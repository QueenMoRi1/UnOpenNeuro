"""Turns game state into decisions, and chat into replies."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import jsonschema

from .llm import ChatModel, parse_json
from .persona import system_prompt


@dataclass
class Action:
    name: str
    description: str = ""
    schema: dict | None = None


@dataclass
class Decision:
    action: str
    data: dict | None
    say: str = ""


@dataclass
class GameMemory:
    """Everything one connected game has told us."""

    game: str
    window: int = 40
    actions: dict[str, Action] = field(default_factory=dict)
    context: deque = field(init=False)

    def __post_init__(self) -> None:
        self.context = deque(maxlen=self.window)

    def add_context(self, text: str) -> None:
        self.context.append(text)


NO_ACTION = "__none__"


def _usable_schema(schema: dict | None) -> dict | None:
    # The SDK says an action with no parameters omits the schema; an empty
    # object schema means the same thing.
    if not schema or schema == {"type": "object"}:
        return None
    return schema


class Brain:
    def __init__(self, llm: ChatModel, persona: dict):
        self.llm = llm
        self.persona = persona

    # ------------------------------------------------------------- games

    def _game_messages(
        self, mem: GameMemory, state: str | None, query: str | None
    ) -> list[dict[str, str]]:
        lines = [f"[game] {c}" for c in mem.context] or ["(nothing yet)"]
        parts = ["Recent events:\n" + "\n".join(lines)]
        if state:
            parts.append("Current state:\n" + state)
        if query:
            parts.append("Instruction from the game:\n" + query)
        return [
            {"role": "system", "content": system_prompt(self.persona, mem.game)},
            {"role": "user", "content": "\n\n".join(parts)},
        ]

    async def decide(
        self,
        mem: GameMemory,
        action_names: list[str],
        state: str | None = None,
        query: str | None = None,
        allow_none: bool = False,
        previous_error: str | None = None,
    ) -> Decision | None:
        """Pick an action and fill in its parameters.

        Done in two steps (pick, then fill) because small models are far
        more reliable that way than with one giant oneOf schema.
        """
        options = [mem.actions[n] for n in action_names if n in mem.actions]
        if not options:
            return None

        messages = self._game_messages(mem, state, query)
        menu = "\n".join(f"- {a.name}: {a.description}" for a in options)
        names = [a.name for a in options]
        if allow_none:
            names.append(NO_ACTION)
            menu += f"\n- {NO_ACTION}: do nothing right now"
        pick_prompt = f"Available actions:\n{menu}\n\n"
        if previous_error:
            pick_prompt += (
                f"Your last attempt failed. The game said: {previous_error}\n\n"
            )
        pick_prompt += (
            'Reply with JSON: {"action": <one action name>, '
            '"say": <one short in-character line, or "">}'
        )
        messages.append({"role": "user", "content": pick_prompt})

        pick_schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": names},
                "say": {"type": "string"},
            },
            "required": ["action", "say"],
        }
        pick = parse_json(await self.llm.chat(messages, pick_schema))
        name = pick.get("action")
        if name == NO_ACTION:
            return None
        if name not in mem.actions:
            # Model ignored the enum (some backends don't enforce it).
            name = options[0].name

        action = mem.actions[name]
        data = None
        schema = _usable_schema(action.schema)
        if schema is not None:
            messages.append({"role": "assistant", "content": json.dumps(pick)})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Now give the parameters for {name} as JSON matching "
                        f"this schema:\n{json.dumps(schema)}"
                    ),
                }
            )
            data = parse_json(await self.llm.chat(messages, schema))
            jsonschema.validate(data, schema)
        return Decision(action=name, data=data, say=str(pick.get("say") or ""))

    # -------------------------------------------------------------- chat

    async def reply(self, history: list[dict[str, str]]) -> str:
        messages = [{"role": "system", "content": system_prompt(self.persona)}]
        messages.extend(history)
        return await self.llm.chat(messages)


def action_from_wire(raw: dict[str, Any]) -> Action:
    return Action(
        name=raw["name"],
        description=raw.get("description", ""),
        schema=raw.get("schema"),
    )

"""A local server that speaks the Neuro Game SDK websocket protocol.

In the real setup, Neuro is the websocket *server* and the game (your mod)
is the *client*. Point your mod at ws://localhost:8000 (the SDK default, or
set the NEURO_SDK_WS_URL environment variable) and it will talk to this
instead of the real Neuro.

Protocol reference: https://github.com/VedalAI/neuro-game-sdk/blob/main/API/SPECIFICATION.md
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Awaitable, Callable

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from .brain import Brain, GameMemory, action_from_wire

log = logging.getLogger("unopenneuro.api")

Speaker = Callable[[str], Awaitable[None]]


class GameSession:
    """One connected game."""

    def __init__(self, ws: ServerConnection, brain: Brain, cfg: dict, speaker: Speaker | None):
        self.ws = ws
        self.brain = brain
        self.cfg = cfg
        self.speaker = speaker
        self.mem = GameMemory(game="unknown", window=cfg["context_window"])
        self.pending: dict[str, asyncio.Future] = {}
        self.force_lock = asyncio.Lock()
        self.tasks: set[asyncio.Task] = set()

    # ---------------------------------------------------------- plumbing

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        task.add_done_callback(_log_task_error)

    async def _send(self, command: str, data: dict | None = None) -> None:
        msg: dict[str, Any] = {"command": command}
        if data is not None:
            msg["data"] = data
        await self.ws.send(json.dumps(msg))

    def _say(self, text: str) -> None:
        if not text:
            return
        print(f"\n{self.brain.persona.get('name', 'Neuro')}: {text}\n", flush=True)
        if self.speaker:
            self._spawn(self.speaker(text))

    async def run(self) -> None:
        if self.cfg.get("idle_action_interval", 0) > 0:
            self._spawn(self._idle_loop())
        try:
            async for raw in self.ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    log.warning("Ignoring non-JSON message: %r", raw[:200])
                    continue
                await self.handle(msg)
        except ConnectionClosed:
            pass
        finally:
            for task in list(self.tasks):
                task.cancel()
            log.info("Game %r disconnected", self.mem.game)

    # ---------------------------------------------------------- messages

    async def handle(self, msg: dict[str, Any]) -> None:
        command = msg.get("command")
        data = msg.get("data") or {}
        if msg.get("game"):
            self.mem.game = msg["game"]

        if command == "startup":
            self.mem.actions.clear()
            log.info("Game %r started up", self.mem.game)

        elif command == "context":
            text = data.get("message", "")
            self.mem.add_context(text)
            log.info("[context%s] %s", " silent" if data.get("silent") else "", text)

        elif command == "actions/register":
            for raw in data.get("actions", []):
                action = action_from_wire(raw)
                self.mem.actions[action.name] = action
            log.info("Actions now: %s", ", ".join(self.mem.actions) or "(none)")

        elif command == "actions/unregister":
            for name in data.get("action_names", []):
                self.mem.actions.pop(name, None)
            log.info("Actions now: %s", ", ".join(self.mem.actions) or "(none)")

        elif command == "actions/force":
            self._spawn(self._forced(data))

        elif command == "action/result":
            message = data.get("message")
            if message:
                self.mem.add_context(message)
            future = self.pending.pop(data.get("id"), None)
            if future and not future.done():
                future.set_result(data)
            log.info(
                "[result %s] %s",
                "ok" if data.get("success") else "FAILED",
                message or "",
            )

        else:
            log.warning("Unknown command %r", command)

    # ----------------------------------------------------------- actions

    async def _execute(self, decision) -> dict:
        """Send an action to the game and wait for its action/result."""
        action_id = str(uuid.uuid4())
        future = asyncio.get_running_loop().create_future()
        self.pending[action_id] = future
        payload: dict[str, Any] = {"id": action_id, "name": decision.action}
        if decision.data is not None:
            payload["data"] = json.dumps(decision.data)
        log.info("-> action %s %s", decision.action, payload.get("data", ""))
        await self._send("action", payload)
        self._say(decision.say)
        try:
            return await asyncio.wait_for(future, timeout=60)
        except asyncio.TimeoutError:
            self.pending.pop(action_id, None)
            log.warning("Game never sent action/result for %s", action_id)
            return {"success": True}

    async def _forced(self, data: dict) -> None:
        async with self.force_lock:
            names = data.get("action_names", [])
            state, query = data.get("state"), data.get("query")
            error = None
            for _attempt in range(1 + self.cfg["max_retries"]):
                try:
                    decision = await self.brain.decide(
                        self.mem, names, state=state, query=query, previous_error=error
                    )
                except Exception as exc:  # bad JSON from the model, etc.
                    log.warning("Model produced an unusable answer: %s", exc)
                    error = f"Your answer was invalid: {exc}"
                    continue
                if decision is None:
                    log.warning("Force named no registered actions: %s", names)
                    return
                result = await self._execute(decision)
                if result.get("success"):
                    break
                error = result.get("message") or "the action failed"
            if not data.get("ephemeral_context"):
                for text in (state, query):
                    if text:
                        self.mem.add_context(text)

    async def _idle_loop(self) -> None:
        interval = self.cfg["idle_action_interval"]
        while True:
            await asyncio.sleep(interval)
            if not self.mem.actions or self.force_lock.locked():
                continue
            async with self.force_lock:
                try:
                    decision = await self.brain.decide(
                        self.mem, list(self.mem.actions), allow_none=True
                    )
                except Exception as exc:
                    log.warning("Idle decision failed: %s", exc)
                    continue
                if decision:
                    await self._execute(decision)


def _log_task_error(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception():
        log.error("Background task failed", exc_info=task.exception())


async def run_server(brain: Brain, cfg: dict, speaker: Speaker | None = None) -> None:
    async def handler(ws: ServerConnection) -> None:
        log.info("Game connected from %s", ws.remote_address)
        await GameSession(ws, brain, cfg, speaker).run()

    async with serve(handler, cfg["host"], cfg["port"]) as server:
        log.info("Neuro API listening on ws://%s:%s", cfg["host"], cfg["port"])
        await server.serve_forever()

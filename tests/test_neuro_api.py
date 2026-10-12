import asyncio
import json

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from unopenneuro.brain import Brain
from unopenneuro.neuro_api import GameSession

CFG = {"context_window": 10, "max_retries": 2, "idle_action_interval": 0}
PERSONA = {"name": "Neuro", "creator": "Vedal"}


class ScriptedLLM:
    """Returns canned answers in order and records every prompt."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    async def chat(self, messages, schema=None):
        self.prompts.append((list(messages), schema))
        return json.dumps(self.answers.pop(0))


async def start(llm):
    brain = Brain(llm, PERSONA)

    async def handler(ws):
        await GameSession(ws, brain, CFG, speaker=None).run()

    server = await serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, f"ws://127.0.0.1:{port}"


async def game_setup(ws):
    for msg in (
        {"command": "startup", "game": "Tic Tac Toe"},
        {"command": "context", "game": "Tic Tac Toe", "data": {"message": "You are O.", "silent": True}},
        {
            "command": "actions/register",
            "game": "Tic Tac Toe",
            "data": {
                "actions": [
                    {
                        "name": "play",
                        "description": "Place an O",
                        "schema": {
                            "type": "object",
                            "properties": {"cell": {"type": "integer", "minimum": 0, "maximum": 8}},
                            "required": ["cell"],
                        },
                    },
                    {"name": "forfeit", "description": "Give up"},
                ]
            },
        },
    ):
        await ws.send(json.dumps(msg))


async def recv_action(ws):
    msg = json.loads(await asyncio.wait_for(ws.recv(), 5))
    assert msg["command"] == "action"
    return msg["data"]


async def test_forced_action_with_parameters():
    llm = ScriptedLLM([{"action": "play", "say": "Middle, obviously."}, {"cell": 4}])
    server, url = await start(llm)
    async with server, connect(url) as ws:
        await game_setup(ws)
        await ws.send(json.dumps({
            "command": "actions/force", "game": "Tic Tac Toe",
            "data": {"state": "board empty", "query": "Your turn", "action_names": ["play", "forfeit"]},
        }))
        action = await recv_action(ws)
        assert action["name"] == "play"
        assert json.loads(action["data"]) == {"cell": 4}
        await ws.send(json.dumps({
            "command": "action/result", "game": "Tic Tac Toe",
            "data": {"id": action["id"], "success": True},
        }))

    pick_messages, pick_schema = llm.prompts[0]
    assert pick_schema["properties"]["action"]["enum"] == ["play", "forfeit"]
    prompt = pick_messages[-2]["content"]
    assert "You are O." in prompt and "board empty" in prompt and "Your turn" in prompt
    assert "Tic Tac Toe" in pick_messages[0]["content"]


async def test_failed_action_is_retried_with_error():
    llm = ScriptedLLM([
        {"action": "play", "say": ""}, {"cell": 4},
        {"action": "play", "say": ""}, {"cell": 0},
    ])
    server, url = await start(llm)
    async with server, connect(url) as ws:
        await game_setup(ws)
        await ws.send(json.dumps({
            "command": "actions/force", "game": "Tic Tac Toe",
            "data": {"query": "Your turn", "action_names": ["play"]},
        }))
        first = await recv_action(ws)
        await ws.send(json.dumps({
            "command": "action/result", "game": "Tic Tac Toe",
            "data": {"id": first["id"], "success": False, "message": "Cell 4 is taken"},
        }))
        second = await recv_action(ws)
        assert json.loads(second["data"]) == {"cell": 0}
        assert second["id"] != first["id"]

    retry_prompt = llm.prompts[2][0][-1]["content"]
    assert "Cell 4 is taken" in retry_prompt


async def test_action_without_schema_sends_no_data():
    llm = ScriptedLLM([{"action": "forfeit", "say": "I let you win."}])
    server, url = await start(llm)
    async with server, connect(url) as ws:
        await game_setup(ws)
        await ws.send(json.dumps({
            "command": "actions/force", "game": "Tic Tac Toe",
            "data": {"query": "Give up?", "action_names": ["forfeit"]},
        }))
        action = await recv_action(ws)
        assert action["name"] == "forfeit"
        assert "data" not in action
    assert len(llm.prompts) == 1


async def test_unregistered_actions_are_not_offered():
    llm = ScriptedLLM([{"action": "play", "say": ""}, {"cell": 1}])
    server, url = await start(llm)
    async with server, connect(url) as ws:
        await game_setup(ws)
        await ws.send(json.dumps({
            "command": "actions/unregister", "game": "Tic Tac Toe",
            "data": {"action_names": ["forfeit"]},
        }))
        await ws.send(json.dumps({
            "command": "actions/force", "game": "Tic Tac Toe",
            "data": {"query": "Your turn", "action_names": ["play", "forfeit"]},
        }))
        await recv_action(ws)
    assert llm.prompts[0][1]["properties"]["action"]["enum"] == ["play"]

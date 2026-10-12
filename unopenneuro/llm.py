"""Minimal async client for a local Ollama server.

Uses Ollama's native /api/chat endpoint because it supports constrained
JSON output (`format` = JSON schema), which is how we make a small model
reliably produce valid Neuro API action payloads.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

import httpx

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class ChatModel(Protocol):
    async def chat(
        self, messages: list[dict[str, str]], schema: dict | None = None
    ) -> str: ...


class OllamaChat:
    def __init__(self, cfg: dict[str, Any]):
        self.base_url = cfg["base_url"].rstrip("/")
        self.model = cfg["model"]
        self.temperature = cfg.get("temperature", 0.9)
        self.think = cfg.get("think")  # None = leave the model's default
        self._client = httpx.AsyncClient(timeout=cfg.get("timeout", 120))

    async def chat(
        self, messages: list[dict[str, str]], schema: dict | None = None
    ) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if schema is not None:
            body["format"] = schema
        if self.think is not None:
            body["think"] = self.think
        resp = await self._client.post(f"{self.base_url}/api/chat", json=body)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        return _THINK_RE.sub("", content).strip()

    async def chat_json(
        self, messages: list[dict[str, str]], schema: dict
    ) -> dict:
        return parse_json(await self.chat(messages, schema))

    async def aclose(self) -> None:
        await self._client.aclose()


def parse_json(text: str) -> dict:
    """Parse a JSON object, tolerating code fences or chatter around it."""
    text = _THINK_RE.sub("", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])

"""Tiny VTube Studio API client: authenticate, then move the mouth.

VTube Studio must have its API enabled:
  Settings (gear icon) -> "Start API (allow plugins)" -> ON, port 8001.
"""

from __future__ import annotations

import json
import logging
import uuid

from websockets.asyncio.client import ClientConnection, connect

log = logging.getLogger("unopenneuro.vts")

PLUGIN_NAME = "UnOpenNeuro"
PLUGIN_DEVELOPER = "UnOpenNeuro"


class VTubeStudio:
    def __init__(self, cfg: dict):
        self.url = cfg["url"]
        self.token = cfg.get("token") or ""
        self.mouth_parameter = cfg.get("mouth_parameter", "MouthOpen")
        self.ws: ClientConnection | None = None

    async def _request(self, message_type: str, data: dict | None = None) -> dict:
        assert self.ws is not None
        await self.ws.send(
            json.dumps(
                {
                    "apiName": "VTubeStudioPublicAPI",
                    "apiVersion": "1.0",
                    "requestID": str(uuid.uuid4()),
                    "messageType": message_type,
                    "data": data or {},
                }
            )
        )
        reply = json.loads(await self.ws.recv())
        if reply.get("messageType") == "APIError":
            raise RuntimeError(f"VTube Studio error: {reply['data']}")
        return reply.get("data", {})

    async def connect(self) -> None:
        self.ws = await connect(self.url)
        if not self.token:
            raise RuntimeError("No VTube Studio token. Run `unopenneuro vts-auth` first.")
        data = await self._request(
            "AuthenticationRequest",
            {
                "pluginName": PLUGIN_NAME,
                "pluginDeveloper": PLUGIN_DEVELOPER,
                "authenticationToken": self.token,
            },
        )
        if not data.get("authenticated"):
            raise RuntimeError(f"VTube Studio refused the token: {data.get('reason')}")
        log.info("Connected to VTube Studio")

    async def request_token(self) -> str:
        """Pops up an 'allow plugin?' dialog inside VTube Studio."""
        self.ws = await connect(self.url)
        data = await self._request(
            "AuthenticationTokenRequest",
            {"pluginName": PLUGIN_NAME, "pluginDeveloper": PLUGIN_DEVELOPER},
        )
        await self.ws.close()
        return data["authenticationToken"]

    async def set_mouth(self, value: float) -> None:
        if self.ws is None:
            return
        try:
            await self._request(
                "InjectParameterDataRequest",
                {
                    "faceFound": False,
                    "mode": "set",
                    "parameterValues": [{"id": self.mouth_parameter, "value": value}],
                },
            )
        except Exception as exc:
            log.warning("Lost VTube Studio connection: %s", exc)
            self.ws = None

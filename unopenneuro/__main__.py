"""Command line entry point.

  unopenneuro serve      run the Neuro API server for your mods
  unopenneuro chat       talk to her in the terminal (add --listen for mic)
  unopenneuro vts-auth   connect to VTube Studio once and save the token
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from .brain import Brain
from .config import load_config, save_value
from .llm import OllamaChat


async def _make_speaker(cfg: dict):
    if cfg["voice"]["tts"] == "none":
        return None
    from .voice import Speaker

    avatar = None
    if cfg["vtube_studio"]["enabled"]:
        from .vtube_studio import VTubeStudio

        avatar = VTubeStudio(cfg["vtube_studio"])
        await avatar.connect()
    return Speaker(cfg["voice"], avatar=avatar)


async def cmd_serve(cfg: dict) -> None:
    from .neuro_api import run_server

    llm = OllamaChat(cfg["llm"])
    speaker = await _make_speaker(cfg)
    await run_server(Brain(llm, cfg["persona"]), cfg["neuro_api"], speaker)


async def cmd_chat(cfg: dict, listen: bool) -> None:
    llm = OllamaChat(cfg["llm"])
    brain = Brain(llm, cfg["persona"])
    speaker = await _make_speaker(cfg)
    listener = None
    if listen:
        from .voice import Listener

        listener = Listener(cfg["voice"])
    name = cfg["persona"]["name"]
    history: list[dict[str, str]] = []
    print(f"Chatting with {name}. Ctrl+C or type /quit to exit.\n")
    while True:
        if listener:
            input("Press Enter to talk... ")
            text = await asyncio.to_thread(
                lambda: listener.transcribe(listener.record_until_enter())
            )
            print(f"you: {text}")
        else:
            text = await asyncio.to_thread(input, "you: ")
        text = text.strip()
        if not text:
            continue
        if text == "/quit":
            break
        history.append({"role": "user", "content": text})
        reply = await brain.reply(history[-20:])
        history.append({"role": "assistant", "content": reply})
        print(f"{name}: {reply}\n")
        if speaker:
            await speaker(reply)


async def cmd_vts_auth(cfg: dict, config_path: str) -> None:
    from .vtube_studio import VTubeStudio

    print("Look at VTube Studio now and click 'Allow' on the popup...")
    token = await VTubeStudio(cfg["vtube_studio"]).request_token()
    save_value(config_path, "vtube_studio", "token", token)
    save_value(config_path, "vtube_studio", "enabled", True)
    print(f"Saved token to {config_path}. Lip-sync is now enabled.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="unopenneuro")
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", help="run the Neuro API server for mods")
    chat = sub.add_parser("chat", help="talk to her in the terminal")
    chat.add_argument("--listen", action="store_true", help="use the microphone")
    sub.add_parser("vts-auth", help="pair with VTube Studio")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config(args.config)
    try:
        if args.command == "serve":
            asyncio.run(cmd_serve(cfg))
        elif args.command == "chat":
            asyncio.run(cmd_chat(cfg, args.listen))
        elif args.command == "vts-auth":
            asyncio.run(cmd_vts_auth(cfg, args.config))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

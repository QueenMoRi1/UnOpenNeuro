# UnOpenNeuro

A private recreation of Neuro-sama, for testing and showcasing mods.
Not for release, streaming, or monetisation.

It runs a local stand-in that speaks the **official Neuro Game SDK websocket
protocol**, so a mod built with the SDK connects to it exactly as it would to
the real Neuro. A local open-source model makes the decisions, and you can add
a voice (TTS), a microphone (STT) and VTube Studio lip-sync.

```
 your mod / game  ──ws://localhost:8000──▶  UnOpenNeuro  ──▶  Ollama (local LLM)
                  ◀──── "action" ─────────       │
                                                 ├──▶ TTS voice ──▶ VTube Studio mouth
                                                 └──◀ mic (chat --listen)
```

## Setup (Mac, no GPU needed)

1. **Install Ollama** from https://ollama.com and open it once.
2. **Download a model** in Terminal:
   ```bash
   ollama pull qwen3:4b
   ```
   (8 GB Mac: use `qwen3:1.7b` and set `model: qwen3:1.7b` in `config.yaml`.)
3. **Install this project:**
   ```bash
   cd UnOpenNeuro
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[voice,listen]"
   cp config.example.yaml config.yaml
   ```
4. **Say hi:**
   ```bash
   unopenneuro chat
   ```

## Testing a Neuro SDK mod

```bash
unopenneuro serve
```

Start your game/mod. It connects to `ws://localhost:8000` by default (or
whatever `NEURO_SDK_WS_URL` is set to). The terminal shows everything the game
sends (context, registered actions, results) and every action she takes.

Supported commands: `startup`, `context`, `actions/register`,
`actions/unregister`, `actions/force` (including `state`, `query`,
`ephemeral_context`) and `action/result`. A failed forced action is retried
with the game's error message, up to `max_retries` times. Set
`idle_action_interval` to let her act without being forced, like the real
Neuro does.

Action parameters are generated with Ollama's JSON-schema-constrained output
and validated against the schema your mod registered, so small models still
send well-formed `data`.

## Voice and avatar

In `config.yaml`:

- `voice.tts: say` uses the built-in macOS voice. No setup.
- `voice.tts: kokoro` uses Kokoro, a small open-source TTS that sounds much
  better and runs on the CPU. `pitch: 1.2` speeds it up and raises the pitch,
  which is roughly how Neuro's voice is made.
- `unopenneuro chat --listen` lets you talk to her with your microphone
  (press Enter to start, Enter again to stop).

**VTube Studio lip-sync** (Kokoro only):

1. In VTube Studio open Settings (gear icon) and turn on **Start API (allow plugins)**. Leave the port at 8001.
2. Run `unopenneuro vts-auth` and click **Allow** in the popup inside VTube Studio.
3. Done. `serve` and `chat` now move the model's `MouthOpen` parameter while she talks.

## Making her actually sound like Neuro: fine-tuning

The persona prompt in `unopenneuro/persona.py` gets you part of the way. To
get closer you fine-tune a small model on her dialogue. See
[`training/README.md`](training/README.md).

## Development

```bash
pip install -e ".[dev]"
pytest
```

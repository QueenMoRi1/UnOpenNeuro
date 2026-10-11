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

## Setup (Windows, e.g. ROG Ally; no NVIDIA GPU needed)

Do these once. Tip: on the Ally, switch to desktop mode and plug in a keyboard.

1. **Install Python 3.12** from https://www.python.org/downloads/windows/.
   On the first installer screen, tick **"Add python.exe to PATH"**.
2. **Install Git** from https://git-scm.com/download/win (the default options are fine).
3. **Install Ollama** from https://ollama.com/download/windows. It runs in the
   system tray once installed.
4. **Open PowerShell** (Start menu -> type `PowerShell` -> Enter) and download a model:
   ```powershell
   ollama pull qwen3:4b
   ```
   If replies feel slow, use `qwen3:1.7b` instead and set `model: qwen3:1.7b`
   in `config.yaml`.
5. **Get the project and install it** (still in PowerShell):
   ```powershell
   git clone https://github.com/queenmori1/unopenneuro.git
   cd unopenneuro
   git checkout ccr-29042952-9yipdk
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -e ".[voice,listen]"
   copy config.example.yaml config.yaml
   ```
   If the repo is private, a GitHub sign-in window pops up during
   `git clone`. Sign in and it continues.
   If `Activate.ps1` gives a red "running scripts is disabled" error, run
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, answer `Y`, then try again.
6. **Say hi:**
   ```powershell
   unopenneuro chat
   ```

Every time you open a new PowerShell window later, first run
`cd unopenneuro` then `.venv\Scripts\Activate.ps1`.

On a Mac or Linux the steps are the same except: `python3 -m venv .venv`,
`source .venv/bin/activate` and `cp` instead of `copy`.

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

- `voice.tts: system` uses the voice built into Windows (or macOS). No setup.
- `voice.tts: kokoro` uses Kokoro, a small open-source TTS that sounds much
  better and runs on the CPU. On Windows, also install espeak-ng (the
  `espeak-ng.msi` from https://github.com/espeak-ng/espeak-ng/releases), which
  Kokoro uses for words it doesn't know.
- `pitch: 1.2` speeds the voice up and raises it, which is roughly how
  Neuro's voice is made.
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

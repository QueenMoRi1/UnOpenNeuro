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

## Setup (Bazzite / ROG Ally, using distrobox)

Bazzite's system files are read-only, so everything goes in a **distrobox**:
a normal Ubuntu inside a container. It shares your home folder, network,
sound and microphone with Bazzite, so games, VTube Studio and your files all
see it as if it were installed normally. Distrobox comes with Bazzite.

Do these once, in **Desktop Mode** (Steam button -> Power -> Switch to
Desktop). A keyboard helps.

1. **Open Konsole** (the terminal: app menu bottom-left -> type `Konsole` -> Enter).
2. **Create the box and go into it:**
   ```bash
   distrobox create --name neuro --image ubuntu:24.04
   distrobox enter neuro
   ```
   The first time takes a few minutes. When it finishes, your prompt changes to
   show `neuro`. That means you're inside the box.
3. **Install the tools** (inside the box; it asks for your password once):
   ```bash
   sudo apt update
   sudo apt install -y python3-venv python3-dev build-essential git curl ffmpeg libportaudio2 espeak-ng
   ```
4. **Install Ollama** (inside the box):
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
   Ignore any warning about systemd; the box has none, so you start Ollama by
   hand in the next step.
5. **Start Ollama in its own tab.** In Konsole, press **Ctrl+Shift+T** for a new tab, then:
   ```bash
   distrobox enter neuro
   ollama serve
   ```
   Leave this tab open; Ollama runs as long as it is open. Go back to the first tab.
6. **Download a model** (first tab):
   ```bash
   ollama pull qwen3:4b
   ```
   If replies feel slow later, use `qwen3:1.7b` instead and set
   `model: qwen3:1.7b` in `config.yaml`.
7. **Get the project and install it** (first tab):
   ```bash
   cd ~
   git clone https://github.com/queenmori1/unopenneuro.git
   cd unopenneuro
   git checkout ccr-29042952-9yipdk
   python3 -m venv .venv
   source .venv/bin/activate
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -e ".[voice,listen]"
   cp config.example.yaml config.yaml
   ```
   The `torch` line installs the small CPU-only build. Without it, pip
   downloads several GB of NVIDIA files the Ally can't use. If the repo is
   private, `git clone` asks for your GitHub username and a personal access
   token instead of a password.
8. **Say hi:**
   ```bash
   unopenneuro chat
   ```

**Every time after that**, open Konsole and:
- Tab 1: `distrobox enter neuro`, then `ollama serve` (leave it running).
- Tab 2: `distrobox enter neuro`, then `cd ~/unopenneuro`, then `source .venv/bin/activate`.

The project lives in `~/unopenneuro` in your normal home folder, so you can
open it in the Dolphin file manager to drop in recordings or edit `config.yaml`.

**Optional: use the Ally's graphics chip.** Ollama runs on the processor by
default. The Ally's AMD graphics aren't officially supported by ROCm (the
software Ollama uses for AMD GPUs), but a common workaround is to start
Ollama with `HSA_OVERRIDE_GFX_VERSION=11.0.0 ollama serve` instead. If it
crashes or isn't faster, go back to plain `ollama serve`.

**Other systems:** on Windows, install Python, Git and Ollama normally, then
use `py -m venv .venv`, `.venv\Scripts\Activate.ps1` and `copy` instead
of `cp`. On a Mac, skip the `torch` line.

## Testing a Neuro SDK mod

```bash
unopenneuro serve
```

Start your game/mod. It connects to `ws://localhost:8000` by default (or
whatever `NEURO_SDK_WS_URL` is set to). Games launched from Steam (including Windows
games through Proton) reach the box fine, because distrobox shares Bazzite's
network. The terminal shows everything the game
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

- `voice.tts: system` uses espeak-ng (installed in step 3). Robotic, but instant and no setup.
- `voice.tts: kokoro` uses Kokoro, a small open-source TTS that sounds much
  better and runs on the CPU. It also uses espeak-ng for words it doesn't know.
- `pitch: 1.2` speeds the voice up and raises it, which is roughly how
  Neuro's voice is made.
- `unopenneuro chat --listen` lets you talk to her with your microphone
  (press Enter to start, Enter again to stop).

**VTube Studio lip-sync** (Kokoro only):

VTube Studio is on Steam and runs on Bazzite through Proton. Its plugin
connection uses the network, so the box can reach it.


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

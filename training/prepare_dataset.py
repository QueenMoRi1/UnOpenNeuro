"""Turn transcripts into a chat fine-tuning dataset.

Input: a folder of files.
  *.txt / *.srt   transcript lines like "Neuro: hello chat" or "Vedal: no".
                  Lines with no "Speaker:" prefix are given to --default-speaker.
                  SRT numbering and timestamps are stripped automatically.
  *.jsonl         already formatted rows: {"messages": [{"role": ..., "content": ...}]}

Output: train.jsonl and valid.jsonl in the "messages" chat format, which
both MLX-LM (Mac) and Unsloth (Colab) read directly.

Example:
  python training/prepare_dataset.py data/raw data/train --persona Neuro
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unopenneuro.persona import system_prompt  # noqa: E402

SPEAKER_RE = re.compile(r"^\s*([^:\n]{1,32}):\s+(.+)$")
SRT_TIMING_RE = re.compile(r"^\d\d:\d\d:\d\d[,.]\d+\s*-->")


def read_lines(path: Path, default_speaker: str) -> list[tuple[str, str]]:
    """Return (speaker, text) pairs from a .txt or .srt file."""
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.isdigit() or SRT_TIMING_RE.match(line):
            continue
        line = re.sub(r"<[^>]+>", "", line)  # SRT styling tags
        m = SPEAKER_RE.match(line)
        if m:
            out.append((m.group(1).strip(), m.group(2).strip()))
        else:
            out.append((default_speaker, line))
    return out


def to_turns(lines: list[tuple[str, str]], persona: str) -> list[dict[str, str]]:
    """Persona lines become assistant turns, everyone else becomes user.

    Other speakers keep their name as a prefix so the model learns who
    it is talking to (chat vs. her creator vs. a collab partner).
    """
    turns: list[dict[str, str]] = []
    for speaker, text in lines:
        if speaker.lower() == persona.lower():
            role, content = "assistant", text
        else:
            role, content = "user", f"{speaker}: {text}"
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n" + content if role == "user" else " " + content
        else:
            turns.append({"role": role, "content": content})
    return turns


def windows(turns: list[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    """Cut a long transcript into overlapping samples that end on her reply."""
    samples = []
    for end, turn in enumerate(turns):
        if turn["role"] != "assistant":
            continue
        start = max(0, end - size + 1)
        chunk = turns[start : end + 1]
        if chunk[0]["role"] == "assistant":
            chunk = [{"role": "user", "content": "(continue)"}] + chunk
        samples.append(chunk)
    return samples


def build(
    src: Path, persona: dict, default_speaker: str, window: int
) -> list[dict]:
    system = {"role": "system", "content": system_prompt(persona)}
    rows: list[dict] = []
    for path in sorted(src.rglob("*")):
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        elif path.suffix in (".txt", ".srt"):
            turns = to_turns(read_lines(path, default_speaker), persona["name"])
            rows.extend({"messages": [system, *w]} for w in windows(turns, window))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path, help="folder of transcripts")
    p.add_argument("out", type=Path, help="folder to write train/valid.jsonl")
    p.add_argument("--persona", default="Neuro")
    p.add_argument("--creator", default="Vedal")
    p.add_argument("--default-speaker", default="Neuro")
    p.add_argument("--window", type=int, default=8, help="turns per sample")
    p.add_argument("--valid-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    persona = {"name": args.persona, "creator": args.creator}
    rows = build(args.src, persona, args.default_speaker, args.window)
    if not rows:
        sys.exit(f"No usable samples found in {args.src}")
    random.Random(args.seed).shuffle(rows)
    n_valid = max(1, int(len(rows) * args.valid_fraction))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, part in (("valid", rows[:n_valid]), ("train", rows[n_valid:])):
        with open(args.out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in part:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows) - n_valid} train / {n_valid} valid samples to {args.out}")


if __name__ == "__main__":
    main()

"""Transcribe stream audio into speaker-labelled lines for prepare_dataset.py.

Uses WhisperX (speech-to-text with word timestamps) plus pyannote speaker
diarization ("who spoke when"). Each detected speaker is then named:

  1. --names SPEAKER_00=Neuro      you name them yourself, or
  2. --voice Neuro=neuro.wav       auto-match against short reference clips, or
  3. --ask                         it shows sample lines and asks you.

Output per input file:
  data/raw/<name>.txt              "Neuro: ..." / "Vedal: ..." lines
  data/transcripts/<name>.json     everything WhisperX found, so you can
                                   rename speakers later without re-running
                                   (see --relabel)

Tip: put the stream date at the front of the filename
(2024-05-12_karaoke.mp4) so you can later train on one era only.

ONE-TIME SETUP
  1. Make a free account at https://huggingface.co and create a read token at
     https://huggingface.co/settings/tokens
  2. While logged in, open
     https://huggingface.co/pyannote/speaker-diarization-community-1
     and click "Agree and access repository". (Older WhisperX versions use
     speaker-diarization-3.1 instead. If you get an access error, open the
     link in the error message and agree there too.)
  3. Install:  pip install -e ".[transcribe]"
     Also install ffmpeg. Windows: `winget install ffmpeg` in PowerShell,
     then open a new window. Mac: `brew install ffmpeg`. (Colab already has it.)

EXAMPLES
  # Free Colab GPU (fast: a 3 hour stream takes roughly 10-20 minutes)
  python training/transcribe.py stream.mp4 --hf-token hf_xxx \\
      --voice Neuro=refs/neuro.wav --voice Vedal=refs/vedal.wav

  # PC without an NVIDIA GPU, e.g. ROG Ally (works, but expect real time or slower)
  python training/transcribe.py stream.m4a --hf-token hf_xxx --model small.en --ask

  # Fix the names afterwards, no GPU needed
  python training/transcribe.py --relabel data/transcripts/stream.json \\
      --names SPEAKER_00=Neuro,SPEAKER_02=Vedal
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

AUDIO_EXTS = {".mp3", ".mp4", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".webm", ".mkv", ".aac"}
UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------- pure helpers


def parse_pairs(values: list[str] | None) -> dict[str, str]:
    """["A=x,B=y", "C=z"] -> {"A": "x", "B": "y", "C": "z"}"""
    out: dict[str, str] = {}
    for value in values or []:
        for pair in value.split(","):
            if not pair.strip():
                continue
            key, sep, val = pair.partition("=")
            if not sep or not key.strip() or not val.strip():
                raise ValueError(f"Expected NAME=VALUE, got {pair!r}")
            out[key.strip()] = val.strip()
    return out


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def match_speakers(
    embeddings: dict[str, list[float]],
    references: dict[str, list[float]],
    threshold: float,
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    """Name each diarized speaker after its closest reference voice.

    Several speakers may map to the same name: diarization sometimes splits
    one person into two clusters (e.g. when she sings).
    """
    names: dict[str, str] = {}
    scores: dict[str, dict[str, float]] = {}
    for speaker, emb in embeddings.items():
        scores[speaker] = {name: cosine(emb, ref) for name, ref in references.items()}
        if scores[speaker]:
            best = max(scores[speaker], key=scores[speaker].get)
            if scores[speaker][best] >= threshold:
                names[speaker] = best
    return names, scores


def split_by_speaker(segment: dict) -> list[dict]:
    """Split one WhisperX segment wherever the word-level speaker changes.

    Fast back-and-forth (her and Vedal interrupting each other) often lands
    in a single segment; splitting on words keeps the lines attributed right.
    """
    default = segment.get("speaker", UNKNOWN)
    words = segment.get("words") or []
    if not words:
        text = segment.get("text", "").strip()
        return [{"speaker": default, "text": text, "score": None,
                 "start": segment.get("start"), "end": segment.get("end")}] if text else []

    turns: list[dict] = []
    current = None
    for word in words:
        speaker = word.get("speaker") or (current["speaker"] if current else default)
        if current is None or speaker != current["speaker"]:
            current = {"speaker": speaker, "words": [], "scores": [],
                       "start": word.get("start"), "end": word.get("end")}
            turns.append(current)
        current["words"].append(word["word"].strip())
        if "score" in word:
            current["scores"].append(word["score"])
        if word.get("end") is not None:
            current["end"] = word["end"]
    return [
        {
            "speaker": t["speaker"],
            "text": " ".join(w for w in t["words"] if w),
            "score": round(sum(t["scores"]) / len(t["scores"]), 3) if t["scores"] else None,
            "start": t["start"],
            "end": t["end"],
        }
        for t in turns
        if any(t["words"])
    ]


def speaker_stats(turns: list[dict]) -> dict[str, dict]:
    """Talk time and a few example lines per speaker, most talkative first."""
    stats: dict[str, dict] = {}
    for t in turns:
        s = stats.setdefault(t["speaker"], {"seconds": 0.0, "samples": []})
        if t.get("start") is not None and t.get("end") is not None:
            s["seconds"] += t["end"] - t["start"]
        if len(s["samples"]) < 4 and len(t["text"].split()) >= 4:
            s["samples"].append(t["text"])
    return dict(sorted(stats.items(), key=lambda kv: -kv[1]["seconds"]))


def to_lines(
    turns: list[dict], names: dict[str, str], min_score: float = 0.0, other: str | None = None
) -> list[str]:
    """Render "Name: text" lines, merging consecutive turns by the same person.

    Speakers without a name keep their diarization label, or become `other`
    if given. Turns whose average word confidence is under `min_score` are
    dropped (usually music, overlapping speech or mumbling).
    """
    lines: list[tuple[str, str]] = []
    for t in turns:
        if min_score and t.get("score") is not None and t["score"] < min_score:
            continue
        name = names.get(t["speaker"]) or other or t["speaker"]
        if lines and lines[-1][0] == name:
            lines[-1] = (name, lines[-1][1] + " " + t["text"])
        else:
            lines.append((name, t["text"]))
    return [f"{name}: {text}" for name, text in lines]


def ask_names(stats: dict[str, dict], names: dict[str, str]) -> dict[str, str]:
    names = dict(names)
    for speaker, s in stats.items():
        if speaker in names:
            continue
        print(f"\n{speaker}  ({s['seconds'] / 60:.1f} min of speech)")
        for sample in s["samples"]:
            print(f"   \"{sample[:120]}\"")
        answer = input(f"Who is {speaker}? (e.g. Neuro, Vedal; Enter = leave unnamed): ").strip()
        if answer:
            names[speaker] = answer
    return names


# ---------------------------------------------------------------- WhisperX


class Transcriber:
    def __init__(self, model: str, device: str, hf_token: str, batch_size: int):
        import whisperx

        self.whisperx = whisperx
        self.device = device
        self.batch_size = batch_size
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"Loading Whisper '{model}' on {device}...")
        self.asr = whisperx.load_model(model, device, compute_type=compute_type, language="en")
        self.align_model, self.align_meta = whisperx.load_align_model(language_code="en", device=device)
        try:
            from whisperx.diarize import DiarizationPipeline
        except ImportError:  # older WhisperX
            DiarizationPipeline = whisperx.DiarizationPipeline
        try:
            self.diarizer = DiarizationPipeline(use_auth_token=hf_token, device=device)
        except TypeError:  # newer WhisperX renamed the argument
            self.diarizer = DiarizationPipeline(token=hf_token, device=device)

    def _diarize(self, audio, min_speakers=None, max_speakers=None):
        """Returns (diarization, {speaker: embedding}). Embeddings may be empty
        on WhisperX versions that cannot return them."""
        kwargs = {"min_speakers": min_speakers, "max_speakers": max_speakers}
        try:
            diarization, embeddings = self.diarizer(audio, return_embeddings=True, **kwargs)
        except TypeError:
            diarization, embeddings = self.diarizer(audio, **kwargs), None
        clean = {k: [float(x) for x in v] for k, v in (embeddings or {}).items()}
        return diarization, clean

    def reference_embedding(self, path: Path) -> list[float]:
        """Embedding of the main voice in a short clean clip (10-60 s)."""
        audio = self.whisperx.load_audio(str(path))
        diarization, embeddings = self._diarize(audio)
        if not embeddings:
            raise RuntimeError(
                "This WhisperX version can't return voice embeddings, so --voice "
                "won't work. Upgrade WhisperX, or use --ask / --names instead."
            )
        # The clip may contain a bit of a second voice; use whoever talks most.
        talk = diarization.assign(dur=diarization["end"] - diarization["start"])
        main = talk.groupby("speaker")["dur"].sum().idxmax()
        return embeddings[main]

    def transcribe(self, path: Path, min_speakers=None, max_speakers=None) -> dict:
        wx = self.whisperx
        audio = wx.load_audio(str(path))
        print("  speech to text...")
        result = self.asr.transcribe(audio, batch_size=self.batch_size, language="en")
        print("  aligning words...")
        result = wx.align(result["segments"], self.align_model, self.align_meta, audio,
                          self.device, return_char_alignments=False)
        print("  working out who is speaking...")
        diarization, embeddings = self._diarize(audio, min_speakers, max_speakers)
        result = wx.assign_word_speakers(diarization, result)
        turns = [t for seg in result["segments"] for t in split_by_speaker(seg)]
        return {"source": str(path), "turns": turns, "embeddings": embeddings}


# ---------------------------------------------------------------- main


def find_inputs(paths: list[Path], exts: set[str]) -> list[Path]:
    found = []
    for p in paths:
        if p.is_dir():
            found.extend(sorted(f for f in p.rglob("*") if f.suffix.lower() in exts))
        elif p.exists():
            found.append(p)
        else:
            sys.exit(f"Not found: {p}")
    return found


def name_and_write(data: dict, stem: str, args, references: dict[str, list[float]]) -> None:
    # Names saved by an earlier run, then names typed on this run on top.
    names = {**data.get("names", {}), **parse_pairs(args.names)}
    if references and data.get("embeddings"):
        matched, scores = match_speakers(data["embeddings"], references, args.match_threshold)
        print("  voice match (higher = more similar):")
        for speaker, row in scores.items():
            pretty = "  ".join(f"{n}={v:.2f}" for n, v in row.items())
            print(f"    {speaker}: {pretty}  ->  {matched.get(speaker, 'no match')}")
        names = {**matched, **names}  # names you typed always win
    if args.ask:
        names = ask_names(speaker_stats(data["turns"]), names)

    data["names"] = names
    args.transcripts.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.transcripts / f"{stem}.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    lines = to_lines(data["turns"], names, args.min_score, args.other)
    (args.out / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    unnamed = sorted({t["speaker"] for t in data["turns"]} - set(names))
    print(f"  wrote {args.out / (stem + '.txt')} ({len(lines)} lines)")
    if unnamed and not args.other:
        print(f"  unnamed speakers left as-is: {', '.join(unnamed)}  "
              f"(fix with --relabel {args.transcripts / (stem + '.json')} --names ...)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", type=Path, help="audio/video files or folders")
    p.add_argument("--relabel", action="store_true",
                   help="inputs are data/transcripts/*.json; only rename speakers")
    p.add_argument("--names", action="append", metavar="SPEAKER_00=Neuro,...")
    p.add_argument("--voice", action="append", metavar="Neuro=clip.wav",
                   help="reference clip of one person (10-60 s, only them talking)")
    p.add_argument("--match-threshold", type=float, default=0.5,
                   help="minimum voice similarity to accept a match (0-1)")
    p.add_argument("--ask", action="store_true", help="ask you to name unmatched speakers")
    p.add_argument("--other", metavar="NAME",
                   help="name for every unnamed speaker, e.g. 'Guest'")
    p.add_argument("--min-score", type=float, default=0.0,
                   help="drop lines with average word confidence below this (try 0.5)")
    p.add_argument("--min-speakers", type=int)
    p.add_argument("--max-speakers", type=int)
    p.add_argument("--model", default="large-v3",
                   help="Whisper model: large-v3 (GPU), small.en or base.en (no GPU)")
    p.add_argument("--device", default=None, help="cuda or cpu (auto-detected)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    p.add_argument("--transcripts", type=Path, default=Path("data/transcripts"))
    args = p.parse_args()

    try:
        voices = parse_pairs(args.voice)
        parse_pairs(args.names)
    except ValueError as exc:
        sys.exit(str(exc))

    if args.relabel:
        for path in find_inputs(args.inputs, {".json"}):
            print(f"\n{path.name}")
            data = json.loads(path.read_text(encoding="utf-8"))
            if voices:
                print("  (--voice is ignored with --relabel; use --names)")
            name_and_write(data, path.stem, args, {})
        return

    if not args.hf_token:
        sys.exit("Speaker diarization needs a Hugging Face token: pass --hf-token or set HF_TOKEN. "
                 "See the setup steps at the top of this file.")
    files = find_inputs(args.inputs, AUDIO_EXTS)
    if not files:
        sys.exit("No audio/video files found.")

    device = args.device
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and args.model == "large-v3":
            print("No GPU found: switching to the small.en model (pass --model to override).")
            args.model = "small.en"

    tx = Transcriber(args.model, device, args.hf_token, args.batch_size)
    references = {}
    for name, clip in voices.items():
        print(f"Learning {name}'s voice from {clip}...")
        references[name] = tx.reference_embedding(Path(clip))

    for i, path in enumerate(files, 1):
        stem = path.stem
        if (args.out / f"{stem}.txt").exists():
            print(f"\n[{i}/{len(files)}] {path.name}: already done, skipping")
            continue
        print(f"\n[{i}/{len(files)}] {path.name}")
        data = tx.transcribe(path, args.min_speakers, args.max_speakers)
        name_and_write(data, stem, args, references)


if __name__ == "__main__":
    main()

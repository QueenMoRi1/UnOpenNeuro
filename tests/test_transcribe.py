import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
import transcribe as tx  # noqa: E402


def word(w, speaker=None, score=0.9, start=0.0, end=0.5):
    out = {"word": w, "score": score, "start": start, "end": end}
    if speaker:
        out["speaker"] = speaker
    return out


def test_split_by_speaker_on_word_changes():
    seg = {"speaker": "A", "start": 0, "end": 3, "text": "hi there no way", "words": [
        word("hi", "A", start=0, end=0.4), word("there", "A", start=0.4, end=1),
        word("no", "B", start=1.2, end=1.5), word("way", start=1.5, end=2),  # no speaker: inherits B
    ]}
    turns = tx.split_by_speaker(seg)
    assert [(t["speaker"], t["text"]) for t in turns] == [("A", "hi there"), ("B", "no way")]
    assert turns[1]["start"] == 1.2 and turns[1]["end"] == 2


def test_split_segment_without_words():
    assert tx.split_by_speaker({"text": " hello ", "start": 0, "end": 1}) == [
        {"speaker": tx.UNKNOWN, "text": "hello", "score": None, "start": 0, "end": 1}
    ]


def test_match_speakers_allows_many_to_one_and_threshold():
    refs = {"Neuro": [1.0, 0.0], "Vedal": [0.0, 1.0]}
    embs = {"S0": [0.9, 0.1], "S1": [0.95, 0.05], "S2": [0.1, 0.9], "S3": [-1.0, 0.0]}
    names, scores = tx.match_speakers(embs, refs, threshold=0.5)
    assert names == {"S0": "Neuro", "S1": "Neuro", "S2": "Vedal"}
    assert scores["S3"]["Neuro"] < 0


def test_to_lines_merges_names_and_filters():
    turns = [
        {"speaker": "S0", "text": "hello chat", "score": 0.9},
        {"speaker": "S1", "text": "hello chat", "score": 0.9},  # same person as S0
        {"speaker": "S2", "text": "mumble", "score": 0.2},
        {"speaker": "S3", "text": "hi neuro", "score": 0.8},
    ]
    names = {"S0": "Neuro", "S1": "Neuro"}
    assert tx.to_lines(turns, names, min_score=0.5) == ["Neuro: hello chat hello chat", "S3: hi neuro"]
    assert tx.to_lines(turns, names, other="Guest")[-1] == "Guest: mumble hi neuro"


def test_parse_pairs():
    assert tx.parse_pairs(["A=x,B=y", "C=z"]) == {"A": "x", "B": "y", "C": "z"}
    with pytest.raises(ValueError):
        tx.parse_pairs(["oops"])


def test_relabel_end_to_end(tmp_path):
    turns = [
        {"speaker": "SPEAKER_00", "text": "I am very smart", "score": 0.9, "start": 0, "end": 2},
        {"speaker": "SPEAKER_01", "text": "you are not", "score": 0.9, "start": 2, "end": 3},
    ]
    for stem in ("a", "b"):
        (tmp_path / f"{stem}.json").write_text(json.dumps({"turns": turns, "embeddings": {}}))
    out = tmp_path / "raw"
    run = lambda *extra: subprocess.run(  # noqa: E731
        [sys.executable, str(ROOT / "training/transcribe.py"), "--relabel", *extra,
         "--out", str(out), "--transcripts", str(tmp_path)],
        check=True, capture_output=True, text=True,
    )
    run(str(tmp_path / "a.json"), "--names", "SPEAKER_00=Neuro")
    assert (out / "a.txt").read_text() == "Neuro: I am very smart\nSPEAKER_01: you are not\n"

    # A second run keeps the saved name and adds the new one.
    run(str(tmp_path / "a.json"), str(tmp_path / "b.json"), "--names", "SPEAKER_01=Vedal")
    assert (out / "a.txt").read_text() == "Neuro: I am very smart\nVedal: you are not\n"
    # Names saved for a.json must not leak into b.json.
    assert (out / "b.txt").read_text() == "SPEAKER_00: I am very smart\nVedal: you are not\n"

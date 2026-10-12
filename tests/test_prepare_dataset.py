import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
import prepare_dataset as pd  # noqa: E402

PERSONA = {"name": "Neuro", "creator": "Vedal"}


def test_srt_and_speakers(tmp_path):
    (tmp_path / "clip.srt").write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nVedal: say hi\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nhi chat\n\n"
        "3\n00:00:03,000 --> 00:00:04,000\n<i>I am very smart</i>\n"
    )
    rows = pd.build(tmp_path, PERSONA, default_speaker="Neuro", window=8)
    assert len(rows) == 1
    msgs = rows[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[1]["content"] == "Vedal: say hi"
    assert msgs[2]["content"] == "hi chat I am very smart"


def test_windows_end_on_assistant_and_start_on_user():
    turns = pd.to_turns(
        [("Neuro", "a"), ("chat", "b"), ("Neuro", "c"), ("chat", "d")], "Neuro"
    )
    samples = pd.windows(turns, size=2)
    assert all(s[-1]["role"] == "assistant" for s in samples)
    assert all(s[0]["role"] == "user" for s in samples)
    assert len(samples) == 2


def test_jsonl_passthrough(tmp_path):
    row = {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]}
    (tmp_path / "extra.jsonl").write_text(json.dumps(row) + "\n")
    assert pd.build(tmp_path, PERSONA, "Neuro", 8) == [row]

# Fine-tuning

The goal: teach a small open model Neuro's voice (timing, humour, how she
talks to chat and to Vedal), then run it locally through Ollama.

**Set expectations:** the real Neuro is a private model with years of tuning,
memory and tooling behind it. A 3-8B LoRA fine-tune gets the *style* convincingly
close. It will not match her knowledge or long-term consistency.

## 1. Collect data

Put files in `data/raw/` (git-ignored, never committed):

- `.txt` or `.srt` transcripts with `Speaker: text` lines:
  ```
  chat: neuro what's 2+2
  Neuro: Four. Unless Vedal coded me, then it's five.
  Vedal: I did code you.
  Neuro: Exactly.
  ```
  Lines with no speaker are assumed to be hers (change with `--default-speaker`).
- `.jsonl` you wrote yourself, one `{"messages": [...]}` per line.

Start each filename with the stream date (`2024-05-12_karaoke.txt`) so you
can train on one era of her personality later. Quality beats quantity. A few thousand clean, correctly attributed lines of
her speech beat 100k lines where you can't tell who is talking. Auto-generated
subtitles mix speakers up, so fix attribution before training. Keep the data
private: the transcripts are other people's content, and this project stays
personal.

### Turning stream recordings into transcripts

`training/transcribe.py` takes audio or video files and writes `Speaker: text`
lines into `data/raw/` for you. It works out *who* is talking by voice, then
names each voice.

**One-time setup** (full details at the top of `transcribe.py`):

1. Make a free Hugging Face account and create a **read** token at
   https://huggingface.co/settings/tokens.
2. Open https://huggingface.co/pyannote/speaker-diarization-community-1 and
   click **Agree and access repository**.

**Make reference clips** (this lets it name speakers automatically):

1. Cut 10-60 seconds where *only* Neuro talks, with no music or game audio, and
   save it as `refs/neuro.wav`.
2. Do the same for Vedal (`refs/vedal.wav`) and anyone else who appears often.

**Run it on a free Colab GPU** (recommended; your own PC also works without an NVIDIA GPU, just slowly):

1. In a new Colab notebook, choose **Runtime -> Change runtime type -> T4 GPU**.
2. Upload this repo's `training/transcribe.py`, your `refs/` clips and your
   recordings using the folder icon on the left.
3. Run these in a cell, replacing `hf_xxx` with your token:
   ```
   !pip install whisperx
   !python transcribe.py recordings/ --hf-token hf_xxx --voice Neuro=refs/neuro.wav --voice Vedal=refs/vedal.wav --other Chat
   ```
4. Download the `data/raw/` and `data/transcripts/` folders, and put them in
   the same place in your copy of this repo.

For each speaker it prints a similarity score per reference voice, so you can
see how sure it is. Then **open a couple of the `.txt` files and check them**. If
a voice was named wrong, fix the names without re-running anything:

```bash
python training/transcribe.py --relabel data/transcripts/2024-05-12_stream.json --names SPEAKER_03=Vedal
```

Useful options: `--ask` (it shows sample lines and asks who each voice is),
`--min-score 0.5` (drops low-confidence lines such as music, singing or
overlapping speech) and `--max-speakers 3` (if you know how many people are
in the stream).

Then:

```bash
python training/prepare_dataset.py data/raw data/train --persona Neuro --creator Vedal
```

This writes `data/train/train.jsonl` and `data/train/valid.jsonl`.

## 2a. Mac only: train on Apple Silicon (16 GB+ RAM)

Skip this on anything that isn't an Apple Silicon Mac (including the ROG Ally) and use 2b.

```bash
pip install mlx-lm
mlx_lm.lora --model Qwen/Qwen2.5-3B-Instruct --train --data data/train \
  --iters 1000 --batch-size 2 --num-layers 16 --adapter-path adapters
# Try it:
mlx_lm.generate --model Qwen/Qwen2.5-3B-Instruct --adapter-path adapters \
  --prompt "chat: neuro say something evil"
# Merge the adapter into the model:
mlx_lm.fuse --model Qwen/Qwen2.5-3B-Instruct --adapter-path adapters --save-path outputs/neuro
```

Then in `training/Modelfile` change the first line to `FROM ../outputs/neuro`.

## 2b. Train on a free Colab GPU (ROG Ally, Linux, Windows, anything)

Follow the steps at the top of [`train_colab.py`](train_colab.py). You get a
`.gguf` file; put it next to `training/Modelfile` and make the `FROM` line
point at it.

## 3. Load it into Ollama

```bash
ollama create neuro -f training/Modelfile --quantize q4_K_M
```

(Leave out `--quantize` for the Colab `.gguf`, which is already quantized.)

Then in `config.yaml` set `llm.model: neuro`. Optionally shorten
`persona.py`'s prompt, since the model now knows the style itself.

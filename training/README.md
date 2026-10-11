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

Quality beats quantity. A few thousand clean, correctly attributed lines of
her speech beat 100k lines where you can't tell who is talking. Auto-generated
subtitles mix speakers up, so fix attribution before training. Keep the data
private: the transcripts are other people's content, and this project stays
personal.

Then:

```bash
python training/prepare_dataset.py data/raw data/train --persona Neuro --creator Vedal
```

This writes `data/train/train.jsonl` and `data/train/valid.jsonl`.

## 2a. Train on your Mac (Apple Silicon, 16 GB+ RAM)

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

## 2b. Or train on a free Colab GPU

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

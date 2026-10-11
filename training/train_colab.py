"""LoRA fine-tune on a free Google Colab GPU with Unsloth, export to GGUF.

How to use (step by step):
  1. Go to https://colab.research.google.com and create a new notebook.
  2. Runtime -> Change runtime type -> T4 GPU -> Save.
  3. In the first cell run:   !pip install unsloth
  4. Upload train.jsonl and valid.jsonl (folder icon on the left -> upload).
  5. Paste this whole file into the next cell and run it (~20-60 min).
  6. Download neuro-gguf/*.gguf from the file panel to your Mac and follow
     "Load it into Ollama" in training/README.md.
"""

BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct"  # 3B fits a free T4; 7B/8B also works
MAX_SEQ_LEN = 2048
EPOCHS = 2
OUT_DIR = "neuro-gguf"

from datasets import load_dataset  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402
from unsloth import FastLanguageModel  # noqa: E402
from unsloth.chat_templates import train_on_responses_only  # noqa: E402

model, tokenizer = FastLanguageModel.from_pretrained(
    BASE_MODEL, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
)

data = load_dataset(
    "json", data_files={"train": "train.jsonl", "valid": "valid.jsonl"}
)
data = data.map(
    lambda row: {"text": tokenizer.apply_chat_template(row["messages"], tokenize=False)}
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=data["train"],
    eval_dataset=data["valid"],
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=EPOCHS,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="epoch",
        output_dir="outputs",
        report_to="none",
    ),
)
# Only learn from her lines, not from chat / other speakers.
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
trainer.train()

model.save_pretrained_gguf(OUT_DIR, tokenizer, quantization_method="q4_k_m")
print(f"Done. Download the .gguf file from {OUT_DIR}/")

import os
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model  # type: ignore
from datasets import Dataset

#
# ---- Config ----
base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Get the script directory and construct absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
train_path = os.path.join(project_root, "data", "train.jsonl")
output_dir = os.path.join(project_root, "outputs", "twinkle-lora-mps")
max_len = 512

# ---- Data ----
# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

ds: Dataset = load_dataset("json", data_files={"train": train_path})["train"]  # type: ignore


def format_example(ex):
    msgs = ex["messages"]
    chunks = []
    for m in msgs:
        role = m["role"]
        if role == "system":
            chunks.append(f"<|system|>\n{m['content']}\n")
        elif role == "user":
            chunks.append(f"<|user|>\n{m['content']}\n")
        elif role == "assistant":
            chunks.append(f"<|assistant|>\n{m['content']}\n")
    return {"text": "".join(chunks).strip() + "\n"}


ds = ds.map(format_example, remove_columns=ds.column_names)

# ---- Tokenizer & model (no 4-bit) ----
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Use MPS if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32  # Use float32 for MPS compatibility

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=dtype,
)
model.to(device)

# ---- LoRA (light adapter layers) ----
peft_cfg = LoraConfig(
    r=16,  # Rank of adaptation - lower values use less memory but may reduce performance Range: 4-64
    lora_alpha=32,  # LoRA scaling parameter - controls the magnitude of adapter weights (typically 2x the rank)
    lora_dropout=0.05,  # Dropout rate for LoRA layers - helps prevent overfitting (5% dropout)
    target_modules=[  # Specific model layers to apply LoRA adapters to
        "q_proj",  # Query projection in attention
        "k_proj",  # Key projection in attention
        "v_proj",  # Value projection in attention
        "o_proj",  # Output projection in attention
        "gate_proj",  # Gate projection in feed-forward network
        "up_proj",  # Up projection in feed-forward network
        "down_proj",  # Down projection in feed-forward network
    ],
    task_type="CAUSAL_LM",  # Type of task options:
    # - "CAUSAL_LM": For text generation (GPT-style models)
    # - "SEQ_2_SEQ_LM": For sequence-to-sequence tasks (T5, BART)
    # - "TOKEN_CLS": For token classification (NER, POS tagging)
    # - "SEQ_CLS": For sequence classification (sentiment analysis)
    # - "QUESTION_ANS": For question answering tasks
    # - "FEATURE_EXTRACTION": For embedding extraction
)
model = get_peft_model(model, peft_cfg)

# ---- Train ----
args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=2,  # keep small on Mac
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_steps=200,
    fp16=False,  # MPS doesn't support fp16 mixed precision
    bf16=False,
    optim="adamw_torch",
    report_to="none",
    dataloader_pin_memory=False,  # Disable pin_memory for MPS compatibility
    # SFT-specific parameters (no longer passed to SFTTrainer)
    dataset_text_field="text",
    max_seq_length=max_len,
    packing=False,  # Set to False when using DataCollatorForCompletionOnlyLM
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    args=args,
)

trainer.train()  # type: ignore
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print("Saved to", output_dir)

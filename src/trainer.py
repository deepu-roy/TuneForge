import os
import argparse
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model  # type: ignore
from datasets import Dataset

#
# ---- Parse Arguments ----
parser = argparse.ArgumentParser(description="Fine-tune TinyLlama with LoRA")
parser.add_argument(
    "--train-data",
    type=str,
    default=None,
    help="Path to training data JSONL file (default: data/train.jsonl)",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Output directory for the fine-tuned model (default: outputs/adaptor/fine-tuned-model-mps)",
)
parser.add_argument(
    "--base-model",
    type=str,
    default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    help="Base model to fine-tune (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)",
)
parser.add_argument(
    "--max-seq-length",
    type=int,
    default=512,
    help="Maximum sequence length (default: 512)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=3,
    help="Number of training epochs (default: 3)",
)
parser.add_argument(
    "--learning-rate",
    type=float,
    default=2e-4,
    help="Learning rate (default: 2e-4)",
)

args = parser.parse_args()

# ---- Config ----
base_model = args.base_model

# Get the script directory and construct absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Use provided train path or default
if args.train_data:
    train_path = (
        args.train_data
        if os.path.isabs(args.train_data)
        else os.path.join(os.getcwd(), args.train_data)
    )
else:
    train_path = os.path.join(project_root, "data", "train.jsonl")

# Use provided output dir or default
if args.output_dir:
    output_dir = (
        args.output_dir
        if os.path.isabs(args.output_dir)
        else os.path.join(os.getcwd(), args.output_dir)
    )
else:
    output_dir = os.path.join(
        project_root, "outputs", "adaptor", "fine-tuned-model-mps"
    )

max_len = args.max_seq_length

print("Training configuration:")
print(f"  Base model: {base_model}")
print(f"  Training data: {train_path}")
print(f"  Output directory: {output_dir}")
print(f"  Max sequence length: {max_len}")
print(f"  Epochs: {args.epochs}")
print(f"  Learning rate: {args.learning_rate}")
print()

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
training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=2,  # keep small on Mac
    gradient_accumulation_steps=8,
    num_train_epochs=args.epochs,
    learning_rate=args.learning_rate,
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
    args=training_args,
)

trainer.train()  # type: ignore
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print("Saved to", output_dir)

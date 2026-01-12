import os
import sys
import argparse
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model  # type: ignore
from datasets import Dataset

# ============================================
# Training Hyperparameters
# ============================================
# Optimized for Apple Silicon MPS, adjust for other hardware
DEFAULT_BATCH_SIZE = 2  # Small batch size for memory efficiency on Mac
DEFAULT_GRADIENT_ACCUMULATION = 2  # Effective batch size = 2 * 2 = 4, If the samples size is large, increase this
DEFAULT_WARMUP_RATIO = 0.1  # 10% of training for learning rate warmup
DEFAULT_LOGGING_STEPS = 10  # Log every 10 steps for monitoring
DEFAULT_CHECKPOINT_STEPS = 200  # Save checkpoint every 200 steps

# ============================================
# LoRA Configuration Constants
# ============================================
DEFAULT_LORA_R = 64  # Rank of adaptation (range: 4-64, lower uses less memory)
DEFAULT_LORA_ALPHA = 128  # LoRA scaling parameter (typically 2x the rank)
DEFAULT_LORA_DROPOUT = 0.05  # Dropout rate for LoRA layers (5% dropout)

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
try:
    os.makedirs(output_dir, exist_ok=True)
except PermissionError:
    print(f"❌ Error: No permission to create output directory: {output_dir}")
    sys.exit(1)

# Load and validate training data
try:
    ds: Dataset = load_dataset("json", data_files={"train": train_path})["train"]  # type: ignore
except FileNotFoundError:
    print(f"❌ Error: Training data file not found: {train_path}")
    print("💡 Tip: Check the file path and ensure the file exists")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error loading training data: {e}")
    print("💡 Tip: Ensure the file is valid JSONL format")
    sys.exit(1)

# Validate training data format
if len(ds) == 0:
    print("❌ Error: Training data is empty")
    sys.exit(1)

if "text" not in ds.column_names:
    print("❌ Error: Training data must have 'text' field")
    print("💡 Expected format: {\"text\":\"<|system|>...<|user|>...<|assistant|>...</s>\"}")
    sys.exit(1)

# Data is already in the expected format with special tokens, no conversion needed

# ---- Tokenizer & model (no 4-bit) ----
try:
    print(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)  # Getting the tokenizer from the basemodel
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
except OSError:
    print(f"❌ Error: Tokenizer for '{base_model}' not found")
    print("💡 Tip: Check the model name on HuggingFace or verify your authentication")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error loading tokenizer: {e}")
    sys.exit(1)

# Use MPS if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32  # Use float32 for MPS compatibility

try:
    print(f"Loading base model: {base_model}")
    print(f"Using device: {device}, dtype: {dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
    )  # We load the base model for causal language modeling
    model.to(device)
    print("✅ Model loaded successfully")
except OSError:
    print(f"❌ Error: Model '{base_model}' not found")
    print("💡 Tip: Check the model name on HuggingFace (e.g., 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')")
    print("💡 Tip: For private models, set your HuggingFace token: huggingface-cli login")
    sys.exit(1)
except RuntimeError as e:
    print(f"❌ Error: Failed to load model to device '{device}': {e}")
    print("💡 Tip: Try reducing model size or check available memory")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# ---- LoRA (light adapter layers) ----
peft_cfg = LoraConfig(
    r=DEFAULT_LORA_R,
    lora_alpha=DEFAULT_LORA_ALPHA,
    lora_dropout=DEFAULT_LORA_DROPOUT,
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
model = get_peft_model(model, peft_cfg) # This freezes the base model and adds LoRA matrices (A and B)

# ---- Train ----
training_args = SFTConfig( # Gets the supervised fine-tuning configuration, overiding the origininal SFTConfig parameters with our own
    output_dir=output_dir,
    per_device_train_batch_size=DEFAULT_BATCH_SIZE,
    gradient_accumulation_steps=DEFAULT_GRADIENT_ACCUMULATION,
    num_train_epochs=args.epochs,
    learning_rate=args.learning_rate,
    lr_scheduler_type="cosine",
    warmup_ratio=DEFAULT_WARMUP_RATIO,
    logging_steps=DEFAULT_LOGGING_STEPS,
    save_steps=DEFAULT_CHECKPOINT_STEPS,
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

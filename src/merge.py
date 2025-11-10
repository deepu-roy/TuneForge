import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # type: ignore

# Parse command line arguments
parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
parser.add_argument(
    "--base-model",
    type=str,
    default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    help="Base model to merge with (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)",
)
parser.add_argument(
    "--adapter",
    type=str,
    default="outputs/adaptor/fine-tuned-model-mps",
    help="Path to adapter folder (default: outputs/adaptor/fine-tuned-model-mps)",
)
parser.add_argument(
    "--output",
    type=str,
    default="outputs/merged/merged-fine-tuned-model",
    help="Output directory for merged model (default: outputs/merged/merged-fine-tuned-model)",
)

args = parser.parse_args()

BASE = args.base_model
ADAPTER = args.adapter
OUT = args.output

print("Merge configuration:")
print(f"  Base model: {BASE}")
print(f"  Adapter: {ADAPTER}")
print(f"  Output: {OUT}")
print()

# Create output directory if it doesn't exist
os.makedirs(OUT, exist_ok=True)

dtype = torch.float32 if torch.backends.mps.is_available() else torch.float32

print("Loading base...")
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dtype)
tok = AutoTokenizer.from_pretrained(BASE)

print("Attaching adapter...")
model = PeftModel.from_pretrained(base, ADAPTER)

print("Merging LoRA weights...")
model = model.merge_and_unload()  # type: ignore # applies LoRA deltas into base weights

print("Saving merged model...")
model.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)

print(f"Done. Merged model at: {OUT}")

import argparse
import os
import sys
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
try:
    os.makedirs(OUT, exist_ok=True)
except PermissionError:
    print(f"❌ Error: No permission to create output directory: {OUT}")
    sys.exit(1)

# Validate adapter path exists
if not os.path.exists(ADAPTER):
    print(f"❌ Error: Adapter path not found: {ADAPTER}")
    print("💡 Tip: Run training first or check the adapter path")
    sys.exit(1)

if not os.path.isdir(ADAPTER):
    print(f"❌ Error: Adapter path is not a directory: {ADAPTER}")
    sys.exit(1)

# Use float32 for MPS compatibility, float16 for CUDA if available
if torch.backends.mps.is_available():
    dtype = torch.float32
elif torch.cuda.is_available():
    dtype = torch.float16
else:
    dtype = torch.float32

try:
    print("Loading base...")
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dtype)
    tok = AutoTokenizer.from_pretrained(BASE)
    print("✅ Base model loaded successfully")
except OSError:
    print(f"❌ Error: Base model '{BASE}' not found")
    print("💡 Tip: Check the model name on HuggingFace")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error loading base model: {e}")
    sys.exit(1)

try:
    print("Attaching adapter...")
    model = PeftModel.from_pretrained(base, ADAPTER)
    print("✅ Adapter loaded successfully")
except Exception as e:
    print(f"❌ Error loading adapter: {e}")
    print("💡 Tip: Ensure the adapter is compatible with the base model")
    sys.exit(1)

try:
    print("Merging LoRA weights...")
    model = model.merge_and_unload()  # type: ignore # applies LoRA deltas into base weights
    print("✅ Merge completed successfully")
except Exception as e:
    print(f"❌ Error during merge: {e}")
    sys.exit(1)

try:
    print("Saving merged model...")
    model.save_pretrained(OUT, safe_serialization=True)
    tok.save_pretrained(OUT)
    print(f"✅ Done. Merged model at: {OUT}")
except Exception as e:
    print(f"❌ Error saving merged model: {e}")
    print("💡 Tip: Check disk space and write permissions")
    sys.exit(1)

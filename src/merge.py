import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # type: ignore

BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # your base
ADAPTER = "outputs/twinkle-lora-mps"  # your adapter folder
OUT = "merged-twinkle-llama"  # output dir

dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32

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

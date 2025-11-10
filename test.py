from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

ckpt = "outputs/merged/merged-fine-tuned-model"
tok = AutoTokenizer.from_pretrained(ckpt)
mdl = AutoModelForCausalLM.from_pretrained(ckpt)
device = "mps" if torch.backends.mps.is_available() else "cpu"
mdl.to(device)

# prompt = "<|user|>\nComplete the rhyme: Twinkle, twinkle, little star,\n<|assistant|>\n"
prompt = "<|user|>\nComplete the rest of the rhyme: As your bright and tiny spark,\n<|assistant|>\n"
inputs = tok(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    out = mdl.generate(
        **inputs,
        max_new_tokens=24,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
print(tok.decode(out[0], skip_special_tokens=False))

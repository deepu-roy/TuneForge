#!/usr/bin/env python3
"""
Test script for fine-tuned models.
Usage: python test.py [--model MODEL_PATH] [--prompt PROMPT] [--max-tokens N]
"""

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Test a fine-tuned model with a custom prompt"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/merged/merged-fine-tuned-model",
        help="Path to the merged model directory (default: outputs/merged/merged-fine-tuned-model)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Complete the rhyme: Twinkle, twinkle, little star,",
        help="Prompt to test the model with",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate (default: 50)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7, use 0 for greedy)",
    )

    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(args.model)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    mdl.to(device)

    # Format prompt with chat template
    formatted_prompt = f"<|user|>\n{args.prompt}\n<|assistant|>\n"
    print(f"\nPrompt: {args.prompt}")
    print("-" * 60)

    inputs = tok(formatted_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            pad_token_id=tok.eos_token_id,
        )

    response = tok.decode(out[0], skip_special_tokens=True)

    # Extract just the assistant's response
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()

    print(f"Response:\n{response}")
    print("-" * 60)


if __name__ == "__main__":
    main()

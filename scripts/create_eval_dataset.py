#!/usr/bin/env python3
"""
TuneForge - Create Evaluation Dataset and Run Ollama Evaluation
Transforms training data into evaluation format and optionally runs against Ollama model.
Usage:
  # Create eval dataset only
  python scripts/create_eval_dataset.py --input data/qa-training.jsonl --output data/qa-eval.jsonl

  # Create and run evaluation against Ollama
  python scripts/create_eval_dataset.py --input data/qa-training.jsonl --output data/qa-eval.jsonl --run-ollama --model qa-model:latest
"""

import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any
import sys
import time


def extract_code_block(text: str) -> str:
    """Extract code from markdown code blocks"""
    if "```" in text:
        # Find content between ``` markers
        parts = text.split("```")
        if len(parts) >= 3:
            # Get the code part (between first and second ```)
            code_part = parts[1]
            # Remove language identifier (ts, typescript, etc.)
            if "\n" in code_part:
                lines = code_part.split("\n", 1)
                if lines[0].strip() in ["ts", "typescript", "javascript", "js"]:
                    return lines[1].strip() if len(lines) > 1 else code_part.strip()
            return code_part.strip()
    return text


def parse_qa_response(response: str) -> Dict[str, str]:
    """Parse QA Mentor response into structured sections"""
    sections = {"summary": "", "steps": "", "code": "", "risks": ""}

    lines = response.strip().split("\n")
    current_section = None
    content_lines = []

    for line in lines:
        line_lower = line.lower().strip()

        # Detect section headers
        if line_lower.startswith("summary:"):
            if current_section and content_lines:
                sections[current_section] = "\n".join(content_lines).strip()
            current_section = "summary"
            content_lines = [line.split(":", 1)[1].strip() if ":" in line else ""]
        elif line_lower.startswith("steps:"):
            if current_section and content_lines:
                sections[current_section] = "\n".join(content_lines).strip()
            current_section = "steps"
            content_lines = []
        elif line_lower.startswith("code:"):
            if current_section and content_lines:
                sections[current_section] = "\n".join(content_lines).strip()
            current_section = "code"
            content_lines = []
        elif line_lower.startswith("risks:"):
            if current_section and content_lines:
                sections[current_section] = "\n".join(content_lines).strip()
            current_section = "risks"
            content_lines = [line.split(":", 1)[1].strip() if ":" in line else ""]
        else:
            if current_section:
                content_lines.append(line)

    # Save last section
    if current_section and content_lines:
        sections[current_section] = "\n".join(content_lines).strip()

    # Clean up code section
    if sections["code"]:
        sections["code"] = extract_code_block(sections["code"])

    return sections


def create_eval_entry(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Transform a training item into evaluation format"""
    messages = item.get("messages", [])
    assistant_response = item.get("assistant", "")

    # Extract system prompt, user query
    system_content = ""
    user_query = ""

    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        elif msg["role"] == "user":
            user_query = msg["content"]

    # Parse the assistant response into sections
    parsed = parse_qa_response(assistant_response)

    # Create evaluation entry
    eval_entry = {
        "id": f"qa_eval_{index:03d}",
        "input": {"query": user_query, "system_prompt": system_content},
        "output": {
            "answer": assistant_response,
            "structured": {
                "summary": parsed["summary"],
                "steps": parsed["steps"],
                "code": parsed["code"],
                "risks": parsed["risks"],
            },
        },
        "reference": {
            "expected": assistant_response,
            "format": "QA Mentor format with Summary, Steps, Code (TypeScript Playwright), and Risks sections",
        },
        "context": [
            {
                "text": "QA Mentor is a specialized assistant that provides software QA guidance in a strict format: Summary, Steps (3 numbered items), Code (TypeScript Playwright), and Risks.",
                "source": "system_instructions",
            },
            {"text": f"User Query: {user_query}", "source": "user_input"},
        ],
    }

    return eval_entry


def query_ollama(model: str, prompt: str, system_prompt: str = "") -> str:
    """Query Ollama model and return response"""
    try:
        cmd = ["ollama", "run", model]

        # Combine system and user prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        result = subprocess.run(
            cmd, input=full_prompt, capture_output=True, text=True, timeout=60
        )

        if result.returncode != 0:
            return f"ERROR: {result.stderr}"

        return result.stdout.strip()

    except subprocess.TimeoutExpired:
        return "ERROR: Timeout waiting for response"
    except Exception as e:
        return f"ERROR: {str(e)}"


def run_ollama_evaluation(eval_data: list, model: str, output_path: Path) -> int:
    """Run evaluation against Ollama model and update dataset with actual outputs"""
    print(f"\n🤖 Running evaluation against Ollama model: {model}")
    print(f"   Total test cases: {len(eval_data)}")

    # Check if ollama is available
    try:
        subprocess.run(["ollama", "list"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: Ollama not found. Please install Ollama first.")
        return 1

    # Check if model exists
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if model not in result.stdout:
        print(f"❌ Error: Model '{model}' not found in Ollama.")
        print("   Available models:")
        print(result.stdout)
        return 1

    print(f"✅ Model '{model}' found")
    print("\n🔄 Running evaluations...")

    success_count = 0
    error_count = 0

    for idx, entry in enumerate(eval_data, start=1):
        query = entry["input"]["query"]
        system_prompt = entry["input"].get("system_prompt", "")

        print(f"\n[{idx}/{len(eval_data)}] Querying model...")
        print(f"   Query: {query[:60]}...")

        start_time = time.time()
        response = query_ollama(model, query, system_prompt)
        elapsed = time.time() - start_time

        if response.startswith("ERROR:"):
            print(f"   ❌ Failed: {response}")
            error_count += 1
            entry["output"]["model_response"] = response
            entry["output"]["success"] = False
        else:
            print(f"   ✅ Success ({elapsed:.1f}s)")
            print(f"   Response length: {len(response)} chars")
            success_count += 1

            # Parse the response
            parsed = parse_qa_response(response)

            # Update output with actual model response
            entry["output"]["model_response"] = response
            entry["output"]["model_parsed"] = parsed
            entry["output"]["success"] = True
            entry["output"]["response_time"] = elapsed

            # Check if response has all required sections
            has_all_sections = all(
                parsed.get(key) for key in ["summary", "steps", "code", "risks"]
            )
            entry["output"]["has_all_sections"] = has_all_sections

            if not has_all_sections:
                print(
                    f"   ⚠️  Warning: Missing sections - {[k for k in ['summary', 'steps', 'code', 'risks'] if not parsed.get(k)]}"
                )

    # Save updated evaluation data
    print(f"\n💾 Saving evaluation results to: {output_path}")
    with open(output_path, "w") as f:
        for entry in eval_data:
            f.write(json.dumps(entry) + "\n")

    print(f"\n📊 Evaluation Summary:")
    print(f"   Total: {len(eval_data)}")
    print(f"   Success: {success_count}")
    print(f"   Errors: {error_count}")
    print(f"   Success rate: {success_count / len(eval_data) * 100:.1f}%")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Create evaluation dataset from QA training data and optionally run Ollama evaluation"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/qa-training.jsonl",
        help="Input training JSONL file (default: data/qa-training.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/qa-eval.jsonl",
        help="Output evaluation JSONL file (default: data/qa-eval.jsonl)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of samples to include (default: all)",
    )
    parser.add_argument(
        "--run-ollama", action="store_true", help="Run evaluation against Ollama model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qa-model:latest",
        help="Ollama model to use for evaluation (default: qa-model:latest)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        return 1

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📖 Reading training data from: {input_path}")

    # Read training data
    training_data = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                training_data.append(json.loads(line))

    print(f"✅ Found {len(training_data)} training examples")

    # Optionally sample
    if args.sample and args.sample < len(training_data):
        training_data = training_data[: args.sample]
        print(f"📊 Using {args.sample} samples")

    # Transform to evaluation format
    print("🔄 Transforming to evaluation format...")
    eval_data = []
    for idx, item in enumerate(training_data, start=1):
        eval_entry = create_eval_entry(item, idx)
        eval_data.append(eval_entry)

    # Write evaluation data (initial version)
    print(f"💾 Writing evaluation data to: {output_path}")
    with open(output_path, "w") as f:
        for entry in eval_data:
            f.write(json.dumps(entry) + "\n")

    print(f"✅ Created {len(eval_data)} evaluation examples")
    print("\n📋 Evaluation dataset structure:")
    print("   - ID: Unique identifier for each test case")
    print("   - Input: User query and system prompt")
    print("   - Output: Full answer with structured sections")
    print("   - Reference: Expected format and content")
    print("   - Context: Background information")

    # Show sample
    if eval_data:
        print("\n🔍 Sample entry (first):")
        sample = eval_data[0]
        print(f"   ID: {sample['id']}")
        print(f"   Query: {sample['input']['query'][:80]}...")
        print(f"   Has structured output: {bool(sample['output']['structured'])}")

    # Run Ollama evaluation if requested
    if args.run_ollama:
        return run_ollama_evaluation(eval_data, args.model, output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

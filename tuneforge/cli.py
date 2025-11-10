#!/usr/bin/env python3
"""
TuneForge CLI - Command-line interface for fine-tuning LLMs
"""

import sys
import argparse
import subprocess
from pathlib import Path


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"\n🔨 {description}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(cmd, cwd=get_project_root())

    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        sys.exit(result.returncode)

    print(f"\n✅ Completed: {description}")
    return result.returncode


def train_command(args):
    """Run the training pipeline"""
    cmd = ["python", "src/trainer.py"]

    if args.train_data:
        cmd.extend(["--train-data", args.train_data])
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    if args.base_model:
        cmd.extend(["--base-model", args.base_model])
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.learning_rate:
        cmd.extend(["--learning-rate", str(args.learning_rate)])
    if args.max_seq_length:
        cmd.extend(["--max-seq-length", str(args.max_seq_length)])

    return run_command(cmd, "Training model with LoRA")


def merge_command(args):
    """Run the merge pipeline"""
    cmd = ["python", "src/merge.py"]

    if args.base_model:
        cmd.extend(["--base-model", args.base_model])
    if args.adapter:
        cmd.extend(["--adapter", args.adapter])
    if args.output:
        cmd.extend(["--output", args.output])

    return run_command(cmd, "Merging adapter with base model")


def convert_command(args):
    """Run the GGUF conversion"""
    if not args.llama_cpp_path:
        print("❌ Error: --llama-cpp-path is required for conversion")
        sys.exit(1)

    llama_cpp = Path(args.llama_cpp_path)
    convert_script = llama_cpp / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        print(f"❌ Error: convert_hf_to_gguf.py not found in {llama_cpp}")
        sys.exit(1)

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(convert_script),
        args.model,
        "--outtype",
        args.outtype,
        "--outfile",
        args.output,
    ]

    return run_command(cmd, f"Converting to GGUF format ({args.outtype})")


def pipeline_command(args):
    """Run the complete pipeline"""
    project_root = get_project_root()
    script = project_root / "train-and-convert.sh"

    cmd = [str(script)]

    if args.config:
        cmd.extend(["--config", args.config])
    if args.model_name:
        cmd.extend(["--model-name", args.model_name])
    if args.llama_cpp_path:
        cmd.extend(["--llama-cpp-path", args.llama_cpp_path])
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.learning_rate:
        cmd.extend(["--learning-rate", str(args.learning_rate)])
    if args.gguf_outtype:
        cmd.extend(["--gguf-outtype", args.gguf_outtype])
    if args.skip_training:
        cmd.append("--skip-training")
    if args.skip_merge:
        cmd.append("--skip-merge")
    if args.skip_gguf:
        cmd.append("--skip-gguf")

    return run_command(cmd, "Running complete pipeline")


def test_command(args):
    """Test a fine-tuned model"""
    cmd = ["python", "test.py"]

    if args.model:
        cmd.extend(["--model", args.model])
    if args.prompt:
        cmd.extend(["--prompt", args.prompt])
    if args.max_tokens:
        cmd.extend(["--max-tokens", str(args.max_tokens)])
    if args.temperature:
        cmd.extend(["--temperature", str(args.temperature)])

    return run_command(cmd, "Testing model")


def create_eval_command(args):
    """Create evaluation dataset from training data"""
    cmd = ["python", "scripts/create_eval_dataset.py"]

    if args.input:
        cmd.extend(["--input", args.input])
    if args.output:
        cmd.extend(["--output", args.output])
    if args.sample:
        cmd.extend(["--sample", str(args.sample)])

    return run_command(cmd, "Creating evaluation dataset")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog="tuneforge",
        description="🔨 TuneForge - Automated pipeline for fine-tuning LLMs with LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline with config
  tuneforge pipeline --config config.env

  # Train only
  tuneforge train --train-data data/train.jsonl --epochs 5

  # Merge adapter with base model
  tuneforge merge --adapter outputs/adaptor/my-model --output outputs/merged/my-model

  # Convert to GGUF
  tuneforge convert --model outputs/merged/my-model --llama-cpp-path /path/to/llama.cpp

  # Test the model
  tuneforge test --model outputs/merged/my-model --prompt "Write a poem"

  # Create evaluation dataset
  tuneforge create-eval --input data/training.jsonl --output data/eval.jsonl

For more information, visit: https://github.com/deepu-roy/TuneForge
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline", help="Run the complete pipeline (train, merge, convert)"
    )
    pipeline_parser.add_argument("--config", help="Configuration file")
    pipeline_parser.add_argument("--model-name", help="Model name")
    pipeline_parser.add_argument("--llama-cpp-path", help="Path to llama.cpp")
    pipeline_parser.add_argument("--epochs", type=int, help="Number of epochs")
    pipeline_parser.add_argument("--learning-rate", type=float, help="Learning rate")
    pipeline_parser.add_argument("--gguf-outtype", help="GGUF quantization type")
    pipeline_parser.add_argument(
        "--skip-training", action="store_true", help="Skip training"
    )
    pipeline_parser.add_argument("--skip-merge", action="store_true", help="Skip merge")
    pipeline_parser.add_argument(
        "--skip-gguf", action="store_true", help="Skip GGUF conversion"
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model with LoRA")
    train_parser.add_argument("--train-data", help="Training data JSONL file")
    train_parser.add_argument("--output-dir", help="Output directory for adapter")
    train_parser.add_argument("--base-model", help="Base model to fine-tune")
    train_parser.add_argument("--epochs", type=int, help="Number of epochs")
    train_parser.add_argument("--learning-rate", type=float, help="Learning rate")
    train_parser.add_argument(
        "--max-seq-length", type=int, help="Maximum sequence length"
    )

    # Merge command
    merge_parser = subparsers.add_parser(
        "merge", help="Merge LoRA adapter with base model"
    )
    merge_parser.add_argument("--base-model", help="Base model path")
    merge_parser.add_argument("--adapter", help="Adapter path")
    merge_parser.add_argument("--output", help="Output directory for merged model")

    # Convert command
    convert_parser = subparsers.add_parser(
        "convert", help="Convert model to GGUF format"
    )
    convert_parser.add_argument(
        "--model", required=True, help="Model directory to convert"
    )
    convert_parser.add_argument("--output", required=True, help="Output GGUF file path")
    convert_parser.add_argument(
        "--llama-cpp-path", required=True, help="Path to llama.cpp"
    )
    convert_parser.add_argument(
        "--outtype", default="f16", help="Quantization type (default: f16)"
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Test a fine-tuned model")
    test_parser.add_argument("--model", help="Model directory to test")
    test_parser.add_argument("--prompt", help="Test prompt")
    test_parser.add_argument(
        "--max-tokens", type=int, help="Maximum tokens to generate"
    )
    test_parser.add_argument("--temperature", type=float, help="Sampling temperature")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate command
    if args.command == "pipeline":
        sys.exit(pipeline_command(args))
    elif args.command == "train":
        sys.exit(train_command(args))
    elif args.command == "merge":
        sys.exit(merge_command(args))
    elif args.command == "convert":
        sys.exit(convert_command(args))
    elif args.command == "test":
        sys.exit(test_command(args))
    elif args.command == "create-eval":
        sys.exit(create_eval_command(args))


if __name__ == "__main__":
    main()

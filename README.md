# TuneForge 🔨

> Automated pipeline for fine-tuning LLMs with LoRA and deploying to Ollama

TuneForge simplifies the process of fine-tuning large language models using LoRA (Low-Rank Adaptation) and automatically converts them to GGUF format for deployment with Ollama. With a single command, you can train, merge, and deploy your custom models.

## Features

✅ **One-Command Pipeline** - Train, merge, and convert with `tuneforge pipeline`  
✅ **Modular CLI** - Run individual steps: `tuneforge train`, `tuneforge merge`, `tuneforge convert`  
✅ **Configuration File Support** - Reusable configs with `config.env`  
✅ **Auto-Generated Modelfile** - Ready-to-use Ollama Modelfile with correct paths  
✅ **Flexible Quantization** - Support for f16, f32, q8_0, q4_0, and more  
✅ **Modern APIs** - Uses `SFTConfig` instead of deprecated `TrainingArguments`  
✅ **Apple Silicon Optimized** - MPS support for GPU acceleration on Mac  
✅ **No Warnings** - Clean training output without deprecation warnings  
✅ **Efficient Training** - LoRA enables fine-tuning with minimal memory  

## Prerequisites

- Python 3.12+
- uv package manager
- (Optional) llama.cpp for GGUF conversion
- (Optional) Ollama for running the model locally

## Setup

Install dependencies:

```bash
uv sync
```

This will install:

- `torch` (with MPS support for Apple Silicon)
- `transformers>=4.44`
- `datasets>=2.20`
- `peft>=0.12` (for LoRA)
- `accelerate>=0.34`
- `trl>=0.9.6` (for SFTTrainer)

## Quick Start: Automated Pipeline 🚀

**The recommended way to fine-tuning a model is using the TuneForge CLI.** The pipeline handles the entire workflow from training to deployment with a single command.

### Install the app to run CLI commands

`pip install -e .`

### What the Pipeline Does

TuneForge automatically:

1. ✅ Trains your model with LoRA adapters
2. ✅ Merges the adapter with the base model
3. ✅ Converts to GGUF format for Ollama
4. ✅ Generates a ready-to-use Modelfile

### Method 1: Using the CLI (Recommended)

#### Basic Usage

```bash
# Run complete pipeline with config file
tuneforge pipeline --config config.env

# Run with specific settings
tuneforge pipeline --model-name my-model --llama-cpp-path /path/to/llama.cpp

# Skip specific steps
tuneforge pipeline --config config.env --skip-training --skip-merge
```

#### Individual Commands

```bash
# Train only
tuneforge train --train-data data/train.jsonl --epochs 5

# Merge adapter with base model
tuneforge merge \
  --adapter outputs/adaptor/my-model \
  --output outputs/merged/my-model

# Convert to GGUF
tuneforge convert \
  --model outputs/merged/my-model \
  --output outputs/gguf/my-model-f16.gguf \
  --llama-cpp-path /path/to/llama.cpp \
  --outtype f16

# Test the model
tuneforge test --model outputs/merged/my-model --prompt "Write a poem"
```

### Method 2: Using the Shell Script

Alternatively, you can use the `train-and-convert.sh` script directly:

#### Step 1: Create Your Configuration

Copy and customize the example config:

```bash
cp config.env.example config.env
```

Edit `config.env` to set your preferences:

```bash
# Model name (used in all output paths)
MODEL_NAME="my-custom-model"

# Training data
TRAIN_DATA="data/train.jsonl"

# Training parameters
EPOCHS=3
LEARNING_RATE=2e-4
MAX_SEQ_LENGTH=512

# GGUF quantization (f16, f32, q8_0, q4_0, etc.)
GGUF_OUTTYPE="f16"

# Path to llama.cpp (required for GGUF conversion)
LLAMA_CPP_PATH="/path/to/llama.cpp"

# Ollama Modelfile parameters
MODELFILE_TEMPERATURE=1
MODELFILE_TOP_P=0.9
MODELFILE_REPEAT_PENALTY=1.05
```

#### Step 2: Run the Pipeline

```bash
./train-and-convert.sh --config config.env
```

That's it! The script will:

- Train your model with the specified parameters
- Merge the LoRA adapter with the base model
- Convert to GGUF format
- Generate a `Modelfile` with the correct path and parameters

#### Step 3: Deploy to Ollama

After the pipeline completes, deploy your model:

```bash
ollama create my-custom-model -f Modelfile
ollama run my-custom-model "Your prompt here"
```

## Advanced Usage Examples

### CLI Examples

```bash
# Train with custom parameters
tuneforge train \
  --train-data data/my_data.jsonl \
  --base-model meta-llama/Llama-2-7b-hf \
  --epochs 5 \
  --learning-rate 1e-4 \
  --output-dir outputs/adaptor/my-7b-model

# Merge the trained adapter
tuneforge merge \
  --base-model meta-llama/Llama-2-7b-hf \
  --adapter outputs/adaptor/my-7b-model \
  --output outputs/merged/my-7b-model

# Convert with different quantizations
tuneforge convert \
  --model outputs/merged/my-7b-model \
  --output outputs/gguf/my-model-f16.gguf \
  --llama-cpp-path /path/to/llama.cpp \
  --outtype f16

# Complete pipeline with options
tuneforge pipeline \
  --model-name my-assistant \
  --epochs 5 \
  --gguf-outtype q8_0 \
  --llama-cpp-path /path/to/llama.cpp

# Skip steps
tuneforge pipeline --config config.env --skip-training --skip-merge
```

### Shell Script Examples

```bash
# Minimal - uses defaults for most settings
./train-and-convert.sh \
  --model-name rhyme-generator \
  --llama-cpp-path /path/to/llama.cpp

# Full customization
./train-and-convert.sh \
  --model-name sql-expert \
  --train-data data/sql_training.jsonl \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --epochs 5 \
  --learning-rate 1e-4 \
  --max-seq-length 1024 \
  --gguf-outtype q8_0 \
  --llama-cpp-path /path/to/llama.cpp

# Use config.env but override specific settings
./train-and-convert.sh --config config.env --epochs 5 --gguf-outtype q4_0

# Only merge and convert (skip training)
./train-and-convert.sh --config config.env --skip-training

# Create multiple quantization versions
./train-and-convert.sh --config config.env --skip-training --skip-merge --gguf-outtype q8_0
./train-and-convert.sh --config config.env --skip-training --skip-merge --gguf-outtype q4_0
```

## Configuration Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_NAME` | Base name for your model (used in all paths) | `fine-tuned-model` |
| `TRAIN_DATA` | Path to training JSONL file | `data/train.jsonl` |
| `BASE_MODEL` | HuggingFace model ID | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `EPOCHS` | Number of training epochs | `3` |
| `LEARNING_RATE` | Learning rate | `2e-4` |
| `MAX_SEQ_LENGTH` | Maximum sequence length | `512` |
| `GGUF_OUTTYPE` | Quantization type (f32, f16, q8_0, q4_0, etc.) | `f16` |
| `LLAMA_CPP_PATH` | Path to llama.cpp directory | (empty - required for GGUF) |
| `MODELFILE_TEMPERATURE` | Ollama temperature parameter | `1` |
| `MODELFILE_TOP_P` | Ollama top_p parameter | `0.9` |
| `MODELFILE_REPEAT_PENALTY` | Ollama repeat penalty | `1.05` |
| `SKIP_TRAINING` | Skip training step | `false` |
| `SKIP_MERGE` | Skip merge step | `false` |
| `SKIP_GGUF` | Skip GGUF conversion | `false` |

## Testing Your Model

### CLI Testing

```bash
# Default test
tuneforge test

# Custom prompt  
tuneforge test --prompt "Write a poem about the moon"

# Test different model
tuneforge test --model outputs/merged/my-other-model

# Adjust generation parameters
tuneforge test --max-tokens 100 --temperature 0.9
```

### Python Script Testing

Test the merged model directly with Python:

```bash
# Default test with sample prompt
uv run python test.py

# Custom prompt
uv run python test.py --prompt "Write a poem about the moon"

# Test a different model
uv run python test.py --model outputs/merged/my-other-model

# Adjust generation parameters
uv run python test.py --max-tokens 100 --temperature 0.9

# Get help
uv run python test.py --help
```

**Available arguments:**

- `--model` - Path to merged model directory (default: `outputs/merged/merged-fine-tuned-model`)
- `--prompt` - Custom prompt to test (default: rhyme completion)
- `--max-tokens` - Maximum tokens to generate (default: 50)
- `--temperature` - Sampling temperature, 0 for greedy decoding (default: 0.7)

### Ollama Testing

After running the pipeline, test with Ollama:

```bash
# The pipeline auto-generates the Modelfile
ollama create my-model -f Modelfile
ollama run my-model "Write a rhyme about stars"
```

## Model Evaluation

### Create Evaluation Dataset

Transform your training data into evaluation format:

```bash
# Create evaluation dataset
uv run python scripts/create_eval_dataset.py \
  --input data/qa-training.jsonl \
  --output data/qa-eval.jsonl
```

### Run Ollama Evaluation

Test your fine-tuned model against the evaluation dataset:

```bash
# Run full evaluation
uv run python scripts/create_eval_dataset.py \
  --input data/qa-training.jsonl \
  --output data/qa-eval-results.jsonl \
  --run-ollama \
  --model qa-model:latest

# Quick test with samples
uv run python scripts/create_eval_dataset.py \
  --input data/qa-training.jsonl \
  --output data/qa-eval-results.jsonl \
  --run-ollama \
  --model qa-model:latest \
  --sample 3
```

The evaluation script will:

- Query your Ollama model with each test case
- Capture actual responses
- Compare against expected outputs
- Track success rate and response times
- Generate detailed results in JSONL format

See `scripts/README.md` for more details on evaluation.

## Manual Steps (Advanced)

If you prefer to run each step manually instead of using the automated pipeline:

### 1. Prepare Training Data

Create your training data in `./data/train.jsonl` with the following format:

```json
{"messages":[
  {"role":"user","content":"Your prompt here"},
  {"role":"assistant","content":"Expected response"}
]}
```

Example data is already provided for training on "Twinkle Twinkle Little Star" rhymes.

### 2. Run the Trainer

Execute the training script with default settings:

```bash
uv run python src/trainer.py
```

Or customize with command-line arguments:

```bash
uv run python src/trainer.py \
  --train-data data/train.jsonl \
  --output-dir outputs/my-model \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --max-seq-length 512 \
  --epochs 3 \
  --learning-rate 2e-4
```

**Available Arguments:**

- `--train-data` - Path to training data JSONL file (default: `data/train.jsonl`)
- `--output-dir` - Output directory for the fine-tuned model (default: `outputs/adaptor/fine-tuned-model-mps`)
- `--base-model` - Base model to fine-tune (default: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
- `--max-seq-length` - Maximum sequence length (default: `512`)
- `--epochs` - Number of training epochs (default: `3`)
- `--learning-rate` - Learning rate (default: `2e-4`)

**What happens during training:**

- Load the base model `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Apply LoRA adapters to enable efficient fine-tuning
- Train on your dataset
- Save the LoRA adapter to the specified output directory

### 3. Merge Adapter with Base Model

To create a standalone merged model (without needing the adapter separately):

```bash
uv run python src/merge.py
```

Or with custom paths:

```bash
uv run python src/merge.py \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter outputs/adaptor/fine-tuned-model-mps \
  --output outputs/merged/merged-fine-tuned-model
```

### 4. Create GGUF Model for Ollama

To create a GGUF file compatible with Ollama, you need utilities from `llama.cpp`:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt

# Convert your merged model
python convert_hf_to_gguf.py \
  outputs/merged/merged-fine-tuned-model \
  --outtype f16 \
  --outfile outputs/gguf/fine-tuned-model-f16.gguf
```

## Deploy to Ollama

### 1. Create a Modelfile

The pipeline automatically generates a `Modelfile` with the correct path:

```dockerfile
FROM /absolute/path/to/outputs/gguf/fine-tuned-model-f16.gguf
PARAMETER temperature 1
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.05
```

Or create manually:

```dockerfile
FROM ./fine-tuned-model-f16.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.9
```

### 2. Create and Run the Model

```bash
# Create the model in Ollama
ollama create twinkle -f Modelfile

# Run the model
ollama run twinkle "Complete the rhyme: Twinkle, twinkle, little star,"
```

## CLI Commands Reference

| Command | Description |
|---------|-------------|
| `tuneforge pipeline` | Run complete pipeline (train → merge → convert) |
| `tuneforge train` | Train model with LoRA adapters |
| `tuneforge merge` | Merge LoRA adapter with base model |
| `tuneforge convert` | Convert model to GGUF format |
| `tuneforge test` | Test a fine-tuned model |

For detailed help on any command:

```bash
tuneforge --help
tuneforge train --help
tuneforge convert --help
```

## Project Structure

```text
TuneForge/
├── tuneforge/
│   ├── __init__.py               # Package initialization
│   └── cli.py                    # CLI command interface
├── data/
│   └── train.jsonl               # Training dataset
├── src/
│   ├── trainer.py                # Training script
│   └── merge.py                  # Merge adapter with base model
├── scripts/
│   ├── create_eval_dataset.py    # Evaluation utilities
│   └── README.md                 # Scripts documentation
├── outputs/
│   ├── adaptor/                  # LoRA adapter output
│   ├── merged/                   # Merged model output
│   └── gguf/                     # GGUF converted models
├── train-and-convert.sh          # Shell script (alternative to CLI)
├── config.env                    # Pipeline configuration
├── config.env.example            # Example configuration with documentation
├── Modelfile                     # Auto-generated Ollama Modelfile
├── test.py                       # Test script for models
├── pyproject.toml                # Package definition and dependencies
└── README.md                     # This file
```

## Tips & Best Practices

### Model Naming

Use descriptive `MODEL_NAME` values in your config:

```bash
MODEL_NAME="rhyme-generator-v1"
MODEL_NAME="sql-expert-tinyllama"
MODEL_NAME="customer-support-bot"
```

### Quantization Choices

- **f16**: Best quality, larger file size (~2.2GB for TinyLlama)
- **q8_0**: Good quality, smaller size (~1.2GB)
- **q4_0**: Smallest, faster inference, some quality loss (~600MB)

### Multiple Experiments

Create different config files for different experiments:

```bash
cp config.env experiments/rhyme-v1.env
cp config.env experiments/rhyme-v2.env

./train-and-convert.sh --config experiments/rhyme-v1.env
./train-and-convert.sh --config experiments/rhyme-v2.env
```

### Re-running with Different Quantizations

Save training time by skipping training and merge steps:

```bash
# Initial training
tuneforge pipeline --config config.env

# Create additional quantized versions
tuneforge convert --model outputs/merged/my-model --output outputs/gguf/my-model-q8_0.gguf --llama-cpp-path /path/to/llama.cpp --outtype q8_0
tuneforge convert --model outputs/merged/my-model --output outputs/gguf/my-model-q4_0.gguf --llama-cpp-path /path/to/llama.cpp --outtype q4_0
```

## Quick Reference

### Essential Commands

```bash
# Complete workflow
tuneforge pipeline --config config.env

# Individual steps
tuneforge train --train-data data/train.jsonl
tuneforge merge --adapter outputs/adaptor/model --output outputs/merged/model
tuneforge convert --model outputs/merged/model --output outputs/gguf/model.gguf --llama-cpp-path /path/to/llama.cpp
tuneforge test --model outputs/merged/model --prompt "Your prompt"

# Deploy to Ollama
ollama create my-model -f Modelfile
ollama run my-model "Your prompt"
```

### Common Workflows

```bash
# Experiment with different epochs
tuneforge pipeline --config config.env --epochs 5

# Create multiple quantizations
tuneforge pipeline --config config.env
tuneforge convert --model outputs/merged/model --output outputs/gguf/model-q8.gguf --llama-cpp-path /path --outtype q8_0

# Resume from existing adapter
tuneforge pipeline --config config.env --skip-training

# Test before deploying
tuneforge test --model outputs/merged/model --prompt "Test prompt"
```

## Technical Notes

- **Base Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B parameters) - easily adaptable to other models
- **LoRA Adapters**: Add only ~1-5% additional parameters for efficient fine-tuning
- **Apple Silicon**: Uses float32 for MPS compatibility, GPU-accelerated training
- **No Warnings**: Modern `SFTConfig` API eliminates deprecation warnings
- **Quantization**: f16 for quality, q8_0 for balance, q4_0 for speed
- **Deployment**: Merged model works with transformers or GGUF for Ollama

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Support

- **Issues**: [GitHub Issues](https://github.com/deepu-roy/TuneForge/issues)
- **Discussions**: [GitHub Discussions](https://github.com/deepu-roy/TuneForge/discussions)

---

Made with 🔨 by [Deepu](https://github.com/deepu-roy)

# Fine-Tuning Example with TinyLlama

This project demonstrates how to fine-tune the TinyLlama model using LoRA (Low-Rank Adaptation) and export it for use with Ollama.

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

> **Note:** The automated pipeline (`train-and-convert.sh`) handles all these steps automatically and is the recommended approach.

---

## Testing Your Model

### Test with Python

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

### Test with Ollama

After running the pipeline, test with Ollama:

```bash
# The pipeline auto-generates the Modelfile
ollama create my-model -f Modelfile
ollama run my-model "Write a rhyme about stars"
```

---

## Quick Start: Automated Pipeline 🚀

**The recommended way to fine-tune a model is using the automated pipeline script `train-and-convert.sh`.** This script handles the entire workflow from training to deployment with a single command.

### What the Pipeline Does

The script automatically:

1. ✅ Trains your model with LoRA adapters
2. ✅ Merges the adapter with the base model
3. ✅ Converts to GGUF format for Ollama
4. ✅ Generates a ready-to-use Modelfile

### Basic Usage

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

### Advanced Usage Examples

#### Using Command-Line Arguments Only

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
```

#### Mixing Config File with Overrides

```bash
# Use config.env but override specific settings
./train-and-convert.sh --config config.env --epochs 5 --gguf-outtype q4_0

# Override model name for a different variant
./train-and-convert.sh --config config.env --model-name my-model-v2
```

#### Skip Steps (Partial Pipeline)

```bash
# Only merge and convert (skip training)
./train-and-convert.sh --config config.env --skip-training

# Only training and merging (skip GGUF conversion)
./train-and-convert.sh --config config.env --skip-gguf

# Only convert existing merged model to GGUF
./train-and-convert.sh \
  --skip-training \
  --skip-merge \
  --merged-output outputs/merged/my-existing-model \
  --llama-cpp-path /path/to/llama.cpp
```

#### Multiple Quantization Versions

Create multiple GGUF files with different quantizations:

```bash
# Create f16 version (high quality)
./train-and-convert.sh --config config.env --skip-training --gguf-outtype f16

# Create q8_0 version (smaller, good quality)
./train-and-convert.sh --config config.env --skip-training --skip-merge --gguf-outtype q8_0

# Create q4_0 version (smallest, faster inference)
./train-and-convert.sh --config config.env --skip-training --skip-merge --gguf-outtype q4_0
```

### Pipeline Output

After successful completion, you'll have:

```text
outputs/
├── adaptor/
│   └── my-custom-model-mps/     # LoRA adapter
├── merged/
│   └── merged-my-custom-model/  # Merged model
└── gguf/
    └── my-custom-model-f16.gguf # GGUF for Ollama

Modelfile                        # Auto-generated, ready to use
```

### Configuration Variables Reference

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

### Get Help

View all available options:

```bash
./train-and-convert.sh --help
```

---

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

Use `--help` to see all options:

```bash
uv run python src/trainer.py --help
```

**What happens during training:**

- Load the base model `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Apply LoRA adapters to enable efficient fine-tuning
- Train on your dataset
- Save the LoRA adapter to the specified output directory

**Configuration:**

- Uses MPS (Metal Performance Shaders) for Apple Silicon GPU acceleration
- LoRA parameters: rank=16, alpha=32, dropout=0.05
- Training: batch size=2, gradient accumulation=8
- No deprecation warnings - uses modern `SFTConfig` API

#### 3. Merge Adapter with Base Model

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

This creates a merged model in `/output/fine-tuned-model/` directory.

## Creating GGUF Model for Ollama

To create a GGUF file compatible with Ollama, you need utilities from `llama.cpp`:

### Install llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt
```

### Convert to GGUF

```bash
python convert_hf_to_gguf.py \
  --outtype f16 \
  --model ./fine-tuned-model \
  --outfile ./fine-tuned-model-f16.gguf
```

## Run with Ollama

### 1. Create a Modelfile

Create a `Modelfile` in the project root:

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

---

## Project Files

```text
.
├── data/
│   └── train.jsonl               # Training dataset
├── src/
│   ├── trainer.py                # Main training script (with CLI args)
│   └── merge.py                  # Merge LoRA adapter with base model (with CLI args)
├── outputs/
│   ├── adaptor/                  # LoRA adapter output
│   ├── merged/                   # Merged model output
│   └── gguf/                     # GGUF converted models
├── train-and-convert.sh          # 🚀 Automated pipeline script
├── config.env                    # Pipeline configuration
├── config.env.example            # Example configuration with documentation
├── Modelfile                     # Auto-generated Ollama Modelfile
├── test.py                       # Test script for fine-tuned model
├── pyproject.toml                # UV/Python dependencies
└── README.md                     # This file
```

## Key Features

✅ **Automated Pipeline** - Single command to train, merge, and convert (`train-and-convert.sh`)  
✅ **Configuration File Support** - Reusable configs with `config.env`  
✅ **Auto-Generated Modelfile** - Ready-to-use Ollama Modelfile with correct paths  
✅ **Flexible Quantization** - Support for f16, f32, q8_0, q4_0, and more  
✅ **Modern APIs** - Uses `SFTConfig` instead of deprecated `TrainingArguments`  
✅ **Apple Silicon Optimized** - MPS support for GPU acceleration on Mac  
✅ **No Warnings** - Clean training output without deprecation warnings  
✅ **Efficient Training** - LoRA enables fine-tuning with minimal memory  
✅ **Skip Steps** - Re-run only specific parts of the pipeline  
✅ **Ollama Ready** - Easy deployment to Ollama with one command  

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
./train-and-convert.sh --config config.env

# Create additional quantized versions
./train-and-convert.sh --config config.env --skip-training --skip-merge --gguf-outtype q8_0
./train-and-convert.sh --config config.env --skip-training --skip-merge --gguf-outtype q4_0
```

## Notes

- The base model is `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B parameters)
- LoRA adapters typically add only ~1-5% additional parameters
- Training on Apple Silicon uses float32 for MPS compatibility
- The merged model can be used with transformers or converted to GGUF for Ollama

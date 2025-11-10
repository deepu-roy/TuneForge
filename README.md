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

## How to Fine-Tune a Base Model

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

Execute the training script:

```bash
uv run python src/trainer.py
```

This will:

- Load the base model `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Apply LoRA adapters to enable efficient fine-tuning
- Train on your dataset
- Save the LoRA adapter to `outputs/twinkle-lora-mps/`

**Configuration in `trainer.py`:**

- Uses MPS (Metal Performance Shaders) for Apple Silicon GPU acceleration
- LoRA parameters: rank=16, alpha=32, dropout=0.05
- Training: 3 epochs, batch size=2, learning rate=2e-4
- No deprecation warnings - uses modern `SFTConfig` API

### 3. Test the Fine-Tuned Model

Test the LoRA adapter with the base model:

```bash
uv run python test.py
```

### 4. Merge Adapter with Base Model

To create a standalone merged model (without needing the adapter separately):

```bash
uv run python src/merge.py
```

This creates a merged model in `merged-twinkle-llama/` directory.

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
python llama.cpp/convert_hf_to_gguf.py \
  --outtype f16 \
  --model ./merged-twinkle-llama \
  --outfile ./merged-twinkle-f16.gguf
```

## Run with Ollama

### 1. Create a Modelfile

Create a `Modelfile` in the project root:

```
FROM ./merged-twinkle-f16.gguf
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

## Project Structure

```
.
├── data/
│   └── train.jsonl          # Training dataset
├── src/
│   ├── trainer.py           # Main training script
│   └── merge.py             # Merge LoRA adapter with base model
├── outputs/
│   └── twinkle-lora-mps/    # LoRA adapter output
├── test.py                  # Test script for fine-tuned model
├── pyproject.toml           # UV/Python dependencies
└── README.md                # This file
```

## Key Features

✅ **Modern APIs** - Uses `SFTConfig` instead of deprecated `TrainingArguments`  
✅ **Apple Silicon Optimized** - MPS support for GPU acceleration on Mac  
✅ **No Warnings** - Clean training output without deprecation warnings  
✅ **Efficient Training** - LoRA enables fine-tuning with minimal memory  
✅ **Ollama Compatible** - Easy conversion to GGUF format  

## Notes

- The base model is `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B parameters)
- LoRA adapters typically add only ~1-5% additional parameters
- Training on Apple Silicon uses float32 for MPS compatibility
- The merged model can be used with transformers or converted to GGUF for Ollama
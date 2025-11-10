#!/bin/bash

# Fine-tune, merge, and convert to GGUF pipeline script
# Usage: ./train-and-convert.sh [options]

set -e  # Exit on error

# Default values
TRAIN_DATA="data/train.jsonl"
LLAMA_CPP_PATH=""
BASE_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EPOCHS=3
LEARNING_RATE=2e-4
MAX_SEQ_LENGTH=512
ADAPTER_OUTPUT="outputs/adaptor/fine-tuned-model-mps"
MERGED_OUTPUT="outputs/merged/merged-fine-tuned-model"
GGUF_OUTPUT="outputs/gguf/fine-tuned-model-f16.gguf"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Help message
show_help() {
    cat << EOF
Usage: ./train-and-convert.sh [OPTIONS]

This script runs the complete fine-tuning pipeline:
  1. Train the model with LoRA
  2. Merge the adapter with the base model
  3. Convert to GGUF format for Ollama

Options:
  --train-data PATH          Path to training data JSONL file (default: data/train.jsonl)
  --llama-cpp-path PATH      Path to llama.cpp directory (required for GGUF conversion)
  --base-model MODEL         Base model to fine-tune (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
  --epochs N                 Number of training epochs (default: 3)
  --learning-rate RATE       Learning rate (default: 2e-4)
  --max-seq-length LENGTH    Maximum sequence length (default: 512)
  --adapter-output PATH      Adapter output directory (default: outputs/adaptor/fine-tuned-model-mps)
  --merged-output PATH       Merged model output directory (default: outputs/merged/merged-fine-tuned-model)
  --gguf-output PATH         GGUF output file path (default: outputs/gguf/fine-tuned-model-f16.gguf)
  --skip-training            Skip training step (use existing adapter)
  --skip-merge               Skip merge step (use existing merged model)
  --skip-gguf                Skip GGUF conversion step
  -h, --help                 Show this help message

Examples:
  # Full pipeline with default settings
  ./train-and-convert.sh --llama-cpp-path /path/to/llama.cpp

  # Custom training data and output paths
  ./train-and-convert.sh \\
    --train-data my_data.jsonl \\
    --llama-cpp-path /path/to/llama.cpp \\
    --gguf-output my-model.gguf

  # Skip training, only merge and convert existing adapter
  ./train-and-convert.sh --skip-training --llama-cpp-path /path/to/llama.cpp
EOF
}

# Parse command line arguments
SKIP_TRAINING=false
SKIP_MERGE=false
SKIP_GGUF=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --train-data)
            TRAIN_DATA="$2"
            shift 2
            ;;
        --llama-cpp-path)
            LLAMA_CPP_PATH="$2"
            shift 2
            ;;
        --base-model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --max-seq-length)
            MAX_SEQ_LENGTH="$2"
            shift 2
            ;;
        --adapter-output)
            ADAPTER_OUTPUT="$2"
            shift 2
            ;;
        --merged-output)
            MERGED_OUTPUT="$2"
            shift 2
            ;;
        --gguf-output)
            GGUF_OUTPUT="$2"
            shift 2
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-merge)
            SKIP_MERGE=true
            shift
            ;;
        --skip-gguf)
            SKIP_GGUF=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Print configuration
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Fine-Tuning Pipeline Configuration${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "Training data:      ${GREEN}$TRAIN_DATA${NC}"
echo -e "Base model:         ${GREEN}$BASE_MODEL${NC}"
echo -e "Epochs:             ${GREEN}$EPOCHS${NC}"
echo -e "Learning rate:      ${GREEN}$LEARNING_RATE${NC}"
echo -e "Max seq length:     ${GREEN}$MAX_SEQ_LENGTH${NC}"
echo -e "Adapter output:     ${GREEN}$ADAPTER_OUTPUT${NC}"
echo -e "Merged output:      ${GREEN}$MERGED_OUTPUT${NC}"
echo -e "GGUF output:        ${GREEN}$GGUF_OUTPUT${NC}"
echo -e "llama.cpp path:     ${GREEN}${LLAMA_CPP_PATH:-Not set}${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Validate inputs
if [ "$SKIP_TRAINING" = false ]; then
    if [ ! -f "$TRAIN_DATA" ]; then
        echo -e "${RED}Error: Training data file not found: $TRAIN_DATA${NC}"
        exit 1
    fi
fi

if [ "$SKIP_GGUF" = false ]; then
    if [ -z "$LLAMA_CPP_PATH" ]; then
        echo -e "${YELLOW}Warning: llama.cpp path not provided. GGUF conversion will be skipped.${NC}"
        SKIP_GGUF=true
    elif [ ! -f "$LLAMA_CPP_PATH/convert_hf_to_gguf.py" ]; then
        echo -e "${RED}Error: convert_hf_to_gguf.py not found in: $LLAMA_CPP_PATH${NC}"
        exit 1
    fi
fi

# Step 1: Training
if [ "$SKIP_TRAINING" = false ]; then
    echo -e "${BLUE}Step 1/3: Training model with LoRA...${NC}"
    uv run python src/trainer.py \
        --train-data "$TRAIN_DATA" \
        --output-dir "$ADAPTER_OUTPUT" \
        --base-model "$BASE_MODEL" \
        --epochs "$EPOCHS" \
        --learning-rate "$LEARNING_RATE" \
        --max-seq-length "$MAX_SEQ_LENGTH"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Training completed successfully${NC}"
    else
        echo -e "${RED}✗ Training failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Skipping training step (using existing adapter)${NC}"
fi

echo ""

# Step 2: Merging
if [ "$SKIP_MERGE" = false ]; then
    echo -e "${BLUE}Step 2/3: Merging adapter with base model...${NC}"
    uv run python src/merge.py \
        --base-model "$BASE_MODEL" \
        --adapter "$ADAPTER_OUTPUT" \
        --output "$MERGED_OUTPUT"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Merge completed successfully${NC}"
    else
        echo -e "${RED}✗ Merge failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Skipping merge step (using existing merged model)${NC}"
fi

echo ""

# Step 3: GGUF Conversion
if [ "$SKIP_GGUF" = false ]; then
    echo -e "${BLUE}Step 3/3: Converting to GGUF format...${NC}"
    
    # Create output directory if it doesn't exist
    mkdir -p "$(dirname "$GGUF_OUTPUT")"
    
    python "$LLAMA_CPP_PATH/convert_hf_to_gguf.py" \
        "$MERGED_OUTPUT" \
        --outtype f16 \
        --outfile "$GGUF_OUTPUT"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ GGUF conversion completed successfully${NC}"
    else
        echo -e "${RED}✗ GGUF conversion failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Skipping GGUF conversion step${NC}"
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Pipeline completed successfully!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "Outputs:"
echo -e "  Adapter:      ${GREEN}$ADAPTER_OUTPUT${NC}"
echo -e "  Merged model: ${GREEN}$MERGED_OUTPUT${NC}"
if [ "$SKIP_GGUF" = false ]; then
    echo -e "  GGUF file:    ${GREEN}$GGUF_OUTPUT${NC}"
    echo ""
    echo -e "Next steps:"
    echo -e "  1. Test with Ollama:"
    echo -e "     ${YELLOW}ollama create my-model -f Modelfile${NC}"
    echo -e "     ${YELLOW}ollama run my-model \"Your prompt here\"${NC}"
fi
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

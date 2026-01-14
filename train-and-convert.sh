#!/bin/bash

# TuneForge - Automated pipeline script
# Fine-tune, merge, and convert to GGUF
# Usage: ./train-and-convert.sh [options]
# Or with config file: ./train-and-convert.sh --config config.env
# Or use the CLI: tuneforge pipeline --config config.env

set -e  # Exit on error

# Default values
MODEL_NAME="fine-tuned-model"
TRAIN_DATA="data/train.jsonl"
LLAMA_CPP_PATH=""
BASE_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EPOCHS=3
LEARNING_RATE=2e-4
MAX_SEQ_LENGTH=512
GGUF_OUTTYPE="f16"
ADAPTER_OUTPUT="outputs/adaptor/${MODEL_NAME}-mps"
MERGED_OUTPUT="outputs/merged/merged-${MODEL_NAME}"
GGUF_OUTPUT="outputs/gguf/${MODEL_NAME}-${GGUF_OUTTYPE}.gguf"
SKIP_TRAINING=false
SKIP_MERGE=false
SKIP_GGUF=false
CONFIG_FILE=""
MODELFILE_TEMPERATURE=1
MODELFILE_TOP_P=0.9
MODELFILE_REPEAT_PENALTY=1.05
SYSTEM_MESSAGE="You are a helpful AI assistant."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Help message
show_help() {
    cat << EOF
TuneForge - Automated Pipeline Script
Usage: ./train-and-convert.sh [OPTIONS]

This script runs the complete fine-tuning pipeline:
  1. Train the model with LoRA
  2. Merge the adapter with the base model
  3. Convert to GGUF format for Ollama

Note: You can also use the CLI: tuneforge pipeline --config config.env

Options:
  --config FILE              Load configuration from file (see config.env.example)
  --model-name NAME          Base name for your model (default: fine-tuned-model)
  --train-data PATH          Path to training data JSONL file (default: data/train.jsonl)
  --llama-cpp-path PATH      Path to llama.cpp directory (required for GGUF conversion)
  --base-model MODEL         Base model to fine-tune (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
  --epochs N                 Number of training epochs (default: 3)
  --learning-rate RATE       Learning rate (default: 2e-4)
  --max-seq-length LENGTH    Maximum sequence length (default: 512)
  --adapter-output PATH      Adapter output directory (default: outputs/adaptor/fine-tuned-model-mps)
  --merged-output PATH       Merged model output directory (default: outputs/merged/merged-fine-tuned-model)
  --gguf-output PATH         GGUF output file path (default: outputs/gguf/fine-tuned-model-f16.gguf)
  --gguf-outtype TYPE        GGUF quantization type: f32, f16, q8_0, q4_0, etc. (default: f16)
  --skip-training            Skip training step (use existing adapter)
  --skip-merge               Skip merge step (use existing merged model)
  --skip-gguf                Skip GGUF conversion step
  -h, --help                 Show this help message

Examples:
  # Using config file (recommended)
  ./train-and-convert.sh --config my-config.env

  # Full pipeline with default settings
  ./train-and-convert.sh --llama-cpp-path /path/to/llama.cpp

  # Custom training data and output paths
  ./train-and-convert.sh \\
    --train-data my_data.jsonl \\
    --llama-cpp-path /path/to/llama.cpp \\
    --gguf-output my-model.gguf

  # Skip training, only merge and convert existing adapter
  ./train-and-convert.sh --skip-training --llama-cpp-path /path/to/llama.cpp

  # Config file with command-line override
  ./train-and-convert.sh --config my-config.env --epochs 5
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
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
        --gguf-outtype)
            GGUF_OUTTYPE="$2"
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

# Load config file if specified
if [ -n "$CONFIG_FILE" ]; then
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
        exit 1
    fi
    echo -e "${BLUE}Loading configuration from: ${CONFIG_FILE}${NC}"
    
    # Validate config file format before sourcing (security check)
    if ! grep -qE '^[A-Z_]+=' "$CONFIG_FILE" 2>/dev/null; then
        echo -e "${RED}Error: Config file appears to be empty or invalid${NC}"
        exit 1
    fi
    
    # Check for dangerous patterns (semicolons, backticks, command substitution with $())
    # Allow ${VAR} for variable substitution but block $() and backticks
    if grep -qE '[;`]|\$\(.*\)' "$CONFIG_FILE" 2>/dev/null; then
        echo -e "${RED}Error: Config file contains suspicious characters${NC}"
        echo -e "${YELLOW}Config files should only contain KEY=VALUE pairs${NC}"
        exit 1
    fi
    
    # Source the config file (safely)
    set -a  # Export all variables
    source "$CONFIG_FILE"
    set +a
    echo -e "${GREEN}✓ Configuration loaded${NC}"
    echo ""
fi

# Print configuration
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Fine-Tuning Pipeline Configuration${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "Model name:         ${GREEN}$MODEL_NAME${NC}"
echo -e "Training data:      ${GREEN}$TRAIN_DATA${NC}"
echo -e "Base model:         ${GREEN}$BASE_MODEL${NC}"
echo -e "Epochs:             ${GREEN}$EPOCHS${NC}"
echo -e "Learning rate:      ${GREEN}$LEARNING_RATE${NC}"
echo -e "Max seq length:     ${GREEN}$MAX_SEQ_LENGTH${NC}"
echo -e "Adapter output:     ${GREEN}$ADAPTER_OUTPUT${NC}"
echo -e "Merged output:      ${GREEN}$MERGED_OUTPUT${NC}"
echo -e "GGUF output:        ${GREEN}$GGUF_OUTPUT${NC}"
echo -e "GGUF outtype:       ${GREEN}$GGUF_OUTTYPE${NC}"
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
        --outtype "$GGUF_OUTTYPE" \
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
    
    # Generate Modelfile
    echo ""
    echo -e "${BLUE}Generating Modelfile...${NC}"
    
    # Convert to absolute path if relative
    if [[ "$GGUF_OUTPUT" = /* ]]; then
        GGUF_ABSOLUTE_PATH="$GGUF_OUTPUT"
    else
        GGUF_ABSOLUTE_PATH="$(pwd)/$GGUF_OUTPUT"
    fi
    
    cat > Modelfile << EOF
FROM ${GGUF_ABSOLUTE_PATH}

TEMPLATE """<|system|>
{{ .System }}</s>
<|user|>
{{ .Prompt }}</s>
<|assistant|>
"""
SYSTEM """${SYSTEM_MESSAGE}"""


PARAMETER temperature ${MODELFILE_TEMPERATURE}
PARAMETER top_p ${MODELFILE_TOP_P}
PARAMETER repeat_penalty ${MODELFILE_REPEAT_PENALTY}
PARAMETER top_k 40

PARAMETER repeat_penalty 1.3
PARAMETER repeat_last_n 256
PARAMETER presence_penalty 0.6
PARAMETER frequency_penalty 0.4

PARAMETER mirostat 2
PARAMETER mirostat_tau 5
PARAMETER mirostat_eta 0.1

PARAMETER num_predict 512

EOF
    
    echo -e "${GREEN}✓ Modelfile generated${NC}"
    echo ""
    echo -e "Next steps:"
    echo -e "  1. Create Ollama model:"
    echo -e "     ${YELLOW}ollama create my-model -f Modelfile${NC}"
    echo -e "  2. Run the model:"
    echo -e "     ${YELLOW}ollama run my-model \"Your prompt here\"${NC}"
fi
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

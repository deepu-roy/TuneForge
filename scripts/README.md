# TuneForge Evaluation Scripts

## Overview

This directory contains scripts for creating evaluation datasets and testing fine-tuned models.

## create_eval_dataset.py

Transform training data into evaluation format and optionally run evaluations against Ollama models.

### Usage

#### 1. Create Evaluation Dataset Only

```bash
# Create eval dataset from training data
uv run python scripts/create_eval_dataset.py \
  --input data/qa-training.jsonl \
  --output data/qa-eval.jsonl
```

#### 2. Create and Run Ollama Evaluation

```bash
# Create eval dataset and run against Ollama model
uv run python scripts/create_eval_dataset.py \
  --input data/qa-training.jsonl \
  --output data/qa-eval-results.jsonl \
  --run-ollama \
  --model qa-model:latest
```

#### 3. Sample a Subset for Testing

```bash
# Test with just 3 examples
uv run python scripts/create_eval_dataset.py \
  --input data/qa-training.jsonl \
  --output data/qa-eval-results.jsonl \
  --run-ollama \
  --model qa-model:latest \
  --sample 3
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input training JSONL file | `data/qa-training.jsonl` |
| `--output` | Output evaluation JSONL file | `data/qa-eval.jsonl` |
| `--sample` | Number of samples to use | All |
| `--run-ollama` | Run evaluation against Ollama | False |
| `--model` | Ollama model name | `qa-model:latest` |

### Output Format

The evaluation dataset follows this structure (compatible with ULEI format):

```json
{
  "id": "qa_eval_001",
  "input": {
    "query": "User question",
    "system_prompt": "System instructions"
  },
  "output": {
    "answer": "Expected full answer",
    "structured": {
      "summary": "...",
      "steps": "...",
      "code": "...",
      "risks": "..."
    },
    "model_response": "Actual model output (if --run-ollama)",
    "model_parsed": {...},
    "success": true,
    "response_time": 4.5,
    "has_all_sections": false
  },
  "reference": {
    "expected": "Expected answer format",
    "format": "Description of expected format"
  },
  "context": [
    {"text": "Context info", "source": "system_instructions"}
  ]
}
```

### Evaluation Metrics

When running with `--run-ollama`, the script tracks:

- **Success rate**: Percentage of queries that got responses
- **Response time**: Time taken for each query
- **Section completeness**: Whether all required sections are present
- **Format compliance**: Whether response follows expected structure

### Example Workflow

```bash
# 1. Fine-tune your model with TuneForge
tuneforge pipeline --config config.env

# 2. Create Ollama model
ollama create qa-model -f Modelfile

# 3. Create evaluation dataset
uv run python scripts/create_eval_dataset.py \
  --input data/qa-training.jsonl \
  --output data/qa-eval.jsonl

# 4. Run evaluation against your fine-tuned model
uv run python scripts/create_eval_dataset.py \
  --input data/qa-training.jsonl \
  --output data/qa-eval-results.jsonl \
  --run-ollama \
  --model qa-model:latest

# 5. Analyze results
cat data/qa-eval-results.jsonl | python -m json.tool | less
```

### Prerequisites

- Ollama installed and running (for `--run-ollama`)
- Your model created in Ollama (`ollama create model-name -f Modelfile`)

### Tips

- Use `--sample` for quick testing during development
- Run full evaluation after training to measure model quality
- Compare results before and after fine-tuning
- Use different model names to compare versions

## Future Enhancements

- [ ] Add evaluation metrics calculation (accuracy, format compliance)
- [ ] Support for batch evaluation
- [ ] Comparison reports between models
- [ ] Automated scoring system

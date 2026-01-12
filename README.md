# Week 7 Assignment: Fine-Tuning Llama 3.1 8B with QLoRA

Complete implementation for generating synthetic Q&A data and fine-tuning a Llama model.

## Quick Start

### Option 1: Run Full Pipeline (Recommended)

```bash
# Set environment variables
export OPENAI_API_KEY="your-openai-key"
export HF_HUB_TOKEN="your-hf-token"

# Install dependencies
pip install -r requirements.txt

# Run complete pipeline (all 3 steps)
python run_full_pipeline.py
```

This will:
1. Generate synthetic_qa.jsonl (~500 Q&A pairs from 100 PDFs)
2. Fine-tune Llama 3.1 8B using QLoRA
3. Evaluate base vs fine-tuned model

### Option 2: Run Steps Individually

```bash
# Step 1: Generate synthetic Q&A data
python generate_synthetic_qa.py

# Step 2: Fine-tune the model
python finetune_llama.py

# Step 3: Evaluate models
python evaluate_models.py
```

## Files Overview

### Scripts
- **`generate_synthetic_qa.py`** - Generate Q&A pairs from PDFs using GPT-4
- **`finetune_llama.py`** - Fine-tune Llama 3.1 8B with QLoRA
- **`evaluate_models.py`** - Compare base vs fine-tuned model
- **`run_full_pipeline.py`** - Run all steps automatically

### Documentation
- **`FINETUNING_GUIDE.md`** - Detailed step-by-step guide with explanations
- **`README_HOMEWORK7.md`** - This file

### Outputs (Generated)
- **`synthetic_qa.jsonl`** - Training dataset (~500 Q&A pairs)
- **`llama3-8b-qlora-finetuned/`** - Fine-tuned model directory
- **`evaluation_results.txt`** - Evaluation comparison results

## Requirements

### Environment Variables
```bash
OPENAI_API_KEY  # For GPT-4 synthetic data generation
HF_HUB_TOKEN    # For accessing Llama models on HuggingFace
```

### Hardware
- **GPU**: 12GB+ VRAM (T4, V100, A100)
- **RAM**: 16GB+
- **Storage**: 20GB+

### Software
All dependencies in `requirements.txt`:
- `openai>=1.0.0` - GPT-4 API
- `unsloth` - Fast LoRA training
- `transformers` - Model loading
- `peft` - LoRA implementation
- `bitsandbytes` - 4-bit quantization
- `datasets` - Data loading
- `PyMuPDF` - PDF text extraction
- `tqdm` - Progress bars

## Expected Outputs

### 1. synthetic_qa.jsonl
Format:
```json
{"text": "<|system|>You are a helpful academic Q&A assistant...<|user|>What is...?<|assistant|>The paper shows..."}
```

### 2. Fine-Tuned Model
Directory structure:
```
llama3-8b-qlora-finetuned/
├── adapter_config.json      # LoRA config
├── adapter_model.safetensors # LoRA weights
├── tokenizer.json
└── ...
```

### 3. Evaluation Results
Text file comparing base vs fine-tuned answers for 10 test questions.

## Time Estimates

- **Data Generation**: 5-10 minutes (100 PDFs × GPT-4 calls)
- **Fine-Tuning**: 10-30 minutes (depending on GPU)
  - T4: ~30 min
  - V100: ~15 min
  - A100: ~10 min
- **Evaluation**: 5-10 minutes (10 questions × 2 models)

**Total**: ~30-60 minutes for complete pipeline

## Customization

### Adjust Training Epochs
Edit `finetune_llama.py`:
```python
num_train_epochs=5  # Default: 3
```

### Change Batch Size
```python
per_device_train_batch_size=1  # Default: 2 (if OOM)
gradient_accumulation_steps=8  # Default: 4
```

### Use Different Model
```python
model_name = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"  # Larger
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `per_device_train_batch_size=1`
- Reduce `max_seq_length=1024`
- Use smaller model

### HuggingFace Auth Error
```bash
huggingface-cli login
# Or export token
export HF_HUB_TOKEN="your-token"
```

### GPT-4 Rate Limits
- Add delays in `generate_synthetic_qa.py`
- Use GPT-4o or GPT-3.5 instead

## Deliverables for Assignment

1. **synthetic_qa.jsonl** - Your generated dataset
2. **Fine-tuning code** - Scripts with any modifications
3. **evaluation_results.txt** - Comparison showing improvements

## Additional Resources

- See `FINETUNING_GUIDE.md` for detailed explanations
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)

## License

This code is for educational purposes as part of the course assignment.

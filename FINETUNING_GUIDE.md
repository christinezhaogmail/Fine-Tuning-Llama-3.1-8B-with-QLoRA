# Llama 3.1 8B Fine-Tuning Guide

This guide walks through the complete process of fine-tuning Llama 3.1 8B using QLoRA on synthetic Q&A data.

## Prerequisites

### Environment Variables
Ensure these are set:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export HF_HUB_TOKEN="your-huggingface-token"
```

### Dependencies
Install all required packages:
```bash
pip install -r requirements.txt
```

### Hardware Requirements
- **GPU**: NVIDIA GPU with at least 12GB VRAM (T4, V100, A100, etc.)
- **RAM**: 16GB+ recommended
- **Storage**: ~20GB for models and data

## Step-by-Step Process

### Step 1: Generate Synthetic Q&A Dataset

Generate ~500 Q&A pairs from 100 PDF papers:

```bash
python generate_synthetic_qa.py
```

**What it does:**
- Extracts text from first 5 pages of each PDF (abstract + intro)
- Uses GPT-4 to generate 5 Q&A pairs per paper
- Includes edge-case questions (~10% of papers)
- Outputs `synthetic_qa.jsonl` with instruction-tuning format

**Expected output:**
```
Processing 100 PDF files...
Generating Q&A pairs: 100%|████████████| 100/100 [08:32<00:00, 5.12s/it]
============================================================
Successfully generated 500 Q&A pairs
Saved to: synthetic_qa.jsonl
============================================================
```

**Verify the output:**
```bash
wc -l synthetic_qa.jsonl  # Should show ~500 lines
head -n 1 synthetic_qa.jsonl | python -m json.tool  # View formatted sample
```

---

### Step 2: Fine-Tune the Model

Train Llama 3.1 8B using QLoRA:

```bash
python finetune_llama.py
```

**What it does:**
- Loads `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` (4-bit quantized)
- Configures LoRA adapters (rank=16) on attention and MLP layers
- Trains for 3 epochs with batch size 2, gradient accumulation 4
- Uses 8-bit AdamW optimizer and linear LR schedule
- Saves checkpoints to `llama3-8b-qlora-finetuned/`

**Training progress:**
```
Starting Fine-Tuning Process
============================================================
1. Loading model: unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
2. Configuring LoRA adapters
3. Loading synthetic Q&A dataset
   - Total examples: 500
4. Setting up training configuration
5. Initializing SFT Trainer
6. Starting training...
============================================================
[Training logs with loss values...]
============================================================
Training completed!
Training loss: 0.XXXX
============================================================
7. Saving fine-tuned model
8. Saving 16-bit version for faster inference (optional)
============================================================
Fine-tuning process completed successfully!
============================================================
```

**Training time:** ~10-30 minutes depending on GPU
- T4: ~30 minutes
- V100: ~15 minutes
- A100: ~10 minutes

**Memory usage:** ~8-10GB VRAM

---

### Step 3: Evaluate the Model

Compare base model vs. fine-tuned model:

```bash
python evaluate_models.py
```

**What it does:**
- Loads both base and fine-tuned models
- Runs 10 test questions through each model
- Compares answers side-by-side
- Saves results to `evaluation_results.txt`

**Output:**
```
Model Evaluation: Base vs. Fine-Tuned
============================================================
1. Loading BASE model...
2. Loading FINE-TUNED model...
3. Running evaluation on test questions...
============================================================
QUESTION 1/10
============================================================
Q: What is the main hypothesis...

BASE MODEL ANSWER:
[Answer from base model]

FINE-TUNED MODEL ANSWER:
[Answer from fine-tuned model]
[... continues for all 10 questions ...]
============================================================
Results saved to: evaluation_results.txt
```

**Review the results:**
```bash
cat evaluation_results.txt
```

Look for improvements in:
- **Accuracy**: More correct information
- **Relevance**: Better aligned with academic content
- **Detail**: More comprehensive answers
- **Terminology**: Proper use of academic terms
- **Hallucination handling**: Better at saying "not in the paper"

---

## Output Files

After completing all steps, you'll have:

```
synthetic_qa.jsonl                      # 500 Q&A pairs in JSONL format
llama3-8b-qlora-finetuned/             # Fine-tuned model with LoRA adapters
  ├── adapter_config.json               # LoRA configuration
  ├── adapter_model.safetensors         # LoRA weights
  ├── tokenizer.json                    # Tokenizer files
  └── ...
llama3-8b-qlora-finetuned-16bit/       # Merged 16-bit model (optional)
evaluation_results.txt                  # Comparison of base vs fine-tuned
```

---

## Troubleshooting

### Out of Memory Error
If you get CUDA OOM:
1. Reduce batch size in `finetune_llama.py`: `per_device_train_batch_size=1`
2. Reduce max sequence length: `max_seq_length=1024`
3. Enable gradient checkpointing (already enabled)

### HuggingFace Token Error
If you get authentication errors:
```bash
huggingface-cli login
# Or set token:
export HF_HUB_TOKEN="your-token-here"
```

### Slow Training
If training is too slow:
- Check GPU is being used: `nvidia-smi`
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Consider using a more powerful GPU instance

### API Rate Limits (Step 1)
If GPT-4 API rate limits hit:
- Add delay between requests in `generate_synthetic_qa.py`
- Process in batches with `time.sleep()`

---

## Customization Options

### Adjust Training Parameters

In `finetune_llama.py`, you can modify:

```python
# Longer training
num_train_epochs=5  # Default: 3

# Larger effective batch size
gradient_accumulation_steps=8  # Default: 4

# Different learning rate
learning_rate=1e-4  # Default: 2e-4

# Longer sequences
max_seq_length=4096  # Default: 2048
```

### Adjust LoRA Configuration

```python
# Higher rank (more parameters, better performance, more memory)
r=32  # Default: 16

# Different target modules (fine-tune more layers)
target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "embed_tokens", "lm_head"]  # Add embedding/output layers
```

### Change Model

To use a different base model:

```python
model_name = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"  # Larger model
# Or
model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"  # Smaller model
```

---

## Next Steps

After fine-tuning:

1. **Test on your own questions**: Modify `evaluate_models.py` with domain-specific questions
2. **Deploy the model**: Use the fine-tuned model in your applications
3. **Continue training**: Load the checkpoint and train more epochs if needed
4. **Share the model**: Push to HuggingFace Hub for others to use

```python
# Push to HuggingFace Hub
model.push_to_hub("your-username/llama3-academic-qa", token=hf_token)
```

---

## References

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Llama 3.1 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)

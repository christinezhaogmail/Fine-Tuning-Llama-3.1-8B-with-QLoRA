import os
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

# Ensure HF token is available
hf_token = os.environ.get("HF_HUB_TOKEN")
if not hf_token:
    raise ValueError("HF_HUB_TOKEN environment variable not set!")

print("="*60)
print("Starting Fine-Tuning Process")
print("="*60)

# Model configuration
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
max_seq_length = 2048  # Can increase to 4096, but 2048 saves memory
dtype = None  # Auto-detect, or set to Float16/BFloat16
load_in_4bit = True  # Use 4-bit quantization for memory efficiency

print(f"\n1. Loading model: {model_name}")
print(f"   - Max sequence length: {max_seq_length}")
print(f"   - 4-bit quantization: {load_in_4bit}")

# Load the base model with LoRA adapters
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    token=hf_token,
)

print("\n2. Configuring LoRA adapters")
# Configure LoRA for fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",     # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # Use Unsloth's gradient checkpointing
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

print("\n3. Loading synthetic Q&A dataset")
# Load the synthetic Q&A dataset
dataset = load_dataset("json", data_files="synthetic_qa.jsonl", split="train")
print(f"   - Total examples: {len(dataset)}")

# Show a sample
if len(dataset) > 0:
    print(f"\n   Sample entry:")
    print(f"   {dataset[0]['text'][:200]}...")

print("\n4. Setting up training configuration")
# Training arguments
training_args = TrainingArguments(
    output_dir="llama3-8b-qlora-finetuned",
    per_device_train_batch_size=2,      # Small batch size for memory
    gradient_accumulation_steps=4,       # Accumulate gradients (effective batch = 2*4=8)
    warmup_steps=5,                      # Warmup for 5 steps
    num_train_epochs=3,                  # Train for 3 epochs
    learning_rate=2e-4,                  # Learning rate
    fp16=not is_bfloat16_supported(),    # Use FP16 if BF16 not supported
    bf16=is_bfloat16_supported(),        # Use BF16 if supported
    logging_steps=10,                    # Log every 10 steps
    optim="adamw_8bit",                  # 8-bit Adam optimizer
    weight_decay=0.01,                   # Weight decay
    lr_scheduler_type="linear",          # Linear learning rate schedule
    seed=3407,                           # Random seed
    save_strategy="epoch",               # Save checkpoint after each epoch
    save_total_limit=2,                  # Keep only last 2 checkpoints
    report_to="none",                    # Don't report to wandb/tensorboard
)

print(f"   - Batch size: {training_args.per_device_train_batch_size}")
print(f"   - Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"   - Epochs: {training_args.num_train_epochs}")
print(f"   - Learning rate: {training_args.learning_rate}")
print(f"   - Using {'BF16' if training_args.bf16 else 'FP16'}")

print("\n5. Initializing SFT Trainer")
# Initialize trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences
    args=training_args,
)

print("\n6. Starting training...")
print("="*60)

# Train the model
trainer_stats = trainer.train()

print("\n" + "="*60)
print("Training completed!")
print(f"Training loss: {trainer_stats.training_loss:.4f}")
print("="*60)

print("\n7. Saving fine-tuned model")
# Save the fine-tuned model
model.save_pretrained("llama3-8b-qlora-finetuned")
tokenizer.save_pretrained("llama3-8b-qlora-finetuned")

print(f"\nModel saved to: llama3-8b-qlora-finetuned/")

# Optional: Save to 16-bit for faster inference
print("\n8. Saving 16-bit version for faster inference (optional)")
model.save_pretrained_merged(
    "llama3-8b-qlora-finetuned-16bit",
    tokenizer,
    save_method="merged_16bit",
)
print(f"16-bit model saved to: llama3-8b-qlora-finetuned-16bit/")

print("\n" + "="*60)
print("Fine-tuning process completed successfully!")
print("="*60)

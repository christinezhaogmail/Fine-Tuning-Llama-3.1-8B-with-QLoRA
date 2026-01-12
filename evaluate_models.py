import os
import torch
from unsloth import FastLanguageModel

# Ensure HF token is available
hf_token = os.environ.get("HF_HUB_TOKEN")
if not hf_token:
    raise ValueError("HF_HUB_TOKEN environment variable not set!")

# Model configuration
base_model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
finetuned_model_path = "llama3-8b-qlora-finetuned"
max_seq_length = 2048

# System prompt used during training
system_prompt = "You are a helpful academic Q&A assistant specialized in scholarly content."

# Test questions (create 10 diverse questions covering various academic topics)
test_questions = [
    "What is the main hypothesis or research question addressed in recent quantum computing papers?",
    "How do researchers typically evaluate the performance of deep learning models in academic studies?",
    "What are the common methodologies used in natural language processing research?",
    "Explain the significance of transformer architectures in modern AI research.",
    "What evaluation metrics are commonly used in computer vision papers?",
    "How do academic papers typically structure their experimental validation?",
    "What is the role of ablation studies in machine learning research?",
    "Describe common approaches to handling data imbalance in classification tasks.",
    "What are the key differences between supervised and unsupervised learning approaches?",
    "How do researchers address reproducibility in their experiments?",
]

print("="*80)
print("Model Evaluation: Base vs. Fine-Tuned")
print("="*80)

print("\n1. Loading BASE model...")
base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
    token=hf_token,
)
FastLanguageModel.for_inference(base_model)  # Enable native 2x faster inference
print("   Base model loaded successfully")

print("\n2. Loading FINE-TUNED model...")
ft_model, ft_tokenizer = FastLanguageModel.from_pretrained(
    model_name=finetuned_model_path,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(ft_model)  # Enable native 2x faster inference
print("   Fine-tuned model loaded successfully")

print("\n3. Running evaluation on test questions...")
print("="*80)

results = []

for idx, question in enumerate(test_questions, 1):
    print(f"\n{'='*80}")
    print(f"QUESTION {idx}/{len(test_questions)}")
    print(f"{'='*80}")
    print(f"Q: {question}\n")

    # Create prompt in the same format as training
    prompt = f"<|system|>{system_prompt}<|user|>{question}<|assistant|>"

    # Tokenize input
    inputs = base_tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate with base model
    print("BASE MODEL ANSWER:")
    print("-" * 80)
    with torch.no_grad():
        base_outputs = base_model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
        )
    base_answer = base_tokenizer.decode(base_outputs[0], skip_special_tokens=False)
    # Extract only the assistant's response
    if "<|assistant|>" in base_answer:
        base_answer = base_answer.split("<|assistant|>")[-1].strip()
    # Remove end tokens if present
    base_answer = base_answer.replace("<|end|>", "").replace("</s>", "").strip()
    print(base_answer)

    # Generate with fine-tuned model
    print("\n" + "-" * 80)
    print("FINE-TUNED MODEL ANSWER:")
    print("-" * 80)
    with torch.no_grad():
        ft_outputs = ft_model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
        )
    ft_answer = ft_tokenizer.decode(ft_outputs[0], skip_special_tokens=False)
    # Extract only the assistant's response
    if "<|assistant|>" in ft_answer:
        ft_answer = ft_answer.split("<|assistant|>")[-1].strip()
    # Remove end tokens if present
    ft_answer = ft_answer.replace("<|end|>", "").replace("</s>", "").strip()
    print(ft_answer)

    results.append({
        "question": question,
        "base_answer": base_answer,
        "finetuned_answer": ft_answer
    })

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)

# Save results to file
print("\n4. Saving evaluation results to file...")
with open("evaluation_results.txt", "w") as f:
    f.write("="*80 + "\n")
    f.write("MODEL EVALUATION RESULTS: Base vs. Fine-Tuned\n")
    f.write("="*80 + "\n\n")

    for idx, result in enumerate(results, 1):
        f.write(f"\n{'='*80}\n")
        f.write(f"QUESTION {idx}/{len(results)}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Q: {result['question']}\n\n")
        f.write(f"BASE MODEL:\n")
        f.write(f"{'-'*80}\n")
        f.write(f"{result['base_answer']}\n\n")
        f.write(f"FINE-TUNED MODEL:\n")
        f.write(f"{'-'*80}\n")
        f.write(f"{result['finetuned_answer']}\n\n")

print("Results saved to: evaluation_results.txt")

print("\n" + "="*80)
print("Summary:")
print(f"- Evaluated {len(test_questions)} questions")
print(f"- Results saved to evaluation_results.txt")
print("- Review the outputs to assess improvement in:")
print("  * Accuracy and relevance")
print("  * Use of academic terminology")
print("  * Depth and detail of answers")
print("  * Correction of base model mistakes")
print("="*80)

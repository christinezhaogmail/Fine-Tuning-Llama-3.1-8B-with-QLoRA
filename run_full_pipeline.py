"""
Full Pipeline: Synthetic Data Generation + Fine-Tuning + Evaluation

This script runs the complete homework pipeline:
1. Generate synthetic Q&A data from PDFs using GPT-4
2. Fine-tune Llama 3.1 8B using QLoRA
3. Evaluate base vs fine-tuned model

Usage:
    python run_full_pipeline.py [--skip-generation] [--skip-training] [--skip-evaluation]

Options:
    --skip-generation: Skip step 1 if synthetic_qa.jsonl already exists
    --skip-training: Skip step 2 if model is already fine-tuned
    --skip-evaluation: Skip step 3 evaluation
"""

import os
import sys
import argparse
import subprocess

def check_env_vars():
    """Check required environment variables."""
    required_vars = ["OPENAI_API_KEY", "HF_HUB_TOKEN"]
    missing = []

    for var in required_vars:
        if not os.environ.get(var):
            missing.append(var)

    if missing:
        print("ERROR: Missing required environment variables:")
        for var in missing:
            print(f"  - {var}")
        print("\nPlease set them:")
        print("  export OPENAI_API_KEY='your-key'")
        print("  export HF_HUB_TOKEN='your-token'")
        return False

    return True

def run_step(step_name, script_name, description):
    """Run a pipeline step."""
    print("\n" + "="*80)
    print(f"STEP: {step_name}")
    print(f"DESCRIPTION: {description}")
    print("="*80 + "\n")

    try:
        result = subprocess.run(
            ["python", script_name],
            check=True,
            text=True,
            capture_output=False  # Show output in real-time
        )
        print(f"\n✓ {step_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {step_name} failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n✗ {step_name} interrupted by user")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Run the full fine-tuning pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip synthetic data generation step"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model fine-tuning step"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip model evaluation step"
    )

    args = parser.parse_args()

    print("="*80)
    print("LLAMA 3.1 8B FINE-TUNING PIPELINE")
    print("="*80)

    # Check environment variables
    print("\nChecking environment variables...")
    if not check_env_vars():
        sys.exit(1)
    print("✓ All required environment variables are set")

    # Track which steps to run
    steps_to_run = []

    if not args.skip_generation:
        steps_to_run.append(("1: Data Generation", "generate_synthetic_qa.py",
                            "Generate synthetic Q&A pairs from PDFs using GPT-4"))
    else:
        print("\n⊘ Skipping data generation (--skip-generation)")
        if not os.path.exists("synthetic_qa.jsonl"):
            print("ERROR: synthetic_qa.jsonl not found! Cannot skip generation.")
            sys.exit(1)

    if not args.skip_training:
        steps_to_run.append(("2: Model Fine-Tuning", "finetune_llama.py",
                            "Fine-tune Llama 3.1 8B using QLoRA on synthetic data"))
    else:
        print("\n⊘ Skipping fine-tuning (--skip-training)")
        if not os.path.exists("llama3-8b-qlora-finetuned"):
            print("ERROR: llama3-8b-qlora-finetuned/ not found! Cannot skip training.")
            sys.exit(1)

    if not args.skip_evaluation:
        steps_to_run.append(("3: Model Evaluation", "evaluate_models.py",
                            "Compare base model vs fine-tuned model performance"))
    else:
        print("\n⊘ Skipping evaluation (--skip-evaluation)")

    # Run pipeline
    print(f"\n{'='*80}")
    print(f"PIPELINE: Running {len(steps_to_run)} step(s)")
    print(f"{'='*80}")

    for step_name, script_name, description in steps_to_run:
        success = run_step(step_name, script_name, description)

        if not success:
            print("\n" + "="*80)
            print("PIPELINE FAILED")
            print("="*80)
            print(f"Failed at: {step_name}")
            print("Please fix the error and try again.")
            sys.exit(1)

    # All steps completed
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)

    print("\nGenerated files:")
    if os.path.exists("synthetic_qa.jsonl"):
        num_lines = sum(1 for _ in open("synthetic_qa.jsonl"))
        print(f"  ✓ synthetic_qa.jsonl ({num_lines} Q&A pairs)")

    if os.path.exists("llama3-8b-qlora-finetuned"):
        print(f"  ✓ llama3-8b-qlora-finetuned/ (fine-tuned model)")

    if os.path.exists("llama3-8b-qlora-finetuned-16bit"):
        print(f"  ✓ llama3-8b-qlora-finetuned-16bit/ (16-bit merged model)")

    if os.path.exists("evaluation_results.txt"):
        print(f"  ✓ evaluation_results.txt (evaluation results)")

    print("\nNext steps:")
    print("  1. Review evaluation_results.txt to see improvements")
    print("  2. Test the fine-tuned model with your own questions")
    print("  3. Use the model in your applications or push to HuggingFace Hub")
    print("\nFor detailed guidance, see FINETUNING_GUIDE.md")

if __name__ == "__main__":
    main()

"""
EnronBot Fine-tuning Script
Fine-tunes Mistral 7B using QLoRA on Enron employee emails.

Usage:
    # Train on all personas combined
    python scripts/train.py

    # Train on a specific persona
    python scripts/train.py --persona vince_kaminski

    # Custom settings
    python scripts/train.py --epochs 5 --batch-size 2 --output-dir ./my_model
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# Default configuration
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_OUTPUT_DIR = "./models/enronbot"
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_MAX_SEQ_LENGTH = 1024

# Available personas (must match filenames in data/training/)
PERSONAS = [
    "vince_kaminski",
    "kate_symes",
    "jeff_dasovich",
    "phillip_allen",
    "enron_announcements",
    "all_personas",  # Combined dataset
]


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Mistral 7B on Enron emails")
    parser.add_argument(
        "--persona",
        type=str,
        default="all_personas",
        choices=PERSONAS,
        help="Which persona to train on (default: all_personas)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Base model to fine-tune (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for model (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Per-device batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help=f"Maximum sequence length (default: {DEFAULT_MAX_SEQ_LENGTH})",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="enronbot",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    return parser.parse_args()


def get_quantization_config():
    """4-bit quantization config for QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config():
    """LoRA configuration for efficient fine-tuning."""
    return LoraConfig(
        r=16,  # Rank
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def format_chat(example, tokenizer):
    """Format example using the chat template."""
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


def main():
    args = parse_args()

    # Login to Hugging Face if token provided
    if args.hf_token:
        login(token=args.hf_token)

    # Set up paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "training" / f"{args.persona}.jsonl"
    output_dir = Path(args.output_dir) / args.persona

    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    print(f"=" * 60)
    print(f"EnronBot Fine-tuning")
    print(f"=" * 60)
    print(f"Base model: {args.model}")
    print(f"Persona: {args.persona}")
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    print(f"Loaded {len(dataset)} examples")

    # Split into train/eval (95/5)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Format datasets with chat template
    print("Formatting datasets...")
    train_dataset = train_dataset.map(
        lambda x: {"text": format_chat(x, tokenizer)},
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda x: {"text": format_chat(x, tokenizer)},
        remove_columns=eval_dataset.column_names,
    )

    # Load model with quantization
    print("\nLoading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=get_quantization_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Prepare model for QLoRA training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, get_lora_config())

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="wandb" if not args.no_wandb else "none",
        run_name=f"enronbot-{args.persona}",
        bf16=True,
        optim="paged_adamw_32bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        max_seq_length=args.max_seq_length,
        packing=False,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save the final model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print("\nTraining complete!")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()

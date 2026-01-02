"""
EnronBot Fine-tuning Script
Fine-tunes LLMs using LoRA/QLoRA on Enron employee emails.

Usage:
    # Train on all personas combined
    python scripts/train.py

    # Train on a specific persona
    python scripts/train.py --persona vince_kaminski

    # Custom settings
    python scripts/train.py --epochs 5 --batch-size 2 --output-dir ./my_model

    # Smoketest on Apple Silicon (MPS)
    python scripts/train.py --smoketest
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
)
from trl import SFTConfig, SFTTrainer

# Check for Apple Silicon / MPS
IS_MPS = torch.backends.mps.is_available()
IS_CUDA = torch.cuda.is_available()

# Only import BitsAndBytes if CUDA is available
if IS_CUDA:
    from transformers import BitsAndBytesConfig

# Default configuration
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
SMOKETEST_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
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
    parser.add_argument(
        "--smoketest",
        action="store_true",
        help="Run a quick smoketest with small model and few steps (good for MPS/Apple Silicon)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Maximum training steps (overrides epochs if set)",
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

    # Smoketest mode overrides
    if args.smoketest:
        args.model = SMOKETEST_MODEL
        args.max_steps = 20
        args.batch_size = 2
        args.gradient_accumulation_steps = 1
        args.no_wandb = True
        args.max_seq_length = 512
        print("\n🧪 SMOKETEST MODE: Using TinyLlama with 20 steps\n")

    # Login to Hugging Face if token provided
    if args.hf_token:
        login(token=args.hf_token)

    # Set up paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "training" / f"{args.persona}.jsonl"
    output_dir = Path(args.output_dir) / args.persona

    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    # Determine device
    if IS_MPS:
        device = "mps"
        print("🍎 Apple Silicon detected - using MPS backend")
    elif IS_CUDA:
        device = "cuda"
        print("🔥 CUDA detected - using GPU")
    else:
        device = "cpu"
        print("💻 No GPU detected - using CPU")

    print(f"=" * 60)
    print(f"EnronBot Fine-tuning")
    print(f"=" * 60)
    print(f"Base model: {args.model}")
    print(f"Device: {device}")
    print(f"Persona: {args.persona}")
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {args.epochs}" + (f" (max_steps={args.max_steps})" if args.max_steps > 0 else ""))
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

    # Load model - use quantization on CUDA, full precision on MPS/CPU
    if IS_CUDA and not args.smoketest:
        print("\nLoading model with 4-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=get_quantization_config(),
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print(f"\nLoading model in {'float16' if IS_MPS else 'float32'}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if IS_MPS else torch.float32,
            device_map={"": device} if IS_MPS else "auto",
            trust_remote_code=True,
        )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Apply LoRA
    model = get_peft_model(model, get_lora_config())

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Training arguments - adjust for device capabilities
    use_bf16 = IS_CUDA and not args.smoketest
    use_fp16 = IS_MPS or args.smoketest

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=5 if args.smoketest else 10,
        eval_strategy="steps" if not args.smoketest else "no",
        eval_steps=100 if not args.smoketest else None,
        save_strategy="steps" if not args.smoketest else "no",
        save_steps=100 if not args.smoketest else None,
        save_total_limit=3,
        load_best_model_at_end=False if args.smoketest else True,
        report_to="wandb" if not args.no_wandb else "none",
        run_name=f"enronbot-{args.persona}",
        bf16=use_bf16,
        fp16=use_fp16,
        optim="adamw_torch" if IS_MPS else "paged_adamw_32bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,
        dataloader_pin_memory=False if IS_MPS else True,
        max_length=args.max_seq_length,
        packing=False,
        dataset_text_field="text",
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
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

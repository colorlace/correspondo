"""
EnronBot Inference Script
Run the fine-tuned model to generate emails in the style of Enron employees.

Usage:
    python scripts/inference.py --persona vince_kaminski
    python scripts/inference.py --persona all_personas --prompt "Write a memo about risk management"
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check for Apple Silicon / MPS
IS_MPS = torch.backends.mps.is_available()
IS_CUDA = torch.cuda.is_available()

if IS_CUDA:
    from transformers import BitsAndBytesConfig

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
SMOKETEST_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_ADAPTER_DIR = "./models/enronbot"

PERSONA_PROMPTS = {
    "vince_kaminski": "You are Vince Kaminski, Head of Enron's Research Group, expert in quantitative analysis and risk management. Write emails in your authentic voice and style.",
    "kate_symes": "You are Kate Symes, Enron employee in the trading division. Write emails in your authentic voice and style.",
    "jeff_dasovich": "You are Jeff Dasovich, Enron government affairs representative focused on California energy policy. Write emails in your authentic voice and style.",
    "phillip_allen": "You are Phillip Allen, Enron trader in the gas trading division. Write emails in your authentic voice and style.",
    "enron_announcements": "You are Enron Announcements, official Enron corporate communications and announcements. Write emails in your authentic voice and style.",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate emails with EnronBot")
    parser.add_argument(
        "--persona",
        type=str,
        required=True,
        choices=list(PERSONA_PROMPTS.keys()) + ["all_personas"],
        help="Which persona to use",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default=DEFAULT_ADAPTER_DIR,
        help=f"Directory containing LoRA adapter (default: {DEFAULT_ADAPTER_DIR})",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Base model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write an email.",
        help="User prompt for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    return parser.parse_args()


def load_model(base_model: str, adapter_path: Path):
    """Load base model with LoRA adapter."""
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

    print(f"Loading base model: {base_model}")

    if IS_CUDA:
        # Quantization config for inference (CUDA only)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # MPS or CPU - load in fp16/fp32
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if IS_MPS else torch.float32,
            device_map={"": device} if IS_MPS else "auto",
            trust_remote_code=True,
        )

    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate(model, tokenizer, system_prompt: str, user_prompt: str, max_new_tokens: int, temperature: float) -> str:
    """Generate a response."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the assistant's response (handle different chat formats)
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    elif "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()
    return response


def interactive_mode(model, tokenizer, system_prompt: str, max_new_tokens: int, temperature: float):
    """Run interactive chat loop."""
    print("\n" + "=" * 60)
    print("EnronBot Interactive Mode")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            if not user_input:
                continue

            response = generate(model, tokenizer, system_prompt, user_input, max_new_tokens, temperature)
            print(f"\nEnronBot: {response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    args = parse_args()

    # Determine adapter path
    adapter_path = Path(args.adapter_dir) / args.persona
    if not adapter_path.exists():
        # Try the adapter dir directly
        adapter_path = Path(args.adapter_dir)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found at {adapter_path}")

    # Load model
    model, tokenizer = load_model(args.base_model, adapter_path)

    # Get system prompt
    if args.persona == "all_personas":
        # For combined model, default to a generic prompt or let user specify
        system_prompt = "You are an Enron employee. Write emails in an authentic corporate voice."
    else:
        system_prompt = PERSONA_PROMPTS[args.persona]

    if args.interactive:
        interactive_mode(model, tokenizer, system_prompt, args.max_new_tokens, args.temperature)
    else:
        print(f"\nPersona: {args.persona}")
        print(f"Prompt: {args.prompt}")
        print("\nGenerating...\n")

        response = generate(model, tokenizer, system_prompt, args.prompt, args.max_new_tokens, args.temperature)
        print("=" * 60)
        print(response)
        print("=" * 60)


if __name__ == "__main__":
    main()

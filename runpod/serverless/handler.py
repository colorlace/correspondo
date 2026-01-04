"""
RunPod Serverless Handler for EnronBot Inference

Provides HTTP API for generating emails in Enron employee styles.

API Request Format:
{
    "input": {
        "persona": "vince_kaminski",
        "prompt": "Write an email about the quarterly report",
        "max_tokens": 512,
        "temperature": 0.7
    }
}

API Response Format:
{
    "output": {
        "email": "Generated email text...",
        "persona": "vince_kaminski"
    }
}
"""

import os
from pathlib import Path

import runpod
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Configuration
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/workspace/models/enronbot"))
BASE_MODEL = os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

# Persona system prompts
PERSONA_PROMPTS = {
    "vince_kaminski": (
        "You are Vince Kaminski, Head of Enron's Research Group, expert in "
        "quantitative analysis and risk management. Write emails in your "
        "authentic voice and style."
    ),
    "kate_symes": (
        "You are Kate Symes, Enron employee in the trading division. "
        "Write emails in your authentic voice and style."
    ),
    "jeff_dasovich": (
        "You are Jeff Dasovich, Enron government affairs representative "
        "focused on California energy policy. Write emails in your "
        "authentic voice and style."
    ),
    "phillip_allen": (
        "You are Phillip Allen, Enron trader in the gas trading division. "
        "Write emails in your authentic voice and style."
    ),
    "enron_announcements": (
        "You are Enron Announcements, official Enron corporate communications "
        "and announcements. Write emails in your authentic voice and style."
    ),
    "all_personas": (
        "You are an Enron employee. Write emails in an authentic corporate voice."
    ),
}

# Global model cache - models are loaded once and reused
_model_cache = {}
_tokenizer = None


def get_quantization_config():
    """4-bit quantization for memory-efficient inference."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_model(persona: str):
    """Load model with LoRA adapter, caching for reuse."""
    global _model_cache, _tokenizer

    # Load tokenizer once
    if _tokenizer is None:
        print(f"Loading tokenizer from {BASE_MODEL}...")
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        _tokenizer.pad_token = _tokenizer.eos_token

    # Load model for persona if not cached
    if persona not in _model_cache:
        adapter_path = MODEL_DIR / persona
        if not adapter_path.exists():
            available = [p.name for p in MODEL_DIR.iterdir() if p.is_dir()]
            raise ValueError(
                f"Adapter not found for persona '{persona}'. "
                f"Available: {available}"
            )

        print(f"Loading base model with 4-bit quantization...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=get_quantization_config(),
            device_map="auto",
            trust_remote_code=True,
        )

        print(f"Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        _model_cache[persona] = model
        print(f"Model loaded and cached for persona: {persona}")

    return _model_cache[persona], _tokenizer


def generate_email(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Generate an email response."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
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

    # Extract assistant response (handle different chat formats)
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    elif "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()

    return response


def handler(event):
    """
    RunPod Serverless Handler

    Input:
        event["input"]["persona"] - Which persona to use (required)
        event["input"]["prompt"] - User prompt for generation (required)
        event["input"]["max_tokens"] - Max tokens to generate (default: 512)
        event["input"]["temperature"] - Sampling temperature (default: 0.7)

    Returns:
        {"output": {"email": "...", "persona": "..."}}
        or {"error": "error message"}
    """
    try:
        input_data = event.get("input", {})

        # Extract and validate parameters
        persona = input_data.get("persona")
        prompt = input_data.get("prompt")

        if not persona:
            return {"error": "Missing required field: persona"}
        if not prompt:
            return {"error": "Missing required field: prompt"}

        if persona not in PERSONA_PROMPTS:
            return {
                "error": f"Invalid persona '{persona}'. "
                f"Available: {list(PERSONA_PROMPTS.keys())}"
            }

        max_tokens = input_data.get("max_tokens", 512)
        temperature = input_data.get("temperature", 0.7)

        # Validate ranges
        max_tokens = min(max(int(max_tokens), 1), 2048)
        temperature = min(max(float(temperature), 0.1), 2.0)

        # Load model and generate
        model, tokenizer = load_model(persona)
        system_prompt = PERSONA_PROMPTS[persona]

        email = generate_email(
            model,
            tokenizer,
            system_prompt,
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

        return {
            "output": {
                "email": email,
                "persona": persona,
            }
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# Start the serverless worker
if __name__ == "__main__":
    print("Starting EnronBot RunPod Serverless Handler...")
    print(f"Base model: {BASE_MODEL}")
    print(f"Model directory: {MODEL_DIR}")
    print(f"Available personas: {list(PERSONA_PROMPTS.keys())}")
    runpod.serverless.start({"handler": handler})

#!/bin/bash
# RunPod Training Entrypoint for EnronBot
#
# Usage:
#   ./start_training.sh [persona] [additional_args...]
#
# Examples:
#   ./start_training.sh all_personas
#   ./start_training.sh vince_kaminski --epochs 5
#   ./start_training.sh all_personas --epochs 3 --batch-size 4
#
# Environment variables:
#   HF_TOKEN        - Hugging Face token (required for Mistral access)
#   WANDB_API_KEY   - Weights & Biases API key (optional)

set -e

PERSONA="${1:-all_personas}"
shift || true  # Remove first arg, continue if empty

echo "=============================================="
echo "  EnronBot Training on RunPod"
echo "=============================================="

# Show GPU info
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo "GPU: $GPU_NAME ($GPU_MEM)"
else
    echo "GPU: Not detected"
fi

echo "Persona: $PERSONA"
echo "=============================================="

# Login to Hugging Face if token provided
if [ -n "$HF_TOKEN" ]; then
    echo "Logging into Hugging Face..."
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    echo "HuggingFace login successful"
else
    echo "WARNING: HF_TOKEN not set. May fail for gated models like Mistral."
fi

# Handle Weights & Biases
EXTRA_ARGS=""
if [ -n "$WANDB_API_KEY" ]; then
    echo "Weights & Biases logging enabled"
    wandb login "$WANDB_API_KEY" 2>/dev/null || true
else
    echo "W&B disabled (no WANDB_API_KEY)"
    EXTRA_ARGS="--no-wandb"
fi

echo "=============================================="
echo "Starting training..."
echo "=============================================="

# Run training
python scripts/train.py \
    --persona "$PERSONA" \
    --output-dir /workspace/models/enronbot \
    $EXTRA_ARGS \
    "$@"

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo "Adapter saved to: /workspace/models/enronbot/$PERSONA/"
echo ""
echo "To download the adapter:"
echo "  1. Use RunPod UI file browser"
echo "  2. Or zip and download:"
echo "     cd /workspace && zip -r adapter.zip models/enronbot/$PERSONA/"
echo "=============================================="

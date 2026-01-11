# RunPod Training Dockerfile for Correspondo
# Fine-tunes LLMs using QLoRA to write in the voice of different personas
#
# Build:
#   docker build -f runpod/training/Dockerfile -t yourdockerhub/correspondo-train .
#
# Run locally (test):
#   docker run --gpus all -it yourdockerhub/correspondo-train bash
#
# On RunPod:
#   1. Create Pod with this image
#   2. SSH in and run: ./start_training.sh all_personas

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training scripts and data
COPY scripts/ ./scripts/
COPY data/training/ ./data/training/

# Copy entrypoint script
COPY runpod/training/start_training.sh .
RUN chmod +x start_training.sh

# Create models directory for output
RUN mkdir -p /workspace/models/correspondo /workspace/.cache/huggingface

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface

# Default command - interactive shell for training
CMD ["bash", "-lc", "sleep infinity"]

# RunPod Training Dockerfile for EnronBot
# Fine-tunes LLMs using QLoRA on Enron employee emails
#
# Build:
#   docker build -f runpod/training/Dockerfile -t yourdockerhub/enronbot-train .
#
# Run locally (test):
#   docker run --gpus all -it yourdockerhub/enronbot-train bash
#
# On RunPod:
#   1. Create Pod with this image
#   2. SSH in and run: ./start_training.sh all_personas

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training scripts
COPY scripts/ ./scripts/

# Copy training data
COPY data/training/ ./data/training/

# Copy entrypoint script
COPY runpod/training/start_training.sh .
RUN chmod +x start_training.sh

# Create models directory for output
RUN mkdir -p models/enronbot

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface

# Default command - interactive shell for training
CMD ["bash"]

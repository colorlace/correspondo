# RunPod Serverless Dockerfile for Correspondo Inference
#
# This creates a serverless endpoint that serves the fine-tuned model via HTTP API.
#
# Build (after training, with adapters):
#   docker build -f runpod/serverless/Dockerfile -t yourdockerhub/correspondo-api .
#
# The image expects trained adapters at /workspace/models/correspondo/[persona]/
# You can either:
#   1. COPY them into the image (larger image, faster cold start)
#   2. Mount a RunPod Network Volume (smaller image, slower cold start)

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /workspace

# Install inference dependencies
COPY runpod/serverless/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the handler
COPY runpod/serverless/handler.py .

# Create models directory (will be populated by volume or COPY)
RUN mkdir -p models/correspondo

# Uncomment to bake trained adapters into image (after training):
# COPY models/correspondo/ ./models/correspondo/

# Pre-download tokenizer to speed up cold start
# Uncomment after setting up HF token:
# ARG HF_TOKEN
# RUN python -c "from transformers import AutoTokenizer; \
#     AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2', \
#     token='${HF_TOKEN}')"

# Environment
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV MODEL_DIR=/workspace/models/correspondo
ENV BASE_MODEL=mistralai/Mistral-7B-Instruct-v0.2

# Start the serverless handler
CMD ["python", "-u", "handler.py"]

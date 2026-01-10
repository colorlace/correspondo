# Correspondo Training Guide

Fine-tune LLMs using QLoRA to write in the voice of different personas.

## Requirements

- GPU with 24GB+ VRAM (A100, A6000, RTX 4090)
- ~50GB disk space
- Python 3.10+

## Cloud GPU Options

### Option 1: RunPod (Recommended)
1. Go to [runpod.io](https://runpod.io)
2. Deploy a GPU Pod:
   - Template: `RunPod Pytorch 2.1`
   - GPU: A100 40GB (~$1.50/hr) or A6000 (~$0.80/hr)
   - Storage: 50GB
3. Connect via SSH or web terminal

### Option 2: Lambda Labs
1. Go to [lambdalabs.com](https://lambdalabs.com)
2. Launch an instance with A100 or A6000
3. SSH in

### Option 3: Vast.ai (Budget)
1. Go to [vast.ai](https://vast.ai)
2. Filter for A100/A6000/4090 with PyTorch template
3. Rent and connect

---

## Setup

```bash
# Clone/upload your project
cd /workspace  # or your cloud instance directory
git clone <your-repo> grandadbot
cd grandadbot

# Install dependencies
pip install -r requirements.txt

# Login to Hugging Face (required for Mistral)
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens

# Optional: Login to Weights & Biases for tracking
wandb login
```

## Training

### Train on all personas (recommended for single model)
```bash
python scripts/train.py --persona all_personas
```

### Train on a specific persona
```bash
python scripts/train.py --persona vince_kaminski
python scripts/train.py --persona jeff_dasovich
python scripts/train.py --persona kate_symes
python scripts/train.py --persona phillip_allen
python scripts/train.py --persona enron_announcements
```

### Custom parameters
```bash
python scripts/train.py \
    --persona all_personas \
    --epochs 5 \
    --batch-size 2 \
    --learning-rate 1e-4 \
    --max-seq-length 2048 \
    --output-dir ./models/correspondo-custom
```

### Disable Weights & Biases
```bash
python scripts/train.py --no-wandb
```

## Expected Training Time

| GPU | all_personas (~30k examples) | Single persona (~5-8k) |
|-----|------------------------------|------------------------|
| A100 40GB | ~2-3 hours | ~30-60 min |
| A6000 48GB | ~3-4 hours | ~45-90 min |
| RTX 4090 | ~3-4 hours | ~45-90 min |

## Output

After training, you'll find:
```
models/correspondo/
└── all_personas/           # or persona name
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── ...
```

## Inference

### Single generation
```bash
python scripts/inference.py \
    --persona all_personas \
    --prompt "Write a memo about the quarterly earnings report"
```

### Interactive mode
```bash
python scripts/inference.py --persona vince_kaminski --interactive
```

### Switch personas (if trained separately)
```bash
python scripts/inference.py --persona kate_symes --prompt "Write about trading strategies"
python scripts/inference.py --persona enron_announcements --prompt "Write a company announcement"
```

## Download Model to Local

After training, download your adapter:
```bash
# On cloud instance
zip -r correspondo-adapter.zip models/correspondo/all_personas/

# Then use scp, rsync, or cloud provider's download feature
```

## Troubleshooting

### Out of memory
- Reduce `--batch-size` to 1 or 2
- Reduce `--max-seq-length` to 512
- Increase `--gradient-accumulation-steps` to compensate

### Slow training
- Make sure you're using GPU (`nvidia-smi` should show usage)
- Check that bf16 is enabled (default)

### Model quality issues
- Try more epochs (5-10)
- Increase LoRA rank (edit `get_lora_config()` in train.py, try r=32 or r=64)
- Lower learning rate (1e-4 or 5e-5)

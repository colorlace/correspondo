#!/bin/bash
# Correspondo Setup Script
# Creates conda environment and installs dependencies for training

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ENV_NAME="correspondo"
PYTHON_VERSION="3.10"

echo "================================================"
echo "  Correspondo Fine-Tuning Environment Setup"
echo "================================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}✗${NC} Conda not found. Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo -e "${GREEN}✓${NC} Conda found"

# Initialize conda for this shell session
eval "$(conda shell.bash hook)"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${GREEN}✓${NC} Conda environment '$ENV_NAME' already exists"
    read -p "Recreate environment from scratch? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
        echo "Creating fresh conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
        conda create -n $ENV_NAME python=$PYTHON_VERSION -y
    fi
else
    echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
fi

# Activate the environment
echo ""
echo "Activating conda environment '$ENV_NAME'..."
conda activate $ENV_NAME
echo -e "${GREEN}✓${NC} Environment activated"

# Verify Python version
echo ""
echo "Checking Python version..."
ACTUAL_PYTHON=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓${NC} Python $ACTUAL_PYTHON detected"

# Upgrade pip
echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel --quiet

# Install PyTorch with appropriate backend
echo ""
echo "Detecting compute backend..."
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} NVIDIA GPU detected"
    echo "Installing PyTorch with CUDA support..."
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
elif [[ $(uname) == "Darwin" ]] && [[ $(uname -m) == "arm64" ]]; then
    echo -e "${GREEN}✓${NC} Apple Silicon detected"
    echo "    Installing PyTorch with MPS support..."
    python -m pip install torch torchvision torchaudio --quiet
else
    echo -e "${YELLOW}⚠${NC}  No GPU detected, installing CPU-only PyTorch"
    echo "Note: Training will be very slow on CPU. Consider using cloud GPU."
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
fi

# Install other requirements
echo ""
echo "Installing ML dependencies from requirements.txt..."
python -m pip install -r requirements.txt --quiet

# Verify installations
echo ""
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
python -c "import trl; print(f'TRL: {trl.__version__}')"

# Check GPU availability
echo ""
echo "Checking compute devices..."
python << EOF
import torch
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
elif torch.backends.mps.is_available():
    print("✓ MPS (Apple Silicon) available")
else:
    print("⚠ No GPU acceleration available (CPU only)")
    print("  Training will be very slow. Consider using:")
    print("  - RunPod (https://runpod.io)")
    print("  - Lambda Labs (https://lambdalabs.com)")
    print("  - Google Colab Pro (https://colab.research.google.com)")
EOF

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p models data/raw data/training

# Check for data files
echo ""
echo "Checking data availability..."
if [ -f "emails.csv" ]; then
    echo -e "${GREEN}✓${NC} emails.csv found"
else
    echo -e "${YELLOW}⚠${NC}  emails.csv not found (required for data preparation)"
fi

if [ -f "data/training/vince_kaminski.jsonl" ]; then
    TRAIN_COUNT=$(wc -l < data/training/vince_kaminski.jsonl)
    echo -e "${GREEN}✓${NC} Training data found ($TRAIN_COUNT examples in vince_kaminski.jsonl)"
else
    echo -e "${YELLOW}⚠${NC}  No training data found. Run: python scripts/prepare_data.py"
fi

# Summary
echo ""
echo "================================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "================================================"
echo ""
echo "Environment: $ENV_NAME"
echo ""
echo "To activate this environment in a new terminal:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Next steps:"
echo "  1. Prepare training data (if not done):"
echo "     python scripts/prepare_data.py"
echo ""
echo "  2. Start training:"
echo "     python scripts/train.py --persona vince_kaminski"
echo ""
echo "  3. Test the model:"
echo "     python scripts/inference.py --persona vince_kaminski --interactive"
echo ""
echo "For detailed instructions, see TRAINING.md"
echo ""

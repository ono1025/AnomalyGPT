#!/bin/bash
# Script to download all required checkpoints for AnomalyGPT

set -e  # Exit on error

echo "================================================"
echo "AnomalyGPT Checkpoint Download Script"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Create directories
echo "Creating checkpoint directories..."
mkdir -p pretrained_ckpt/imagebind_ckpt
mkdir -p pretrained_ckpt/pandagpt_ckpt/7b
mkdir -p pretrained_ckpt/vicuna_ckpt
mkdir -p code/ckpt/train_mvtec
print_status "Directories created"
echo ""

# Download ImageBind checkpoint
echo "================================================"
echo "1. Downloading ImageBind checkpoint (~5GB)"
echo "================================================"
if [ -f "pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth" ]; then
    print_warning "ImageBind checkpoint already exists, skipping download"
else
    wget -c -O pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth \
        https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth
    print_status "ImageBind checkpoint downloaded"
fi
echo ""

# Download PandaGPT checkpoint
echo "================================================"
echo "2. Downloading PandaGPT 7B checkpoint (~13GB)"
echo "================================================"
if [ -f "pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt" ]; then
    print_warning "PandaGPT checkpoint already exists, skipping download"
else
    wget -c -O pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt \
        https://huggingface.co/openllmplayground/pandagpt_7b_max_len_1024/resolve/main/pytorch_model.pt
    print_status "PandaGPT checkpoint downloaded"
fi
echo ""

# Vicuna checkpoint
echo "================================================"
echo "3. Vicuna Checkpoint Setup"
echo "================================================"
print_warning "Vicuna requires manual setup (LLaMA weights + delta)"
print_warning "Follow instructions in COLAB_SETUP_GUIDE.md Section 3.2"
echo ""
echo "Quick steps:"
echo "  1. Obtain LLaMA 7B weights from Meta"
echo "  2. Convert to HuggingFace format"
echo "  3. Download Vicuna delta weights"
echo "  4. Combine using FastChat tools"
echo ""

if [ -f "pretrained_ckpt/vicuna_ckpt/7b_v0/config.json" ]; then
    print_status "Vicuna checkpoint found"
else
    print_error "Vicuna checkpoint not found"
    echo "  Expected location: pretrained_ckpt/vicuna_ckpt/7b_v0/"
fi
echo ""

# Download pre-trained AnomalyGPT weights (optional)
echo "================================================"
echo "4. Pre-trained AnomalyGPT Weights (Optional)"
echo "================================================"
echo "These weights allow you to skip training."
read -p "Download pre-trained AnomalyGPT MVTec weights (~13GB)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "code/ckpt/train_mvtec/pytorch_model.pt" ]; then
        print_warning "AnomalyGPT weights already exist, skipping download"
    else
        wget -c -O code/ckpt/train_mvtec/pytorch_model.pt \
            https://huggingface.co/FantasticGNU/AnomalyGPT/resolve/main/train_mvtec/pytorch_model.pt
        print_status "AnomalyGPT weights downloaded"
    fi
else
    print_warning "Skipped AnomalyGPT weights download"
    echo "  You will need to train the model yourself"
fi
echo ""

# Verify downloads
echo "================================================"
echo "5. Verification"
echo "================================================"

verify_file() {
    if [ -f "$1" ]; then
        size=$(du -h "$1" | cut -f1)
        print_status "$1 ($size)"
        return 0
    else
        print_error "$1 (missing)"
        return 1
    fi
}

all_ok=true

echo "Checking downloaded files:"
verify_file "pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth" || all_ok=false
verify_file "pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt" || all_ok=false

if [ -f "pretrained_ckpt/vicuna_ckpt/7b_v0/config.json" ]; then
    print_status "pretrained_ckpt/vicuna_ckpt/7b_v0/ (found)"
else
    print_error "pretrained_ckpt/vicuna_ckpt/7b_v0/ (missing)"
    all_ok=false
fi

if [ -f "code/ckpt/train_mvtec/pytorch_model.pt" ]; then
    verify_file "code/ckpt/train_mvtec/pytorch_model.pt" || all_ok=false
else
    print_warning "code/ckpt/train_mvtec/pytorch_model.pt (not downloaded - training required)"
fi

echo ""

if [ "$all_ok" = true ] && [ -f "pretrained_ckpt/vicuna_ckpt/7b_v0/config.json" ]; then
    print_status "All essential checkpoints ready!"
    echo ""
    echo "Next steps:"
    echo "  1. Download MVTec-AD dataset (see COLAB_SETUP_GUIDE.md Section 4)"
    echo "  2. Run inference: cd code && python test_mvtec.py"
    echo "  3. Or start training: cd code && bash scripts/train_mvtec.sh"
else
    print_error "Some checkpoints are missing"
    echo ""
    echo "Missing items:"
    [ -f "pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth" ] || echo "  - ImageBind checkpoint"
    [ -f "pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt" ] || echo "  - PandaGPT checkpoint"
    [ -f "pretrained_ckpt/vicuna_ckpt/7b_v0/config.json" ] || echo "  - Vicuna checkpoint"
    echo ""
    echo "Refer to COLAB_SETUP_GUIDE.md for detailed setup instructions."
fi

echo ""
echo "================================================"
echo "Download script completed"
echo "================================================"

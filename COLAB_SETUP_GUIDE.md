# AnomalyGPT: Complete Google Colab Setup and Training Guide

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [Environment Setup](#environment-setup)
3. [Checkpoint Preparation](#checkpoint-preparation)
4. [Dataset Preparation (MVTec-AD)](#dataset-preparation-mvtec-ad)
5. [Training Pipeline](#training-pipeline)
6. [Inference and Evaluation](#inference-and-evaluation)
7. [Common Issues and Solutions](#common-issues-and-solutions)

---

## 1. Repository Overview

### 1.1 Repository Structure

```
AnomalyGPT/
├── code/                          # Main code directory
│   ├── train_mvtec.py            # Training script for MVTec-AD
│   ├── train_visa.py             # Training script for VisA
│   ├── test_mvtec.py             # Inference/evaluation for MVTec-AD
│   ├── test_visa.py              # Inference/evaluation for VisA
│   ├── web_demo.py               # Web-based demo interface
│   ├── model/                     # Model implementations
│   │   ├── openllama.py          # Main LLM wrapper (OpenLLAMAPEFTModel)
│   │   ├── AnomalyGPT_models.py  # PromptLearner, LinearLayer
│   │   ├── ImageBind/            # Vision encoder (from Meta)
│   │   ├── modeling_llama.py     # Modified LLaMA architecture
│   │   └── agent.py              # Agent wrapper for training
│   ├── datasets/                  # Dataset loaders
│   │   ├── mvtec.py              # MVTec-AD dataset with self-supervised augmentation
│   │   ├── visa.py               # VisA dataset
│   │   ├── sft_dataset.py        # Supervised fine-tuning dataset (PandaGPT)
│   │   └── self_sup_tasks.py     # Patch exchange augmentation
│   ├── config/                    # Configuration files
│   │   ├── openllama_peft.yaml   # Training hyperparameters
│   │   └── base.yaml
│   ├── dsconfig/                  # DeepSpeed configurations
│   │   └── openllama_peft_stage_1.json
│   ├── scripts/                   # Shell scripts
│   │   ├── train_mvtec.sh        # Training launcher for MVTec-AD
│   │   ├── train_visa.sh
│   │   └── train_all_supervised_cn.sh
│   └── utils/                     # Utility functions
├── data/                          # Data directory (you'll add datasets here)
├── pretrained_ckpt/              # Pre-trained model checkpoints
│   ├── imagebind_ckpt/           # ImageBind checkpoint
│   ├── vicuna_ckpt/              # Vicuna LLM checkpoint
│   └── pandagpt_ckpt/            # PandaGPT delta weights
└── requirements.txt              # Python dependencies
```

### 1.2 Key Components

**Architecture Overview:**

1. **Vision Encoder (ImageBind)**: 
   - Multi-modal encoder from Meta
   - Converts images to 1024-dimensional embeddings
   - Location: `code/model/ImageBind/`
   
2. **Prompt Learner**:
   - Generates image-specific prompts via CNN meta-network
   - Combines base prompts with image-derived prompts
   - Location: `code/model/AnomalyGPT_models.py` → `PromptLearner` class

3. **LLM Interface (Vicuna)**:
   - Based on LLaMA 7B/13B with instruction tuning
   - Uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
   - Location: `code/model/openllama.py` → `OpenLLAMAPEFTModel` class

4. **Anomaly Decoder**:
   - Feature-matching between test image and normal references
   - Produces pixel-level anomaly maps
   - Integrated in `OpenLLAMAPEFTModel.generate()` method

### 1.3 Training Modes

1. **Unsupervised (One-Class)**: Train on single dataset (e.g., MVTec-AD only)
   - Uses self-supervised patch exchange for simulated anomalies
   - Script: `train_mvtec.py` or `train_visa.py`
   
2. **Supervised (Multi-Dataset)**: Train on multiple datasets with real anomalies
   - Requires labeled anomaly data
   - Script: `train_all_supervised_cn.py`

### 1.4 Data Flow

**Training:**
```
Normal Images (MVTec-AD train/good/) 
    → Self-supervised Augmentation (patch_ex)
    → ImageBind Encoder 
    → Prompt Learner
    → Vicuna LLM (with LoRA)
    → Text Response + Anomaly Map
```

**Inference:**
```
Test Image + Few Normal References
    → ImageBind Encoder
    → Feature Matching (compute anomaly score)
    → Vicuna LLM Generation
    → Text Response ("Yes/No") + Anomaly Map (224×224)
```

---

## 2. Environment Setup

### 2.1 System Requirements

**Minimum:**
- GPU: NVIDIA T4 (16GB VRAM) or better
- RAM: 12GB+ system RAM
- Storage: 50GB+ free space
- CUDA: 11.7+

**Recommended:**
- GPU: NVIDIA A100 (40GB) or V100 (32GB)
- RAM: 25GB+ system RAM
- Storage: 100GB+ free space

### 2.2 Python and CUDA Versions

```bash
# Check your Colab environment
!nvidia-smi
!python --version
```

**Target versions** (based on requirements.txt):
- Python: 3.8-3.10
- CUDA: 11.7
- PyTorch: 1.13.1+cu117

### 2.3 Installation Steps for Google Colab

#### Step 1: Clone the Repository

```python
import os

# Clone repository
!git clone https://github.com/CASIA-IVA-Lab/AnomalyGPT.git
%cd AnomalyGPT
```

#### Step 2: Install Dependencies

```bash
# Install core dependencies
!pip install -q torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# Install other requirements
!pip install -q deepspeed==0.9.2
!pip install -q transformers==4.29.1
!pip install -q peft==0.3.0
!pip install -q sentencepiece
!pip install -q einops==0.6.1
!pip install -q timm==0.6.7
!pip install -q opencv-python==4.8.0.74
!pip install -q scikit-learn==1.3.0
!pip install -q gradio==3.41.2
!pip install -q kornia==0.7.0
!pip install -q ftfy==6.1.1
!pip install -q regex==2022.10.31
!pip install -q Pillow==10.0.0
!pip install -q easydict==1.10
!pip install -q iopath==0.1.10
!pip install -q pytorchvideo==0.1.5
!pip install -q matplotlib==3.7.2
!pip install -q tqdm==4.64.1
```

**Alternative (using requirements.txt):**
```bash
!pip install -q -r requirements.txt
```

**Note:** If you encounter dependency conflicts:
```bash
!pip install --upgrade pip
!pip install -q torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 --force-reinstall
```

#### Step 3: Verify Installation

```python
import torch
import transformers
import deepspeed
import peft

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Transformers version: {transformers.__version__}")
print(f"DeepSpeed version: {deepspeed.__version__}")
print(f"PEFT version: {peft.__version__}")
```

Expected output:
```
PyTorch version: 1.13.1+cu117
CUDA available: True
CUDA version: 11.7
Transformers version: 4.29.1
DeepSpeed version: 0.9.2
PEFT version: 0.3.0
```

### 2.4 Mount Google Drive (Optional but Recommended)

For persistent storage of checkpoints and datasets:

```python
from google.colab import drive
drive.mount('/content/drive')

# Create working directory
!mkdir -p /content/drive/MyDrive/AnomalyGPT_workspace
%cd /content/drive/MyDrive/AnomalyGPT_workspace

# Clone or move repository here
```

---

## 3. Checkpoint Preparation

You need **three** sets of pre-trained weights before training:

### 3.1 ImageBind Checkpoint

**Download:**
```bash
# Create directory
!mkdir -p pretrained_ckpt/imagebind_ckpt

# Download ImageBind weights (~5GB)
!wget -O pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth \
    https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth

# Verify
!ls -lh pretrained_ckpt/imagebind_ckpt/
```

Expected: `imagebind_huge.pth` (~4.9GB)

### 3.2 Vicuna Checkpoint

Vicuna requires **LLaMA base weights** + **Vicuna delta weights**.

#### Option A: Direct Download (Easier, if available)

Some repositories provide pre-combined Vicuna weights:

```bash
# This is a placeholder - check HuggingFace for actual links
# Example structure:
!mkdir -p pretrained_ckpt/vicuna_ckpt/7b_v0
```

#### Option B: Manual Combination (Official Method)

**Step 1: Get LLaMA Base Weights**

LLaMA weights require Meta approval:
1. Fill form: https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform
2. Convert to HuggingFace format following: https://huggingface.co/docs/transformers/main/model_doc/llama

**Step 2: Get Vicuna Delta Weights**

```bash
!pip install -q huggingface_hub

from huggingface_hub import snapshot_download

# Download Vicuna 7B delta v0
snapshot_download(
    repo_id="lmsys/vicuna-7b-delta-v0",
    local_dir="pretrained_ckpt/vicuna_delta_7b_v0",
    local_dir_use_symlinks=False
)
```

**Step 3: Combine Weights**

```bash
# Install FastChat
!pip install -q git+https://github.com/lm-sys/FastChat.git@v0.1.10

# Combine weights
!python -m fastchat.model.apply_delta \
    --base {path_to_llama_7b} \
    --target pretrained_ckpt/vicuna_ckpt/7b_v0/ \
    --delta pretrained_ckpt/vicuna_delta_7b_v0
```

**Expected structure:**
```
pretrained_ckpt/vicuna_ckpt/7b_v0/
├── config.json
├── generation_config.json
├── pytorch_model-00001-of-00002.bin
├── pytorch_model-00002-of-00002.bin
├── pytorch_model.bin.index.json
├── special_tokens_map.json
├── tokenizer.model
└── tokenizer_config.json
```

### 3.3 PandaGPT Delta Weights

PandaGPT provides initialization for AnomalyGPT:

```bash
!mkdir -p pretrained_ckpt/pandagpt_ckpt/7b

# Download PandaGPT 7B weights (~13GB)
!wget -O pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt \
    https://huggingface.co/openllmplayground/pandagpt_7b_max_len_1024/resolve/main/pytorch_model.pt

# Verify
!ls -lh pretrained_ckpt/pandagpt_ckpt/7b/
```

**Available PandaGPT versions:**
- 7B, max_len=512: `openllmplayground/pandagpt_7b_max_len_512`
- **7B, max_len=1024** (recommended): `openllmplayground/pandagpt_7b_max_len_1024`
- 13B, max_len=256: `openllmplayground/pandagpt_13b_max_len_256`
- 13B, max_len=400: `openllmplayground/pandagpt_13b_max_len_400`

### 3.4 Download PandaGPT Pre-training Data

For the supervised fine-tuning component:

```bash
!mkdir -p data/images

# Download PandaGPT visual instruction data
!wget -O data/pandagpt4_visual_instruction_data.json \
    https://huggingface.co/datasets/openllmplayground/pandagpt_visual_instruction_dataset/resolve/main/pandagpt4_visual_instruction_data.json

# Download images (this is large, ~50GB)
# Consider downloading a subset or using wget with resume
!wget -c -O data/images.tar.gz \
    https://huggingface.co/datasets/openllmplayground/pandagpt_visual_instruction_dataset/resolve/main/images.tar.gz

# Extract
!tar -xzf data/images.tar.gz -C data/
!rm data/images.tar.gz  # Clean up
```

**Alternative for limited storage:**
```python
# Download only the JSON for now, images can be fetched on-demand
!wget -O data/pandagpt4_visual_instruction_data.json \
    https://huggingface.co/datasets/openllmplayground/pandagpt_visual_instruction_dataset/resolve/main/pandagpt4_visual_instruction_data.json
```

### 3.5 Checkpoint Verification

```bash
# Final checkpoint structure
!tree -L 3 pretrained_ckpt/

# Expected output:
# pretrained_ckpt/
# ├── imagebind_ckpt/
# │   └── imagebind_huge.pth
# ├── pandagpt_ckpt/
# │   └── 7b/
# │       └── pytorch_model.pt
# └── vicuna_ckpt/
#     └── 7b_v0/
#         ├── config.json
#         └── [other files...]
```

---

## 4. Dataset Preparation (MVTec-AD)

### 4.1 Download MVTec-AD

**Option 1: Official Download (Requires Registration)**

1. Visit: https://www.mvtec.com/company/research/datasets/mvtec-ad
2. Fill registration form
3. Download `mvtec_anomaly_detection.tar.xz` (~4.9GB compressed)

```bash
# After download, upload to Colab or Google Drive
!mkdir -p data
!tar -xf mvtec_anomaly_detection.tar.xz -C data/
```

**Option 2: Direct Download (if available)**

```bash
# Some mirrors may exist, but verify integrity
# Example placeholder:
!mkdir -p data
!wget -O data/mvtec_ad.tar.xz [MIRROR_URL]
!tar -xf data/mvtec_ad.tar.xz -C data/
```

**Option 3: Upload from Local Machine**

```python
from google.colab import files

# Upload the tar.xz file
uploaded = files.upload()

# Extract
!tar -xf mvtec_anomaly_detection.tar.xz -C data/
```

### 4.2 MVTec-AD Dataset Structure

**Expected directory layout:**

```
data/mvtec_anomaly_detection/
├── bottle/
│   ├── train/
│   │   └── good/              # Normal training images (e.g., 000.png - 209.png)
│   ├── test/
│   │   ├── good/              # Normal test images
│   │   ├── broken_large/      # Anomaly type 1
│   │   ├── broken_small/      # Anomaly type 2
│   │   └── contamination/     # Anomaly type 3
│   └── ground_truth/
│       ├── broken_large/      # Masks: {img_name}_mask.png
│       ├── broken_small/
│       └── contamination/
├── cable/
│   ├── train/good/
│   ├── test/...
│   └── ground_truth/...
├── capsule/
├── carpet/
├── grid/
├── hazelnut/
├── leather/
├── metal_nut/
├── pill/
├── screw/
├── tile/
├── toothbrush/
├── transistor/
├── wood/
└── zipper/
```

**15 object/texture categories** in total.

### 4.3 Dataset Characteristics

| Category    | Type    | Train (Normal) | Test (Normal) | Test (Anomaly) | Anomaly Types |
|-------------|---------|----------------|---------------|----------------|---------------|
| Bottle      | Object  | 209            | 20            | 63             | 3             |
| Cable       | Object  | 224            | 58            | 92             | 8             |
| Capsule     | Object  | 219            | 23            | 109            | 5             |
| Carpet      | Texture | 280            | 28            | 89             | 5             |
| Grid        | Texture | 264            | 21            | 57             | 5             |
| Hazelnut    | Object  | 391            | 40            | 70             | 4             |
| Leather     | Texture | 245            | 32            | 92             | 5             |
| Metal Nut   | Object  | 220            | 22            | 93             | 4             |
| Pill        | Object  | 267            | 26            | 141            | 7             |
| Screw       | Object  | 320            | 41            | 119            | 5             |
| Tile        | Texture | 230            | 33            | 84             | 5             |
| Toothbrush  | Object  | 60             | 12            | 30             | 1             |
| Transistor  | Object  | 213            | 60            | 40             | 4             |
| Wood        | Texture | 247            | 19            | 60             | 5             |
| Zipper      | Object  | 240            | 32            | 119            | 7             |

### 4.4 Verify Dataset Loading

```python
import os

# Count samples
def count_mvtec_samples(root_dir):
    categories = os.listdir(root_dir)
    for cat in sorted(categories):
        cat_path = os.path.join(root_dir, cat)
        if not os.path.isdir(cat_path):
            continue
        
        train_good = os.path.join(cat_path, 'train/good')
        test_dir = os.path.join(cat_path, 'test')
        
        train_count = len(os.listdir(train_good)) if os.path.exists(train_good) else 0
        
        test_good = os.path.join(test_dir, 'good')
        test_good_count = len(os.listdir(test_good)) if os.path.exists(test_good) else 0
        
        test_anomaly_count = 0
        if os.path.exists(test_dir):
            for subdir in os.listdir(test_dir):
                if subdir != 'good':
                    test_anomaly_count += len(os.listdir(os.path.join(test_dir, subdir)))
        
        print(f"{cat:15s} | Train: {train_count:3d} | Test Normal: {test_good_count:3d} | Test Anomaly: {test_anomaly_count:3d}")

count_mvtec_samples('data/mvtec_anomaly_detection')
```

### 4.5 Dataset Path Configuration

The dataset path is **hardcoded** in `code/datasets/__init__.py`:

```python
# Line 52 in code/datasets/__init__.py
data = MVtecDataset('../data/mvtec_anomaly_detection')
```

**To modify for Colab:**

```python
# If your data is at /content/drive/MyDrive/AnomalyGPT_workspace/data/mvtec_anomaly_detection
# Edit code/datasets/__init__.py, line 52:

# Original:
data = MVtecDataset('../data/mvtec_anomaly_detection')

# Change to absolute path:
data = MVtecDataset('/content/AnomalyGPT/data/mvtec_anomaly_detection')
```

Or create a symbolic link:
```bash
!ln -s /content/drive/MyDrive/mvtec_anomaly_detection /content/AnomalyGPT/data/mvtec_anomaly_detection
```

### 4.6 How Dataset is Loaded

**Training (code/datasets/mvtec.py):**

1. **Reads all normal training images** from `{category}/train/good/*.png`
2. **Self-supervised augmentation** via `patch_ex()`:
   - Randomly extracts patches from another normal image
   - Pastes patches onto current image to simulate anomalies
   - Generates ground truth masks for pasted regions
3. **Creates two samples per image**:
   - Original image → label: "No anomaly"
   - Augmented image → label: "Yes, anomaly at [position]"

**Inference (code/test_mvtec.py):**

1. **Test images** from `{category}/test/{defect_type}/*.png`
2. **Ground truth masks** from `{category}/ground_truth/{defect_type}/{img_name}_mask.png`
3. **Normal reference images** (few-shot): 4 images from `{category}/train/good/`
   - Specified by `--k_shot` parameter (default: 1)
   - Selected by `--round` parameter for reproducibility

---

## 5. Training Pipeline

### 5.1 Training Configuration

**Key files:**
- **Hyperparameters**: `code/config/openllama_peft.yaml`
- **DeepSpeed config**: `code/dsconfig/openllama_peft_stage_1.json`
- **Training script**: `code/scripts/train_mvtec.sh`

#### Hyperparameters (from openllama_peft.yaml):

```yaml
# LoRA parameters (trainable)
lora_r: 32                    # LoRA rank
lora_alpha: 32                # LoRA scaling
lora_dropout: 0.1             # Dropout rate

# Training parameters
epochs: 50                    # Number of epochs
max_length: 1024              # Max sequence length
warmup_rate: 0.1              # Warmup ratio
```

#### DeepSpeed Configuration (from openllama_peft_stage_1.json):

```json
{
  "train_batch_size": 16,                    # Global batch size
  "train_micro_batch_size_per_gpu": 1,       # Batch size per GPU
  "gradient_accumulation_steps": 8,          # Accumulation steps
  "learning_rate": 0.001,                    # Learning rate (1e-3)
  "optimizer": "Adam",
  "fp16": {"enabled": true},                 # Mixed precision
  "zero_optimization": {
    "stage": 2,                              # ZeRO stage 2
    "offload_optimizer": {"device": "cpu"}   # Offload to CPU
  }
}
```

**Effective batch size** = `train_micro_batch_size_per_gpu` × `gradient_accumulation_steps` × `num_gpus`
- For 2 GPUs: 1 × 8 × 2 = 16

### 5.2 What is Frozen vs. Trainable

**Frozen (not updated):**
- ImageBind encoder (1024-dim embeddings)
- Vicuna LLM base weights (all transformer layers)

**Trainable (updated during training):**
- **LoRA adapters** in Vicuna LLM:
  - Applied to query, key, value, output projection matrices
  - Only ~1-2% of total parameters
- **Prompt Learner** (PromptLearner class):
  - CNN-based meta-network
  - Base prompts (9 learnable embeddings)
- **Linear projection layers** (LinearLayer class):
  - Maps ImageBind features to LLM input space

**Total trainable parameters**: ~50M out of ~7B (< 1%)

### 5.3 Training Command

#### For Colab (Single GPU):

**Important**: Colab provides 1 GPU, so we need to adjust the DeepSpeed script.

**Step 1: Modify training script**

Create a Colab-friendly version:

```bash
# Navigate to code directory
%cd /content/AnomalyGPT/code

# Create Colab training script
!cat > train_mvtec_colab.sh << 'EOF'
#!/bin/bash

deepspeed --include localhost:0 --master_port 28400 train_mvtec.py \
    --model openllama_peft \
    --stage 1 \
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth \
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/ \
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt \
    --max_tgt_len 1024 \
    --data_path ../data/pandagpt4_visual_instruction_data.json \
    --image_root_path ../data/images/ \
    --save_path ./ckpt/train_mvtec/ \
    --log_path ./ckpt/train_mvtec/log/
EOF

!chmod +x train_mvtec_colab.sh
```

**Step 2: Adjust batch size for single GPU**

Edit `code/dsconfig/openllama_peft_stage_1.json`:

```json
{
  "train_batch_size": 8,                     # Reduce for single GPU
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 8,
  ...
}
```

**Step 3: Run training**

```bash
%cd /content/AnomalyGPT/code
!bash train_mvtec_colab.sh
```

**Alternative (direct Python call):**

```python
%cd /content/AnomalyGPT/code

!deepspeed --include localhost:0 --master_port 28400 train_mvtec.py \
    --model openllama_peft \
    --stage 1 \
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth \
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/ \
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt \
    --max_tgt_len 1024 \
    --data_path ../data/pandagpt4_visual_instruction_data.json \
    --image_root_path ../data/images/ \
    --save_path ./ckpt/train_mvtec/ \
    --log_path ./ckpt/train_mvtec/log/
```

### 5.4 Training Process Explanation

**The training alternates between two datasets:**

1. **MVTec-AD self-supervised data** (anomaly simulation)
   - Loaded by `load_mvtec_dataset()` in `datasets/__init__.py`
   - Uses `patch_ex()` augmentation to create fake anomalies
   - Teaches model to detect spatial anomalies

2. **PandaGPT visual instruction data** (general vision-language)
   - Loaded by `load_sft_dataset()` in `datasets/__init__.py`
   - General image captioning and VQA samples
   - Maintains general vision-language understanding

**Training loop (from train_mvtec.py, line 86):**
```python
for epoch in range(epochs):
    for batch_mvtec, batch_sft in zip(train_iter, train_iter_sft):
        # Step 1: Train on MVTec-AD (anomaly detection)
        agent.train_model(batch_mvtec)
        
        # Step 2: Train on PandaGPT (general vision-language)
        agent.train_model(batch_sft)
```

**Loss function**:
- **Language modeling loss** (cross-entropy on text generation)
- **Anomaly segmentation loss** (Focal Loss + Dice Loss on pixel masks)

### 5.5 Expected Training Time

**On Colab T4 (16GB):**
- **Epoch duration**: ~3-4 hours
- **Full training (50 epochs)**: ~150-200 hours (6-8 days)

**On Colab A100 (40GB):**
- **Epoch duration**: ~1-1.5 hours
- **Full training (50 epochs)**: ~50-75 hours (2-3 days)

**Tips for Colab:**
- Use Colab Pro/Pro+ for longer runtimes and better GPUs
- Save checkpoints frequently (already done after each epoch)
- Resume training if disconnected (load from last checkpoint)

### 5.6 Checkpoint Saving

Checkpoints are saved **after each epoch** to:
```
code/ckpt/train_mvtec/pytorch_model.pt
```

**Checkpoint contains:**
- LoRA adapter weights
- Prompt Learner weights
- Linear projection weights

**To resume training**, the script automatically loads from this path if it exists.

### 5.7 Monitoring Training

**Check training logs:**
```bash
# View latest log
!tail -f /content/AnomalyGPT/code/ckpt/train_mvtec/log/train_*.log
```

**Check GPU usage:**
```python
!nvidia-smi -l 1
```

**Monitor loss in real-time:**
```python
import time
log_file = '/content/AnomalyGPT/code/ckpt/train_mvtec/log/train_*.log'

while True:
    !tail -n 20 {log_file} | grep -E "loss|Loss"
    time.sleep(10)
```

### 5.8 Reducing Training Time (Quick Start)

For **testing/debugging**, reduce epochs:

Edit `code/config/openllama_peft.yaml`:
```yaml
train:
    epochs: 5  # Reduce from 50 to 5
```

Or modify the training script:
```python
# In train_mvtec.py, line 112
args['epochs'] = 5  # Override config
```

**Note**: Performance will be suboptimal with fewer epochs.

---

## 6. Inference and Evaluation

### 6.1 Pre-trained Model Weights

To **skip training** and use pre-trained AnomalyGPT weights:

```bash
!mkdir -p code/ckpt/train_mvtec

# Download MVTec-AD trained weights (~13GB)
!wget -O code/ckpt/train_mvtec/pytorch_model.pt \
    https://huggingface.co/FantasticGNU/AnomalyGPT/resolve/main/train_mvtec/pytorch_model.pt

# Verify
!ls -lh code/ckpt/train_mvtec/
```

**Other available weights:**
- VisA: `https://huggingface.co/FantasticGNU/AnomalyGPT/resolve/main/train_visa/pytorch_model.pt`
- Supervised: `https://huggingface.co/FantasticGNU/AnomalyGPT/resolve/main/train_supervised/pytorch_model.pt`

### 6.2 Inference Script Overview

**Key file**: `code/test_mvtec.py`

**What it does:**
1. Loads model with PandaGPT + AnomalyGPT weights
2. For each category in MVTec-AD:
   - Selects few-shot normal reference images
   - Iterates through all test images (normal + anomalies)
   - Generates text response ("Yes/No") and anomaly map
   - Computes metrics: accuracy, image-AUROC, pixel-AUROC

### 6.3 Running Inference

#### Step 1: Modify Checkpoint Path (if needed)

Edit `code/test_mvtec.py`, line 44:

```python
# Original (for VisA weights):
'anomalygpt_ckpt_path': './ckpt/train_visa/pytorch_model.pt',

# Change to MVTec-AD weights:
'anomalygpt_ckpt_path': './ckpt/train_mvtec/pytorch_model.pt',
```

#### Step 2: Run Inference

```bash
%cd /content/AnomalyGPT/code

# Run with default settings (1-shot, round 3)
!python test_mvtec.py

# Run with 4-shot
!python test_mvtec.py --k_shot 4

# Run without few-shot (zero-shot)
!python test_mvtec.py --few_shot False
```

**Parameters:**
- `--few_shot`: Enable/disable few-shot learning (default: True)
- `--k_shot`: Number of normal reference images (default: 1)
- `--round`: Random seed for reference selection (default: 3)

#### Step 3: Interpret Results

**Output format:**
```
bottle right: 20 wrong: 63
bottle i_AUROC: 95.23
bottle p_AUROC: 97.45

cable right: 35 wrong: 115
cable i_AUROC: 87.65
cable p_AUROC: 94.32

...

i_AUROC: 91.34  # Average image-level AUROC across all categories
p_AUROC: 95.12  # Average pixel-level AUROC across all categories
precision: 52.3  # Classification accuracy (Yes/No)
```

**Metrics:**
- **Image-level AUROC (i_AUROC)**: Area under ROC curve for anomaly/normal classification
  - Uses max anomaly score as image score
- **Pixel-level AUROC (p_AUROC)**: Area under ROC curve for anomaly localization
  - Uses pixel-wise anomaly map vs. ground truth masks
- **Precision (Accuracy)**: % of correct "Yes"/"No" responses

### 6.4 How Anomaly Localization Works

**Feature-matching approach (in `OpenLLAMAPEFTModel.generate()`):**

1. **Extract features** from test image via ImageBind encoder
2. **Extract features** from K normal reference images
3. **Compute similarity** between test features and normal features at each spatial location
4. **Anomaly score** = 1 - max(similarity)
   - High similarity to normals → low anomaly score
   - Low similarity to normals → high anomaly score
5. **Generate anomaly map**: 224×224 heatmap
6. **Threshold** to determine "Yes"/"No" response

### 6.5 Visualizing Anomaly Maps

The current `test_mvtec.py` doesn't save visualizations. To add:

```python
# Add after line 145 in test_mvtec.py

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Visualize some examples
if i_label[0] == 1:  # If anomaly
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    img = Image.open(file_path)
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(img_mask, cmap='Reds', alpha=0.6)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Predicted anomaly map
    axes[2].imshow(anomaly_map, cmap='jet')
    axes[2].set_title(f'Predicted (Response: {resp})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'results/{c_name}_{file_path.split("/")[-1]}')
    plt.close()
```

### 6.6 Running on Custom Images

To test on your own images:

```python
import torch
from model.openllama import OpenLLAMAPEFTModel
from PIL import Image

# Initialize model
args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
    'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/7b_v0',
    'anomalygpt_ckpt_path': './ckpt/train_mvtec/pytorch_model.pt',
    'delta_ckpt_path': '../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt',
    'stage': 2,
    'max_tgt_len': 128,
    'lora_r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
}

model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['delta_ckpt_path'], map_location='cpu')
model.load_state_dict(delta_ckpt, strict=False)
delta_ckpt = torch.load(args['anomalygpt_ckpt_path'], map_location='cpu')
model.load_state_dict(delta_ckpt, strict=False)
model = model.eval().half().cuda()

# Run inference
test_image = '/path/to/test/image.png'
normal_refs = ['/path/to/normal1.png', '/path/to/normal2.png']  # Few-shot references

prompt = "This is a photo of [object description] for anomaly detection. Is there any anomaly in the image?"

response, anomaly_map = model.generate({
    'prompt': prompt,
    'image_paths': [test_image],
    'normal_img_paths': normal_refs,
    'audio_paths': [],
    'video_paths': [],
    'thermal_paths': [],
    'top_p': 0.1,
    'temperature': 1.0,
    'max_tgt_len': 512,
    'modality_embeds': []
})

print(f"Response: {response}")
print(f"Anomaly map shape: {anomaly_map.shape}")
```

---

## 7. Common Issues and Solutions

### 7.1 Dependency Conflicts

**Issue**: `ERROR: Package X has requirement Y, but you have Z`

**Solution**:
```bash
# Force install specific versions
!pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --force-reinstall --no-deps
!pip install transformers==4.29.1 --force-reinstall

# Restart runtime after installation
import IPython
IPython.Application.instance().kernel.do_shutdown(True)
```

### 7.2 CUDA Out of Memory

**Issue**: `RuntimeError: CUDA out of memory`

**Solutions**:

1. **Reduce batch size**:
   ```json
   // In dsconfig/openllama_peft_stage_1.json
   "train_micro_batch_size_per_gpu": 1,  // Already minimum
   "gradient_accumulation_steps": 4,     // Reduce from 8
   ```

2. **Enable CPU offloading** (already enabled in config):
   ```json
   "zero_optimization": {
       "stage": 2,
       "offload_optimizer": {"device": "cpu"}
   }
   ```

3. **Use gradient checkpointing**:
   ```json
   "activation_checkpointing": {
       "partition_activations": true,
       "cpu_checkpointing": true
   }
   ```

4. **Reduce sequence length**:
   ```bash
   # In training command
   --max_tgt_len 512  # Reduce from 1024
   ```

5. **Upgrade Colab tier**:
   - Colab Pro: More memory, longer runtime
   - Colab Pro+: A100 GPU with 40GB VRAM

### 7.3 Missing Checkpoint Error

**Issue**: `FileNotFoundError: [Errno 2] No such file or directory: '../pretrained_ckpt/vicuna_ckpt/7b_v0/config.json'`

**Solution**:
```bash
# Verify all checkpoints exist
!ls -lh pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth
!ls -lh pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt
!ls pretrained_ckpt/vicuna_ckpt/7b_v0/

# If missing, re-download following Section 3
```

### 7.4 Dataset Path Error

**Issue**: `FileNotFoundError: ../data/mvtec_anomaly_detection`

**Solution**:
```bash
# Check dataset location
!ls -d data/mvtec_anomaly_detection

# If in different location, edit code/datasets/__init__.py line 52
# Or create symlink:
!ln -s /path/to/actual/location /content/AnomalyGPT/data/mvtec_anomaly_detection
```

### 7.5 DeepSpeed Launch Issues

**Issue**: `ModuleNotFoundError: No module named 'deepspeed'`

**Solution**:
```bash
!pip install deepspeed==0.9.2
```

**Issue**: `RuntimeError: NCCL error`

**Solution**:
```bash
# For single GPU, ensure correct device specification
deepspeed --include localhost:0 ...
```

### 7.6 PandaGPT Data Missing

**Issue**: Training fails with "Cannot find pandagpt4_visual_instruction_data.json"

**Solution**:
```bash
# The PandaGPT data is large (~50GB images)
# For testing, you can skip it by modifying train_mvtec.py:

# Comment out line 73:
# train_data_sft, train_iter_sft, sampler = load_sft_dataset(args)

# Comment out lines 95-100 (SFT training step)
```

**Alternative**: Use only MVTec-AD training:
```python
# In train_mvtec.py, modify training loop (line 86):
for epoch_i in range(args['epochs']):
    for batch in train_iter:  # Remove zip with train_iter_sft
        agent.train_model(batch, current_step=current_step, pbar=pbar)
        current_step += 1
```

### 7.7 Slow Training Speed

**Issue**: Training is very slow (< 1 iteration/minute)

**Solutions**:

1. **Check dataloader workers**:
   ```python
   # In datasets/__init__.py, line 68
   num_workers=8,  # Reduce to 2-4 if CPU bottleneck
   ```

2. **Disable checkpointing for intermediate steps**:
   ```python
   # In train_mvtec.py, already commented out (line 103)
   ```

3. **Profile bottlenecks**:
   ```python
   import torch.profiler
   
   with torch.profiler.profile() as prof:
       # Run one training iteration
       agent.train_model(batch)
   
   print(prof.key_averages().table())
   ```

### 7.8 Test Accuracy is Low

**Issue**: Test accuracy much lower than reported in paper

**Possible causes & solutions**:

1. **Using wrong checkpoint**:
   ```python
   # Ensure using MVTec-trained weights, not VisA weights
   'anomalygpt_ckpt_path': './ckpt/train_mvtec/pytorch_model.pt'
   ```

2. **Not using few-shot**:
   ```bash
   # Enable few-shot with adequate K
   !python test_mvtec.py --few_shot True --k_shot 4
   ```

3. **Insufficient training**:
   - Full training requires 50 epochs (~150-200 hours on T4)
   - Use pre-trained weights from HuggingFace

4. **Object descriptions mismatch**:
   - Check `describles` dict in test_mvtec.py (line 20-35)
   - Ensure descriptions match training

### 7.9 Colab Disconnection During Training

**Issue**: Colab disconnects after 12 hours (or 24h with Pro)

**Solutions**:

1. **Save checkpoints frequently** (already done after each epoch)

2. **Resume from checkpoint**:
   ```python
   # Training script automatically loads existing checkpoint
   # Just re-run the training command
   ```

3. **Use Google Drive for persistence**:
   ```python
   # Mount Drive at start
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Save checkpoints to Drive
   --save_path /content/drive/MyDrive/AnomalyGPT/ckpt/train_mvtec/
   ```

4. **Keep Colab alive** (use with caution, may violate ToS):
   ```javascript
   // In browser console (F12)
   function ClickConnect(){
       console.log("Clicking connect");
       document.querySelector("colab-connect-button").click()
   }
   setInterval(ClickConnect, 60000);
   ```

### 7.10 Vicuna Checkpoint Issues

**Issue**: Cannot obtain LLaMA weights from Meta

**Alternative solution**:

1. **Use community-converted weights** (if available and licensed):
   - Check HuggingFace for pre-combined Vicuna weights
   - Example: Some repositories provide full Vicuna v0 without needing LLaMA

2. **Use LLaMA 2** (with code modifications):
   - LLaMA 2 is publicly available without approval
   - May require changing model architecture slightly

3. **Contact authors** for pre-trained AnomalyGPT weights with bundled LLM

---

## 8. Web Demo

### 8.1 Running the Gradio Demo

After training (or with pre-trained weights):

```bash
%cd /content/AnomalyGPT/code

# Run web demo
!python web_demo.py
```

**Note**: In Colab, you'll need to expose the port:

```python
# In web_demo.py, change the last line:
# demo.launch(share=False)
# to:
demo.launch(share=True)  # This creates a public URL
```

### 8.2 Accessing the Demo

After running, you'll see:
```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxx.gradio.live
```

Click the public URL to access the demo from anywhere.

### 8.3 Demo Features

- Upload test image
- Optionally upload 1-4 normal reference images (few-shot)
- Enter text description of the object/texture
- Ask: "Is there any anomaly in the image?"
- View text response + anomaly heatmap overlay

---

## 9. Advanced Topics

### 9.1 Training on Custom Dataset

To train on your own anomaly detection dataset:

1. **Organize data** in MVTec-AD format:
   ```
   data/my_custom_dataset/
   ├── category1/
   │   ├── train/good/
   │   ├── test/good/
   │   ├── test/defect_type1/
   │   └── ground_truth/defect_type1/
   └── category2/...
   ```

2. **Create dataset loader** (copy from `datasets/mvtec.py`):
   ```python
   class CustomDataset(Dataset):
       def __init__(self, root_dir):
           # Same structure as MVtecDataset
           pass
   ```

3. **Add descriptions**:
   ```python
   describles['category1'] = "This is a photo of category1 for anomaly detection..."
   ```

4. **Update dataset loading** in `datasets/__init__.py`:
   ```python
   data = CustomDataset('../data/my_custom_dataset')
   ```

5. **Run training**:
   ```bash
   !bash train_mvtec.sh  # Same command
   ```

### 9.2 Multi-GPU Training

If you have access to multi-GPU setup:

```bash
# Original script uses 2 GPUs (GPUs 0 and 1)
deepspeed --include localhost:0,1 --master_port 28400 train_mvtec.py ...

# For 4 GPUs:
deepspeed --include localhost:0,1,2,3 --master_port 28400 train_mvtec.py ...
```

**Adjust batch size accordingly** in DeepSpeed config:
```json
{
  "train_batch_size": 32,  // Scale with number of GPUs
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 8
}
```

### 9.3 Using 13B Vicuna

For better performance:

1. **Download 13B checkpoints**:
   ```bash
   # PandaGPT 13B
   !wget -O pretrained_ckpt/pandagpt_ckpt/13b/pytorch_model.pt \
       https://huggingface.co/openllmplayground/pandagpt_13b_max_len_400/resolve/main/pytorch_model.pt
   
   # Vicuna 13B (follow same steps as 7B but with 13B weights)
   ```

2. **Update paths in training script**:
   ```bash
   --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/13b_v0/ \
   --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/13b/pytorch_model.pt \
   ```

**Note**: 13B requires significantly more VRAM (~40GB for inference).

---

## 10. Expected Results

### 10.1 Performance Benchmarks (from paper)

**MVTec-AD (unsupervised, 1-shot):**
- Image-level AUROC: 86.0%
- Pixel-level AUROC: 94.1%
- Classification accuracy: ~85%

**Per-category results (approximate):**

| Category    | Image AUROC | Pixel AUROC |
|-------------|-------------|-------------|
| Bottle      | 99.5        | 97.8        |
| Cable       | 88.7        | 92.5        |
| Capsule     | 94.2        | 96.9        |
| Carpet      | 97.1        | 98.4        |
| Grid        | 99.8        | 99.2        |
| Hazelnut    | 98.7        | 97.5        |
| Leather     | 99.9        | 99.1        |
| Metal Nut   | 98.2        | 96.8        |
| Pill        | 94.5        | 95.7        |
| Screw       | 65.4        | 97.2        |
| Tile        | 99.6        | 96.8        |
| Toothbrush  | 100.0       | 98.9        |
| Transistor  | 94.8        | 86.7        |
| Wood        | 99.1        | 94.6        |
| Zipper      | 97.2        | 97.8        |

### 10.2 Comparison with Other Methods

AnomalyGPT advantages:
- No manual threshold tuning required
- Provides natural language explanations
- Supports few-shot learning
- Can describe anomaly location ("at the top left of the image")

---

## 11. Citation

If you use this code or guide, please cite:

```bibtex
@article{gu2023anomalygpt,
  title={AnomalyGPT: Detecting Industrial Anomalies using Large Vision-Language Models},
  author={Gu, Zhaopeng and Zhu, Bingke and Zhu, Guibo and Chen, Yingying and Tang, Ming and Wang, Jinqiao},
  journal={arXiv preprint arXiv:2308.15366},
  year={2023}
}
```

---

## 12. Additional Resources

- **Paper**: https://arxiv.org/abs/2308.15366
- **Project Page**: https://anomalygpt.github.io
- **Official Repo**: https://github.com/CASIA-IVA-Lab/AnomalyGPT
- **HuggingFace Demo**: https://huggingface.co/spaces/FantasticGNU/AnomalyGPT
- **Model Weights**: https://huggingface.co/FantasticGNU/AnomalyGPT

- **ImageBind**: https://github.com/facebookresearch/ImageBind
- **PandaGPT**: https://github.com/yxuansu/PandaGPT
- **Vicuna**: https://github.com/lm-sys/FastChat
- **MVTec-AD**: https://www.mvtec.com/company/research/datasets/mvtec-ad

---

## Quick Start Checklist

- [ ] Install dependencies (Section 2.3)
- [ ] Download ImageBind checkpoint (Section 3.1)
- [ ] Download PandaGPT checkpoint (Section 3.3)
- [ ] Obtain Vicuna checkpoint (Section 3.2)
- [ ] Download MVTec-AD dataset (Section 4.1)
- [ ] Verify dataset structure (Section 4.4)
- [ ] Download PandaGPT training data (Section 3.4)
- [ ] Modify training script for Colab (Section 5.3)
- [ ] Start training OR download pre-trained weights (Section 5.3 / 6.1)
- [ ] Run inference and evaluation (Section 6.3)
- [ ] (Optional) Launch web demo (Section 8.1)

---

**Last Updated**: 2026-01-02

**Maintained by**: Community contribution for AnomalyGPT

For issues or questions, please open an issue in the official repository.

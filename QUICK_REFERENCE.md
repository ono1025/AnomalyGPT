# AnomalyGPT Quick Reference Guide

## Quick Links
- [Full Setup Guide](./COLAB_SETUP_GUIDE.md)
- [Colab Notebook](./colab_setup.ipynb)
- [Official Repo](https://github.com/CASIA-IVA-Lab/AnomalyGPT)
- [Paper](https://arxiv.org/abs/2308.15366)

## Essential Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download checkpoints
bash scripts/download_checkpoints.sh
```

### Training
```bash
cd code/
bash scripts/train_mvtec.sh
```

### Inference
```bash
cd code/
python test_mvtec.py --few_shot True --k_shot 1
```

### Demo
```bash
cd code/
python web_demo.py
```

## File Structure Quick Reference

```
AnomalyGPT/
├── code/
│   ├── train_mvtec.py          # Train on MVTec-AD
│   ├── test_mvtec.py           # Evaluate on MVTec-AD
│   ├── web_demo.py             # Gradio web interface
│   ├── model/
│   │   ├── openllama.py        # Main model (OpenLLAMAPEFTModel)
│   │   ├── AnomalyGPT_models.py # PromptLearner, LinearLayer
│   │   └── ImageBind/          # Vision encoder
│   ├── datasets/
│   │   ├── mvtec.py            # MVTec-AD dataset loader
│   │   └── sft_dataset.py      # PandaGPT dataset loader
│   ├── config/
│   │   └── openllama_peft.yaml # Training hyperparameters
│   └── dsconfig/
│       └── openllama_peft_stage_1.json # DeepSpeed config
├── data/
│   └── mvtec_anomaly_detection/
│       ├── bottle/
│       │   ├── train/good/
│       │   ├── test/{defect_types}/
│       │   └── ground_truth/{defect_types}/
│       └── [14 more categories]
├── pretrained_ckpt/
│   ├── imagebind_ckpt/
│   │   └── imagebind_huge.pth
│   ├── vicuna_ckpt/7b_v0/
│   └── pandagpt_ckpt/7b/
│       └── pytorch_model.pt
└── COLAB_SETUP_GUIDE.md
```

## Key Hyperparameters

### Training (openllama_peft.yaml)
- **epochs**: 50
- **max_length**: 1024
- **lora_r**: 32
- **lora_alpha**: 32
- **lora_dropout**: 0.1

### DeepSpeed (openllama_peft_stage_1.json)
- **train_batch_size**: 16
- **train_micro_batch_size_per_gpu**: 1
- **gradient_accumulation_steps**: 8
- **learning_rate**: 0.001

## Model Architecture

1. **ImageBind** (Frozen)
   - Input: 224×224 RGB image
   - Output: 1024-dim embeddings

2. **Prompt Learner** (Trainable)
   - Input: Image features
   - Output: 18 prompt embeddings (9 base + 9 image-specific)

3. **Vicuna LLM** (LoRA fine-tuned)
   - Base: LLaMA 7B/13B
   - LoRA rank: 32
   - Trainable params: ~50M (~1% of total)

4. **Anomaly Decoder** (Feature matching)
   - Compares test image with normal references
   - Outputs: 224×224 anomaly heatmap

## Data Flow

### Training
```
Normal Image
  → Self-supervised Augmentation (patch_ex)
  → ImageBind Encoder
  → Prompt Learner
  → Vicuna LLM (LoRA)
  → Text: "Yes, anomaly at [position]" + Anomaly Map
```

### Inference
```
Test Image + K Normal References
  → ImageBind Encoder
  → Feature Matching
  → Anomaly Map (224×224)
  → Vicuna LLM
  → Text: "Yes/No" + Localization
```

## MVTec-AD Dataset Structure

**15 Categories**: bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper

**Per Category**:
```
{category}/
├── train/
│   └── good/              # Normal images (50-400 images)
├── test/
│   ├── good/              # Normal test images
│   └── {defect_type}/     # Anomaly images (3-8 defect types)
└── ground_truth/
    └── {defect_type}/     # Binary masks: {img}_mask.png
```

## Checkpoint Requirements

| Checkpoint | Size | Source | Path |
|------------|------|--------|------|
| ImageBind | 4.9GB | [Meta](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth) | `pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth` |
| PandaGPT 7B | 13GB | [HF](https://huggingface.co/openllmplayground/pandagpt_7b_max_len_1024) | `pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt` |
| Vicuna 7B v0 | 13GB | [Combine LLaMA + Delta](https://github.com/lm-sys/FastChat) | `pretrained_ckpt/vicuna_ckpt/7b_v0/` |
| AnomalyGPT (MVTec) | 13GB | [HF](https://huggingface.co/FantasticGNU/AnomalyGPT) | `code/ckpt/train_mvtec/pytorch_model.pt` |

**Total Storage**: ~45GB (without training data)

## Performance Benchmarks

### MVTec-AD (1-shot, unsupervised)
- **Image AUROC**: 86.0%
- **Pixel AUROC**: 94.1%
- **Accuracy**: ~85%

### Top-performing categories:
- Toothbrush: 100.0% (Image AUROC)
- Leather: 99.9%
- Grid: 99.8%
- Tile: 99.6%
- Bottle: 99.5%

### Challenging categories:
- Screw: 65.4%
- Transistor: 94.8%

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce batch size, use gradient checkpointing |
| Missing checkpoint | Verify all 3 checkpoints downloaded correctly |
| Dataset not found | Check path in `datasets/__init__.py` line 52 |
| Slow training | Reduce num_workers, use A100 GPU |
| DeepSpeed error | Ensure single GPU: `--include localhost:0` |
| Vicuna setup | Follow [Section 3.2](./COLAB_SETUP_GUIDE.md#32-vicuna-checkpoint) carefully |

## Training Time Estimates

| GPU | Batch Size | Time per Epoch | Full Training (50 epochs) |
|-----|------------|----------------|---------------------------|
| T4 (16GB) | 1 | 3-4 hours | 150-200 hours |
| V100 (32GB) | 1-2 | 2-3 hours | 100-150 hours |
| A100 (40GB) | 2-4 | 1-1.5 hours | 50-75 hours |

## Memory Requirements

### Training
- Minimum: 16GB GPU (T4)
- Recommended: 32GB+ GPU (V100, A100)
- System RAM: 12GB+

### Inference
- Minimum: 12GB GPU (T4)
- Recommended: 16GB+ GPU
- System RAM: 8GB+

## API Usage Example

```python
from model.openllama import OpenLLAMAPEFTModel
import torch

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
# Load checkpoints...
model = model.eval().half().cuda()

# Run inference
response, anomaly_map = model.generate({
    'prompt': 'This is a photo of a bottle. Is there any anomaly?',
    'image_paths': ['/path/to/test_image.png'],
    'normal_img_paths': ['/path/to/normal1.png', '/path/to/normal2.png'],
    'audio_paths': [],
    'video_paths': [],
    'thermal_paths': [],
    'top_p': 0.1,
    'temperature': 1.0,
    'max_tgt_len': 512,
    'modality_embeds': []
})

print(f"Response: {response}")
print(f"Anomaly map shape: {anomaly_map.shape}")  # (224, 224)
```

## Configuration Files

### Modify Training Epochs
Edit `code/config/openllama_peft.yaml`:
```yaml
train:
    epochs: 5  # Reduce from 50 for testing
```

### Modify Learning Rate
Edit `code/dsconfig/openllama_peft_stage_1.json`:
```json
{
  "optimizer": {
    "params": {
      "lr": 0.001  # Adjust as needed
    }
  }
}
```

### Modify Batch Size
Edit `code/dsconfig/openllama_peft_stage_1.json`:
```json
{
  "train_batch_size": 8,  # Reduce for single GPU
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 8
}
```

## Important Code Locations

| Function | File | Line | Description |
|----------|------|------|-------------|
| MVTec Dataset | `datasets/mvtec.py` | 58-195 | Dataset class with self-supervised augmentation |
| Model Definition | `model/openllama.py` | ~100-500 | OpenLLAMAPEFTModel class |
| Training Loop | `train_mvtec.py` | 84-107 | Main training iteration |
| Inference | `test_mvtec.py` | 66-100 | Prediction function |
| Prompt Learner | `model/AnomalyGPT_models.py` | 32-73 | PromptLearner class |
| Anomaly Decoder | `model/openllama.py` | ~300-400 | Feature matching in generate() |

## Useful Debugging Commands

```bash
# Check GPU usage
nvidia-smi -l 1

# Monitor training logs
tail -f code/ckpt/train_mvtec/log/*.log

# Check dataset loading
python -c "from datasets import load_mvtec_dataset; print('OK')"

# Test model loading
python -c "from model.openllama import OpenLLAMAPEFTModel; print('OK')"

# Verify checkpoint structure
tree -L 3 pretrained_ckpt/

# Count dataset samples
find data/mvtec_anomaly_detection -name "*.png" | wc -l
```

## Citation

```bibtex
@article{gu2023anomalygpt,
  title={AnomalyGPT: Detecting Industrial Anomalies using Large Vision-Language Models},
  author={Gu, Zhaopeng and Zhu, Bingke and Zhu, Guibo and Chen, Yingying and Tang, Ming and Wang, Jinqiao},
  journal={arXiv preprint arXiv:2308.15366},
  year={2023}
}
```

## Support

- **Issues**: https://github.com/CASIA-IVA-Lab/AnomalyGPT/issues
- **Full Guide**: [COLAB_SETUP_GUIDE.md](./COLAB_SETUP_GUIDE.md)
- **Paper**: https://arxiv.org/abs/2308.15366
- **Demo**: https://huggingface.co/spaces/FantasticGNU/AnomalyGPT

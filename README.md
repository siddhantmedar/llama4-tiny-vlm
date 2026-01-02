# LLaMA 4 From Scratch

Building a multimodal LLM from scratch, implementing key LLaMA 4 architectural innovations.

## Overview

This project implements a vision-language model in three phases:

1. **Text Pretraining** - Train a decoder-only LLM with modern architecture
2. **Vision-Language Alignment** - Align a pretrained ViT encoder with the LLM
3. **Visual Instruction Tuning** - Fine-tune for instruction following (coming soon)

## Architecture

### LLM Architecture (~380M params)

| Component | Value |
|-----------|-------|
| Layers | 12 |
| Hidden Dim | 768 |
| Attention Heads | 12 |
| KV Heads | 4 (GQA) |
| Vocab Size | 32,000 |
| Max Seq Len | 1,024 |

**Key Features:**
- **Grouped Query Attention (GQA)** - 4 KV heads shared across 12 query heads
- **Mixture of Experts (MoE)** - 8 experts, top-2 routing, every 2nd layer
- **iRoPE** - Interleaved RoPE (75% RoPE, 25% NoPE layers)
- **SwiGLU** - Gated activation in FFN layers
- **RMSNorm** - Pre-normalization

### Vision-Language Architecture (~470M total params)

```
Image [B, 3, 224, 224]
    │
    ▼
┌─────────────────────────────┐
│  ViT-B/16 Encoder (frozen)  │  90M params
│  Pretrained on ImageNet-1K  │
└─────────────────────────────┘
    │
    ▼ [B, 196, 768]
┌─────────────────────────────┐
│  MLP Projector (trainable)  │  4.7M params
│  768 → 3072 → 768           │
└─────────────────────────────┘
    │
    ▼ [B, 196, 768]
    │
    ├──────────────────┐
    │                  │
    ▼                  ▼
[Vision Tokens]   [Text Tokens]
    │                  │
    └────────┬─────────┘
             │
             ▼
┌─────────────────────────────┐
│  LLM Decoder (frozen)       │  380M params
│  12 layers, GQA, MoE        │
└─────────────────────────────┘
             │
             ▼
        [Logits]
```

---

## Phase 1: Text Pretraining

### Dataset
- **Source**: HuggingFace `smollm-corpus/cosmopedia-v2`
- **Format**: Streaming for memory efficiency
- **Tokenizer**: Custom BPE (32,000 vocab)

### Training Config
```toml
batch_size = 7
max_seq_len = 1024
learning_rate = 3e-4
batches_per_epoch = 20000
```

### Results

```
Training on device: cuda (RTX 3090)
Trainable params: 379,547,916

Epoch 1 Training (20k batches):
Step 1000  | loss: 6.23 | lr: 0.000150
Step 5000  | loss: 4.18 | lr: 0.000285
Step 10000 | loss: 3.57 | lr: 0.000298
Step 15000 | loss: 3.12 | lr: 0.000250
Step 20000 | loss: 2.85 | lr: 0.000180

Final Results:
├── Train Loss: ~2.85
├── Val Loss: ~2.90
└── Perplexity: ~18
```

### Sample Generations (Actual Model Output)

**Prompt:** `"The capital of France is"`
```
The capital of France is the city of Paris, France, known for its stunning
architecture, rich culture, and delicious food. But did you know that there
are also some amazing places in France? Today, we will explore one such place
called the "Les Rivier de L'Ouverture," which is...
```

**Prompt:** `"Machine learning is"`
```
Machine learning is a powerful tool for enhancing the performance of machine
learning models. By combining the strengths of both approaches, we can create
more robust and adaptive machine learning models that can accurately predict
and optimize their performance...
```

**Prompt:** `"def fibonacci(n):"`
```
def fibonacci(n):
    """
    This function takes a single number as input and returns a single number.
    This function can be used to perform various operations, such as addition,
    subtraction, multiplication, and division...
```

**Prompt:** `"The solar system"`
```
The solar system, it is crucial to understand the role of the solar system in
regulating the planet's climate. The solar system is a complex system that
governs the distribution of energy and the distribution of it across space
and time...
```

---

## Phase 2: Vision-Language Alignment

### Dataset
- **Source**: COCO Captions (`jxie/coco_captions`)
- **Train**: 567k image-caption pairs
- **Val**: 25k pairs

### Training Config
```toml
vlm_batch_size = 48
vlm_epochs = 3
vlm_learning_rate = 2e-4
vlm_max_seq_len = 64
```

### Training Approach
- **Frozen**: ViT encoder (90M) + LLM decoder (380M)
- **Trainable**: MLP projector only (4.7M params, 1%)
- **Objective**: Next-token prediction on captions

### Results

```
Training on device: cuda (RTX 3090)
Trainable params: 4,722,432 / 470,069,772 (1.0%)
VRAM Usage: ~22GB
Batch size: 48

Epoch 1/3 | train_loss: 4.98 | val_loss: 4.21
Epoch 2/3 | train_loss: 3.57 | val_loss: 3.49
Epoch 3/3 | train_loss: 3.12 | val_loss: 3.23

Best Val Loss: 3.227
Training Time: ~1.5 hours
```

### Sample Captions (Actual Model Output)

| Image | Generated Caption |
|-------|------------------|
| 🐕 Dog | "A dog is sitting on the beach with a nose." |
| 🐈 Cat | "A cat laying on top of a table." |
| 🍲 Food | "A bowl of food is placed on a table." |
| 🏙️ City | "A busy city with lots of buildings and people." |
| 🏖️ Beach | "A beach with a water of the ocean on the beach." |
| 🏔️ Mountain | "A person sitting on a snow-capped mountain landscape." |

### Inference

```python
from model import create_vlm
from dataset import get_tokenizer, get_image_transform

# Load model
model = create_vlm(config, llm_checkpoint)
model.load_state_dict(torch.load("checkpoints/best_vlm.pt"))

# Generate caption
image = transform(Image.open("photo.jpg")).unsqueeze(0).cuda()
input_ids = torch.tensor([[IMAGE_TOKEN]]).cuda()

for _ in range(50):
    logits = model(image, input_ids)
    next_token = logits[0, -1].argmax()
    if next_token == EOS_TOKEN:
        break
    input_ids = torch.cat([input_ids, next_token.view(1,1)], dim=1)

caption = tokenizer.decode(input_ids[0, 1:].tolist())
```

---

## Project Structure

```
llama4-from-scratch/
├── config.toml                 # Model & training configuration
│
├── text_pretraining/           # Phase 1: LLM Pretraining
│   ├── model.py               # Llama model (GQA, MoE, iRoPE)
│   ├── dataset.py             # Streaming text dataset
│   ├── run.py                 # Training script
│   ├── tokenizer/             # BPE tokenizer
│   └── notebooks/             # Development notebooks
│
└── vision_language_alignment/  # Phase 2: VL Alignment
    ├── model.py               # VisionEncoder + VLM
    ├── dataset.py             # COCO captions dataset
    ├── run.py                 # Training script
    ├── inference.py           # Caption generation
    └── notebooks/
        └── inference.ipynb    # Interactive testing
```

---

## Quick Start

### Install Dependencies
```bash
uv sync
```

### Phase 1: Text Pretraining
```bash
cd text_pretraining
uv run python run.py --train
```

### Phase 2: Vision-Language Alignment
```bash
cd vision_language_alignment
uv run python run.py --train
```

### Inference
```bash
cd vision_language_alignment
uv run python inference.py your_image.jpg --greedy
```

---

## Hardware Requirements

Tested on NVIDIA RTX 3090 (24GB).

---

## License

MIT

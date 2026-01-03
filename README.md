# LLaMA4-Tiny-VLM

A **470M parameter Vision-Language Model** built entirely from scratch, implementing LLaMA 4 architecture innovations.

[![Model on HF](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/medarsiddhant/llama4-tiny-vlm)

## Overview

This project implements a complete VLM training pipeline in four phases:

1. **Text Pretraining** - Train a decoder-only LLM with modern architecture
2. **Vision-Language Alignment** - Align a pretrained ViT encoder with the LLM
3. **Visual Instruction Tuning** - Fine-tune for instruction following
4. **DPO Preference Tuning** - Align with human preferences

## Architecture

| Component | Details |
|-----------|---------|
| **LLM** | 380M params, 12 layers, 768 hidden dim |
| **Vision Encoder** | ViT-B/16 (86M params, frozen) |
| **Projector** | 2-layer MLP (4.7M params) |
| **Total** | ~470M parameters |

**Key Innovations:**
- **Grouped Query Attention (GQA)**: 12 query heads, 4 KV heads (3x memory savings)
- **iRoPE**: Interleaved RoPE/NoPE layers (3:1 pattern) with chunked attention
- **Mixture of Experts**: 8 experts, top-2 routing, shared expert
- **SwiGLU**: Gated activation in all FFN blocks

### Training Phases

1. **Phase 1 - Text Pretraining**: Train the LLM from scratch on text data (cosmopedia-v2). All 380M parameters trained.

2. **Phase 2 - Vision-Language Alignment**: Attach a frozen ViT-B/16 encoder and train only a 2-layer MLP projector (4.7M params) to align vision embeddings with the frozen LLM. Uses COCO captions.

3. **Phase 3 - Visual Instruction Tuning**: Unfreeze the LLM and train both projector + LLM (385M params) on instruction-following conversations (LLaVA-Instruct). ViT stays frozen.

4. **Phase 4 - DPO Preference Tuning**: Fine-tune the model using Direct Preference Optimization on chosen/rejected response pairs (RLAIF-V) to improve response quality.

---

## Training Pipeline

| Phase | Dataset | Trainable | Result |
|-------|---------|-----------|--------|
| 1. Text Pretraining | [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) (cosmopedia-v2) | 380M (100%) | LLM base |
| 2. VL Alignment | [jxie/coco_captions](https://huggingface.co/datasets/jxie/coco_captions) (567K) | 4.7M (1%) | val_loss: 3.23 |
| 3. Instruction Tuning | [liuhaotian/LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) (142K COCO) | 385M (82%) | val_loss: 1.69 |
| 4. DPO Tuning | [HuggingFaceH4/rlaif-v_formatted](https://huggingface.co/datasets/HuggingFaceH4/rlaif-v_formatted) (79K) | 385M (82%) | val_acc: 64.5% |

---

## Quick Start

### Install Dependencies
```bash
pip install torch torchvision huggingface_hub tokenizers
```

### Download & Run
```python
import torch
import tomllib
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms
from tokenizers import Tokenizer

# Download checkpoints (cached automatically)
llm_ckpt = hf_hub_download("medarsiddhant/llama4-tiny-vlm", "checkpoints/text_pretraining_best.pt")
vlm_ckpt = hf_hub_download("medarsiddhant/llama4-tiny-vlm", "checkpoints/vision_alignment_best.pt")
dpo_ckpt = hf_hub_download("medarsiddhant/llama4-tiny-vlm", "checkpoints/dpo_best.pt")

# Load model
from visual_instruction_tuning.model import create_instruct_vlm

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

model = create_instruct_vlm(config, llm_ckpt, vlm_ckpt)
ckpt = torch.load(dpo_ckpt, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to("cuda").eval()

# Load tokenizer
tokenizer = Tokenizer.from_file("visual_instruction_tuning/bpe_tokenizer_with_image_tag.json")
IMAGE_TOKEN_ID = tokenizer.token_to_id("<image>")
EOS_TOKEN_ID = tokenizer.token_to_id("</s>")

# Image transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Inference function
@torch.no_grad()
def ask(image_path, question, max_tokens=100):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to("cuda")

    prompt = f" USER: {question} ASSISTANT:"
    input_ids = [IMAGE_TOKEN_ID] + tokenizer.encode(prompt).ids
    input_ids = torch.tensor([input_ids], device="cuda")

    generated = []
    for _ in range(max_tokens):
        logits = model(image_tensor, input_ids)
        next_token = logits[0, -1, :].argmax().item()
        if next_token == EOS_TOKEN_ID:
            break
        generated.append(next_token)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device="cuda")], dim=1)

    return tokenizer.decode(generated)

# Example usage
response = ask("path/to/image.jpg", "What is in this image?")
print(response)
```

---

## Training from Scratch

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

### Phase 3: Visual Instruction Tuning
```bash
cd visual_instruction_tuning
uv run python run.py --train
```

### Phase 4: DPO Preference Tuning
```bash
cd preference_tuning
uv run python run.py --train
```

---

## Project Structure

```
llama4-tiny-vlm/
├── config.toml                     # Model & training configuration
│
├── text_pretraining/               # Phase 1: LLM Pretraining
│   ├── model.py                    # LLaMA model (GQA, MoE, iRoPE)
│   ├── dataset.py                  # Streaming text dataset
│   ├── run.py                      # Training script
│   └── notebooks/
│
├── vision_language_alignment/      # Phase 2: VL Alignment
│   ├── model.py                    # VisionEncoder + VLM
│   ├── dataset.py                  # COCO captions dataset
│   ├── run.py                      # Training script
│   └── notebooks/
│
├── visual_instruction_tuning/      # Phase 3: Instruction Tuning
│   ├── model.py                    # InstructVLM
│   ├── instruction_dataset.py      # LLaVA-Instruct dataset
│   ├── run.py                      # Training script
│   └── notebooks/
│
└── preference_tuning/              # Phase 4: DPO
    ├── dataset.py                  # RLAIF-V dataset
    ├── run.py                      # DPO training script
    └── notebooks/
```

---

## Checkpoints

Pre-trained checkpoints available on [HuggingFace](https://huggingface.co/medarsiddhant/llama4-tiny-vlm):

| File | Description | Size |
|------|-------------|------|
| `checkpoints/text_pretraining_best.pt` | Phase 1: Pretrained LLM | 3.1GB |
| `checkpoints/vision_alignment_best.pt` | Phase 2: Aligned projector | 1.8GB |
| `checkpoints/instruction_tuning_best.pt` | Phase 3: Instruction-tuned | 3.5GB |
| `checkpoints/dpo_best.pt` | Phase 4: DPO-tuned (final) | 3.5GB |

---

## Hardware Requirements

- GPU: NVIDIA RTX 3090 (24GB)
- Total training time: ~35-40 hours across all phases

---

## Sample Outputs

See the inference notebooks for example model outputs at each training phase:

| Phase | Notebook | Description |
|-------|----------|-------------|
| Phase 1 | [`text_pretraining/notebooks/inference.ipynb`](text_pretraining/notebooks/inference.ipynb) | Text generation examples |
| Phase 2 | [`vision_language_alignment/notebooks/inference.ipynb`](vision_language_alignment/notebooks/inference.ipynb) | Image captioning examples |
| Phase 3 | [`visual_instruction_tuning/notebooks/inference.ipynb`](visual_instruction_tuning/notebooks/inference.ipynb) | Visual Q&A examples |
| Phase 4 | [`preference_tuning/notebooks/inference.ipynb`](preference_tuning/notebooks/inference.ipynb) | Before/after DPO comparison |

---

## Limitations

- **Small model size**: 470M params limits reasoning capacity compared to 7B+ models
- **Repetition**: May produce repetitive outputs (use repetition_penalty=1.2)
- **Training data**: Limited to COCO images and synthetic captions
- **Educational purpose**: Not intended for production use

---

## License

MIT
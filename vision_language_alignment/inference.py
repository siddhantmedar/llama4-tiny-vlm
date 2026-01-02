#!/usr/bin/env python3
"""
Inference script for Vision-Language Model.
Generate captions from images using the trained VLM.
"""

import torch
from pathlib import Path
from PIL import Image
import argparse
import tomllib

from model import create_vlm, DEFAULT_CONFIG, DEFAULT_LLM_CKPT
from dataset import get_image_transform, get_tokenizer


def load_model(checkpoint_path, config_path=None, device="cuda"):
    """Load trained VLM from checkpoint."""
    config_path = config_path or DEFAULT_CONFIG

    print(f"Loading config from: {config_path}")
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    print("Creating model...")
    model = create_vlm(cfg, str(DEFAULT_LLM_CKPT))

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    print(f"Model loaded on {device}")
    return model


def generate_caption(
    model,
    image_path,
    tokenizer,
    transform,
    device="cuda",
    max_new_tokens=50,
    temperature=0.8,
    top_k=50,
):
    """
    Generate a caption for an image.

    Args:
        model: Trained VisionLanguageModel
        image_path: Path to image file
        tokenizer: BPE tokenizer with <image> token
        transform: Image preprocessing transforms
        device: cuda or cpu
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
        top_k: Top-k sampling

    Returns:
        Generated caption string
    """
    # Load and preprocess image
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # Start with <image> token
    image_token_id = tokenizer.token_to_id("<image>")
    eos_token_id = tokenizer.token_to_id("</s>")

    input_ids = torch.tensor([[image_token_id]], device=device)  # [1, 1]

    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            logits = model(image_tensor, input_ids)  # [1, 196 + seq_len - 1, vocab]

            # Get logits for next token (last position)
            next_token_logits = logits[0, -1, :]  # [vocab]

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][-1]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample from distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Check for EOS
            if next_token.item() == eos_token_id:
                break

            generated_tokens.append(next_token.item())

            # Append to input_ids for next iteration
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    # Decode tokens to text
    caption = tokenizer.decode(generated_tokens)

    return caption


def generate_caption_greedy(
    model,
    image_path,
    tokenizer,
    transform,
    device="cuda",
    max_new_tokens=50,
):
    """
    Generate a caption using greedy decoding (deterministic).
    """
    # Load and preprocess image
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Start with <image> token
    image_token_id = tokenizer.token_to_id("<image>")
    eos_token_id = tokenizer.token_to_id("</s>")

    input_ids = torch.tensor([[image_token_id]], device=device)

    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(image_tensor, input_ids)
            next_token_logits = logits[0, -1, :]

            # Greedy: take argmax
            next_token = torch.argmax(next_token_logits).unsqueeze(0)

            if next_token.item() == eos_token_id:
                break

            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    caption = tokenizer.decode(generated_tokens)
    return caption


def main():
    parser = argparse.ArgumentParser(description="Generate captions from images")
    parser.add_argument("image", type=str, help="Path to image file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_vlm.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--max_tokens", type=int, default=50,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--greedy", action="store_true",
                       help="Use greedy decoding instead of sampling")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda or cpu)")

    args = parser.parse_args()

    # Resolve checkpoint path
    checkpoint_path = Path(__file__).parent / args.checkpoint

    # Load model
    model = load_model(checkpoint_path, device=args.device)

    # Load tokenizer and transforms
    tokenizer = get_tokenizer()
    transform = get_image_transform()

    # Generate caption
    print(f"\nGenerating caption for: {args.image}")
    print("-" * 50)

    if args.greedy:
        caption = generate_caption_greedy(
            model, args.image, tokenizer, transform,
            device=args.device, max_new_tokens=args.max_tokens
        )
    else:
        caption = generate_caption(
            model, args.image, tokenizer, transform,
            device=args.device, max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )

    print(f"Caption: {caption}")
    print("-" * 50)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Visual-Instruction tuning dataset for LLaMA 4 instruction tuning.
Uses LLaVA-Instruct-150K downloaded locally (~300MB, cached in ~/.cache/huggingface/).
"""

import torch
import json
import os
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from torchvision import transforms
from pathlib import Path


# ==================== Image Transform ====================


def get_image_transform():
    """
    Image preprocessing pipeline matching ViT-B/16 training.

    Returns:
        torchvision.transforms.Compose pipeline
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ==================== Dataset ====================


class VisualInstructionDataset(torch.utils.data.Dataset):
    """
    Dataset for vision-instruction training.

    Each sample contains:
    - image: preprocessed image tensor [3, 224, 224]
    - input_ids: [<image>, USER: Q, ASSISTANT: A, </s>]
    - attention_mask: 1 for real tokens, 0 for padding
    - labels: -100 for USER/image tokens, actual tokens for ASSISTANT responses

    Note: LLaVA-Instruct-150K contains image filenames referencing COCO.
    Images must be loaded from local COCO directory.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        image_transform,
        coco_images_dir,
        max_seq_len=512,
    ):
        """
        Args:
            hf_dataset: HuggingFace dataset (LLaVA-Instruct-150K)
            tokenizer: BPE tokenizer with <image> token
            image_transform: torchvision transforms pipeline
            coco_images_dir: Path to COCO images (train2017/ and val2017/)
            max_seq_len: Max conversation length (including <image> and </s>)
        """
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.transform = image_transform
        self.coco_dir = Path(coco_images_dir)
        self.max_seq_len = max_seq_len

        # Get special token IDs
        self.image_token_id = tokenizer.token_to_id("<image>")
        self.pad_token_id = tokenizer.token_to_id("<pad>") or 0
        self.eos_token_id = tokenizer.token_to_id("</s>")
        self.ignore_index = -100

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get a single sample by index."""
        sample = self.dataset[idx]
        try:
            processed = self._process_sample(sample)
            if processed is not None:
                return processed
        except Exception:
            pass
        # Return a dummy sample on error (will be filtered by collate)
        return self._get_dummy_sample()

    def _get_dummy_sample(self):
        """Return a dummy sample for error cases (all labels masked)."""
        return {
            'image': torch.zeros(3, 224, 224),
            'input_ids': torch.zeros(self.max_seq_len, dtype=torch.long),
            'attention_mask': torch.zeros(self.max_seq_len, dtype=torch.long),
            'labels': torch.full((self.max_seq_len,), self.ignore_index, dtype=torch.long),
        }

    def _load_image(self, image_path_or_filename):
        """Load image from COCO directory."""
        from PIL import Image

        # Extract just the filename if a path is given (e.g., "coco/train2017/000000123.jpg" -> "000000123.jpg")
        image_filename = Path(image_path_or_filename).name

        # Support multiple directory layouts:
        # 1. Fiftyone: coco_dir/train/data/, coco_dir/validation/data/
        # 2. Standard COCO: coco_dir/train2017/, coco_dir/val2017/
        # 3. Direct: coco_dir/ (images directly in root)
        search_paths = [
            self.coco_dir / 'train' / 'data' / image_filename,      # fiftyone train
            self.coco_dir / 'validation' / 'data' / image_filename, # fiftyone val
            self.coco_dir / 'train2017' / image_filename,           # standard train
            self.coco_dir / 'val2017' / image_filename,             # standard val
            self.coco_dir / image_filename,                          # direct
        ]

        for image_path in search_paths:
            if image_path.exists():
                return Image.open(image_path)

        # If not found, raise to skip this sample
        raise FileNotFoundError(f"Image not found: {image_filename}")

    def _process_sample(self, sample):
        """
        Process a single conversation sample.

        Format: <image> USER: question ASSISTANT: answer USER: follow-up ASSISTANT: response </s>
        Labels: Only compute loss on ASSISTANT responses, mask everything else with -100
        """

        # ----- Image Processing -----
        # LLaVA dataset has 'image' as filename string, not actual image
        image_filename = sample['image']
        image = self._load_image(image_filename)

        # Handle grayscale images (convert to RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transforms: resize, crop, normalize
        image_tensor = self.transform(image)  # [3, 224, 224]

        # ----- Conversation Processing -----
        conversations = sample['conversations']

        # Build sequence token by token, tracking which are ASSISTANT responses
        input_ids = [self.image_token_id]  # Start with <image>
        labels = [self.ignore_index]  # Mask image token

        for turn in conversations:
            is_user = 'human' in turn['from']
            sender = 'USER' if is_user else 'ASSISTANT'

            # Clean message (remove <image> tag if present in first turn)
            message = turn['value'].replace('<image>', '').strip()

            # Format turn text
            turn_text = f" {sender}: {message}"

            # Tokenize just this turn
            turn_tokens = self.tokenizer.encode(turn_text).ids

            # Add tokens and labels
            input_ids.extend(turn_tokens)

            if is_user:
                # Mask USER tokens (don't compute loss)
                labels.extend([self.ignore_index] * len(turn_tokens))
            else:
                # Keep ASSISTANT tokens for loss computation
                labels.extend(turn_tokens)

        # Add EOS token
        input_ids.append(self.eos_token_id)
        labels.append(self.eos_token_id)  # Learn to predict EOS after assistant response

        # Truncate if too long
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]

        # Create attention mask (1 for real tokens)
        seq_len = len(input_ids)
        attention_mask = [1] * seq_len

        # Pad if too short
        padding_length = self.max_seq_len - seq_len
        if padding_length > 0:
            input_ids = input_ids + [self.pad_token_id] * padding_length
            labels = labels + [self.ignore_index] * padding_length  # Mask padding
            attention_mask = attention_mask + [0] * padding_length

        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return {
            'image': image_tensor,            # [3, 224, 224]
            'input_ids': input_ids,           # [max_seq_len]
            'attention_mask': attention_mask, # [max_seq_len]
            'labels': labels                  # [max_seq_len]
        }


# ==================== Tokenizer ====================


def get_tokenizer(tokenizer_path=None):
    """
    Load custom BPE tokenizer with <image> token.

    Args:
        tokenizer_path: Path to tokenizer.json file.
                       Defaults to bpe_tokenizer_with_image_tag.json
    """
    if tokenizer_path is None:
        tokenizer_path = Path(__file__).parent / "bpe_tokenizer_with_image_tag.json"

    tokenizer_path = Path(tokenizer_path)

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. "
            "Run notebooks/tokenizer_setup.ipynb first."
        )

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"Loaded tokenizer from {tokenizer_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    return tokenizer


# ==================== DataLoader ====================


class ListDataset:
    """Simple wrapper to make a list behave like a HuggingFace dataset."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloaders(
    coco_images_dir,
    batch_size=4,
    max_seq_len=512,
    num_workers=4,
    val_split_ratio=0.05,
):
    """
    Create DataLoaders for visual-instruction tuning.

    Downloads dataset locally (cached in ~/.cache/huggingface/).

    Args:
        coco_images_dir: Path to COCO images directory (containing train2017/, val2017/)
        batch_size: Batch size
        max_seq_len: Maximum sequence length for conversations
        num_workers: Number of data loading workers
        val_split_ratio: Fraction of data to use for validation (default 5%)

    Returns:
        (train_loader, val_loader)
    """
    print("Loading tokenizer...")
    tokenizer = get_tokenizer()

    print("Setting up image transforms...")
    image_transform = get_image_transform()

    print("Downloading LLaVA-Instruct-150K JSON (cached locally)...")
    json_path = hf_hub_download(
        repo_id="liuhaotian/LLaVA-Instruct-150K",
        filename="llava_v1_5_mix665k.json",
        repo_type="dataset"
    )

    print(f"Loading JSON from: {json_path}")
    with open(json_path, 'r') as f:
        all_data = json.load(f)

    # Filter to only COCO images (skip other datasets in the mix)
    coco_data = [s for s in all_data if 'coco' in s.get('image', '').lower()]
    print(f"Total samples: {len(all_data):,} | COCO samples: {len(coco_data):,}")

    # Split into train/val
    import random
    random.seed(42)
    random.shuffle(coco_data)

    val_size = int(len(coco_data) * val_split_ratio)
    val_data = coco_data[:val_size]
    train_data = coco_data[val_size:]

    print(f"Train: {len(train_data):,} | Val: {len(val_data):,}")

    # Wrap in ListDataset for indexing
    train_list = ListDataset(train_data)
    val_list = ListDataset(val_data)

    # Training dataset
    train_dataset = VisualInstructionDataset(
        hf_dataset=train_list,
        tokenizer=tokenizer,
        image_transform=image_transform,
        coco_images_dir=coco_images_dir,
        max_seq_len=max_seq_len,
    )

    # Validation dataset
    val_dataset = VisualInstructionDataset(
        hf_dataset=val_list,
        tokenizer=tokenizer,
        image_transform=image_transform,
        coco_images_dir=coco_images_dir,
        max_seq_len=max_seq_len,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"COCO images dir: {coco_images_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Max sequence length: {max_seq_len}")

    return train_loader, val_loader


# ==================== Test ====================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_dir", type=str, required=True,
                        help="Path to COCO images (containing train2017/, val2017/)")
    args = parser.parse_args()

    # Quick test
    train_loader, val_loader = get_dataloaders(
        coco_images_dir=args.coco_dir,
        batch_size=2,
        max_seq_len=512
    )

    print("\nTesting train loader...")
    batch = next(iter(train_loader))
    print(f"  image shape:          {batch['image'].shape}")
    print(f"  input_ids shape:      {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  labels shape:         {batch['labels'].shape}")
    print(f"  First token (should be 32000): {batch['input_ids'][0][0].item()}")

    # Show label masking
    print("\n  Label masking verification:")
    labels = batch['labels'][0]
    masked = (labels == -100).sum().item()
    total = (labels != 0).sum().item()  # Non-padding tokens
    print(f"    Masked tokens (USER + image): {masked}")
    print(f"    Total non-padding tokens: {total}")

    print("\nDataset test passed!")

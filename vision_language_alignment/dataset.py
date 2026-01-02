#!/usr/bin/env python3
"""
Vision-Language dataset for LLaMA 4 alignment training.
Uses COCO captions with streaming for memory efficiency.
"""

import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
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


class VisionLanguageDataset(IterableDataset):
    """
    Dataset for vision-language alignment training.

    Each sample contains:
    - image: preprocessed image tensor [3, 224, 224]
    - input_ids: [<image>, caption_tokens..., </s>]
    - attention_mask: 1 for real tokens, 0 for padding
    - labels: same as input_ids (for next-token prediction loss)
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        image_transform,
        max_seq_len=128,
    ):
        """
        Args:
            hf_dataset: HuggingFace streaming dataset
            tokenizer: BPE tokenizer with <image> token
            image_transform: torchvision transforms pipeline
            max_seq_len: Max caption length (including <image> and </s>)
        """
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.transform = image_transform
        self.max_seq_len = max_seq_len

        # Get special token IDs
        self.image_token_id = tokenizer.token_to_id("<image>")
        self.pad_token_id = tokenizer.token_to_id("<pad>") or 0
        self.eos_token_id = tokenizer.token_to_id("</s>")

    def __iter__(self):
        """Iterate through the streaming dataset."""
        for sample in self.dataset:
            try:
                processed = self._process_sample(sample)
                if processed is not None:
                    yield processed
            except Exception:
                # Skip corrupted samples
                continue

    def _process_sample(self, sample):
        """Process a single (image, caption) pair."""

        # ----- Image Processing -----
        image = sample['image']

        # Handle grayscale images (convert to RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transforms: resize, crop, normalize
        image_tensor = self.transform(image)  # [3, 224, 224]

        # ----- Caption Processing -----
        caption = sample['caption']

        # Prepend <image> token to caption
        # This placeholder will be replaced by 196 vision tokens during forward pass
        caption_with_image = f"<image> {caption}"

        # Tokenize
        encoding = self.tokenizer.encode(caption_with_image)
        token_ids = encoding.ids

        # Truncate (leave room for EOS)
        if len(token_ids) > self.max_seq_len - 1:
            token_ids = token_ids[:self.max_seq_len - 1]

        # Add EOS
        token_ids = token_ids + [self.eos_token_id]

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(token_ids)

        # Pad if too short
        padding_length = self.max_seq_len - len(token_ids)
        if padding_length > 0:
            token_ids = token_ids + [self.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        # Convert to tensors
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        # Labels are same as input_ids for language modeling
        labels = input_ids.clone()

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


def get_dataloaders(
    split="all",
    batch_size=4,
    max_seq_len=128,
    num_workers=0,
):
    """
    Create DataLoaders for vision-language alignment training.

    Args:
        split: Which loader(s) to return - 'train', 'val', or 'all'
        batch_size: Batch size
        max_seq_len: Maximum sequence length for captions
        num_workers: Number of data loading workers (0 for streaming)

    Returns:
        Single DataLoader if split specified, or (train_loader, val_loader) if 'all'
    """
    if split.lower() not in {"train", "val", "all"}:
        raise ValueError("split not valid. should be train/val/all")

    print("Loading tokenizer...")
    tokenizer = get_tokenizer()

    print("Setting up image transforms...")
    image_transform = get_image_transform()

    print("Loading COCO captions dataset (streaming)...")
    ds = load_dataset("jxie/coco_captions", streaming=True)

    # Training dataset
    train_dataset = VisionLanguageDataset(
        hf_dataset=ds['train'],
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_seq_len=max_seq_len,
    )

    # Validation dataset
    val_dataset = VisionLanguageDataset(
        hf_dataset=ds['validation'],
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_seq_len=max_seq_len,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
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

    print(f"Batch size: {batch_size}")
    print(f"Max sequence length: {max_seq_len}")

    if split.lower() == "train":
        return train_loader
    elif split.lower() == "val":
        return val_loader
    else:
        return train_loader, val_loader


# ==================== Test ====================


if __name__ == "__main__":
    # Quick test
    train_loader, val_loader = get_dataloaders(batch_size=4, max_seq_len=128)

    print("\nTesting train loader...")
    batch = next(iter(train_loader))
    print(f"  image shape:          {batch['image'].shape}")
    print(f"  input_ids shape:      {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  labels shape:         {batch['labels'].shape}")
    print(f"  First token (should be 32000): {batch['input_ids'][0][0].item()}")

    print("\nDataset test passed!")

    import os

    os._exit(0)

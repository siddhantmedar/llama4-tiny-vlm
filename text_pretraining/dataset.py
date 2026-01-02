#!/usr/bin/env python3
"""
Llama 4 pretraining dataset loading and DataLoader utilities.
Uses streaming dataset from HuggingFace for memory efficiency.
"""

import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from tokenizers import Tokenizer
from pathlib import Path
import tomllib


with open(Path(__file__).parent.parent / "config.toml", "rb") as f:
    cfg = tomllib.load(f)


# ==================== Dataset ====================


class PretrainDataset(IterableDataset):
    """
    Streaming dataset for LLM pretraining.
    Tokenizes text and creates (input_ids, labels) pairs for next token prediction.
    """

    def __init__(self, tokenizer, seq_len=2048, split="train", max_tokens=None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_tokens = max_tokens
        self.split = split

        self.ds = load_dataset(
            "HuggingFaceTB/smollm-corpus",
            "cosmopedia-v2",
            split=split,
            streaming=True,
        )

    def __iter__(self):
        buffer = []
        tokens_seen = 0

        for example in self.ds:
            # Tokenize text
            text = example.get("text", "")
            if not text:
                continue

            tokens = self.tokenizer.encode(text).ids
            buffer.extend(tokens)

            # Yield chunks of seq_len + 1 (for input and target)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)

                yield input_ids, labels

                buffer = buffer[self.seq_len :]
                tokens_seen += self.seq_len

                # Stop if max_tokens reached
                if self.max_tokens and tokens_seen >= self.max_tokens:
                    return


class ValidationDataset(IterableDataset):
    """
    Validation dataset - takes a fixed number of samples for evaluation.
    """

    def __init__(self, tokenizer, seq_len=2048, num_samples=1000):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.num_samples = num_samples

        # Use a different seed/skip for validation
        self.ds = load_dataset(
            "HuggingFaceTB/smollm-corpus",
            "cosmopedia-v2",
            split="train",
            streaming=True,
        )

    def __iter__(self):
        buffer = []
        samples_yielded = 0

        # Skip first 500k examples to create pseudo-validation split
        ds_iter = iter(self.ds.skip(500_000))

        for example in ds_iter:
            if samples_yielded >= self.num_samples:
                return

            text = example.get("text", "")
            if not text:
                continue

            tokens = self.tokenizer.encode(text).ids
            buffer.extend(tokens)

            while (
                len(buffer) >= self.seq_len + 1 and samples_yielded < self.num_samples
            ):
                chunk = buffer[: self.seq_len + 1]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)

                yield input_ids, labels

                buffer = buffer[self.seq_len :]
                samples_yielded += 1


# ==================== Tokenizer ====================


def get_tokenizer(tokenizer_path=None):
    """
    Load custom BPE tokenizer.

    Args:
        tokenizer_path: Path to tokenizer.json file.
                       Defaults to tokenizer/bpe_tokenizer.json
    """
    if tokenizer_path is None:
        # Default path relative to project root
        tokenizer_path = Path(__file__).parent / "tokenizer" / "bpe_tokenizer.json"

    tokenizer_path = Path(tokenizer_path)

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. "
            "Run notebooks/build_tokenizer.ipynb first."
        )

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"Loaded tokenizer from {tokenizer_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    return tokenizer


# ==================== DataLoader ====================


def get_dataloaders(
    split="all",
    batch_size=None,
    num_workers=4,
    seq_len=None,
    max_tokens=None,
    val_samples=1000,
):
    """
    Create DataLoaders for Llama pretraining.

    Args:
        split: Which loader(s) to return - 'train', 'val', or 'all'
        batch_size: Batch size (defaults to config)
        num_workers: Number of data loading workers
        seq_len: Sequence length (defaults to config)
        max_tokens: Max tokens per epoch for training (None = unlimited)
        val_samples: Number of validation samples

    Returns:
        Single DataLoader if split specified, or (train_loader, val_loader) if 'all'
    """
    if split.lower() not in {"train", "val", "all"}:
        raise ValueError("split not valid. should be train/val/all")

    # Use config defaults if not specified
    batch_size = batch_size or cfg.get("batch_size", 32)
    seq_len = seq_len or cfg.get("max_seq_len", 2048)

    print("Loading tokenizer...")
    tokenizer = get_tokenizer()

    print("Setting up streaming datasets...")

    # Training dataset
    train_dataset = PretrainDataset(
        tokenizer=tokenizer,
        seq_len=seq_len,
        split="train",
        max_tokens=max_tokens,
    )

    # Validation dataset
    val_dataset = ValidationDataset(
        tokenizer=tokenizer,
        seq_len=seq_len,
        num_samples=val_samples,
    )

    # Create DataLoaders
    # Note: IterableDataset doesn't support shuffle, num_workers > 0 needs care
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Streaming datasets handle their own ordering
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Validation samples: {val_samples}")

    if split.lower() == "train":
        return train_loader
    elif split.lower() == "val":
        return val_loader
    else:
        return train_loader, val_loader


# ==================== Test ====================


if __name__ == "__main__":
    # Quick test
    train_loader, val_loader = get_dataloaders(
        batch_size=4, seq_len=512, val_samples=100
    )

    print("\nTesting train loader...")
    for i, (input_ids, labels) in enumerate(train_loader):
        print(f"Batch {i}: input_ids={input_ids.shape}, labels={labels.shape}")
        if i >= 2:
            break

    print("\nTesting val loader...")
    for i, (input_ids, labels) in enumerate(val_loader):
        print(f"Batch {i}: input_ids={input_ids.shape}, labels={labels.shape}")
        if i >= 2:
            break

    print("\nDataset test passed!")

    import os

    os._exit(0)

#!/usr/bin/env python3
"""
TODO
CIFAR-100 dataset loading and DataLoader utilities.
"""


from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import tomllib


with open("config.toml", "rb") as f:
    cfg = tomllib.load(f)

# ==================== DataLoader ====================


class CIFARDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["img"]
        label = sample["fine_label"]

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(is_train=True):
    """CIFAR-100 transforms for Vision Transformer."""
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                ),
            ]
        )


def get_dataloaders(split="all", batch_size=32, num_workers=4):
    """
    Create DataLoaders for CIFAR-100.

    Args:
        split: Which loader(s) to return - 'train', 'val', 'test', or 'all'
        batch_size: Batch size for training
        num_workers: Number of data loading workers

    Returns:
        Single DataLoader if split specified, or (train_loader, val_loader, test_loader) if 'all'
    """
    if split.lower() not in {"train", "val", "test", "all"}:
        raise ValueError("split not valid. should be train/val/test/all")

    print("Loading CIFAR 100 datasets...")
    train_dataset = load_dataset(cfg["DATASET_NAME"], split="train")
    full_test_dataset = load_dataset(cfg["DATASET_NAME"], split="test")

    indices = list(range(len(full_test_dataset)))
    labels = full_test_dataset["fine_label"]

    val_idx, test_idx = train_test_split(
        indices, test_size=0.5, stratify=labels, random_state=313
    )

    val_dataset = full_test_dataset.select(val_idx)
    test_dataset = full_test_dataset.select(test_idx)

    train_dataset = CIFARDataset(train_dataset, get_transforms(is_train=True))
    val_dataset = CIFARDataset(val_dataset, get_transforms(is_train=False))
    test_dataset = CIFARDataset(test_dataset, get_transforms(is_train=False))

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"Test:  {len(test_dataset)} samples, {len(test_loader)} batches")

    if split.lower() == "train":
        return train_loader
    elif split.lower() == "val":
        return val_loader
    elif split.lower() == "test":
        return test_loader
    else:
        return train_loader, val_loader, test_loader

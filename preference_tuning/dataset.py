#!/usr/bin/env python3
"""
DPO Dataset for Vision-Language Model preference tuning.
Uses HuggingFaceH4/rlaif-v_formatted with chosen/rejected response pairs.
"""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from torchvision import transforms
from pathlib import Path
from PIL import Image
import io


# ==================== Image Transform ====================

def get_image_transform():
    """Image preprocessing matching ViT-B/16."""
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

class DPODataset(torch.utils.data.Dataset):
    """
    Dataset for DPO training on VLMs.

    Each sample contains:
    - image: preprocessed image tensor [3, 224, 224]
    - chosen_ids: full sequence [<image> + prompt + chosen_response]
    - chosen_labels: [-100 for prompt, token_ids for response]
    - rejected_ids: full sequence [<image> + prompt + rejected_response]
    - rejected_labels: [-100 for prompt, token_ids for response]
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        image_transform,
        max_seq_len=512,
    ):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.transform = image_transform
        self.max_seq_len = max_seq_len

        # Special tokens
        self.image_token_id = tokenizer.token_to_id("<image>")
        self.pad_token_id = tokenizer.token_to_id("<pad>") or 0
        self.eos_token_id = tokenizer.token_to_id("</s>")
        self.ignore_index = -100

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        try:
            return self._process_sample(sample)
        except Exception as e:
            # Return dummy on error
            return self._get_dummy_sample()

    def _get_dummy_sample(self):
        """Dummy sample for error cases."""
        dummy_len = self.max_seq_len
        return {
            'image': torch.zeros(3, 224, 224),
            'chosen_ids': torch.zeros(dummy_len, dtype=torch.long),
            'chosen_labels': torch.full((dummy_len,), self.ignore_index, dtype=torch.long),
            'rejected_ids': torch.zeros(dummy_len, dtype=torch.long),
            'rejected_labels': torch.full((dummy_len,), self.ignore_index, dtype=torch.long),
        }

    def _process_sample(self, sample):
        """Process a single DPO sample."""

        # ----- Image -----
        # Dataset has 'images' as a list of PIL images
        image = sample['images']
        if isinstance(image, list):
            image = image[0]  # Take first image
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, str):
            image = Image.open(image)
        # image is already a PIL Image from HF dataset
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_tensor = self.transform(image)

        # ----- Text -----
        # HuggingFaceH4/rlaif-v_formatted uses conversation format:
        # prompt: [{'content': [{'type': 'image'}, {'type': 'text', 'text': '...'}], 'role': 'user'}]
        # chosen/rejected: [{'content': [{'text': '...', 'type': 'text'}], 'role': 'assistant'}]

        # Extract question from prompt
        prompt_content = sample['prompt'][0]['content']
        question = next(item['text'] for item in prompt_content if item['type'] == 'text')

        # Extract chosen/rejected responses
        chosen = sample['chosen'][0]['content'][0]['text']
        rejected = sample['rejected'][0]['content'][0]['text']

        # Format: <image> USER: {question} ASSISTANT: {response} </s>
        # Tokenize prompt and response SEPARATELY then merge for exact prompt_len
        prompt_text = f" USER: {question} ASSISTANT:"
        prompt_tokens = self.tokenizer.encode(prompt_text).ids

        # Response tokens (with leading space to match natural formatting)
        chosen_response_tokens = self.tokenizer.encode(f" {chosen}").ids
        rejected_response_tokens = self.tokenizer.encode(f" {rejected}").ids

        # Build full sequences by concatenating
        chosen_ids = [self.image_token_id] + prompt_tokens + chosen_response_tokens
        rejected_ids = [self.image_token_id] + prompt_tokens + rejected_response_tokens
        prompt_len = 1 + len(prompt_tokens)  # Exact: <image> + prompt tokens

        # Create labels (mask prompt, keep response)
        chosen_labels = [self.ignore_index] * prompt_len + chosen_ids[prompt_len:]
        rejected_labels = [self.ignore_index] * prompt_len + rejected_ids[prompt_len:]

        # Truncate (leave room for EOS)
        max_len_without_eos = self.max_seq_len - 1
        chosen_ids = chosen_ids[:max_len_without_eos]
        chosen_labels = chosen_labels[:max_len_without_eos]
        rejected_ids = rejected_ids[:max_len_without_eos]
        rejected_labels = rejected_labels[:max_len_without_eos]

        # Add EOS at the end (always present)
        chosen_ids = chosen_ids + [self.eos_token_id]
        chosen_labels = chosen_labels + [self.eos_token_id]
        rejected_ids = rejected_ids + [self.eos_token_id]
        rejected_labels = rejected_labels + [self.eos_token_id]

        # Pad
        chosen_ids = self._pad(chosen_ids, self.pad_token_id)
        chosen_labels = self._pad(chosen_labels, self.ignore_index)
        rejected_ids = self._pad(rejected_ids, self.pad_token_id)
        rejected_labels = self._pad(rejected_labels, self.ignore_index)

        return {
            'image': image_tensor,
            'chosen_ids': torch.tensor(chosen_ids, dtype=torch.long),
            'chosen_labels': torch.tensor(chosen_labels, dtype=torch.long),
            'rejected_ids': torch.tensor(rejected_ids, dtype=torch.long),
            'rejected_labels': torch.tensor(rejected_labels, dtype=torch.long),
        }

    def _pad(self, ids, pad_value):
        """Pad sequence to max_seq_len."""
        if len(ids) < self.max_seq_len:
            ids = ids + [pad_value] * (self.max_seq_len - len(ids))
        return ids


# ==================== Tokenizer ====================

def get_tokenizer(tokenizer_path=None):
    """Load tokenizer with <image> token."""
    if tokenizer_path is None:
        # Use the one from visual_instruction_tuning
        tokenizer_path = Path(__file__).parent.parent / "visual_instruction_tuning" / "bpe_tokenizer_with_image_tag.json"

    tokenizer_path = Path(tokenizer_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"Loaded tokenizer: {tokenizer_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer


# ==================== DataLoader ====================

def get_dataloaders(
    batch_size=4,
    max_seq_len=512,
    num_workers=4,
):
    """
    Create DataLoaders for DPO training.

    Uses HuggingFaceH4/rlaif-v_formatted (83K train, ~800 test).
    """
    print("Loading tokenizer...")
    tokenizer = get_tokenizer()

    print("Setting up image transforms...")
    image_transform = get_image_transform()

    print("Loading RLAIF-V dataset (HuggingFaceH4 formatted)...")
    ds = load_dataset("HuggingFaceH4/rlaif-v_formatted")

    # Use existing train/test splits
    train_data = ds['train']
    val_data = ds['test']

    print(f"Train: {len(train_data):,} | Val: {len(val_data):,}")

    # Create datasets
    train_dataset = DPODataset(
        hf_dataset=train_data,
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_seq_len=max_seq_len,
    )

    val_dataset = DPODataset(
        hf_dataset=val_data,
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_seq_len=max_seq_len,
    )

    # Create dataloaders
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
    )

    return train_loader, val_loader, tokenizer


# ==================== Test ====================

if __name__ == "__main__":
    train_loader, val_loader, tokenizer = get_dataloaders(batch_size=2, max_seq_len=512)

    print("\nTesting dataloader...")
    batch = next(iter(train_loader))

    print(f"  image shape: {batch['image'].shape}")
    print(f"  chosen_ids shape: {batch['chosen_ids'].shape}")
    print(f"  chosen_labels shape: {batch['chosen_labels'].shape}")
    print(f"  rejected_ids shape: {batch['rejected_ids'].shape}")
    print(f"  rejected_labels shape: {batch['rejected_labels'].shape}")

    # Check label masking
    chosen_masked = (batch['chosen_labels'][0] == -100).sum().item()
    chosen_unmasked = (batch['chosen_labels'][0] != -100).sum().item()
    print(f"\n  Chosen: {chosen_masked} masked (prompt) / {chosen_unmasked} unmasked (response)")

    # Decode sample to verify
    sample_ids = batch['chosen_ids'][0].tolist()
    sample_labels = batch['chosen_labels'][0].tolist()
    # Find first non-masked label (response start)
    resp_start = next(i for i, l in enumerate(sample_labels) if l != -100)
    print(f"  Response starts at position: {resp_start}")

    print("\nDataset ready for DPO training!")

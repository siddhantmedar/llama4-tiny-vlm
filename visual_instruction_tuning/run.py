#!/usr/bin/env python3
"""
Training script for Visual Instruction Tuning (Phase 3).

Trains MLP projector (~4.7M) + LLM (~380M) = ~385M trainable params.
ViT encoder remains frozen (~86M).

Usage:
    python run.py --train --coco_dir /path/to/coco
    python run.py --train --coco_dir /path/to/coco --batch_size 8 --epochs 1
"""

import os
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import tomllib

from instruction_dataset import get_dataloaders
from model import create_instruct_vlm, DEFAULT_CONFIG, DEFAULT_LLM_CKPT, DEFAULT_VLM_CKPT


# ==================== Constants ====================

CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"
VOCAB_SIZE = 32001  # 32000 + <image> token


# ==================== Logging ====================

LOG_FILE = None


def setup_logging(save_path):
    """Setup log file in save directory."""
    global LOG_FILE
    os.makedirs(save_path, exist_ok=True)
    LOG_FILE = open(os.path.join(save_path, "training.log"), "a")


def log(msg):
    """Print to console and write to log file."""
    tqdm.write(msg)
    if LOG_FILE:
        LOG_FILE.write(msg + "\n")
        LOG_FILE.flush()


# ==================== Loss Computation ====================

def compute_loss(logits, labels):
    """
    Compute cross-entropy loss for instruction tuning.

    The labels already have -100 for:
    - <image> token (position 0)
    - USER tokens
    - Padding tokens

    Only ASSISTANT tokens contribute to loss.

    Args:
        logits: [B, 196 + seq_len - 1, vocab_size]
                196 vision tokens + (seq_len - 1) text tokens
        labels: [B, seq_len] with -100 for masked positions

    Returns:
        loss: scalar tensor
    """
    _, _, vocab_size = logits.shape
    text_seq_len = labels.shape[1]

    # Vision tokens = 196, so text starts at position 196 in logits
    # logits[:, 195, :] predicts labels[:, 1] (first text token after <image>)
    # logits[:, 196, :] predicts labels[:, 2]
    # ...

    # Slice logits to align with labels
    # Skip first 195 positions (vision-to-vision predictions)
    # Take positions 195 to 195 + text_seq_len - 1
    shift_logits = logits[:, 195:195 + text_seq_len - 1, :].contiguous()  # [B, seq-1, vocab]
    shift_labels = labels[:, 1:].contiguous()  # [B, seq-1]

    # Flatten and compute loss
    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100
    )

    return loss


# ==================== Training ====================

def train_epoch(model, train_loader, optimizer, scheduler, device, epoch,
                max_batches=-1, log_interval=100, save_interval=2000, save_dir="checkpoints"):
    """Train for one epoch with periodic checkpointing."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False, unit="batch")

    for batch_idx, batch in enumerate(pbar):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        # Move to device
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward
        optimizer.zero_grad()
        logits = model(images, input_ids)

        # Loss (labels already masked with -100)
        loss = compute_loss(logits, labels)

        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            log(f"Warning: NaN/inf loss at batch {batch_idx}, skipping")
            continue

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0
        )
        optimizer.step()
        scheduler.step()

        # Accumulate
        total_loss += loss.item()
        num_batches += 1

        # Progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
        })

        # Periodic logging
        if num_batches % log_interval == 0:
            log(f"  Step {num_batches} | loss: {loss.item():.4f} | lr: {optimizer.param_groups[0]['lr']:.2e}")

        # Periodic checkpoint saving
        if num_batches % save_interval == 0:
            ckpt_path = os.path.join(save_dir, f"checkpoint_step_{num_batches}.pt")
            torch.save({
                'epoch': epoch,
                'step': num_batches,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, ckpt_path)
            log(f"  → Saved checkpoint: {ckpt_path}")

    return total_loss / max(num_batches, 1)


def validate(model, val_loader, device, max_batches=500):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = model(images, input_ids)
            loss = compute_loss(logits, labels)

            total_loss += loss.item()
            num_batches += 1

            if num_batches >= max_batches:
                break

    return total_loss / max(num_batches, 1)


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, path)


# ==================== Main Training ====================

def train(
    model,
    train_loader,
    val_loader,
    epochs=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=500,
    save_dir="checkpoints",
    device=None,
    max_batches_per_epoch=-1,
    val_batches=500,
    log_interval=100,
):
    """
    Main training loop for instruction tuning.

    Args:
        model: InstructionTunedVLM
        train_loader: Training dataloader
        val_loader: Validation dataloader
        epochs: Number of epochs
        learning_rate: Learning rate
        weight_decay: AdamW weight decay
        warmup_steps: LR warmup steps
        save_dir: Directory to save checkpoints
        device: Training device
        max_batches_per_epoch: Max batches per epoch (-1 for full epoch)
        val_batches: Number of validation batches
        log_interval: Log every N batches
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    # Optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95)
    )

    # Scheduler: warmup + cosine decay
    # Estimate total steps: ~142k samples / batch_size
    effective_batches = max_batches_per_epoch if max_batches_per_epoch > 0 else 15000
    total_steps = epochs * effective_batches

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Tensorboard
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "runs", timestamp))

    # Param counts
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())
    frozen_count = total_count - trainable_count

    log("=" * 60)
    log("Visual Instruction Tuning - Phase 3")
    log("=" * 60)
    log(f"Device: {device}")
    log(f"Trainable params: {trainable_count:,} ({100*trainable_count/total_count:.1f}%)")
    log(f"Frozen params:    {frozen_count:,} ({100*frozen_count/total_count:.1f}%)")
    log(f"Total params:     {total_count:,}")
    log(f"Learning rate: {learning_rate}")
    log(f"Epochs: {epochs}")
    log(f"Warmup steps: {warmup_steps}")
    log(f"Total steps (est): {total_steps}")
    log("=" * 60)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        log(f"\nEpoch {epoch + 1}/{epochs}")
        log("-" * 40)

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            max_batches=max_batches_per_epoch,
            log_interval=log_interval,
            save_interval=2000,
            save_dir=save_dir
        )

        # Validate
        val_loss = validate(model, val_loader, device, max_batches=val_batches)

        # Log
        log(f"Epoch {epoch + 1} Complete | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                os.path.join(save_dir, "best_instruct.pt")
            )
            log(f"  → Saved best model (val_loss: {val_loss:.4f})")

        # Save last
        save_checkpoint(
            model, optimizer, epoch, train_loss, val_loss,
            os.path.join(save_dir, "last_instruct.pt")
        )

    writer.close()

    log("\n" + "=" * 60)
    log("Training Complete!")
    log(f"Best validation loss: {best_val_loss:.4f}")
    log("=" * 60)

    return best_val_loss


# ==================== CLI ====================

def parse_args():
    # Load config for defaults
    with open(DEFAULT_CONFIG, "rb") as f:
        cfg = tomllib.load(f)

    parser = argparse.ArgumentParser(
        description="Visual Instruction Tuning (Phase 3)"
    )

    # Required
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--coco_dir", type=str, required=False,
                        help="Path to COCO images (train2017/, val2017/ or fiftyone format)")

    # Training hyperparameters (defaults from config.toml)
    parser.add_argument("--epochs", type=int,
                        default=cfg.get("instruct_epochs", 1))
    parser.add_argument("--batch_size", type=int,
                        default=cfg.get("instruct_batch_size", 8))
    parser.add_argument("--lr", type=float,
                        default=cfg.get("instruct_learning_rate", 2e-5))
    parser.add_argument("--weight_decay", type=float,
                        default=cfg.get("instruct_weight_decay", 0.01))
    parser.add_argument("--warmup_steps", type=int,
                        default=cfg.get("instruct_warmup_steps", 500))
    parser.add_argument("--max_seq_len", type=int,
                        default=cfg.get("instruct_max_seq_len", 512))
    parser.add_argument("--max_batches", type=int,
                        default=cfg.get("instruct_batches_per_epoch", -1),
                        help="Max batches per epoch (-1 for full epoch)")
    parser.add_argument("--val_batches", type=int,
                        default=cfg.get("instruct_val_batches", 500))
    parser.add_argument("--log_interval", type=int,
                        default=cfg.get("instruct_log_interval", 100))

    # Paths
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--llm_ckpt", type=str, default=str(DEFAULT_LLM_CKPT))
    parser.add_argument("--vlm_ckpt", type=str, default=str(DEFAULT_VLM_CKPT))

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    # Create model
    print("\nCreating InstructionTunedVLM...")
    model = create_instruct_vlm(cfg, args.llm_ckpt, args.vlm_ckpt)

    # Param summary
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")

    if args.train:
        if not args.coco_dir:
            print("\nError: --coco_dir is required for training")
            print("Usage: python run.py --train --coco_dir /path/to/coco")
            exit(1)

        # Setup logging
        os.makedirs(args.save_dir, exist_ok=True)
        setup_logging(args.save_dir)

        # Create dataloaders
        print(f"\nLoading dataloaders...")
        print(f"  COCO dir: {args.coco_dir}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Max seq len: {args.max_seq_len}")

        train_loader, val_loader = get_dataloaders(
            coco_images_dir=args.coco_dir,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
        )

        # Train
        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            save_dir=args.save_dir,
            max_batches_per_epoch=args.max_batches,
            val_batches=args.val_batches,
            log_interval=args.log_interval,
        )
    else:
        print("\nUsage:")
        print("  python run.py --train --coco_dir /path/to/coco")
        print("  python run.py --train --coco_dir /path/to/coco --batch_size 8 --epochs 1 --lr 2e-5")
        print("\nOptions:")
        print("  --epochs N         Number of training epochs (default: 1)")
        print("  --batch_size N     Batch size (default: 8)")
        print("  --lr FLOAT         Learning rate (default: 2e-5)")
        print("  --max_seq_len N    Max sequence length (default: 512)")
        print("  --max_batches N    Max batches per epoch, -1 for full (default: -1)")

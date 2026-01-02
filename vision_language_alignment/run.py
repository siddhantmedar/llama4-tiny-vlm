#!/usr/bin/env python3
"""
Training script for Vision-Language Alignment (Phase 2).
Trains only the MLP projector (~4.7M params) to align vision and text.

Usage:
    python run.py --train --epochs 5
    python run.py --train --epochs 5 --batch_size 8
"""

import os
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import tomllib

from dataset import get_dataloaders
from model import create_vlm, DEFAULT_CONFIG


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


# ==================== Training ====================

def compute_loss(logits, input_ids, model, loss_fn):
    """
    Compute cross-entropy loss with proper masking.

    Args:
        logits: [B, 196 + seq_len - 1, vocab_size + 1]
        input_ids: [B, seq_len] original input with <image> at position 0
        model: VisionLanguageModel (for pad_token_id)
        loss_fn: CrossEntropyLoss with ignore_index=-100

    Returns:
        loss: scalar tensor
    """
    batch_size = logits.size(0)
    seq_len = logits.size(1)  # 196 + text_len - 1
    vocab_size = logits.size(2)

    # Build labels
    # Position 0-195: vision tokens -> no prediction target (-100)
    # Position 196+: text tokens -> predict next token
    labels = torch.full((batch_size, seq_len), -100, device=logits.device, dtype=torch.long)

    # Fill text positions with token IDs
    # labels[196:] should contain input_ids[1:] (the actual caption tokens)
    text_start = model.num_vision_tokens  # 196
    labels[:, text_start:] = input_ids[:, 1:]  # Skip <image> token

    # Mask padding tokens
    labels[labels == model.pad_token_id] = -100

    # Standard next-token prediction shift
    # logits[i] predicts token at position i+1
    logits_flat = logits[:, :-1, :].reshape(-1, vocab_size)  # [B*(seq_len-1), vocab]
    labels_flat = labels[:, 1:].reshape(-1)  # [B*(seq_len-1)]

    loss = loss_fn(logits_flat, labels_flat)
    return loss


def train(
    model,
    train_loader,
    val_loader,
    learning_rate=2e-4,
    weight_decay=0.01,
    epochs=3,
    max_grad_norm=1.0,
    save_path="checkpoints",
    device=None,
    batches_per_epoch=-1,
    val_batches=500,
    log_interval=100,
):
    """
    Train the VLM projector.

    Only ~4.7M params are trainable (MLP projector).
    """
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Only optimize trainable params (projector)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    # LR scheduler: warmup + cosine decay
    # For streaming dataset with -1, estimate ~12k batches per epoch (567k / 48)
    effective_batches = batches_per_epoch if batches_per_epoch > 0 else 12000
    total_steps = epochs * effective_batches
    warmup_steps = min(1000, total_steps // 100)  # 1% warmup

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Tensorboard
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=os.path.join(save_path, "runs", timestamp))

    # Count params
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())

    log(f"Training on device: {device}")
    log(f"Trainable params: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.1f}%)")
    log(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")
    log(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    log("-" * 60)

    best_val_loss = float("inf")
    global_step = 0

    # Training loop
    epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch")

    for epoch in epoch_pbar:
        # === TRAIN ===
        model.train()
        train_loss = 0.0
        num_batches = 0

        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}", leave=False, unit="batch"
        )

        for batch_idx, batch in enumerate(train_pbar):
            # batches_per_epoch=-1 means full epoch
            if batches_per_epoch > 0 and batch_idx >= batches_per_epoch:
                break

            image = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)

            opt.zero_grad()

            # Forward
            logits = model(image, input_ids)

            # Loss
            loss = compute_loss(logits, input_ids, model, loss_fn)

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                log(f"Warning: NaN/inf loss at step {global_step}, skipping")
                continue

            # Backward
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

            opt.step()
            scheduler.step()

            train_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Progress bar
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Log to tensorboard
            if global_step % log_interval == 0:
                writer.add_scalar("Loss/train_step", loss.item(), global_step)
                writer.add_scalar("LR", opt.param_groups[0]["lr"], global_step)

        train_loss /= max(num_batches, 1)

        # === VALIDATE ===
        model.eval()
        val_loss = 0.0
        val_batch_count = 0

        with torch.no_grad():
            for batch in val_loader:
                image = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)

                logits = model(image, input_ids)
                loss = compute_loss(logits, input_ids, model, loss_fn)

                val_loss += loss.item()
                val_batch_count += 1

                # Limit validation batches
                if val_batch_count >= val_batches:
                    break

        val_loss /= max(val_batch_count, 1)

        # Logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        log(f"Epoch {epoch+1}/{epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

        epoch_pbar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}",
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(save_path, "best_vlm.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, best_path)
            log(f"  → Saved best model (val_loss: {val_loss:.4f})")

    # Save final
    final_path = os.path.join(save_path, "last_vlm.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "epoch": epochs - 1,
    }, final_path)

    writer.close()

    log("-" * 60)
    log("Training complete!")
    log(f"Best validation loss: {best_val_loss:.4f}")

    return best_val_loss


# ==================== Main ====================

def parse_args():
    # Load config for defaults
    with open(DEFAULT_CONFIG, "rb") as f:
        cfg = tomllib.load(f)

    parser = argparse.ArgumentParser(description="Train Vision-Language Model")

    # Training
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--epochs", type=int, default=cfg.get("vlm_epochs", 3))
    parser.add_argument("--lr", type=float, default=cfg.get("vlm_learning_rate", 2e-4))
    parser.add_argument("--weight_decay", type=float, default=cfg.get("vlm_weight_decay", 0.01))
    parser.add_argument("--max_grad_norm", type=float, default=cfg.get("vlm_max_grad_norm", 1.0))
    parser.add_argument("--batches_per_epoch", type=int, default=cfg.get("vlm_batches_per_epoch", -1))
    parser.add_argument("--val_batches", type=int, default=cfg.get("vlm_val_batches", 500))

    # Data
    parser.add_argument("--batch_size", type=int, default=cfg.get("vlm_batch_size", 8))
    parser.add_argument("--max_seq_len", type=int, default=cfg.get("vlm_max_seq_len", 64))

    # Paths
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--llm_ckpt", type=str,
                       default="../text_pretraining/checkpoints/best.pt")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load config
    config_path = Path(args.config)
    print(f"Loading config from: {config_path}")
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    # Resolve LLM checkpoint path
    llm_ckpt_path = Path(__file__).parent / args.llm_ckpt
    print(f"LLM checkpoint: {llm_ckpt_path}")

    # Create model
    print("Creating VisionLanguageModel...")
    model = create_vlm(cfg, str(llm_ckpt_path))

    # Count params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")

    if args.train:
        setup_logging(args.save_dir)

        # Get dataloaders
        print("Loading dataloaders...")
        train_loader, val_loader = get_dataloaders(
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
        )

        # Train
        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            max_grad_norm=args.max_grad_norm,
            save_path=args.save_dir,
            batches_per_epoch=args.batches_per_epoch,
            val_batches=args.val_batches,
        )
    else:
        print("\nUsage:")
        print("  python run.py --train")
        print("  python run.py --train --epochs 5 --batch_size 8 --lr 2e-4")

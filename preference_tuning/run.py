#!/usr/bin/env python3
"""
DPO Training for Vision-Language Model (Phase 4).

Direct Preference Optimization to improve response quality
by learning from chosen/rejected response pairs.

Usage:
    python run.py --train
    python run.py --train --batch_size 4 --epochs 1
"""

import os
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import tomllib

from dataset import get_dataloaders

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VL_INSTRUCT_DIR = PROJECT_ROOT / "visual_instruction_tuning"
DEFAULT_CONFIG = PROJECT_ROOT / "config.toml"
DEFAULT_INSTRUCT_CKPT = VL_INSTRUCT_DIR / "checkpoints" / "best_instruct.pt"

# Import model from Phase 3
import importlib.util
spec = importlib.util.spec_from_file_location("instruct_model", VL_INSTRUCT_DIR / "model.py")
instruct_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(instruct_module)
create_instruct_vlm = instruct_module.create_instruct_vlm
DEFAULT_LLM_CKPT = instruct_module.DEFAULT_LLM_CKPT
DEFAULT_VLM_CKPT = instruct_module.DEFAULT_VLM_CKPT


# ==================== Logging ====================

LOG_FILE = None


def setup_logging(save_path):
    global LOG_FILE
    os.makedirs(save_path, exist_ok=True)
    LOG_FILE = open(os.path.join(save_path, "training.log"), "a")


def log(msg):
    tqdm.write(msg)
    if LOG_FILE:
        LOG_FILE.write(msg + "\n")
        LOG_FILE.flush()


# ==================== DPO Loss ====================

def compute_log_probs(logits, labels, vocab_size):
    """
    Compute per-token log probabilities.

    Args:
        logits: [B, 196 + seq_len - 1, vocab_size]
        labels: [B, seq_len] with -100 for masked positions

    Returns:
        log_probs: [B] sum of log probs for non-masked tokens
        num_tokens: [B] count of non-masked tokens
    """
    batch_size = logits.size(0)                                      # B
    text_seq_len = labels.size(1)                                    # seq_len (e.g., 512)

    # Align logits with labels (same as instruction tuning)
    # logits[195:] predicts labels[1:]
    shift_logits = logits[:, 195:195 + text_seq_len - 1, :].contiguous()  # [B, seq_len-1, vocab]
    shift_labels = labels[:, 1:].contiguous()                             # [B, seq_len-1]

    # Compute log softmax
    log_softmax = F.log_softmax(shift_logits, dim=-1)                # [B, seq_len-1, vocab]

    # Create mask for response tokens (not -100)
    mask = shift_labels != -100                                       # [B, seq_len-1] bool
    safe_labels = shift_labels.clone()                                # [B, seq_len-1]
    safe_labels[~mask] = 0  # Replace -100 with 0 for gather

    # Gather log prob of each target token
    token_log_probs = log_softmax.gather(
        dim=-1, index=safe_labels.unsqueeze(-1)                       # [B, seq_len-1, 1]
    ).squeeze(-1)                                                     # [B, seq_len-1]

    # Zero out prompt positions (where mask is False)
    token_log_probs = token_log_probs * mask.float()                  # [B, seq_len-1]

    # Sum log probs per sequence → total log P(response | prompt, image)
    log_probs = token_log_probs.sum(dim=-1)                           # [B]
    num_tokens = mask.sum(dim=-1).float()                             # [B]

    return log_probs, num_tokens


def dpo_loss(
    policy_chosen_logits,    # [B, 196+seq_len-1, vocab]
    policy_rejected_logits,  # [B, 196+seq_len-1, vocab]
    ref_chosen_logits,       # [B, 196+seq_len-1, vocab]
    ref_rejected_logits,     # [B, 196+seq_len-1, vocab]
    chosen_labels,           # [B, seq_len]
    rejected_labels,         # [B, seq_len]
    vocab_size,
    beta=0.1,
):
    """
    Compute DPO loss.

    DPO Loss = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))

    Where log_ratio = log(policy/ref) = log(policy) - log(ref)
    """
    # Compute log probs for each (model, response) pair
    policy_chosen_logps, chosen_tokens = compute_log_probs(
        policy_chosen_logits, chosen_labels, vocab_size
    )                                                                 # [B], [B]
    policy_rejected_logps, rejected_tokens = compute_log_probs(
        policy_rejected_logits, rejected_labels, vocab_size
    )                                                                 # [B], [B]
    ref_chosen_logps, _ = compute_log_probs(
        ref_chosen_logits, chosen_labels, vocab_size
    )                                                                 # [B]
    ref_rejected_logps, _ = compute_log_probs(
        ref_rejected_logits, rejected_labels, vocab_size
    )                                                                 # [B]

    # Compute log ratios: how much policy differs from reference
    chosen_log_ratio = policy_chosen_logps - ref_chosen_logps         # [B]
    rejected_log_ratio = policy_rejected_logps - ref_rejected_logps   # [B]

    # DPO loss: push chosen_log_ratio up, rejected_log_ratio down
    logits_diff = beta * (chosen_log_ratio - rejected_log_ratio)      # [B]
    loss = -F.logsigmoid(logits_diff).mean()                          # scalar

    # Metrics
    chosen_reward = beta * chosen_log_ratio.detach().mean()           # scalar
    rejected_reward = beta * rejected_log_ratio.detach().mean()       # scalar
    reward_margin = (chosen_reward - rejected_reward).item()          # float
    accuracy = (logits_diff > 0).float().mean().item()                # float (0-1)

    return loss, {
        'reward_margin': reward_margin,
        'accuracy': accuracy,
        'chosen_reward': chosen_reward.item(),
        'rejected_reward': rejected_reward.item(),
    }


# ==================== Training ====================

def train_epoch(
    policy_model,
    ref_model,
    train_loader,
    optimizer,
    scheduler,
    device,
    epoch,
    beta=0.1,
    max_batches=-1,
    log_interval=50,
    save_interval=500,
    save_dir="checkpoints",
):
    """Train for one epoch."""
    policy_model.train()
    ref_model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    vocab_size = 32001  # 32000 + <image>

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False, unit="batch")

    for batch_idx, batch in enumerate(pbar):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        # Move to device
        images = batch['image'].to(device)              # [B, 3, 224, 224]
        chosen_ids = batch['chosen_ids'].to(device)     # [B, seq_len]
        chosen_labels = batch['chosen_labels'].to(device)   # [B, seq_len]
        rejected_ids = batch['rejected_ids'].to(device)     # [B, seq_len]
        rejected_labels = batch['rejected_labels'].to(device)  # [B, seq_len]

        optimizer.zero_grad()

        # Forward pass - policy model
        policy_chosen_logits = policy_model(images, chosen_ids)      # [B, 196+seq_len-1, vocab]
        policy_rejected_logits = policy_model(images, rejected_ids)  # [B, 196+seq_len-1, vocab]

        # Forward pass - reference model (no grad)
        with torch.no_grad():
            ref_chosen_logits = ref_model(images, chosen_ids)      # [B, 196+seq_len-1, vocab]
            ref_rejected_logits = ref_model(images, rejected_ids)  # [B, 196+seq_len-1, vocab]

        # Compute DPO loss
        loss, metrics = dpo_loss(
            policy_chosen_logits,
            policy_rejected_logits,
            ref_chosen_logits,
            ref_rejected_logits,
            chosen_labels,
            rejected_labels,
            vocab_size,
            beta=beta,
        )

        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            log(f"Warning: NaN/inf loss at batch {batch_idx}, skipping")
            continue

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy_model.parameters() if p.requires_grad],
            max_norm=1.0
        )
        optimizer.step()
        scheduler.step()

        # Accumulate
        total_loss += loss.item()
        total_accuracy += metrics['accuracy']
        num_batches += 1

        # Progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{metrics['accuracy']:.2%}",
            "margin": f"{metrics['reward_margin']:.3f}",
        })

        # Periodic logging
        if num_batches % log_interval == 0:
            log(f"  Step {num_batches} | loss: {loss.item():.4f} | acc: {metrics['accuracy']:.2%} | margin: {metrics['reward_margin']:.3f}")

        # Periodic checkpoint
        if num_batches % save_interval == 0:
            ckpt_path = os.path.join(save_dir, f"checkpoint_step_{num_batches}.pt")
            torch.save({
                'step': num_batches,
                'model_state_dict': policy_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, ckpt_path)
            log(f"  → Saved checkpoint: {ckpt_path}")

    avg_loss = total_loss / max(num_batches, 1)
    avg_acc = total_accuracy / max(num_batches, 1)

    return avg_loss, avg_acc


def validate(policy_model, ref_model, val_loader, device, beta=0.1, max_batches=200):
    """Validate the model."""
    policy_model.eval()
    ref_model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    vocab_size = 32001

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            images = batch['image'].to(device)
            chosen_ids = batch['chosen_ids'].to(device)
            chosen_labels = batch['chosen_labels'].to(device)
            rejected_ids = batch['rejected_ids'].to(device)
            rejected_labels = batch['rejected_labels'].to(device)

            # Forward
            policy_chosen_logits = policy_model(images, chosen_ids)
            policy_rejected_logits = policy_model(images, rejected_ids)
            ref_chosen_logits = ref_model(images, chosen_ids)
            ref_rejected_logits = ref_model(images, rejected_ids)

            # Loss
            loss, metrics = dpo_loss(
                policy_chosen_logits,
                policy_rejected_logits,
                ref_chosen_logits,
                ref_rejected_logits,
                chosen_labels,
                rejected_labels,
                vocab_size,
                beta=beta,
            )

            total_loss += loss.item()
            total_accuracy += metrics['accuracy']
            num_batches += 1

            if num_batches >= max_batches:
                break

    return total_loss / max(num_batches, 1), total_accuracy / max(num_batches, 1)


# ==================== Main ====================

def train(
    policy_model,
    ref_model,
    train_loader,
    val_loader,
    epochs=1,
    learning_rate=5e-6,
    weight_decay=0.01,
    beta=0.1,
    warmup_steps=100,
    save_dir="checkpoints",
    device=None,
    max_batches_per_epoch=-1,
    val_batches=200,
    log_interval=50,
    save_interval=500,
):
    """Main DPO training loop."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    policy_model = policy_model.to(device)
    ref_model = ref_model.to(device)

    # Freeze reference model
    for param in ref_model.parameters():
        param.requires_grad = False

    # Optimizer
    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95)
    )

    # Scheduler - calculate actual steps from dataloader
    actual_batches = len(train_loader)
    effective_batches = max_batches_per_epoch if max_batches_per_epoch > 0 else actual_batches
    total_steps = epochs * effective_batches
    log(f"Total training steps: {total_steps:,} ({effective_batches:,} batches × {epochs} epochs)")

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
    total_count = sum(p.numel() for p in policy_model.parameters())

    log("=" * 60)
    log("DPO Training - Phase 4")
    log("=" * 60)
    log(f"Device: {device}")
    log(f"Trainable params: {trainable_count:,}")
    log(f"Total params: {total_count:,}")
    log(f"Learning rate: {learning_rate}")
    log(f"Beta (DPO temperature): {beta}")
    log(f"Epochs: {epochs}")
    log("=" * 60)

    best_val_acc = 0.0

    for epoch in range(epochs):
        log(f"\nEpoch {epoch + 1}/{epochs}")
        log("-" * 40)

        # Train
        train_loss, train_acc = train_epoch(
            policy_model, ref_model, train_loader, optimizer, scheduler,
            device, epoch, beta=beta, max_batches=max_batches_per_epoch,
            log_interval=log_interval, save_interval=save_interval, save_dir=save_dir
        )

        # Validate
        val_loss, val_acc = validate(
            policy_model, ref_model, val_loader, device, beta=beta, max_batches=val_batches
        )

        # Log
        log(f"Epoch {epoch + 1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.2%}")
        log(f"           | val_loss: {val_loss:.4f} | val_acc: {val_acc:.2%}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': policy_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, os.path.join(save_dir, "best_dpo.pt"))
            log(f"  → Saved best model (val_acc: {val_acc:.2%})")

        # Save last
        torch.save({
            'epoch': epoch,
            'model_state_dict': policy_model.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, os.path.join(save_dir, "last_dpo.pt"))

    writer.close()

    log("\n" + "=" * 60)
    log("DPO Training Complete!")
    log(f"Best validation accuracy: {best_val_acc:.2%}")
    log("=" * 60)

    if LOG_FILE:
        LOG_FILE.close()

    return best_val_acc


# ==================== CLI ====================

def parse_args(cfg):
    """Parse args with defaults from config.toml."""
    parser = argparse.ArgumentParser(description="DPO Training (Phase 4)")

    parser.add_argument("--train", action="store_true", help="Run training")

    # Hyperparameters (defaults from config.toml)
    parser.add_argument("--epochs", type=int, default=cfg.get('dpo_epochs', 1))
    parser.add_argument("--batch_size", type=int, default=cfg.get('dpo_batch_size', 4))
    parser.add_argument("--lr", type=float, default=cfg.get('dpo_learning_rate', 5e-6))
    parser.add_argument("--weight_decay", type=float, default=cfg.get('dpo_weight_decay', 0.01))
    parser.add_argument("--beta", type=float, default=cfg.get('dpo_beta', 0.1), help="DPO temperature")
    parser.add_argument("--max_seq_len", type=int, default=cfg.get('dpo_max_seq_len', 512))
    parser.add_argument("--max_batches", type=int, default=cfg.get('dpo_batches_per_epoch', -1))
    parser.add_argument("--val_batches", type=int, default=cfg.get('dpo_val_batches', 200))
    parser.add_argument("--warmup_steps", type=int, default=cfg.get('dpo_warmup_steps', 100))
    parser.add_argument("--log_interval", type=int, default=cfg.get('dpo_log_interval', 50))
    parser.add_argument("--save_interval", type=int, default=cfg.get('dpo_save_interval', 500))

    # Paths
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--instruct_ckpt", type=str, default=str(DEFAULT_INSTRUCT_CKPT))

    return parser.parse_args()


if __name__ == "__main__":
    # Load config first (needed for arg defaults)
    print(f"Loading config from: {DEFAULT_CONFIG}")
    with open(DEFAULT_CONFIG, "rb") as f:
        cfg = tomllib.load(f)

    args = parse_args(cfg)

    # Create policy model (will be trained)
    print("\nCreating policy model...")
    policy_model = create_instruct_vlm(cfg, str(DEFAULT_LLM_CKPT), str(DEFAULT_VLM_CKPT))

    # Load instruction-tuned weights
    print(f"Loading instruction-tuned weights: {args.instruct_ckpt}")
    ckpt = torch.load(args.instruct_ckpt, map_location='cpu', weights_only=False)
    policy_model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded! (val_loss: {ckpt.get('val_loss', 'N/A')})")

    # Create reference model (frozen copy)
    print("\nCreating reference model (frozen copy)...")
    ref_model = create_instruct_vlm(cfg, str(DEFAULT_LLM_CKPT), str(DEFAULT_VLM_CKPT))
    ref_model.load_state_dict(ckpt['model_state_dict'])

    # Param count
    trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy_model.parameters())
    print(f"\nParameters: {trainable:,} trainable / {total:,} total")

    if args.train:
        setup_logging(args.save_dir)

        print("\nLoading dataloaders...")
        train_loader, val_loader, tokenizer = get_dataloaders(
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
        )

        print(f"\nConfig: batch_size={args.batch_size}, epochs={args.epochs}, lr={args.lr}, beta={args.beta}")

        train(
            policy_model=policy_model,
            ref_model=ref_model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            beta=args.beta,
            warmup_steps=args.warmup_steps,
            save_dir=args.save_dir,
            max_batches_per_epoch=args.max_batches,
            val_batches=args.val_batches,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
        )
    else:
        print("\nUsage:")
        print("  python run.py --train")
        print("  python run.py --train --batch_size 4 --epochs 1 --beta 0.1")

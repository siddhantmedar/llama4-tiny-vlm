#!/usr/bin/env python3
"""
Resume training from checkpoint for Llama 4
Optimized for RTX 3090

Usage:
    python resume.py --checkpoint checkpoints/best.pt --epochs 8
"""

import os
import math
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import tomllib

from dataset import get_dataloaders, get_tokenizer
from model import Llama


LOG_FILE = None


def setup_logging(save_path):
    global LOG_FILE
    os.makedirs(save_path, exist_ok=True)
    LOG_FILE = open(os.path.join(save_path, "training_resume.log"), "a")


def log(msg):
    tqdm.write(msg)
    if LOG_FILE:
        LOG_FILE.write(msg + "\n")
        LOG_FILE.flush()


def resume_train(
    model,
    train_loader,
    val_loader,
    vocab_size,
    checkpoint_path,
    aux_loss_alpha=0.01,
    weight_decay=0.1,
    learning_rate=1e-4,  # Lower LR for fine-tuning
    epochs=8,
    max_grad_norm=1.0,
    save_path="checkpoints",
    device=None,
    batches_per_epoch=5000,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    # Load checkpoint
    log(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Try to load optimizer state (may fail if LR changed)
    try:
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        # Override LR
        for pg in opt.param_groups:
            pg["lr"] = learning_rate
        log("Loaded optimizer state, using new LR")
    except:
        log("Could not load optimizer state, starting fresh optimizer")

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_val_loss = checkpoint.get("val_loss", float("inf"))
    best_val_ppl = checkpoint.get("val_ppl", float("inf"))

    log(f"Resuming from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")

    # Cosine decay LR scheduler (no warmup since resuming)
    total_steps = epochs * batches_per_epoch

    def lr_lambda(step):
        progress = step / total_steps
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=os.path.join(save_path, "runs", f"resume_{timestamp}"))

    log(f"Training on device: {device}")
    log(f"LR: {learning_rate}, WD: {weight_decay}")
    log(f"Epochs: {epochs}, Batches/epoch: {batches_per_epoch}")
    log(f"Total new steps: {total_steps}")
    log("-" * 60)

    global_step = 0
    epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch")

    for epoch in epoch_pbar:
        actual_epoch = start_epoch + epoch

        # === TRAIN ===
        model.train()
        train_loss = 0.0
        train_aux_loss = 0.0
        num_batches = 0

        train_pbar = tqdm(
            train_loader, desc=f"Epoch {actual_epoch + 1}", leave=False, unit="batch"
        )
        for batch_idx, (input_ids, labels) in enumerate(train_pbar):
            if batch_idx >= batches_per_epoch:
                break
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            opt.zero_grad()
            logits = model(input_ids)
            ce_loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
            loss = ce_loss + (aux_loss_alpha * model.aux_loss)

            if torch.isnan(loss) or torch.isinf(loss):
                log(f"Warning: NaN/inf loss at step {global_step}, skipping")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            scheduler.step()

            train_loss += ce_loss.item()
            train_aux_loss += (
                model.aux_loss.item()
                if torch.is_tensor(model.aux_loss)
                else model.aux_loss
            )
            num_batches += 1
            global_step += 1

            ppl = math.exp(min(ce_loss.item(), 100))
            train_pbar.set_postfix({
                "loss": f"{ce_loss.item():.4f}",
                "ppl": f"{ppl:.2f}",
                "aux": f"{model.aux_loss:.4f}" if torch.is_tensor(model.aux_loss) else f"{model.aux_loss:.4f}",
            })

            if global_step % 100 == 0:
                writer.add_scalar("Loss/train_step", ce_loss.item(), global_step)
                writer.add_scalar("Perplexity/train_step", ppl, global_step)

        train_loss /= num_batches
        train_aux_loss /= num_batches
        train_ppl = math.exp(min(train_loss, 100))

        # === VALIDATE ===
        model.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
                val_loss += loss.item()
                num_batches += 1

        val_loss /= num_batches
        val_ppl = math.exp(min(val_loss, 100))

        writer.add_scalar("Loss/train", train_loss, actual_epoch)
        writer.add_scalar("Loss/val", val_loss, actual_epoch)
        writer.add_scalar("Perplexity/val", val_ppl, actual_epoch)

        current_lr = opt.param_groups[0]["lr"]
        log(f"Epoch {actual_epoch + 1} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | val_ppl: {val_ppl:.2f} | lr: {current_lr:.6f}")

        epoch_pbar.set_postfix({
            "val_loss": f"{val_loss:.4f}",
            "val_ppl": f"{val_ppl:.2f}",
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_ppl = val_ppl
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "epoch": actual_epoch,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
            }, os.path.join(save_path, "best.pt"))
            log(f"  → Saved best model (val_ppl: {best_val_ppl:.2f})")

        # Save checkpoint every 2 epochs
        if (epoch + 1) % 2 == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "epoch": actual_epoch,
            }, os.path.join(save_path, "last.pt"))

    # Final save
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "epoch": start_epoch + epochs - 1,
    }, os.path.join(save_path, "last.pt"))

    writer.close()
    log("-" * 60)
    log("Resume training complete!")
    log(f"Best val_loss: {best_val_loss:.4f}, Best val_ppl: {best_val_ppl:.2f}")

    return best_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume Llama 4 training")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--config", type=str, default="../config.toml")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batches_per_epoch", type=int, default=20000)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    print(f"Loading config from: {args.config}")
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    cfg["d_head"] = cfg["d_model"] // cfg["n_heads"]
    cfg["kv_d_head"] = cfg["d_model"] // cfg["n_kv_heads"]

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size}")

    model = Llama(
        vocab_size=vocab_size,
        n_layers=cfg["n_layers"],
        d_model=cfg["d_model"],
        d_head=cfg["d_head"],
        n_heads=cfg["n_heads"],
        n_kv_heads=cfg["n_kv_heads"],
        kv_d_head=cfg["kv_d_head"],
        d_ff_standard=cfg["d_ff_standard"],
        num_experts=cfg["num_experts"],
        num_experts_per_tok=cfg["num_experts_per_tok"],
        d_expert=cfg["d_expert"],
        rope_layers_ratio=cfg["rope_layers_ratio"],
        chunk_size=cfg["chunk_size"],
        rope_theta=cfg["rope_theta"],
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    setup_logging(args.save_dir)

    batch_size = cfg.get("batch_size", 8)
    seq_len = cfg.get("max_seq_len", 1024)

    train_loader, val_loader = get_dataloaders(
        split="all",
        batch_size=batch_size,
        seq_len=seq_len,
        val_samples=500,
    )

    resume_train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab_size=vocab_size,
        checkpoint_path=args.checkpoint,
        learning_rate=args.lr,
        epochs=args.epochs,
        save_path=args.save_dir,
        batches_per_epoch=args.batches_per_epoch,
    )
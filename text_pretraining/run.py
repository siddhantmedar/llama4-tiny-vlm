#!/usr/bin/env python3
"""
Training script for Llama 4
Includes training loop, validation, and text generation

Usage:
    python run.py --train --epochs 10
    python run.py --train --epochs 10 --config config_h200.toml
    python run.py --generate --prompt "Once upon a time"
"""

import os
import math
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import tomllib

from dataset import get_dataloaders, get_tokenizer
from model import Llama


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


def train(
    model,
    train_loader,
    val_loader,
    vocab_size,
    aux_loss_alpha=0.01,
    optimizer="adamw",
    weight_decay=0.1,
    learning_rate=3e-4,
    epochs=10,
    max_grad_norm=1.0,
    save_path="checkpoints",
    device=None,
    batches_per_epoch=1000,
):
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    # LR scheduler: warmup + cosine decay
    total_steps = epochs * batches_per_epoch
    warmup_steps = min(2000, total_steps // 10)  # 10% warmup, max 2000 steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2  # decay to 10% of max

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=os.path.join(save_path, "runs", timestamp))

    log(f"Training on device: {device}")
    log(f"Optimizer: {optimizer}, LR: {learning_rate}, WD: {weight_decay}")
    log(f"Aux loss alpha: {aux_loss_alpha}")
    log(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    log("-" * 60)

    best_val_loss = float("inf")
    best_val_ppl = float("inf")
    global_step = 0

    # Training loop with progress bar
    epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch")

    for epoch in epoch_pbar:
        # === TRAIN ===
        model.train()
        train_loss = 0.0
        train_aux_loss = 0.0
        num_batches = 0

        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}", leave=False, unit="batch"
        )
        for batch_idx, (input_ids, labels) in enumerate(train_pbar):
            if batch_idx >= batches_per_epoch:
                break
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            opt.zero_grad()

            logits = model(input_ids)  # [b, s, vocab_size]

            # Compute CE loss
            ce_loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))

            # Total loss = CE + aux loss (load balancing)
            loss = ce_loss + (aux_loss_alpha * model.aux_loss)

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                log(f"Warning: NaN/inf loss at step {global_step}, skipping batch")
                opt.zero_grad()
                continue

            # Backward pass
            loss.backward()

            # Gradient clipping
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

            # Perplexity
            ppl = math.exp(min(ce_loss.item(), 100))  # Cap to avoid overflow

            train_pbar.set_postfix(
                {
                    "loss": f"{ce_loss.item():.4f}",
                    "ppl": f"{ppl:.2f}",
                    "aux": (
                        f"{model.aux_loss:.4f}"
                        if torch.is_tensor(model.aux_loss)
                        else f"{model.aux_loss:.4f}"
                    ),
                }
            )

            # Log every 100 steps
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

        # Logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Loss/aux", train_aux_loss, epoch)
        writer.add_scalar("Perplexity/train", train_ppl, epoch)
        writer.add_scalar("Perplexity/val", val_ppl, epoch)

        current_lr = opt.param_groups[0]["lr"]
        writer.add_scalar("Learning_rate", current_lr, epoch)

        # Log epoch summary
        log(f"Epoch {epoch+1}/{epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | val_ppl: {val_ppl:.2f}")

        # Update progress bar
        epoch_pbar.set_postfix(
            {
                "train_loss": f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_ppl": f"{val_ppl:.2f}",
                "lr": f"{current_lr:.6f}",
            }
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_ppl = val_ppl
            best_path = os.path.join(save_path, "best.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                },
                best_path,
            )
            log(f"  → Saved best model (val_loss: {best_val_loss:.4f}, val_ppl: {best_val_ppl:.2f})")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            last_path = os.path.join(save_path, "last.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "epoch": epoch,
                },
                last_path,
            )

    # Save final checkpoint
    final_path = os.path.join(save_path, "last.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "epoch": epochs - 1,
        },
        final_path,
    )

    writer.close()

    log("-" * 60)
    log("Training complete!")
    log(f"Best validation loss: {best_val_loss:.4f}")
    log(f"Best validation perplexity: {best_val_ppl:.2f}")

    return best_val_loss


def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, device=None):
    """Generate text from a prompt."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt).ids  # [seq_len]
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)  # [1, seq_len]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)  # [1, seq_len, vocab_size]

            next_token_logits = logits[:, -1, :] / temperature  # [1, vocab_size]

            # Numerical stability: clamp extreme values and handle NaN
            next_token_logits = torch.clamp(next_token_logits, min=-100, max=100)
            if torch.isnan(next_token_logits).any() or torch.isinf(next_token_logits).any():
                print("Warning: NaN/inf in logits, using uniform sampling")
                next_token = torch.randint(0, logits.size(-1), (1, 1), device=device)
            else:
                probs = F.softmax(next_token_logits, dim=-1)  # [1, vocab_size]
                next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]

            input_ids = torch.cat([input_ids, next_token], dim=1)  # [1, seq_len+1]

            if next_token.item() == 2:  # EOS token
                break

    output_ids = input_ids[0].tolist()  # [seq_len]
    output_text = tokenizer.decode(output_ids)

    return output_text


def parse_args():
    parser = argparse.ArgumentParser(description="Train Llama 4 on cosmopedia-v2")

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--aux_loss_alpha", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Data
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--val_samples", type=int, default=500)

    # Misc
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--config", type=str, default="../config.toml", help="Config file path")
    parser.add_argument("--train", action="store_true", help="Enable training mode")
    parser.add_argument("--generate", action="store_true", help="Generate text")
    parser.add_argument(
        "--prompt", type=str, default="Once upon a time", help="Prompt for generation"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Loading config from: {args.config}")
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    # Compute derived config values
    cfg["d_head"] = cfg["d_model"] // cfg["n_heads"]
    cfg["kv_d_head"] = cfg["d_model"] // cfg["n_kv_heads"]

    # Get tokenizer and vocab size
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    print(f"Vocab size: {vocab_size}")
    print(
        f"Model config: d_model={cfg['d_model']}, n_layers={cfg['n_layers']}, n_heads={cfg['n_heads']}"
    )

    # Create model
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

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    if args.train:
        setup_logging(args.save_dir)

        # Get dataloaders
        batch_size = args.batch_size or cfg.get("batch_size", 32)
        seq_len = args.seq_len or cfg.get("max_seq_len", 2048)

        train_loader, val_loader = get_dataloaders(
            split="all",
            batch_size=batch_size,
            seq_len=seq_len,
            val_samples=args.val_samples,
        )

        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            vocab_size=vocab_size,
            aux_loss_alpha=args.aux_loss_alpha,
            optimizer="adamw",
            weight_decay=args.weight_decay,
            learning_rate=args.lr,
            epochs=args.epochs,
            max_grad_norm=args.max_grad_norm,
            save_path=args.save_dir,
            batches_per_epoch=cfg.get("batches_per_epoch", 1000),
        )

    elif args.generate:
        # Load checkpoint
        checkpoint_path = os.path.join(args.save_dir, "best.pt")
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            print("Checkpoint loaded successfully")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        print(f"\nPrompt: {args.prompt}")
        print("-" * 60)
        output = generate(model, tokenizer, args.prompt, max_new_tokens=100)
        print(f"Generated: {output}")

    else:
        print("Use --train to train or --generate to generate text")
        print("Example: python run.py --train --epochs 10")
        print("Example: python run.py --generate --prompt 'The meaning of life is'")

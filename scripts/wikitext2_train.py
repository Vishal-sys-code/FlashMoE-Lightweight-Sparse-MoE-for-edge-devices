# wikitext2_train.py
import argparse
import os
import time
import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from scripts.data import get_wikitext2_data
from scripts.model import create_model
from scripts.utils import set_seed, setup_logging, MetricsLogger


def train_one_epoch(model: nn.Module,
                    data_loader,
                    optimizer: torch.optim.Optimizer,
                    scheduler: Optional[object],
                    criterion: nn.Module,
                    device: torch.device,
                    pad_id: int,
                    grad_clip: float = 1.0,
                    lambda_bal: float = 1e-3,
                    debug: bool = False):
    """
    Train one epoch. Loss computed as sum over tokens (ignore_index=pad_id),
    aggregated to per-token average for reporting. Scheduler is stepped per optimizer.step().
    """
    model.train()
    total_loss_sum = 0.0        # sum of token losses across batches
    total_nonpad = 0           # total non-pad tokens across batches
    start_time = time.time()
    step_idx = 0

    for step, (inputs, targets) in enumerate(data_loader, 1):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs, aux_loss, _ = model(inputs)  # outputs: [B, T, V]
        B, T, V = outputs.size()

        outputs_flat = outputs.view(-1, V)     # [B*T, V]
        targets_flat = targets.view(-1)        # [B*T]

        # loss_sum sums only non-ignored tokens
        loss_sum = criterion(outputs_flat, targets_flat)  # scalar sum over tokens (ignore_index accounted)
        nonpad = (targets_flat != pad_id).sum().item()

        # avoid divide by zero
        if nonpad == 0:
            # nothing to learn in this batch
            continue

        # aux_loss might be None
        if aux_loss is None:
            aux_loss = torch.tensor(0.0, device=loss_sum.device, dtype=loss_sum.dtype)

        # per-token main loss for autograd
        main_loss_per_token = loss_sum / float(nonpad)
        total_loss = main_loss_per_token + (lambda_bal * aux_loss)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss_sum += loss_sum.item()
        total_nonpad += nonpad
        step_idx += 1

        if debug and (step % 50 == 0):
            grads = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
            grad_min = min(grads) if grads else 0.0
            grad_max = max(grads) if grads else 0.0
            lr = optimizer.param_groups[0]["lr"]
            logging.info(f"[debug] step={step} main_loss_per_token={main_loss_per_token.item():.6f} "
                         f"aux_loss={aux_loss.item():.6f} grad_min={grad_min:.3e} grad_max={grad_max:.3e} lr={lr:.2e}")

    elapsed = time.time() - start_time
    avg_token_loss = (total_loss_sum / max(1, total_nonpad)) if total_nonpad > 0 else 0.0
    return avg_token_loss, elapsed


def evaluate(model: nn.Module, data_loader, criterion: nn.Module, device: torch.device, pad_id: int):
    model.eval()
    total_loss_sum = 0.0
    total_nonpad = 0
    start_time = time.time()

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _, _ = model(inputs)
            B, T, V = outputs.size()
            outputs_flat = outputs.view(-1, V)
            targets_flat = targets.view(-1)
            loss_sum = criterion(outputs_flat, targets_flat)  # sum over non-pad tokens
            nonpad = (targets_flat != pad_id).sum().item()

            total_loss_sum += loss_sum.item()
            total_nonpad += nonpad

    avg_token_loss = (total_loss_sum / max(1, total_nonpad)) if total_nonpad > 0 else float("inf")
    perplexity = float(torch.exp(torch.tensor(avg_token_loss)).item()) if total_nonpad > 0 else float("inf")
    elapsed = time.time() - start_time
    return avg_token_loss, perplexity, elapsed


def main(args):
    # Setup output and logging
    outdir = os.path.join(args.outdir, args.model)
    os.makedirs(outdir, exist_ok=True)
    setup_logging(os.path.join(outdir, "train.log"))
    set_seed(args.seed)
    device = torch.device("cpu")

    logging.info(f"Starting training for model={args.model} with args: {args}")

    # Load data and tokenizer (ensure tokenizer is same object returned)
    train_loader, val_loader, _, tokenizer = get_wikitext2_data(
        args.batch_size, args.seq_len, debug=args.debug
    )

    # determine pad id safely
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        # fallback to 0 if tokenizer doesn't expose pad id
        pad_id = 0
        logging.warning("tokenizer.pad_token_id not found; using pad_id=0 as fallback")

    # Model
    model = create_model(args, tokenizer).to(device)

    # weight tying (best-effort)
    try:
        if hasattr(model, "decoder") and hasattr(model, "embedding"):
            model.decoder.weight = model.embedding.weight
            logging.info("Tied decoder weight to embedding weight")
    except Exception:
        logging.warning("Weight tying failed; continuing without it")

    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.3f}M")

    # Optimizer and scheduler (warmup + linear)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = max(100, int(0.03 * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # criterion sums token losses and uses ignore_index to exclude padding
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")

    metrics_logger = MetricsLogger(os.path.join(outdir, "metrics.csv"))

    best_val_loss = float("inf")
    best_ckpt_path = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_time = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, pad_id, grad_clip=args.grad_clip,
            lambda_bal=args.load_balancing_lambda, debug=args.debug
        )

        val_loss, val_ppl, val_time = evaluate(model, val_loader, criterion, device, pad_id)

        epoch_elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        logging.info(
            f"Epoch {epoch}/{args.epochs} | Time: {epoch_elapsed:.2f}s "
            f"| Train Loss (per-token): {train_loss:.6f} | Val Loss (per-token): {val_loss:.6f} "
            f"| Val PPL: {val_ppl:.2f} | LR: {current_lr:.2e}"
        )

        metrics_logger.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_perplexity": val_ppl,
            "lr": current_lr
        })

        # Save checkpoint and keep best
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args)
        }
        ckpt_path = os.path.join(outdir, f"epoch_{epoch}.pt")
        torch.save(ckpt, ckpt_path)
        logging.info(f"Saved checkpoint: {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = ckpt_path
            no_improve = 0
            logging.info(f"New best model (val_loss={best_val_loss:.6f}) saved: {best_ckpt_path}")
        else:
            no_improve += 1

        # optional early stopping if no improvement for N epochs
        if args.early_stop_patience is not None and no_improve >= args.early_stop_patience:
            logging.info(f"No improvement for {no_improve} epochs; early stopping.")
            break

    metrics_logger.save()
    logging.info("Training complete.")
    if best_ckpt_path is not None:
        logging.info(f"Best checkpoint saved at: {best_ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wikitext-2 Language Model Training (FlashMoE)")

    # model
    parser.add_argument("--model", type=str, required=True, choices=["flashmoe", "dense", "switch"])
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--ff_hidden", type=int, default=512)

    # MoE specifics
    parser.add_argument("--expert_type", type=str, default="lowrank", choices=["lowrank", "dense"])
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--capacity_factor", type=float, default=1.25)
    parser.add_argument("--load_balancing_lambda", type=float, default=1e-2)

    # training
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="./artifacts/wikitext2")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=None,
                        help="Number of epochs without val improvement before stopping (None disables)")

    args = parser.parse_args()
    main(args)

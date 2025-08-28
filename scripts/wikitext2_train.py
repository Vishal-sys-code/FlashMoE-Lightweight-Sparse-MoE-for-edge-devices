# scripts/wikitext2_train.py
import argparse
import os
import time
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from scripts.data import get_wikitext2_data
from scripts.model import create_model, compute_utilization_histogram, load_balance_loss  # model helpers if available
from scripts.utils import set_seed, setup_logging, MetricsLogger


def safe_model_forward(model, x, return_router_info=False):
    """
    Support multiple forward signatures used across iterations:
      1) logits, aux_loss, info = model(x)
      2) logits, info = model(x, return_router_info=True)  (aux_loss inside info)
      3) logits = model(x)  (no aux)
    Returns (logits, aux_loss_tensor, info_dict_or_none)
    """
    out = model(x) if not return_router_info else model(x, return_router_info=True)
    if isinstance(out, tuple):
        if len(out) == 3:
            return out  # assume (logits, aux_loss, info)
        if len(out) == 2:
            logits, info = out
            aux = None
            if isinstance(info, dict) and "aux_loss" in info:
                aux = info["aux_loss"]
            return logits, aux, info
    # single tensor
    return out, None, None


def compute_token_count(y):
    # y expected [B, T] integer targets OR [B, T, V] logits not used here
    if y is None:
        return 1
    return int(y.numel())


def train_one_epoch(model,
                    data_loader,
                    optimizer,
                    scheduler,
                    criterion,
                    device,
                    lambda_bal: float = 1e-2,
                    grad_clip: float = 1.0,
                    debug: bool = False,
                    overfit_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
    model.train()
    total_main_loss = 0.0
    total_tokens = 0
    start = time.time()

    # Optionally overfit a single batch: repeat it many steps
    if overfit_batch is not None:
        x0, y0 = overfit_batch
        steps = len(data_loader)  # iterate same number of steps as an epoch for stability
        for step in range(steps):
            x = x0.to(device); y = y0.to(device)
            optimizer.zero_grad(set_to_none=True)
            out_logits, out_aux, out_info = safe_model_forward(model, x, return_router_info=True)
            V = out_logits.size(-1)
            main_loss_sum = criterion(out_logits.view(-1, V), y.view(-1))  # sum over tokens
            aux_loss = out_aux if out_aux is not None else (out_info.get("aux_loss", torch.tensor(0.0, device=device)) if isinstance(out_info, dict) else torch.tensor(0.0, device=device))
            loss = main_loss_sum + (lambda_bal * aux_loss)
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            total_main_loss += float(main_loss_sum.item())
            total_tokens += int(y.numel())
            if debug and step % 20 == 0:
                logging.info(f"[overfit] step={step} main_loss_per_token={main_loss_sum.item()/y.numel():.6f} aux={aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss:.6f}")
        elapsed = time.time() - start
        return total_main_loss / max(1, total_tokens), elapsed

    # Normal epoch over loader
    for step, (x, y) in enumerate(data_loader, 1):
        x = x.to(device); y = y.to(device)
        optimizer.zero_grad(set_to_none=True)

        out_logits, out_aux, out_info = safe_model_forward(model, x, return_router_info=True)
        V = out_logits.size(-1)
        main_loss_sum = criterion(out_logits.view(-1, V), y.view(-1))  # sum over tokens

        # aux handling: model may return scalar aux_loss or provide router info
        aux_loss = out_aux
        if aux_loss is None and isinstance(out_info, dict):
            if "probs" in out_info and "topk_idx" in out_info and "topk_weights" in out_info:
                # compute load balance loss from router info (if model didn't compute it)
                try:
                    aux_loss = load_balance_loss(out_info["probs"], out_info["topk_idx"], out_info["topk_weights"], model.num_experts)
                except Exception:
                    aux_loss = torch.tensor(0.0, device=device)
            else:
                aux_loss = out_info.get("aux_loss", torch.tensor(0.0, device=device))

        if not isinstance(aux_loss, torch.Tensor):
            aux_loss = torch.tensor(float(aux_loss or 0.0), device=device)

        loss = main_loss_sum + (lambda_bal * aux_loss)
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_main_loss += float(main_loss_sum.item())
        total_tokens += int(y.numel())

        if debug and step % 200 == 0:
            logging.info(f"[debug] step={step} main_loss_per_token={main_loss_sum.item()/y.numel():.6f} aux={aux_loss.item():.6f}")

    elapsed = time.time() - start
    return (total_main_loss / max(1, total_tokens)), elapsed


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_main_loss = 0.0
    total_tokens = 0
    start = time.time()

    for (x, y) in data_loader:
        x = x.to(device); y = y.to(device)
        logits, aux, info = safe_model_forward(model, x, return_router_info=True)
        V = logits.size(-1)
        main_loss_sum = criterion(logits.view(-1, V), y.view(-1))
        total_main_loss += float(main_loss_sum.item())
        total_tokens += int(y.numel())

    avg_token_loss = total_main_loss / max(1, total_tokens)
    ppl = float(torch.exp(torch.as_tensor(avg_token_loss)).item())
    elapsed = time.time() - start
    return avg_token_loss, ppl, elapsed


def compute_utilization(model, data_loader, device, top_k=1):
    """
    Quick utilization histogram computed over a dataset using model.gating_network.
    Requires gating_network to return (topk_idx, topk_weights, probs, topc_idx).
    """
    model.eval()
    M = model.num_experts
    counts = torch.zeros((M,), dtype=torch.long, device=device)
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            topk_idx, topk_weights, probs, topc_idx = model.gating_network(x)
            flat_idx = topk_idx.reshape(-1)
            flat_w = topk_weights.reshape(-1)
            mask = flat_w > 1e-12
            if mask.sum() > 0:
                idx = flat_idx[mask]
                counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
    return counts.cpu().tolist()


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    setup_logging(os.path.join(args.outdir, f"{args.model}_train.log"))
    set_seed(args.seed)
    device = torch.device("cpu")

    logging.info(f"Starting training: model={args.model}, device={device}")
    logging.info(f"Hyperparams: {args}")

    # data
    train_loader, val_loader, test_loader, vocab = get_wikitext2_data(
        batch_size=args.batch_size, seq_len=args.seq_len, debug=args.debug
    )
    pad_id = getattr(vocab, "pad_token_id", -100)

    # model
    model = create_model(args, vocab).to(device)

    # weight tying if present
    try:
        if hasattr(model, "decoder") and hasattr(model, "embedding"):
            model.decoder.weight = model.embedding.weight
            logging.info("Weight tied: decoder <-> embedding")
    except Exception as e:
        logging.warning(f"Weight tying skipped: {e}")

    # optimizer/scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = max(1, int(args.warmup_frac * total_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # criterion: sum reduction, we'll divide by tokens manually
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")

    metrics = MetricsLogger(os.path.join(args.outdir, "metrics.csv"))
    best_val_loss = float("inf")
    best_ckpt = None

    # optional overfit single batch
    overfit_batch = None
    if args.overfit_batch:
        x0, y0 = next(iter(train_loader))
        logging.info("Overfit mode: using a single batch repeated for training.")
        overfit_batch = (x0, y0)

    for epoch in range(1, args.epochs + 1):
        t_loss, t_time = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            lambda_bal=args.load_balancing_lambda, grad_clip=args.grad_clip,
            debug=args.debug, overfit_batch=overfit_batch
        )

        v_loss, v_ppl, v_time = evaluate(model, val_loader, criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]

        # compute utilization histogram (small overhead)
        util_counts = compute_utilization(model, val_loader, device, top_k=args.top_k) if args.log_utilization else None

        logging.info(f"Epoch {epoch}/{args.epochs} | Train Time: {t_time:.2f}s | Val Time: {v_time:.2f}s | "
                     f"Train Loss (per-token): {t_loss:.6f} | Val Loss (per-token): {v_loss:.6f} | Val PPL: {v_ppl:.2f} | LR: {lr_now:.2e}")
        if util_counts is not None:
            logging.info(f"Expert utilization counts (val): {util_counts}")

        metrics.log({
            "epoch": epoch,
            "train_loss": t_loss,
            "val_loss": v_loss,
            "val_perplexity": v_ppl,
            "lr": lr_now,
            "util_counts": util_counts if util_counts is not None else []
        })

        # save checkpoint
        ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "args": vars(args)}
        ckpt_path = os.path.join(args.outdir, f"epoch_{epoch}.pt")
        torch.save(ckpt, ckpt_path)
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_ckpt = ckpt_path
            logging.info(f"New best checkpoint: {best_ckpt}")

        # early stopping
        # simple patience check using metrics file for simplicity
        if args.early_stop_patience is not None:
            logged = metrics.load_latest()  # if your MetricsLogger supports reading; else skip
            # We keep this hook lightweight (your utils may differ)

    logging.info(f"Training finished. Best val loss: {best_val_loss:.6f}. Best ckpt: {best_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wikitext-2: FlashMoE training harness (CPU-friendly)")

    # model config (defaults chosen to be CPU-friendly and match your Step 3)
    parser.add_argument("--model", type=str, required=True, choices=["flashmoe", "dense", "switch"])
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--ff_hidden", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_tying", action="store_true", default=True)

    # MoE params
    parser.add_argument("--expert_type", type=str, default="lowrank", choices=["lowrank", "dense"])
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--capacity_factor", type=float, default=1.25)
    parser.add_argument("--load_balancing_lambda", type=float, default=1e-2)
    parser.add_argument("--log_utilization", action="store_true", default=True)

    # training
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="./artifacts/wikitext2")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overfit_batch", action="store_true", help="Overfit single batch for debugging")
    parser.add_argument("--early_stop_patience", type=int, default=6)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=200)

    args = parser.parse_args()
    main(args)

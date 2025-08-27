import argparse
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import logging
import time

from scripts.data import get_wikitext2_data
from scripts.model import create_model
from scripts.utils import set_seed, setup_logging, MetricsLogger

# inside wikitext2_train.py

def train_one_epoch(model, data_loader, optimizer, criterion, device, grad_clip=1.0, lambda_bal=1e-3, debug=False):
    model.train()
    total_loss = 0.
    start_time = time.time()

    for step, (inputs, targets) in enumerate(data_loader, 1):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs, aux_loss, _ = model(inputs)  # outputs: [B, T, V], aux_loss: scalar tensor
        outputs_flat = outputs.view(-1, outputs.size(-1))
        targets_flat = targets.view(-1)

        main_loss = criterion(outputs_flat, targets_flat)
        # scale balancing loss (lambda hyperparam)
        total_epoch_loss = main_loss + (lambda_bal * aux_loss)

        total_epoch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += main_loss.item()

        if debug and (step % 50 == 0):
            # quick debug prints to sanity-check shapes/gradients
            print(f"[debug] step {step} main_loss={main_loss.item():.6f} aux_loss={aux_loss.item():.6f}")
            grads = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
            if grads:
                print(f"[debug] grad_norms min/max = {min(grads):.6e}/{max(grads):.6e}")

    avg_loss = total_loss / len(data_loader)
    elapsed = time.time() - start_time
    return avg_loss, elapsed

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _, _ = model(inputs)
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = targets.view(-1)
            loss = criterion(outputs_flat, targets_flat)
            total_loss += loss.item()
            
    avg_loss = total_loss / len(data_loader)
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity.item()

def main(args):
    # Setup
    outdir = os.path.join(args.outdir, args.model)
    os.makedirs(outdir, exist_ok=True)
    setup_logging(os.path.join(outdir, 'train.log'))
    set_seed(args.seed)
    device = torch.device("cpu")
    metrics_logger = MetricsLogger(os.path.join(outdir, 'metrics.csv'))

    logging.info(f"Starting training for model: {args.model}")
    logging.info(f"Args: {args}")

    # Data
    train_loader, val_loader, _, tokenizer = get_wikitext2_data(
        args.batch_size, args.seq_len, debug=args.debug
    )
    
    # Model
     # Model
    model = create_model(args, tokenizer).to(device)
    # Weight tying: share embedding and decoder weights (improves LM)
    try:
        model.decoder.weight = model.embedding.weight
    except Exception:
        pass
    logging.info(f"Model created. Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        train_loss, train_elapsed = train_one_epoch(model, train_loader, optimizer, criterion, device, args.grad_clip, lambda_bal=args.load_balancing_lambda, debug=args.debug)
        val_loss, val_ppl = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        epoch_elapsed = time.time() - epoch_start_time
        logging.info(f'Epoch {epoch}/{args.epochs} | Time: {epoch_elapsed:.2f}s | '
                     f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}')
        
        metrics_logger.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_perplexity': val_ppl,
            'lr': scheduler.get_last_lr()[0]
        })
        
        # Save checkpoint
        checkpoint_path = os.path.join(outdir, f'epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args
        }, checkpoint_path)
        logging.info(f"Checkpoint saved to {checkpoint_path}")

    metrics_logger.save()
    logging.info("Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wikitext-2 Language Model Training')
    # Model args
    parser.add_argument('--model', type=str, required=True, choices=['flashmoe', 'dense', 'switch'], help='Model type')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--ff_hidden', type=int, default=512, help='Feed-forward hidden size for MoE experts')
    
    # MoE specific args
    parser.add_argument('--expert_type', type=str, default='lowrank', choices=['lowrank', 'dense'], help='Type of expert network')
    parser.add_argument('--num_experts', type=int, default=8, help='Number of experts')
    parser.add_argument('--top_k', type=int, default=2, help='Top-k routing')
    parser.add_argument('--capacity_factor', type=float, default=1.25, help='Capacity factor for expert routing')
    parser.add_argument('--load_balancing_lambda', type=float, default=1e-2, help='Lambda for load balancing loss')

    # Training args
    parser.add_argument('--epochs', type=int, default=6, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=256, help='Sequence length')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--outdir', type=str, default='./artifacts/wikitext2', help='Output directory for checkpoints and logs')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (uses a small subset of data)')

    args = parser.parse_args()
    main(args)

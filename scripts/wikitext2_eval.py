import argparse
import os
import torch
import psutil
import time
import numpy as np
import pandas as pd
import logging

from scripts.data import get_wikitext2_data
from scripts.model import create_model
from scripts.utils import setup_logging, get_model_flops_per_token, choose_lowrank_r, analyze_router_stats

def measure_latency(model, dummy_input, device):
    """Measures average inference latency per token."""
    model.to(device)
    dummy_input = dummy_input.to(device)
    
    # Warm-up runs
    for _ in range(3):
        _ = model(dummy_input)
        
    # Timed runs
    latencies = []
    for _ in range(20):
        start_time = time.time()
        _, _, _ = model(dummy_input)
        end_time = time.time()
        latencies.append(end_time - start_time)
        
    avg_latency_batch = np.mean(latencies)
    tokens_per_batch = dummy_input.numel()
    avg_latency_token = avg_latency_batch / tokens_per_batch
    return avg_latency_token * 1000 # Return in ms

def measure_memory(model, dummy_input, device):
    """Measures peak memory usage during a forward pass."""
    model.to(device)
    dummy_input = dummy_input.to(device)
    
    process = psutil.Process(os.getpid())
    
    # Measure memory before
    mem_before = process.memory_info().rss / (1024 * 1024) # in MB
    
    # Run forward pass
    _, _, _ = model(dummy_input)
    
    # Measure memory after
    mem_after = process.memory_info().rss / (1024 * 1024) # in MB
    
    return mem_after - mem_before

def run_evaluation(model, val_loader, device):
    """Calculates perplexity on the validation set."""
    model.eval()
    total_loss = 0.
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _, _ = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    perplexity = np.exp(avg_loss)
    return perplexity

def main(args):
    setup_logging(os.path.join(os.path.dirname(args.checkpoint), 'eval.log'))
    device = torch.device("cpu")

    # Load checkpoint
    logging.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    train_args = checkpoint['args']

    # Load data
    _, val_loader, _, tokenizer = get_wikitext2_data(
        train_args.batch_size, train_args.seq_len, debug=args.debug
    )

    # Re-create model and load weights
    model = create_model(train_args, tokenizer)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    logging.info("Model loaded successfully.")

    # --- Run Evaluations ---
    results = {'checkpoint': args.checkpoint, 'model': train_args.model}

    # 1. Perplexity
    logging.info("Calculating perplexity...")
    perplexity = run_evaluation(model, val_loader, device)
    results['perplexity'] = perplexity
    logging.info(f"Perplexity: {perplexity:.4f}")

    dummy_input, _ = next(iter(val_loader))

    # 2. Latency
    logging.info("Measuring latency...")
    latency_ms = measure_latency(model, dummy_input, device)
    results['latency_ms_per_token'] = latency_ms
    logging.info(f"Latency per token: {latency_ms:.4f} ms")

    # 3. Memory
    logging.info("Measuring peak memory...")
    peak_mem_mb = measure_memory(model, dummy_input, device)
    results['peak_memory_mb'] = peak_mem_mb
    logging.info(f"Peak memory usage: {peak_mem_mb:.2f} MB")

    # 4. FLOPs
    logging.info("Calculating FLOPs...")
    r = choose_lowrank_r(train_args.d_model)
    dense_ffn_hidden_equiv = None
    if train_args.model == 'dense':
        # This is tricky because the actual hidden dim is already in train_args
        # We need to find what it was for the original MoE model it was based on
        # This part of the logic might need refinement. For now, we assume it's stored.
        dense_ffn_hidden_equiv = train_args.ff_hidden # Placeholder
    
    flops = get_model_flops_per_token(
        d_model=train_args.d_model,
        ff_hidden=train_args.ff_hidden,
        num_experts=train_args.num_experts,
        top_k=train_args.top_k,
        r=r,
        expert_type=train_args.expert_type,
        model_type=train_args.model,
        dense_ffn_hidden=dense_ffn_hidden_equiv
    )
    results['flops_per_token'] = flops
    logging.info(f"FLOPs per token: {flops}")

    # 5. Router Stats
    if train_args.model in ['flashmoe', 'switch']:
        logging.info("Analyzing router statistics...")
        all_stats = []
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                _, _, router_stats = model(inputs)
                if router_stats:
                    all_stats.append(router_stats)
        
        router_analysis_results = analyze_router_stats(all_stats, train_args.num_experts, train_args.top_k)
        logging.info(f"Router analysis results: {router_analysis_results}")
        results.update(router_analysis_results)

    # --- Exporting ---
    if args.export:
        export_path = os.path.join(os.path.dirname(args.checkpoint), f'model_export.{args.export}')
        logging.info(f"Exporting model to {export_path}...")
        dummy_input, _ = next(iter(val_loader))
        dummy_input = dummy_input.to(device)

        if args.export == 'ts':
            try:
                # Trace the forward_export method specifically
                traced_model = torch.jit.trace(lambda x: model.forward_export(x), dummy_input)
                torch.jit.save(traced_model, export_path)
                logging.info(f"Model exported to TorchScript at {export_path}")
            except Exception as e:
                logging.error(f"Failed to export to TorchScript: {e}")

        elif args.export == 'onnx':
            original_forward = model.forward
            model.forward = model.forward_export
            try:
                torch.onnx.export(
                    model,
                    dummy_input,
                    export_path,
                    input_names=['input_ids'],
                    output_names=['logits'],
                    dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                                  'logits': {0: 'batch_size', 1: 'sequence_length'}},
                    opset_version=11
                )
                logging.info(f"Model exported to ONNX at {export_path}")
            except Exception as e:
                logging.error(f"Failed to export to ONNX: {e}")
            finally:
                model.forward = original_forward # Restore original forward

    # Save results
    df = pd.DataFrame([results])
    df.to_csv(args.outpath, index=False)
    logging.info(f"Evaluation results saved to {args.outpath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wikitext-2 Model Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--outpath', type=str, required=True, help='Path to save evaluation results CSV')
    parser.add_argument('--export', type=str, choices=['ts', 'onnx'], help='Export model to TorchScript (ts) or ONNX (onnx)')
    parser.add_argument('--debug', action='store_true', help='Use debug data for quick evaluation')
    args = parser.parse_args()
    main(args)

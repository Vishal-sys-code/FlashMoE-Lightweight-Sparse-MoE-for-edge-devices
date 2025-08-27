import random
import os
import numpy as np
import torch
import logging
import pandas as pd

def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def choose_lowrank_r(d_model: int) -> int:
    """
    Chooses the rank 'r' for the low-rank expert decomposition.
    A simple heuristic.
    """
    return d_model // 4

def calculate_expert_flops(d_model: int, ff_hidden: int, r: int, expert_type: str) -> int:
    """
    Calculates the approximate FLOPs for a single forward pass of one expert.
    An FFN is treated as two linear layers. FLOPs for one linear layer X@W is 2 * M * K * N for X(M,K) and W(K,N).
    Here, input is (1, d_model).
    """
    if expert_type == 'dense':
        # Layer 1: (1, d_model) @ (d_model, ff_hidden) -> (1, ff_hidden) => 2 * d_model * ff_hidden
        # Layer 2: (1, ff_hidden) @ (ff_hidden, d_model) -> (1, d_model) => 2 * ff_hidden * d_model
        return 2 * d_model * ff_hidden + 2 * ff_hidden * d_model
    elif expert_type == 'lowrank':
        # Layer 1 is decomposed: d_model -> r -> ff_hidden
        # (1, d_model) @ (d_model, r) -> (1, r) => 2 * d_model * r
        # (1, r) @ (r, ff_hidden) -> (1, ff_hidden) => 2 * r * ff_hidden
        # Layer 2: (1, ff_hidden) @ (ff_hidden, d_model) -> (1, d_model) => 2 * ff_hidden * d_model
        return 2 * d_model * r + 2 * r * ff_hidden + 2 * ff_hidden * d_model
    else:
        raise ValueError(f"Unknown expert_type: {expert_type}")

def get_model_flops_per_token(
    d_model: int,
    ff_hidden: int,
    num_experts: int,
    top_k: int,
    r: int,
    expert_type: str,
    model_type: str,
    dense_ffn_hidden: int = None
) -> int:
    """
    Calculates the expected FLOPs per token for the feed-forward part of the model.
    """
    if model_type == 'dense':
        if dense_ffn_hidden is None:
            raise ValueError("dense_ffn_hidden must be provided for dense model")
        # FFN FLOPs: (d_model -> dense_ffn_hidden -> d_model)
        return 2 * d_model * dense_ffn_hidden + 2 * dense_ffn_hidden * d_model

    # For MoE models, add gating FLOPs
    gate_flops = 2 * d_model * num_experts
    
    expert_flops = calculate_expert_flops(d_model, ff_hidden, r, expert_type)
    
    # Expected FLOPs is gate_flops + weighted sum of expert_flops
    expected_flops = gate_flops + top_k * expert_flops
    return expected_flops

def calculate_dense_hidden_dim_for_flops_match(
    d_model: int,
    ff_hidden: int,
    num_experts: int,
    top_k: int,
    r: int,
    expert_type: str
) -> int:
    """
    Calculates the hidden dimension for a dense FFN to match the expected FLOPs
    of a FlashMoE layer.
    """
    moe_flops = get_model_flops_per_token(
        d_model=d_model,
        ff_hidden=ff_hidden,
        num_experts=num_experts,
        top_k=top_k,
        r=r,
        expert_type=expert_type,
        model_type='flashmoe'
    )
    
    # Dense FFN FLOPs = 2 * d_model * d_hidden + 2 * d_hidden * d_model = 4 * d_model * d_hidden
    # Solve for d_hidden: d_hidden = moe_flops / (4 * d_model)
    dense_hidden_dim = moe_flops / (4 * d_model)
    return int(dense_hidden_dim)

def setup_logging(log_path: str):
    """Sets up logging to file and console."""
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

class MetricsLogger:
    """A simple logger to store metrics in a list of dicts and save to CSV."""
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.metrics = []
        log_dir = os.path.dirname(self.csv_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def log(self, metric_dict: dict):
        self.metrics.append(metric_dict)
        logging.info(f"Logged metrics: {metric_dict}")

    def save(self):
        if not self.metrics:
            return
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.csv_path, index=False)
        logging.info(f"Metrics saved to {self.csv_path}")

def analyze_router_stats(all_router_stats, num_experts, top_k):
    """
    Analyzes the router statistics collected from the model during evaluation.

    Args:
        all_router_stats (list): A list of stats from each batch. Each element is a list of stats
                                 from each MoE layer in the model.
        num_experts (int): Number of experts.
        top_k (int): The k value for routing.

    Returns:
        dict: A dictionary containing various router statistics.
    """
    if not all_router_stats or not all_router_stats[0]:
        return {
            "utilization_counts": {},
            "utilization_mass": {},
            "avg_gating_accuracy": 0
        }

    # Aggregate stats from all batches and layers
    total_tokens = 0
    total_argmax_in_topk = 0
    
    # Let's assume for now we average the stats over layers
    # A more detailed analysis could report per-layer stats
    agg_router_probs = torch.cat([stats_dict['router_probs'] for batch_stats in all_router_stats for stats_dict in batch_stats])
    agg_top_k_indices = torch.cat([stats_dict['top_k_indices'] for batch_stats in all_router_stats for stats_dict in batch_stats])

    # 1. Expert utilization (counts)
    utilization_counts = torch.bincount(agg_top_k_indices.flatten(), minlength=num_experts).cpu().numpy()
    
    # 2. Expert utilization (mass)
    # Sum of probabilities for tokens routed to each expert
    one_hot_indices = F.one_hot(agg_top_k_indices, num_classes=num_experts).float()
    # one_hot_indices shape: (S, top_k, num_experts)
    # agg_router_probs shape: (S, num_experts)
    # We want to sum the router_probs for the chosen top_k indices
    # This is not straightforward. Let's simplify: sum the probabilities of all tokens for each expert.
    utilization_mass = agg_router_probs.sum(dim=0).cpu().numpy()

    # 3. Top-k gating accuracy
    argmax_choices = torch.argmax(agg_router_probs, dim=1)
    # Check if the argmax choice is present in the top-k choices for each token
    is_in_topk = (agg_top_k_indices == argmax_choices.unsqueeze(1)).any(dim=1)
    gating_accuracy = torch.sum(is_in_topk).item() / len(is_in_topk)

    return {
        "utilization_counts": {i: int(c) for i, c in enumerate(utilization_counts)},
        "utilization_mass": {i: float(m) for i, m in enumerate(utilization_mass)},
        "avg_gating_accuracy": float(gating_accuracy)
    }

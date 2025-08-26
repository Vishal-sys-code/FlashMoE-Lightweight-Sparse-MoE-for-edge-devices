import torch
import pytest
import math

from flash_moe.model import FlashMoEModel

@pytest.fixture
def params():
    return {
        "d_model": 128, 
        "num_experts": 8, 
        "d_hidden": 256, 
        "top_k": 2, 
        "capacity_factor": 1000.0
    }


def test_output_shape(params):
    B = 16
    model = FlashMoEModel(**params)
    x = torch.randn(B, params["d_model"])  # [B, d]
    y = model(x)
    assert y.shape == x.shape, f"expected {x.shape}, got {y.shape}"


def test_training_step(params):
    B = 8
    model = FlashMoEModel(**params)
    x = torch.randn(B, params["d_model"])  # [B, d]
    target = torch.randn_like(x)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    y = model(x)
    loss = torch.nn.MSELoss()(y, target)
    loss.backward()
    opt.step()

    # If we reached here without error, training step succeeded
    assert True


def test_masked_softmax_sums_to_one(params):
    """Ensure masked softmax over top-k logits sums to 1 per token."""
    B = 10
    model = FlashMoEModel(**params)
    x = torch.randn(B, params["d_model"])  # [B, d]

    encoded = model.shared_encoder(x)
    logits = model.gating_network(encoded)
    topk_vals, _ = torch.topk(logits, k=model.top_k, dim=-1)
    weights = model._stable_masked_softmax(topk_vals / model.tau, dim=-1)

    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def naive_forward(model: FlashMoEModel, x: torch.Tensor) -> torch.Tensor:
    """
    Naive reference forward that mirrors the FlashMoEModel vectorized dispatch,
    including capacity clamping per-expert. This produces outputs that match
    the model's forward when using the same params (top_k, capacity_factor, tau).
    """
    device = x.device
    B = x.size(0)

    # Encode and router decisions (same as model.forward)
    encoded = model.shared_encoder(x)                    # [B, d]
    logits = model.gating_network(encoded)              # [B, M]
    topk_vals, topk_idx = torch.topk(logits, k=model.top_k, dim=-1)  # [B, K]
    weights = model._stable_masked_softmax(topk_vals / float(model.tau), dim=-1)  # [B, K]

    # Prepare per-expert buckets (lists of token indices and weights)
    per_expert_tokens = {e: [] for e in range(model.num_experts)}
    per_expert_weights = {e: [] for e in range(model.num_experts)}

    # Fill buckets
    for t in range(B):
        for k in range(model.top_k):
            e = int(topk_idx[t, k].item())
            w = float(weights[t, k].item())
            # Only store positive-weight assignments
            if w > 0.0:
                per_expert_tokens[e].append(t)
                per_expert_weights[e].append(w)

    # Capacity calculation (same formula)
    cap = math.ceil(model.capacity_factor * (model.top_k * B) / float(model.num_experts))

    # Initialize output as encoded (residual base)
    y = encoded.clone()

    # For each expert, apply clamping, pack, compute, and scatter-add residuals
    for e in range(model.num_experts):
        toks = per_expert_tokens[e]
        ws = per_expert_weights[e]
        if len(toks) == 0:
            continue

        toks_tensor = torch.tensor(toks, dtype=torch.long, device=device)
        ws_tensor = torch.tensor(ws, dtype=encoded.dtype, device=device)

        # Capacity clamp: if overflow, keep top-weighted tokens
        if toks_tensor.numel() > cap:
            _, keep_idx = torch.topk(ws_tensor, k=cap, largest=True)
            keep_idx_sorted, _ = torch.sort(keep_idx)  # preserve within-selection order
            toks_tensor = toks_tensor[keep_idx_sorted]
            ws_tensor = ws_tensor[keep_idx_sorted]

        # Pack inputs for this expert
        x_block = encoded[toks_tensor]        # [n_tokens, d]

        # Batched expert compute (residual returned by expert)
        y_block = model.experts[e](x_block)   # [n_tokens, d] (includes residual)
        delta = y_block - x_block             # residual

        # Weighted residual add: y[tokens] += weights * delta
        # Use scatter/add by indexing (same as model.index_add_)
        # Create weighted delta
        weighted_delta = delta * ws_tensor.unsqueeze(-1)  # [n_tokens, d]
        # Accumulate
        y.index_add_(0, toks_tensor, weighted_delta)

    return y


def test_pack_unpack_equivalence(params):
    """Compare naive per-token outputs with the vectorized implementation."""
    B = 12
    model = FlashMoEModel(**params)
    x = torch.randn(B, params["d_model"])  # [B, d]

    y_naive = naive_forward(model, x)
    y_vec = model(x)

    assert torch.allclose(y_naive, y_vec, atol=1e-6), "Vectorized output differs from naive reference"
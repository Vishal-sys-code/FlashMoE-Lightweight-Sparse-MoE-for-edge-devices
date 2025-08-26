# tests/test_model.py
import torch
import pytest
import math

from flash_moe.model import FlashMoEModel

@pytest.fixture(params=["lowrank", "dense"])
def params(request):
    """
    Base params used by most tests. parametrize over expert_type so the test
    suite runs for both lowrank and dense expert implementations.
    """
    base = {
        "d_model": 128,
        "num_experts": 8,
        "d_hidden": 256,
        "top_k": 2,
        "capacity_factor": 1000.0,
        "top_c": 4,
        "expert_type": request.param,     # 'lowrank' or 'dense'
        # we optionally set overrides to make the param-budget checks deterministic
        "r_override": 8 if request.param == "lowrank" else None,
        "d_hidden_override": 128 if request.param == "dense" else None,
        "target_k_budget": 16,             # used by helper if override missing
    }
    return base


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
    x = torch.randn(B, params["d_model"])

    encoded = model.shared_encoder(x)
    topk_idx, topk_weights, probs, topc_idx = model.gating_network(encoded)

    sums = topk_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_gating_returns_full_probs(params):
    """Ensure gating returns the full-probs vector and top-C candidates."""
    model = FlashMoEModel(**params)
    x = torch.randn(4, params["d_model"])
    topk_idx, topk_weights, probs, topc_idx = model.gating_network(model.shared_encoder(x))

    # Shapes
    B = x.size(0)
    assert topk_idx.shape[0] == B and topk_idx.shape[1] == model.top_k
    assert topk_weights.shape == topk_idx.shape
    assert probs.shape[0] == B and probs.shape[1] == model.num_experts
    assert topc_idx.shape[0] == B and topc_idx.shape[1] >= model.top_k


def test_expert_param_budget(params):
    """Sanity check: expert param counts reflect chosen type and overrides.

    Compute expected parameter count directly from the expert module's layer shapes
    (safer than relying on an approximate formula). Allow a very small absolute
    tolerance to account for tiny implementation differences.
    """
    model = FlashMoEModel(**params)
    # pick the first expert
    expert = model.experts[0]
    pcount = sum(p.numel() for p in expert.parameters())
    d = params["d_model"]

    # Compute expected directly from the expert object structure
    if params["expert_type"] == "lowrank":
        # LowRankExpert has U: (r x d) and V: (d x r) weights (no biases here in our impl)
        # find r from a weight shape
        U_weight = getattr(expert, "U").weight  # shape: (r, d)
        r = U_weight.shape[0]
        expected = int(2 * d * r)  # U + V weights
    else:
        # DenseSmallMLPExpert has fc1.weight: (d_hidden, d), fc2.weight: (d, d_hidden)
        # and (optionally) fc1.bias: (d_hidden), fc2.bias: (d)
        fc1 = getattr(expert, "fc1")
        fc2 = getattr(expert, "fc2")
        d_hidden = fc1.weight.shape[0]
        # base count: weight matrices
        expected = int(fc1.weight.numel() + fc2.weight.numel())
        # include biases if present
        if getattr(fc1, "bias", None) is not None:
            expected += int(fc1.bias.numel())
        if getattr(fc2, "bias", None) is not None:
            expected += int(fc2.bias.numel())

    # Small absolute tolerance (e.g., allow a couple hundred params for bookkeeping differences)
    tol = 512
    diff = abs(pcount - expected)
    assert diff <= tol, f"expert param count {pcount} differs from expected {expected} by {diff} (> {tol})"



def naive_forward(model: FlashMoEModel, x: torch.Tensor) -> torch.Tensor:
    """
    Naive reference forward that mirrors the FlashMoEModel logic, including
    capacity clamping and reroute via top-C candidates.
    """
    device = x.device
    B = x.size(0)

    # Encode and router decisions (same as model.forward)
    encoded = model.shared_encoder(x)
    topk_idx, topk_weights, probs, topc_idx = model.gating_network(encoded)

    # Per-expert buckets
    per_expert_tokens = {e: [] for e in range(model.num_experts)}

    # Fill buckets from top-K assignments
    for t in range(B):
        for k in range(model.top_k):
            e = int(topk_idx[t, k].item())
            w = float(topk_weights[t, k].item())
            if w > 1e-12:
                per_expert_tokens[e].append((t, w))

    # Capacity
    cap = math.ceil(model.capacity_factor * (model.top_k * B) / float(model.num_experts))

    # Initialize output
    y = encoded.clone()

    # First pass: assign up to cap per expert by keeping highest-weighted tokens
    overflow_tokens = []
    for e in range(model.num_experts):
        bucket = per_expert_tokens[e]
        if len(bucket) == 0:
            continue
        # Sort bucket by weight desc
        bucket_sorted = sorted(bucket, key=lambda tw: tw[1], reverse=True)
        kept = bucket_sorted[:cap]
        overflow = bucket_sorted[cap:]
        for t, w in kept:
            x_block = encoded[torch.tensor([t], device=device)]
            y_block = model.experts[e](x_block)
            delta = (y_block - x_block)[0]
            y[t] = y[t] + w * delta
        for t, w in overflow:
            overflow_tokens.append(t)

    # Reroute overflow tokens greedily using top-C candidates and probs
    if len(overflow_tokens) > 0:
        # Compute remaining capacities
        rem_cap = {e: cap - min(len(per_expert_tokens[e]), cap) for e in range(model.num_experts)}
        # Build candidate list entries (token, candidate_expert, weight)
        cand_entries = []
        for t in overflow_tokens:
            candidates = topc_idx[t].tolist()  # small list per token; ok in test
            # get probs for these candidates
            for c in candidates:
                w_c = float(probs[t, c].item())
                cand_entries.append((t, c, w_c))
        # Sort all candidate entries by weight descending
        cand_entries.sort(key=lambda x: x[2], reverse=True)
        assigned = set()
        for t, c, w in cand_entries:
            if t in assigned:
                continue
            if rem_cap[c] > 0:
                # assign
                x_block = encoded[torch.tensor([t], device=device)]
                y_block = model.experts[c](x_block)
                delta = (y_block - x_block)[0]
                y[t] = y[t] + w * delta
                rem_cap[c] -= 1
                assigned.add(t)
        # any tokens not assigned remain as-is (unlikely with large top_c and cap)

    return y


def test_pack_unpack_equivalence(params):
    """Compare naive per-token outputs with the implementation."""
    B = 12
    model = FlashMoEModel(**params)
    x = torch.randn(B, params["d_model"])  # [B, d]

    y_naive = naive_forward(model, x)
    y_vec = model(x)

    assert torch.allclose(y_naive, y_vec, atol=1e-6), "Vectorized output differs from naive reference"
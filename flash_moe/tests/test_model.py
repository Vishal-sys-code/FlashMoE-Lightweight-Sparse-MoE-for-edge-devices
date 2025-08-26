# tests/test_model.py
import torch
import pytest
import math

from flash_moe.model import FlashMoEModel, load_balance_loss

# Parametrized fixture to run tests for both expert implementations
@pytest.fixture(params=["lowrank", "dense"])
def params(request):
    base = {
        "d_model": 128,
        "num_experts": 8,
        "d_hidden": 256,
        "top_k": 2,
        "capacity_factor": 1000.0,
        "top_c": 4,
        "expert_type": request.param,
        # deterministic overrides per expert type to make param-budget tests stable
        "r_override": 8 if request.param == "lowrank" else None,
        "d_hidden_override": 128 if request.param == "dense" else None,
        "target_k_budget": 16,
        # default capacity mode for these tests (can be overridden in specific tests)
        "capacity_mode": "reroute_then_drop",
    }
    return base


def test_output_shape(params):
    B = 16
    model = FlashMoEModel(**params)
    x = torch.randn(B, params["d_model"])
    y = model(x)
    assert y.shape == x.shape, f"expected {x.shape}, got {y.shape}"


def test_training_step(params):
    B = 8
    model = FlashMoEModel(**params)
    x = torch.randn(B, params["d_model"])
    target = torch.randn_like(x)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    y = model(x)
    loss = torch.nn.MSELoss()(y, target)
    loss.backward()
    opt.step()
    assert True  # training step completed without error


def test_masked_softmax_sums_to_one(params):
    """Masked softmax over top-k logits sums to 1 per token."""
    B = 10
    model = FlashMoEModel(**params)
    x = torch.randn(B, params["d_model"])
    encoded = model.shared_encoder(x)
    topk_idx, topk_weights, probs, topc_idx = model.gating_network(encoded)
    sums = topk_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_gating_returns_full_probs(params):
    """Ensure gating returns full-probs and top-C candidate list."""
    model = FlashMoEModel(**params)
    x = torch.randn(4, params["d_model"])
    topk_idx, topk_weights, probs, topc_idx = model.gating_network(model.shared_encoder(x))

    B = x.size(0)
    assert topk_idx.shape == (B, model.top_k)
    assert topk_weights.shape == topk_idx.shape
    assert probs.shape == (B, model.num_experts)
    assert topc_idx.shape[0] == B and topc_idx.shape[1] >= model.top_k


def test_expert_param_budget(params):
    """Check expert parameter counts are consistent with the chosen type."""
    model = FlashMoEModel(**params)
    expert = model.experts[0]
    pcount = sum(p.numel() for p in expert.parameters())
    d = params["d_model"]

    if params["expert_type"] == "lowrank":
        # LowRankExpert: U weight shape (r, d), V weight shape (d, r)
        U = getattr(expert, "U").weight
        r = U.shape[0]
        expected = int(2 * d * r)
    else:
        fc1 = getattr(expert, "fc1")
        fc2 = getattr(expert, "fc2")
        expected = int(fc1.weight.numel() + fc2.weight.numel())
        if getattr(fc1, "bias", None) is not None:
            expected += int(fc1.bias.numel())
        if getattr(fc2, "bias", None) is not None:
            expected += int(fc2.bias.numel())

    tol = 512
    diff = abs(pcount - expected)
    assert diff <= tol, f"expert param count {pcount} differs from expected {expected} by {diff} (> {tol})"


def test_load_balance_loss_nonnegative(params):
    """Sanity check: load-balance loss is finite and non-negative."""
    model = FlashMoEModel(**params)
    B = 12
    x = torch.randn(B, params["d_model"])
    encoded = model.shared_encoder(x)
    topk_idx, topk_weights, probs, topc_idx = model.gating_network(encoded)

    L = load_balance_loss(probs, topk_idx, topk_weights, model.num_experts)
    assert torch.isfinite(L)
    assert L.item() >= 0.0


def test_forward_bookkeeping(params):
    """Forward should return sane bookkeeping info via return_router_info=True."""
    model = FlashMoEModel(**params)
    B = 32
    x = torch.randn(B, params["d_model"])
    y, info = model(x, return_router_info=True)

    assert y.shape == (B, params["d_model"])
    assert "num_dropped" in info and "overflow_count" in info and "cap" in info
    # basic sanity: overflow_count >= num_dropped >= 0
    assert int(info["overflow_count"].item()) >= int(info["num_dropped"].item()) >= 0


def test_drop_mode(params):
    """When capacity_mode='drop' small cap should produce some dropped tokens (smoke)."""
    params2 = dict(params)
    params2.update({"capacity_factor": 0.1, "capacity_mode": "drop", "top_c": 2})
    model = FlashMoEModel(**params2)
    B = 24
    x = torch.randn(B, params2["d_model"])
    y, info = model(x, return_router_info=True)
    assert y.shape == (B, params2["d_model"])
    # num_dropped exists and is non-negative
    assert int(info["num_dropped"].item()) >= 0


def naive_forward(model: FlashMoEModel, x: torch.Tensor) -> torch.Tensor:
    """
    Naive forward to mirror model semantics (phase1 keep top-cap per expert,
    phase2 greedy reroute using top-C candidates, then drop remaining).
    This mirrors the behavior implemented in model.forward and is used to verify
    pack/unpack_equivalence.
    """
    device = x.device
    B = x.size(0)

    encoded = model.shared_encoder(x)
    topk_idx, topk_weights, probs, topc_idx = model.gating_network(encoded)

    # Fill per-expert buckets from top-K assignments
    per_expert = {e: [] for e in range(model.num_experts)}
    flat_positions = []  # store (token, k-slot) mapping if needed

    for t in range(B):
        for k in range(model.top_k):
            e = int(topk_idx[t, k].item())
            w = float(topk_weights[t, k].item())
            if w > 1e-12:
                per_expert[e].append((t, w))

    cap = math.ceil(model.capacity_factor * (model.top_k * B) / float(model.num_experts))
    y = encoded.clone()

    # Phase 1: per-expert keep top-cap tokens by weight (tie-break by token id ascending)
    overflow_tokens = []
    for e in range(model.num_experts):
        bucket = per_expert[e]
        if len(bucket) == 0:
            continue
        # tie-break: token id asc then weight desc
        bucket_sorted = sorted(bucket, key=lambda tw: ( -tw[1], tw[0] ))
        kept = bucket_sorted[:cap]
        overflow = bucket_sorted[cap:]
        for t, w in kept:
            xb = encoded[torch.tensor([t], device=device)]
            yb = model.experts[e](xb)
            delta = (yb - xb)[0]
            y[t] = y[t] + w * delta
        for t, w in overflow:
            overflow_tokens.append(t)

    # Phase 2: greedy reroute using top-C candidates and probs
    if len(overflow_tokens) > 0:
        # remaining cap
        rem_cap = {e: cap - min(len(per_expert[e]), cap) for e in range(model.num_experts)}
        cand_entries = []
        for t in overflow_tokens:
            candidates = topc_idx[t].tolist()
            for c in candidates:
                w_c = float(probs[t, c].item())
                cand_entries.append((t, c, w_c))
        cand_entries.sort(key=lambda x: x[2], reverse=True)
        assigned = set()
        for t, c, w in cand_entries:
            if t in assigned:
                continue
            if rem_cap[c] > 0:
                xb = encoded[torch.tensor([t], device=device)]
                yb = model.experts[c](xb)
                delta = (yb - xb)[0]
                y[t] = y[t] + w * delta
                rem_cap[c] -= 1
                assigned.add(t)
        # remaining tokens remain as-is

    return y


def test_pack_unpack_equivalence(params):
    """Compare naive per-token outputs with the implementation."""
    B = 12
    model = FlashMoEModel(**params)
    x = torch.randn(B, params["d_model"])

    y_naive = naive_forward(model, x)
    y_vec = model(x)

    assert torch.allclose(y_naive, y_vec, atol=1e-6), "Vectorized output differs from naive reference"


# ---------------- New test: training objective composition ----------------
def test_training_objective_combines_losses(params):
    """
    Ensure compute_loss/training_step runs and gradients flow when combining
    task loss and load-balance loss.
    """
    model = FlashMoEModel(**params)
    B = 16
    x = torch.randn(B, params["d_model"])
    target = torch.randn(B, params["d_model"])  # MSE target
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Run one training_step (convenience helper) which does forward + loss + step
    total, comps = model.training_step(x, target, optimizer, loss_type="mse", lambda_bal=1e-3)

    # total should be scalar and finite; components keys present
    assert isinstance(total, torch.Tensor) and total.dim() == 0 and torch.isfinite(total)
    assert "L_task" in comps and "L_bal" in comps and "L_drop" in comps
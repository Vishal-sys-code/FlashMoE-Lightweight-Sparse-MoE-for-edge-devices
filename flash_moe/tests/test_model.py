import torch
import pytest

# from ..model import FlashMoEModel
from ...flash_moe.model import FlashMoEModel

@pytest.fixture
def params():
    return {"d_model": 128, "num_experts": 8, "d_hidden": 256, "top_k": 2}


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
    """Reference (naive) per-token forward matching the vectorized semantics.

    This uses the same masked-softmax over top-k and then calls each expert
    on the corresponding encoded token and accumulates the weighted residuals.
    It's intentionally simple and inefficient — used only for numerical checks.
    """
    encoded = model.shared_encoder(x)
    logits = model.gating_network(encoded)
    topk_vals, topk_idx = torch.topk(logits, k=model.top_k, dim=-1)
    weights = model._stable_masked_softmax(topk_vals / model.tau, dim=-1)

    B = x.size(0)
    y = encoded.clone()

    for i in range(B):
        for k in range(model.top_k):
            expert_id = int(topk_idx[i, k].item())
            w = float(weights[i, k].item())
            # call expert on a single example (keep batch dim)
            y_i = model.experts[expert_id](encoded[i : i + 1])[0]
            delta = y_i - encoded[i]
            y[i] = y[i] + w * delta

    return y


def test_pack_unpack_equivalence(params):
    """Compare naive per-token outputs with the vectorized implementation."""
    B = 12
    model = FlashMoEModel(**params)
    x = torch.randn(B, params["d_model"])  # [B, d]

    y_naive = naive_forward(model, x)
    y_vec = model(x)

    assert torch.allclose(y_naive, y_vec, atol=1e-6), "Vectorized output differs from naive reference"
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """A simple residual low-rank-style MLP expert (keeps residual pattern).

    Forward: out = x + delta(x)
    where delta(x) = Linear(relu(Linear(x)))
    """

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden, bias=False),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, d]
        return x + self.net(x)


class FlashMoEModel(nn.Module):
    """FlashMoE minimal model with Top-K masked softmax routing and vectorized dispatch.

    Features implemented:
      - Top-K selection on router logits
      - Masked softmax over selected logits (stable)
      - Vectorized dispatch: sort-pack-grouped compute-scatter
      - Capacity clamping per-expert and overflow bookkeeping
      - Device / dtype safe tensor creation

    Not implemented here (left for next steps):
      - Coarse->fine router
      - Gumbel warmup / ST relaxations
      - Quantization / export
      - Custom Triton/CUDA kernels for even higher perf
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        d_hidden: int,
        top_k: int = 1,
        capacity_factor: float = 1.25,
        tau: float = 1.0,
    ):
        super().__init__()
        assert top_k >= 1 and top_k <= num_experts
        self.d_model = d_model
        self.num_experts = num_experts
        self.d_hidden = d_hidden
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.tau = tau

        # Shared encoder
        self.shared_encoder = nn.Linear(d_model, d_model)

        # Gating network (simple linear head producing logits over experts)
        self.gating_network = nn.Linear(d_model, num_experts)

        # Experts list
        self.experts = nn.ModuleList([Expert(d_model, d_hidden) for _ in range(num_experts)])

        # Bookkeeping / stats
        self.register_buffer("last_overflow_rate", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_overflow_count", torch.tensor(0, dtype=torch.long), persistent=False)

    # --------------------- Utilities ---------------------
    @staticmethod
    def _stable_masked_softmax(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Stable softmax for small vectors. values shape [..., K]."""
        # subtract max for numerical stability
        v_max = values.max(dim=dim, keepdim=True).values
        e = (values - v_max).exp()
        return e / (e.sum(dim=dim, keepdim=True) + 1e-12)

    # --------------------- Forward ---------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Top-K routing and vectorized expert dispatch.

        Args:
            x: [B, d]
        Returns:
            y: [B, d]
        """
        device = x.device
        dtype = x.dtype
        B = x.size(0)

        # Encode input
        encoded = self.shared_encoder(x)  # [B, d]

        # Router logits
        router_logits = self.gating_network(encoded)  # [B, M]

        # Top-K selection from logits (indices and selected logits)
        # topk_vals: [B, K], topk_idx: [B, K]
        topk_vals, topk_idx = torch.topk(router_logits, k=self.top_k, dim=-1)

        # Masked softmax over the selected K logits (apply temperature)
        scaled = topk_vals / float(self.tau)
        topk_weights = self._stable_masked_softmax(scaled, dim=-1)  # [B, K]

        # Prepare flattened assignment lists
        tok_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        exp_idx = topk_idx.reshape(-1)
        w_flat = topk_weights.reshape(-1)

        # Filter tiny weights (optional, keeps things robust if K>actual)
        mask = w_flat > 0
        tok_idx = tok_idx[mask]
        exp_idx = exp_idx[mask]
        w_flat = w_flat[mask]

        if exp_idx.numel() == 0:
            # No assignments (edge-case) - return encoded input directly
            return encoded

        # Sort by expert id so tokens for same expert are contiguous
        order = torch.argsort(exp_idx)
        exp_sorted = exp_idx[order]
        tok_sorted = tok_idx[order]
        w_sorted = w_flat[order]

        # Find group boundaries for each expert (unique consecutive)
        unique_experts, counts = torch.unique_consecutive(exp_sorted, return_counts=True)

        # Capacity calculation
        cap = math.ceil(self.capacity_factor * (self.top_k * B) / float(self.num_experts))

        # Initialize output as encoded (residual base) and zeros for aggregation
        y = encoded.clone()

        overflow_count = 0

        # Iterate only over active experts (small loop since typically few active experts)
        offset = 0
        for expert_id, n_tokens in zip(unique_experts.tolist(), counts.tolist()):
            start = offset
            end = offset + n_tokens
            offset = end

            tokens_for_e = tok_sorted[start:end]  # indices in [0, B)
            weights_for_e = w_sorted[start:end]

            # Capacity clamp: keep top-weighted tokens if overflow
            if n_tokens > cap:
                overflow = n_tokens - cap
                overflow_count += overflow
                # keep top `cap` tokens by weight
                _, keep_idx = torch.topk(weights_for_e, k=cap, largest=True)
                keep_idx_sorted, _ = torch.sort(keep_idx)  # keep original ordering within the selection
                tokens_for_e = tokens_for_e[keep_idx_sorted]
                weights_for_e = weights_for_e[keep_idx_sorted]
                n_tokens = tokens_for_e.numel()

            # Pack inputs for this expert
            x_block = encoded[tokens_for_e]  # [n_tokens, d]

            # Compute expert outputs in batch
            expert = self.experts[expert_id]
            y_block = expert(x_block)  # [n_tokens, d] (this already includes residual)

            # Convert to residual form: delta = y_block - x_block
            delta = y_block - x_block

            # Weighted residual add: y[tokens] += w * delta
            y.index_add_(0, tokens_for_e, delta * weights_for_e.unsqueeze(-1))

        # Bookkeeping: store last overflow rate and count
        total_assignments = int((self.top_k * B))
        self.last_overflow_count = torch.tensor(overflow_count, device=device)
        self.last_overflow_rate = torch.tensor(float(overflow_count) / max(1.0, total_assignments), device=device)

        return y


# --------------------- Smoke test when run directly ---------------------
if __name__ == "__main__":
    # Basic smoke run to validate shapes and a training step
    d_model = 128
    num_experts = 8
    d_hidden = 256
    top_k = 2

    model = FlashMoEModel(d_model=d_model, num_experts=num_experts, d_hidden=d_hidden, top_k=top_k)
    model.train()

    B = 16
    x = torch.randn(B, d_model)
    y = model(x)
    print("output shape:", y.shape)

    # Single training step
    target = torch.randn_like(y)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = F.mse_loss(y, target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("train step ok; last_overflow_rate=", model.last_overflow_rate.item())
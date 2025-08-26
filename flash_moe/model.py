# flash_moe/model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


"""
DenseSmallMLPExpert implements a basic MLP block:
    W2 * ReLU(W1 * x + b1) + b2
with a residual connection. This version prioritizes clarity and works well when
the hidden dimension (d_hidden) is small.

LowRankExpert implements a parameter-efficient alternative:
    x + γ * V * ϕ(U * x)
where U and V are low-rank projections and ϕ is a nonlinearity.
This design uses approximately 2 * d * r parameters.

To meet a parameter budget of O(d^2 / k), set:
    2 * d * r ≈ d^2 / k  ⇒  r ≈ d / (2k)

This makes LowRankExpert a good fit for edge devices and other
resource-constrained environments where efficiency matters.
"""

class DenseSmallMLPExpert(nn.Module):
    """
    Expert: residual MLP with biases.
    Implements: E(x) = x + (W2 * ReLU(W1 x + b1) + b2)

    Args:
      d_model: model dimension (d)
      d_hidden: hidden dim of the small MLP (d_hidden)
      bias: whether to use biases (True)
    """
    def __init__(self, d_model: int, d_hidden: int, bias: bool = True):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.fc1 = nn.Linear(d_model, d_hidden, bias=bias)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d_model, bias=bias)

        # initialization: small weights to help stability (optional)
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc2.weight)
        if bias:
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, d]
        hidden = self.act(self.fc1(x))         # [N, d_hidden]
        out = self.fc2(hidden)                 # [N, d]
        return x + out                         # residual


class LowRankExpert(nn.Module):
    """
    Low-rank residual expert (LoRA-style) for parameter-efficiency.
    Implements: E(x) = x + gamma * V * phi(U x)
    Params ~ 2 * d * r where r << d.

    Args:
      d_model: model dim (d)
      r: low-rank dimension
      act: activation (default SiLU for smoothness), can be ReLU
      gamma_init: initial scale for residual
    """
    def __init__(self, d_model: int, r: int, act: Optional[nn.Module] = None, gamma_init: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.r = r
        self.U = nn.Linear(d_model, r, bias=False)
        self.V = nn.Linear(r, d_model, bias=False)
        self.act = act if act is not None else nn.SiLU()
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init), dtype=torch.float32))

        # init small
        nn.init.kaiming_uniform_(self.U.weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.V.weight, nonlinearity='linear')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, d]
        s = self.act(self.U(x))          # [N, r]
        out = self.V(s)                  # [N, d]
        return x + (self.gamma * out)    # residual scaled


# Helper constructor that chooses appropriate d_hidden relative to d and target k
def choose_dense_hidden(d: int, target_k: Optional[int] = None, d_hidden_override: Optional[int] = None) -> int:
    """
    Compute a reasonable d_hidden for DenseSmallMLPExpert such that the expert param
    count ~ O(d^2 / k) if target_k provided.

    Math:
      Params_dense = d * d_hidden + d_hidden * d = 2 d d_hidden
      We want approx 2 d d_hidden ≈ d^2 / k  => d_hidden ≈ d / (2 k)
    """
    if d_hidden_override is not None:
        return d_hidden_override
    if target_k is None or target_k <= 0:
        # default small bottleneck
        return max(4, d // 8)
    r_est = max(4, d // (2 * max(1, target_k)))
    return int(r_est)


def choose_lowrank_r(d: int, target_k: Optional[int] = None, r_override: Optional[int] = None) -> int:
    """
    Choose low-rank r so that params ≈ 2 d r ≈ d^2 / k  => r ≈ d/(2k)
    """
    if r_override is not None:
        return r_override
    if target_k is None or target_k <= 0:
        return max(4, d // 16)
    r_est = max(2, d // (2 * max(1, target_k)))
    return int(r_est)


class GatingNetwork(nn.Module):
    """Gating network with Top-C candidate selection, then Top-K routing.

    Returns:
      topk_idx: [B, K]
      topk_weights: [B, K]
      probs: [B, M]         # full softmax probabilities (pre-topk)
      topc_idx: [B, C]      # candidate list for reroute if overflow
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int, tau: float = 1.0, top_c: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.tau = tau
        self.top_c = top_c if top_c is not None else min(num_experts, max(4, 2 * top_k))
        # gating produces logits across all experts
        self.gating_network = nn.Linear(d_model, num_experts, bias=False)

    @staticmethod
    def _stable_softmax(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Stable softmax over a full vector (values shape [..., M])."""
        v_max = values.max(dim=dim, keepdim=True).values
        e = (values - v_max).exp()
        return e / (e.sum(dim=dim, keepdim=True) + 1e-12)

    @staticmethod
    def _stable_masked_softmax(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Stable softmax for small vectors. values shape [..., K]."""
        v_max = values.max(dim=dim, keepdim=True).values
        e = (values - v_max).exp()
        return e / (e.sum(dim=dim, keepdim=True) + 1e-12)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, d]
        Returns:
            topk_idx: [B, K] indices of top-k experts (int64)
            topk_weights: [B, K] normalized weights for top-k experts
            probs: [B, M] full softmax probs for balancing losses
            topc_idx: [B, C] candidate indices for rerouting (int64)
        """
        # Router logits (B, M)
        router_logits = self.gating_network(x)  # [B, M]

        # Full softmax probabilities (for balancing losses)
        probs = self._stable_softmax(router_logits / float(self.tau), dim=-1)  # [B, M]

        # Coarse candidate selection: top-C on scaled logits (so tau affects selection)
        topc_vals, topc_idx = torch.topk(router_logits / float(self.tau), k=self.top_c, dim=-1)  # [B, C]

        # Fine selection: pick top-K among the top-C candidates (we already have their logits)
        # topc_vals shape [B, C] - compute masked softmax on those C values then pick TopK
        # Note: selecting top-K from top-C is equivalent to picking the K largest of topc_vals.
        topk_vals, topk_pos_in_C = torch.topk(topc_vals, k=self.top_k, dim=-1)  # values and positions within C
        # Map positions to absolute expert ids
        # topc_idx: [B, C], topk_pos_in_C: [B, K]
        topk_idx = torch.gather(topc_idx, dim=1, index=topk_pos_in_C)  # [B, K]

        # Now compute normalized weights over the chosen K using a stable masked softmax on topk_vals
        topk_weights = self._stable_masked_softmax(topk_vals, dim=-1)  # [B, K]

        return topk_idx, topk_weights, probs, topc_idx


class FlashMoEModel(nn.Module):
    """FlashMoE minimal model with Top-K masked softmax routing and vectorized-ish dispatch.

    Improvements included:
      - Top-C candidate selection + Top-K routing (so we can reroute to next candidates)
      - Masked softmax over selected logits
      - Vectorized grouping (sort -> group -> batch compute -> scatter)
      - Overflow re-routing to next-best candidate (instead of dropping)
      - Exposes full probs for balancing losses
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        d_hidden: int,
        top_k: int = 1,
        capacity_factor: float = 1.25,
        tau: float = 1.0,
        top_c: Optional[int] = None,
        expert_type: str = "lowrank",            # 'lowrank' or 'dense'
        d_hidden_override: Optional[int] = None, # if provided, forces dense hidden dim
        r_override: Optional[int] = None,        # if provided, forces low-rank r
        target_k_budget: Optional[int] = None,   # target_k used in O(d^2/k) budget helper
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

        # Gating network (Top-C candidates, then Top-K)
        self.gating_network = GatingNetwork(d_model, num_experts, top_k, tau, top_c=top_c)

                # ---------------- Expert factory (flexible) ----------------
        # New optional args (add to signature): expert_type='lowrank'|'dense',
        # d_hidden_override=None, r_override=None, target_k_budget=None
        #
        # Explanation of args:
        # - expert_type: 'lowrank' (LoRA-style) or 'dense' (small MLP)
        # - d_hidden_override: if you want to force dense hidden dim
        # - r_override: if you want to force low-rank r
        # - target_k_budget: integer 'k' used in O(d^2/k) budget formulas (optional)
        #
        # Example signature (modify __init__ params accordingly):
        # def __init__(..., d_hidden: int, ..., expert_type: str = 'lowrank',
        #              d_hidden_override: Optional[int] = None,
        #              r_override: Optional[int] = None,
        #              target_k_budget: Optional[int] = None):
        #
        # Use the helpers choose_dense_hidden / choose_lowrank_r above.

        if expert_type not in ('lowrank', 'dense'):
            raise ValueError(f"expert_type must be 'lowrank' or 'dense', got {expert_type}")

        if expert_type == 'dense':
            # choose dense hidden dim if not explicitly provided
            chosen_hidden = choose_dense_hidden(d_model, target_k=target_k_budget, d_hidden_override=d_hidden)
            self.experts = nn.ModuleList(
                [DenseSmallMLPExpert(d_model, chosen_hidden) for _ in range(num_experts)]
            )
        else:  # 'lowrank'
            chosen_r = choose_lowrank_r(d_model, target_k=target_k_budget, r_override=r_override)
            self.experts = nn.ModuleList(
                [LowRankExpert(d_model, chosen_r) for _ in range(num_experts)]
            )

        # Bookkeeping / stats
        # persistent=False so not saved in checkpoints (runtime-only stats)
        self.register_buffer("last_overflow_rate", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_overflow_count", torch.tensor(0, dtype=torch.long), persistent=False)

    # --------------------- Forward ---------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Top-K routing and vectorized-ish expert dispatch.

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

        # Gating decisions - returns candidates for reroute and full probs for balancing
        topk_idx, topk_weights, probs, topc_idx = self.gating_network(encoded)
        # topk_idx: [B, K], topk_weights: [B, K], topc_idx: [B, C]

        # Prepare flattened assignment lists (token index repeated K times)
        tok_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)  # [B*K]
        exp_idx = topk_idx.reshape(-1)   # [B*K]
        w_flat = topk_weights.reshape(-1)  # [B*K]

        # Filter tiny weights (guard against numerical zeros)
        mask = w_flat > 1e-12
        tok_idx = tok_idx[mask]
        exp_idx = exp_idx[mask]
        w_flat = w_flat[mask]

        # If no assignments, return encoded and zero bookkeeping
        total_assignments = int(self.top_k * B)
        if exp_idx.numel() == 0:
            self.last_overflow_count = torch.tensor(0, device=device)
            self.last_overflow_rate = torch.tensor(0.0, device=device)
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

        # Initialize output as encoded (residual base)
        y = encoded.clone()

        overflow_count = 0

        # We'll iterate over *active* experts only. The number of active experts is typically small.
        offset = 0
        # unique_experts is a 1D tensor of active expert ids (ascending)
        num_active = unique_experts.size(0)
        for i in range(num_active):
            expert_id_tensor = unique_experts[i]  # tensor scalar on device
            n_tokens = int(counts[i].item())
            start = offset
            end = offset + n_tokens
            offset = end

            # Convert to python int to index ModuleList (cheap: small loop over active experts)
            expert_id = int(expert_id_tensor.item())

            tokens_for_e = tok_sorted[start:end]  # indices in [0, B)
            weights_for_e = w_sorted[start:end]

            # Capacity clamp: if overflow, re-route excess tokens to next-best candidates
            if n_tokens > cap:
                overflow = n_tokens - cap
                overflow_count += overflow

                # We want to keep the top-`cap` tokens by weight for this expert.
                # Keep indices of the kept tokens
                _, keep_idx = torch.topk(weights_for_e, k=cap, largest=True)
                keep_idx_sorted, _ = torch.sort(keep_idx)  # preserve order within kept set
                kept_tokens = tokens_for_e[keep_idx_sorted]
                kept_weights = weights_for_e[keep_idx_sorted]

                # Determine overflow tokens (those not kept)
                mask_kept = torch.zeros(n_tokens, dtype=torch.bool, device=device)
                mask_kept[keep_idx] = True
                overflow_mask = ~mask_kept
                overflow_tokens = tokens_for_e[overflow_mask]
                overflow_weights = weights_for_e[overflow_mask]

                # Assign kept tokens to this expert (pack & compute below)
                tokens_for_e = kept_tokens
                weights_for_e = kept_weights
                n_tokens = tokens_for_e.numel()

                # --- REROUTE overflow tokens to next-best candidates from top-C list ---
                # For each overflow token t, find the next best candidate index (in topc_idx)
                # that is not the current expert and attempt to assign it there.
                if overflow_tokens.numel() > 0:
                    # topc_idx: [B, C] on device
                    # for each overflow token t we have a candidate list; iterate in python
                    # (overflow_count small typically). We'll append these rerouted assignments
                    # to a lists to be processed after main loop.
                    # To avoid complicated in-loop mutation of sorted arrays, collect reroutes.
                    # We'll build lists reroute_tokens, reroute_experts, reroute_weights
                    # Note: weight used for rerouted assignment can be taken from the original
                    # topC probabilities for that token. We'll use the probs vector to find
                    # next-best candidate's weight.
                    # (we use python loop because overflow tokens are expected to be few)
                    pass  # we handle reroutes after main loop for simplicity

            # Pack inputs for this expert
            if n_tokens == 0:
                continue
            x_block = encoded[tokens_for_e]  # [n_tokens, d]

            # Compute expert outputs in batch
            expert = self.experts[expert_id]
            y_block = expert(x_block)  # [n_tokens, d] (includes residual)

            # Convert to residual form: delta = y_block - x_block
            delta = y_block - x_block

            # Weighted residual add: y[tokens] += w * delta
            y.index_add_(0, tokens_for_e, delta * weights_for_e.unsqueeze(-1))

        # --- Handle reroutes if any overflow occurred ---
        # For simplicity and determinism: compute reroutes in a separate pass using top-C candidates.
        # Build reroute assignment lists
        if overflow_count > 0:
            reroute_tok_list = []
            reroute_exp_list = []
            reroute_weight_list = []

            # Reconstruct assignments to search for overflow tokens:
            # For each token t, check how many times it was assigned originally and whether it
            # was kept for that expert. If any overflow tokens exist, reassign them to next best.
            # Simpler approach: iterate tokens and check their assigned experts from topk_idx and whether
            # they ended up in final output (we know y was updated for kept tokens). Because we didn't
            # store a direct mask of kept tokens per expert, we'll recompute per-token reserved slot counts:
            # For each token t: scan its top-K candidates and weights; for the first candidate that still
            # has capacity available (we recompute remaining slots per expert), assign token there.
            #
            # This is more straightforward: compute remaining capacities per expert and fill them by
            # scanning tokens in decreasing order of their topk weight.
            remaining_cap = torch.full((self.num_experts,), cap, dtype=torch.long, device=device)
            # Deduct what we've already assigned (quick scan over unique_experts counts kept)
            offset = 0
            for i in range(num_active):
                expert_id = int(unique_experts[i].item())
                assigned = int(counts[i].item())
                # Some of these were overflowed and not actually assigned; approximate by min(assigned, cap)
                assigned_kept = min(assigned, cap)
                remaining_cap[expert_id] = max(0, cap - assigned_kept)

            # Build a list of candidate assignment tuples (token, candidate_expert, weight) for all tokens,
            # sorted by decreasing weight so high-weight tokens get priority to fill remaining slots.
            # We'll consider each token's candidates in order provided by topc_idx and use probs to set weights.
            B_range = torch.arange(B, device=device)
            # For each token, get candidate experts and their probs
            topc = topc_idx  # [B, C]
            # Use the precomputed probs (softmax over all experts) to get candidate weights
            # Extract candidate probs by gather
            topc_probs = torch.gather(probs, 1, topc)  # [B, C]

            # Create flattened candidate list (B*C entries)
            token_vals = B_range.unsqueeze(1).expand(-1, topc.size(1)).reshape(-1)  # [B*C]
            cand_exps = topc.reshape(-1)  # [B*C]
            cand_weights = topc_probs.reshape(-1)  # [B*C]

            # Filter out candidates that correspond to experts that already have full capacity
            # But we want to fill slots fairly: sort all candidate entries by weight descending and assign greedily.
            order_cand = torch.argsort(cand_weights, descending=True)
            token_vals = token_vals[order_cand]
            cand_exps = cand_exps[order_cand]
            cand_weights = cand_weights[order_cand]

            # Track which tokens already assigned in reroute pass
            token_assigned_mask = torch.zeros(B, dtype=torch.bool, device=device)

            for t_val, e_val, w_val in zip(token_vals.tolist(), cand_exps.tolist(), cand_weights.tolist()):
                if token_assigned_mask[t_val]:
                    continue
                e_int = int(e_val)
                if remaining_cap[e_int].item() <= 0:
                    continue
                # Assign token t_val to expert e_int
                token_assigned_mask[t_val] = True
                remaining_cap[e_int] -= 1
                reroute_tok_list.append(t_val)
                reroute_exp_list.append(e_int)
                reroute_weight_list.append(float(w_val))

            # Now perform grouped compute for reroutes by grouping reroute lists per expert
            if len(reroute_tok_list) > 0:
                reroute_tok_tensor = torch.tensor(reroute_tok_list, dtype=torch.long, device=device)
                reroute_exp_tensor = torch.tensor(reroute_exp_list, dtype=torch.long, device=device)
                reroute_w_tensor = torch.tensor(reroute_weight_list, dtype=dtype, device=device)

                # Sort by expert id to group
                order_r = torch.argsort(reroute_exp_tensor)
                reroute_exp_sorted = reroute_exp_tensor[order_r]
                reroute_tok_sorted = reroute_tok_tensor[order_r]
                reroute_w_sorted = reroute_w_tensor[order_r]

                unique_r_exps, r_counts = torch.unique_consecutive(reroute_exp_sorted, return_counts=True)
                off = 0
                for i in range(unique_r_exps.numel()):
                    rid = int(unique_r_exps[i].item())
                    cnt = int(r_counts[i].item())
                    s = off
                    e = off + cnt
                    off = e
                    toks = reroute_tok_sorted[s:e]
                    ws = reroute_w_sorted[s:e]
                    x_block = encoded[toks]
                    expert = self.experts[rid]
                    y_block = expert(x_block)
                    delta = y_block - x_block
                    y.index_add_(0, toks, delta * ws.unsqueeze(-1))

        # Bookkeeping: store last overflow rate and count
        self.last_overflow_count = torch.tensor(overflow_count, device=device)
        self.last_overflow_rate = torch.tensor(float(overflow_count) / max(1.0, total_assignments), device=device)

        return y


# --------------------- Smoke test when run directly ---------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    d_model = 128
    num_experts = 8
    d_hidden = 256
    top_k = 2
    B = 16

    print("=== Smoke: LowRank experts (default) ===")
    model_lr = FlashMoEModel(
        d_model=d_model,
        num_experts=num_experts,
        d_hidden=d_hidden,
        top_k=top_k,
        expert_type='lowrank',   # explicit, but this is the default in our design
        r_override=8             # example low-rank size
    )
    model_lr.train()
    x = torch.randn(B, d_model)
    y = model_lr(x)
    print("output shape (lowrank):", y.shape)
    print("example expert param count:", sum(p.numel() for p in model_lr.experts[0].parameters()))
    # one training step
    target = torch.randn_like(y)
    opt = torch.optim.Adam(model_lr.parameters(), lr=1e-3)
    loss = F.mse_loss(y, target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("train step ok; last_overflow_rate (lowrank) =", model_lr.last_overflow_rate.item())
    print()

    print("=== Smoke: Dense small-MLP experts ===")
    model_dense = FlashMoEModel(
        d_model=d_model,
        num_experts=num_experts,
        d_hidden=d_hidden,
        top_k=top_k,
        expert_type='dense',
        d_hidden_override=128   # example hidden dim for dense small-MLP expert
    )
    model_dense.train()
    y2 = model_dense(x)  # reuse x to compare deterministically
    print("output shape (dense):", y2.shape)
    print("example expert param count (dense):", sum(p.numel() for p in model_dense.experts[0].parameters()))
    # one training step
    target2 = torch.randn_like(y2)
    opt2 = torch.optim.Adam(model_dense.parameters(), lr=1e-3)
    loss2 = F.mse_loss(y2, target2)
    opt2.zero_grad()
    loss2.backward()
    opt2.step()
    print("train step ok; last_overflow_rate (dense) =", model_dense.last_overflow_rate.item())
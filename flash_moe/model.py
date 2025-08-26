# flash_moe/model.py
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- Experts ----------------
class DenseSmallMLPExpert(nn.Module):
    """Dense small MLP expert: E(x) = x + W2(ReLU(W1 x + b1)) + b2 (residual)."""

    def __init__(self, d_model: int, d_hidden: int, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden, bias=bias)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d_model, bias=bias)

        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.fc2.weight)
        if bias:
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        out = self.fc2(h)
        return x + out


class LowRankExpert(nn.Module):
    """Low-rank residual expert: E(x) = x + gamma * V(phi(U x))."""

    def __init__(self, d_model: int, r: int, act: Optional[nn.Module] = None, gamma_init: float = 1.0):
        super().__init__()
        self.U = nn.Linear(d_model, r, bias=False)
        self.V = nn.Linear(r, d_model, bias=False)
        self.act = act if act is not None else nn.SiLU()
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init), dtype=torch.float32))

        nn.init.kaiming_uniform_(self.U.weight, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.V.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.act(self.U(x))
        out = self.V(s)
        return x + (self.gamma * out)


# ---------------- Helpers for choosing sizes ----------------
def choose_dense_hidden(d: int, target_k: Optional[int] = None, d_hidden_override: Optional[int] = None) -> int:
    if d_hidden_override is not None:
        return int(d_hidden_override)
    if target_k is None or target_k <= 0:
        return max(8, d // 8)
    return max(4, int(d // (2 * max(1, target_k))))


def choose_lowrank_r(d: int, target_k: Optional[int] = None, r_override: Optional[int] = None) -> int:
    if r_override is not None:
        return int(r_override)
    if target_k is None or target_k <= 0:
        return max(4, d // 16)
    return max(2, int(d // (2 * max(1, target_k))))


# ---------------- Gating network ----------------
class GatingNetwork(nn.Module):
    """
    Gating network: returns (topk_idx, topk_weights, probs, topc_idx)
      - topk_idx: [B, K] int64
      - topk_weights: [B, K] float (masked softmax)
      - probs: [B, M] full softmax over all experts (scaled by tau)
      - topc_idx: [B, C] candidate list for reroute
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int, tau: float = 1.0, top_c: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.tau = tau
        self.top_c = top_c if top_c is not None else min(num_experts, max(4, 2 * top_k))
        self.gating_network = nn.Linear(d_model, num_experts, bias=False)

    @staticmethod
    def _stable_softmax(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
        vmax = values.max(dim=dim, keepdim=True).values
        e = (values - vmax).exp()
        return e / (e.sum(dim=dim, keepdim=True) + 1e-12)

    @staticmethod
    def _stable_masked_softmax(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
        vmax = values.max(dim=dim, keepdim=True).values
        e = (values - vmax).exp()
        return e / (e.sum(dim=dim, keepdim=True) + 1e-12)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # router logits
        router_logits = self.gating_network(x)  # [B, M]
        scaled_logits = router_logits / float(self.tau)

        # full probs for balancing
        probs = self._stable_softmax(scaled_logits, dim=-1)  # [B, M]

        # top-C candidates on scaled logits
        topc_vals, topc_idx = torch.topk(scaled_logits, k=self.top_c, dim=-1)  # [B, C]

        # top-K among top-C
        topk_vals, topk_pos_in_C = torch.topk(topc_vals, k=self.top_k, dim=-1)  # [B, K]
        topk_idx = torch.gather(topc_idx, dim=1, index=topk_pos_in_C)  # [B, K]

        # weights over the chosen K
        topk_weights = self._stable_masked_softmax(topk_vals, dim=-1)  # [B, K]

        return topk_idx, topk_weights, probs, topc_idx


# ---------------- Load-balance loss (Switch-style) ----------------
def load_balance_loss(probs: torch.Tensor,
                      topk_idx: torch.Tensor,
                      topk_weights: torch.Tensor,
                      num_experts: int,
                      eps: float = 1e-12) -> torch.Tensor:
    """
    L = M * sum_i f_i * p_i
    where f_i = fraction of routed mass to expert i
          p_i = mean_t probs[t,i]
    """
    B = probs.size(0)
    M = num_experts
    device = probs.device
    dtype = probs.dtype

    K = topk_idx.size(1)
    flat_idx = topk_idx.reshape(-1)  # [B*K]
    flat_w = topk_weights.reshape(-1)

    mask = flat_w > eps
    if mask.sum() == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    flat_idx = flat_idx[mask]
    flat_w = flat_w[mask]

    hat_ell = torch.zeros((M,), device=device, dtype=dtype)
    hat_ell = hat_ell.index_add(0, flat_idx, flat_w)

    total_mass = hat_ell.sum().clamp_min(eps)
    f = hat_ell / total_mass

    p_bar = probs.mean(dim=0).clamp_min(eps)

    f = f.clamp_min(eps)

    L = float(M) * torch.dot(f, p_bar)
    return L


# ---------------- FlashMoEModel ----------------
class FlashMoEModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        d_hidden: int,
        top_k: int = 1,
        capacity_factor: float = 1.25,
        tau: float = 1.0,
        top_c: Optional[int] = None,
        # expert selection
        expert_type: str = "lowrank",            # 'lowrank' or 'dense'
        d_hidden_override: Optional[int] = None,
        r_override: Optional[int] = None,
        target_k_budget: Optional[int] = None,
        # capacity behavior
        capacity_mode: str = "reroute_then_drop",  # 'reroute_then_drop', 'drop', 'none'
        drop_penalty: float = 0.0,
        # misc
        return_router_info_default: bool = False,
    ):
        super().__init__()
        assert top_k >= 1 and top_k <= num_experts
        if capacity_mode not in ("reroute_then_drop", "drop", "none"):
            raise ValueError("capacity_mode must be in {'reroute_then_drop','drop','none'}")

        self.d_model = d_model
        self.num_experts = num_experts
        self.d_hidden = d_hidden
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.tau = tau
        self.top_c = top_c if top_c is not None else min(num_experts, max(4, 2 * top_k))
        self.capacity_mode = capacity_mode
        self.drop_penalty = float(drop_penalty)
        self.return_router_info_default = bool(return_router_info_default)

        # Shared encoder
        self.shared_encoder = nn.Linear(d_model, d_model)

        # Gating network
        self.gating_network = GatingNetwork(d_model, num_experts, top_k, tau, top_c=self.top_c)

        # Expert factory
        if expert_type not in ("lowrank", "dense"):
            raise ValueError("expert_type must be 'lowrank' or 'dense'")
        if expert_type == "dense":
            chosen_hidden = choose_dense_hidden(d_model, target_k=target_k_budget, d_hidden_override=d_hidden_override or d_hidden)
            self.experts = nn.ModuleList([DenseSmallMLPExpert(d_model, chosen_hidden) for _ in range(num_experts)])
        else:
            chosen_r = choose_lowrank_r(d_model, target_k=target_k_budget, r_override=r_override)
            self.experts = nn.ModuleList([LowRankExpert(d_model, chosen_r) for _ in range(num_experts)])

        # Bookkeeping
        self.register_buffer("last_overflow_rate", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_overflow_count", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer("last_dropped_count", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer("last_drop_rate", torch.tensor(0.0), persistent=False)

    def forward(self, x: torch.Tensor, return_router_info: Optional[bool] = None):
        """
        Forward with strict capacity enforcement (depending on capacity_mode).
        If return_router_info True, returns (y, info_dict) else returns y.
        """
        if return_router_info is None:
            return_router_info = self.return_router_info_default

        device = x.device
        dtype = x.dtype
        B = x.size(0)
        M = self.num_experts
        K = self.top_k
        C = self.top_c

        # Encode
        encoded = self.shared_encoder(x)  # [B, d]

        # Router
        topk_idx, topk_weights, probs, topc_idx = self.gating_network(encoded)  # [B,K],[B,K],[B,M],[B,C]

        # Flatten assignments
        tok_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, K).reshape(-1)  # [B*K]
        exp_idx = topk_idx.reshape(-1)  # [B*K]
        w_flat = topk_weights.reshape(-1)  # [B*K]

        # Filter tiny weights
        keep_mask_flat = w_flat > 1e-12
        if keep_mask_flat.sum() == 0:
            # nothing routed
            self.last_overflow_count = torch.tensor(0, device=device)
            self.last_overflow_rate = torch.tensor(0.0, device=device)
            self.last_dropped_count = torch.tensor(0, device=device)
            self.last_drop_rate = torch.tensor(0.0, device=device)
            if return_router_info:
                info = {
                    "probs": probs,
                    "topk_idx": topk_idx,
                    "topk_weights": topk_weights,
                    "topc_idx": topc_idx,
                    "cap": 0,
                    "num_dropped": torch.tensor(0, device=device),
                    "overflow_count": torch.tensor(0, device=device),
                }
                return encoded, info
            return encoded

        tok_idx = tok_idx[keep_mask_flat]
        exp_idx = exp_idx[keep_mask_flat]
        w_flat = w_flat[keep_mask_flat]

        total_assignments = int(B * K)

        # Sort by expert id to make groups contiguous
        order = torch.argsort(exp_idx)
        exp_sorted = exp_idx[order]
        tok_sorted = tok_idx[order]
        w_sorted = w_flat[order]

        # Unique consecutive experts and counts
        unique_experts, counts = torch.unique_consecutive(exp_sorted, return_counts=True)
        num_active = unique_experts.size(0)

        # Capacity
        cap = math.ceil(self.capacity_factor * (K * B) / float(M))

        # Prepare output
        y = encoded.clone()

        # Track overflow tokens per expert (list of tensors)
        overflow_tokens_per_expert = []
        overflow_count = 0

        # Track assigned_count per expert to compute remaining_cap later
        assigned_count = torch.zeros((M,), dtype=torch.long, device=device)

        # Phase 1: per-expert keep top-cap tokens
        offset = 0
        for i in range(num_active):
            e = int(unique_experts[i].item())
            n = int(counts[i].item())
            s = offset
            t = offset + n
            offset = t

            toks = tok_sorted[s:t]   # [n]
            ws = w_sorted[s:t]      # [n]

            if n <= cap:
                # keep all
                assigned_count[e] += n
                if n > 0:
                    xb = encoded[toks]
                    expert = self.experts[e]
                    yb = expert(xb)
                    delta = yb - xb
                    y.index_add_(0, toks, delta * ws.unsqueeze(-1))
            else:
                # overflow: keep top cap by weight (tie-break by token id ascending)
                overflow_count += (n - cap)
                # compute order with stable sorts: first token id asc, then weight desc
                idx_tok = torch.argsort(toks, stable=True)
                toks = toks[idx_tok]
                ws = ws[idx_tok]
                idx_w = torch.argsort(-ws, stable=True)
                toks = toks[idx_w]
                ws = ws[idx_w]

                kept_toks = toks[:cap]
                kept_ws = ws[:cap]
                overflow_toks = toks[cap:]
                # compute for kept
                assigned_count[e] += kept_toks.numel()
                if kept_toks.numel() > 0:
                    xb = encoded[kept_toks]
                    expert = self.experts[e]
                    yb = expert(xb)
                    delta = yb - xb
                    y.index_add_(0, kept_toks, delta * kept_ws.unsqueeze(-1))
                # collect overflow tokens
                overflow_tokens_per_expert.append(overflow_toks)

        # If no overflow, finalize
        if overflow_count == 0:
            self.last_overflow_count = torch.tensor(0, device=device)
            self.last_overflow_rate = torch.tensor(0.0, device=device)
            self.last_dropped_count = torch.tensor(0, device=device)
            self.last_drop_rate = torch.tensor(0.0, device=device)
            if return_router_info:
                info = {
                    "probs": probs,
                    "topk_idx": topk_idx,
                    "topk_weights": topk_weights,
                    "topc_idx": topc_idx,
                    "cap": cap,
                    "num_dropped": torch.tensor(0, device=device),
                    "overflow_count": torch.tensor(0, device=device),
                }
                return y, info
            return y

        # Phase 2: greedy reroute across Top-C candidates
        # Build all overflow tokens list
        if len(overflow_tokens_per_expert) > 0:
            overflow_tokens_all = torch.cat(overflow_tokens_per_expert, dim=0)  # [N_over]
        else:
            overflow_tokens_all = torch.empty((0,), dtype=torch.long, device=device)

        # remaining capacity per expert
        remaining_cap = torch.full((M,), cap, dtype=torch.long, device=device)
        # subtract what has been assigned in phase1
        remaining_cap -= assigned_count
        remaining_cap = remaining_cap.clamp_min(0)

        num_dropped = 0
        if overflow_tokens_all.numel() > 0 and self.capacity_mode != "none":
            # candidates for overflow tokens
            # topc_idx: [B, C]; pick rows corresponding to overflow tokens
            C = min(C, topc_idx.size(1))
            cand_exps = topc_idx[overflow_tokens_all]  # [N_over, C]
            cand_probs = probs[overflow_tokens_all]     # [N_over, M]
            # gather candidate probs in shape [N_over, C]
            cand_w = torch.gather(cand_probs, 1, cand_exps)  # [N_over, C]

            # flatten and sort candidates by weight desc, then token id asc for determinism
            N_over = overflow_tokens_all.size(0)
            token_expand = overflow_tokens_all.unsqueeze(1).expand(-1, C).reshape(-1)  # [N_over*C]
            cand_exps_flat = cand_exps.reshape(-1)  # [N_over*C]
            cand_w_flat = cand_w.reshape(-1)

            # sort indices by (-weight, token id) -> use argsort twice (stable) for lexicographic
            order_tok = torch.argsort(token_expand, stable=True)
            token_expand = token_expand[order_tok]
            cand_exps_flat = cand_exps_flat[order_tok]
            cand_w_flat = cand_w_flat[order_tok]
            order_w = torch.argsort(-cand_w_flat, stable=True)
            token_expand = token_expand[order_w]
            cand_exps_flat = cand_exps_flat[order_w]
            cand_w_flat = cand_w_flat[order_w]

            # greedy assign
            token_assigned = torch.zeros((B,), dtype=torch.bool, device=device)
            reroute_tok = []
            reroute_exp = []
            reroute_w = []

            # iterate (N_over * C moderate): use python loop for clarity
            for t_val, e_val, w_val in zip(token_expand.tolist(), cand_exps_flat.tolist(), cand_w_flat.tolist()):
                if token_assigned[t_val]:
                    continue
                e_int = int(e_val)
                if remaining_cap[e_int].item() <= 0:
                    continue
                # assign
                token_assigned[t_val] = True
                remaining_cap[e_int] -= 1
                reroute_tok.append(t_val)
                reroute_exp.append(e_int)
                reroute_w.append(float(w_val))

            # perform grouped compute for reroute assignments
            if len(reroute_tok) > 0:
                rt = torch.tensor(reroute_tok, dtype=torch.long, device=device)
                re = torch.tensor(reroute_exp, dtype=torch.long, device=device)
                rw = torch.tensor(reroute_w, dtype=dtype, device=device)

                ord_r = torch.argsort(re)
                re_s = re[ord_r]; rt_s = rt[ord_r]; rw_s = rw[ord_r]
                uniq, r_counts = torch.unique_consecutive(re_s, return_counts=True)
                off = 0
                for i in range(uniq.numel()):
                    eid = int(uniq[i].item())
                    cnt = int(r_counts[i].item())
                    s = off; e = off + cnt; off = e
                    toks = rt_s[s:e]; ws = rw_s[s:e]
                    if toks.numel() == 0:
                        continue
                    xb = encoded[toks]
                    expert = self.experts[eid]
                    yb = expert(xb)
                    delta = yb - xb
                    y.index_add_(0, toks, delta * ws.unsqueeze(-1))

            # count dropped tokens (those overflow tokens which were not assigned)
            # token_assigned mask contains assigned status for tokens in [0..B-1]
            # but we only consider overflow_tokens_all
            assigned_mask_over = token_assigned[overflow_tokens_all]
            num_assigned_over = int(assigned_mask_over.sum().item())
            num_dropped = int(overflow_tokens_all.numel() - num_assigned_over)
        else:
            # no reroute (capacity_mode == 'none' or no overflow tokens)
            if self.capacity_mode == "drop":
                # dropped tokens are overflow tokens
                num_dropped = int(overflow_tokens_all.numel())
            else:
                num_dropped = 0

        # Bookkeeping
        self.last_overflow_count = torch.tensor(overflow_count, device=device)
        self.last_overflow_rate = torch.tensor(float(overflow_count) / max(1.0, total_assignments), device=device)
        self.last_dropped_count = torch.tensor(num_dropped, device=device)
        self.last_drop_rate = torch.tensor(float(num_dropped) / max(1.0, B), device=device)

        if return_router_info:
            info = {
                "probs": probs,
                "topk_idx": topk_idx,
                "topk_weights": topk_weights,
                "topc_idx": topc_idx,
                "cap": cap,
                "num_dropped": torch.tensor(num_dropped, device=device),
                "overflow_count": torch.tensor(overflow_count, device=device),
            }
            return y, info

        return y


# ---------------- Smoke test ----------------
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
        expert_type="lowrank",
        r_override=8,
    )
    model_lr.train()
    x = torch.randn(B, d_model)
    y = model_lr(x)
    print("output shape (lowrank):", y.shape)
    print("example expert param count:", sum(p.numel() for p in model_lr.experts[0].parameters()))
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
        expert_type="dense",
        d_hidden_override=128,
    )
    model_dense.train()
    y2 = model_dense(x)
    print("output shape (dense):", y2.shape)
    print("example expert param count (dense):", sum(p.numel() for p in model_dense.experts[0].parameters()))
    target2 = torch.randn_like(y2)
    opt2 = torch.optim.Adam(model_dense.parameters(), lr=1e-3)
    loss2 = F.mse_loss(y2, target2)
    opt2.zero_grad()
    loss2.backward()
    opt2.step()
    print("train step ok; last_overflow_rate (dense) =", model_dense.last_overflow_rate.item())
# flash_moe/model.py
import os
import math
from typing import Optional, Tuple, Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------- Quantization helpers ----------------
def quantize_tensor(t: torch.Tensor):
    max_val = float(t.abs().max().clamp_min(1e-8))
    scale = max_val / 127.0
    q = torch.quantize_per_tensor(t, scale=scale, zero_point=0, dtype=torch.qint8)
    return q


def dequantize_tensor(q: torch.Tensor) -> torch.Tensor:
    return q.dequantize()


# ---------------- Helpers for expert size choices ----------------
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


# ---------------- Expert implementations ----------------
class DenseSmallMLPExpert(nn.Module):
    """Dense small MLP expert with optional low-rank adapter and optional quantized-eval path."""

    def __init__(self, d_model: int, d_hidden: int, bias: bool = True, use_quant: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden, bias=bias)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d_model, bias=bias)

        # small rank-decomposed adapter (per-expert)
        rank = max(4, d_model // 32)
        self.adapter_U = nn.Linear(d_model, rank, bias=False)
        self.adapter_V = nn.Linear(rank, d_model, bias=False)

        self.use_quant = bool(use_quant)

        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.adapter_U.weight, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.adapter_V.weight, nonlinearity="linear")

    def _dense_forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        out = self.fc2(h)
        out = out + self.adapter_V(self.adapter_U(x))
        return x + out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # training path uses checkpoint for memory savings
        if self.use_quant and not self.training:
            # eval quant path (dequantize locally, no param mutation)
            w1_q = quantize_tensor(self.fc1.weight)
            w2_q = quantize_tensor(self.fc2.weight)
            w1 = dequantize_tensor(w1_q)
            w2 = dequantize_tensor(w2_q)
            b1 = self.fc1.bias.detach() if self.fc1.bias is not None else None
            b2 = self.fc2.bias.detach() if self.fc2.bias is not None else None
            h = F.relu(F.linear(x, w1, b1))
            out = F.linear(h, w2, b2)
            out = out + self.adapter_V(self.adapter_U(x))
            return x + out
        # training path with checkpoint (keep for memory)
        return checkpoint(self._dense_forward, x)

        # ---------------- export-safe forward (traceable) ----------------
    def forward_export(self, x: torch.Tensor) -> torch.Tensor:
        """
        Export-safe forward:
        - deterministic, no Python branch on tensors
        - uses Top-K routing but does NOT perform capacity clamping/reroute
        - groups tokens per expert and calls expert.forward_export when available
          to avoid hitting checkpoint() during tracing.
        """
        device = x.device
        B = x.size(0)
        K = self.top_k

        encoded = self.shared_encoder(x)  # [B, d]
        topk_idx, topk_weights, probs, topc_idx = self.gating_network(encoded)  # tensors only

        tok_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, K).reshape(-1)  # [B*K]
        exp_idx = topk_idx.reshape(-1)  # [B*K]
        w_flat = topk_weights.reshape(-1)  # [B*K]

        mask = w_flat > 1e-12
        tok_idx = tok_idx[mask]
        exp_idx = exp_idx[mask]
        w_flat = w_flat[mask]

        # if nothing to dispatch, return encoded
        if exp_idx.numel() == 0:
            return encoded

        # sort by expert id so tokens for same expert are contiguous
        order = torch.argsort(exp_idx)
        exp_sorted = exp_idx[order]
        tok_sorted = tok_idx[order]
        w_sorted = w_flat[order]

        y = encoded.clone()

        # iterate fixed-range over experts (python int). This is fine for tracing.
        for e in range(self.num_experts):
            matches = exp_sorted == e  # boolean tensor
            pos = torch.nonzero(matches, as_tuple=False).reshape(-1)
            if pos.numel() == 0:
                continue
            toks = tok_sorted[pos]
            ws = w_sorted[pos]
            xb = encoded[toks]

            expert = self.experts[e]
            # prefer export-safe entrypoint if available (no checkpoint)
            if hasattr(expert, "forward_export"):
                yb = expert.forward_export(xb)
            else:
                # fallback (may call checkpoint; undesirable for tracing but kept as a last resort)
                yb = expert(xb)

            delta = yb - xb
            y.index_add_(0, toks, delta * ws.unsqueeze(-1))

        return y



class LowRankExpert(nn.Module):
    """Low-rank expert (LoRA-style) with optional quantized-eval path."""

    def __init__(self, d_model: int, r: int, act: Optional[nn.Module] = None, use_quant: bool = False):
        super().__init__()
        self.U = nn.Linear(d_model, r, bias=False)
        self.V = nn.Linear(r, d_model, bias=False)
        self.act = act if act is not None else nn.SiLU()
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.use_quant = bool(use_quant)

        nn.init.kaiming_uniform_(self.U.weight, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.V.weight, nonlinearity="linear")

    def _lowrank_forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.act(self.U(x))
        out = self.V(s)
        return x + self.gamma * out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quant and not self.training:
            qU = quantize_tensor(self.U.weight)
            qV = quantize_tensor(self.V.weight)
            U_dq = dequantize_tensor(qU)
            V_dq = dequantize_tensor(qV)
            s = self.act(F.linear(x, U_dq, None))
            out = F.linear(s, V_dq, None)
            return x + self.gamma * out
        # training path with checkpoint
        return checkpoint(self._lowrank_forward, x)

    def forward_export(self, x: torch.Tensor) -> torch.Tensor:
        """
        Export-safe forward: no checkpoint, pure tensor ops.
        """
        if self.use_quant and not self.training:
            qU = quantize_tensor(self.U.weight)
            qV = quantize_tensor(self.V.weight)
            U_dq = dequantize_tensor(qU)
            V_dq = dequantize_tensor(qV)
            s = self.act(F.linear(x, U_dq, None))
            out = F.linear(s, V_dq, None)
            return x + self.gamma * out

        s = self.act(self.U(x))
        out = self.V(s)
        return x + self.gamma * out



# ---------------- Gating network (with eval cache) ----------------
class GatingNetwork(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int, tau: float = 1.0, top_c: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.tau = tau
        self.top_c = top_c if top_c is not None else min(num_experts, max(4, 2 * top_k))
        self.gating = nn.Linear(d_model, num_experts, bias=False)
        self._cache: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        m = x.max(dim=dim, keepdim=True).values
        e = (x - m).exp()
        return e / (e.sum(dim=dim, keepdim=True) + 1e-12)

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        cache_key = (B, str(x.device))
        if not self.training and cache_key in self._cache:
            return self._cache[cache_key]

        logits = self.gating(x) / float(self.tau)
        probs = self._stable_softmax(logits, dim=-1)

        topc_vals, topc_idx = torch.topk(logits, k=self.top_c, dim=-1)
        topk_vals, pos_in_c = torch.topk(topc_vals, k=self.top_k, dim=-1)
        topk_idx = torch.gather(topc_idx, 1, pos_in_c)
        topk_weights = self._stable_softmax(topk_vals, dim=-1)

        out = (topk_idx, topk_weights, probs, topc_idx)
        if not self.training:
            self._cache[cache_key] = out
        return out


# ---------------- Load-balance loss ----------------
def load_balance_loss(probs: torch.Tensor,
                      topk_idx: torch.Tensor,
                      topk_weights: torch.Tensor,
                      num_experts: int,
                      eps: float = 1e-12) -> torch.Tensor:
    flat_idx = topk_idx.reshape(-1)
    flat_w = topk_weights.reshape(-1)
    mask = flat_w > eps
    if mask.sum() == 0:
        return torch.tensor(0.0, device=probs.device, dtype=probs.dtype)

    flat_idx = flat_idx[mask]
    flat_w = flat_w[mask]
    hat_ell = torch.zeros((num_experts,), device=probs.device, dtype=probs.dtype)
    hat_ell = hat_ell.index_add(0, flat_idx, flat_w)
    total_mass = hat_ell.sum().clamp_min(eps)
    f = hat_ell / total_mass
    p_bar = probs.mean(dim=0).clamp_min(eps)
    L = float(num_experts) * torch.dot(f, p_bar)
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
        expert_type: str = "lowrank",
        d_hidden_override: Optional[int] = None,
        r_override: Optional[int] = None,
        target_k_budget: Optional[int] = None,
        capacity_mode: str = "reroute_then_drop",
        drop_penalty: float = 0.0,
        use_quant: bool = False,
        return_router_info_default: bool = False,
    ):
        super().__init__()
        assert top_k >= 1 and top_k <= num_experts
        if capacity_mode not in ("reroute_then_drop", "drop", "none"):
            raise ValueError("capacity_mode must be one of 'reroute_then_drop','drop','none'")

        self.d_model = d_model
        self.num_experts = num_experts
        self.d_hidden = d_hidden
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.tau = tau
        self.top_c = top_c if top_c is not None else min(num_experts, max(4, 2 * top_k))
        self.capacity_mode = capacity_mode
        self.drop_penalty = float(drop_penalty)
        self.use_quant = bool(use_quant)
        self.return_router_info_default = bool(return_router_info_default)

        self.shared_encoder = nn.Linear(d_model, d_model)
        self.gating_network = GatingNetwork(d_model, num_experts, top_k, tau, top_c=self.top_c)

        if expert_type not in ("lowrank", "dense"):
            raise ValueError("expert_type must be 'lowrank' or 'dense'")

        if expert_type == "dense":
            chosen_hidden = choose_dense_hidden(d_model, target_k=target_k_budget, d_hidden_override=d_hidden_override or d_hidden)
            self.experts = nn.ModuleList([DenseSmallMLPExpert(d_model, chosen_hidden, use_quant=self.use_quant) for _ in range(num_experts)])
        else:
            chosen_r = choose_lowrank_r(d_model, target_k=target_k_budget, r_override=r_override)
            self.experts = nn.ModuleList([LowRankExpert(d_model, chosen_r, use_quant=self.use_quant) for _ in range(num_experts)])

        self.register_buffer("last_overflow_rate", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_overflow_count", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer("last_dropped_count", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer("last_drop_rate", torch.tensor(0.0), persistent=False)

    # ---------------- training/eval forward (strict capacity, as before) ----------------
    def forward(self, x: torch.Tensor, return_router_info: Optional[bool] = None):
        if return_router_info is None:
            return_router_info = self.return_router_info_default

        device = x.device
        B = x.size(0)
        M = self.num_experts
        K = self.top_k
        C = self.top_c

        encoded = self.shared_encoder(x)
        topk_idx, topk_weights, probs, topc_idx = self.gating_network(encoded)

        tok_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, K).reshape(-1)
        exp_idx = topk_idx.reshape(-1)
        w_flat = topk_weights.reshape(-1)

        keep_mask = w_flat > 1e-12
        if keep_mask.sum() == 0:
            self.last_overflow_count = torch.tensor(0, device=device)
            self.last_overflow_rate = torch.tensor(0.0, device=device)
            self.last_dropped_count = torch.tensor(0, device=device)
            self.last_drop_rate = torch.tensor(0.0, device=device)
            if return_router_info:
                info = {"probs": probs, "topk_idx": topk_idx, "topk_weights": topk_weights, "topc_idx": topc_idx,
                        "cap": 0, "num_dropped": torch.tensor(0, device=device), "overflow_count": torch.tensor(0, device=device)}
                return encoded, info
            return encoded

        tok_idx = tok_idx[keep_mask]
        exp_idx = exp_idx[keep_mask]
        w_flat = w_flat[keep_mask]
        total_assignments = int(B * K)

        order = torch.argsort(exp_idx)
        exp_sorted = exp_idx[order]
        tok_sorted = tok_idx[order]
        w_sorted = w_flat[order]

        unique_experts, counts = torch.unique_consecutive(exp_sorted, return_counts=True)
        num_active = unique_experts.size(0)

        cap = math.ceil(self.capacity_factor * (K * B) / float(M))

        y = encoded.clone()
        overflow_tokens_per_expert = []
        overflow_count = 0
        assigned_count = torch.zeros((M,), dtype=torch.long, device=device)

        offset = 0
        for i in range(num_active):
            e = int(unique_experts[i].item())
            n = int(counts[i].item())
            s = offset
            t = offset + n
            offset = t

            toks = tok_sorted[s:t]
            ws = w_sorted[s:t]

            if n <= cap:
                assigned_count[e] += n
                if n > 0:
                    xb = encoded[toks]
                    yb = self.experts[e](xb)
                    delta = yb - xb
                    y.index_add_(0, toks, delta * ws.unsqueeze(-1))
            else:
                overflow_count += (n - cap)
                idx_tok = torch.argsort(toks, stable=True)
                toks = toks[idx_tok]; ws = ws[idx_tok]
                idx_w = torch.argsort(-ws, stable=True)
                toks = toks[idx_w]; ws = ws[idx_w]

                kept_toks = toks[:cap]
                kept_ws = ws[:cap]
                overflow_toks = toks[cap:]

                assigned_count[e] += kept_toks.numel()
                if kept_toks.numel() > 0:
                    xb = encoded[kept_toks]
                    yb = self.experts[e](xb)
                    delta = yb - xb
                    y.index_add_(0, kept_toks, delta * kept_ws.unsqueeze(-1))
                overflow_tokens_per_expert.append(overflow_toks)

        if overflow_count == 0:
            self.last_overflow_count = torch.tensor(0, device=device)
            self.last_overflow_rate = torch.tensor(0.0, device=device)
            self.last_dropped_count = torch.tensor(0, device=device)
            self.last_drop_rate = torch.tensor(0.0, device=device)
            if return_router_info:
                info = {"probs": probs, "topk_idx": topk_idx, "topk_weights": topk_weights, "topc_idx": topc_idx,
                        "cap": cap, "num_dropped": torch.tensor(0, device=device), "overflow_count": torch.tensor(0, device=device)}
                return y, info
            return y

        if len(overflow_tokens_per_expert) > 0:
            overflow_tokens_all = torch.cat(overflow_tokens_per_expert, dim=0)
        else:
            overflow_tokens_all = torch.empty((0,), dtype=torch.long, device=device)

        remaining_cap = torch.full((M,), cap, dtype=torch.long, device=device)
        remaining_cap -= assigned_count
        remaining_cap = remaining_cap.clamp_min(0)

        num_dropped = 0
        if overflow_tokens_all.numel() > 0 and self.capacity_mode != "none":
            C = min(C, topc_idx.size(1))
            cand_exps = topc_idx[overflow_tokens_all]
            cand_probs = probs[overflow_tokens_all]
            cand_w = torch.gather(cand_probs, 1, cand_exps)

            N_over = overflow_tokens_all.size(0)
            token_expand = overflow_tokens_all.unsqueeze(1).expand(-1, C).reshape(-1)
            cand_exps_flat = cand_exps.reshape(-1)
            cand_w_flat = cand_w.reshape(-1)

            order_tok = torch.argsort(token_expand, stable=True)
            token_expand = token_expand[order_tok]
            cand_exps_flat = cand_exps_flat[order_tok]
            cand_w_flat = cand_w_flat[order_tok]
            order_w = torch.argsort(-cand_w_flat, stable=True)
            token_expand = token_expand[order_w]
            cand_exps_flat = cand_exps_flat[order_w]
            cand_w_flat = cand_w_flat[order_w]

            token_assigned = torch.zeros((B,), dtype=torch.bool, device=device)
            reroute_tok = []
            reroute_exp = []
            reroute_w = []

            for t_val, e_val, w_val in zip(token_expand.tolist(), cand_exps_flat.tolist(), cand_w_flat.tolist()):
                if token_assigned[t_val]:
                    continue
                e_int = int(e_val)
                if remaining_cap[e_int].item() <= 0:
                    continue
                token_assigned[t_val] = True
                remaining_cap[e_int] -= 1
                reroute_tok.append(t_val)
                reroute_exp.append(e_int)
                reroute_w.append(float(w_val))

            if len(reroute_tok) > 0:
                rt = torch.tensor(reroute_tok, dtype=torch.long, device=device)
                re = torch.tensor(reroute_exp, dtype=torch.long, device=device)
                rw = torch.tensor(reroute_w, dtype=torch.float32, device=device)

                ord_r = torch.argsort(re)
                re_s = re[ord_r]; rt_s = rt[ord_r]; rw_s = rw[ord_r]
                uniq, r_counts = torch.unique_consecutive(re_s, return_counts=True)
                off = 0
                for i in range(uniq.numel()):
                    eid = int(uniq[i].item())
                    cnt = int(r_counts[i].item())
                    s = off; e = off + cnt; off = e
                    toks = rt_s[s:e]; ws = rw_s[s:e]
                    xb = encoded[toks]
                    yb = self.experts[eid](xb)
                    delta = yb - xb
                    y.index_add_(0, toks, delta * ws.unsqueeze(-1))

            assigned_mask_over = token_assigned[overflow_tokens_all]
            num_assigned_over = int(assigned_mask_over.sum().item())
            num_dropped = int(overflow_tokens_all.numel() - num_assigned_over)
        else:
            if self.capacity_mode == "drop":
                num_dropped = int(overflow_tokens_all.numel())
            else:
                num_dropped = 0

        self.last_overflow_count = torch.tensor(overflow_count, device=device)
        self.last_overflow_rate = torch.tensor(float(overflow_count) / max(1.0, total_assignments), device=device)
        self.last_dropped_count = torch.tensor(num_dropped, device=device)
        self.last_drop_rate = torch.tensor(float(num_dropped) / max(1.0, B), device=device)

        if return_router_info:
            info = {"probs": probs, "topk_idx": topk_idx, "topk_weights": topk_weights, "topc_idx": topc_idx,
                    "cap": cap, "num_dropped": torch.tensor(num_dropped, device=device),
                    "overflow_count": torch.tensor(overflow_count, device=device)}
            return y, info

        return y

    # ---------------- export-safe forward (traceable) ----------------
    def forward_export(self, x: torch.Tensor) -> torch.Tensor:
        """
        Export-safe forward:
        - deterministic, no Python branch on tensors
        - uses Top-K routing but does NOT perform capacity clamping/reroute
        - vectorized grouping by expert (loops over fixed num_experts)
        """
        device = x.device
        B = x.size(0)
        K = self.top_k

        encoded = self.shared_encoder(x)  # [B, d]
        topk_idx, topk_weights, probs, topc_idx = self.gating_network(encoded)  # tensors only

        tok_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, K).reshape(-1)  # [B*K]
        exp_idx = topk_idx.reshape(-1)  # [B*K]
        w_flat = topk_weights.reshape(-1)  # [B*K]

        mask = w_flat > 1e-12
        tok_idx = tok_idx[mask]
        exp_idx = exp_idx[mask]
        w_flat = w_flat[mask]

        # sort by expert id so tokens per expert are contiguous
        if exp_idx.numel() == 0:
            return encoded

        order = torch.argsort(exp_idx)
        exp_sorted = exp_idx[order]
        tok_sorted = tok_idx[order]
        w_sorted = w_flat[order]

        y = encoded.clone()

        # For exportability we avoid data-dependent python branching on tensors:
        # iterate fixed-range over experts (python int)
        for e in range(self.num_experts):
            # find positions where exp_sorted == e
            matches = exp_sorted == e  # boolean tensor
            # get indices of matches in exp_sorted
            pos = torch.nonzero(matches, as_tuple=False).reshape(-1)
            if pos.numel() == 0:
                continue
            toks = tok_sorted[pos]
            ws = w_sorted[pos]
            xb = encoded[toks]
            yb = self.experts[e](xb)
            delta = yb - xb
            y.index_add_(0, toks, delta * ws.unsqueeze(-1))

        return y

    # ---------------- Training objective helpers ----------------
    def compute_loss(self,
                     y: torch.Tensor,
                     target: torch.Tensor,
                     info: Optional[dict] = None,
                     loss_type: str = "mse",
                     lambda_bal: float = 1e-3,
                     drop_penalty: Optional[float] = None) -> Tuple[torch.Tensor, dict]:
        if loss_type == "mse":
            L_task = F.mse_loss(y, target)
        elif loss_type == "cross_entropy":
            L_task = F.cross_entropy(y, target)
        else:
            raise ValueError("loss_type must be 'mse' or 'cross_entropy'")

        if info is not None and "probs" in info and "topk_idx" in info and "topk_weights" in info:
            L_bal = load_balance_loss(info["probs"], info["topk_idx"], info["topk_weights"], self.num_experts)
        else:
            L_bal = torch.tensor(0.0, device=y.device, dtype=y.dtype)

        if drop_penalty is None:
            drop_penalty = float(self.drop_penalty or 0.0)

        L_drop = torch.tensor(0.0, device=y.device, dtype=y.dtype)
        if drop_penalty > 0.0:
            B = y.size(0)
            drop_frac = float(self.last_dropped_count.item()) / max(1.0, float(B))
            L_drop = torch.tensor(drop_penalty * drop_frac, device=y.device, dtype=y.dtype)

        total = L_task + float(lambda_bal) * L_bal + L_drop
        comps = {"L_task": L_task.detach() if isinstance(L_task, torch.Tensor) else L_task,
                 "L_bal": L_bal.detach() if isinstance(L_bal, torch.Tensor) else L_bal,
                 "L_drop": L_drop.detach() if isinstance(L_drop, torch.Tensor) else L_drop}
        return total, comps

    def training_step(self,
                      x: torch.Tensor,
                      target: torch.Tensor,
                      optimizer: torch.optim.Optimizer,
                      loss_type: str = "mse",
                      lambda_bal: float = 1e-3,
                      drop_penalty: Optional[float] = None) -> Tuple[torch.Tensor, dict]:
        self.train()
        optimizer.zero_grad()
        y, info = self(x, return_router_info=True)
        total, comps = self.compute_loss(y, target, info=info, loss_type=loss_type, lambda_bal=lambda_bal, drop_penalty=drop_penalty)
        total.backward()
        optimizer.step()
        return total.detach(), comps

# ---------------- Evaluation & Export Helpers ----------------
def evaluate_router_topk_accuracy(model: FlashMoEModel, dataloader: Iterable[torch.Tensor], device: torch.device):
    model.eval()
    total_tokens = 0
    hits = 0
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            encoded = model.shared_encoder(x)
            topk_idx, topk_weights, probs, topc_idx = model.gating_network(encoded)
            argmax = torch.argmax(probs, dim=-1)
            B = x.size(0)
            total_tokens += B
            matches = (topk_idx == argmax.unsqueeze(-1)).any(dim=-1)
            hits += int(matches.sum().item())
    rate = float(hits) / float(max(1, total_tokens))
    return {"topk_contains_argmax_rate": rate, "total_tokens": total_tokens}


def estimate_flops_saved(model: FlashMoEModel, dataloader: Iterable[torch.Tensor], device: torch.device):
    model.eval()
    dense_flops_total = 0.0
    moe_flops_total = 0.0
    total_tokens = 0

    def linear_flops(in_f, out_f):
        return 2.0 * in_f * out_f

    d = model.d_model
    d_hidden = model.d_hidden
    dense_encoder = linear_flops(d, d)
    dense_mlp = linear_flops(d, d_hidden) + linear_flops(d_hidden, d)
    dense_per_token = dense_encoder + dense_mlp

    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            B = x.size(0)
            total_tokens += B
            dense_flops_total += dense_per_token * B

            moe_flops = linear_flops(d, d) * B
            moe_flops += linear_flops(d, model.num_experts) * B

            encoded = model.shared_encoder(x)
            topk_idx, topk_weights, probs, topc_idx = model.gating_network(encoded)
            flat_idx = topk_idx.reshape(-1)
            flat_w = topk_weights.reshape(-1)
            mask = flat_w > 1e-12
            if mask.sum() == 0:
                moe_flops_total += moe_flops
                continue
            flat_idx = flat_idx[mask]
            counts = torch.bincount(flat_idx, minlength=model.num_experts).to(torch.long)

            if isinstance(model.experts[0], LowRankExpert):
                r = model.experts[0].U.weight.shape[0]
                expert_flops_per_token = 2.0 * d * r
            else:
                expert_flops_per_token = linear_flops(d, d_hidden) + linear_flops(d_hidden, d)

            for e in range(model.num_experts):
                n = int(counts[e].item())
                moe_flops += expert_flops_per_token * n

            moe_flops_total += moe_flops

    if total_tokens == 0:
        return {"dense_flops": 0.0, "moe_flops": 0.0, "flops_saved_ratio": 0.0}

    flops_saved = dense_flops_total - moe_flops_total
    saved_ratio = float(flops_saved / max(1.0, dense_flops_total))
    return {"dense_flops": dense_flops_total, "moe_flops": moe_flops_total, "flops_saved_ratio": saved_ratio}


def compute_utilization_histogram(model: FlashMoEModel, dataloader: Iterable[torch.Tensor], device: torch.device):
    model.eval()
    M = model.num_experts
    counts = torch.zeros((M,), dtype=torch.long, device=device)
    mass = torch.zeros((M,), dtype=torch.float32, device=device)
    probs_accum = torch.zeros((M,), dtype=torch.float32, device=device)
    total_tokens = 0

    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            B = x.size(0)
            total_tokens += B
            encoded = model.shared_encoder(x)
            topk_idx, topk_weights, probs, topc_idx = model.gating_network(encoded)
            flat_idx = topk_idx.reshape(-1)
            flat_w = topk_weights.reshape(-1)
            mask = flat_w > 1e-12
            if mask.sum() > 0:
                idx = flat_idx[mask]
                w = flat_w[mask]
                mass.index_add_(0, idx, w)
                counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
            probs_accum += probs.sum(dim=0)

    total_mass = float(mass.sum().item()) if mass.sum().item() > 0 else 1.0
    f = (mass / total_mass).cpu().numpy().tolist()
    p_bar = (probs_accum / max(1, total_tokens)).cpu().numpy().tolist()
    counts_list = counts.cpu().numpy().tolist()

    return {"expert_counts": counts_list, "f": f, "p_bar": p_bar, "total_tokens": total_tokens}


def export_torchscript(model: FlashMoEModel, example_input: torch.Tensor, path: str, strict: bool = True):
    """
    Export the model's export-safe forward (forward_export) to TorchScript.
    We use torch.jit.trace_module(model, {'forward_export': example_input}) so that
    module parameters remain parameters (not captured as constants).
    """
    model.eval()
    # make sure input is detached and doesn't require grad
    example_input = example_input.detach()
    if example_input.requires_grad:
        example_input = example_input.clone().detach()

    # Trace the module method (so parameters are preserved as parameters)
    traced = torch.jit.trace_module(model, {'forward_export': example_input}, strict=strict)

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    traced.save(path)
    return path


def export_onnx(model: FlashMoEModel, example_input: torch.Tensor, path: str, opset: int = 13, dynamic_axes: bool = True):
    """
    Export the model's export-safe forward (forward_export) to ONNX.
    Call torch.onnx.export with the bound method model.forward_export (no closure).
    """
    model.eval()
    example_input = example_input.detach()
    if example_input.requires_grad:
        example_input = example_input.clone().detach()

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    dynamic_axis = {"input": {0: "batch_size"}, "output": {0: "batch_size"}} if dynamic_axes else None

    # Export the bound method directly (avoids closure-captured tensors becoming constants)
    torch.onnx.export(model.forward_export,
                      example_input,
                      path,
                      input_names=["input"],
                      output_names=["output"],
                      dynamic_axes=dynamic_axis if dynamic_axes else None,
                      opset_version=opset)
    return path


# ---------------- Smoke test ----------------
if __name__ == "__main__":
    torch.manual_seed(0)
    d_model = 128
    num_experts = 8
    d_hidden = 256
    top_k = 2
    B = 16

    model = FlashMoEModel(d_model=d_model, num_experts=num_experts, d_hidden=d_hidden, top_k=top_k)
    model.train()
    x = torch.randn(B, d_model)
    y = model(x)
    print("output shape:", y.shape)
    target = torch.randn_like(y)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = F.mse_loss(y, target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("train step ok; last_overflow_rate=", model.last_overflow_rate.item())

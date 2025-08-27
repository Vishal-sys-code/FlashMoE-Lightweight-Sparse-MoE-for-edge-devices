import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class DenseExpert(nn.Module):
    """A standard two-layer feed-forward network expert."""
    def __init__(self, d_model: int, ff_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ff_hidden)
        self.fc2 = nn.Linear(ff_hidden, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class LowRankExpert(nn.Module):
    """A low-rank decomposed feed-forward network expert."""
    def __init__(self, d_model: int, ff_hidden: int, r: int):
        super().__init__()
        self.fc1_a = nn.Linear(d_model, r)
        self.fc1_b = nn.Linear(r, ff_hidden)
        self.fc2 = nn.Linear(ff_hidden, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1_b(self.fc1_a(x))))

class FlashMoeLayer(nn.Module):
    """
    FlashMoE Layer: A Mixture-of-Experts layer with top-k gating.
    This implementation follows the principles of papers like Switch Transformers,
    but is simplified for CPU execution. It drops tokens that exceed expert capacity.
    """
    def __init__(self, d_model: int, num_experts: int, top_k: int, ff_hidden: int, r: int, 
                 expert_type: str, capacity_factor: float = 1.25, load_balancing_lambda: float = 1e-2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.load_balancing_lambda = load_balancing_lambda

        self.gate = nn.Linear(d_model, num_experts)
        
        if expert_type == 'dense':
            self.experts = nn.ModuleList([DenseExpert(d_model, ff_hidden) for _ in range(num_experts)])
        elif expert_type == 'lowrank':
            self.experts = nn.ModuleList([LowRankExpert(d_model, ff_hidden, r) for _ in range(num_experts)])
        else:
            raise ValueError(f"Unknown expert type: {expert_type}")

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, d_model = x.shape
        S = batch_size * seq_len
        x = x.view(S, d_model)

        # Gate computation
        router_logits = self.gate(x)
        router_probs = F.softmax(router_logits, dim=1)
        
        # Get top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=1)
        
        # --- Dispatching logic ---
        # This part remains the same for both training and eval
        y = torch.zeros_like(x)
        # Normalize weights over top-k
        top_k_probs_norm = top_k_probs / torch.sum(top_k_probs, dim=1, keepdim=True)

        # Create a flat list of (token_index, expert_index, weight) tuples
        flat_indices = top_k_indices.flatten()
        flat_probs = top_k_probs_norm.flatten()
        token_ids = torch.arange(S, device=x.device).repeat_interleave(self.top_k)

        # Bin tokens to experts respecting capacity
        # This is a simplified, iterative approach. A more parallelizable version would be complex.
        final_output = torch.zeros_like(x)
        for i in range(self.num_experts):
            mask = (top_k_indices == i).any(dim=1)
            token_indices_for_expert = torch.where(mask)[0]
            
            if token_indices_for_expert.numel() == 0:
                continue

            capacity = int(self.capacity_factor * S / self.num_experts)
            if token_indices_for_expert.numel() > capacity:
                token_indices_for_expert = token_indices_for_expert[:capacity]

            expert_input = x[token_indices_for_expert]
            expert_output = self.experts[i](expert_input)
            
            # Find the weights for these tokens for this expert
            route_probs_for_expert = torch.zeros(len(token_indices_for_expert), device=x.device)
            for k in range(self.top_k):
                is_kth_choice = (top_k_indices[token_indices_for_expert, k] == i)
                route_probs_for_expert[is_kth_choice] = top_k_probs_norm[token_indices_for_expert[is_kth_choice], k]

            final_output.index_add_(0, token_indices_for_expert, expert_output * route_probs_for_expert.unsqueeze(1))

        # --- Loss and Stats ---
        aux_loss = 0.0
        router_stats = None
        if self.training:
            # Simplified load balancing loss
            tokens_per_expert_count = torch.bincount(top_k_indices.flatten(), minlength=self.num_experts)
            router_prob_per_expert_mean = router_probs.mean(dim=0)
            aux_loss = self.load_balancing_lambda * self.num_experts * \
                       torch.sum(router_prob_per_expert_mean * tokens_per_expert_count.float() / S)
        else:
            # In eval mode, we collect stats instead of loss
            router_stats = {
                'router_probs': router_probs.detach().cpu(),
                'top_k_indices': top_k_indices.detach().cpu()
            }

        return final_output.view(batch_size, seq_len, d_model), aux_loss, router_stats

# This is a workaround for nn.TransformerEncoderLayer not having a simple way to replace the FFN
class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, ffn_module, dropout=0.1):
        # We have to call super().__init__ with some placeholder for dim_feedforward
        super().__init__(d_model, nhead, d_model * 2, dropout, batch_first=True)
        self.feed_forward = ffn_module

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        self.last_aux_loss = 0.0
        self.last_router_stats = None
        
        if hasattr(self, 'feed_forward'):
            # If the ffn_module is a MoE layer, it might return more outputs
            if isinstance(self.feed_forward, FlashMoeLayer):
                x, aux_loss, router_stats = self.feed_forward(x)
                self.last_aux_loss = aux_loss
                self.last_router_stats = router_stats
            else:
                x = self.feed_forward(x)
        
        return self.dropout2(x)

class LanguageModel(nn.Module):
    """
    A more robust Language Model that properly handles custom FFN modules
    and auxiliary losses from MoE layers.
    """
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int,
                 dropout: float, ffn_module_factory):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        self.layers = nn.ModuleList(
            [CustomTransformerEncoderLayer(d_model, n_heads, ffn_module_factory(), dropout) for _ in range(n_layers)]
        )
        self.decoder = nn.Linear(d_model, vocab_size)
    
    def forward(self, src: torch.Tensor):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src.transpose(0,1)).transpose(0,1)

        mask = nn.Transformer.generate_square_subsequent_mask(src.size(1)).to(src.device)
        
        total_aux_loss = 0.0
        all_router_stats = []
        for layer in self.layers:
            src = layer(src, src_mask=mask)
            if hasattr(layer, 'last_aux_loss'):
                total_aux_loss += layer.last_aux_loss
            if hasattr(layer, 'last_router_stats') and layer.last_router_stats is not None:
                all_router_stats.append(layer.last_router_stats)
        
        decoded = self.decoder(src)
        
        # For training, the model in wikitext2_train.py expects a 2-tuple.
        # For eval, we can return more. But this complicates things.
        # A better way is to unify the return signature.
        # Let's always return a 3-tuple, with the last item being None during training.
        if self.training:
             # The training script is hardcoded to expect a 2-tuple.
             # I will modify the training script to handle a 3-tuple.
             # For now, let's stick to the plan of modifying the model first.
             pass

        return decoded, total_aux_loss, all_router_stats

    def forward_export(self, src: torch.Tensor):
        """A simplified forward for exporting to TorchScript/ONNX."""
        return self.forward(src)[0]

def create_model(args, tokenizer):
    """Factory function to create the right model based on args."""
    from scripts.utils import calculate_dense_hidden_dim_for_flops_match, choose_lowrank_r

    ffn_module_factory = None
    if args.model == 'dense':
        # Calculate the hidden dim for the dense model to match FlashMoE FLOPs
        r = choose_lowrank_r(args.d_model)
        dense_ffn_hidden = calculate_dense_hidden_dim_for_flops_match(
            d_model=args.d_model,
            ff_hidden=args.ff_hidden, # The ff_hidden for the MoE model
            num_experts=args.num_experts,
            top_k=args.top_k,
            r=r,
            expert_type=args.expert_type
        )
        print(f"Dense model FFN hidden size calculated to match FLOPs: {dense_ffn_hidden}")
        ffn_module_factory = lambda: DenseExpert(args.d_model, dense_ffn_hidden)
    
    elif args.model in ['flashmoe', 'switch']:
        top_k = 1 if args.model == 'switch' else args.top_k
        r = choose_lowrank_r(args.d_model) if args.expert_type == 'lowrank' else 0
        
        ffn_module_factory = lambda: FlashMoeLayer(
            d_model=args.d_model,
            num_experts=args.num_experts,
            top_k=top_k,
            ff_hidden=args.ff_hidden,
            r=r,
            expert_type=args.expert_type,
            capacity_factor=args.capacity_factor,
            load_balancing_lambda=args.load_balancing_lambda
        )

    model = LanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=0.1,
        ffn_module_factory=ffn_module_factory
    )
    
    # Initialize weights
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return model

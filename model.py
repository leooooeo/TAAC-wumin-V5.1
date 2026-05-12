"""PCVRHyFormer: A hybrid transformer model for post-click conversion rate prediction."""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, NamedTuple, Tuple, Optional, Union

from dataset import TS_STAT_DIM, TS_FLOAT_DIM


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    pair_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    pair_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict  # {domain: tensor [B, S, L]}
    seq_lens: dict  # {domain: tensor [B]}
    seq_time_buckets: dict  # {domain: tensor [B, L]}
    seq_ts_float_feats: dict  # {domain: tensor [B, 8, L]} float time-derived feats
    seq_ts_stat_feats: dict  # {domain: tensor [B, 6]} precomputed time stats
    # Item-history-user pools (only populated when ItemHistUserModule is enabled
    # and the dataset was built via build_item_hist_users.py). Each row gets two
    # variable-length pools of past users that interacted with the same item:
    #   pos pool: label_type==2 (converters)
    #   neg pool: label_type==1 (shown but did not convert)
    # ``hist_*_lens`` masks the K axis. None when the hist branch is disabled.
    hist_pos_scalars: Optional[torch.Tensor] = None   # (B, K_pos, 7) int64
    hist_pos_dense: Optional[torch.Tensor] = None     # (B, K_pos, 256) float
    hist_neg_scalars: Optional[torch.Tensor] = None   # (B, K_neg, 7) int64
    hist_neg_dense: Optional[torch.Tensor] = None     # (B, K_neg, 256) float
    hist_pos_lens: Optional[torch.Tensor] = None      # (B,) int32
    hist_neg_lens: Optional[torch.Tensor] = None      # (B,) int32


# ═══════════════════════════════════════════════════════════════════════════════
# Rotary Position Embedding (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════


class RotaryEmbedding(nn.Module):
    """Precomputes and caches RoPE cos/sin values.

    Attributes:
        dim: Rotary embedding dimension.
        max_seq_len: Maximum sequence length for cache.
        base: Base frequency for rotary encoding.
    """

    def __init__(
        self, dim: int, max_seq_len: int = 2048, base: float = 10000.0
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inv_freq: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(
            seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer(
            "cos_cached", emb.cos().unsqueeze(0), persistent=False
        )  # (1, seq_len, dim)
        self.register_buffer(
            "sin_cached", emb.sin().unsqueeze(0), persistent=False
        )  # (1, seq_len, dim)

    def forward(
        self, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes cos/sin values for the given sequence length.

        Returns pre-computed slices from the cache. The cache is built once
        in __init__ with max_seq_len; no runtime expansion is performed so
        that the forward pass remains compatible with torch.compile().
        """
        cos = self.cos_cached[:, :seq_len, :].to(device)
        sin = self.sin_cached[:, :seq_len, :].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swaps and negates the first and second halves of the last dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_tensor(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Applies Rotary Position Embedding to a single tensor.

    Args:
        x: (B, num_heads, L, head_dim)
        cos: (1, L_max, head_dim) or (B, L, head_dim) for batch-specific positions.
        sin: Same shape as cos.

    Returns:
        Rotated tensor of shape (B, num_heads, L, head_dim).
    """
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)  # (*, 1, L, head_dim)
    sin_ = sin[:, :L, :].unsqueeze(1)
    return x * cos_ + rotate_half(x) * sin_


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Basic Components
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLU(nn.Module):
    """SwiGLU activation: x1 * SiLU(x2)."""

    def __init__(self, d_model: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc = nn.Linear(d_model, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        x = self.fc_out(x)
        return x


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with Rotary Position Embedding support.

    Manually projects Q/K/V and reshapes for multi-head, then injects RoPE
    after projection and before dot-product. Uses F.scaled_dot_product_attention
    for efficient computation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_on_q: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_on_q = rope_on_q
        self.dropout = dropout

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)

        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, 1.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        q_rope_cos: Optional[torch.Tensor] = None,
        q_rope_sin: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> tuple:
        """Computes multi-head attention with optional RoPE.

        Args:
            query: (B, Lq, D)
            key: (B, Lk, D)
            value: (B, Lk, D)
            key_padding_mask: (B, Lk), True indicates padding positions.
            attn_mask: (Lq, Lk) or (B*num_heads, Lq, Lk), additive mask.
            rope_cos: (1, L, head_dim), RoPE for KV side (also used for Q
                unless q_rope_* is provided).
            rope_sin: Same shape as rope_cos.
            q_rope_cos: (B, Lq, head_dim) or (1, Lq, head_dim), Q-specific
                RoPE for cross-attention with gathered positions.
            q_rope_sin: Same shape as q_rope_cos.
            need_weights: Compatibility parameter, not used.

        Returns:
            Tuple of (output, None).
        """
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        # 1. Linear projection
        Q = self.W_q(query)  # (B, Lq, D)
        K = self.W_k(key)  # (B, Lk, D)
        V = self.W_v(value)  # (B, Lk, D)

        # 2. Reshape to (B, num_heads, L, head_dim)
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE independently to Q and K
        if rope_cos is not None and rope_sin is not None:
            # K always uses rope_cos/rope_sin (KV-side positional encoding)
            K = apply_rope_to_tensor(K, rope_cos, rope_sin)

            if self.rope_on_q:
                # Q side: prefer dedicated q_rope_cos/sin (top_k positions in LongerEncoder cross-attn)
                q_cos = q_rope_cos if q_rope_cos is not None else rope_cos
                q_sin = q_rope_sin if q_rope_sin is not None else rope_sin
                Q = apply_rope_to_tensor(Q, q_cos, q_sin)

        # 4. Convert key_padding_mask to SDPA format
        sdpa_attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, Lk), True = padding
            # SDPA expects (B, 1, 1, Lk) bool mask, True = attend
            sdpa_attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(
                2
            )  # (B, 1, 1, Lk)
            sdpa_attn_mask = sdpa_attn_mask.expand(B, self.num_heads, Lq, Lk)

        if attn_mask is not None:
            # attn_mask: additive float mask (Lq, Lk), -inf means do not attend
            # Convert to bool: positions that are not -inf are True
            bool_attn = attn_mask == 0  # (Lq, Lk)
            bool_attn = (
                bool_attn.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, Lq, Lk)
            )
            if sdpa_attn_mask is not None:
                sdpa_attn_mask = sdpa_attn_mask & bool_attn
            else:
                sdpa_attn_mask = bool_attn

        # 5. Scaled Dot-Product Attention
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=sdpa_attn_mask,
            dropout_p=dropout_p,
        )  # (B, num_heads, Lq, head_dim)

        # Replace NaN from all-padding softmax with 0 (zero vectors preserve original input via residual)
        out = torch.nan_to_num(out, nan=0.0)

        # 6. Reshape back and output projection
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        G = self.W_g(query)
        out = out * torch.sigmoid(G)
        out = self.W_o(out)

        return out, None



class CrossAttention(nn.Module):
    """Cross-attention module.

    Query comes from global tokens (Q tokens), Key/Value comes from sequence
    tokens. Only applies RoPE to KV side (rope_on_q=False).
    """

    def __init__(
        self, d_model: int, num_heads: int, dropout: float = 0.0, ln_mode: str = "pre"
    ) -> None:
        super().__init__()
        self.ln_mode = ln_mode

        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=False,
        )

        if ln_mode in ["pre", "post"]:
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes cross-attention between query tokens and sequence tokens.

        Args:
            query: (B, Nq, D), query tokens.
            key_value: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), KV-side RoPE cosine values.
            rope_sin: (1, L, head_dim), KV-side RoPE sine values.

        Returns:
            Output tensor of shape (B, Nq, D).
        """
        residual = query

        if self.ln_mode == "pre":
            query = self.norm_q(query)
            key_value = self.norm_kv(key_value)

        out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )

        out = residual + out

        if self.ln_mode == "post":
            out = self.norm_q(out)

        return out


class RankMixerBlock(nn.Module):
    """HyFormer Query Boosting block.

    Performs three steps:
    1. Token Mixing: Parameter-free tensor reshaping.
    2. Per-token FFN: Shared-parameter feedforward network.
    3. Residual connection: Q_boost = Q + Q_e.

    Constraint: d_model must be divisible by n_total in 'full' mode.
    """

    def __init__(
        self,
        d_model: int,
        n_total: int,  # T = Nq + Nns
        hidden_mult: int = 4,
        dropout: float = 0.0,
        mode: str = "full",  # 'full' | 'ffn_only' | 'none'
    ) -> None:
        super().__init__()
        self.T = n_total
        self.D = d_model
        self.mode = mode

        if mode == "none":
            # Pure identity mapping, no submodules created
            return

        if mode == "full":
            if d_model % n_total != 0:
                raise ValueError(
                    f"d_model={d_model} must be divisible by T={n_total} for token mixing."
                )
            self.d_sub = d_model // n_total

        # Per-token FFN (shared parameters) — used by both 'full' and 'ffn_only'
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * hidden_mult)
        self.fc2 = nn.Linear(d_model * hidden_mult, d_model)
        self.dropout = nn.Dropout(dropout)
        # Post-LN after residual to stabilize stacked block outputs
        self.post_norm = nn.LayerNorm(d_model)

    def token_mixing(self, Q: torch.Tensor) -> torch.Tensor:
        """Performs parameter-free token mixing via reshape and transpose.

        Steps:
        1. Splits channels into T subspaces: (B, T, D) -> (B, T, T, d_sub).
        2. Swaps token and subspace axes: (B, token, h, d_sub) -> (B, h, token, d_sub).
        3. Flattens back: (B, T, D).

        Args:
            Q: (B, T, D)

        Returns:
            Mixed tensor of shape (B, T, D).
        """
        B, T, D = Q.shape

        # (B, T, D) -> (B, T, T, d_sub)
        Q_split = Q.view(B, T, self.T, self.d_sub)

        # (B, token, h, d_sub) -> (B, h, token, d_sub)
        Q_rewired = Q_split.transpose(1, 2).contiguous()

        # (B, T, T, d_sub) -> (B, T, D)
        Q_hat = Q_rewired.view(B, T, D)
        return Q_hat

    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        """Applies query boosting: token mixing, FFN, and residual connection.

        Args:
            Q: (B, T, D) where T = Nq + Nns.

        Returns:
            Boosted tensor of shape (B, T, D).
        """
        if self.mode == "none":
            return Q

        # Token Mixing (parameter-free rewire) or identity
        if self.mode == "full":
            Q_hat = self.token_mixing(Q)
        else:  # 'ffn_only'
            Q_hat = Q

        # Per-token FFN
        x = self.norm(Q_hat)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        Q_e = self.fc2(x)

        # Residual from original Q
        Q_boost = Q + Q_e
        Q_boost = self.post_norm(Q_boost)
        return Q_boost


class MultiSeqQueryGenerator(nn.Module):
    """Multi-sequence query generation module.

    Generates Q tokens independently for each sequence:
    For each sequence i:
        GlobalInfo_i = Concat(F1..FM, MeanPool(Seq_i), Proj(stat_feats_i))
        Q_i = [FFN_{i,1}(GlobalInfo_i), ..., FFN_{i,N}(GlobalInfo_i)]
    """

    def __init__(
        self,
        d_model: int,
        num_ns: int,
        num_queries: int,
        num_sequences: int,
        hidden_mult: int = 4,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        self.d_model = d_model

        global_info_dim = (num_ns + 2) * d_model

        self.global_info_norm = nn.LayerNorm(global_info_dim)

        self.stat_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(TS_STAT_DIM, d_model),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_sequences)
            ]
        )

        self.query_ffns_per_seq = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(global_info_dim, d_model * hidden_mult),
                            nn.SiLU(),
                            nn.Linear(d_model * hidden_mult, d_model),
                            nn.LayerNorm(d_model),
                        )
                        for _ in range(num_queries)
                    ]
                )
                for _ in range(num_sequences)
            ]
        )

    def forward(
        self,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        seq_ts_stat_feats_list: List[torch.Tensor],
    ) -> list:
        """Generates query tokens for each sequence.

        Args:
            ns_tokens: (B, M, D), shared NS tokens.
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S. True
                indicates padding.
            seq_ts_stat_feats_list: List of (B, 6) precomputed time stats per domain.

        Returns:
            List of (B, Nq, D) query token tensors, length S.
        """
        B = ns_tokens.shape[0]
        ns_flat = ns_tokens.view(B, -1)  # (B, M*D)
        ns_repr = ns_tokens.mean(dim=1)  # (B, D) — target-aware query for attn pooling

        q_tokens_list = []
        for i in range(self.num_sequences):
            padding_mask = seq_padding_masks[i]          # (B, L), True=padding
            seq_tok = seq_tokens_list[i]                 # (B, L, D)

            valid = (~padding_mask).float().unsqueeze(-1)  # (B, L, 1)
            denom = valid.sum(dim=1).clamp(min=1)
            seq_pooled = (seq_tok * valid).sum(dim=1) / denom  # (B, D)

            raw_stats = seq_ts_stat_feats_list[i].to(ns_tokens.device)
            stat_vec = self.stat_proj[i](raw_stats)

            global_info = torch.cat([ns_flat, seq_pooled, stat_vec], dim=-1)
            global_info = self.global_info_norm(global_info)

            queries = [ffn(global_info) for ffn in self.query_ffns_per_seq[i]]
            q_tokens = torch.stack(queries, dim=1)  # (B, Nq, D)
            q_tokens_list.append(q_tokens)

        return q_tokens_list


# ═══════════════════════════════════════════════════════════════════════════════
# Sequence Encoders
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLUEncoder(nn.Module):
    """Efficient attention-free sequence encoder.

    Structure: x + Dropout(SwiGLU(LN(x))).
    """

    def __init__(
        self, d_model: int, hidden_mult: int = 4, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model, hidden_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        """Applies the SwiGLU encoder with residual connection.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding. Not used by
                this encoder variant.
            **kwargs: Absorbs rope_cos/rope_sin and other unused parameters.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        residual = x
        x = self.norm(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = residual + x
        return x, key_padding_mask


class TransformerEncoder(nn.Module):
    """High-capacity sequence encoder with self-attention and RoPE.

    Structure: Standard Transformer Encoder Layer (Pre-LN).
    """

    def __init__(
        self, d_model: int, num_heads: int, hidden_mult: int = 4, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies one Transformer encoder layer.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), RoPE cosine values.
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        # Self-Attention (Pre-LN) with RoPE
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        x = residual + x

        # FFN (Pre-LN)
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, key_padding_mask


class LongerEncoder(nn.Module):
    """Top-K compressed sequence encoder.

    Adapts behavior based on input length:
    - L > top_k (first MultiSeqHyFormerBlock): Cross Attention.
      Q = latest top_k tokens, K/V = all seq tokens -> output (B, top_k, D).
    - L <= top_k (subsequent MultiSeqHyFormerBlocks): Self Attention.
      Q = K = V = top_k tokens -> output (B, top_k, D).

    Causal mask is only applied among top_k tokens (self-attention layers);
    the first cross-attention layer does not use a causal mask since Q and K
    have different lengths.

    Returns (output, new_key_padding_mask) so downstream can update the mask.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        top_k: int = 50,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.causal = causal

        # Pre-LN for attention
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        # Shared RoPEMHA for both cross and self attention
        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        # FFN (Pre-LN + residual)
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def _gather_top_k(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selects the latest top_k valid tokens from each sample.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding.

        Returns:
            top_k_tokens: (B, top_k, D)
            new_padding_mask: (B, top_k), True indicates padding.
            position_indices: (B, top_k), original position index for each
                selected token, used for Q-side RoPE.
        """
        B, L, D = x.shape
        device = x.device

        # Valid lengths per sample
        valid_len = (~key_padding_mask).sum(dim=1)  # (B,)

        # Start position for each sample: max(valid_len - top_k, 0)
        actual_k = torch.clamp(valid_len, max=self.top_k)  # (B,)
        start_pos = valid_len - actual_k  # (B,)

        # Build gather indices: (B, top_k)
        offsets = (
            torch.arange(self.top_k, device=device).unsqueeze(0).expand(B, -1)
        )  # (B, top_k)
        indices = start_pos.unsqueeze(1) + offsets  # (B, top_k)

        # For samples with valid_len < top_k, early indices may exceed valid range;
        # clamp to [0, L-1] and handle via mask below
        indices = torch.clamp(indices, min=0, max=L - 1)

        # Gather: (B, top_k, D)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)  # (B, top_k, D)
        top_k_tokens = torch.gather(x, dim=1, index=indices_expanded)

        # New padding mask: first (top_k - actual_k) positions are padding
        new_valid_len = actual_k  # (B,)
        pad_count = self.top_k - new_valid_len  # (B,)
        pos_indices = torch.arange(self.top_k, device=device).unsqueeze(0)  # (1, top_k)
        new_padding_mask = pos_indices < pad_count.unsqueeze(1)  # (B, top_k)

        # Zero out tokens at padding positions
        top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).float()

        # position_indices for Q-side RoPE
        position_indices = indices  # (B, top_k)

        return top_k_tokens, new_padding_mask, position_indices

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the LongerEncoder with adaptive cross/self attention.

        Args:
            x: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding.
            rope_cos: (1, L, head_dim), RoPE cosine values (length must cover
                original sequence length L).
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            output: (B, top_k, D), compressed sequence.
            new_key_padding_mask: (B, top_k), updated padding mask.
        """
        B, L, D = x.shape

        if L > self.top_k:
            # === Cross Attention mode (first MultiSeqHyFormerBlock) ===
            # 1. Extract latest top_k tokens as query
            q, new_mask, q_pos_indices = self._gather_top_k(x, key_padding_mask)

            # 2. Pre-LN
            q_normed = self.norm_q(q)
            kv_normed = self.norm_kv(x)

            # 3. Build Q-side RoPE cos/sin by gathering from global cos/sin at top_k positions
            q_rope_cos = None
            q_rope_sin = None
            if rope_cos is not None and rope_sin is not None:
                # rope_cos: (1, L_max, head_dim), q_pos_indices: (B, top_k)
                head_dim = rope_cos.shape[2]
                # Expand to batch dimension
                cos_expanded = rope_cos.expand(B, -1, -1)  # (B, L_max, head_dim)
                sin_expanded = rope_sin.expand(B, -1, -1)
                idx = q_pos_indices.unsqueeze(-1).expand(
                    -1, -1, head_dim
                )  # (B, top_k, head_dim)
                q_rope_cos = torch.gather(cos_expanded, 1, idx)  # (B, top_k, head_dim)
                q_rope_sin = torch.gather(sin_expanded, 1, idx)

            # 4. Cross Attention (no causal mask since Q and K have different lengths)
            attn_out, _ = self.attn(
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,  # Original (B, L) mask
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                q_rope_cos=q_rope_cos,
                q_rope_sin=q_rope_sin,
            )
            out = q + attn_out  # Residual based on q
        else:
            # === Self Attention mode (subsequent MultiSeqHyFormerBlocks) ===
            new_mask = key_padding_mask

            # Pre-LN (Q and KV share norm_q)
            x_normed = self.norm_q(x)

            # Causal mask
            attn_mask = None
            if self.causal:
                attn_mask = nn.Transformer.generate_square_subsequent_mask(
                    L, device=x.device
                )

            attn_out, _ = self.attn(
                query=x_normed,
                key=x_normed,
                value=x_normed,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
            out = x + attn_out

        # FFN (Pre-LN + residual)
        residual = out
        out = self.ffn_norm(out)
        out = self.ffn(out)
        out = residual + out

        return out, new_mask


def create_sequence_encoder(
    encoder_type: str,
    d_model: int,
    num_heads: int = 4,
    hidden_mult: int = 4,
    dropout: float = 0.0,
    top_k: int = 50,
    causal: bool = False,
) -> nn.Module:
    """Creates a sequence encoder of the specified type.

    Args:
        encoder_type: One of 'swiglu', 'transformer', or 'longer'.
        d_model: Model dimension.
        num_heads: Number of attention heads (used by transformer/longer).
        hidden_mult: FFN expansion multiplier.
        dropout: Dropout rate.
        top_k: Compression length for LongerEncoder (only used by longer).
        causal: Whether to use causal mask in LongerEncoder (only used by
            longer).

    Returns:
        A sequence encoder module.
    """
    if encoder_type == "swiglu":
        return SwiGLUEncoder(d_model, hidden_mult, dropout)
    elif encoder_type == "transformer":
        return TransformerEncoder(d_model, num_heads, hidden_mult, dropout)
    elif encoder_type == "longer":
        return LongerEncoder(d_model, num_heads, top_k, hidden_mult, dropout, causal)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Blocks
# ═══════════════════════════════════════════════════════════════════════════════


class MultiSeqHyFormerBlock(nn.Module):
    """Multi-sequence HyFormer block.

    Each of the S sequences independently performs Sequence Evolution and
    Query Decoding, then all Q tokens and shared NS tokens are merged for
    joint Query Boosting.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_queries: int,
        num_ns: int,
        num_sequences: int,
        seq_encoder_type: str = "swiglu",
        hidden_mult: int = 4,
        dropout: float = 0.0,
        top_k: int = 50,
        causal: bool = False,
        rank_mixer_mode: str = "full",
    ) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        self.num_queries = num_queries
        self.num_ns = num_ns

        # Independent sequence encoder per sequence
        self.seq_encoders = nn.ModuleList(
            [
                create_sequence_encoder(
                    encoder_type=seq_encoder_type,
                    d_model=d_model,
                    num_heads=num_heads,
                    hidden_mult=hidden_mult,
                    dropout=dropout,
                    top_k=top_k,
                    causal=causal,
                )
                for _ in range(num_sequences)
            ]
        )

        # Independent cross-attention per sequence
        self.cross_attns = nn.ModuleList(
            [
                CrossAttention(
                    d_model=d_model, num_heads=num_heads, dropout=dropout, ln_mode="pre"
                )
                for _ in range(num_sequences)
            ]
        )

        # RankMixer: input token count = Nq * S + Nns
        n_total = num_queries * num_sequences + num_ns
        self.mixer = RankMixerBlock(
            d_model=d_model,
            n_total=n_total,
            hidden_mult=hidden_mult,
            dropout=dropout,
            mode=rank_mixer_mode,
        )

    def forward(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        rope_cos_list: Optional[List[torch.Tensor]] = None,
        rope_sin_list: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[list, torch.Tensor, list, list]:
        """Processes one multi-sequence HyFormer block step.

        Args:
            q_tokens_list: List of (B, Nq, D) tensors, length S.
            ns_tokens: (B, Nns, D)
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S.
            rope_cos_list: List of (1, L_i, head_dim) tensors, length S.
            rope_sin_list: List of (1, L_i, head_dim) tensors, length S.

        Returns:
            A tuple (next_q_list, next_ns, next_seq_list, next_masks), where
            next_q_list is a list of (B, Nq, D) updated query tensors,
            next_ns is (B, Nns, D) updated non-sequence tokens,
            next_seq_list is a list of (B, L_i', D) encoded sequence tensors,
            and next_masks is a list of (B, L_i') updated padding masks.
        """
        S = self.num_sequences
        Nq = self.num_queries

        # 1. Independent Sequence Evolution per sequence
        next_seqs = []
        next_masks = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            result = self.seq_encoders[i](
                seq_tokens_list[i],
                seq_padding_masks[i],
                rope_cos=rc,
                rope_sin=rs,
            )
            next_seq_i, mask_i = result
            next_seqs.append(next_seq_i)
            next_masks.append(mask_i)

        # 2. Independent Query Decoding per sequence
        decoded_qs = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            decoded_q_i = self.cross_attns[i](
                q_tokens_list[i],
                next_seqs[i],
                next_masks[i],
                rope_cos=rc,
                rope_sin=rs,
            )
            decoded_qs.append(decoded_q_i)

        # 3. Token Fusion: concatenate all decoded_q + ns_tokens
        combined = torch.cat(decoded_qs + [ns_tokens], dim=1)  # (B, Nq*S + Nns, D)

        # 4. Query Boosting
        boosted = self.mixer(combined)  # (B, Nq*S + Nns, D)

        # 5. Split back into per-sequence Q and NS
        next_q_list = []
        offset = 0
        for i in range(S):
            next_q_list.append(boosted[:, offset : offset + Nq, :])
            offset += Nq
        next_ns = boosted[:, offset:, :]

        return next_q_list, next_ns, next_seqs, next_masks


# ═══════════════════════════════════════════════════════════════════════════════
# PCVRHyFormer Main Model
# ═══════════════════════════════════════════════════════════════════════════════


class GroupNSTokenizer(nn.Module):
    """NS tokenizer used by ns_tokenizer_type='group'.

    Groups discrete features by fid, applies shared embedding with mean
    pooling per multi-valued feature, then projects each group to a single
    NS token (one token per group).
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        groups: List[List[int]],
        emb_dim: int,
        d_model: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        # self.emb_dim = emb_dim
        self.emb_skip_threshold = emb_skip_threshold
        self.emb_dim_list = []

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (
                emb_skip_threshold > 0 and int(vs) > emb_skip_threshold
            )
            if skip:
                embs.append(None)
                self.emb_dim_list.append(0)
            else:
                emb_dim = get_emb_dim(vs, 64)
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
                self.emb_dim_list.append(emb_dim)

        self.embs = nn.ModuleList([e for e in embs if e is not None])

        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        self.group_dims = [
            sum(self.emb_dim_list[fid] for fid in group) for group in groups
        ]

        # Per-group projection: num_fids_in_group * emb_dim -> d_model (with LayerNorm)
        self.group_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, d_model),
                    nn.LayerNorm(d_model),
                )
                for dim in self.group_dims
            ]
        )

        # ---- gate ----
        self.group_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, max(8, dim // 4)),
                    nn.SiLU(),
                    nn.Linear(max(8, dim // 4), dim),
                    nn.Sigmoid(),
                )
                for dim in self.group_dims
            ]
        )

        self.attn_layers = nn.ModuleList(
            [nn.Linear(dim, 1) if dim > 0 else None for dim in self.emb_dim_list]
        )

    def forward(self, int_feats):
        B = int_feats.size(0)
        tokens = []

        for group_idx, (group, proj) in enumerate(zip(self.groups, self.group_projs)):
            fid_embs = []

            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                dim = self.emb_dim_list[fid_idx]

                if emb_real_idx == -1:
                    fid_emb = int_feats.new_zeros(B, dim)
                else:
                    emb_layer = self.embs[emb_real_idx]

                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())

                    else:
                        vals = int_feats[:, offset : offset + length].long()
                        emb_all = emb_layer(vals)  # (B, L, D)

                        # mask
                        mask = (vals != 0).unsqueeze(-1)  # (B, L, 1)

                        # attention pooling
                        attn = self.attn_layers[fid_idx](emb_all)  # (B, L, 1)
                        attn = attn.masked_fill(~mask, -1e9)
                        attn = torch.softmax(attn, dim=1)

                        fid_emb = (emb_all * attn).sum(dim=1)

                fid_embs.append(fid_emb)

            cat_emb = torch.cat(fid_embs, dim=-1)

            # gating
            gate = self.group_gates[group_idx](cat_emb)
            cat_emb = cat_emb * (1.0 + gate)

            tokens.append(F.silu(proj(cat_emb)).unsqueeze(1))

        return torch.cat(tokens, dim=1)


def get_emb_dim(vocab_size: int, emb_dim: int) -> int:
    # return 64
    if vocab_size <= 4:
        return 4
    elif vocab_size <= 10:
        return 8
    elif vocab_size <= 50:
        return 16
    elif vocab_size <= 600:
        return 32
    else:
        return 64


class RankMixerNSTokenizer(nn.Module):
    """NS Tokenizer following the RankMixer paper's approach.

    All group embedding vectors are concatenated into a single long vector,
    then equally split into num_ns_tokens segments, each projected to d_model.
    This allows num_ns_tokens to be chosen freely (independent of group count).
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        groups: List[List[int]],
        emb_dim: int,
        d_model: int,
        num_ns_tokens: int,
        emb_skip_threshold: int = 0,
        extra_emb_dim: int = 0,
    ) -> None:
        """Initializes RankMixerNSTokenizer.

        Args:
            feature_specs: [(vocab_size, offset, length), ...] per feature.
            groups: List of feature index groups (defines semantic ordering).
            emb_dim: Embedding dimension per feature.
            d_model: Output token dimension.
            num_ns_tokens: Number of NS tokens to produce (T segments).
            emb_skip_threshold: Skip embedding for features with vocab > threshold.
        """
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.num_ns_tokens = num_ns_tokens
        self.emb_skip_threshold = emb_skip_threshold
        self.total_emb_dim = 0
        self.offset_to_index = {}

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        count = 0
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (
                emb_skip_threshold > 0 and int(vs) > emb_skip_threshold
            )
            if skip:
                embs.append(None)
                # Skipped features still contribute a zero vector of size emb_dim
                # in forward(), so account for them here.
                self.total_emb_dim += emb_dim
            else:
                vs_emb_dim = get_emb_dim(vs, emb_dim)
                self.total_emb_dim += vs_emb_dim
                embs.append(nn.Embedding(int(vs) + 1, vs_emb_dim, padding_idx=0))
                self.offset_to_index[offset] = count
                count += 1

        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Compute total embedding dim: sum of all fids across all groups
        total_num_fids = sum(len(g) for g in groups)

        # Fold in extra embeddings (e.g. pair features injected from outside)
        self.total_emb_dim += extra_emb_dim

        # Pad total_emb_dim to be divisible by num_ns_tokens
        self.chunk_dim = math.ceil(self.total_emb_dim / num_ns_tokens)
        self.padded_total_dim = self.chunk_dim * num_ns_tokens
        self._pad_size = self.padded_total_dim - self.total_emb_dim

        self.lhuc = nn.Sequential(
            nn.Linear(self.total_emb_dim, self.total_emb_dim // 4),
            nn.SiLU(),
            nn.Linear(self.total_emb_dim // 4, self.total_emb_dim),
            nn.Sigmoid(),
        )

        # Per-chunk projection: chunk_dim -> d_model with LayerNorm
        self.token_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.chunk_dim, d_model),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_ns_tokens)
            ]
        )

        logging.info(
            f"RankMixerNSTokenizer: {total_num_fids} fids, "
            f"total_emb_dim={self.total_emb_dim}, chunk_dim={self.chunk_dim}, "
            f"num_ns_tokens={num_ns_tokens}, pad={self._pad_size}"
        )

    def forward(self, int_feats: torch.Tensor, extra_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Embeds all features, concatenates, splits, and projects.

        Args:
            int_feats: (B, total_int_dim) concatenated integer features.
            extra_emb: optional (B, extra_emb_dim) tensor appended before LHUC,
                e.g. pair feature embeddings from CrossRankMixerNSTokenizer.

        Returns:
            (B, num_ns_tokens, d_model) tensor.
        """
        B = int_feats.size(0)
        group_outputs = []

        for group in self.groups:
            fid_outputs = []

            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]

                x = int_feats[:, offset : offset + length]

                if emb_real_idx == -1:
                    fid_outputs.append(x.new_zeros(B, self.emb_dim))
                    continue

                emb_layer = self.embs[emb_real_idx]

                # ---- scalar feature ----
                if length == 1:
                    fid_outputs.append(emb_layer(x.squeeze(-1)))
                    continue

                # ---- sequence feature (vectorized pooling) ----
                vals = x.long()  # (B, L)
                emb_all = emb_layer(vals)  # (B, L, D)

                mask = (vals != 0).unsqueeze(-1)  # (B, L, 1)
                denom = mask.sum(dim=1).clamp(min=1)

                fid_emb = (emb_all * mask).sum(dim=1) / denom
                fid_outputs.append(fid_emb)

            group_emb = torch.cat(fid_outputs, dim=-1)
            group_outputs.append(group_emb)

        cat_emb = torch.cat(group_outputs, dim=-1)
        if extra_emb is not None:
            cat_emb = torch.cat([cat_emb, extra_emb], dim=-1)
        gate = self.lhuc(cat_emb)
        cat_emb = cat_emb * gate * 2.0

        if self._pad_size > 0:
            cat_emb = F.pad(cat_emb, (0, self._pad_size))

        cat_emb = cat_emb.view(B, self.num_ns_tokens, self.chunk_dim)

        # projection
        outs = []
        for i, proj in enumerate(self.token_projs):
            outs.append(proj(cat_emb[:, i]).unsqueeze(1))

        return torch.cat(outs, dim=1)


class CrossRankMixerNSTokenizer(nn.Module):
    """
    Pair feature embedder (not a standalone NS tokenizer).

    Each pair feature has aligned int (category ID) and dense (weight/score) arrays.
    For each feature: embed int → weight by dense → aggregate → (B, emb_dim).

    - Head features (62-66): dense = log1p counts → L1-normalised weights.
    - Tail features (89-91): dense = similarity scores (can be negative)
      → masked softmax weights (padding masked to -inf).

    Output: flat (B, num_features * emb_dim) injected into user_ns_tokenizer.
    """

    def __init__(
        self,
        feature_specs,
        d_model: int,
        emb_dim: int = 64,
        num_pos: int = 10,
    ):
        super().__init__()

        self.feature_specs = feature_specs
        self.d_model = d_model
        self.emb_dim = emb_dim
        self.num_pos = num_pos

        self.num_head = len(feature_specs) - 3
        self.num_tail = 3

        # one embedding table per feature, using get_emb_dim for vocab-adaptive sizing
        self.embs = nn.ModuleList(
            [
                nn.Embedding(vs + 1, get_emb_dim(vs, emb_dim), padding_idx=0)
                for (vs, _, _) in feature_specs
            ]
        )
        # store per-feature emb dims for forward slicing
        self._emb_dims = [get_emb_dim(vs, emb_dim) for (vs, _, _) in feature_specs]

        # flat output dim — injected into user_ns_tokenizer before LHUC
        self.out_dim = sum(self._emb_dims)
        self.dense_projs = nn.ModuleList([nn.Linear(1, d) for d in self._emb_dims])
        self.fusion_projs = nn.ModuleList(
            [nn.Linear(d * 2 + 5, d) for d in self._emb_dims]
        )

    def _slice(self, x, offset, length):
        return x[:, offset : offset + length]

    def forward(self, pair_int_feats, pair_dense_feats):
        """
        pair_int_feats:  (B, total_dim)
        pair_dense_feats: (B, total_dim)

        Returns (B, out_dim) flat embedding.
        """
        outs = []

        # Position-aligned int/dense pooling for all pair features.
        for i in range(len(self.feature_specs)):
            _, offset, length = self.feature_specs[i]
            x = self._slice(pair_int_feats, offset, length)
            dense = self._slice(pair_dense_feats, offset, length)

            valid = (x != 0) & torch.isfinite(dense)
            mask = valid.float()
            valid_count = mask.sum(dim=1, keepdim=True)
            count = valid_count.clamp(min=1.0)
            mask_3d = mask.unsqueeze(-1)

            int_emb = self.embs[i](x.long())
            int_pool = (int_emb * mask_3d).sum(dim=1) / count

            dense_clean = torch.where(valid, dense, torch.zeros_like(dense))
            dense_emb = self.dense_projs[i](dense_clean.unsqueeze(-1))
            dense_pool = (dense_emb * mask_3d).sum(dim=1) / count

            has_valid = valid.any(dim=1, keepdim=True)
            dense_max = torch.where(
                valid, dense, torch.full_like(dense, float("-inf"))
            ).max(dim=1, keepdim=True).values
            dense_max = torch.where(has_valid, dense_max, torch.zeros_like(dense_max))
            dense_mean = dense_clean.sum(dim=1, keepdim=True) / count
            dense_var = (((dense_clean - dense_mean) * mask) ** 2).sum(
                dim=1, keepdim=True
            ) / count
            dense_std = torch.sqrt(dense_var.clamp_min(0.0) + 1e-6)
            dense_std = torch.where(has_valid, dense_std, torch.zeros_like(dense_std))
            coverage = valid_count / float(length)

            stats = torch.cat(
                [valid_count.log1p(), coverage, dense_max, dense_mean, dense_std],
                dim=-1,
            )
            fused = torch.cat([int_pool, dense_pool, stats], dim=-1)
            outs.append(F.silu(self.fusion_projs[i](fused)))

        # softmax so the result is uniform weights over zero embeddings → zero output.
        return torch.cat(outs, dim=-1)  # (B, out_dim)


# ═══════════════════════════════════════════════════════════════════════════════
# Item-history-user module (audience matching, no item_id required)
# ═══════════════════════════════════════════════════════════════════════════════


class UserQueryPool(nn.Module):
    """Pool the ``num_user_ns`` pre-HyFormer user NS tokens into a single query
    token via one learnable cross-attention.

    Used as the Q side of ItemHistUserModule's two cross-attentions. We use the
    PRE-HyFormer user tokens (not the post-HyFormer mixed output) so that the
    Q and K/V live in the same pure-user semantic space — see project notes.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = CrossAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout, ln_mode="pre"
        )

    def forward(self, user_ns_tokens: torch.Tensor) -> torch.Tensor:
        """user_ns_tokens: (B, num_user_ns, D) → (B, 1, D)."""
        B = user_ns_tokens.size(0)
        q = self.query.expand(B, -1, -1)
        return self.attn(q, user_ns_tokens, key_padding_mask=None)


class ItemHistUserModule(nn.Module):
    """Encode the per-row item-history-user pools and cross-attend with the
    current user, producing two ``(B, d_model)`` representations (one per
    pool). The caller injects them as 2 extra "virtual domain" reprs into
    ``_domain_sequence_gate`` so they participate in the same softmax-weighted
    merge as the 4 sequence domains.

    Design notes:
    - The 7 scalar features per historical user have their OWN Embedding
      tables in this module (no longer shared with ``user_ns_tokenizer``).
      Sharing was an alignment hypothesis that didn't pan out — owning the
      tables gives the hist branch a clean independent representation space.
    - The 256-d dense_61 user embedding goes through a private MLP.
    - Per-user token = ``Linear(cat[scalar_embs, dense_proj]) → d_model``.
    - Cross-attention: query = single pooled user token (B,1,D); KV = K hist
      tokens; key_padding_mask is built from ``hist_*_lens``.
    - Cold-pool fallback: when ``hist_*_lens==0`` or random history dropout
      fires (training only, prob ``hist_dropout``), the attention output is
      OVERWRITTEN by a learnable empty token (per pool).
    - This module is INTENTIONALLY headless on fusion: it does NOT decide how
      its two outputs are mixed with the backbone. ``forward`` returns the
      tuple ``(pos_t, neg_t)`` and PCVRHyFormer feeds those tensors into
      ``_domain_sequence_gate`` as 2 extra reprs, with dedicated score MLPs.
    """

    def __init__(
        self,
        scalar_vocab_sizes: List[int],
        hist_dense_dim: int,
        d_model: int,
        num_heads: int,
        emb_dim_base: int = 64,
        dropout: float = 0.0,
        history_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.history_dropout = history_dropout

        # Own scalar embedding tables (one per hist scalar fid). vocab_size+1
        # accounts for the implicit padding row at index 0, matching the
        # convention used by RankMixerNSTokenizer.
        embs: List[nn.Embedding] = []
        emb_dims: List[int] = []
        for vs in scalar_vocab_sizes:
            emb_d = get_emb_dim(int(vs), emb_dim_base)
            embs.append(nn.Embedding(int(vs) + 1, emb_d, padding_idx=0))
            emb_dims.append(emb_d)
        self.scalar_embs = nn.ModuleList(embs)
        self.scalar_emb_dims = emb_dims
        scalar_total_dim = sum(emb_dims)

        # Private dense_61 projection (fresh params)
        self.dense_proj = nn.Sequential(
            nn.Linear(hist_dense_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Per-user token projection: (scalar_emb_cat | dense_proj) → d_model
        self.token_proj = nn.Sequential(
            nn.Linear(scalar_total_dim + d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Cross-attention modules — independent params for pos vs neg pool
        self.pos_attn = CrossAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout, ln_mode="pre"
        )
        self.neg_attn = CrossAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout, ln_mode="pre"
        )

        # Learnable cold-pool fallback tokens (one per pool)
        self.empty_pos = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.empty_neg = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Gated residual fusion: hist contribution is added on top of the
        # backbone output, with a per-dim sigmoid gate so the model can
        # shrink the add-on when hist is unreliable. Gate input is
        # [pos_t, neg_t] only (no baseline output) so the gate can't collapse
        # into a copy of the backbone signal.
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def encode_hist_users(
        self, hist_scalars: torch.Tensor, hist_dense: torch.Tensor
    ) -> torch.Tensor:
        """Encode K historical users per sample → (B, K, d_model) tokens.

        ``hist_scalars``: (B, K, len(scalar_embs)) int64
        ``hist_dense``:   (B, K, hist_dense_dim) float
        """
        B, K, n_fids = hist_scalars.shape

        # Scalar embeddings — flat across (B*K) for fewer kernel launches
        flat = hist_scalars.reshape(B * K, n_fids).long()
        per_feat: List[torch.Tensor] = []
        for j, emb in enumerate(self.scalar_embs):
            per_feat.append(emb(flat[:, j]))
        scalar_emb = torch.cat(per_feat, dim=-1)              # (B*K, scalar_dim)
        scalar_emb = scalar_emb.view(B, K, -1)

        # Dense projection
        dense_emb = self.dense_proj(hist_dense)               # (B, K, D)

        # Combine to one token per user
        tok = torch.cat([scalar_emb, dense_emb], dim=-1)
        tok = self.token_proj(tok)                            # (B, K, D)
        return tok

    def _cross_attend_with_fallback(
        self,
        query: torch.Tensor,           # (B, 1, D)
        kv: torch.Tensor,              # (B, K, D)
        kv_lens: torch.Tensor,         # (B,)
        empty_token: torch.Tensor,     # (1, 1, D)
        attn: nn.Module,
        is_dropped: torch.Tensor,      # (B,) bool
    ) -> torch.Tensor:
        """Run cross-attention with a key_padding_mask built from ``kv_lens``,
        then OVERWRITE the output of rows whose pool is empty or whose history
        was dropped this step — using the learnable ``empty_token``.
        """
        B, K, D = kv.shape
        device = kv.device
        idx = torch.arange(K, device=device).unsqueeze(0)             # (1, K)
        pad_mask = idx >= kv_lens.to(device=device).unsqueeze(1)      # (B, K)

        pool_empty = (kv_lens == 0)
        use_empty = pool_empty | is_dropped                            # (B,)
        if use_empty.any():
            # All-padding rows: flip slot 0 to non-pad to avoid NaN softmax.
            first_slot = use_empty.unsqueeze(1) & (idx == 0)
            pad_mask = pad_mask & ~first_slot

        out = attn(query, kv, key_padding_mask=pad_mask)              # (B,1,D)

        out = torch.where(
            use_empty.view(B, 1, 1),
            empty_token.expand(B, 1, D),
            out,
        )
        return out  # (B, 1, D)

    def forward(
        self,
        current_scalars: torch.Tensor,     # (B, n_fids) int64 — current user's scalars
        current_dense: torch.Tensor,       # (B, hist_dense_dim) float — current user's dense_61
        hist_pos_scalars: torch.Tensor,
        hist_pos_dense: torch.Tensor,
        hist_pos_lens: torch.Tensor,
        hist_neg_scalars: torch.Tensor,
        hist_neg_dense: torch.Tensor,
        hist_neg_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Return the gated-residual delta ``(B, d_model)`` to be added onto
        the HyFormer output by the caller.

        Q (current user) and K/V (historical users) are encoded by the SAME
        ``encode_hist_users`` path, so the cross-attention dot products live
        in one space.
        """
        B = current_scalars.size(0)
        device = current_scalars.device

        # Encode current user as a 1-element pool with the same encoder.
        query = self.encode_hist_users(
            current_scalars.unsqueeze(1),  # (B, 1, n_fids)
            current_dense.unsqueeze(1),     # (B, 1, hist_dense_dim)
        )                                   # (B, 1, D)

        if self.training and self.history_dropout > 0:
            is_dropped = (
                torch.rand(B, device=device) < self.history_dropout
            )
        else:
            is_dropped = torch.zeros(B, dtype=torch.bool, device=device)

        pos_kv = self.encode_hist_users(hist_pos_scalars, hist_pos_dense)
        neg_kv = self.encode_hist_users(hist_neg_scalars, hist_neg_dense)

        pos_out = self._cross_attend_with_fallback(
            query, pos_kv, hist_pos_lens,
            self.empty_pos, self.pos_attn, is_dropped,
        )
        neg_out = self._cross_attend_with_fallback(
            query, neg_kv, hist_neg_lens,
            self.empty_neg, self.neg_attn, is_dropped,
        )

        pos_t = pos_out.squeeze(1)                       # (B, D)
        neg_t = neg_out.squeeze(1)
        cat = torch.cat([pos_t, neg_t], dim=-1)          # (B, 2D)
        gate = self.fusion_gate(cat)                     # (B, D)
        delta = self.fusion_proj(cat)                    # (B, D)
        return gate * delta                              # (B, D)


class PCVRHyFormer(nn.Module):
    """PCVRHyFormer model for post-click conversion rate prediction.

    Combines MultiSeqHyFormerBlock and MultiSeqQueryGenerator to process
    multiple input sequences with non-sequence features.
    """

    def __init__(
        self,
        # Data schema
        user_int_feature_specs: List[Tuple[int, int, int]],
        pair_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: "dict[str, List[int]]",  # {domain: [vocab_size_per_fid, ...]}
        # NS grouping config (grouped by fid index)
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        # Model hyperparameters
        d_model: int = 64,
        emb_dim: int = 64,
        num_queries: int = 1,
        num_hyformer_blocks: int = 2,
        num_heads: int = 4,
        seq_encoder_type: str = "transformer",
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        seq_top_k: int = 50,
        seq_causal: bool = False,
        action_num: int = 1,
        num_time_buckets: int = 65,
        rank_mixer_mode: str = "full",
        use_rope: bool = False,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        # NS tokenizer variant
        ns_tokenizer_type: str = "rankmixer",
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
        # user_dense split: fid=61 (user embedding) | fid=87 (history seq) | hod(2)
        user_emb_dim: int = 256,
        user_seq_block_dim: int = 32,
        user_seq_num: int = 10,
        # DIN: item_ns attends over each sequence domain before HyFormer blocks.
        # Updates item_ns in-place (residual), no change to num_ns or T.
        use_din: bool = False,
        # ── Item-history-user (audience matching) module ──
        # If ``enable_hist_users`` is True, the model builds an
        # ItemHistUserModule that encodes the current user AND the per-row
        # pos/neg historical-user pools with one shared encoder, producing
        # two pool reprs that enter ``_domain_sequence_gate`` as 2 extra
        # virtual domains in the same softmax-weighted merge as the 4 seq
        # domains. Q and K/V share weights and feature set, so the cross-
        # attention dot product is strictly aligned.
        enable_hist_users: bool = False,
        # Positions of the 7 stable user scalar fids inside ``user_int_feature_specs``;
        # required only when ``enable_hist_users=True``. The model uses these
        # positions to BORROW the relevant Embedding tables from
        # ``user_ns_tokenizer`` (shared weights, same semantic space).
        hist_scalar_fid_positions: Optional[List[int]] = None,
        hist_dense_dim: int = 256,
        hist_dropout: float = 0.1,
        hist_num_heads: Optional[int] = None,  # default = num_heads
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.action_num = action_num
        self.num_queries = num_queries
        self.seq_domains = sorted(seq_vocab_sizes.keys())  # deterministic order
        self.num_sequences = len(self.seq_domains)
        self.num_time_buckets = num_time_buckets
        self.rank_mixer_mode = rank_mixer_mode
        self.use_rope = use_rope
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold
        self.ns_tokenizer_type = ns_tokenizer_type
        self.cross_ns_tokenizer = None

        # ================== NS Tokens Construction ==================

        if ns_tokenizer_type == "group":
            # Original: one NS token per group
            self.user_ns_tokenizer = GroupNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = len(user_ns_groups)

            self.item_ns_tokenizer = GroupNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = len(item_ns_groups)
        elif ns_tokenizer_type == "rankmixer":
            # RankMixer paper style: all embeddings cat → split → project
            # 0 means auto: fall back to group count
            if user_ns_tokens <= 0:
                user_ns_tokens = len(user_ns_groups)
            if item_ns_tokens <= 0:
                item_ns_tokens = len(item_ns_groups)

            # Pair feature embedder: produces a flat (B, pair_emb_dim) vector
            # that is injected into user_ns_tokenizer before LHUC.
            self.cross_ns_tokenizer = CrossRankMixerNSTokenizer(
                pair_int_feature_specs, self.d_model, emb_dim=emb_dim
            )
            _pair_emb_dim = self.cross_ns_tokenizer.out_dim

            self.user_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=user_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
                extra_emb_dim=_pair_emb_dim,
            )
            num_user_ns = user_ns_tokens

            self.item_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=item_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = item_ns_tokens
        else:
            raise ValueError(f"Unknown ns_tokenizer_type: {ns_tokenizer_type}")

        # User dense feature projection (if available)
        # Split into two NS tokens:
        #   token 1 — fid=61 (user embedding, user_emb_dim) + timestamp_hod (remaining dims)
        #   token 2 — fid=87 (history item embeddings, user_seq_num x user_seq_block_dim)
        #             aggregated via learned attention pooling over non-zero blocks
        self.has_user_dense = user_dense_dim > 0
        if self.has_user_dense:
            _hod_dim = user_dense_dim - user_emb_dim - user_seq_block_dim * user_seq_num
            self._user_emb_dim = user_emb_dim
            self._user_seq_block_dim = user_seq_block_dim
            self._user_seq_num = user_seq_num
            self.user_emb_proj = nn.Sequential(
                nn.Linear(user_emb_dim + _hod_dim, d_model),
                nn.LayerNorm(d_model),
            )
            self.user_seq_attn = nn.Linear(user_seq_block_dim, 1)
            self.user_seq_proj = nn.Sequential(
                nn.Linear(user_seq_block_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # Item dense feature projection (if available)
        self.has_item_dense = item_dense_dim > 0
        if self.has_item_dense:
            self.item_dense_proj = nn.Sequential(
                nn.Linear(item_dense_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # Total NS token count
        # cross_ns_tokenizer is now injected into user_ns, not a separate token
        self.num_ns = (
            num_user_ns
            + (2 if self.has_user_dense else 0)
            + num_item_ns
            + (1 if self.has_item_dense else 0)
        )

        # ================== Check d_model % T == 0 constraint (full mode only) ==================
        T = num_queries * self.num_sequences + self.num_ns
        if rank_mixer_mode == "full" and d_model % T != 0:
            valid_T_values = [t for t in range(1, d_model + 1) if d_model % t == 0]
            raise ValueError(
                f"d_model={d_model} must be divisible by T=num_queries*num_sequences+num_ns="
                f"{num_queries}*{self.num_sequences}+{self.num_ns}={T}. "
                f"Valid T values for d_model={d_model}: {valid_T_values}"
            )

        # ================== Seq Tokens Embedding ==================
        # seq_id_threshold decides which features inside the seq tokenizer are
        # treated as id features (they receive extra dropout). It is fully
        # independent of emb_skip_threshold (which skips Embedding creation).
        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        def _make_seq_embs(vocab_sizes):
            """Create embedding list, returning None for features skipped via
            emb_skip_threshold or with no vocab info (vs<=0)."""
            embs_raw = []
            for vs in vocab_sizes:
                skip = int(vs) <= 0 or (
                    emb_skip_threshold > 0 and int(vs) > emb_skip_threshold
                )
                if skip:
                    embs_raw.append(None)
                else:
                    embs_raw.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
            module_list = nn.ModuleList([e for e in embs_raw if e is not None])
            # Map from position index to real index in module_list (-1 if skipped)
            index_map = []
            real_idx = 0
            for e in embs_raw:
                if e is not None:
                    index_map.append(real_idx)
                    real_idx += 1
                else:
                    index_map.append(-1)
            is_id = [int(vs) > seq_id_threshold for vs in vocab_sizes]
            return module_list, index_map, is_id

        # ================== Dynamic Sequence Embeddings ==================
        self._seq_embs = nn.ModuleDict()
        self._seq_emb_index = {}  # domain -> index_map
        self._seq_is_id = {}  # domain -> is_id list
        self._seq_vocab_sizes = {}  # domain -> vocab_sizes list
        self._seq_proj = nn.ModuleDict()

        self._seq_ts_float_proj = nn.ModuleDict()

        for domain in self.seq_domains:
            vs = seq_vocab_sizes[domain]
            embs, idx_map, is_id = _make_seq_embs(vs)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id
            self._seq_vocab_sizes[domain] = vs
            self._seq_ts_float_proj[domain] = nn.Sequential(
                nn.Linear(TS_FLOAT_DIM, emb_dim),
                nn.LayerNorm(emb_dim),
            )
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear((len(vs) + 1) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )

        self.seq_gate_stat_proj = nn.ModuleDict(
            {
                domain: nn.Sequential(
                    nn.Linear(TS_STAT_DIM + 2, d_model),
                    nn.LayerNorm(d_model),
                    nn.SiLU(),
                )
                for domain in self.seq_domains
            }
        )
        self.seq_gate_score = nn.ModuleDict(
            {
                domain: nn.Sequential(
                    nn.Linear(d_model * 3, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, 1),
                )
                for domain in self.seq_domains
            }
        )
        self.seq_gate_temperature = 2.0
        self.seq_gate_uniform_alpha = 0.15
        self.last_seq_weights: Optional[torch.Tensor] = None

        # ================== Time Interval Bucket Embedding (optional) ==================
        if num_time_buckets > 0:
            self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0)

        # ================== HyFormer Components ==================
        # MultiSeqQueryGenerator
        self.query_generator = MultiSeqQueryGenerator(
            d_model=d_model,
            num_ns=self.num_ns,
            num_queries=num_queries,
            num_sequences=self.num_sequences,
            hidden_mult=hidden_mult,
        )

        # MultiSeqHyFormerBlock stack
        self.blocks = nn.ModuleList(
            [
                MultiSeqHyFormerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    num_queries=num_queries,
                    num_ns=self.num_ns,
                    num_sequences=self.num_sequences,
                    seq_encoder_type=seq_encoder_type,
                    hidden_mult=hidden_mult,
                    dropout=dropout_rate,
                    top_k=seq_top_k,
                    causal=seq_causal,
                    rank_mixer_mode=rank_mixer_mode,
                )
                for _ in range(num_hyformer_blocks)
            ]
        )

        # ================== RoPE ==================
        if use_rope:
            head_dim = d_model // num_heads
            self.rotary_emb = RotaryEmbedding(dim=head_dim, base=rope_base)
        else:
            self.rotary_emb = None

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Dropout
        self.emb_dropout = nn.Dropout(dropout_rate)

        # Classifier
        self.clsfier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num),
        )

        # ── Item-history-user module (optional add-on) ──
        self.enable_hist_users = enable_hist_users
        if enable_hist_users:
            if hist_scalar_fid_positions is None:
                raise ValueError(
                    "enable_hist_users=True requires hist_scalar_fid_positions"
                )
            # Resolve vocab sizes AND offsets in user_int_feats for the chosen
            # scalar fid positions from user_int_feature_specs (vs, off, len).
            # Each fid must be length=1 (a true scalar), otherwise the offset
            # alone isn't enough to slice it out of user_int_feats.
            scalar_vocab_sizes: List[int] = []
            scalar_int_offsets: List[int] = []
            for pos in hist_scalar_fid_positions:
                vs, offset, length = user_int_feature_specs[pos]
                if int(length) != 1:
                    raise ValueError(
                        f"hist scalar at fid_position={pos} has length={length}; "
                        f"only scalar (length=1) features are supported as hist scalars."
                    )
                scalar_vocab_sizes.append(int(vs))
                scalar_int_offsets.append(int(offset))

            # Strict Q/KV feature alignment: current user is encoded via the
            # SAME ItemHistUserModule.encode_hist_users path as historical
            # users, using only the 7 scalar fids + dense_61. This puts Q and
            # K/V in literally the same space (same weights, same features).
            # The previous UserQueryPool over the 12 user_ns tokens is gone —
            # those tokens carry many features that hist users don't have,
            # making the cross-attention dot product asymmetric.
            self.user_query_pool = None
            self.hist_user_module = ItemHistUserModule(
                scalar_vocab_sizes=scalar_vocab_sizes,
                hist_dense_dim=hist_dense_dim,
                d_model=d_model,
                num_heads=hist_num_heads or num_heads,
                emb_dim_base=emb_dim,
                dropout=dropout_rate,
                history_dropout=hist_dropout,
            )
            # Column indices of the 7 scalar fids inside user_int_feats; used
            # at forward time to gather current user's scalars (same column
            # order as build_item_hist_users.py wrote into hist tensors).
            self.register_buffer(
                "hist_scalar_user_int_offsets",
                torch.tensor(scalar_int_offsets, dtype=torch.long),
                persistent=False,
            )
            # dense_61 in user_dense_feats lives at [0, user_emb_dim) by the
            # same convention as _encode_user_dense. Verify the size matches
            # hist_dense_dim so the encoder can be shared between sides.
            if not self.has_user_dense or int(user_emb_dim) != int(hist_dense_dim):
                raise ValueError(
                    f"enable_hist_users requires user_dense to be present and "
                    f"user_emb_dim ({user_emb_dim}) to equal hist_dense_dim "
                    f"({hist_dense_dim}); current layout incompatible."
                )
        else:
            self.user_query_pool = None
            self.hist_user_module = None

        # Initialize parameters
        self._init_params()

        # Log emb_skip_threshold filtering stats
        if emb_skip_threshold > 0:

            def _count_filtered(vocab_sizes, emb_index):
                filtered = sum(1 for idx in emb_index if idx == -1)
                return filtered, len(vocab_sizes)

            for domain in self.seq_domains:
                f, t = _count_filtered(
                    self._seq_vocab_sizes[domain], self._seq_emb_index[domain]
                )
                if f > 0:
                    logging.info(
                        f"emb_skip_threshold={emb_skip_threshold}: {domain} skipped {f}/{t} features"
                    )
            for name, tokenizer in [
                ("user_ns", self.user_ns_tokenizer),
                ("item_ns", self.item_ns_tokenizer),
            ]:
                f = sum(1 for idx in tokenizer._emb_index if idx == -1)
                t = len(tokenizer._emb_index)
                if f > 0:
                    logging.info(
                        f"emb_skip_threshold={emb_skip_threshold}: {name} skipped {f}/{t} features"
                    )

    def _init_params(self) -> None:
        """Applies Xavier initialization to all embedding weights."""
        for domain in self.seq_domains:
            for emb in self._seq_embs[domain]:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        for tokenizer in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            for emb in tokenizer.embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        if self.num_time_buckets > 0:
            nn.init.xavier_normal_(self.time_embedding.weight.data)
            self.time_embedding.weight.data[0, :] = 0

        # ItemHistUserModule owns its own scalar embedding tables (no longer
        # shared with user_ns_tokenizer); initialise them the same way.
        if self.hist_user_module is not None:
            for emb in self.hist_user_module.scalar_embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

    def reinit_high_cardinality_params(
        self, cardinality_threshold: int = 10000
    ) -> "set[int]":
        """Reinitializes only high-cardinality embeddings.

        Preserves low-cardinality and time feature embeddings.

        Args:
            cardinality_threshold: Only embeddings with vocab_size exceeding
                this value are reinitialized.

        Returns:
            A set of data_ptr() values for reinitialized parameters.
        """
        reinit_count = 0
        skip_count = 0
        reinit_ptrs = set()

        for emb_list, vocab_sizes, emb_index in [
            (self._seq_embs[d], self._seq_vocab_sizes[d], self._seq_emb_index[d])
            for d in self.seq_domains
        ]:
            for i, vs in enumerate(vocab_sizes):
                real_idx = emb_index[i]
                if real_idx == -1:
                    # Skipped by emb_skip_threshold, no embedding to reinit
                    continue
                emb = emb_list[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        for tokenizer, specs in [
            (self.user_ns_tokenizer, self.user_ns_tokenizer.feature_specs),
            (self.item_ns_tokenizer, self.item_ns_tokenizer.feature_specs),
        ]:
            for i, (vs, offset, length) in enumerate(specs):
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1:
                    continue
                emb = tokenizer.embs[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        # time_embedding is always preserved
        if self.num_time_buckets > 0:
            skip_count += 1

        logging.info(
            f"Re-initialized {reinit_count} high-cardinality Embeddings "
            f"(vocab>{cardinality_threshold}), kept {skip_count}"
        )
        return reinit_ptrs

    def get_sparse_params(self) -> List[nn.Parameter]:
        """Returns all embedding table parameters (optimized with Adagrad)."""
        sparse_params = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_params.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_params]

    def get_dense_params(self) -> List[nn.Parameter]:
        """Returns all non-embedding parameters (optimized with AdamW)."""
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def _embed_seq_domain(
        self,
        seq: torch.Tensor,
        sideinfo_embs: nn.ModuleList,
        proj: nn.Module,
        is_id: List[bool],
        emb_index: List[int],
        time_bucket_ids: torch.Tensor,
        ts_float_feats: torch.Tensor,
        ts_float_proj: nn.Module,
    ) -> torch.Tensor:
        """Embeds a sequence domain by concatenating sideinfo embeddings and projecting to d_model."""
        B, S, L = seq.shape
        emb_list = []
        for i in range(S):
            real_idx = emb_index[i] if i < len(emb_index) else -1
            if real_idx == -1:
                # Feature skipped by emb_skip_threshold: output zero vector
                emb_list.append(seq.new_zeros(B, L, self.emb_dim, dtype=torch.float))
            else:
                emb = sideinfo_embs[real_idx]
                e = emb(seq[:, i, :])  # (B, L, emb_dim)
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)

        ts_emb = ts_float_proj(
            ts_float_feats.transpose(1, 2).contiguous()
        )  # (B, L, emb_dim)
        emb_list.append(ts_emb)

        cat_emb = torch.cat(emb_list, dim=-1)  # (B, L, (S+1)*emb_dim)
        token_emb = F.gelu(proj(cat_emb))  # (B, L, D)

        # Add time bucket embedding (all-zero ids produce zero vectors via padding_idx=0)
        if self.num_time_buckets > 0:
            token_emb = token_emb + self.time_embedding(time_bucket_ids)

        return token_emb

    def _make_padding_mask(self, seq_len: torch.Tensor, max_len: int) -> torch.Tensor:
        """Generates a padding mask from sequence lengths."""
        device = seq_len.device
        idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
        return idx >= seq_len.unsqueeze(1)  # (B, max_len)

    def _encode_user_dense(self, user_dense_feats: torch.Tensor):
        """Encode user_dense into two NS tokens.

        Token 1 — fid=61 (user embedding) + timestamp_hod:
            projected via user_emb_proj.
        Token 2 — fid=87 (up to user_seq_num x user_seq_block_dim history item embeddings):
            each non-zero block is scored by user_seq_attn, masked softmax over valid
            blocks, then attention-weighted sum projected via user_seq_proj.

        Returns a list of two tensors, each (B, 1, D).
        """
        B = user_dense_feats.size(0)
        seq_start = self._user_emb_dim
        seq_end = seq_start + self._user_seq_block_dim * self._user_seq_num

        # --- token 1: user embedding + hod ---
        emb_input = torch.cat(
            [user_dense_feats[:, :seq_start], user_dense_feats[:, seq_end:]], dim=-1
        )
        tok1 = F.silu(self.user_emb_proj(emb_input)).unsqueeze(1)  # (B, 1, D)

        # --- token 2: history item embeddings with attention pooling ---
        seq_vecs = user_dense_feats[:, seq_start:seq_end].view(
            B, self._user_seq_num, self._user_seq_block_dim
        )  # (B, N, 32)
        valid = seq_vecs.norm(dim=-1) > 0.1          # (B, N)
        has_valid = valid.any(dim=-1, keepdim=True)  # (B, 1)
        scores = self.user_seq_attn(seq_vecs).squeeze(-1)  # (B, N)
        scores = scores.masked_fill(~valid, float("-inf"))
        # rows where ALL blocks are zero (null user) → replace -inf with 0
        # so softmax gives uniform weights over zero vectors → zero output
        scores = scores.masked_fill(~has_valid, 0.0)
        attn = torch.softmax(scores, dim=-1)          # (B, N)
        seq_agg = (seq_vecs * attn.unsqueeze(-1)).sum(dim=1)  # (B, 32)
        tok2 = F.silu(self.user_seq_proj(seq_agg)).unsqueeze(1)  # (B, 1, D)

        return [tok1, tok2]

    def _time_bucket_pool(
        self,
        time_bucket_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Mask-mean pool per-position time bucket embeddings into (B, D)."""
        B = time_bucket_ids.size(0)
        if self.num_time_buckets <= 0:
            return time_bucket_ids.new_zeros(B, self.d_model, dtype=dtype)

        valid = ~padding_mask
        time_emb = self.time_embedding(time_bucket_ids)  # (B, L, D)
        mask = valid.unsqueeze(-1).to(dtype=time_emb.dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (time_emb * mask).sum(dim=1) / denom
        has_valid = valid.any(dim=1, keepdim=True)
        pooled = torch.where(has_valid, pooled, torch.zeros_like(pooled))
        return pooled.to(dtype=dtype)

    def _domain_sequence_gate(
        self,
        q_tokens_list: list,
        seq_masks_list: list,
        seq_lens_list: list,
        seq_time_buckets_list: list,
        seq_ts_stat_feats_list: list,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse domain-specific sequence representations with learned gates."""
        seq_reprs = []
        scores = []
        valid_domains = []

        for i, domain in enumerate(self.seq_domains):
            q = q_tokens_list[i]
            B = q.size(0)
            seq_repr = q.mean(dim=1)  # (B, D)
            dtype = seq_repr.dtype

            valid_len = seq_lens_list[i].to(device=q.device)
            total_len = max(1, int(seq_masks_list[i].shape[1]))
            coverage = (valid_len.to(dtype=dtype) / float(total_len)).unsqueeze(-1)
            log_len = torch.log1p(valid_len.to(dtype=dtype)).unsqueeze(-1)

            stats = seq_ts_stat_feats_list[i].to(device=q.device, dtype=dtype)
            stat_input = torch.cat([stats, coverage, log_len], dim=-1)
            stat_emb = self.seq_gate_stat_proj[domain](stat_input).to(dtype=dtype)
            time_pool = self._time_bucket_pool(
                seq_time_buckets_list[i].to(device=q.device),
                seq_masks_list[i].to(device=q.device),
                dtype,
            )

            gate_input = torch.cat([seq_repr, stat_emb, time_pool], dim=-1)
            score = self.seq_gate_score[domain](gate_input)
            valid_domain = valid_len > 0

            seq_reprs.append(seq_repr)
            scores.append(score)
            valid_domains.append(valid_domain)

        seq_repr_stack = torch.stack(seq_reprs, dim=1)  # (B, S, D)
        scores_t = torch.cat(scores, dim=1)             # (B, S)
        valid_mask = torch.stack(valid_domains, dim=1)  # (B, S)
        has_valid = valid_mask.any(dim=1, keepdim=True)

        masked_scores = scores_t.masked_fill(~valid_mask, float("-inf"))
        safe_scores = torch.where(has_valid, masked_scores, torch.zeros_like(scores_t))
        weights = torch.softmax(safe_scores / self.seq_gate_temperature, dim=1)
        weights = torch.where(has_valid, weights, torch.zeros_like(weights))
        uniform = valid_mask.to(dtype=weights.dtype)
        uniform = uniform / uniform.sum(dim=1, keepdim=True).clamp(min=1.0)
        weights = (
            (1.0 - self.seq_gate_uniform_alpha) * weights
            + self.seq_gate_uniform_alpha * uniform
        )
        weights = torch.where(has_valid, weights, torch.zeros_like(weights))

        output = torch.sum(weights.unsqueeze(-1) * seq_repr_stack, dim=1)  # (B, D)
        return self.output_proj(output), weights

    def _run_multi_seq_blocks(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_masks_list: list,
        seq_lens_list: list,
        seq_time_buckets_list: list,
        seq_ts_stat_feats_list: list,
        apply_dropout: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs the block stack and fuses sequence Q tokens with domain gates."""
        if apply_dropout:
            q_tokens_list = [self.emb_dropout(q) for q in q_tokens_list]
            ns_tokens = self.emb_dropout(ns_tokens)
            seq_tokens_list = [self.emb_dropout(s) for s in seq_tokens_list]

        curr_qs = q_tokens_list
        curr_ns = ns_tokens
        curr_seqs = seq_tokens_list
        curr_masks = seq_masks_list

        for block in self.blocks:
            # Precompute RoPE cos/sin for each sequence
            rope_cos_list = None
            rope_sin_list = None
            if self.rotary_emb is not None:
                rope_cos_list = []
                rope_sin_list = []
                device = curr_seqs[0].device
                for seq_i in curr_seqs:
                    seq_len = seq_i.shape[1]
                    cos, sin = self.rotary_emb(seq_len, device)
                    rope_cos_list.append(cos)
                    rope_sin_list.append(sin)

            curr_qs, curr_ns, curr_seqs, curr_masks = block(
                q_tokens_list=curr_qs,
                ns_tokens=curr_ns,
                seq_tokens_list=curr_seqs,
                seq_padding_masks=curr_masks,
                rope_cos_list=rope_cos_list,
                rope_sin_list=rope_sin_list,
            )

        return self._domain_sequence_gate(
            curr_qs,
            curr_masks,
            seq_lens_list,
            seq_time_buckets_list,
            seq_ts_stat_feats_list,
        )

    def forward(
        self, inputs: ModelInput, return_output: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Runs the forward pass of the PCVRHyFormer model."""
        # 1. NS tokens: grouped projection
        # Pair features are embedded first and injected into user_ns before LHUC.
        pair_emb = None
        if self.cross_ns_tokenizer is not None:
            pair_emb = self.cross_ns_tokenizer(
                inputs.pair_int_feats, inputs.pair_dense_feats
            )  # (B, pair_emb_dim)

        user_ns = self.user_ns_tokenizer(
            inputs.user_int_feats, extra_emb=pair_emb
        )  # (B, num_user_groups, D)
        item_ns = self.item_ns_tokenizer(
            inputs.item_int_feats
        )  # (B, num_item_groups, D)

        ns_parts = [user_ns]
        if self.has_user_dense:
            ns_parts.extend(self._encode_user_dense(inputs.user_dense_feats))
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok = F.silu(
                self.item_dense_proj(inputs.item_dense_feats)
            ).unsqueeze(
                1
            )  # (B, 1, D)
            ns_parts.append(item_dense_tok)

        ns_tokens = torch.cat(ns_parts, dim=1)  # (B, num_ns, D)

        # 2. Embed each sequence domain (dynamic)
        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain],
                self._seq_proj[domain],
                self._seq_is_id[domain],
                self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
                inputs.seq_ts_float_feats[domain],
                self._seq_ts_float_proj[domain],
            )
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(
                inputs.seq_lens[domain], inputs.seq_data[domain].shape[2]
            )
            seq_masks_list.append(mask)

        # 3. Generate independent Q tokens per sequence via MultiSeqQueryGenerator
        q_tokens_list = self.query_generator(
            ns_tokens,
            seq_tokens_list,
            seq_masks_list,
            [inputs.seq_ts_stat_feats[d] for d in self.seq_domains],
        )

        # 4.0 Optional item-history-user encoding: pool the PRE-HyFormer user
        # NS tokens to one query, cross-attend over the per-row pos/neg pools,
        # and forward the two pool reprs into ``_run_multi_seq_blocks`` so
        # they participate in the same softmax-weighted merge as the 4 seq
        # domain reprs (6-way softmax instead of 4-way).
        # 4. Dropout + MultiSeqHyFormerBlock stack + 4-way domain gate
        output, seq_weights = self._run_multi_seq_blocks(
            q_tokens_list,
            ns_tokens,
            seq_tokens_list,
            seq_masks_list,
            [inputs.seq_lens[d] for d in self.seq_domains],
            [inputs.seq_time_buckets[d] for d in self.seq_domains],
            [inputs.seq_ts_stat_feats[d] for d in self.seq_domains],
            apply_dropout=self.training,
        )
        self.last_seq_weights = seq_weights.detach()

        # 4.5 Optional item-history-user gated-residual fusion.
        # Q (current user) and K/V (historical users) are encoded by the SAME
        # encode_hist_users path, so they share an aligned space. The hist
        # module returns a gated delta which is added on top of the backbone.
        if self.hist_user_module is not None:
            if inputs.hist_pos_scalars is None:
                raise ValueError(
                    "Model was built with enable_hist_users=True but received "
                    "ModelInput.hist_pos_scalars=None. The dataset must be "
                    "constructed with hist_users_dir set to the same lookup "
                    "directory the checkpoint was trained against."
                )
            cur_scalars = inputs.user_int_feats.index_select(
                1, self.hist_scalar_user_int_offsets
            )
            cur_dense = inputs.user_dense_feats[
                :, : self.hist_user_module.dense_proj[0].in_features
            ]
            hist_delta = self.hist_user_module(
                cur_scalars,
                cur_dense,
                inputs.hist_pos_scalars,
                inputs.hist_pos_dense,
                inputs.hist_pos_lens,
                inputs.hist_neg_scalars,
                inputs.hist_neg_dense,
                inputs.hist_neg_lens,
            )
            output = output + hist_delta

        # 5. Classifier
        logits = self.clsfier(output)  # (B, action_num)
        if return_output:
            return logits, output
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs inference without dropout, returning both logits and embeddings."""
        # Reuses forward logic but without dropout
        pair_emb = None
        if self.cross_ns_tokenizer is not None:
            pair_emb = self.cross_ns_tokenizer(
                inputs.pair_int_feats, inputs.pair_dense_feats
            )  # (B, pair_emb_dim)

        user_ns = self.user_ns_tokenizer(inputs.user_int_feats, extra_emb=pair_emb)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)

        ns_parts = [user_ns]
        if self.has_user_dense:
            ns_parts.extend(self._encode_user_dense(inputs.user_dense_feats))
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok = F.silu(
                self.item_dense_proj(inputs.item_dense_feats)
            ).unsqueeze(1)
            ns_parts.append(item_dense_tok)

        ns_tokens = torch.cat(ns_parts, dim=1)

        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain],
                self._seq_proj[domain],
                self._seq_is_id[domain],
                self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
                inputs.seq_ts_float_feats[domain],
                self._seq_ts_float_proj[domain],
            )
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(
                inputs.seq_lens[domain], inputs.seq_data[domain].shape[2]
            )
            seq_masks_list.append(mask)

        q_tokens_list = self.query_generator(
            ns_tokens,
            seq_tokens_list,
            seq_masks_list,
            [inputs.seq_ts_stat_feats[d] for d in self.seq_domains],
        )

        output, seq_weights = self._run_multi_seq_blocks(
            q_tokens_list,
            ns_tokens,
            seq_tokens_list,
            seq_masks_list,
            [inputs.seq_lens[d] for d in self.seq_domains],
            [inputs.seq_time_buckets[d] for d in self.seq_domains],
            [inputs.seq_ts_stat_feats[d] for d in self.seq_domains],
            apply_dropout=False,
        )
        self.last_seq_weights = seq_weights.detach()

        if self.hist_user_module is not None:
            cur_scalars = inputs.user_int_feats.index_select(
                1, self.hist_scalar_user_int_offsets
            )
            cur_dense = inputs.user_dense_feats[
                :, : self.hist_user_module.dense_proj[0].in_features
            ]
            hist_delta = self.hist_user_module(
                cur_scalars,
                cur_dense,
                inputs.hist_pos_scalars,
                inputs.hist_pos_dense,
                inputs.hist_pos_lens,
                inputs.hist_neg_scalars,
                inputs.hist_neg_dense,
                inputs.hist_neg_lens,
            )
            output = output + hist_delta

        logits = self.clsfier(output)
        return logits, output

"""
Conformer block with pervasive FiLM conditioning for AURA's Stegaformer embedder.

Paper key innovation — FiLM placement (corrected architecture):
    FiLM is inserted immediately after the LayerNorm of EACH of the four
    sub-modules (FF1, MHSA, Conv, FF2) in every Conformer block.

    One block (corrected):
        x = x + 0.5 * FF1(LayerNorm(x) → FiLM_0 → Linear → SiLU → Linear)
        x = x + MHSA(LayerNorm(x) → FiLM_1 → MultiheadAttention)
        x = x + Conv(LayerNorm(x) → FiLM_2 → PointwiseConv → GLU → DepthwiseConv → BN → SiLU → PointwiseConv)
        x = x + 0.5 * FF2(LayerNorm(x) → FiLM_3 → Linear → SiLU → Linear)
        x = LayerNorm(x)                    # final block norm

    This gives 4 × 8 = 32 FiLM applications total, vs. only 8 in the naive
    "FiLM at block end" formulation. The deep, pervasive conditioning is the
    reason AURA achieves SOTA robustness against microphone re-recording
    and other extreme transformations.

FiLM generator (corrected — unique vectors, no reuse):
    message [B, 32]
    → Embedding(64, d_model) → sum-pool → msg_repr [B, d_model]
    → gamma_proj: Linear(d_model → n_blocks * n_submodules * d_model)
    → beta_proj:  Linear(d_model → n_blocks * n_submodules * d_model)
    → reshape → gamma [B, n_blocks, n_submodules, d_model]
                beta  [B, n_blocks, n_submodules, d_model]

    Each of the 8 × 4 = 32 (block, sub-module) positions gets its own unique
    (gamma, beta) pair — 64 independent d_model-dimensional vectors total.
    Sharing a single vector across all positions (the naive approach) destroys
    the network's ability to learn hierarchical message representations.

Gradient checkpointing:
    Each ConformerBlock forward is wrapped with torch.utils.checkpoint when
    cfg.use_gradient_checkpointing=True and the model is in training mode.
    This recomputes block activations during backward instead of caching them,
    saving ~60% VRAM at the cost of ~30% extra compute. Essential on V100 16 GB.

Confirmed hyperparameters:
    d_model=512, n_heads=8, conv_kernel=31, dropout=0.1, ff_expansion=4, n_blocks=8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ConformerConfig, MessageConfig


# ── Sub-modules (each accepts gamma/beta for its own FiLM position) ───────────

class FeedForwardModule(nn.Module):
    """
    Position-wise Feed-Forward sub-module with integrated FiLM conditioning.

    Corrected structure (FiLM immediately after LayerNorm):
        LayerNorm → FiLM(gamma, beta) → Linear(d→4d) → SiLU → Dropout
                  → Linear(4d→d) → Dropout

    FiLM scales and shifts the normalized features before the linear
    transformation, letting the message control what information each
    frequency region amplifies or suppresses.

    Applied with 0.5 half-step scaling at the ConformerBlock level.

    Args:
        d_model:      hidden dimension (512)
        ff_expansion: inner dim multiplier (4 → inner = 2048)
        dropout:      dropout probability (0.1)
    """

    def __init__(self, d_model: int, ff_expansion: int, dropout: float):
        super().__init__()
        inner_dim = d_model * ff_expansion

        self.norm     = nn.LayerNorm(d_model)
        self.linear1  = nn.Linear(d_model, inner_dim)
        self.act      = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2  = nn.Linear(inner_dim, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x:     torch.Tensor,
        gamma: torch.Tensor,
        beta:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:     [B, T, d_model]  input sequence
            gamma: [B, d_model]     FiLM scale for this sub-module position
            beta:  [B, d_model]     FiLM shift for this sub-module position
        Returns:
            out: [B, T, d_model]    branch output (before residual + 0.5 scaling)
        """
        out = self.norm(x)
        # FiLM: gamma/beta are [B, d_model]; unsqueeze(1) broadcasts over T
        out = gamma.unsqueeze(1) * out + beta.unsqueeze(1)
        out = self.linear1(out)
        out = self.act(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.dropout2(out)
        return out


class MultiHeadSelfAttentionModule(nn.Module):
    """
    Multi-Head Self-Attention sub-module with integrated FiLM conditioning.

    Corrected structure (FiLM immediately after LayerNorm):
        LayerNorm → FiLM(gamma, beta) → MultiheadAttention(Q=K=V) → Dropout

    FiLM conditions the normalized query/key/value features before attention
    is computed, allowing the message to steer which temporal patterns the
    attention focuses on.

    Args:
        d_model: hidden dimension (512)
        n_heads: number of attention heads (8, head_dim = 64)
        dropout: dropout on attention weights (0.1)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.norm    = nn.LayerNorm(d_model)
        self.attn    = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x:     torch.Tensor,
        gamma: torch.Tensor,
        beta:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:     [B, T, d_model]
            gamma: [B, d_model]
            beta:  [B, d_model]
        Returns:
            out: [B, T, d_model]  branch output (before residual)
        """
        out = self.norm(x)
        out = gamma.unsqueeze(1) * out + beta.unsqueeze(1)     # FiLM before attn
        out, _ = self.attn(out, out, out, need_weights=False)
        out = self.dropout(out)
        return out


class ConvolutionModule(nn.Module):
    """
    Convolution sub-module with integrated FiLM conditioning.

    Corrected structure (FiLM immediately after LayerNorm):
        LayerNorm → FiLM(gamma, beta) → PointwiseConv(d→2d) → GLU
                  → DepthwiseConv(k=31) → BatchNorm → SiLU
                  → PointwiseConv(d→d) → Dropout

    FiLM conditions the normalized features before the convolution operations,
    letting the message modulate the local-context extraction behaviour.

    Args:
        d_model:     hidden dimension (512)
        kernel_size: depthwise conv kernel (31, padding=15 for same-length output)
        dropout:     dropout probability (0.1)
    """

    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        assert kernel_size % 2 == 1, (
            f"kernel_size must be odd for same-length output, got {kernel_size}"
        )
        padding = (kernel_size - 1) // 2

        self.norm = nn.LayerNorm(d_model)

        # Pointwise expand: d → 2d (for GLU)
        self.pointwise_expand = nn.Conv1d(
            d_model, 2 * d_model, kernel_size=1
        )

        # Depthwise conv: local context (~330 ms at 48 kHz / hop 512)
        self.depthwise = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size, padding=padding, groups=d_model,
        )

        self.bn  = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()

        # Pointwise contract: d → d
        self.pointwise_contract = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x:     torch.Tensor,
        gamma: torch.Tensor,
        beta:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:     [B, T, d_model]
            gamma: [B, d_model]
            beta:  [B, d_model]
        Returns:
            out: [B, T, d_model]  branch output (before residual)
        """
        out = self.norm(x)
        out = gamma.unsqueeze(1) * out + beta.unsqueeze(1)     # FiLM before conv

        out = out.transpose(1, 2)                              # [B, d_model, T]
        out = self.pointwise_expand(out)                       # [B, 2d, T]
        out = F.glu(out, dim=1)                                # [B, d, T]
        out = self.depthwise(out)                              # [B, d, T]
        out = self.bn(out)
        out = self.act(out)
        out = self.pointwise_contract(out)                     # [B, d, T]
        out = self.dropout(out)
        out = out.transpose(1, 2)                              # [B, T, d_model]
        return out


# ── FiLM Generator ────────────────────────────────────────────────────────────

class FiLMGenerator(nn.Module):
    """
    Converts a binary message to unique FiLM (gamma, beta) vectors for
    every sub-module in every ConformerBlock.

    For n_blocks=8 blocks, each with n_submodules=4 sub-modules:
        Total positions: 8 × 4 = 32
        Total vectors:   32 × 2 (gamma + beta) = 64 independent d_model vectors

    Architecture:
        message [B, n_bits]
        → Embedding(2*n_bits, d_model) indexed by (2*i + bit_value)
        → sum over n_bits → msg_repr [B, d_model]
        → gamma_proj: Linear(d_model → n_blocks * n_submodules * d_model)
        → beta_proj:  Linear(d_model → n_blocks * n_submodules * d_model)
        → reshape → gamma [B, n_blocks, n_submodules, d_model]
        → reshape → beta  [B, n_blocks, n_submodules, d_model]

    Each (block_i, submodule_j) position gets its own learned (gamma_ij, beta_ij).
    Sharing a single pair across all positions collapses the network's ability
    to learn hierarchical message representations — a critical correctness bug.

    Args:
        n_bits:       number of message bits (32)
        d_model:      hidden dimension (512)
        n_blocks:     number of Conformer blocks (8)
        n_submodules: FiLM applications per block (4)
    """

    def __init__(
        self,
        n_bits:       int,
        d_model:      int,
        n_blocks:     int,
        n_submodules: int,
    ):
        super().__init__()
        self.n_bits       = n_bits
        self.d_model      = d_model
        self.n_blocks     = n_blocks
        self.n_submodules = n_submodules

        n_positions = n_blocks * n_submodules       # 32

        # 2*n_bits = 64 embedding rows: index 2i = bit i is 0, index 2i+1 = bit i is 1
        self.embedding  = nn.Embedding(2 * n_bits, d_model)

        # Project message representation to ALL gamma/beta values at once
        # Output reshaped to [B, n_blocks, n_submodules, d_model]
        self.gamma_proj = nn.Linear(d_model, n_positions * d_model)
        self.beta_proj  = nn.Linear(d_model, n_positions * d_model)

    def forward(
        self, message: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            message: [B, n_bits]  binary tensor, values in {0, 1}

        Returns:
            gamma: [B, n_blocks, n_submodules, d_model]  unique scale per position
            beta:  [B, n_blocks, n_submodules, d_model]  unique shift per position
        """
        B, n_bits = message.shape
        assert n_bits == self.n_bits, f"Expected {self.n_bits} bits, got {n_bits}"

        # Lookup: for bit i with value v → embedding index = 2*i + v
        base    = (2 * torch.arange(n_bits, device=message.device)).unsqueeze(0).expand(B, -1)
        indices = base + message.long()                     # [B, n_bits]

        msg_repr = self.embedding(indices).sum(dim=1)       # [B, d_model]

        B_d, Nb, Ns, D = B, self.n_blocks, self.n_submodules, self.d_model

        gamma = self.gamma_proj(msg_repr).view(B_d, Nb, Ns, D)  # [B, 8, 4, 512]
        beta  = self.beta_proj(msg_repr).view(B_d, Nb, Ns, D)   # [B, 8, 4, 512]

        return gamma, beta


# ── Conformer Block ───────────────────────────────────────────────────────────

class ConformerBlock(nn.Module):
    """
    One Conformer block with pervasive FiLM conditioning (corrected architecture).

    FiLM is applied inside each of the 4 sub-modules, immediately after
    that sub-module's own LayerNorm.  Each sub-module gets a different
    (gamma, beta) pair from the FiLMGenerator.

    Full forward pass:
        x = x + 0.5 * FF1(LN(x) → FiLM_0 → ...)
        x = x + MHSA(LN(x) → FiLM_1 → ...)
        x = x + Conv(LN(x) → FiLM_2 → ...)
        x = x + 0.5 * FF2(LN(x) → FiLM_3 → ...)
        x = LayerNorm(x)                           # final block norm

    Args:
        cfg: ConformerConfig
    """

    # Sub-module index constants for readability
    FF1_IDX  = 0
    MHSA_IDX = 1
    CONV_IDX = 2
    FF2_IDX  = 3

    def __init__(self, cfg: ConformerConfig):
        super().__init__()
        d = cfg.d_model

        self.ff1  = FeedForwardModule(d, cfg.ff_expansion, cfg.dropout)
        self.attn = MultiHeadSelfAttentionModule(d, cfg.n_heads, cfg.dropout)
        self.conv = ConvolutionModule(d, cfg.conv_kernel_size, cfg.dropout)
        self.ff2  = FeedForwardModule(d, cfg.ff_expansion, cfg.dropout)
        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        x:     torch.Tensor,
        gamma: torch.Tensor,
        beta:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:     [B, T, d_model]              input sequence
            gamma: [B, n_submodules, d_model]   unique scale per sub-module
            beta:  [B, n_submodules, d_model]   unique shift per sub-module
                   (sliced from [B, n_blocks, n_submodules, d_model] by backbone)

        Returns:
            x: [B, T, d_model]
        """
        x = x + 0.5 * self.ff1(x,  gamma[:, self.FF1_IDX],  beta[:, self.FF1_IDX])
        x = x +       self.attn(x,  gamma[:, self.MHSA_IDX], beta[:, self.MHSA_IDX])
        x = x +       self.conv(x,  gamma[:, self.CONV_IDX],  beta[:, self.CONV_IDX])
        x = x + 0.5 * self.ff2(x,  gamma[:, self.FF2_IDX],  beta[:, self.FF2_IDX])
        x = self.norm(x)
        return x


# ── Stacked Conformer with pervasive FiLM ─────────────────────────────────────

class StegaformerBackbone(nn.Module):
    """
    Stack of N ConformerBlocks with pervasive, unique FiLM conditioning
    and optional gradient checkpointing for V100 16 GB memory efficiency.

    The FiLMGenerator produces unique (gamma, beta) pairs for every
    (block, sub-module) position in one forward call, then dispatches
    the per-block slice to each ConformerBlock.

    Gradient checkpointing:
        When cfg.use_gradient_checkpointing=True and the model is training,
        each block's forward is wrapped with torch.utils.checkpoint.checkpoint.
        Activations are NOT stored during the forward pass — they are
        recomputed on demand during backward.
        Savings: ~60% VRAM.  Cost: ~30% extra compute per step.

    Args:
        cfg_conformer: ConformerConfig
        cfg_message:   MessageConfig
    """

    def __init__(
        self,
        cfg_conformer: ConformerConfig = ConformerConfig(),
        cfg_message:   MessageConfig   = MessageConfig(),
    ):
        super().__init__()

        self.use_gradient_checkpointing = cfg_conformer.use_gradient_checkpointing

        self.film_generator = FiLMGenerator(
            n_bits       = cfg_message.n_bits,
            d_model      = cfg_conformer.d_model,
            n_blocks     = cfg_conformer.n_blocks,
            n_submodules = cfg_conformer.n_film_per_block,
        )

        self.blocks = nn.ModuleList([
            ConformerBlock(cfg_conformer)
            for _ in range(cfg_conformer.n_blocks)
        ])

    def forward(
        self,
        x:       torch.Tensor,
        message: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:       [B, T, d_model]
            message: [B, n_bits]  binary, values in {0, 1}

        Returns:
            x: [B, T, d_model]
        """
        # Compute ALL unique FiLM vectors in one projection
        gamma_all, beta_all = self.film_generator(message)
        # gamma_all: [B, n_blocks, n_submodules, d_model]
        # beta_all:  [B, n_blocks, n_submodules, d_model]

        for i, block in enumerate(self.blocks):
            gamma_i = gamma_all[:, i]    # [B, n_submodules, d_model]
            beta_i  = beta_all[:, i]     # [B, n_submodules, d_model]

            if self.use_gradient_checkpointing and self.training:
                # Recompute block activations on backward instead of storing
                x = checkpoint(block, x, gamma_i, beta_i, use_reentrant=False)
            else:
                x = block(x, gamma_i, beta_i)

        return x

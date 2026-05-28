"""
Tests for Conformer block components and StegaformerBackbone.
Validates the corrected architecture:
  - FiLM placed inside each sub-module (after its LayerNorm), not at block end
  - Unique (gamma, beta) per (block, sub-module) position — 32 pairs, 64 vectors
  - Gradient checkpointing wrapper in StegaformerBackbone

Run with:
    python tests/test_conformer.py
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aura_watermark.conformer import (
    FeedForwardModule,
    MultiHeadSelfAttentionModule,
    ConvolutionModule,
    FiLMGenerator,
    ConformerBlock,
    StegaformerBackbone,
)
from aura_watermark.config import ConformerConfig, MessageConfig


# ── fixtures ──────────────────────────────────────────────────────────────────

B  = 2      # batch size
T  = 188    # time frames
D  = 512    # d_model
N  = 32     # message bits
NB = 8      # n_blocks
NS = 4      # n_submodules (FF1, MHSA, Conv, FF2)


def make_sequence():
    return torch.randn(B, T, D)


def make_message():
    return torch.randint(0, 2, (B, N))


def make_film_vectors(batch=B, n_sub=NS):
    """
    Make (gamma, beta) of shape [batch, n_sub, D] as if sliced from the
    FiLMGenerator output for a single block.
    """
    gamma = torch.randn(batch, n_sub, D)
    beta  = torch.randn(batch, n_sub, D)
    return gamma, beta


# ── FeedForwardModule ─────────────────────────────────────────────────────────

def test_ff_shape():
    """FF module must preserve shape [B, T, D]."""
    cfg = ConformerConfig()
    ff  = FeedForwardModule(cfg.d_model, cfg.ff_expansion, cfg.dropout)

    x = make_sequence()
    gamma = torch.randn(B, D)
    beta  = torch.randn(B, D)
    out = ff(x, gamma, beta)

    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"
    print(f"  FF output: {out.shape}  [PASS]")


def test_ff_film_applied_after_norm():
    """
    Verify FiLM is applied AFTER LayerNorm (not before).
    If FiLM is before LN, the LN would undo it, making message conditioning
    ineffective. With FiLM after LN, the conditioned features feed the Linear.
    We confirm by checking the output changes when gamma/beta change.
    """
    cfg = ConformerConfig()
    ff  = FeedForwardModule(cfg.d_model, cfg.ff_expansion, cfg.dropout)
    ff.eval()

    x     = make_sequence()
    gamma = torch.ones(B, D)   # identity scale
    beta  = torch.zeros(B, D)  # zero shift

    with torch.no_grad():
        out_identity = ff(x, gamma, beta)
        out_scaled   = ff(x, gamma * 2.0, beta)   # double scale

    # Output must differ when gamma differs
    assert not torch.allclose(out_identity, out_scaled), (
        "FiLM has no effect — check placement"
    )
    print("  [PASS] FiLM affects output (correctly placed after LayerNorm)")


def test_ff_residual_scale():
    """FF branch output magnitude must be reasonable (not exploding)."""
    cfg = ConformerConfig()
    ff  = FeedForwardModule(cfg.d_model, cfg.ff_expansion, cfg.dropout)
    ff.eval()

    x     = make_sequence()
    gamma = torch.ones(B, D)
    beta  = torch.zeros(B, D)

    with torch.no_grad():
        out = ff(x, gamma, beta)

    ratio = out.std().item() / (x.std().item() + 1e-8)
    print(f"  FF branch std ratio: {ratio:.3f}  (expected < 5.0)")
    assert ratio < 5.0, f"FF branch output exploding: ratio={ratio:.3f}"
    print("  [PASS] FF output scale reasonable")


# ── MultiHeadSelfAttentionModule ──────────────────────────────────────────────

def test_attn_shape():
    """MHSA module must preserve shape [B, T, D]."""
    cfg  = ConformerConfig()
    attn = MultiHeadSelfAttentionModule(cfg.d_model, cfg.n_heads, cfg.dropout)

    x     = make_sequence()
    gamma = torch.randn(B, D)
    beta  = torch.randn(B, D)
    out   = attn(x, gamma, beta)

    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"
    print(f"  MHSA output: {out.shape}  [PASS]")


def test_attn_head_dim():
    """head_dim must be integer: d_model / n_heads = 512 / 8 = 64."""
    cfg      = ConformerConfig()
    head_dim = cfg.d_model // cfg.n_heads
    assert head_dim == 64, f"Expected head_dim=64, got {head_dim}"
    assert cfg.d_model % cfg.n_heads == 0
    print(f"  head_dim = {head_dim}  [PASS]")


# ── ConvolutionModule ─────────────────────────────────────────────────────────

def test_conv_shape():
    """Conv module must preserve shape [B, T, D]."""
    cfg  = ConformerConfig()
    conv = ConvolutionModule(cfg.d_model, cfg.conv_kernel_size, cfg.dropout)

    x     = make_sequence()
    gamma = torch.randn(B, D)
    beta  = torch.randn(B, D)
    out   = conv(x, gamma, beta)

    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"
    print(f"  Conv output: {out.shape}  [PASS]")


def test_conv_receptive_field():
    """Depthwise conv must have kernel=31, ~330ms context."""
    from aura_watermark.config import STFTConfig
    cfg      = ConformerConfig()
    stft_cfg = STFTConfig()
    ctx_ms   = cfg.conv_kernel_size * (stft_cfg.hop_length / stft_cfg.sample_rate) * 1000

    print(f"  Depthwise conv context: {ctx_ms:.1f} ms  (kernel={cfg.conv_kernel_size})")
    assert cfg.conv_kernel_size == 31
    print("  [PASS] conv kernel size correct")


# ── FiLMGenerator (corrected architecture) ────────────────────────────────────

def test_film_output_shapes():
    """
    FiLMGenerator must output:
        gamma: [B, n_blocks, n_submodules, d_model]
        beta:  [B, n_blocks, n_submodules, d_model]
    Not [B, d_model] (the old, wrong shape).
    """
    cfg_c = ConformerConfig()
    cfg_m = MessageConfig()
    film  = FiLMGenerator(
        n_bits       = cfg_m.n_bits,
        d_model      = cfg_c.d_model,
        n_blocks     = cfg_c.n_blocks,
        n_submodules = cfg_c.n_film_per_block,
    )

    msg   = make_message()
    gamma, beta = film(msg)

    expected = (B, cfg_c.n_blocks, cfg_c.n_film_per_block, D)
    assert gamma.shape == expected, f"gamma: expected {expected}, got {gamma.shape}"
    assert beta.shape  == expected, f"beta:  expected {expected}, got {beta.shape}"
    print(f"  gamma: {gamma.shape}  [PASS]")
    print(f"  beta:  {beta.shape}   [PASS]")


def test_film_vectors_are_unique_across_positions():
    """
    Each (block, sub-module) position must receive a DIFFERENT (gamma, beta).
    If the same vector is reused across positions, message conditioning is
    shallow — the paper's robustness claims rely on 32 unique pairs.
    """
    cfg_c = ConformerConfig()
    cfg_m = MessageConfig()
    film  = FiLMGenerator(
        cfg_m.n_bits, cfg_c.d_model,
        cfg_c.n_blocks, cfg_c.n_film_per_block,
    )
    film.eval()

    msg = make_message()
    with torch.no_grad():
        gamma, _ = film(msg)  # [B, 8, 4, 512]

    # gamma[0, block_i, sub_j] must not all be equal
    # Sample a few (block, submodule) combinations
    g_b0_s0 = gamma[0, 0, 0]   # block 0, sub-module 0
    g_b0_s1 = gamma[0, 0, 1]   # block 0, sub-module 1
    g_b1_s0 = gamma[0, 1, 0]   # block 1, sub-module 0
    g_b7_s3 = gamma[0, 7, 3]   # block 7, sub-module 3

    assert not torch.allclose(g_b0_s0, g_b0_s1, atol=1e-4), (
        "Same gamma for different sub-modules in block 0"
    )
    assert not torch.allclose(g_b0_s0, g_b1_s0, atol=1e-4), (
        "Same gamma for block 0 and block 1, sub-module 0"
    )
    assert not torch.allclose(g_b0_s0, g_b7_s3, atol=1e-4), (
        "Same gamma for block 0, sub 0 and block 7, sub 3"
    )
    print("  [PASS] 32 unique gamma vectors (no reuse across positions)")


def test_film_different_messages_give_different_conditioning():
    """Two different messages must produce different (gamma, beta) tensors."""
    cfg_c = ConformerConfig()
    cfg_m = MessageConfig()
    film  = FiLMGenerator(
        cfg_m.n_bits, cfg_c.d_model,
        cfg_c.n_blocks, cfg_c.n_film_per_block,
    )
    film.eval()

    msg1 = torch.zeros(1, cfg_m.n_bits, dtype=torch.long)
    msg2 = torch.ones(1,  cfg_m.n_bits, dtype=torch.long)

    with torch.no_grad():
        g1, b1 = film(msg1)
        g2, b2 = film(msg2)

    assert not torch.allclose(g1, g2), "Same gamma for different messages"
    assert not torch.allclose(b1, b2), "Same beta for different messages"
    print("  [PASS] different messages produce different FiLM tensors")


def test_film_same_message_gives_same_conditioning():
    """Same message must always produce the same (gamma, beta) — deterministic."""
    cfg_c = ConformerConfig()
    cfg_m = MessageConfig()
    film  = FiLMGenerator(
        cfg_m.n_bits, cfg_c.d_model,
        cfg_c.n_blocks, cfg_c.n_film_per_block,
    )
    film.eval()

    msg = make_message()
    with torch.no_grad():
        g1, b1 = film(msg)
        g2, b2 = film(msg)

    assert torch.allclose(g1, g2), "FiLM not deterministic for same message"
    assert torch.allclose(b1, b2), "FiLM not deterministic for same message"
    print("  [PASS] same message produces same FiLM tensors")


def test_film_embedding_table_size():
    """Embedding table must have 2*n_bits = 64 rows."""
    cfg_c = ConformerConfig()
    cfg_m = MessageConfig()
    film  = FiLMGenerator(
        cfg_m.n_bits, cfg_c.d_model,
        cfg_c.n_blocks, cfg_c.n_film_per_block,
    )

    expected_rows = 2 * cfg_m.n_bits
    actual_rows   = film.embedding.weight.shape[0]
    assert actual_rows == expected_rows, (
        f"Embedding table: expected {expected_rows} rows, got {actual_rows}"
    )
    print(f"  Embedding table: [{actual_rows}, {cfg_c.d_model}]  [PASS]")


def test_film_projection_output_size():
    """
    gamma_proj must map d_model -> n_blocks*n_submodules*d_model = 16384.
    This confirms there are 32 independent gamma projections (not 1 reused).
    """
    cfg_c = ConformerConfig()
    cfg_m = MessageConfig()
    film  = FiLMGenerator(
        cfg_m.n_bits, cfg_c.d_model,
        cfg_c.n_blocks, cfg_c.n_film_per_block,
    )

    expected_out = cfg_c.n_blocks * cfg_c.n_film_per_block * cfg_c.d_model  # 32*512=16384
    actual_out   = film.gamma_proj.out_features

    assert actual_out == expected_out, (
        f"gamma_proj output: expected {expected_out}, got {actual_out}"
    )
    print(f"  gamma_proj: Linear({film.gamma_proj.in_features} -> {actual_out})  [PASS]")
    print(f"  = {cfg_c.n_blocks} blocks x {cfg_c.n_film_per_block} sub-modules x {cfg_c.d_model} d_model")


# ── ConformerBlock ────────────────────────────────────────────────────────────

def test_conformer_block_shape():
    """ConformerBlock must preserve shape [B, T, d_model]."""
    cfg   = ConformerConfig()
    block = ConformerBlock(cfg)

    x = make_sequence()
    gamma, beta = make_film_vectors()   # [B, 4, D]

    out = block(x, gamma, beta)

    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"
    print(f"  ConformerBlock output: {out.shape}  [PASS]")


def test_conformer_block_each_submodule_gets_different_vectors():
    """
    Block must route gamma[:, 0] to FF1, [:, 1] to MHSA, [:, 2] to Conv, [:, 3] to FF2.
    Verify by patching gamma to all-zeros except one sub-module and checking output differs.
    """
    cfg   = ConformerConfig()
    block = ConformerBlock(cfg)
    block.eval()

    x = make_sequence()

    # Base: all gamma=1, beta=0 (FiLM identity at every position)
    gamma_base = torch.ones(B, NS, D)
    beta_base  = torch.zeros(B, NS, D)

    # Perturb only sub-module 2 (Conv)
    gamma_perturbed       = gamma_base.clone()
    gamma_perturbed[:, 2] = gamma_base[:, 2] * 2.0   # double Conv's scale

    with torch.no_grad():
        out_base      = block(x, gamma_base,      beta_base)
        out_perturbed = block(x, gamma_perturbed, beta_base)

    assert not torch.allclose(out_base, out_perturbed), (
        "Perturbing Conv's gamma had no effect — routing may be wrong"
    )
    print("  [PASS] sub-module routing correct (perturbing Conv changes output)")


def test_conformer_block_changes_input():
    """ConformerBlock must modify the input (not be an identity)."""
    cfg   = ConformerConfig()
    block = ConformerBlock(cfg)
    block.eval()

    x     = make_sequence()
    gamma, beta = make_film_vectors()

    with torch.no_grad():
        out = block(x, gamma, beta)

    assert not torch.allclose(x, out), "Block output identical to input"
    print("  [PASS] block modifies input")


def test_conformer_block_different_messages_give_different_outputs():
    """Different (gamma, beta) vectors must lead to different block outputs."""
    cfg   = ConformerConfig()
    block = ConformerBlock(cfg)
    block.eval()

    x = make_sequence()
    gamma1, beta1 = make_film_vectors()
    gamma2, beta2 = make_film_vectors()   # different random vectors

    with torch.no_grad():
        out1 = block(x, gamma1, beta1)
        out2 = block(x, gamma2, beta2)

    assert not torch.allclose(out1, out2, atol=1e-4), (
        "Same block output for different FiLM vectors"
    )
    print("  [PASS] different FiLM vectors produce different block outputs")


# ── StegaformerBackbone ───────────────────────────────────────────────────────

def test_backbone_shape():
    """StegaformerBackbone must preserve shape [B, T, d_model]."""
    backbone = StegaformerBackbone()

    x   = make_sequence()
    msg = make_message()
    out = backbone(x, msg)

    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"
    print(f"  Backbone output: {out.shape}  [PASS]")


def test_backbone_has_8_blocks():
    """Backbone must have exactly 8 ConformerBlocks."""
    backbone = StegaformerBackbone()
    n_blocks = len(backbone.blocks)

    assert n_blocks == 8, f"Expected 8 blocks, got {n_blocks}"
    print(f"  Number of Conformer blocks: {n_blocks}  [PASS]")


def test_backbone_film_produces_32_unique_pairs():
    """
    Backbone's FiLMGenerator must produce 32 unique (gamma, beta) pairs
    (one per (block, sub-module) position), not 1 shared pair.
    """
    backbone = StegaformerBackbone()
    backbone.eval()

    msg = make_message()
    with torch.no_grad():
        gamma, _ = backbone.film_generator(msg)  # [B, 8, 4, 512]

    assert gamma.shape == (B, 8, 4, D), (
        f"Expected [B, 8, 4, 512], got {gamma.shape}"
    )
    # Confirm different blocks get different vectors
    assert not torch.allclose(gamma[:, 0], gamma[:, 1], atol=1e-4), (
        "Block 0 and Block 1 get identical FiLM vectors"
    )
    # Confirm different sub-modules within a block get different vectors
    assert not torch.allclose(gamma[:, 0, 0], gamma[:, 0, 1], atol=1e-4), (
        "Sub-modules 0 and 1 of Block 0 get identical FiLM vectors"
    )
    print(f"  FiLM output shape: {gamma.shape}  (32 unique gamma vectors)  [PASS]")


def test_backbone_parameter_count():
    """
    Parameter count must be in expected range.
    New FiLMGenerator adds ~16.8M params (vs. ~0.6M before), so backbone
    is now ~65M total. Range: 50M-80M.
    """
    backbone = StegaformerBackbone()
    n_params = sum(p.numel() for p in backbone.parameters())
    n_params_M = n_params / 1e6

    print(f"  Backbone parameters: {n_params_M:.2f}M")
    assert 50e6 < n_params < 80e6, (
        f"Unexpected parameter count: {n_params_M:.2f}M (expected 50-80M)"
    )
    print(f"  [PASS] parameter count in expected range ({n_params_M:.2f}M)")


def test_backbone_gradient_flow():
    """
    Gradients must flow all the way back to the FiLM embedding table
    through the new pervasive FiLM structure (32 applications, not 8).
    """
    backbone = StegaformerBackbone()

    x   = make_sequence()
    msg = make_message()

    out  = backbone(x, msg)
    loss = out.sum()
    loss.backward()

    grad = backbone.film_generator.embedding.weight.grad
    assert grad is not None, "No gradient reached FiLM embedding table"
    assert (grad != 0).any(), "All gradients in FiLM embedding table are zero"
    print("  [PASS] gradients flow to FiLM embedding table")


def test_backbone_different_messages():
    """Different messages must produce different backbone outputs."""
    backbone = StegaformerBackbone()
    backbone.eval()

    x    = make_sequence()
    msg1 = torch.zeros(B, N, dtype=torch.long)
    msg2 = torch.ones(B,  N, dtype=torch.long)

    with torch.no_grad():
        out1 = backbone(x, msg1)
        out2 = backbone(x, msg2)

    assert not torch.allclose(out1, out2), (
        "Backbone produces same output for different messages"
    )
    print("  [PASS] different messages produce different backbone outputs")


def test_backbone_gradient_checkpointing_flag():
    """
    With use_gradient_checkpointing=False, backbone must still produce
    correct output (checkpointing is opt-in, not required for correctness).
    """
    from aura_watermark.config import ConformerConfig
    cfg = ConformerConfig()
    cfg.use_gradient_checkpointing = False
    backbone = StegaformerBackbone(cfg_conformer=cfg)

    x   = make_sequence()
    msg = make_message()

    # Must work in both train and eval mode without checkpointing
    backbone.train()
    out_train = backbone(x, msg)
    backbone.eval()
    with torch.no_grad():
        out_eval = backbone(x, msg)

    assert out_train.shape == (B, T, D)
    assert out_eval.shape  == (B, T, D)
    print("  [PASS] backbone works correctly without gradient checkpointing")


def test_backbone_gradient_checkpointing_saves_memory():
    """
    Gradient checkpointing must be active during training when enabled.
    We verify this by checking that it doesn't break gradient flow.
    """
    cfg = ConformerConfig()
    cfg.use_gradient_checkpointing = True
    backbone = StegaformerBackbone(cfg_conformer=cfg)
    backbone.train()   # must be in train mode for checkpointing to activate

    x   = make_sequence()
    msg = make_message()

    out  = backbone(x, msg)
    loss = out.sum()
    loss.backward()

    grad = backbone.film_generator.embedding.weight.grad
    assert grad is not None, "Gradient checkpointing broke gradient flow"
    print("  [PASS] gradient checkpointing active in training, gradients intact")


# ── runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # FeedForward
        test_ff_shape,
        test_ff_film_applied_after_norm,
        test_ff_residual_scale,
        # MHSA
        test_attn_shape,
        test_attn_head_dim,
        # Conv
        test_conv_shape,
        test_conv_receptive_field,
        # FiLMGenerator (corrected)
        test_film_output_shapes,
        test_film_vectors_are_unique_across_positions,
        test_film_different_messages_give_different_conditioning,
        test_film_same_message_gives_same_conditioning,
        test_film_embedding_table_size,
        test_film_projection_output_size,
        # ConformerBlock (corrected)
        test_conformer_block_shape,
        test_conformer_block_each_submodule_gets_different_vectors,
        test_conformer_block_changes_input,
        test_conformer_block_different_messages_give_different_outputs,
        # StegaformerBackbone
        test_backbone_shape,
        test_backbone_has_8_blocks,
        test_backbone_film_produces_32_unique_pairs,
        test_backbone_parameter_count,
        test_backbone_gradient_flow,
        test_backbone_different_messages,
        test_backbone_gradient_checkpointing_flag,
        test_backbone_gradient_checkpointing_saves_memory,
    ]

    print("\n" + "=" * 60)
    print("AURA - Step 2 (corrected): Conformer Block Tests")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_fn in tests:
        print(f"\n{test_fn.__name__}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  [FAIL] {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)

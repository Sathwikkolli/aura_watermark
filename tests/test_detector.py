"""
Tests for AURADecoder (watermark detector).

Run with:
    python tests/test_detector.py

Architecture under test:
    [B, 1, 1025, 188]
    → 4x (Conv2d stride=2 + GroupNorm + LeakyReLU)
    → AdaptiveAvgPool2d((1,1))
    → Linear(512→32)
    → logits [B, 32]
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aura_watermark.detector import AURADecoder
from aura_watermark.config import AURAConfig


# ── fixtures ──────────────────────────────────────────────────────────────────

B  = 2
F  = 1025    # n_freq_bins
T  = 188     # n_time_frames
N  = 32      # message bits


def make_spectrogram(batch: int = B) -> torch.Tensor:
    """Random non-negative magnitude spectrogram [B, 1, 1025, 188]."""
    return torch.abs(torch.randn(batch, 1, F, T))


def make_message(batch: int = B) -> torch.Tensor:
    """Random binary message [B, 32]."""
    return torch.randint(0, 2, (batch, N)).float()


# ── shape tests ───────────────────────────────────────────────────────────────

def test_forward_shape():
    """forward() must return logits [B, 32]."""
    detector = AURADecoder()
    s_mag = make_spectrogram()

    logits = detector(s_mag)

    assert logits.shape == (B, N), (
        f"Expected ({B}, {N}), got {logits.shape}"
    )
    print(f"  logits: {logits.shape}  [PASS]")


def test_detect_shape():
    """detect() must return probabilities [B, 32]."""
    detector = AURADecoder()
    s_mag = make_spectrogram()

    probs = detector.detect(s_mag)

    assert probs.shape == (B, N), (
        f"Expected ({B}, {N}), got {probs.shape}"
    )
    print(f"  probs: {probs.shape}  [PASS]")


def test_decode_bits_shape():
    """decode_bits() must return long tensor [B, 32] with values in {0, 1}."""
    detector = AURADecoder()
    s_mag = make_spectrogram()

    bits = detector.decode_bits(s_mag)

    assert bits.shape == (B, N), (
        f"Expected ({B}, {N}), got {bits.shape}"
    )
    assert bits.dtype == torch.long, f"Expected long, got {bits.dtype}"
    assert bits.min() >= 0 and bits.max() <= 1, (
        f"Bits out of {{0,1}}: min={bits.min()}, max={bits.max()}"
    )
    print(f"  bits: {bits.shape}, dtype={bits.dtype}  [PASS]")


# ── output range tests ────────────────────────────────────────────────────────

def test_logits_are_unbounded():
    """
    forward() outputs raw logits — they should NOT be in [0,1].
    (If all logits are in [0,1] the sigmoid is being applied in forward.)
    """
    detector = AURADecoder()
    detector.eval()

    with torch.no_grad():
        logits = detector(make_spectrogram())

    # With random weights, logits will span a wide range outside [0,1]
    has_positive_large = (logits > 1.0).any()
    has_negative = (logits < 0.0).any()

    assert has_positive_large or has_negative, (
        "All logits are in [0,1] — sigmoid may have been applied in forward(). "
        f"logits min={logits.min():.3f}, max={logits.max():.3f}"
    )
    print(f"  logits range: [{logits.min():.3f}, {logits.max():.3f}]  [PASS]")


def test_probs_in_zero_one():
    """detect() must return values strictly in (0, 1)."""
    detector = AURADecoder()
    detector.eval()

    with torch.no_grad():
        probs = detector.detect(make_spectrogram())

    assert (probs > 0).all(), f"Probs <= 0 found: min={probs.min():.6f}"
    assert (probs < 1).all(), f"Probs >= 1 found: max={probs.max():.6f}"
    print(f"  probs range: ({probs.min():.4f}, {probs.max():.4f})  [PASS]")


# ── architecture tests ────────────────────────────────────────────────────────

def test_num_conv_blocks():
    """Detector must have exactly 4 Conv blocks."""
    detector = AURADecoder()
    assert len(detector.blocks) == 4, (
        f"Expected 4 blocks, got {len(detector.blocks)}"
    )
    print(f"  Number of conv blocks: {len(detector.blocks)}  [PASS]")


def test_channel_progression():
    """
    Channel progression must match: 1 → 64 → 128 → 256 → 512.
    Verified by inspecting each block's Conv2d in/out channels.
    """
    detector = AURADecoder()
    expected_in  = [1,   64,  128, 256]
    expected_out = [64, 128,  256, 512]

    for i, block in enumerate(detector.blocks):
        conv = block[0]   # Conv2d is the first layer in the Sequential
        assert conv.in_channels == expected_in[i], (
            f"Block {i}: in_channels expected {expected_in[i]}, got {conv.in_channels}"
        )
        assert conv.out_channels == expected_out[i], (
            f"Block {i}: out_channels expected {expected_out[i]}, got {conv.out_channels}"
        )
        print(f"  Block {i}: {conv.in_channels} -> {conv.out_channels}  [OK]")
    print("  [PASS] channel progression correct")


def test_groupnorm_groups():
    """Each block must use GroupNorm with 32 groups."""
    detector = AURADecoder()
    for i, block in enumerate(detector.blocks):
        gn = block[1]   # GroupNorm is the second layer
        assert isinstance(gn, __import__('torch').nn.GroupNorm), (
            f"Block {i}: expected GroupNorm, got {type(gn)}"
        )
        assert gn.num_groups == 32, (
            f"Block {i}: expected 32 groups, got {gn.num_groups}"
        )
    print("  [PASS] all blocks use GroupNorm(32)")


def test_conv_stride_2():
    """Each block's Conv2d must have stride=2 (spatial downsampling)."""
    detector = AURADecoder()
    for i, block in enumerate(detector.blocks):
        conv = block[0]
        assert conv.stride == (2, 2), (
            f"Block {i}: expected stride (2,2), got {conv.stride}"
        )
    print("  [PASS] all convolutions use stride=2")


def test_head_shape():
    """Final linear layer must be Linear(512 → 32)."""
    detector = AURADecoder()
    head = detector.head
    assert head.in_features == 512, (
        f"head.in_features: expected 512, got {head.in_features}"
    )
    assert head.out_features == 32, (
        f"head.out_features: expected 32, got {head.out_features}"
    )
    print(f"  head: Linear({head.in_features} -> {head.out_features})  [PASS]")


# ── parameter count ───────────────────────────────────────────────────────────

def test_parameter_count():
    """
    Total detector params should be ~1-3M for the 4-block
    1->64->128->256->512 2D CNN architecture.
    (The paper reports 113.3M total; embedder ~50M; detector ~1.5M;
    BigVGAN discriminator makes up the remainder.)
    """
    detector = AURADecoder()
    counts = detector.count_parameters()

    for name, n in counts.items():
        print(f"  {name:10s}: {n/1e6:.3f}M")

    total_M = counts["total"] / 1e6
    assert 1 < total_M < 5, (
        f"Unexpected total parameter count: {total_M:.3f}M (expected 1-5M)"
    )
    print(f"  [PASS] total parameters: {total_M:.3f}M")


# ── gradient flow ─────────────────────────────────────────────────────────────

def test_gradient_flow():
    """BCEWithLogitsLoss gradients must flow back through all conv blocks."""
    import torch.nn as nn

    detector = AURADecoder()
    criterion = nn.BCEWithLogitsLoss()

    s_mag   = make_spectrogram()
    targets = make_message()

    logits = detector(s_mag)
    loss   = criterion(logits, targets)
    loss.backward()

    # Check gradients reached the first conv block
    first_conv = detector.blocks[0][0]
    assert first_conv.weight.grad is not None, "No gradient in first conv block"
    assert (first_conv.weight.grad != 0).any(), "All gradients zero in first conv"

    # Check gradients reached the final linear head
    assert detector.head.weight.grad is not None, "No gradient in head"

    print(f"  loss: {loss.item():.4f}")
    print("  [PASS] gradients flow through all layers")


# ── determinism tests ─────────────────────────────────────────────────────────

def test_deterministic_in_eval():
    """Same input → same output in eval mode."""
    detector = AURADecoder()
    detector.eval()

    s_mag = make_spectrogram(batch=1)

    with torch.no_grad():
        out1 = detector(s_mag)
        out2 = detector(s_mag)

    assert torch.allclose(out1, out2), "Detector not deterministic in eval mode"
    print("  [PASS] detector is deterministic in eval mode")


def test_different_inputs_give_different_outputs():
    """Different spectrograms must produce different logits."""
    detector = AURADecoder()
    detector.eval()

    s_mag1 = make_spectrogram(batch=1)
    s_mag2 = make_spectrogram(batch=1)   # different random tensor

    with torch.no_grad():
        out1 = detector(s_mag1)
        out2 = detector(s_mag2)

    assert not torch.allclose(out1, out2, atol=1e-5), (
        "Same output for different spectrograms"
    )
    diff = (out1 - out2).abs().mean().item()
    print(f"  Mean diff between outputs: {diff:.4f}  [PASS]")


# ── batch size flexibility ────────────────────────────────────────────────────

def test_batch_size_1():
    """Detector must work with batch_size=1 (GroupNorm is batch-agnostic)."""
    detector = AURADecoder()
    detector.eval()

    s_mag = make_spectrogram(batch=1)
    with torch.no_grad():
        logits = detector(s_mag)

    assert logits.shape == (1, N), f"Expected (1, {N}), got {logits.shape}"
    print(f"  batch=1 logits: {logits.shape}  [PASS]")


def test_batch_size_4():
    """Detector must work with larger batches."""
    detector = AURADecoder()
    detector.eval()

    s_mag = make_spectrogram(batch=4)
    with torch.no_grad():
        logits = detector(s_mag)

    assert logits.shape == (4, N), f"Expected (4, {N}), got {logits.shape}"
    print(f"  batch=4 logits: {logits.shape}  [PASS]")


# ── integration: embedder → detector pipeline ─────────────────────────────────

def test_embedder_detector_pipeline():
    """
    End-to-end: embed a message into audio, extract magnitude, detect bits.
    Verifies the shapes connect correctly through the full pipeline.
    Note: at random init, decoded bits will NOT match the original message.
    """
    from aura_watermark.embedder import StegaformerEmbedder
    from aura_watermark.stft import STFTProcessor

    cfg      = AURAConfig()
    embedder = StegaformerEmbedder(cfg)
    detector = AURADecoder(cfg)
    stft     = STFTProcessor(cfg.stft)

    embedder.eval()
    detector.eval()
    stft.eval()

    waveform = torch.randn(1, 1, 96000)
    waveform = waveform / waveform.abs().amax() * 0.9  # peak normalise
    message  = torch.randint(0, 2, (1, N))

    with torch.no_grad():
        watermarked, _, _ = embedder(waveform, message)
        s_wm, _           = stft(watermarked)           # [1, 1025, 188]
        s_wm_4d           = s_wm.unsqueeze(1)           # [1, 1, 1025, 188]
        logits            = detector(s_wm_4d)           # [1, 32]
        bits              = detector.decode_bits(s_wm_4d)

    assert logits.shape == (1, N), f"logits shape: {logits.shape}"
    assert bits.shape == (1, N), f"bits shape: {bits.shape}"
    assert bits.dtype == torch.long

    print(f"  logits: {logits.shape}, bits: {bits.shape}  [PASS]")
    print("  [PASS] embedder -> stft -> detector pipeline works")


# ── runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # Shape
        test_forward_shape,
        test_detect_shape,
        test_decode_bits_shape,
        # Output range
        test_logits_are_unbounded,
        test_probs_in_zero_one,
        # Architecture
        test_num_conv_blocks,
        test_channel_progression,
        test_groupnorm_groups,
        test_conv_stride_2,
        test_head_shape,
        # Parameters
        test_parameter_count,
        # Gradients
        test_gradient_flow,
        # Determinism
        test_deterministic_in_eval,
        test_different_inputs_give_different_outputs,
        # Batch flexibility
        test_batch_size_1,
        test_batch_size_4,
        # Integration
        test_embedder_detector_pipeline,
    ]

    print("\n" + "=" * 60)
    print("AURA - Step 4: AURADecoder Tests")
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

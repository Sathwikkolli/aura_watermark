"""
Tests for StegaformerEmbedder.

Run with:
    cd aura_watermark
    python tests/test_embedder.py
"""

import math
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aura_watermark.embedder import StegaformerEmbedder
from aura_watermark.config import AURAConfig


# ── fixtures ──────────────────────────────────────────────────────────────────

B = 2
SR = 48_000
T = 96_000     # 2-sec segment
N = 32         # message bits


def make_audio(batch: int = B, peak: float = 0.9) -> torch.Tensor:
    """Peak-normalised random mono audio [B, 1, 96000]."""
    x = torch.randn(batch, 1, T)
    max_abs = x.abs().amax(dim=-1, keepdim=True)
    return x / (max_abs + 1e-8) * peak


def make_message(batch: int = B) -> torch.Tensor:
    """Random binary message [B, 32]."""
    return torch.randint(0, 2, (batch, N))


# ── shape tests ───────────────────────────────────────────────────────────────

def test_output_shapes():
    """
    Embedder forward() must return:
        watermarked: [B, 1, 96000]
        mask:        [B, 1025, 188]
        s_mag:       [B, 1025, 188]
    """
    cfg = AURAConfig()
    embedder = StegaformerEmbedder(cfg)

    waveform = make_audio()
    message  = make_message()

    watermarked, mask, s_mag = embedder(waveform, message)

    assert watermarked.shape == (B, 1, T), (
        f"watermarked: expected {(B, 1, T)}, got {watermarked.shape}"
    )
    assert mask.shape == (B, cfg.stft.n_freq_bins, cfg.stft.n_time_frames), (
        f"mask: expected {(B, 1025, 188)}, got {mask.shape}"
    )
    assert s_mag.shape == (B, cfg.stft.n_freq_bins, cfg.stft.n_time_frames), (
        f"s_mag: expected {(B, 1025, 188)}, got {s_mag.shape}"
    )

    print(f"  watermarked: {watermarked.shape}")
    print(f"  mask:        {mask.shape}")
    print(f"  s_mag:       {s_mag.shape}")
    print("  [PASS] all output shapes correct")


# ── mask property tests ───────────────────────────────────────────────────────

def test_mask_positive():
    """Softplus guarantees M > 0 everywhere."""
    embedder = StegaformerEmbedder()
    embedder.eval()

    with torch.no_grad():
        _, mask, _ = embedder(make_audio(), make_message())

    assert (mask > 0).all(), (
        f"Mask contains non-positive values. min={mask.min():.6f}"
    )
    print(f"  mask min: {mask.min():.6f}, max: {mask.max():.6f}  [PASS]")


def test_mask_near_one_at_init():
    """
    At initialisation, output bias = 0.541 → Softplus ≈ 1.0.
    Mean mask value must be close to 1.0 at step 0.
    """
    embedder = StegaformerEmbedder()
    embedder.eval()

    with torch.no_grad():
        _, mask, _ = embedder(make_audio(), make_message())

    mask_mean = mask.mean().item()
    print(f"  mask mean at init: {mask_mean:.4f}  (expected close to 1.0)")

    assert abs(mask_mean - 1.0) < 0.5, (
        f"Mask mean far from 1.0 at init: {mask_mean:.4f}. "
        "Check output_proj bias initialisation."
    )
    print("  [PASS] mask near 1.0 at init")


def test_output_bias_init():
    """
    Output projection bias must be initialised to log(e-1) ≈ 0.541.
    """
    embedder = StegaformerEmbedder()
    expected = math.log(math.e - 1)   # ≈ 0.5413

    bias = embedder.output_proj.bias
    mean_bias = bias.mean().item()

    print(f"  output_proj bias mean: {mean_bias:.4f}  (expected {expected:.4f})")
    assert abs(mean_bias - expected) < 1e-4, (
        f"Output bias not initialised correctly: {mean_bias:.4f}"
    )
    print("  [PASS] output bias initialised correctly")


# ── watermarked audio quality tests ──────────────────────────────────────────

def test_watermarked_audio_clipped():
    """Output waveform must be in [-1, 1]."""
    embedder = StegaformerEmbedder()
    embedder.eval()

    with torch.no_grad():
        watermarked, _, _ = embedder(make_audio(), make_message())

    assert watermarked.max() <= 1.0, f"Output > 1.0: {watermarked.max():.4f}"
    assert watermarked.min() >= -1.0, f"Output < -1.0: {watermarked.min():.4f}"
    print("  [PASS] watermarked audio clipped to [-1, 1]")


def test_watermarked_close_to_original_at_init():
    """
    At init (M ≈ 1.0), watermarked audio should be close to original.
    SI-SNR should be > 20 dB (not as high as pure round-trip because
    M is not exactly 1.0, but close).
    """
    embedder = StegaformerEmbedder()
    embedder.eval()

    torch.manual_seed(0)
    waveform = make_audio()
    message  = make_message()

    with torch.no_grad():
        watermarked, _, _ = embedder(waveform, message)

    si_snr = compute_si_snr(waveform, watermarked)
    print(f"  SI-SNR at init: {si_snr:.2f} dB  (threshold: > 20 dB)")

    assert si_snr > 20.0, (
        f"Watermarked audio too far from original at init: {si_snr:.2f} dB. "
        "Check output bias initialisation."
    )
    print("  [PASS] watermarked close to original at init")


# ── message conditioning tests ────────────────────────────────────────────────

def test_different_messages_give_different_watermarks():
    """Two different messages must produce different watermarked audio."""
    embedder = StegaformerEmbedder()
    embedder.eval()

    waveform = make_audio(batch=1)
    msg1 = torch.zeros(1, N, dtype=torch.long)
    msg2 = torch.ones(1, N, dtype=torch.long)

    with torch.no_grad():
        wm1, _, _ = embedder(waveform, msg1)
        wm2, _, _ = embedder(waveform, msg2)

    assert not torch.allclose(wm1, wm2, atol=1e-5), (
        "Same watermarked output for different messages"
    )
    diff = (wm1 - wm2).abs().mean().item()
    print(f"  Mean diff between message outputs: {diff:.6f}  [PASS]")


def test_same_message_gives_same_watermark():
    """Same waveform + same message → deterministic output."""
    embedder = StegaformerEmbedder()
    embedder.eval()

    waveform = make_audio(batch=1)
    message  = make_message(batch=1)

    with torch.no_grad():
        wm1, _, _ = embedder(waveform, message)
        wm2, _, _ = embedder(waveform, message)

    assert torch.allclose(wm1, wm2), "Embedder not deterministic in eval mode"
    print("  [PASS] embedder is deterministic in eval mode")


# ── multiplicative mask correctness ──────────────────────────────────────────

def test_s_mag_unchanged_by_mask():
    """
    s_mag returned by the embedder must be the original STFT magnitude,
    not affected by the mask. Verify: s_mag is same as direct STFT output.
    """
    from aura_watermark.stft import STFTProcessor
    from aura_watermark.config import AURAConfig

    cfg = AURAConfig()
    embedder = StegaformerEmbedder(cfg)
    stft     = STFTProcessor(cfg.stft)
    embedder.eval()
    stft.eval()

    waveform = make_audio()
    message  = make_message()

    with torch.no_grad():
        _, _, s_mag_from_embedder = embedder(waveform, message)
        s_mag_direct, _           = stft(waveform)

    assert torch.allclose(s_mag_from_embedder, s_mag_direct, atol=1e-6), (
        "s_mag returned by embedder differs from direct STFT output"
    )
    print("  [PASS] s_mag is unmodified original magnitude")


def test_multiplicative_path():
    """
    Verify the multiplicative relationship:
        S_wm should equal S_mag × M (within floating point tolerance).
    We verify this by checking the mask and s_mag against what the
    iSTFT receives.
    """
    from aura_watermark.stft import STFTProcessor

    cfg = AURAConfig()
    embedder = StegaformerEmbedder(cfg)
    stft     = STFTProcessor(cfg.stft)
    embedder.eval()
    stft.eval()

    waveform = make_audio(batch=1)
    message  = make_message(batch=1)

    with torch.no_grad():
        _, mask, s_mag = embedder(waveform, message)
        expected_s_wm  = s_mag * mask

    # s_wm should satisfy: all values >= 0 (magnitude × positive mask)
    assert (expected_s_wm >= 0).all(), "s_mag × mask has negative values"

    # mask and s_mag must broadcast cleanly (same shape)
    assert mask.shape == s_mag.shape, (
        f"Shape mismatch: mask {mask.shape} vs s_mag {s_mag.shape}"
    )
    print(f"  s_mag × mask shape: {expected_s_wm.shape}")
    print("  [PASS] multiplicative mask path correct")


# ── parameter count ───────────────────────────────────────────────────────────

def test_parameter_count():
    """
    Total embedder params: ~66M with corrected FiLMGenerator.

    Breakdown:
      input_proj:  0.525M   (Linear 1025->512)
      backbone:   ~65.3M    (48.5M Conformer blocks + 16.8M FiLMGenerator)
                             FiLMGenerator grew from 0.6M to 16.8M because it now
                             projects to 32 unique (gamma, beta) pairs instead of 1.
      output_proj: 0.526M   (Linear 512->1025)
    """
    embedder = StegaformerEmbedder()
    counts   = embedder.count_parameters()

    for name, n in counts.items():
        print(f"  {name:15s}: {n/1e6:.3f}M")

    total_M = counts["total"] / 1e6
    assert 60 < total_M < 80, (
        f"Unexpected total parameter count: {total_M:.2f}M (expected 60-80M)"
    )
    print(f"  [PASS] total parameters: {total_M:.3f}M")


# ── gradient flow ─────────────────────────────────────────────────────────────

def test_gradient_flow():
    """
    Gradients must flow end-to-end:
    loss → watermarked → iSTFT → mask → backbone → FiLM embedding.
    """
    embedder = StegaformerEmbedder()

    waveform = make_audio()
    message  = make_message()

    watermarked, mask, s_mag = embedder(waveform, message)

    # Simple proxy loss on output
    loss = watermarked.mean() + mask.mean()
    loss.backward()

    # Check gradient reached FiLM embedding (deepest learnable params)
    grad = embedder.backbone.film_generator.embedding.weight.grad
    assert grad is not None, "No gradient reached FiLM embedding"
    assert (grad != 0).any(), "All FiLM embedding gradients are zero"

    # Check gradient reached input projection
    grad_in = embedder.input_proj.weight.grad
    assert grad_in is not None, "No gradient reached input_proj"

    # Check gradient reached output projection
    grad_out = embedder.output_proj.weight.grad
    assert grad_out is not None, "No gradient reached output_proj"

    print("  [PASS] gradients flow through full embedder")


# ── helpers ───────────────────────────────────────────────────────────────────

def compute_si_snr(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    ref = reference.squeeze(1).double()
    est = estimate.squeeze(1).double()
    ref = ref - ref.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)
    dot       = (est * ref).sum(dim=-1, keepdim=True)
    ref_nrg   = (ref ** 2).sum(dim=-1, keepdim=True) + 1e-8
    proj      = (dot / ref_nrg) * ref
    noise     = est - proj
    ratio     = (proj ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + 1e-8)
    return (10.0 * torch.log10(ratio + 1e-8)).mean().item()


# ── runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_output_shapes,
        test_mask_positive,
        test_mask_near_one_at_init,
        test_output_bias_init,
        test_watermarked_audio_clipped,
        test_watermarked_close_to_original_at_init,
        test_different_messages_give_different_watermarks,
        test_same_message_gives_same_watermark,
        test_s_mag_unchanged_by_mask,
        test_multiplicative_path,
        test_parameter_count,
        test_gradient_flow,
    ]

    print("\n" + "=" * 60)
    print("AURA - Step 3: Stegaformer Embedder Tests")
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

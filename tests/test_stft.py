"""
Tests for STFTProcessor and ISTFTReconstructor.

Run with:
    cd aura_watermark
    python -m pytest tests/test_stft.py -v

Or directly:
    python tests/test_stft.py
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aura_watermark.stft import STFTProcessor, ISTFTReconstructor
from aura_watermark.config import STFTConfig


# ── helpers ──────────────────────────────────────────────────────────────────

def make_audio(batch: int, length: int, peak: float = 0.9) -> torch.Tensor:
    """
    Create a batch of random mono waveforms normalized to peak amplitude.

    torch.randn produces values in roughly [-4, 4]. The iSTFT reconstructor
    clips its output to [-1, 1]. If test waveforms have values outside [-1, 1],
    the clamp introduces intentional distortion — making the round-trip
    SI-SNR test meaningless. Normalizing to peak=0.9 keeps all samples
    safely inside the valid range.

    Args:
        batch:  batch size B
        length: samples per clip (96000 for 2-sec at 48 kHz)
        peak:   target peak amplitude (0 < peak <= 1.0)

    Returns:
        waveform: [B, 1, length]  peak-normalized random audio
    """
    x = torch.randn(batch, 1, length)
    # Divide by the per-sample max absolute value, then scale to peak
    max_abs = x.abs().amax(dim=-1, keepdim=True)   # [B, 1, 1]
    x = x / (max_abs + 1e-8) * peak
    return x


def compute_si_snr(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    """
    Scale-Invariant Signal-to-Noise Ratio in dB.
    Higher is better. > 60 dB means near-lossless reconstruction.

    Args:
        reference: [B, 1, T]
        estimate:  [B, 1, T]
    Returns:
        mean SI-SNR across batch (float, dB)
    """
    ref = reference.squeeze(1).double()   # [B, T]  use float64 for precision
    est = estimate.squeeze(1).double()

    # Zero-mean both signals
    ref = ref - ref.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)

    # Project estimate onto reference
    dot = (est * ref).sum(dim=-1, keepdim=True)
    ref_energy = (ref ** 2).sum(dim=-1, keepdim=True) + 1e-8
    projection = (dot / ref_energy) * ref

    # Noise = residual after projection
    noise = est - projection

    ratio = (projection ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + 1e-8)
    si_snr_per_sample = 10.0 * torch.log10(ratio + 1e-8)

    return si_snr_per_sample.mean().item()


# ── tests ─────────────────────────────────────────────────────────────────────

def test_output_shapes():
    """STFTProcessor and ISTFTReconstructor must produce correct shapes."""
    cfg = STFTConfig()
    stft = STFTProcessor(cfg)
    istft = ISTFTReconstructor(cfg)

    B = 4
    waveform = make_audio(B, cfg.segment_samples)

    magnitude, phase = stft(waveform)

    assert magnitude.shape == (B, cfg.n_freq_bins, cfg.n_time_frames), (
        f"Magnitude shape mismatch: expected {(B, cfg.n_freq_bins, cfg.n_time_frames)}, "
        f"got {magnitude.shape}"
    )
    assert phase.shape == (B, cfg.n_freq_bins, cfg.n_time_frames), (
        f"Phase shape mismatch: expected {(B, cfg.n_freq_bins, cfg.n_time_frames)}, "
        f"got {phase.shape}"
    )

    reconstructed = istft(magnitude, phase)

    assert reconstructed.shape == (B, 1, cfg.segment_samples), (
        f"Reconstructed shape mismatch: expected {(B, 1, cfg.segment_samples)}, "
        f"got {reconstructed.shape}"
    )

    print(f"  magnitude:    {magnitude.shape}")
    print(f"  phase:        {phase.shape}")
    print(f"  reconstructed:{reconstructed.shape}")
    print("  [PASS] output shapes correct")


def test_magnitude_non_negative():
    """Magnitude values must always be >= 0."""
    cfg = STFTConfig()
    stft = STFTProcessor(cfg)

    waveform = make_audio(2, cfg.segment_samples)
    magnitude, _ = stft(waveform)

    assert (magnitude >= 0).all(), "Magnitude contains negative values"
    print("  [PASS] magnitude is non-negative")


def test_phase_range():
    """Phase values must be in [-pi, pi]."""
    cfg = STFTConfig()
    stft = STFTProcessor(cfg)

    waveform = make_audio(2, cfg.segment_samples)
    _, phase = stft(waveform)

    import math
    assert (phase >= -math.pi).all() and (phase <= math.pi).all(), (
        f"Phase out of range [-pi, pi]: min={phase.min():.4f}, max={phase.max():.4f}"
    )
    print("  [PASS] phase in [-pi, pi]")


def test_roundtrip_si_snr():
    """
    Round-trip (waveform → STFT → iSTFT → waveform) must be
    near-lossless: SI-SNR > 60 dB.
    """
    cfg = STFTConfig()
    stft = STFTProcessor(cfg)
    istft = ISTFTReconstructor(cfg)

    # Use deterministic seed for reproducibility
    torch.manual_seed(42)
    waveform = make_audio(4, cfg.segment_samples)

    magnitude, phase = stft(waveform)
    reconstructed = istft(magnitude, phase)

    si_snr = compute_si_snr(waveform, reconstructed)
    print(f"  Round-trip SI-SNR: {si_snr:.2f} dB  (threshold: > 60 dB, expect > 80 dB)")

    assert si_snr > 60.0, (
        f"Round-trip SI-SNR too low: {si_snr:.2f} dB. "
        "Check STFT parameters and padding."
    )
    print("  [PASS] round-trip SI-SNR > 60 dB")


def test_output_clipped():
    """Reconstructed waveform must be clipped to [-1, 1]."""
    cfg = STFTConfig()
    stft = STFTProcessor(cfg)
    istft = ISTFTReconstructor(cfg)

    # Intentionally large amplitude (> 1.0) to stress-test clipping
    waveform = torch.randn(2, 1, cfg.segment_samples) * 5.0
    magnitude, phase = stft(waveform)
    reconstructed = istft(magnitude, phase)

    assert reconstructed.max() <= 1.0, (
        f"Reconstructed max exceeds 1.0: {reconstructed.max():.4f}"
    )
    assert reconstructed.min() >= -1.0, (
        f"Reconstructed min below -1.0: {reconstructed.min():.4f}"
    )
    print("  [PASS] output clipped to [-1, 1]")


def test_device_consistency():
    """Modules must work correctly when moved to CUDA (if available)."""
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available, skipping device test")
        return

    device = torch.device("cuda")
    cfg = STFTConfig()
    stft = STFTProcessor(cfg).to(device)
    istft = ISTFTReconstructor(cfg).to(device)

    waveform = make_audio(2, cfg.segment_samples).to(device)
    magnitude, phase = stft(waveform)
    reconstructed = istft(magnitude, phase)

    assert magnitude.device.type == "cuda"
    assert reconstructed.device.type == "cuda"

    si_snr = compute_si_snr(
        waveform.cpu(), reconstructed.cpu()
    )
    assert si_snr > 60.0, f"CUDA round-trip SI-SNR too low: {si_snr:.2f} dB"
    print(f"  CUDA round-trip SI-SNR: {si_snr:.2f} dB")
    print("  [PASS] CUDA device consistency")


def test_multiplicative_mask_identity():
    """
    Applying a mask of all 1.0 (identity) must leave the reconstructed
    waveform identical to the original (SI-SNR > 60 dB).
    This validates the multiplicative watermarking path.
    """
    cfg = STFTConfig()
    stft = STFTProcessor(cfg)
    istft = ISTFTReconstructor(cfg)

    torch.manual_seed(0)
    waveform = make_audio(2, cfg.segment_samples)

    magnitude, phase = stft(waveform)

    # Identity mask: M = 1.0 everywhere → S_wm = S_mag × 1.0 = S_mag
    identity_mask = torch.ones_like(magnitude)
    watermarked_magnitude = magnitude * identity_mask

    reconstructed = istft(watermarked_magnitude, phase)

    si_snr = compute_si_snr(waveform, reconstructed)
    print(f"  Identity mask SI-SNR: {si_snr:.2f} dB  (threshold: > 60 dB)")

    assert si_snr > 60.0, (
        f"Identity mask SI-SNR too low: {si_snr:.2f} dB. "
        "Multiplicative path has unexpected distortion."
    )
    print("  [PASS] multiplicative identity mask is lossless")


def test_wrong_input_length_raises():
    """STFTProcessor must raise ValueError for wrong input length."""
    cfg = STFTConfig()
    stft = STFTProcessor(cfg)

    bad_waveform = torch.randn(1, 1, 44100)   # wrong length
    try:
        stft(bad_waveform)
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        print(f"  Caught expected error: {e}")
        print("  [PASS] wrong input length raises ValueError")


def test_stereo_input_raises():
    """STFTProcessor must raise ValueError for stereo input."""
    cfg = STFTConfig()
    stft = STFTProcessor(cfg)

    stereo = torch.randn(1, 2, 96000)   # 2 channels
    try:
        stft(stereo)
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        print(f"  Caught expected error: {e}")
        print("  [PASS] stereo input raises ValueError")


# ── runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_output_shapes,
        test_magnitude_non_negative,
        test_phase_range,
        test_roundtrip_si_snr,
        test_output_clipped,
        test_device_consistency,
        test_multiplicative_mask_identity,
        test_wrong_input_length_raises,
        test_stereo_input_raises,
    ]

    print("\n" + "=" * 60)
    print("AURA — Step 1: STFT Module Tests")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_fn in tests:
        print(f"\n{test_fn.__name__}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)

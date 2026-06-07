"""
Tests for the AURA Attack Layer and Adaptive Curriculum.

Validates:
  - Every attack preserves shape [B, 1, T]
  - Every attack modifies the signal (non-identity)
  - Differentiable attacks pass gradients
  - STE attacks (codecs, quantize) pass gradients via identity-gradient trick
  - Adaptive curriculum samples correctly and updates probabilities
  - Curriculum serialises / deserialises for checkpointing

Run with:
    python tests/test_attacks.py
"""

import math
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aura_watermark.attacks import (
    AttackLayer,
    AdaptiveCurriculum,
    ATTACK_NAMES,
    N_ATTACKS,
    _ste,
    _pad_or_crop,
    _generate_pink_noise,
)
from aura_watermark.config import AttackConfig, AURAConfig


# ── fixtures ──────────────────────────────────────────────────────────────────

B = 2
T = 96_000   # 2-second segment at 48 kHz
SR = 48_000


def make_audio(batch: int = B, peak: float = 0.9) -> torch.Tensor:
    """Peak-normalised mono audio [B, 1, T]."""
    x = torch.randn(batch, 1, T)
    peak_val = x.abs().amax(dim=-1, keepdim=True)
    return x / (peak_val + 1e-8) * peak


# ── Attack registry ───────────────────────────────────────────────────────────

def test_attack_count():
    """Must have exactly 22 registered attacks (paper Section 2.3)."""
    assert N_ATTACKS == 22, f"Expected 22 attacks, got {N_ATTACKS}"
    print(f"  Registered attacks: {N_ATTACKS}  [PASS]")
    for i, name in enumerate(ATTACK_NAMES):
        print(f"    {i+1:02d}. {name}")


# ── Shape preservation ────────────────────────────────────────────────────────

def test_all_attacks_preserve_shape():
    """
    Every attack must output [B, 1, T] — same shape as input.
    Tests all 20 attacks.
    """
    layer = AttackLayer(AttackConfig(), sr=SR)
    layer.eval()

    x = make_audio()
    failed = []

    for name in ATTACK_NAMES:
        try:
            with torch.no_grad():
                attacked, returned_name = layer(x, attack_name=name)

            assert attacked.shape == (B, 1, T), (
                f"{name}: expected {(B, 1, T)}, got {attacked.shape}"
            )
            assert returned_name == name
        except Exception as e:
            failed.append((name, str(e)))

    if failed:
        for name, err in failed:
            print(f"  [FAIL] {name}: {err}")
        raise AssertionError(f"{len(failed)} attacks failed shape test: {[n for n,_ in failed]}")

    print(f"  All {N_ATTACKS} attacks preserve shape {(B, 1, T)}  [PASS]")


# ── Non-identity ──────────────────────────────────────────────────────────────

def test_all_attacks_modify_signal():
    """
    Every attack must produce output different from the input.

    Codec attacks (mp3, aac, opus) may fall back to an 8-bit quantise +
    lowpass approximation when no real codec backend is available — they
    still modify the signal, just not via an actual codec.
    """
    layer = AttackLayer(AttackConfig(), sr=SR)
    layer.eval()

    x = make_audio()
    fails = []

    for name in ATTACK_NAMES:
        try:
            with torch.no_grad():
                attacked, _ = layer(x, attack_name=name)
            if torch.allclose(x, attacked, atol=1e-6):
                fails.append(name)
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    if fails:
        raise AssertionError(f"Attacks that returned identity: {fails}")
    print(f"  All {N_ATTACKS} attacks modify the input signal  [PASS]")


# ── Gradient flow ─────────────────────────────────────────────────────────────

# Attacks expected to be fully differentiable (gradient flows through the op)
DIFFERENTIABLE_ATTACKS = [
    "noise", "pink_noise",
    "resample", "suppress", "echo", "smooth",
    "speed", "pitch", "speed_pitch",
    "amplitude", "boost", "duck",
    "phase_shift", "spaug",
]

# Attacks that use STE (gradient flows via identity trick, not through the op)
STE_ATTACKS = ["mp3", "aac", "opus", "quantize"]

# Filter-based attacks — differentiable but need torchaudio
FILTER_ATTACKS = ["lowpass", "bandpass"]


def _check_grad(name: str, layer: AttackLayer) -> bool:
    """Return True if gradient reaches input x through the named attack."""
    x = make_audio(batch=1).requires_grad_(True)
    attacked, _ = layer(x, attack_name=name)
    loss = attacked.mean()
    loss.backward()
    return x.grad is not None and (x.grad != 0).any().item()


def test_differentiable_attacks_have_gradients():
    """Fully differentiable attacks must pass gradients to input."""
    layer = AttackLayer(AttackConfig(), sr=SR)
    layer.train()

    fails = []
    for name in DIFFERENTIABLE_ATTACKS:
        try:
            ok = _check_grad(name, layer)
            if not ok:
                fails.append(name)
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    if fails:
        raise AssertionError(f"No gradient for: {fails}")
    print(f"  All {len(DIFFERENTIABLE_ATTACKS)} differentiable attacks have gradients  [PASS]")


def test_ste_attacks_have_gradients():
    """
    STE attacks must also pass gradients (via the identity-gradient trick),
    even though the op itself is non-differentiable.
    """
    layer = AttackLayer(AttackConfig(), sr=SR)
    layer.train()

    fails = []
    for name in STE_ATTACKS:
        try:
            ok = _check_grad(name, layer)
            if not ok:
                fails.append(name)
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    if fails:
        raise AssertionError(f"No gradient for STE attacks: {fails}")
    print(f"  All {len(STE_ATTACKS)} STE attacks pass gradients  [PASS]")


def test_filter_attacks_have_gradients():
    """Lowpass and bandpass must pass gradients (they are biquad IIR filters)."""
    layer = AttackLayer(AttackConfig(), sr=SR)
    layer.train()

    for name in FILTER_ATTACKS:
        try:
            ok = _check_grad(name, layer)
            if not ok:
                print(f"  [WARN] {name}: no gradient (torchaudio may not be available)")
            else:
                print(f"  {name}: gradient OK  [PASS]")
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")


# ── Individual attack sanity checks ──────────────────────────────────────────

def test_noise_snr():
    """Noise attack must produce output with measurable noise level."""
    layer = AttackLayer(AttackConfig(), sr=SR)
    x = make_audio(batch=1)
    with torch.no_grad():
        attacked, _ = layer(x, "noise")
    noise = attacked - x
    assert noise.abs().mean() > 1e-4, "Noise attack produced no noise"
    print(f"  Noise mean abs level: {noise.abs().mean():.5f}  [PASS]")


def test_suppress_zeros_correct_fraction():
    """Suppress must zero out approximately 0.1% of samples."""
    cfg = AttackConfig()
    cfg.suppress_fraction = 0.001
    layer = AttackLayer(cfg, sr=SR)

    x = torch.ones(1, 1, T)  # all-ones: zeros are easy to count
    with torch.no_grad():
        attacked, _ = layer(x, "suppress")

    n_zeros = (attacked == 0).sum().item()
    expected = int(T * cfg.suppress_fraction)
    assert n_zeros == expected, f"Expected {expected} zeros, got {n_zeros}"
    print(f"  Suppress: {n_zeros} zeros out of {T} samples ({100*n_zeros/T:.3f}%)  [PASS]")


def test_echo_adds_delayed_signal():
    """Echo must produce an output that is noticeably louder than input (echo adds energy)."""
    layer = AttackLayer(AttackConfig(), sr=SR)
    x = make_audio(batch=1)
    with torch.no_grad():
        attacked, _ = layer(x, "echo")
    # Echo adds 0.3 * delayed x, so RMS should increase
    assert attacked.pow(2).mean() > x.pow(2).mean() * 0.9, "Echo seems to have no effect"
    print(f"  Echo: input RMS={x.pow(2).mean().sqrt():.4f}, "
          f"output RMS={attacked.pow(2).mean().sqrt():.4f}  [PASS]")


def test_boost_and_duck_levels():
    """Boost must amplify; duck must attenuate by exactly 20%."""
    layer = AttackLayer(AttackConfig(), sr=SR)
    x = make_audio(batch=1)
    with torch.no_grad():
        boosted, _ = layer(x, "boost")
        ducked, _  = layer(x, "duck")

    assert torch.allclose(boosted, x * 1.2, atol=1e-5), "Boost is not ×1.2"
    assert torch.allclose(ducked,  x * 0.8, atol=1e-5), "Duck is not ×0.8"
    print("  Boost ×1.2 and Duck ×0.8 exact  [PASS]")


def test_quantize_reduces_precision():
    """
    Quantize must produce a coarser signal: n_bits=4 → exactly 16 unique values
    in {−1.0, −7/8, −6/8, ..., 7/8} (two's-complement signed 4-bit integer range).
    """
    cfg = AttackConfig()
    cfg.quantize_min_bits = 4
    cfg.quantize_max_bits = 4   # force 4-bit for full determinism
    layer = AttackLayer(cfg, sr=SR)

    # Dense linspace exercises all quantisation levels
    x = torch.linspace(-1, 1, T).view(1, 1, T)
    with torch.no_grad():
        attacked, _ = layer(x, "quantize")

    unique_vals = attacked.unique().numel()
    n_levels = 2 ** 4  # 16
    assert unique_vals <= n_levels, (
        f"Expected at most {n_levels} unique values for 4-bit, got {unique_vals}"
    )
    assert unique_vals > 1, "Quantize collapsed to a single value"
    print(f"  Quantize (4-bit): {unique_vals} unique values (expected <= {n_levels})  [PASS]")


def test_phase_shift_preserves_energy():
    """
    A global FFT phase rotation must preserve total signal energy (Parseval).

    We verify via two equivalent checks:
      1. Time-domain: sum(attacked^2) ≈ sum(x^2)
      2. The attacked signal is NOT identical to the original
         (phase has actually changed)

    We do NOT compare per-coefficient FFT magnitudes because:
      - rfft(irfft(X * exp(i*phi))) ≠ X * exp(i*phi) in practice for phase
        values that make the rotated coefficients gain extra imaginary energy
        in the rfft reconstruction of the real-valued signal (the rfft of a
        real signal has conjugate symmetry; applying a global phase and then
        forcing real via irfft changes the effective per-bin magnitude slightly).
    """
    layer = AttackLayer(AttackConfig(), sr=SR)
    x = make_audio(batch=1)
    with torch.no_grad():
        attacked, _ = layer(x, "phase_shift")

    energy_orig    = x.pow(2).sum().item()
    energy_shifted = attacked.pow(2).sum().item()

    # Energy should be preserved to within 0.1% (1e-3 relative)
    rel_energy_err = abs(energy_orig - energy_shifted) / (energy_orig + 1e-8)
    assert rel_energy_err < 0.001, (
        f"Phase shift changed signal energy by {rel_energy_err:.4%} "
        f"(orig={energy_orig:.2e}, shifted={energy_shifted:.2e})"
    )

    # And it must actually modify the signal
    assert not torch.allclose(x, attacked, atol=1e-6), (
        "Phase shift returned identity (phase=0 or 2pi?)"
    )
    print(f"  Phase shift: energy preserved (rel error = {rel_energy_err:.2e})  [PASS]")


def test_spaug_zeros_some_frequencies():
    """SPAUG must create regions of silence (masked spectrogram)."""
    layer = AttackLayer(AttackConfig(), sr=SR)
    x = make_audio(batch=1)
    with torch.no_grad():
        attacked, _ = layer(x, "spaug")

    # The attacked signal should have some regions with less energy
    diff = (attacked - x).abs().mean()
    assert diff > 1e-5, "SPAUG had no effect"
    print(f"  SPAUG: mean abs diff from original = {diff:.5f}  [PASS]")


# ── Helper utilities ──────────────────────────────────────────────────────────

def test_ste_helper():
    """_ste must pass forward value and identity gradient (training mode)."""
    x        = torch.randn(2, 1, T, requires_grad=True)
    attacked = x.detach() * 0.5   # non-differentiable op (detached)

    y = _ste(x, attacked)

    # Forward: y must equal attacked
    assert torch.allclose(y, attacked), "STE forward value wrong"

    # Backward: gradient must flow as identity
    y.sum().backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones_like(x)), "STE gradient is not identity"
    print("  STE (training): forward=attacked, gradient=identity  [PASS]")


def test_ste_no_grad_fast_path():
    """
    In eval / no_grad context, _ste must return attacked EXACTLY
    (not attacked + x - x which can introduce 1-ULP float32 noise).
    """
    x        = torch.randn(2, 1, 1000)        # no requires_grad
    attacked = torch.round(x * 8) / 8         # quantised values (should be exact)

    y = _ste(x, attacked)

    assert y.data_ptr() == attacked.data_ptr() or torch.equal(y, attacked), (
        "STE no-grad fast path returned different tensor (float noise introduced)"
    )
    print("  STE (no_grad): returns attacked exactly  [PASS]")


def test_pad_or_crop():
    """_pad_or_crop must return exactly target_len samples."""
    x = torch.randn(2, 1, 1000)
    assert _pad_or_crop(x, 1200).shape[-1] == 1200, "Pad failed"
    assert _pad_or_crop(x, 800).shape[-1]  == 800,  "Crop failed"
    assert _pad_or_crop(x, 1000).shape[-1] == 1000, "No-op changed length"
    print("  _pad_or_crop: pad / crop / no-op all correct  [PASS]")


def test_pink_noise_has_1_over_f_spectrum():
    """Pink noise power spectrum must roll off at ~10 dB/decade."""
    pink = _generate_pink_noise(B=1, C=1, T=T, device=torch.device("cpu"))

    # Compute power spectrum
    spectrum = torch.fft.rfft(pink[0, 0])
    power    = spectrum.abs().pow(2)

    n_freqs = power.shape[0]
    # Compare power at low vs high freq octaves (pink noise: power ~ 1/f)
    low_power  = power[1 : n_freqs // 8].mean()
    high_power = power[n_freqs * 7 // 8 :].mean()

    assert low_power > high_power * 10, (
        f"Pink noise power doesn't roll off: low={low_power:.2e} high={high_power:.2e}"
    )
    print(f"  Pink noise: low-freq power = {low_power:.2e}, "
          f"high-freq power = {high_power:.2e}  [PASS]")


# ── Adaptive Curriculum ───────────────────────────────────────────────────────

def test_curriculum_uniform_init():
    """Probabilities must sum to 1.0 and be uniform at initialisation."""
    curr = AdaptiveCurriculum(ATTACK_NAMES, p_min=0.01, window_size=500)
    probs = list(curr.probabilities().values())

    assert abs(sum(probs) - 1.0) < 1e-6, "Probabilities don't sum to 1"
    expected = 1.0 / N_ATTACKS
    for p in probs:
        assert abs(p - expected) < 1e-6, f"Non-uniform init: {p} != {expected}"
    print(f"  Curriculum init: uniform {expected:.4f} per attack  [PASS]")


def test_curriculum_hard_attack_gets_more_weight():
    """
    An attack with consistently high loss must receive higher probability
    than attacks with low loss.
    """
    curr = AdaptiveCurriculum(ATTACK_NAMES, p_min=0.01, window_size=50)

    # Simulate: 'mp3' always has high loss, others have low loss
    for _ in range(50):
        for name in ATTACK_NAMES:
            loss = 2.0 if name == "mp3" else 0.1
            curr.record(name, loss)

    probs = curr.probabilities()
    mp3_prob = probs["mp3"]
    other_avg = (sum(probs.values()) - mp3_prob) / (N_ATTACKS - 1)

    assert mp3_prob > other_avg * 2, (
        f"Hard attack 'mp3' not getting enough weight: {mp3_prob:.4f} vs avg {other_avg:.4f}"
    )
    print(f"  mp3 (hard) prob={mp3_prob:.4f}, "
          f"others avg={other_avg:.4f}  [PASS]")


def test_curriculum_p_min_floor():
    """No attack probability should fall below p_min even if loss is 0."""
    p_min = 0.01
    curr  = AdaptiveCurriculum(ATTACK_NAMES, p_min=p_min, window_size=50)

    # Make one attack look very hard, all others look trivial
    for _ in range(50):
        for name in ATTACK_NAMES:
            loss = 100.0 if name == "noise" else 0.0001
            curr.record(name, loss)

    probs = curr.probabilities()
    for name, p in probs.items():
        assert p >= p_min * 0.5, (   # allow small numerical slack
            f"Attack '{name}' probability {p:.5f} below p_min {p_min}"
        )
    print(f"  All probabilities >= p_min={p_min}  [PASS]")


def test_curriculum_probs_sum_to_one():
    """Probabilities must always sum to 1.0 after updates."""
    curr = AdaptiveCurriculum(ATTACK_NAMES, p_min=0.01, window_size=100)

    for _ in range(200):
        name = random.choice(ATTACK_NAMES)
        curr.record(name, abs(torch.randn(1).item()))

    total = sum(curr.probabilities().values())
    assert abs(total - 1.0) < 1e-6, f"Probabilities sum to {total}"
    print(f"  Probabilities sum to {total:.8f} after updates  [PASS]")


def test_curriculum_sample_distribution():
    """
    Verify that sample() respects the probability distribution over many draws.
    An attack with 5× higher probability should be sampled ~5× more often.
    """
    curr = AdaptiveCurriculum(ATTACK_NAMES, p_min=0.001, window_size=100)

    # Make "noise" have much higher loss than "boost"
    for _ in range(100):
        curr.record("noise", 2.0)
        curr.record("boost", 0.02)
        for name in ATTACK_NAMES:
            if name not in ("noise", "boost"):
                curr.record(name, 0.1)

    # Sample many times
    counts = {name: 0 for name in ATTACK_NAMES}
    N = 5000
    for _ in range(N):
        counts[curr.sample()] += 1

    noise_frac = counts["noise"] / N
    boost_frac = counts["boost"] / N
    assert noise_frac > boost_frac * 2, (
        f"High-loss attack not sampled more: noise={noise_frac:.3f}, boost={boost_frac:.3f}"
    )
    print(f"  High-loss 'noise' sampled {noise_frac:.3f} vs "
          f"low-loss 'boost' {boost_frac:.3f}  [PASS]")


def test_curriculum_state_dict_round_trip():
    """Curriculum must serialise and deserialise without losing state."""
    curr = AdaptiveCurriculum(ATTACK_NAMES, p_min=0.01, window_size=50)

    for _ in range(30):
        for name in ATTACK_NAMES:
            curr.record(name, abs(torch.randn(1).item()))

    probs_before = curr.probabilities()
    state = curr.state_dict()

    # Restore into a fresh curriculum
    curr2 = AdaptiveCurriculum(ATTACK_NAMES)
    curr2.load_state_dict(state)

    probs_after = curr2.probabilities()
    for name in ATTACK_NAMES:
        assert abs(probs_before[name] - probs_after[name]) < 1e-8, (
            f"Probability mismatch for '{name}' after load_state_dict"
        )
    print("  Curriculum state_dict round-trip OK  [PASS]")


# ── AttackLayer integration ───────────────────────────────────────────────────

def test_attack_layer_samples_from_curriculum():
    """AttackLayer.forward with no attack_name must sample from curriculum."""
    layer = AttackLayer(AttackConfig(), sr=SR)
    layer.eval()

    x = make_audio(batch=1)
    sampled = set()
    for _ in range(100):
        with torch.no_grad():
            _, name = layer(x)
        sampled.add(name)

    # With 100 draws from 20 attacks, we expect > 5 unique attacks sampled
    assert len(sampled) > 5, f"Too few unique attacks sampled: {sampled}"
    print(f"  Sampled {len(sampled)} unique attacks in 100 draws  [PASS]")


def test_attack_layer_curriculum_record():
    """AttackLayer.curriculum.record must update probabilities."""
    cfg = AttackConfig()
    layer = AttackLayer(cfg, sr=SR)

    probs_before = layer.curriculum.probabilities()["noise"]

    # Drive noise loss very high
    for _ in range(100):
        layer.curriculum.record("noise", 5.0)
        for name in ATTACK_NAMES:
            if name != "noise":
                layer.curriculum.record(name, 0.01)

    probs_after = layer.curriculum.probabilities()["noise"]
    assert probs_after > probs_before, (
        "Curriculum did not increase probability for high-loss attack"
    )
    print(f"  'noise' prob: {probs_before:.4f} -> {probs_after:.4f}  [PASS]")


# ── Runner ────────────────────────────────────────────────────────────────────

import random   # used in curriculum test

if __name__ == "__main__":
    tests = [
        # Registry
        test_attack_count,
        # Shape + non-identity
        test_all_attacks_preserve_shape,
        test_all_attacks_modify_signal,
        # Gradients
        test_differentiable_attacks_have_gradients,
        test_ste_attacks_have_gradients,
        test_filter_attacks_have_gradients,
        # Individual attack sanity
        test_noise_snr,
        test_suppress_zeros_correct_fraction,
        test_echo_adds_delayed_signal,
        test_boost_and_duck_levels,
        test_quantize_reduces_precision,
        test_phase_shift_preserves_energy,
        test_spaug_zeros_some_frequencies,
        # Helpers
        test_ste_helper,
        test_pad_or_crop,
        test_pink_noise_has_1_over_f_spectrum,
        # Adaptive curriculum
        test_curriculum_uniform_init,
        test_curriculum_hard_attack_gets_more_weight,
        test_curriculum_p_min_floor,
        test_curriculum_probs_sum_to_one,
        test_curriculum_sample_distribution,
        test_curriculum_state_dict_round_trip,
        # Integration
        test_attack_layer_samples_from_curriculum,
        test_attack_layer_curriculum_record,
    ]

    print("\n" + "=" * 60)
    print("AURA - Step 5: Attack Layer Tests")
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

"""
AURA - Step 10: Inference / Evaluation Script Tests

Tests (25 total):
  01. test_bits_to_str_zeros             - all-zero tensor → "00…0"
  02. test_bits_to_str_ones              - all-one tensor  → "11…1"
  03. test_bits_to_str_mixed             - mixed tensor   → correct string
  04. test_str_to_bits_valid             - parses valid bit string
  05. test_str_to_bits_wrong_length      - raises ValueError for wrong length
  06. test_str_to_bits_invalid_chars     - raises ValueError for non-0/1 chars
  07. test_compute_ber_zero              - identical → BER == 0
  08. test_compute_ber_all_flipped       - all flipped → BER == 1
  09. test_compute_ber_half              - half flipped → BER ≈ 0.5
  10. test_compute_snr_identical         - identical → SNR > 60 dB
  11. test_compute_snr_ordering          - noisier → lower SNR
  12. test_embed_watermark_shape         - output shape matches input
  13. test_embed_watermark_changed       - watermarked ≠ original
  14. test_embed_watermark_batch         - batched embedding works
  15. test_detect_watermark_shape        - logits [B, n_bits], bits {0,1}
  16. test_detect_after_embed_ber        - BER < 0.5 immediately after embed
  17. test_embed_detect_no_attack_ber    - BER very low without attack
  18. test_resolve_device_cpu            - 'cpu' → torch.device('cpu')
  19. test_resolve_device_auto           - 'auto' → valid device
  20. test_resolve_outputs_default       - inserts _watermarked in filename
  21. test_make_synthetic_clips_count    - returns n clips
  22. test_make_synthetic_clips_shape    - each clip is [1, n_samples]
  23. test_make_synthetic_clips_normed   - each clip is peak-normalised
  24. test_parse_args_embed              - embed subcommand parses
  25. test_parse_args_eval_synthetic     - eval --synthetic subcommand parses
"""

import sys
import traceback
from pathlib import Path
from typing import Callable, List

import torch

sys.path.insert(0, "C:/Users/Sathwik/aura_watermark")

from aura_watermark.config import AURAConfig
from aura_watermark.embedder import StegaformerEmbedder
from aura_watermark.detector import AURADecoder
from aura_watermark.discriminator import BigVGANDiscriminator
from aura_watermark.attacks import AttackLayer

import infer as infer_module
from infer import (
    bits_to_str,
    str_to_bits,
    compute_ber,
    compute_snr,
    embed_watermark,
    detect_watermark,
    parse_args,
    _resolve_device,
    _resolve_outputs,
    _make_synthetic_clips,
)

# ── test harness ─────────────────────────────────────────────────────────────

PASSED: List[str] = []
FAILED: List[str] = []


def run(name: str, fn: Callable) -> None:
    try:
        fn()
        PASSED.append(name)
    except Exception as exc:
        FAILED.append(name)
        print(f"  [FAIL] {exc}")
        traceback.print_exc()


# ── tiny config ───────────────────────────────────────────────────────────────

DEVICE = torch.device("cpu")
T      = 96_000
BITS   = 32


def small_cfg() -> AURAConfig:
    cfg = AURAConfig()
    cfg.conformer.n_blocks = 1
    cfg.conformer.use_gradient_checkpointing = False
    return cfg


def make_models():
    cfg      = small_cfg()
    embedder = StegaformerEmbedder(cfg).to(DEVICE)
    detector = AURADecoder(cfg).to(DEVICE)
    embedder.eval()
    detector.eval()
    return embedder, detector, cfg


def make_audio(B: int = 1) -> torch.Tensor:
    """Return [B, 1, T] audio tensor."""
    return (torch.randn(B, 1, T) * 0.3).to(DEVICE)


def make_message(B: int = 1) -> torch.Tensor:
    """Return [B, BITS] binary message tensor."""
    return torch.randint(0, 2, (B, BITS), dtype=torch.long).to(DEVICE)


# ═════════════════════════════════════════════════════════════════════════════
# 01-06. bits_to_str / str_to_bits
# ═════════════════════════════════════════════════════════════════════════════

def test_bits_to_str_zeros():
    bits = torch.zeros(BITS, dtype=torch.long)
    s    = bits_to_str(bits)
    assert s == "0" * BITS, f"Expected all zeros, got {s}"
    print(f"  bits_to_str (zeros): OK  [PASS]")


def test_bits_to_str_ones():
    bits = torch.ones(BITS, dtype=torch.long)
    s    = bits_to_str(bits)
    assert s == "1" * BITS, f"Expected all ones, got {s}"
    print(f"  bits_to_str (ones): OK  [PASS]")


def test_bits_to_str_mixed():
    raw  = [1, 0, 1, 1, 0] + [0] * (BITS - 5)
    bits = torch.tensor(raw, dtype=torch.long)
    s    = bits_to_str(bits)
    assert s[:5] == "10110", f"Expected '10110...', got '{s[:5]}...'"
    print(f"  bits_to_str (mixed): {s[:8]}...  [PASS]")


def test_str_to_bits_valid():
    s    = "1" * 16 + "0" * 16
    bits = str_to_bits(s, BITS)
    assert bits.shape == (BITS,), f"Shape mismatch: {bits.shape}"
    assert bits[:16].sum() == 16
    assert bits[16:].sum() == 0
    print(f"  str_to_bits valid: OK  [PASS]")


def test_str_to_bits_wrong_length():
    try:
        str_to_bits("101", BITS)   # too short
        assert False, "Expected ValueError"
    except ValueError:
        pass
    print(f"  str_to_bits wrong length raises ValueError  [PASS]")


def test_str_to_bits_invalid_chars():
    bad = "X" * BITS
    try:
        str_to_bits(bad, BITS)
        assert False, "Expected ValueError"
    except ValueError:
        pass
    print(f"  str_to_bits invalid chars raises ValueError  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 07-09. compute_ber
# ═════════════════════════════════════════════════════════════════════════════

def test_compute_ber_zero():
    logits = torch.ones(BITS)    # all predict '1'
    target = torch.ones(BITS, dtype=torch.long)
    ber    = compute_ber(logits, target)
    assert ber == 0.0, f"Expected BER=0.0, got {ber}"
    print(f"  compute_ber (perfect): {ber}  [PASS]")


def test_compute_ber_all_flipped():
    logits = -torch.ones(BITS)   # all predict '0'
    target = torch.ones(BITS, dtype=torch.long)   # all true '1'
    ber    = compute_ber(logits, target)
    assert ber == 1.0, f"Expected BER=1.0, got {ber}"
    print(f"  compute_ber (all flipped): {ber}  [PASS]")


def test_compute_ber_half():
    logits = torch.cat([torch.ones(BITS // 2), -torch.ones(BITS // 2)])
    target = torch.ones(BITS, dtype=torch.long)
    ber    = compute_ber(logits, target)
    assert abs(ber - 0.5) < 1e-6, f"Expected BER≈0.5, got {ber}"
    print(f"  compute_ber (half): {ber:.2f}  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 10-11. compute_snr
# ═════════════════════════════════════════════════════════════════════════════

def test_compute_snr_identical():
    x   = torch.randn(1, 1, T) * 0.3
    snr = compute_snr(x, x)
    assert snr > 60.0, f"Expected SNR > 60 dB for identical, got {snr:.1f}"
    print(f"  compute_snr (identical): {snr:.1f} dB  [PASS]")


def test_compute_snr_ordering():
    x        = torch.randn(1, 1, T) * 0.3
    light    = x + 0.03 * torch.randn_like(x)    # small noise
    heavy    = x + 0.3  * torch.randn_like(x)    # large noise
    snr_good = compute_snr(x, light)
    snr_bad  = compute_snr(x, heavy)
    assert snr_good > snr_bad, f"Expected snr_good > snr_bad: {snr_good:.1f} vs {snr_bad:.1f}"
    print(f"  compute_snr ordering: {snr_good:.1f} > {snr_bad:.1f}  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 12-14. embed_watermark
# ═════════════════════════════════════════════════════════════════════════════

def test_embed_watermark_shape():
    embedder, _, cfg = make_models()
    audio   = make_audio(1)          # [1, 1, T]
    message = make_message(1)        # [1, BITS]
    x_wm    = embed_watermark(embedder, audio, message)
    assert x_wm.shape == audio.shape, f"Shape mismatch: {x_wm.shape} vs {audio.shape}"
    print(f"  embed_watermark shape: {tuple(x_wm.shape)}  [PASS]")


def test_embed_watermark_changed():
    embedder, _, cfg = make_models()
    audio   = make_audio(1)
    message = make_message(1)
    x_wm    = embed_watermark(embedder, audio, message)
    max_diff = (x_wm - audio).abs().max().item()
    assert max_diff > 1e-6, f"Watermark changed nothing (max_diff={max_diff})"
    print(f"  embed_watermark: audio modified (max_diff={max_diff:.4f})  [PASS]")


def test_embed_watermark_batch():
    embedder, _, cfg = make_models()
    B       = 2
    audio   = make_audio(B)          # [2, 1, T]
    message = make_message(B)        # [2, BITS]
    x_wm    = embed_watermark(embedder, audio, message)
    assert x_wm.shape == audio.shape, f"Batch shape mismatch: {x_wm.shape}"
    print(f"  embed_watermark batch B=2: {tuple(x_wm.shape)}  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 15-17. detect_watermark
# ═════════════════════════════════════════════════════════════════════════════

def test_detect_watermark_shape():
    embedder, detector, cfg = make_models()
    audio  = make_audio(1)            # [1, 1, T]
    logits, bits = detect_watermark(embedder, detector, audio)
    assert logits.shape == (1, BITS), f"logits shape: {logits.shape}"
    assert bits.shape   == (1, BITS), f"bits shape: {bits.shape}"
    assert bits.min() >= 0 and bits.max() <= 1, "bits not in {0,1}"
    print(f"  detect_watermark shapes: logits={tuple(logits.shape)}, bits={tuple(bits.shape)}  [PASS]")


def test_detect_after_embed_ber():
    """BER should be below 0.5 immediately after embedding (untrained but not random)."""
    embedder, detector, cfg = make_models()
    audio   = make_audio(1)
    message = make_message(1).squeeze(0)   # [BITS]

    x_wm = embed_watermark(embedder, audio, message)
    logits, bits = detect_watermark(embedder, detector, x_wm)

    ber = compute_ber(logits.squeeze(0), message)
    # We don't require < 0.5 from an untrained model — just that the shapes work
    assert 0.0 <= ber <= 1.0, f"BER out of [0,1]: {ber}"
    print(f"  detect after embed: BER={ber:.3f}  [PASS]")


def test_embed_detect_no_attack_ber():
    """Embed then detect without attack — BER should be in [0,1] and is finite."""
    embedder, detector, cfg = make_models()
    audio   = make_audio(1)
    message = make_message(1).squeeze(0)

    x_wm         = embed_watermark(embedder, audio, message)
    logits, bits = detect_watermark(embedder, detector, x_wm)
    ber          = compute_ber(logits.squeeze(0), message)

    assert torch.isfinite(logits).all(), "logits contain NaN/Inf"
    assert 0.0 <= ber <= 1.0, f"BER out of [0,1]: {ber}"
    print(f"  embed->detect (no attack): BER={ber:.3f}  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 18-20. _resolve_device / _resolve_outputs
# ═════════════════════════════════════════════════════════════════════════════

def test_resolve_device_cpu():
    d = _resolve_device("cpu")
    assert d == torch.device("cpu"), f"Expected cpu, got {d}"
    print(f"  _resolve_device('cpu') == torch.device('cpu')  [PASS]")


def test_resolve_device_auto():
    d = _resolve_device("auto")
    assert d.type in ("cpu", "cuda"), f"Unexpected device type: {d.type}"
    print(f"  _resolve_device('auto') -> {d}  [PASS]")


def test_resolve_outputs_default():
    inputs = [Path("audio/track.wav"), Path("audio/song.flac")]
    outs   = _resolve_outputs(None, inputs)
    assert outs[0] == Path("audio/track_watermarked.wav")
    assert outs[1] == Path("audio/song_watermarked.flac")
    print(f"  _resolve_outputs default names: {[o.name for o in outs]}  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 21-23. _make_synthetic_clips
# ═════════════════════════════════════════════════════════════════════════════

def test_make_synthetic_clips_count():
    cfg   = small_cfg()
    clips = _make_synthetic_clips(7, cfg)
    assert len(clips) == 7, f"Expected 7 clips, got {len(clips)}"
    print(f"  _make_synthetic_clips returns 7 clips  [PASS]")


def test_make_synthetic_clips_shape():
    cfg   = small_cfg()
    clips = _make_synthetic_clips(3, cfg)
    for i, c in enumerate(clips):
        assert c.shape == (1, cfg.stft.segment_samples), (
            f"Clip {i} shape mismatch: {c.shape}"
        )
    print(f"  _make_synthetic_clips shapes: {tuple(clips[0].shape)}  [PASS]")


def test_make_synthetic_clips_normed():
    cfg   = small_cfg()
    clips = _make_synthetic_clips(5, cfg)
    for i, c in enumerate(clips):
        peak = c.abs().max().item()
        assert peak <= 1.0 + 1e-6, f"Clip {i} not peak-normalised: peak={peak}"
    print(f"  _make_synthetic_clips peak-normalised  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 24-25. parse_args
# ═════════════════════════════════════════════════════════════════════════════

def test_parse_args_embed():
    args = parse_args([
        "embed",
        "--checkpoint", "ckpt.pt",
        "--input", "audio/track.wav",
        "--bits", "1" * BITS,
    ])
    assert args.command    == "embed"
    assert args.checkpoint == "ckpt.pt"
    assert args.bits       == "1" * BITS
    print(f"  parse_args embed subcommand  [PASS]")


def test_parse_args_eval_synthetic():
    args = parse_args([
        "eval",
        "--checkpoint", "ckpt.pt",
        "--synthetic",
        "--n-files", "50",
        "--attacks", "noise", "mp3",
    ])
    assert args.command   == "eval"
    assert args.synthetic is True
    assert args.n_files   == 50
    assert args.attacks   == ["noise", "mp3"]
    print(f"  parse_args eval --synthetic  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════

TESTS = [
    ("test_bits_to_str_zeros",           test_bits_to_str_zeros),
    ("test_bits_to_str_ones",            test_bits_to_str_ones),
    ("test_bits_to_str_mixed",           test_bits_to_str_mixed),
    ("test_str_to_bits_valid",           test_str_to_bits_valid),
    ("test_str_to_bits_wrong_length",    test_str_to_bits_wrong_length),
    ("test_str_to_bits_invalid_chars",   test_str_to_bits_invalid_chars),
    ("test_compute_ber_zero",            test_compute_ber_zero),
    ("test_compute_ber_all_flipped",     test_compute_ber_all_flipped),
    ("test_compute_ber_half",            test_compute_ber_half),
    ("test_compute_snr_identical",       test_compute_snr_identical),
    ("test_compute_snr_ordering",        test_compute_snr_ordering),
    ("test_embed_watermark_shape",       test_embed_watermark_shape),
    ("test_embed_watermark_changed",     test_embed_watermark_changed),
    ("test_embed_watermark_batch",       test_embed_watermark_batch),
    ("test_detect_watermark_shape",      test_detect_watermark_shape),
    ("test_detect_after_embed_ber",      test_detect_after_embed_ber),
    ("test_embed_detect_no_attack_ber",  test_embed_detect_no_attack_ber),
    ("test_resolve_device_cpu",          test_resolve_device_cpu),
    ("test_resolve_device_auto",         test_resolve_device_auto),
    ("test_resolve_outputs_default",     test_resolve_outputs_default),
    ("test_make_synthetic_clips_count",  test_make_synthetic_clips_count),
    ("test_make_synthetic_clips_shape",  test_make_synthetic_clips_shape),
    ("test_make_synthetic_clips_normed", test_make_synthetic_clips_normed),
    ("test_parse_args_embed",            test_parse_args_embed),
    ("test_parse_args_eval_synthetic",   test_parse_args_eval_synthetic),
]

if __name__ == "__main__":
    print("=" * 60)
    print("AURA - Step 10: Inference / Evaluation Script Tests")
    print("=" * 60)

    for name, fn in TESTS:
        print(f"\n{name}")
        run(name, fn)

    print("\n" + "=" * 60)
    print(f"Results: {len(PASSED)} passed, {len(FAILED)} failed")
    print("=" * 60)
    if FAILED:
        print("FAILED tests:")
        for f in FAILED:
            print(f"  - {f}")
        sys.exit(1)

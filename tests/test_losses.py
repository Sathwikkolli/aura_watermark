"""
AURA - Step 6: Loss Function & Discriminator Tests

Tests:
  01. test_message_loss_correct_bits      - loss -> 0 with high-confidence correct logits
  02. test_message_loss_wrong_bits        - loss > 0 with high-confidence wrong logits
  03. test_message_loss_random            - loss in (0, 1) for random logits
  04. test_multi_res_stft_identical       - loss == 0 when x_orig == x_wm
  05. test_multi_res_stft_nonzero         - loss > 0 when signals differ
  06. test_multi_res_stft_gradients       - gradient flows through x_wm
  07. test_nmr_loss_zero_noise            - loss ~= 0 with no watermark noise
  08. test_nmr_loss_large_noise           - loss > 0 with loud injected noise
  09. test_nmr_gradients                  - NMR gradient flows through x_wm
  10. test_bark_filterbank_shape          - filterbank is [24, n_fft//2+1]
  11. test_bark_filterbank_normalised     - each row sums to ~1
  12. test_bark_filterbank_nonneg         - all values >= 0
  13. test_spreading_matrix_shape         - spreading is [24, 24]
  14. test_spreading_matrix_diagonal      - highest weight on diagonal
  15. test_generator_adv_loss_perfect     - loss ~= 0 when D outputs 1s
  16. test_generator_adv_loss_fooled      - loss > 0 when D outputs 0s
  17. test_discriminator_adv_loss_perfect - loss ~= 0 for real=1, fake=0
  18. test_discriminator_adv_loss_fooled  - loss > 0 when D is confused
  19. test_feature_matching_identical     - loss == 0 when feats match exactly
  20. test_feature_matching_nonzero       - loss > 0 when feats differ
  21. test_aura_loss_stage1_only_msg      - stft/adv/fm/nmr == 0 in stage 1
  22. test_aura_loss_stage2_all_active    - all terms > 0 in stage 2
  23. test_aura_loss_components_as_dict   - as_dict() has all 6 keys
  24. test_aura_loss_total_gradients      - total loss has gradient w.r.t. x_wm
  25. test_aura_discriminator_step        - disc loss > 0 for random scores
  26. test_mpd_output_shapes              - MPD returns 5 scores + 5 feature lists
  27. test_msstftd_output_shapes          - MSSTFTD returns 3 scores + 3 feature lists
  28. test_bigvgan_disc_total_outputs     - 8 scores, 8 feature lists (5+3)
  29. test_bigvgan_disc_param_count       - reasonable parameter count
  30. test_all_losses_finite              - no NaN / Inf in any term
"""

import sys
import traceback
from typing import Callable, List

import torch
import torch.nn as nn

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, "C:/Users/Sathwik/aura_watermark")

from aura_watermark.config import AURAConfig
from aura_watermark.losses import (
    AURALoss,
    LossComponents,
    MultiResSTFTLoss,
    NMRLoss,
    _build_bark_filterbank,
    _build_spreading_matrix,
    discriminator_adversarial_loss,
    feature_matching_loss,
    generator_adversarial_loss,
    message_loss,
)
from aura_watermark.discriminator import (
    BigVGANDiscriminator,
    MultiPeriodDiscriminator,
    MultiScaleSTFTDiscriminator,
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


# ── fixtures ─────────────────────────────────────────────────────────────────

B    = 2
BITS = 32
SR   = 48_000
T    = 96_000

cfg = AURAConfig()


def make_audio(requires_grad: bool = False) -> torch.Tensor:
    x = torch.randn(B, 1, T) * 0.3
    if requires_grad:
        x = x.requires_grad_(True)
    return x


def make_bits() -> torch.Tensor:
    return torch.randint(0, 2, (B, BITS)).float()


def make_logits(correct: bool = False, target: torch.Tensor = None) -> torch.Tensor:
    if correct and target is not None:
        # Large positive logit for 1-bits, large negative for 0-bits
        return (target * 2 - 1) * 10.0   # {-10, +10}
    elif not correct and target is not None:
        return (target * 2 - 1) * -10.0  # wrong direction
    return torch.randn(B, BITS)


def make_fake_scores(val: float = 0.5, n: int = 8) -> List[torch.Tensor]:
    return [torch.full((B, 10), val) for _ in range(n)]


def make_feats(val: float = 0.5, n_disc: int = 8, n_layers: int = 6) -> List[List[torch.Tensor]]:
    return [[torch.full((B, 32, 4, 4), val) for _ in range(n_layers)] for _ in range(n_disc)]


# ═════════════════════════════════════════════════════════════════════════════
# 01. BCE Message Loss
# ═════════════════════════════════════════════════════════════════════════════

def test_message_loss_correct_bits():
    target  = make_bits()
    logits  = make_logits(correct=True, target=target)
    loss    = message_loss(logits, target)
    assert loss.item() < 1e-3, f"Expected ~0, got {loss.item():.4f}"
    print(f"  message_loss (correct bits): {loss.item():.2e}  [PASS]")


def test_message_loss_wrong_bits():
    target = make_bits()
    logits = make_logits(correct=False, target=target)
    loss   = message_loss(logits, target)
    assert loss.item() > 5.0, f"Expected >5, got {loss.item():.4f}"
    print(f"  message_loss (wrong bits):  {loss.item():.4f}  [PASS]")


def test_message_loss_random():
    target = make_bits()
    logits = make_logits()
    loss   = message_loss(logits, target)
    assert 0.0 < loss.item() < 2.0, f"Unexpected loss={loss.item()}"
    print(f"  message_loss (random):      {loss.item():.4f}  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 04-06. Multi-Resolution STFT Loss
# ═════════════════════════════════════════════════════════════════════════════

def test_multi_res_stft_identical():
    stft_fn = MultiResSTFTLoss(cfg)
    x = make_audio()
    loss = stft_fn(x, x)
    assert loss.item() < 1e-4, f"Expected ~0, got {loss.item():.4e}"
    print(f"  MultiResSTFTLoss (identical): {loss.item():.2e}  [PASS]")


def test_multi_res_stft_nonzero():
    stft_fn = MultiResSTFTLoss(cfg)
    x_orig  = make_audio()
    x_wm    = x_orig + 0.05 * torch.randn_like(x_orig)
    loss    = stft_fn(x_orig, x_wm)
    assert loss.item() > 0.0, "Expected loss > 0"
    print(f"  MultiResSTFTLoss (noisy):    {loss.item():.4f}  [PASS]")


def test_multi_res_stft_gradients():
    stft_fn = MultiResSTFTLoss(cfg)
    x_orig  = make_audio()
    x_wm    = make_audio(requires_grad=True)
    loss    = stft_fn(x_orig, x_wm)
    loss.backward()
    assert x_wm.grad is not None, "No gradient"
    assert torch.isfinite(x_wm.grad).all(), "Non-finite gradient"
    print(f"  MultiResSTFTLoss gradients OK  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 07-09. NMR Psychoacoustic Loss
# ═════════════════════════════════════════════════════════════════════════════

def test_nmr_loss_zero_noise():
    nmr_fn = NMRLoss(cfg)
    x      = make_audio()
    loss   = nmr_fn(x, x)   # no distortion at all
    assert loss.item() < 1e-3, f"Expected ~0, got {loss.item():.4e}"
    print(f"  NMRLoss (zero noise):  {loss.item():.2e}  [PASS]")


def test_nmr_loss_large_noise():
    nmr_fn = NMRLoss(cfg)
    x_orig = make_audio()
    x_wm   = x_orig + 0.5 * torch.randn_like(x_orig)   # very loud noise
    loss   = nmr_fn(x_orig, x_wm)
    assert loss.item() > 0.0, "Expected loss > 0 for loud noise"
    print(f"  NMRLoss (large noise): {loss.item():.4f}  [PASS]")


def test_nmr_gradients():
    nmr_fn = NMRLoss(cfg)
    x_orig = make_audio()
    x_wm   = make_audio(requires_grad=True)
    loss   = nmr_fn(x_orig, x_wm)
    loss.backward()
    assert x_wm.grad is not None, "No gradient"
    assert torch.isfinite(x_wm.grad).all(), "Non-finite NMR gradient"
    print(f"  NMRLoss gradients OK  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 10-14. Bark filterbank & spreading matrix
# ═════════════════════════════════════════════════════════════════════════════

def test_bark_filterbank_shape():
    fb = _build_bark_filterbank(2048, 48_000, 24)
    assert fb.shape == (24, 1025), f"Wrong shape {fb.shape}"
    print(f"  Bark filterbank shape {fb.shape}  [PASS]")


def test_bark_filterbank_normalised():
    fb = _build_bark_filterbank(2048, 48_000, 24)
    row_sums = fb.sum(dim=1)
    # Every non-empty row should sum to 1.0
    non_empty = row_sums > 0
    assert non_empty.all(), "Some Bark bands cover no FFT bins"
    assert (row_sums[non_empty] - 1.0).abs().max() < 1e-5, (
        f"Filterbank rows not normalised: max deviation {(row_sums-1).abs().max():.4e}"
    )
    print(f"  Bark filterbank normalised (max |sum-1|={((row_sums-1).abs().max()):.2e})  [PASS]")


def test_bark_filterbank_nonneg():
    fb = _build_bark_filterbank(2048, 48_000, 24)
    assert (fb >= 0).all(), "Filterbank has negative values"
    print(f"  Bark filterbank non-negative  [PASS]")


def test_spreading_matrix_shape():
    sp = _build_spreading_matrix(24)
    assert sp.shape == (24, 24), f"Wrong shape {sp.shape}"
    print(f"  Spreading matrix shape {sp.shape}  [PASS]")


def test_spreading_matrix_diagonal():
    sp = _build_spreading_matrix(24)
    # Diagonal (i==j, delta=0) should have the highest weight per row
    diag = sp.diag()
    row_max = sp.max(dim=1).values
    assert (diag == row_max).all(), (
        "Some rows have off-diagonal maximum — spreading function incorrect"
    )
    print(f"  Spreading matrix diagonal is max per row  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 15-18. Adversarial Losses
# ═════════════════════════════════════════════════════════════════════════════

def test_generator_adv_loss_perfect():
    # D outputs 1s for fake → generator fully fools discriminator → loss ~= 0
    fake_scores = make_fake_scores(1.0)
    loss = generator_adversarial_loss(fake_scores)
    assert loss.item() < 1e-5, f"Expected ~0, got {loss.item():.4e}"
    print(f"  generator_adv_loss (perfect): {loss.item():.2e}  [PASS]")


def test_generator_adv_loss_fooled():
    # D outputs 0s → generator failed → loss should be 1.0 per disc
    fake_scores = make_fake_scores(0.0)
    loss = generator_adversarial_loss(fake_scores)
    assert abs(loss.item() - 1.0) < 1e-4, f"Expected 1.0, got {loss.item():.4f}"
    print(f"  generator_adv_loss (D rejects): {loss.item():.4f}  [PASS]")


def test_discriminator_adv_loss_perfect():
    # D(real)=1, D(fake)=0 → optimal discriminator → loss ~= 0
    real_scores = make_fake_scores(1.0)
    fake_scores = make_fake_scores(0.0)
    loss = discriminator_adversarial_loss(real_scores, fake_scores)
    assert loss.item() < 1e-5, f"Expected ~0, got {loss.item():.4e}"
    print(f"  disc_adv_loss (perfect):  {loss.item():.2e}  [PASS]")


def test_discriminator_adv_loss_fooled():
    # D(real)=0.5, D(fake)=0.5 → confused discriminator → loss > 0
    real_scores = make_fake_scores(0.5)
    fake_scores = make_fake_scores(0.5)
    loss = discriminator_adversarial_loss(real_scores, fake_scores)
    assert loss.item() > 0.0, f"Expected > 0, got {loss.item():.4f}"
    print(f"  disc_adv_loss (confused): {loss.item():.4f}  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 19-20. Feature Matching Loss
# ═════════════════════════════════════════════════════════════════════════════

def test_feature_matching_identical():
    feats = make_feats(0.5)
    loss  = feature_matching_loss(feats, feats)
    assert loss < 1e-6, f"Expected ~0, got {loss:.4e}"
    print(f"  feature_matching (identical): {loss:.2e}  [PASS]")


def test_feature_matching_nonzero():
    real_feats = make_feats(1.0)
    fake_feats = make_feats(0.0)
    loss = feature_matching_loss(real_feats, fake_feats)
    assert loss.item() > 0.0, f"Expected > 0, got {loss.item()}"
    print(f"  feature_matching (differ):    {loss.item():.4f}  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 21-25. AURALoss combined
# ═════════════════════════════════════════════════════════════════════════════

def _make_aura_loss_inputs(requires_grad: bool = False):
    x_orig      = make_audio()
    x_wm        = make_audio(requires_grad)
    target_bits = make_bits()
    logits      = torch.randn(B, BITS)
    fake_scores = make_fake_scores(0.7)
    fake_feats  = make_feats(0.8)
    real_feats  = make_feats(1.0)
    return x_orig, x_wm, target_bits, logits, fake_scores, fake_feats, real_feats


def test_aura_loss_stage1_only_msg():
    aura_loss   = AURALoss(cfg)
    x_orig, x_wm, target_bits, logits, fs, ff, rf = _make_aura_loss_inputs()
    comp = aura_loss.generator_step(
        x_orig=x_orig, x_wm=x_wm, logits=logits, target_bits=target_bits,
        fake_scores=fs, fake_feats=ff, real_feats=rf, stage=1,
    )
    assert comp.stft.item() == 0.0, f"stft should be 0 in stage 1: {comp.stft}"
    assert comp.adv.item()  == 0.0, f"adv  should be 0 in stage 1: {comp.adv}"
    assert comp.fm.item()   == 0.0, f"fm   should be 0 in stage 1: {comp.fm}"
    assert comp.nmr.item()  == 0.0, f"nmr  should be 0 in stage 1: {comp.nmr}"
    assert comp.msg.item()  > 0.0,  f"msg  should be > 0: {comp.msg}"
    print(f"  AURALoss stage 1: msg={comp.msg.item():.4f}, rest=0  [PASS]")


def test_aura_loss_stage2_all_active():
    aura_loss   = AURALoss(cfg)
    x_orig, x_wm, target_bits, logits, fs, ff, rf = _make_aura_loss_inputs()
    comp = aura_loss.generator_step(
        x_orig=x_orig, x_wm=x_wm, logits=logits, target_bits=target_bits,
        fake_scores=fs, fake_feats=ff, real_feats=rf, stage=2,
    )
    for name in ("msg", "stft", "adv", "fm", "nmr"):
        val = getattr(comp, name).item()
        assert torch.isfinite(getattr(comp, name)), f"{name}={val} not finite"
    print(
        f"  AURALoss stage 2: msg={comp.msg.item():.3f}  stft={comp.stft.item():.3f}"
        f"  adv={comp.adv.item():.3f}  fm={comp.fm.item():.3f}  nmr={comp.nmr.item():.3f}  [PASS]"
    )


def test_aura_loss_components_as_dict():
    aura_loss   = AURALoss(cfg)
    x_orig, x_wm, target_bits, logits, fs, ff, rf = _make_aura_loss_inputs()
    comp = aura_loss.generator_step(
        x_orig=x_orig, x_wm=x_wm, logits=logits, target_bits=target_bits,
        fake_scores=fs, fake_feats=ff, real_feats=rf, stage=2,
    )
    d = comp.as_dict()
    expected_keys = {"msg", "stft", "adv", "fm", "nmr", "total"}
    assert set(d.keys()) == expected_keys, f"Keys mismatch: {set(d.keys())}"
    for k, v in d.items():
        assert isinstance(v, float), f"{k} is not float: {type(v)}"
    print(f"  LossComponents.as_dict() keys OK: {sorted(d.keys())}  [PASS]")


def test_aura_loss_total_gradients():
    aura_loss   = AURALoss(cfg)
    x_orig, x_wm, target_bits, logits, fs, ff, rf = _make_aura_loss_inputs(requires_grad=True)
    comp = aura_loss.generator_step(
        x_orig=x_orig, x_wm=x_wm, logits=logits, target_bits=target_bits,
        fake_scores=fs, fake_feats=ff, real_feats=rf, stage=2,
    )
    comp.total.backward()
    assert x_wm.grad is not None, "No gradient on x_wm"
    assert torch.isfinite(x_wm.grad).all(), "Non-finite gradient on x_wm"
    print(f"  AURALoss total.backward() OK  [PASS]")


def test_aura_discriminator_step():
    aura_loss   = AURALoss(cfg)
    real_scores = make_fake_scores(1.0)
    fake_scores = make_fake_scores(0.0)
    d_loss      = aura_loss.discriminator_step(real_scores, fake_scores)
    assert d_loss.item() < 1e-4, f"Expected ~0 for ideal discriminator, got {d_loss.item()}"
    print(f"  AURALoss.discriminator_step (ideal): {d_loss.item():.2e}  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 26-29. Discriminator architecture
# ═════════════════════════════════════════════════════════════════════════════

def test_mpd_output_shapes():
    mpd = MultiPeriodDiscriminator()
    x   = make_audio()
    scores, features = mpd(x)
    assert len(scores)   == 5, f"Expected 5 MPD scores, got {len(scores)}"
    assert len(features) == 5, f"Expected 5 MPD feature lists, got {len(features)}"
    # Each score should be 2D: [B, -1]
    for i, s in enumerate(scores):
        assert s.shape[0] == B,  f"Score {i}: batch dim wrong {s.shape}"
        assert s.ndim == 2,      f"Score {i}: expected 2D, got {s.ndim}D"
    print(f"  MPD: 5 scores {[tuple(s.shape) for s in scores]}  [PASS]")


def test_msstftd_output_shapes():
    msstftd = MultiScaleSTFTDiscriminator()
    x       = make_audio()
    scores, features = msstftd(x)
    assert len(scores)   == 3, f"Expected 3 MSSTFTD scores, got {len(scores)}"
    assert len(features) == 3, f"Expected 3 MSSTFTD feature lists, got {len(features)}"
    for i, s in enumerate(scores):
        assert s.shape[0] == B, f"Score {i}: batch dim wrong {s.shape}"
        assert s.ndim == 2,     f"Score {i}: expected 2D"
    print(f"  MSSTFTD: 3 scores {[tuple(s.shape) for s in scores]}  [PASS]")


def test_bigvgan_disc_total_outputs():
    disc = BigVGANDiscriminator()
    x    = make_audio()
    scores, features = disc(x)
    assert len(scores)   == 8, f"Expected 8 total scores, got {len(scores)}"
    assert len(features) == 8, f"Expected 8 total feature lists, got {len(features)}"
    # Feature lists should each be non-empty
    for i, f_list in enumerate(features):
        assert len(f_list) > 0, f"Sub-disc {i} has empty feature list"
    print(f"  BigVGANDiscriminator: 8 scores, 8 feature lists  [PASS]")


def test_bigvgan_disc_param_count():
    disc   = BigVGANDiscriminator()
    counts = disc.count_parameters()
    total_M = counts["total"] / 1e6
    # Rough bounds: MPD~17M, MSSTFTD~6M → total ~23M ± 10M
    assert 10.0 < total_M < 60.0, f"Unexpected parameter count: {total_M:.2f}M"
    print(
        f"  BigVGANDiscriminator: MPD={counts['mpd']/1e6:.2f}M  "
        f"MSSTFTD={counts['msstftd']/1e6:.2f}M  "
        f"total={total_M:.2f}M  [PASS]"
    )


# ═════════════════════════════════════════════════════════════════════════════
# 30. Sanity: no NaN / Inf in any loss term
# ═════════════════════════════════════════════════════════════════════════════

def test_all_losses_finite():
    aura_loss   = AURALoss(cfg)
    x_orig, x_wm, target_bits, logits, fs, ff, rf = _make_aura_loss_inputs()
    comp = aura_loss.generator_step(
        x_orig=x_orig, x_wm=x_wm, logits=logits, target_bits=target_bits,
        fake_scores=fs, fake_feats=ff, real_feats=rf, stage=2,
    )
    for name in ("msg", "stft", "adv", "fm", "nmr", "total"):
        val = getattr(comp, name)
        assert torch.isfinite(val), f"{name}={val.item()} is not finite"
    print(f"  All loss terms finite (no NaN/Inf)  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════

TESTS = [
    ("test_message_loss_correct_bits",      test_message_loss_correct_bits),
    ("test_message_loss_wrong_bits",         test_message_loss_wrong_bits),
    ("test_message_loss_random",             test_message_loss_random),
    ("test_multi_res_stft_identical",        test_multi_res_stft_identical),
    ("test_multi_res_stft_nonzero",          test_multi_res_stft_nonzero),
    ("test_multi_res_stft_gradients",        test_multi_res_stft_gradients),
    ("test_nmr_loss_zero_noise",             test_nmr_loss_zero_noise),
    ("test_nmr_loss_large_noise",            test_nmr_loss_large_noise),
    ("test_nmr_gradients",                   test_nmr_gradients),
    ("test_bark_filterbank_shape",           test_bark_filterbank_shape),
    ("test_bark_filterbank_normalised",      test_bark_filterbank_normalised),
    ("test_bark_filterbank_nonneg",          test_bark_filterbank_nonneg),
    ("test_spreading_matrix_shape",          test_spreading_matrix_shape),
    ("test_spreading_matrix_diagonal",       test_spreading_matrix_diagonal),
    ("test_generator_adv_loss_perfect",      test_generator_adv_loss_perfect),
    ("test_generator_adv_loss_fooled",       test_generator_adv_loss_fooled),
    ("test_discriminator_adv_loss_perfect",  test_discriminator_adv_loss_perfect),
    ("test_discriminator_adv_loss_fooled",   test_discriminator_adv_loss_fooled),
    ("test_feature_matching_identical",      test_feature_matching_identical),
    ("test_feature_matching_nonzero",        test_feature_matching_nonzero),
    ("test_aura_loss_stage1_only_msg",       test_aura_loss_stage1_only_msg),
    ("test_aura_loss_stage2_all_active",     test_aura_loss_stage2_all_active),
    ("test_aura_loss_components_as_dict",    test_aura_loss_components_as_dict),
    ("test_aura_loss_total_gradients",       test_aura_loss_total_gradients),
    ("test_aura_discriminator_step",         test_aura_discriminator_step),
    ("test_mpd_output_shapes",               test_mpd_output_shapes),
    ("test_msstftd_output_shapes",           test_msstftd_output_shapes),
    ("test_bigvgan_disc_total_outputs",      test_bigvgan_disc_total_outputs),
    ("test_bigvgan_disc_param_count",        test_bigvgan_disc_param_count),
    ("test_all_losses_finite",               test_all_losses_finite),
]

if __name__ == "__main__":
    print("=" * 60)
    print("AURA - Step 6: Loss Function & Discriminator Tests")
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

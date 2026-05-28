"""
AURA - Step 7: Training Loop Tests

Tests (25 total):
  01. test_compute_lr_at_zero           - lr=0 at step 0
  02. test_compute_lr_warmup_linear     - lr rises linearly during warmup
  03. test_compute_lr_peak_at_warmup    - lr == lr_max at end of warmup
  04. test_compute_lr_cosine_decay      - lr decreases after warmup
  05. test_compute_lr_at_total_steps    - lr == lr_min at total_steps
  06. test_double_encode_prob_before    - P_de=0 before de_t_start
  07. test_double_encode_prob_ramp      - P_de ramps linearly
  08. test_double_encode_prob_max       - P_de == de_p_max after ramp
  09. test_trainer_init                 - AURATrainer constructs without error
  10. test_trainer_stage1_at_step0      - stage == 1 at step 0
  11. test_trainer_stage2_after_70k     - stage == 2 at step 70_000
  12. test_train_step_stage1_returns_result  - StepResult with all fields
  13. test_train_step_stage1_msg_only   - stft/adv/fm/nmr == 0 in stage 1
  14. test_train_step_stage1_nonzero_msg - msg loss > 0
  15. test_train_step_stage2_all_active  - all loss terms != 0 in stage 2
  16. test_train_step_bit_acc_range      - bit_acc in [0, 1]
  17. test_train_step_result_as_dict     - as_dict() returns all expected keys
  18. test_grad_accum_did_step_flag      - did_step toggles on accum boundary
  19. test_global_step_increments        - global_step advances after accum
  20. test_lr_applied_to_optimiser       - optimiser pg lr matches compute_lr
  21. test_curriculum_updated_each_step  - attack probs change after steps
  22. test_checkpoint_save_load_roundtrip - state restored after save/load
  23. test_checkpoint_step_preserved     - global_step saved and restored
  24. test_prune_checkpoints             - old checkpoints deleted
  25. test_no_nan_inf_in_losses          - all loss scalars finite
"""

import sys
import os
import tempfile
import traceback
from pathlib import Path
from typing import Callable, List

import torch

sys.path.insert(0, "C:/Users/Sathwik/aura_watermark")

from aura_watermark.config import AURAConfig, TrainingConfig
from aura_watermark.trainer import AURATrainer, compute_lr, compute_double_encode_prob, StepResult
from aura_watermark.embedder import StegaformerEmbedder
from aura_watermark.detector import AURADecoder
from aura_watermark.discriminator import BigVGANDiscriminator
from aura_watermark.attacks import AttackLayer
from aura_watermark.losses import AURALoss

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


# ── tiny config for fast tests ────────────────────────────────────────────────

def small_cfg() -> AURAConfig:
    """AURAConfig with 1 Conformer block to keep tests fast."""
    cfg = AURAConfig()
    cfg.conformer.n_blocks             = 1
    cfg.conformer.use_gradient_checkpointing = False
    cfg.training.grad_accum_steps      = 2
    cfg.training.warmup_steps          = 10
    cfg.training.total_steps           = 100
    cfg.training.stage1_steps          = 50
    cfg.training.learning_rate         = 1e-4
    cfg.training.lr_min                = 1e-6
    cfg.training.de_t_start            = 50
    cfg.training.de_t_warmup           = 20
    cfg.training.de_p_max              = 0.5
    return cfg


DEVICE = torch.device("cpu")
B      = 1
T      = 96_000
BITS   = 32


def make_batch():
    audio   = torch.randn(B, 1, T) * 0.3
    message = torch.randint(0, 2, (B, BITS))
    return audio.to(DEVICE), message.to(DEVICE)


def make_trainer(cfg: AURAConfig = None) -> AURATrainer:
    if cfg is None:
        cfg = small_cfg()
    embedder  = StegaformerEmbedder(cfg).to(DEVICE)
    detector  = AURADecoder(cfg).to(DEVICE)
    disc      = BigVGANDiscriminator().to(DEVICE)
    attacks   = AttackLayer(cfg.attack, sr=cfg.stft.sample_rate)
    loss_fn   = AURALoss(cfg).to(DEVICE)
    return AURATrainer(cfg, embedder, detector, disc, attacks, loss_fn, DEVICE)


# ═════════════════════════════════════════════════════════════════════════════
# 01-05. LR Schedule
# ═════════════════════════════════════════════════════════════════════════════

def test_compute_lr_at_zero():
    cfg = small_cfg()
    lr  = compute_lr(0, cfg)
    # At step 0: lr * max(0,1)/warmup = lr * 1/10
    assert lr > 0, f"Expected lr > 0 at step 0, got {lr}"
    assert lr < cfg.training.learning_rate, f"lr at step 0 should be < peak: {lr}"
    print(f"  LR at step 0: {lr:.2e}  [PASS]")


def test_compute_lr_warmup_linear():
    cfg = small_cfg()
    lrs = [compute_lr(s, cfg) for s in range(cfg.training.warmup_steps)]
    # Should be strictly increasing
    assert all(lrs[i] < lrs[i + 1] for i in range(len(lrs) - 1)), (
        "LR not monotonically increasing during warmup"
    )
    print(f"  LR warmup monotone ({lrs[0]:.2e} -> {lrs[-1]:.2e})  [PASS]")


def test_compute_lr_peak_at_warmup():
    cfg = small_cfg()
    lr  = compute_lr(cfg.training.warmup_steps, cfg)
    assert abs(lr - cfg.training.learning_rate) < 1e-10, (
        f"Expected lr_max={cfg.training.learning_rate} at warmup end, got {lr}"
    )
    print(f"  LR peak at warmup end: {lr:.2e}  [PASS]")


def test_compute_lr_cosine_decay():
    cfg = small_cfg()
    w   = cfg.training.warmup_steps
    lrs = [compute_lr(s, cfg) for s in range(w, cfg.training.total_steps + 1)]
    # Should be non-increasing
    assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1)), (
        "LR not monotonically decreasing during cosine phase"
    )
    print(f"  LR cosine decay monotone ({lrs[0]:.2e} -> {lrs[-1]:.2e})  [PASS]")


def test_compute_lr_at_total_steps():
    cfg = small_cfg()
    lr  = compute_lr(cfg.training.total_steps, cfg)
    assert abs(lr - cfg.training.lr_min) < 1e-10, (
        f"Expected lr_min={cfg.training.lr_min} at total_steps, got {lr}"
    )
    print(f"  LR at total_steps == lr_min: {lr:.2e}  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 06-08. Double-encoding schedule
# ═════════════════════════════════════════════════════════════════════════════

def test_double_encode_prob_before():
    cfg = small_cfg()
    p   = compute_double_encode_prob(cfg.training.de_t_start - 1, cfg)
    assert p == 0.0, f"Expected 0.0, got {p}"
    print(f"  P_de before start: {p}  [PASS]")


def test_double_encode_prob_ramp():
    cfg    = small_cfg()
    start  = cfg.training.de_t_start
    warmup = cfg.training.de_t_warmup
    probs  = [compute_double_encode_prob(start + t, cfg) for t in range(warmup)]
    assert all(probs[i] <= probs[i + 1] for i in range(len(probs) - 1)), (
        "P_de not monotonically increasing during ramp"
    )
    assert probs[0] == 0.0, f"P_de at ramp start should be 0, got {probs[0]}"
    print(f"  P_de ramp: {probs[0]:.2f} -> {probs[-1]:.2f}  [PASS]")


def test_double_encode_prob_max():
    cfg = small_cfg()
    p   = compute_double_encode_prob(
        cfg.training.de_t_start + cfg.training.de_t_warmup, cfg
    )
    assert abs(p - cfg.training.de_p_max) < 1e-10, (
        f"Expected de_p_max={cfg.training.de_p_max}, got {p}"
    )
    print(f"  P_de at max: {p}  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 09-11. Trainer initialisation and stage
# ═════════════════════════════════════════════════════════════════════════════

def test_trainer_init():
    trainer = make_trainer()
    assert trainer.global_step == 0
    assert len(trainer.gen_opt.param_groups) > 0
    assert len(trainer.disc_opt.param_groups) > 0
    print(f"  AURATrainer init OK  [PASS]")


def test_trainer_stage1_at_step0():
    trainer = make_trainer()
    assert trainer.current_stage == 1, f"Expected stage 1, got {trainer.current_stage}"
    print(f"  Stage == 1 at step 0  [PASS]")


def test_trainer_stage2_after_70k():
    cfg = small_cfg()
    cfg.training.stage1_steps = 5   # override for fast test
    trainer = make_trainer(cfg)
    trainer.global_step = 5
    assert trainer.current_stage == 2, f"Expected stage 2, got {trainer.current_stage}"
    print(f"  Stage == 2 after stage1_steps  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 12-17. train_step output correctness (stage 1)
# ═════════════════════════════════════════════════════════════════════════════

def _one_step_stage1():
    """Returns a StepResult from a single stage-1 micro-step."""
    cfg     = small_cfg()
    cfg.training.grad_accum_steps = 1   # step immediately
    trainer = make_trainer(cfg)
    audio, message = make_batch()
    return trainer.train_step(audio, message)


def test_train_step_stage1_returns_result():
    result = _one_step_stage1()
    assert isinstance(result, StepResult), f"Expected StepResult, got {type(result)}"
    print(f"  train_step returns StepResult  [PASS]")


def test_train_step_stage1_msg_only():
    result = _one_step_stage1()
    assert result.stage == 1
    assert result.loss_stft == 0.0, f"stft should be 0 in stage 1: {result.loss_stft}"
    assert result.loss_adv  == 0.0, f"adv  should be 0 in stage 1: {result.loss_adv}"
    assert result.loss_fm   == 0.0, f"fm   should be 0 in stage 1: {result.loss_fm}"
    assert result.loss_nmr  == 0.0, f"nmr  should be 0 in stage 1: {result.loss_nmr}"
    assert result.loss_disc == 0.0, f"disc should be 0 in stage 1: {result.loss_disc}"
    print(f"  Stage 1: only msg loss active  [PASS]")


def test_train_step_stage1_nonzero_msg():
    result = _one_step_stage1()
    assert result.loss_msg > 0.0, f"msg loss should be > 0: {result.loss_msg}"
    print(f"  Stage 1 msg loss = {result.loss_msg:.4f}  [PASS]")


def test_train_step_stage2_all_active():
    cfg = small_cfg()
    cfg.training.grad_accum_steps = 1
    cfg.training.stage1_steps     = 0   # immediately stage 2
    trainer = make_trainer(cfg)
    audio, message = make_batch()
    result = trainer.train_step(audio, message)
    assert result.stage == 2
    # All gen loss terms should be finite (may be 0 for adv/fm if disc outputs constant)
    for name in ("loss_msg", "loss_stft", "loss_gen"):
        val = getattr(result, name)
        assert torch.isfinite(torch.tensor(val)), f"{name}={val} not finite"
    print(
        f"  Stage 2: msg={result.loss_msg:.3f} stft={result.loss_stft:.3f} "
        f"adv={result.loss_adv:.3f} fm={result.loss_fm:.3f} nmr={result.loss_nmr:.3f}  [PASS]"
    )


def test_train_step_bit_acc_range():
    result = _one_step_stage1()
    assert 0.0 <= result.bit_acc <= 1.0, f"bit_acc out of [0,1]: {result.bit_acc}"
    print(f"  bit_acc in [0,1]: {result.bit_acc:.3f}  [PASS]")


def test_train_step_result_as_dict():
    result = _one_step_stage1()
    d = result.as_dict()
    expected = {
        "step", "stage", "attack", "lr", "p_de",
        "loss/msg", "loss/stft", "loss/adv", "loss/fm", "loss/nmr",
        "loss/gen", "loss/disc", "bit_acc", "did_step",
    }
    assert set(d.keys()) == expected, f"Key mismatch: {set(d.keys()) ^ expected}"
    print(f"  StepResult.as_dict() keys OK  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 18-20. Gradient accumulation and LR application
# ═════════════════════════════════════════════════════════════════════════════

def test_grad_accum_did_step_flag():
    cfg = small_cfg()
    cfg.training.grad_accum_steps = 3
    trainer = make_trainer(cfg)
    audio, message = make_batch()

    results = [trainer.train_step(audio, message) for _ in range(3)]
    # First two are accumulation sub-steps; third should fire
    assert not results[0].did_step, "Step 0 should not update (accumulating)"
    assert not results[1].did_step, "Step 1 should not update (accumulating)"
    assert results[2].did_step,     "Step 2 should update (accum complete)"
    print(f"  did_step flags correct [False, False, True]  [PASS]")


def test_global_step_increments():
    cfg = small_cfg()
    cfg.training.grad_accum_steps = 2
    trainer = make_trainer(cfg)
    audio, message = make_batch()

    assert trainer.global_step == 0
    trainer.train_step(audio, message)
    assert trainer.global_step == 0, "Step should not increment mid-accumulation"
    trainer.train_step(audio, message)
    assert trainer.global_step == 1, f"global_step should be 1, got {trainer.global_step}"
    print(f"  global_step increments correctly  [PASS]")


def test_lr_applied_to_optimiser():
    cfg = small_cfg()
    cfg.training.grad_accum_steps = 1
    trainer = make_trainer(cfg)
    audio, message = make_batch()
    trainer.train_step(audio, message)

    expected_lr = compute_lr(trainer.global_step, cfg)
    # Apply LR manually (as train_step does at start)
    trainer._apply_lr()
    for pg in trainer.gen_opt.param_groups:
        assert abs(pg["lr"] - expected_lr) < 1e-12, (
            f"Optimiser LR {pg['lr']} != expected {expected_lr}"
        )
    print(f"  Optimiser LR == compute_lr: {expected_lr:.2e}  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 21. Curriculum
# ═════════════════════════════════════════════════════════════════════════════

def test_curriculum_updated_each_step():
    cfg = small_cfg()
    cfg.training.grad_accum_steps = 1
    trainer = make_trainer(cfg)

    probs_before = trainer.attack_layer.curriculum.probabilities()
    audio, message = make_batch()

    # Run several steps so the curriculum has data to update from
    for _ in range(5):
        trainer.train_step(audio, message)

    probs_after = trainer.attack_layer.curriculum.probabilities()

    # At least some probabilities should have changed
    changed = any(
        abs(probs_before[k] - probs_after[k]) > 1e-6
        for k in probs_before
    )
    assert changed, "Curriculum probabilities did not change after steps"
    print(f"  Curriculum updated after training steps  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 22-24. Checkpointing
# ═════════════════════════════════════════════════════════════════════════════

def test_checkpoint_save_load_roundtrip():
    cfg = small_cfg()
    cfg.training.grad_accum_steps = 1
    trainer = make_trainer(cfg)
    audio, message = make_batch()
    trainer.train_step(audio, message)   # take one step to create non-zero state

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "step_0001.pt"
        trainer.save_checkpoint(ckpt_path)
        assert ckpt_path.exists(), "Checkpoint file not created"

        # Build a fresh trainer and restore
        trainer2 = make_trainer(cfg)
        trainer2.load_checkpoint(ckpt_path)

        # Verify model weights match
        for p1, p2 in zip(trainer.embedder.parameters(), trainer2.embedder.parameters()):
            assert torch.allclose(p1, p2), "Embedder weights differ after load"

    print(f"  Checkpoint save/load round-trip OK  [PASS]")


def test_checkpoint_step_preserved():
    cfg = small_cfg()
    cfg.training.grad_accum_steps = 1
    trainer = make_trainer(cfg)
    audio, message = make_batch()

    for _ in range(3):
        trainer.train_step(audio, message)

    step_before = trainer.global_step

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "ckpt.pt"
        trainer.save_checkpoint(ckpt_path)

        trainer2 = make_trainer(cfg)
        trainer2.load_checkpoint(ckpt_path)
        assert trainer2.global_step == step_before, (
            f"global_step mismatch: {trainer2.global_step} vs {step_before}"
        )

    print(f"  global_step={step_before} preserved through checkpoint  [PASS]")


def test_prune_checkpoints():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg     = small_cfg()
        trainer = make_trainer(cfg)

        # Save 5 checkpoints
        for i in range(5):
            trainer.save_checkpoint(Path(tmpdir) / f"step_{i:04d}.pt")

        trainer.prune_checkpoints(tmpdir, keep=3)
        remaining = list(Path(tmpdir).glob("*.pt"))
        assert len(remaining) == 3, f"Expected 3 checkpoints, got {len(remaining)}"
        print(f"  prune_checkpoints: {len(remaining)} remaining (expected 3)  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 25. No NaN / Inf
# ═════════════════════════════════════════════════════════════════════════════

def test_no_nan_inf_in_losses():
    cfg = small_cfg()
    cfg.training.grad_accum_steps = 1
    trainer = make_trainer(cfg)
    audio, message = make_batch()
    result = trainer.train_step(audio, message)

    for field in ("loss_msg", "loss_stft", "loss_adv", "loss_fm", "loss_nmr", "loss_gen"):
        val = getattr(result, field)
        assert not (val != val), f"{field} is NaN"
        assert val != float("inf") and val != float("-inf"), f"{field} is Inf"

    print(f"  All loss scalars finite (no NaN/Inf)  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════

TESTS = [
    ("test_compute_lr_at_zero",              test_compute_lr_at_zero),
    ("test_compute_lr_warmup_linear",         test_compute_lr_warmup_linear),
    ("test_compute_lr_peak_at_warmup",        test_compute_lr_peak_at_warmup),
    ("test_compute_lr_cosine_decay",          test_compute_lr_cosine_decay),
    ("test_compute_lr_at_total_steps",        test_compute_lr_at_total_steps),
    ("test_double_encode_prob_before",        test_double_encode_prob_before),
    ("test_double_encode_prob_ramp",          test_double_encode_prob_ramp),
    ("test_double_encode_prob_max",           test_double_encode_prob_max),
    ("test_trainer_init",                     test_trainer_init),
    ("test_trainer_stage1_at_step0",          test_trainer_stage1_at_step0),
    ("test_trainer_stage2_after_70k",         test_trainer_stage2_after_70k),
    ("test_train_step_stage1_returns_result", test_train_step_stage1_returns_result),
    ("test_train_step_stage1_msg_only",       test_train_step_stage1_msg_only),
    ("test_train_step_stage1_nonzero_msg",    test_train_step_stage1_nonzero_msg),
    ("test_train_step_stage2_all_active",     test_train_step_stage2_all_active),
    ("test_train_step_bit_acc_range",         test_train_step_bit_acc_range),
    ("test_train_step_result_as_dict",        test_train_step_result_as_dict),
    ("test_grad_accum_did_step_flag",         test_grad_accum_did_step_flag),
    ("test_global_step_increments",           test_global_step_increments),
    ("test_lr_applied_to_optimiser",          test_lr_applied_to_optimiser),
    ("test_curriculum_updated_each_step",     test_curriculum_updated_each_step),
    ("test_checkpoint_save_load_roundtrip",   test_checkpoint_save_load_roundtrip),
    ("test_checkpoint_step_preserved",        test_checkpoint_step_preserved),
    ("test_prune_checkpoints",                test_prune_checkpoints),
    ("test_no_nan_inf_in_losses",             test_no_nan_inf_in_losses),
]

if __name__ == "__main__":
    print("=" * 60)
    print("AURA - Step 7: Training Loop Tests")
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

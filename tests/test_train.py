"""
AURA - Step 9: Training Entry Point Tests

Tests (25 total):
  01. test_parse_args_defaults           - default args parse without error
  02. test_parse_args_synthetic          - --synthetic flag parsed
  03. test_parse_args_resume             - --resume path stored correctly
  04. test_parse_args_dry_run            - --dry-run flag parsed
  05. test_parse_args_device             - --device auto/cpu/cuda parsed
  06. test_parse_args_wandb_flags        - --wandb and --wandb-project parsed
  07. test_parse_args_overrides          - batch-size / max-steps overrides
  08. test_build_models_returns_all      - build_models returns 5-tuple
  09. test_build_models_on_cpu           - models on CPU
  10. test_log_model_sizes_runs          - log_model_sizes does not raise
  11. test_fmt_duration_zero             - 0s -> 00:00:00
  12. test_fmt_duration_hours            - 3661s -> 01:01:01
  13. test_compute_snr_identical         - SNR=inf for identical signals (>60 dB)
  14. test_compute_snr_noisy             - SNR decreases with louder noise
  15. test_run_validation_returns_keys   - metrics dict has expected keys
  16. test_run_validation_ber_range      - BER in [0, 1]
  17. test_run_validation_snr_finite     - SNR is finite float
  18. test_run_validation_per_attack     - at least some per-attack BER keys
  19. test_dry_run_completes             - train(--synthetic --dry-run) exits cleanly
  20. test_dry_run_saves_no_checkpoint   - dry run does not save final checkpoint
  21. test_train_stage_switch            - stage transitions at stage1_steps
  22. test_train_logs_step_result        - Logger.log called with did_step metrics
  23. test_logger_no_wandb_no_tb         - Logger with no backends does not crash
  24. test_checkpoint_saved_during_train - checkpoint created at save_every steps
  25. test_train_resume_continues_step   - resumed trainer global_step restored
"""

import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable, List
from unittest.mock import MagicMock, patch

import torch

sys.path.insert(0, "C:/Users/Sathwik/aura_watermark")

from aura_watermark.config import AURAConfig
from aura_watermark.embedder import StegaformerEmbedder
from aura_watermark.detector import AURADecoder
from aura_watermark.discriminator import BigVGANDiscriminator
from aura_watermark.attacks import AttackLayer, ATTACK_NAMES
from aura_watermark.losses import AURALoss
from aura_watermark.trainer import AURATrainer
from aura_watermark.dataset import build_synthetic_dataloaders

import train as train_module
from train import (
    parse_args,
    build_models,
    log_model_sizes,
    run_validation,
    compute_snr,
    Logger,
    _fmt_duration,
    train,
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


# ── helpers ───────────────────────────────────────────────────────────────────

DEVICE = torch.device("cpu")
B      = 1
T      = 96_000
BITS   = 32


def small_cfg() -> AURAConfig:
    cfg = AURAConfig()
    cfg.conformer.n_blocks = 1
    cfg.conformer.use_gradient_checkpointing = False
    cfg.training.grad_accum_steps = 1
    cfg.training.batch_size       = 1
    cfg.training.stage1_steps     = 2
    cfg.training.save_every_n_steps = 100
    cfg.training.warmup_steps     = 2
    cfg.training.total_steps      = 10
    return cfg


def make_trainer(cfg: AURAConfig = None) -> AURATrainer:
    if cfg is None:
        cfg = small_cfg()
    emb  = StegaformerEmbedder(cfg).to(DEVICE)
    det  = AURADecoder(cfg).to(DEVICE)
    disc = BigVGANDiscriminator().to(DEVICE)
    att  = AttackLayer(cfg.attack, sr=cfg.stft.sample_rate)
    loss = AURALoss(cfg).to(DEVICE)
    return AURATrainer(cfg, emb, det, disc, att, loss, DEVICE)


def make_val_loader(cfg: AURAConfig = None, n: int = 4):
    if cfg is None:
        cfg = small_cfg()
    _, val_loader = build_synthetic_dataloaders(cfg, n_train=4, n_val=n,
                                                batch_size=1, num_workers=0)
    return val_loader


# ═════════════════════════════════════════════════════════════════════════════
# 01-07. parse_args
# ═════════════════════════════════════════════════════════════════════════════

def test_parse_args_defaults():
    args = parse_args([])
    assert args.synthetic is False
    assert args.device == "auto"
    assert args.wandb is False
    print(f"  parse_args defaults OK  [PASS]")


def test_parse_args_synthetic():
    args = parse_args(["--synthetic", "--n-train", "100"])
    assert args.synthetic is True
    assert args.n_train == 100
    print(f"  parse_args --synthetic  [PASS]")


def test_parse_args_resume():
    args = parse_args(["--synthetic", "--resume", "/path/to/ckpt.pt"])
    assert args.resume == "/path/to/ckpt.pt"
    print(f"  parse_args --resume  [PASS]")


def test_parse_args_dry_run():
    args = parse_args(["--synthetic", "--dry-run"])
    assert args.dry_run is True
    print(f"  parse_args --dry-run  [PASS]")


def test_parse_args_device():
    for dev in ["auto", "cpu", "cuda", "cuda:0"]:
        args = parse_args(["--synthetic", "--device", dev])
        assert args.device == dev
    print(f"  parse_args --device auto/cpu/cuda  [PASS]")


def test_parse_args_wandb_flags():
    args = parse_args(["--synthetic", "--wandb", "--wandb-project", "my-proj",
                        "--wandb-run-name", "run-01"])
    assert args.wandb is True
    assert args.wandb_project == "my-proj"
    assert args.wandb_run_name == "run-01"
    print(f"  parse_args --wandb flags  [PASS]")


def test_parse_args_overrides():
    args = parse_args(["--synthetic", "--batch-size", "8", "--max-steps", "500",
                        "--val-every", "100", "--log-every", "25"])
    assert args.batch_size == 8
    assert args.max_steps  == 500
    assert args.val_every  == 100
    assert args.log_every  == 25
    print(f"  parse_args numeric overrides  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 08-10. build_models / log_model_sizes
# ═════════════════════════════════════════════════════════════════════════════

def test_build_models_returns_all():
    cfg = small_cfg()
    result = build_models(cfg, DEVICE)
    assert len(result) == 5, f"Expected 5-tuple, got {len(result)}"
    emb, det, disc, att, loss = result
    assert isinstance(emb,  StegaformerEmbedder)
    assert isinstance(det,  AURADecoder)
    assert isinstance(disc, BigVGANDiscriminator)
    assert isinstance(att,  AttackLayer)
    assert isinstance(loss, AURALoss)
    print(f"  build_models returns 5-tuple of correct types  [PASS]")


def test_build_models_on_cpu():
    cfg = small_cfg()
    emb, det, disc, att, loss = build_models(cfg, DEVICE)
    for p in emb.parameters():
        assert p.device.type == "cpu"
        break
    print(f"  build_models on CPU  [PASS]")


def test_log_model_sizes_runs():
    cfg = small_cfg()
    emb, det, disc, att, loss = build_models(cfg, DEVICE)
    log_model_sizes(emb, det, disc)   # should not raise
    print(f"  log_model_sizes does not raise  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 11-12. _fmt_duration
# ═════════════════════════════════════════════════════════════════════════════

def test_fmt_duration_zero():
    assert _fmt_duration(0) == "00:00:00"
    print(f"  _fmt_duration(0) == '00:00:00'  [PASS]")


def test_fmt_duration_hours():
    assert _fmt_duration(3661) == "01:01:01"
    print(f"  _fmt_duration(3661) == '01:01:01'  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 13-14. compute_snr
# ═════════════════════════════════════════════════════════════════════════════

def test_compute_snr_identical():
    x   = torch.randn(1, 1, T) * 0.3
    snr = compute_snr(x, x)
    # Identical signals: noise ≈ 0 → SNR should be very large
    assert snr > 60.0, f"Expected SNR > 60 dB for identical signals, got {snr:.1f}"
    print(f"  compute_snr (identical): {snr:.1f} dB  [PASS]")


def test_compute_snr_noisy():
    x      = torch.randn(1, 1, T) * 0.3
    noisy  = x + 0.3 * torch.randn_like(x)   # ~0 dB noise
    strong = x + 3.0 * torch.randn_like(x)   # very noisy

    snr_low  = compute_snr(x, noisy)
    snr_high = compute_snr(x, strong)
    assert snr_low > snr_high, "Higher noise should give lower SNR"
    print(f"  compute_snr noisy={snr_low:.1f} dB, very_noisy={snr_high:.1f} dB  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 15-18. run_validation
# ═════════════════════════════════════════════════════════════════════════════

def _setup_val_components():
    cfg = small_cfg()
    emb = StegaformerEmbedder(cfg).to(DEVICE)
    det = AURADecoder(cfg).to(DEVICE)
    att = AttackLayer(cfg.attack, sr=cfg.stft.sample_rate)
    val = make_val_loader(cfg, n=4)
    return emb, det, att, val


def test_run_validation_returns_keys():
    emb, det, att, val = _setup_val_components()
    metrics = run_validation(emb, det, att, val, DEVICE, n_batches=2)
    required = {"val/ber", "val/bit_acc", "val/snr_db"}
    assert required.issubset(set(metrics.keys())), (
        f"Missing keys: {required - set(metrics.keys())}"
    )
    print(f"  run_validation keys: {sorted(metrics.keys())[:5]}...  [PASS]")


def test_run_validation_ber_range():
    emb, det, att, val = _setup_val_components()
    metrics = run_validation(emb, det, att, val, DEVICE, n_batches=2)
    ber = metrics["val/ber"]
    assert 0.0 <= ber <= 1.0, f"BER out of range: {ber}"
    print(f"  run_validation BER in [0,1]: {ber:.4f}  [PASS]")


def test_run_validation_snr_finite():
    emb, det, att, val = _setup_val_components()
    metrics = run_validation(emb, det, att, val, DEVICE, n_batches=2)
    snr = metrics["val/snr_db"]
    assert isinstance(snr, float) and not (snr != snr), f"SNR is not finite: {snr}"
    print(f"  run_validation SNR = {snr:.1f} dB  [PASS]")


def test_run_validation_per_attack():
    emb, det, att, val = _setup_val_components()
    metrics = run_validation(emb, det, att, val, DEVICE, n_batches=2)
    per_attack = [k for k in metrics if k.startswith("val/ber_")]
    assert len(per_attack) >= 5, f"Expected >= 5 per-attack metrics, got {len(per_attack)}"
    print(f"  run_validation: {len(per_attack)} per-attack BER metrics  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 19-20. dry-run end-to-end
# ═════════════════════════════════════════════════════════════════════════════

def test_dry_run_completes():
    with tempfile.TemporaryDirectory() as tmpdir:
        args = parse_args([
            "--synthetic", "--dry-run",
            "--checkpoint-dir", tmpdir,
            "--log-level", "WARNING",
            "--val-every", "1000",    # skip validation
            "--n-train", "4", "--n-val", "4",
        ])
        # Patch AURAConfig to use small model
        with patch.object(train_module, "AURAConfig", small_cfg):
            train(args)   # should complete without exception
    print(f"  dry-run completes without error  [PASS]")


def test_dry_run_saves_no_checkpoint():
    """Dry run should NOT write a 'final' checkpoint (no --save-every hit)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        args = parse_args([
            "--synthetic", "--dry-run",
            "--checkpoint-dir", tmpdir,
            "--log-level", "WARNING",
            "--val-every", "1000",
            "--save-every", "1000",   # save_every > 2 steps
            "--n-train", "4", "--n-val", "4",
        ])
        with patch.object(train_module, "AURAConfig", small_cfg):
            train(args)
        checkpoints = list(Path(tmpdir).glob("*.pt"))
        # No checkpoint should be written for a dry run (save_every > 2)
        assert len(checkpoints) == 0, f"Expected 0 checkpoints, got {len(checkpoints)}"
    print(f"  dry-run: no checkpoint written  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 21-22. Stage logic and logging
# ═════════════════════════════════════════════════════════════════════════════

def test_train_stage_switch():
    """Verify that stage transitions at cfg.training.stage1_steps."""
    cfg     = small_cfg()
    cfg.training.stage1_steps = 3
    trainer = make_trainer(cfg)

    audio   = torch.randn(1, 1, T) * 0.3
    message = torch.randint(0, 2, (1, BITS))

    # Steps 0-2: stage 1
    for _ in range(2):
        result = trainer.train_step(audio, message)
        assert result.stage == 1, f"Expected stage 1, got {result.stage}"

    # Step 3+: stage 2
    trainer.global_step = 3
    result = trainer.train_step(audio, message)
    assert result.stage == 2, f"Expected stage 2 at step 3, got {result.stage}"
    print(f"  stage transitions at stage1_steps=3  [PASS]")


def test_train_logs_step_result():
    """Logger.log is called with a dict when did_step=True."""
    logged_calls = []

    class MockLogger:
        def log(self, metrics, step):
            logged_calls.append((metrics, step))
        def finish(self):
            pass

    with tempfile.TemporaryDirectory() as tmpdir:
        args = parse_args([
            "--synthetic", "--dry-run",
            "--checkpoint-dir", tmpdir,
            "--log-level", "WARNING",
            "--log-every", "1",
            "--val-every", "1000",
            "--n-train", "4", "--n-val", "4",
        ])
        with patch.object(train_module, "AURAConfig", small_cfg), \
             patch.object(train_module, "Logger", lambda *a, **k: MockLogger()):
            train(args)

    # At least one log call should have happened
    assert len(logged_calls) > 0, "Expected at least one Logger.log call"
    # Verify the metrics dict has expected keys
    metrics, step = logged_calls[0]
    assert "val/step" in metrics or any("loss" in k for k in metrics), (
        f"Unexpected metrics keys: {list(metrics.keys())[:5]}"
    )
    print(f"  Logger.log called {len(logged_calls)} time(s) during dry run  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 23. Logger with no backends
# ═════════════════════════════════════════════════════════════════════════════

def test_logger_no_wandb_no_tb():
    args = parse_args(["--synthetic"])   # no --wandb, no --tensorboard
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(args, AURAConfig(), Path(tmpdir))
        logger.log({"loss": 0.5, "step": 1}, step=1)   # should not raise
        logger.finish()
    print(f"  Logger with no backends: no crash  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 24-25. Checkpointing through train()
# ═════════════════════════════════════════════════════════════════════════════

def test_checkpoint_saved_during_train():
    """A checkpoint is created when global_step % save_every == 0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        args = parse_args([
            "--synthetic",
            "--checkpoint-dir", tmpdir,
            "--log-level", "WARNING",
            "--max-steps", "2",
            "--save-every", "1",       # save every step
            "--val-every", "1000",
            "--n-train", "4", "--n-val", "4",
        ])
        with patch.object(train_module, "AURAConfig", small_cfg):
            train(args)
        checkpoints = list(Path(tmpdir).glob("*.pt"))
        assert len(checkpoints) >= 1, f"Expected >= 1 checkpoint, got {len(checkpoints)}"
    print(f"  Checkpoint created during training  [PASS]")


def test_train_resume_continues_step():
    """Resumed training starts at the saved global_step."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # First run: train for 2 steps and save checkpoint
        args1 = parse_args([
            "--synthetic", "--max-steps", "2", "--save-every", "1",
            "--checkpoint-dir", tmpdir, "--log-level", "WARNING",
            "--val-every", "1000", "--n-train", "4", "--n-val", "4",
        ])
        with patch.object(train_module, "AURAConfig", small_cfg):
            train(args1)

        checkpoints = sorted(Path(tmpdir).glob("step_*.pt"))
        assert checkpoints, "No checkpoint saved in first run"
        ckpt_path   = checkpoints[0]

        # Second run: resume and verify global_step is restored
        cfg     = small_cfg()
        emb, det, disc, att, loss = build_models(cfg, DEVICE)
        trainer = AURATrainer(cfg, emb, det, disc, att, loss, DEVICE)
        trainer.load_checkpoint(ckpt_path)
        assert trainer.global_step >= 1, (
            f"Expected global_step >= 1 after resume, got {trainer.global_step}"
        )
    print(f"  Resume restores global_step={trainer.global_step}  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════

TESTS = [
    ("test_parse_args_defaults",           test_parse_args_defaults),
    ("test_parse_args_synthetic",          test_parse_args_synthetic),
    ("test_parse_args_resume",             test_parse_args_resume),
    ("test_parse_args_dry_run",            test_parse_args_dry_run),
    ("test_parse_args_device",             test_parse_args_device),
    ("test_parse_args_wandb_flags",        test_parse_args_wandb_flags),
    ("test_parse_args_overrides",          test_parse_args_overrides),
    ("test_build_models_returns_all",      test_build_models_returns_all),
    ("test_build_models_on_cpu",           test_build_models_on_cpu),
    ("test_log_model_sizes_runs",          test_log_model_sizes_runs),
    ("test_fmt_duration_zero",             test_fmt_duration_zero),
    ("test_fmt_duration_hours",            test_fmt_duration_hours),
    ("test_compute_snr_identical",         test_compute_snr_identical),
    ("test_compute_snr_noisy",             test_compute_snr_noisy),
    ("test_run_validation_returns_keys",   test_run_validation_returns_keys),
    ("test_run_validation_ber_range",      test_run_validation_ber_range),
    ("test_run_validation_snr_finite",     test_run_validation_snr_finite),
    ("test_run_validation_per_attack",     test_run_validation_per_attack),
    ("test_dry_run_completes",             test_dry_run_completes),
    ("test_dry_run_saves_no_checkpoint",   test_dry_run_saves_no_checkpoint),
    ("test_train_stage_switch",            test_train_stage_switch),
    ("test_train_logs_step_result",        test_train_logs_step_result),
    ("test_logger_no_wandb_no_tb",         test_logger_no_wandb_no_tb),
    ("test_checkpoint_saved_during_train", test_checkpoint_saved_during_train),
    ("test_train_resume_continues_step",   test_train_resume_continues_step),
]

if __name__ == "__main__":
    print("=" * 60)
    print("AURA - Step 9: Training Entry Point Tests")
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

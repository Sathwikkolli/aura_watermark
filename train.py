#!/usr/bin/env python3
"""
AURA Training Entry Point — Step 9.

Two-stage training as described in the ICASSP 2026 paper:
  Stage 1 (steps 0 → 70 000):   only BCE message loss active.
  Stage 2 (steps 70 000 → 200 000): all five loss terms + discriminator.

Usage examples:

  # Full training with both corpora
  python train.py \\
      --emilia-root /data/emilia \\
      --fma-root    /data/fma \\
      --checkpoint-dir checkpoints/run_001 \\
      --wandb --wandb-project aura-watermark

  # Synthetic data (CI / smoke test — no corpus needed)
  python train.py --synthetic --n-train 512 --n-val 64 --max-steps 10

  # Resume from checkpoint
  python train.py \\
      --emilia-root /data/emilia \\
      --resume      checkpoints/run_001/step_070000.pt

  # Stage 1 only, then hand-off to Stage 2 later
  python train.py --emilia-root /data/emilia --max-steps 70000

  # Dry run (2 steps, verifies end-to-end pipeline)
  python train.py --synthetic --dry-run
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ── project imports ───────────────────────────────────────────────────────────
from aura_watermark.config import AURAConfig
from aura_watermark.embedder import StegaformerEmbedder
from aura_watermark.detector import AURADecoder
from aura_watermark.discriminator import BigVGANDiscriminator
from aura_watermark.attacks import AttackLayer, ATTACK_NAMES
from aura_watermark.losses import AURALoss
from aura_watermark.trainer import AURATrainer, StepResult
from aura_watermark.dataset import (
    build_dataloaders,
    build_synthetic_dataloaders,
)

# ── optional logging backends ─────────────────────────────────────────────────
try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

log = logging.getLogger("aura")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train AURA audio watermarking model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    data = p.add_argument_group("data")
    data.add_argument("--emilia-root",  type=str, default=None,
                      help="Path to Emilia dataset root (EN/ZH/DE/FR/JA/KO subfolders).")
    data.add_argument("--fma-root",     type=str, default=None,
                      help="Path to FMA-large root (fma_large/000-155 subfolders).")
    data.add_argument("--synthetic",    action="store_true",
                      help="Use in-memory synthetic data (no corpus required).")
    data.add_argument("--n-train",      type=int, default=2048,
                      help="Synthetic training clips (only with --synthetic).")
    data.add_argument("--n-val",        type=int, default=256,
                      help="Synthetic validation clips (only with --synthetic).")
    data.add_argument("--speech-ratio", type=float, default=0.75,
                      help="Fraction of training clips drawn from speech (Emilia).")
    data.add_argument("--num-workers",  type=int, default=4,
                      help="DataLoader worker processes.")
    data.add_argument("--val-frac",     type=float, default=0.01,
                      help="Fraction of each corpus reserved for validation.")

    # ── Training ──────────────────────────────────────────────────────────────
    tr = p.add_argument_group("training")
    tr.add_argument("--batch-size",      type=int,   default=None,
                    help="Per-GPU batch size (default: cfg.training.batch_size=32).")
    tr.add_argument("--max-steps",       type=int,   default=None,
                    help="Stop after this many global steps (default: cfg.training.total_steps).")
    tr.add_argument("--val-every",       type=int,   default=1_000,
                    help="Run validation every N global steps.")
    tr.add_argument("--val-batches",     type=int,   default=50,
                    help="Number of validation batches per validation run.")
    tr.add_argument("--log-every",       type=int,   default=50,
                    help="Log training metrics every N global steps.")
    tr.add_argument("--device",          type=str,   default="auto",
                    help="'auto', 'cpu', 'cuda', or 'cuda:N'.")
    tr.add_argument("--seed",            type=int,   default=42)
    tr.add_argument("--dry-run",         action="store_true",
                    help="Run for 2 steps then exit (pipeline smoke test).")

    # ── Checkpointing ─────────────────────────────────────────────────────────
    ck = p.add_argument_group("checkpointing")
    ck.add_argument("--checkpoint-dir",  type=str,  default="checkpoints",
                    help="Directory for saved checkpoints.")
    ck.add_argument("--resume",          type=str,  default=None,
                    help="Path to a .pt checkpoint to resume from.")
    ck.add_argument("--save-every",      type=int,  default=None,
                    help="Save checkpoint every N steps (default: cfg.training.save_every_n_steps).")
    ck.add_argument("--keep-checkpoints",type=int,  default=None,
                    help="Keep last N checkpoints (default: cfg.training.keep_last_n_checkpoints).")

    # ── Logging ───────────────────────────────────────────────────────────────
    lg = p.add_argument_group("logging")
    lg.add_argument("--wandb",           action="store_true",
                    help="Enable Weights & Biases logging.")
    lg.add_argument("--wandb-project",   type=str,  default="aura-watermark")
    lg.add_argument("--wandb-run-name",  type=str,  default=None)
    lg.add_argument("--tensorboard",     action="store_true",
                    help="Enable TensorBoard logging.")
    lg.add_argument("--tensorboard-dir", type=str,  default="runs")
    lg.add_argument("--log-level",       type=str,  default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p.parse_args(argv)


# ═════════════════════════════════════════════════════════════════════════════
# Logging helpers
# ═════════════════════════════════════════════════════════════════════════════

class Logger:
    """
    Thin wrapper around W&B + TensorBoard that degrades gracefully when
    neither is installed or configured.
    """

    def __init__(
        self,
        args:       argparse.Namespace,
        cfg:        AURAConfig,
        run_dir:    Path,
    ):
        self._wb  = False
        self._tb  = None

        if args.wandb:
            if not _WANDB_AVAILABLE:
                log.warning("wandb not installed — install with: pip install wandb")
            else:
                _wandb.init(
                    project = args.wandb_project,
                    name    = args.wandb_run_name,
                    config  = {
                        "learning_rate":   cfg.training.learning_rate,
                        "batch_size":      cfg.training.batch_size,
                        "grad_accum":      cfg.training.grad_accum_steps,
                        "stage1_steps":    cfg.training.stage1_steps,
                        "total_steps":     cfg.training.total_steps,
                        "n_bits":          cfg.message.n_bits,
                        "n_blocks":        cfg.conformer.n_blocks,
                        "n_attacks":       len(ATTACK_NAMES),
                    },
                    dir = str(run_dir),
                )
                self._wb = True
                log.info("W&B initialised: project=%s", args.wandb_project)

        if args.tensorboard:
            if not _TB_AVAILABLE:
                log.warning("TensorBoard not installed — install with: pip install tensorboard")
            else:
                tb_dir = Path(args.tensorboard_dir) / run_dir.name
                self._tb = _SummaryWriter(str(tb_dir))
                log.info("TensorBoard writer: %s", tb_dir)

    def log(self, metrics: Dict, step: int) -> None:
        if self._wb:
            _wandb.log(metrics, step=step)
        if self._tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb.add_scalar(k, v, global_step=step)

    def finish(self) -> None:
        if self._wb:
            _wandb.finish()
        if self._tb is not None:
            self._tb.close()


# ═════════════════════════════════════════════════════════════════════════════
# Validation
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_snr(original: torch.Tensor, watermarked: torch.Tensor) -> float:
    """Signal-to-noise ratio of the watermark distortion (dB)."""
    noise         = watermarked - original
    signal_power  = original.pow(2).mean().clamp(min=1e-10)
    noise_power   = noise.pow(2).mean().clamp(min=1e-10)
    return 10.0 * math.log10(signal_power.item() / noise_power.item())


@torch.no_grad()
def run_validation(
    embedder:     StegaformerEmbedder,
    detector:     AURADecoder,
    attack_layer: AttackLayer,
    val_loader,
    device:       torch.device,
    n_batches:    int = 50,
) -> Dict[str, float]:
    """
    Run validation and return a metrics dict.

    Metrics:
        val/ber           — mean Bit Error Rate across all attacks (0 = perfect)
        val/ber_{attack}  — BER per attack name
        val/snr_db        — mean watermark SNR (dB)
        val/bit_acc       — mean bit accuracy (complementary to BER)
    """
    embedder.eval()
    detector.eval()

    total_acc:   float = 0.0
    total_snr:   float = 0.0
    per_attack_acc: Dict[str, List[float]] = {name: [] for name in ATTACK_NAMES}
    n_seen = 0

    for batch_idx, (audio, message) in enumerate(val_loader):
        if batch_idx >= n_batches:
            break

        audio   = audio.to(device)
        message = message.to(device)

        # Embed
        x_wm, _, _ = embedder(audio, message)

        # Track SNR
        total_snr += compute_snr(audio, x_wm)

        # Evaluate each attack independently
        for attack_name in ATTACK_NAMES:
            try:
                x_attacked, _ = attack_layer(x_wm, attack_name=attack_name)
            except Exception:
                # Skip attacks that fail (e.g. codec without backend)
                continue

            # Detect
            mag, _ = embedder.stft(x_attacked)
            s_mag  = mag.unsqueeze(1)        # [B, 1, F, T]
            logits = detector(s_mag)          # [B, n_bits]

            pred    = (logits > 0).long()
            correct = (pred == message).float().mean().item()
            per_attack_acc[attack_name].append(correct)

        n_seen += 1

    # Aggregate
    metrics: Dict[str, float] = {}
    all_accs: List[float] = []

    for name, accs in per_attack_acc.items():
        if accs:
            mean_acc = sum(accs) / len(accs)
            metrics[f"val/ber_{name}"] = 1.0 - mean_acc
            all_accs.extend(accs)

    metrics["val/bit_acc"] = sum(all_accs) / len(all_accs) if all_accs else 0.0
    metrics["val/ber"]     = 1.0 - metrics["val/bit_acc"]
    metrics["val/snr_db"]  = total_snr / max(n_seen, 1)

    return metrics


# ═════════════════════════════════════════════════════════════════════════════
# Model factory
# ═════════════════════════════════════════════════════════════════════════════

def build_models(
    cfg:    AURAConfig,
    device: torch.device,
) -> Tuple[StegaformerEmbedder, AURADecoder, BigVGANDiscriminator, AttackLayer, AURALoss]:
    """Instantiate all model components and move to device."""
    embedder      = StegaformerEmbedder(cfg).to(device)
    detector      = AURADecoder(cfg).to(device)
    discriminator = BigVGANDiscriminator().to(device)
    attack_layer  = AttackLayer(cfg.attack, sr=cfg.stft.sample_rate)
    aura_loss     = AURALoss(cfg).to(device)
    return embedder, detector, discriminator, attack_layer, aura_loss


def log_model_sizes(
    embedder:      StegaformerEmbedder,
    detector:      AURADecoder,
    discriminator: BigVGANDiscriminator,
) -> None:
    def count(m):
        return sum(p.numel() for p in m.parameters()) / 1e6

    log.info("Model parameters:")
    log.info("  Embedder:       %.2f M", count(embedder))
    log.info("  Detector:       %.2f M", count(detector))
    log.info("  Discriminator:  %.2f M", count(discriminator))
    log.info("  Total:          %.2f M", count(embedder) + count(detector) + count(discriminator))


# ═════════════════════════════════════════════════════════════════════════════
# Main training loop
# ═════════════════════════════════════════════════════════════════════════════

def train(args: argparse.Namespace) -> None:
    # ── Logging setup ─────────────────────────────────────────────────────
    logging.basicConfig(
        level   = getattr(logging, args.log_level),
        format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    # ── Reproducibility ───────────────────────────────────────────────────
    torch.manual_seed(args.seed)

    # ── Device ────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log.info("Device: %s", device)

    # ── Config ────────────────────────────────────────────────────────────
    cfg = AURAConfig()
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    max_steps = args.max_steps or cfg.training.total_steps
    if args.dry_run:
        max_steps = 2

    save_every  = args.save_every        or cfg.training.save_every_n_steps
    keep_ckpts  = args.keep_checkpoints  or cfg.training.keep_last_n_checkpoints

    # ── Run directory ─────────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Logger ────────────────────────────────────────────────────────────
    logger = Logger(args, cfg, ckpt_dir)

    # ── Dataset ───────────────────────────────────────────────────────────
    log.info("Building dataloaders…")
    if args.synthetic:
        train_loader, val_loader = build_synthetic_dataloaders(
            cfg,
            n_train     = args.n_train,
            n_val       = args.n_val,
            batch_size  = cfg.training.batch_size,
            num_workers = 0,
        )
        log.info("Using synthetic dataset (%d train, %d val clips).", args.n_train, args.n_val)
    else:
        if args.emilia_root is None and args.fma_root is None:
            log.error("Provide --emilia-root and/or --fma-root, or use --synthetic.")
            sys.exit(1)
        train_loader, val_loader = build_dataloaders(
            cfg,
            emilia_root  = args.emilia_root,
            fma_root     = args.fma_root,
            speech_ratio = args.speech_ratio,
            batch_size   = cfg.training.batch_size,
            num_workers  = args.num_workers,
            val_frac     = args.val_frac,
        )
        log.info("Train loader: %d batches/epoch", len(train_loader))

    # ── Models ────────────────────────────────────────────────────────────
    log.info("Instantiating models…")
    embedder, detector, discriminator, attack_layer, aura_loss = build_models(cfg, device)
    log_model_sizes(embedder, detector, discriminator)

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = AURATrainer(
        cfg           = cfg,
        embedder      = embedder,
        detector      = detector,
        discriminator = discriminator,
        attack_layer  = attack_layer,
        aura_loss     = aura_loss,
        device        = device,
    )

    # ── Resume ────────────────────────────────────────────────────────────
    if args.resume:
        log.info("Resuming from checkpoint: %s", args.resume)
        trainer.load_checkpoint(args.resume)
        log.info("Resumed at global step %d.", trainer.global_step)

    # ── Training loop ─────────────────────────────────────────────────────
    log.info("Starting training (max_steps=%d, stage1_steps=%d).",
             max_steps, cfg.training.stage1_steps)

    # Track rolling averages for periodic console logging
    _rolling: Dict[str, float] = {}
    _roll_n = 0
    t_start = time.time()

    data_iter = iter(train_loader)

    while trainer.global_step < max_steps:

        # ── Fetch batch ─────────────────────────────────────────────────
        try:
            audio, message = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            audio, message = next(data_iter)

        audio   = audio.to(device, non_blocking=True)
        message = message.to(device, non_blocking=True)

        # ── Train step ──────────────────────────────────────────────────
        result: StepResult = trainer.train_step(audio, message)

        # ── Rolling average ─────────────────────────────────────────────
        for k, v in result.as_dict().items():
            if isinstance(v, float):
                _rolling[k] = _rolling.get(k, 0.0) + v
        _roll_n += 1

        # ── Log to W&B / TB ─────────────────────────────────────────────
        if result.did_step and trainer.global_step % args.log_every == 0:
            avg = {k: v / _roll_n for k, v in _rolling.items()}
            avg["train/step"]  = trainer.global_step
            avg["train/stage"] = result.stage
            logger.log(avg, step=trainer.global_step)
            _rolling.clear()
            _roll_n = 0

        # ── Console logging ──────────────────────────────────────────────
        if result.did_step and trainer.global_step % args.log_every == 0:
            elapsed = time.time() - t_start
            steps_done = trainer.global_step
            steps_left = max_steps - steps_done
            eta_s = elapsed / max(steps_done, 1) * steps_left
            log.info(
                "step=%6d  stage=%d  attack=%-12s  "
                "msg=%.4f  ber=%.3f  lr=%.2e  "
                "eta=%s",
                trainer.global_step,
                result.stage,
                result.attack_name,
                result.loss_msg,
                1.0 - result.bit_acc,
                result.lr,
                _fmt_duration(eta_s),
            )

        # ── Validation ──────────────────────────────────────────────────
        if result.did_step and trainer.global_step % args.val_every == 0:
            log.info("Running validation at step %d…", trainer.global_step)
            val_metrics = run_validation(
                embedder, detector, attack_layer,
                val_loader, device, n_batches=args.val_batches,
            )
            val_metrics["val/step"] = trainer.global_step

            logger.log(val_metrics, step=trainer.global_step)

            log.info(
                "Validation: BER=%.4f  SNR=%.1f dB  bit_acc=%.4f",
                val_metrics["val/ber"],
                val_metrics["val/snr_db"],
                val_metrics["val/bit_acc"],
            )

            # Per-attack BER summary (worst 5)
            attack_bers = [
                (k.removeprefix("val/ber_"), v)
                for k, v in val_metrics.items()
                if k.startswith("val/ber_")
            ]
            attack_bers.sort(key=lambda kv: -kv[1])
            log.info("  Worst attacks: %s",
                     "  ".join(f"{n}={v:.3f}" for n, v in attack_bers[:5]))

            # Put models back in train mode
            embedder.train()
            detector.train()

        # ── Checkpoint ──────────────────────────────────────────────────
        if result.did_step and trainer.global_step % save_every == 0:
            ckpt_path = ckpt_dir / f"step_{trainer.global_step:07d}.pt"
            trainer.save_checkpoint(ckpt_path)
            log.info("Saved checkpoint: %s", ckpt_path)
            trainer.prune_checkpoints(ckpt_dir, keep=keep_ckpts)

        # ── Dry-run exit ─────────────────────────────────────────────────
        if args.dry_run and trainer.global_step >= 2:
            log.info("Dry run complete — exiting after 2 steps.")
            break

    # ── Final checkpoint ──────────────────────────────────────────────────
    if not args.dry_run:
        final_path = ckpt_dir / f"step_{trainer.global_step:07d}_final.pt"
        trainer.save_checkpoint(final_path)
        log.info("Training complete. Final checkpoint: %s", final_path)

    logger.finish()


# ═════════════════════════════════════════════════════════════════════════════
# Utility
# ═════════════════════════════════════════════════════════════════════════════

def _fmt_duration(seconds: float) -> str:
    """Format seconds as hh:mm:ss."""
    seconds = int(seconds)
    h, rem  = divmod(seconds, 3600)
    m, s    = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train(parse_args())

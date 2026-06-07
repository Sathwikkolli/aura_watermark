"""
AURA Step 7 — Training Loop.

Two-stage training (paper Section 3.3):
  Stage 1  steps 0 → 70_000:   only L_msg active; discriminator frozen.
  Stage 2  steps 70_000 → 200_000: all 5 loss terms; discriminator updated.

Architecture:
  Two separate Adam optimisers (generator / discriminator), lr = 1e-4.
  LR schedule: linear warmup (5 k steps) → cosine annealing to lr_min.
  Gradient accumulation × grad_accum_steps (virtual batch = 64 clips).
  AMP FP16 via torch.autocast + GradScaler (separate scaler per optimiser).
  Gradient clipping: max_norm = 1.0 (applied before each optimiser step).

Double-encoding (paper Section 3.4):
  From step de_t_start (70k) to de_t_start + de_t_warmup (90k),
  P_de ramps linearly from 0 → de_p_max (0.5).
  With probability P_de a second random message is embedded into the
  already-watermarked audio before the attack is applied.
  The detector then tries to recover the *second* message.

Adaptive curriculum:
  AttackLayer.curriculum.record(attack_name, bit_loss) updates rolling statistics.
  Curriculum probabilities are re-normalised after every batch.

Checkpoint:
  Saved as a single .pt file every save_every_n_steps.
  Contains: model state_dicts, optimiser state_dicts, scaler state_dicts,
            global_step, and curriculum state_dict.
  Only the last keep_last_n_checkpoints are retained on disk.

Usage:
    trainer = AURATrainer(cfg, embedder, detector, discriminator,
                          attack_layer, aura_loss, device)
    for audio, message in dataloader:
        result = trainer.train_step(audio.to(device), message.to(device))
        print(result.as_dict())

    trainer.save_checkpoint("checkpoints/step_001000.pt")
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import Adam

from .attacks import AttackLayer
from .config import AURAConfig
from .detector import AURADecoder
from .discriminator import BigVGANDiscriminator
from .embedder import StegaformerEmbedder
from .losses import AURALoss, LossComponents

Tensor = torch.Tensor


# ─────────────────────────────────────────────────────────────────────────────
# LR and schedule helpers (pure functions — easy to unit-test)
# ─────────────────────────────────────────────────────────────────────────────

def compute_lr(step: int, cfg: AURAConfig) -> float:
    """
    Linear warmup then cosine annealing.

        step < warmup_steps:
            lr = lr_max * step / warmup_steps

        step >= warmup_steps:
            progress = (step - warmup) / (total - warmup)
            lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + cos(pi * progress))

    Args:
        step: current global step (0-indexed)
        cfg:  AURAConfig — uses training.{learning_rate, warmup_steps,
                           total_steps, lr_min}

    Returns:
        Learning rate as float.
    """
    tc = cfg.training
    if step < tc.warmup_steps:
        return tc.learning_rate * (step + 1) / tc.warmup_steps
    progress = min(
        (step - tc.warmup_steps) / max(tc.total_steps - tc.warmup_steps, 1),
        1.0,
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return tc.lr_min + (tc.learning_rate - tc.lr_min) * cosine


def compute_double_encode_prob(step: int, cfg: AURAConfig) -> float:
    """
    Double-encoding probability ramp (paper Section 3.4).

        step < de_t_start:            P_de = 0
        de_t_start ≤ step < de_t_start + de_t_warmup:
            P_de = de_p_max * (step - de_t_start) / de_t_warmup
        step ≥ de_t_start + de_t_warmup:
            P_de = de_p_max

    Args:
        step: current global step
        cfg:  AURAConfig — uses training.{de_t_start, de_t_warmup, de_p_max}

    Returns:
        Probability in [0, de_p_max].
    """
    tc = cfg.training
    if step < tc.de_t_start:
        return 0.0
    t = step - tc.de_t_start
    if t >= tc.de_t_warmup:
        return tc.de_p_max
    return tc.de_p_max * t / tc.de_t_warmup


# ─────────────────────────────────────────────────────────────────────────────
# Per-step result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """
    All scalars produced by one call to AURATrainer.train_step().
    Suitable for direct logging to W&B / TensorBoard.
    """
    step:         int
    stage:        int          # 1 or 2
    attack_name:  str
    lr:           float
    p_de:         float        # double-encode probability this step

    # Generator loss terms
    loss_msg:     float
    loss_stft:    float
    loss_adv:     float
    loss_fm:      float
    loss_nmr:     float
    loss_gen:     float        # weighted total generator loss

    # Discriminator loss (0.0 in stage 1)
    loss_disc:    float

    # Accuracy proxy: mean bit accuracy on recovered message
    bit_acc:      float

    # Whether the optimiser actually stepped this iteration
    # (False during gradient accumulation sub-steps)
    did_step:     bool

    def as_dict(self) -> Dict[str, float | int | str | bool]:
        return {
            "step":        self.step,
            "stage":       self.stage,
            "attack":      self.attack_name,
            "lr":          self.lr,
            "p_de":        self.p_de,
            "loss/msg":    self.loss_msg,
            "loss/stft":   self.loss_stft,
            "loss/adv":    self.loss_adv,
            "loss/fm":     self.loss_fm,
            "loss/nmr":    self.loss_nmr,
            "loss/gen":    self.loss_gen,
            "loss/disc":   self.loss_disc,
            "bit_acc":     self.bit_acc,
            "did_step":    self.did_step,
        }


# ─────────────────────────────────────────────────────────────────────────────
# AURATrainer
# ─────────────────────────────────────────────────────────────────────────────

class AURATrainer:
    """
    Full AURA training loop.

    Manages both generator (embedder + detector) and discriminator updates,
    AMP scaling, gradient accumulation, LR scheduling, double-encoding,
    and checkpointing.

    Args:
        cfg:            AURAConfig
        embedder:       StegaformerEmbedder  (on ``device``)
        detector:       AURADecoder          (on ``device``)
        discriminator:  BigVGANDiscriminator  (on ``device``)
        attack_layer:   AttackLayer
        aura_loss:      AURALoss
        device:         torch.device

    Example::

        trainer = AURATrainer(cfg, embedder, detector, disc, attacks, loss, device)
        for audio, msg in loader:
            result = trainer.train_step(audio.to(device), msg.to(device))
            wandb.log(result.as_dict(), step=result.step)
    """

    def __init__(
        self,
        cfg:           AURAConfig,
        embedder:      StegaformerEmbedder,
        detector:      AURADecoder,
        discriminator: BigVGANDiscriminator,
        attack_layer:  AttackLayer,
        aura_loss:     AURALoss,
        device:        torch.device = torch.device("cpu"),
    ):
        self.cfg           = cfg
        self.embedder      = embedder
        self.detector      = detector
        self.discriminator = discriminator
        self.attack_layer  = attack_layer
        self.aura_loss     = aura_loss
        self.device        = device

        tc = cfg.training

        # ── Optimisers ─────────────────────────────────────────────────────
        # Generator: embedder + detector parameters.
        gen_params = (
            list(embedder.parameters()) + list(detector.parameters())
        )
        self.gen_opt  = Adam(gen_params,                     lr=tc.learning_rate)
        self.disc_opt = Adam(discriminator.parameters(),     lr=tc.learning_rate)

        # ── AMP scalers (one per optimiser) ────────────────────────────────
        # AMP (float16) is opt-in via cfg.training.use_amp and only on CUDA.
        # Default fp32: the watermark is a small signal float16 corrupts, and
        # float16 over/underflow is what fed NaN/Inf into the LAME codec.
        amp_enabled   = tc.use_amp and device.type == "cuda"
        self.gen_scaler  = GradScaler(enabled=amp_enabled)
        self.disc_scaler = GradScaler(enabled=amp_enabled)

        # ── Training state ─────────────────────────────────────────────────
        self.global_step      = 0
        self._accum_count     = 0     # how many micro-steps accumulated so far
        self._accum_gen_loss: Optional[Tensor]  = None
        self._accum_disc_loss: Optional[Tensor] = None

        # Gradient accumulation denominator
        self._grad_denom = float(tc.grad_accum_steps)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def current_stage(self) -> int:
        """1 for steps < stage1_steps, 2 thereafter."""
        return 1 if self.global_step < self.cfg.training.stage1_steps else 2

    @property
    def current_lr(self) -> float:
        return compute_lr(self.global_step, self.cfg)

    @property
    def current_p_de(self) -> float:
        return compute_double_encode_prob(self.global_step, self.cfg)

    # ── Cold-start curriculum warmup ─────────────────────────────────────────

    # Easy attacks for the second warmup phase: cheap, near-invertible ops the
    # detector can learn robustness to once it can decode the clean watermark.
    _EASY_WARMUP_ATTACKS = ("noise", "boost", "duck", "amplitude", "pink_noise")

    def _warmup_attack_name(self) -> Optional[str]:
        """
        Select the attack for this step's cold-start warmup.

            step < clean_steps                → "identity" (no attack)
            clean_steps ≤ step < warmup_steps → random easy attack
            step ≥ warmup_steps               → None (full adaptive curriculum)
        """
        tc = self.cfg.training
        if self.global_step < tc.clean_steps:
            return "identity"
        if self.global_step < tc.curriculum_warmup_steps:
            return random.choice(self._EASY_WARMUP_ATTACKS)
        return None

    # ── LR update ─────────────────────────────────────────────────────────

    def _apply_lr(self) -> None:
        """Push current LR to both optimisers."""
        lr = self.current_lr
        for opt in (self.gen_opt, self.disc_opt):
            for pg in opt.param_groups:
                pg["lr"] = lr

    # ── Forward pass ──────────────────────────────────────────────────────

    def _forward(
        self,
        audio:   Tensor,   # [B, 1, T]
        message: Tensor,   # [B, n_bits]  {0, 1}
        stage:   int,
        p_de:    float,
    ) -> Tuple[LossComponents, float, Tensor, str]:
        """
        One forward pass through embedder → (optional double-encode) →
        attack → detector → losses.

        Returns:
            gen_components:  LossComponents from generator_step()
            disc_loss_val:   float (0.0 in stage 1)
            logits:          [B, n_bits] detector logits
            attack_name:     str name of the attack applied
        """
        tc   = self.cfg.training
        amp  = tc.use_amp and self.device.type == "cuda"

        with torch.autocast(device_type=self.device.type, enabled=amp):

            # ── 1. Embed primary message ───────────────────────────────────
            x_wm, _, _ = self.embedder(audio, message)   # [B, 1, T]

            # ── 2. Double-encoding (Stage 2 ramp) ─────────────────────────
            active_message = message
            if p_de > 0.0 and random.random() < p_de:
                msg2 = torch.randint(
                    0, 2, message.shape,
                    dtype=message.dtype, device=message.device,
                )
                # Embed a second message into the already-watermarked audio.
                # Detector must recover msg2; x_wm_de is the new "original"
                # for perceptual losses (audio quality relative to x_wm).
                x_wm_de, _, _ = self.embedder(x_wm.detach(), msg2)
                active_message = msg2
                x_wm           = x_wm_de

            # ── 3. Apply attack (cold-start warmup → full curriculum) ──────
            forced = self._warmup_attack_name()
            x_attacked, attack_name = self.attack_layer(x_wm, attack_name=forced)

            # ── 4. Detect from attacked audio ──────────────────────────────
            mag_attacked, _ = self.embedder.stft(x_attacked)    # [B, 1025, 188]
            s_mag_attacked  = mag_attacked.unsqueeze(1)          # [B, 1, 1025, 188]
            logits          = self.detector(s_mag_attacked)      # [B, n_bits]

            # ── 5. Discriminator forward (stage 2 only) ────────────────────
            if stage == 2:
                # Real audio features (detached — used as FM targets)
                with torch.no_grad():
                    real_scores_fm, real_feats = self.discriminator(audio)

                # Fake audio — keep graph for generator FM + adv losses
                fake_scores_g, fake_feats = self.discriminator(x_wm)

                # Fake audio — detached for discriminator update
                fake_scores_d, _          = self.discriminator(x_wm.detach())
                real_scores_d, _          = self.discriminator(audio)

            else:
                real_feats    = []
                fake_scores_g = []
                fake_feats    = []
                fake_scores_d = []
                real_scores_d = []

            # ── 6. Generator loss ──────────────────────────────────────────
            gen_comp = self.aura_loss.generator_step(
                x_orig      = audio,
                x_wm        = x_wm,
                logits      = logits,
                target_bits = active_message.float(),
                fake_scores = fake_scores_g,
                fake_feats  = fake_feats,
                real_feats  = real_feats,
                stage       = stage,
            )

            # ── 7. Discriminator loss (stage 2 only) ──────────────────────
            if stage == 2:
                disc_loss = self.aura_loss.discriminator_step(
                    real_scores_d, fake_scores_d
                )
                disc_loss_val = disc_loss.item()
            else:
                disc_loss     = None
                disc_loss_val = 0.0

        return gen_comp, disc_loss_val, disc_loss, logits.detach(), attack_name

    # ── train_step ────────────────────────────────────────────────────────

    def train_step(self, audio: Tensor, message: Tensor) -> StepResult:
        """
        Process one micro-batch.

        Accumulates gradients for ``grad_accum_steps`` calls, then fires
        both optimisers.  Returns a StepResult on every call regardless of
        whether the optimisers stepped (check ``result.did_step``).

        Args:
            audio:   [B, 1, 96000]  mono waveform, peak-normalised
            message: [B, n_bits]    binary bits {0, 1} as int or float

        Returns:
            StepResult with all scalars for this micro-step.
        """
        self._apply_lr()

        stage = self.current_stage
        p_de  = self.current_p_de

        self.embedder.train()
        self.detector.train()
        if stage == 2:
            self.discriminator.train()

        # ── Forward ────────────────────────────────────────────────────────
        gen_comp, disc_loss_val, disc_loss_tensor, logits, attack_name = (
            self._forward(audio, message, stage, p_de)
        )

        # Scale loss by accumulation steps so effective LR is independent
        scaled_gen_loss  = gen_comp.total / self._grad_denom
        scaled_disc_loss = (
            disc_loss_tensor / self._grad_denom
            if disc_loss_tensor is not None else None
        )

        # ── Backward (generator) ───────────────────────────────────────────
        # Gen loss (msg + stft + adv_G + fm + nmr) flows through both
        # embedder/detector AND discriminator (for adv + FM terms).
        # We zero discriminator grads AFTER gen backward so they don't
        # contaminate the disc update — standard GAN training practice.
        self.gen_scaler.scale(scaled_gen_loss).backward()

        # ── Backward (discriminator) ───────────────────────────────────────
        if scaled_disc_loss is not None:
            # Purge any disc-param gradients that leaked from gen backward
            for p in self.discriminator.parameters():
                p.grad = None
            self.disc_scaler.scale(scaled_disc_loss).backward()

        # ── Update curriculum ──────────────────────────────────────────────
        # Only record once the full adaptive curriculum is active, so warmup
        # attacks (incl. the "identity" no-op) don't pollute adaptive stats.
        if self.global_step >= self.cfg.training.curriculum_warmup_steps:
            bit_loss_for_curriculum = gen_comp.msg.item()
            self.attack_layer.curriculum.record(attack_name, bit_loss_for_curriculum)

        # ── Accumulation counter ───────────────────────────────────────────
        self._accum_count += 1
        did_step = self._accum_count >= self.cfg.training.grad_accum_steps

        if did_step:
            # ── Generator optimiser step ───────────────────────────────────
            self.gen_scaler.unscale_(self.gen_opt)
            nn.utils.clip_grad_norm_(
                [p for pg in self.gen_opt.param_groups for p in pg["params"]],
                self.cfg.training.max_grad_norm,
            )
            self.gen_scaler.step(self.gen_opt)
            self.gen_scaler.update()
            self.gen_opt.zero_grad(set_to_none=True)

            # ── Discriminator optimiser step (stage 2 only) ────────────────
            if stage == 2 and scaled_disc_loss is not None:
                self.disc_scaler.unscale_(self.disc_opt)
                nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(),
                    self.cfg.training.max_grad_norm,
                )
                self.disc_scaler.step(self.disc_opt)
                self.disc_scaler.update()
                self.disc_opt.zero_grad(set_to_none=True)

            self._accum_count = 0
            self.global_step += 1

        # ── Bit accuracy ───────────────────────────────────────────────────
        with torch.no_grad():
            pred_bits = (logits > 0).long()
            target    = message.long() if message.is_floating_point() is False else message.long()
            bit_acc   = (pred_bits == target).float().mean().item()

        return StepResult(
            step        = self.global_step,
            stage       = stage,
            attack_name = attack_name,
            lr          = self.current_lr,
            p_de        = p_de,
            loss_msg    = gen_comp.msg.item(),
            loss_stft   = gen_comp.stft.item(),
            loss_adv    = gen_comp.adv.item(),
            loss_fm     = gen_comp.fm.item(),
            loss_nmr    = gen_comp.nmr.item(),
            loss_gen    = gen_comp.total.item(),
            loss_disc   = disc_loss_val,
            bit_acc     = bit_acc,
            did_step    = did_step,
        )

    # ── Checkpointing ─────────────────────────────────────────────────────

    def save_checkpoint(self, path: str | Path) -> None:
        """
        Save full training state to ``path``.

        Saved keys:
            global_step, gen_state, det_state, disc_state,
            gen_opt, disc_opt, gen_scaler, disc_scaler, curriculum
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "global_step":  self.global_step,
                "gen_state":    self.embedder.state_dict(),
                "det_state":    self.detector.state_dict(),
                "disc_state":   self.discriminator.state_dict(),
                "gen_opt":      self.gen_opt.state_dict(),
                "disc_opt":     self.disc_opt.state_dict(),
                "gen_scaler":   self.gen_scaler.state_dict(),
                "disc_scaler":  self.disc_scaler.state_dict(),
                "curriculum":   self.attack_layer.curriculum.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        """
        Restore full training state from ``path``.

        All tensors are mapped to ``self.device``.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.global_step = ckpt["global_step"]
        self.embedder.load_state_dict(ckpt["gen_state"])
        self.detector.load_state_dict(ckpt["det_state"])
        self.discriminator.load_state_dict(ckpt["disc_state"])
        self.gen_opt.load_state_dict(ckpt["gen_opt"])
        self.disc_opt.load_state_dict(ckpt["disc_opt"])
        self.gen_scaler.load_state_dict(ckpt["gen_scaler"])
        self.disc_scaler.load_state_dict(ckpt["disc_scaler"])
        self.attack_layer.curriculum.load_state_dict(ckpt["curriculum"])

    def prune_checkpoints(
        self,
        checkpoint_dir: str | Path,
        keep: Optional[int] = None,
    ) -> None:
        """
        Delete oldest checkpoints, keeping only the last ``keep`` files.

        Args:
            checkpoint_dir: directory containing *.pt checkpoint files
            keep:           number to retain (defaults to cfg.training.keep_last_n_checkpoints)
        """
        keep = keep or self.cfg.training.keep_last_n_checkpoints
        ckpt_dir = Path(checkpoint_dir)
        ckpts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        for old in ckpts[:-keep]:
            old.unlink()

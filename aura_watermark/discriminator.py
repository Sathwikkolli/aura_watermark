"""
BigVGAN-style discriminator for AURA.

Two sub-discriminators (matching BigVGAN / HiFi-GAN architecture):
  1. MultiPeriodDiscriminator  (MPD)      — periods [2, 3, 5, 7, 11]
  2. MultiScaleSTFTDiscriminator (MSSTFTD) — 3 STFT scales

Each sub-discriminator returns:
    (score, features)
    where score    = final logit map (for adversarial loss)
    and   features = list of intermediate feature maps (for FM loss)

The combined BigVGANDiscriminator returns:
    (all_scores, all_features)  — 8 entries each (5 MPD + 3 MSSTFTD)

Design decisions:
    - Weight-norm on all conv layers (HiFi-GAN style) [paper]
    - Spectral-norm on the first period discriminator only [HiFi-GAN impl]
    - LS-GAN loss (not hinge) — consistent with BigVGAN v2 release
    - STFT discriminator input: [real, imag] stacked as 2 channels [EnCodec]
    - No discriminator inside losses.py — kept separate for two-optimizer setup
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm
from typing import List, Tuple

Tensor = torch.Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _wn(module: nn.Module) -> nn.Module:
    """Apply weight normalisation."""
    return weight_norm(module)


def _sn(module: nn.Module) -> nn.Module:
    """Apply spectral normalisation."""
    return spectral_norm(module)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Period Discriminator (MPD)
# ─────────────────────────────────────────────────────────────────────────────

class PeriodDiscriminator(nn.Module):
    """
    Single-period discriminator from HiFi-GAN (used unchanged in BigVGAN).

    Folds the 1D waveform [B, 1, T] into a 2D grid [B, 1, T//p, p],
    then applies 5 strided 2D Conv layers (kernel 5x1, stride 3x1) to
    capture quasi-periodic patterns at the given period.

    Args:
        period:           folding period (one of 2, 3, 5, 7, 11)
        use_spectral_norm: if True, use SN instead of WN (for first sub-disc)
    """

    def __init__(self, period: int, use_spectral_norm: bool = False):
        super().__init__()
        self.period = period
        norm = _sn if use_spectral_norm else _wn

        # Channel progression: 1 → 32 → 128 → 512 → 1024 → 1024
        # kernel (5,1): 5 steps along the time axis within each period slice
        # stride (3,1): subsample time, keep period dimension intact
        self.convs = nn.ModuleList([
            norm(nn.Conv2d(1,    32,   (5, 1), stride=(3, 1), padding=(2, 0))),
            norm(nn.Conv2d(32,   128,  (5, 1), stride=(3, 1), padding=(2, 0))),
            norm(nn.Conv2d(128,  512,  (5, 1), stride=(3, 1), padding=(2, 0))),
            norm(nn.Conv2d(512,  1024, (5, 1), stride=(3, 1), padding=(2, 0))),
            norm(nn.Conv2d(1024, 1024, (5, 1), stride=(1, 1), padding=(2, 0))),
        ])
        self.conv_post = norm(
            nn.Conv2d(1024, 1, (3, 1), stride=(1, 1), padding=(1, 0))
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """
        Args:
            x: [B, 1, T]

        Returns:
            score:    [B, -1]  flattened logit map
            features: list of 6 intermediate tensors (for FM loss)
        """
        features: List[Tensor] = []
        B, C, T = x.shape
        p = self.period

        # Pad to make T divisible by period (reflect padding, no DC bias)
        if T % p != 0:
            pad_len = p - (T % p)
            x = F.pad(x, (0, pad_len), mode="reflect")
            T = x.shape[-1]

        # Reshape 1D → 2D: [B, 1, T] → [B, 1, T//p, p]
        x = x.view(B, C, T // p, p)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, negative_slope=0.1)
            features.append(x)

        x = self.conv_post(x)
        features.append(x)

        # Flatten all spatial dims to get a 1D score per sample
        score = x.flatten(start_dim=1)   # [B, -1]
        return score, features


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator: 5 PeriodDiscriminators.

    Periods [2, 3, 5, 7, 11] are pairwise coprime, so each sub-discriminator
    sees a different aliasing pattern — collectively they cover all periodicities.

    Returns:
        scores:   List[Tensor] of length 5 — one score per sub-discriminator
        features: List[List[Tensor]] — feature maps per sub-discriminator
    """

    PERIODS: List[int] = [2, 3, 5, 7, 11]

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p, use_spectral_norm=(i == 0))
            for i, p in enumerate(self.PERIODS)
        ])

    def forward(self, x: Tensor) -> Tuple[List[Tensor], List[List[Tensor]]]:
        scores:   List[Tensor]       = []
        features: List[List[Tensor]] = []
        for d in self.discriminators:
            s, f = d(x)
            scores.append(s)
            features.append(f)
        return scores, features


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Scale STFT Discriminator (MSSTFTD)
# ─────────────────────────────────────────────────────────────────────────────

class STFTDiscriminator(nn.Module):
    """
    Single-scale STFT discriminator.

    Computes the STFT of the input waveform, stacks real and imaginary parts
    as 2 input channels, then applies 2D Conv blocks on the spectrogram.

    Architecture adapted from EnCodec / BigVGAN v2:
        [B, 2, F, T'] → Conv blocks → [B, 1, F', T''] → score

    Args:
        n_fft:       FFT size
        hop_length:  hop size
        win_length:  window size
        n_filters:   base channel count (channel progression = n_filters × [1,2,4,4,8,8])
    """

    _CH_MULTS = [1, 2, 4, 4, 8, 8]   # channel multipliers for 6 conv blocks

    def __init__(
        self,
        n_fft:      int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_filters:  int = 32,
    ):
        super().__init__()
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.register_buffer("window", torch.hann_window(win_length))

        # Progressive 2D convolutions
        channels   = [n_filters * m for m in self._CH_MULTS]  # [32,64,128,128,256,256]
        in_ch = 2  # real + imag

        self.convs = nn.ModuleList()
        for i, out_ch in enumerate(channels):
            if i == 0:
                # First layer: wide kernel along frequency axis for spectral context
                k, s, pad = (3, 8), (1, 1), (1, 4)
            elif i < 4:
                # Middle layers: stride 2 along frequency to downsample
                k, s, pad = (3, 3), (2, 1), (1, 1)
            else:
                # Last two layers: no stride
                k, s, pad = (3, 3), (1, 1), (1, 1)
            self.convs.append(
                _wn(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=pad))
            )
            in_ch = out_ch

        self.conv_post = _wn(
            nn.Conv2d(in_ch, 1, kernel_size=(3, 3), padding=(1, 1))
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """
        Args:
            x: [B, 1, T]

        Returns:
            score:    [B, -1]
            features: list of intermediate feature tensors
        """
        features: List[Tensor] = []

        # STFT: [B, T] → [B, F, T'] complex
        X = torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )   # [B, F, T']

        # Stack real + imag → [B, 2, F, T']
        h = torch.stack([X.real, X.imag], dim=1)

        for conv in self.convs:
            h = conv(h)
            h = F.leaky_relu(h, negative_slope=0.1)
            features.append(h)

        h = self.conv_post(h)
        features.append(h)

        score = h.flatten(start_dim=1)   # [B, -1]
        return score, features


class MultiScaleSTFTDiscriminator(nn.Module):
    """
    Multi-Scale STFT Discriminator: 3 STFTDiscriminators at different resolutions.

    Scales chosen to match MultiResSTFTConfig (48 kHz):
        scale 0: n_fft=512,  hop=128,  win=512   (fine temporal resolution)
        scale 1: n_fft=1024, hop=256,  win=1024  (balanced)
        scale 2: n_fft=2048, hop=512,  win=2048  (coarse, matches encoder STFT)

    Returns:
        scores:   List[Tensor] of length 3
        features: List[List[Tensor]] of length 3
    """

    SCALES = [
        dict(n_fft=512,  hop_length=128, win_length=512),
        dict(n_fft=1024, hop_length=256, win_length=1024),
        dict(n_fft=2048, hop_length=512, win_length=2048),
    ]

    def __init__(self, n_filters: int = 32):
        super().__init__()
        self.discriminators = nn.ModuleList([
            STFTDiscriminator(**s, n_filters=n_filters)
            for s in self.SCALES
        ])

    def forward(self, x: Tensor) -> Tuple[List[Tensor], List[List[Tensor]]]:
        scores:   List[Tensor]       = []
        features: List[List[Tensor]] = []
        for d in self.discriminators:
            s, f = d(x)
            scores.append(s)
            features.append(f)
        return scores, features


# ─────────────────────────────────────────────────────────────────────────────
# BigVGANDiscriminator — combines MPD + MSSTFTD
# ─────────────────────────────────────────────────────────────────────────────

class BigVGANDiscriminator(nn.Module):
    """
    Combined BigVGAN discriminator: MPD (5) + MSSTFTD (3) = 8 sub-discriminators.

    Used during AURA Stage 2 training to compute adversarial and
    feature-matching losses.

    Usage:
        disc = BigVGANDiscriminator()

        # Real audio (for discriminator update)
        real_scores, real_feats = disc(x_orig)

        # Fake audio — detach for discriminator update, keep graph for generator
        fake_scores_d, _ = disc(x_wm.detach())   # discriminator step
        fake_scores_g, fake_feats_g = disc(x_wm) # generator step

    Returns:
        scores:   List[Tensor] of length 8
        features: List[List[Tensor]] of length 8
    """

    N_TOTAL = 8   # 5 MPD + 3 MSSTFTD

    def __init__(self):
        super().__init__()
        self.mpd     = MultiPeriodDiscriminator()
        self.msstftd = MultiScaleSTFTDiscriminator()

    def forward(self, x: Tensor) -> Tuple[List[Tensor], List[List[Tensor]]]:
        """
        Args:
            x: [B, 1, T] mono waveform

        Returns:
            scores:   8 score tensors  [MPD×5, MSSTFTD×3]
            features: 8 feature lists  [MPD×5, MSSTFTD×3]
        """
        mpd_scores,     mpd_feats     = self.mpd(x)
        msstftd_scores, msstftd_feats = self.msstftd(x)

        all_scores   = mpd_scores   + msstftd_scores    # 8 tensors
        all_features = mpd_feats    + msstftd_feats     # 8 lists

        return all_scores, all_features

    def count_parameters(self) -> dict:
        def count(m):
            return sum(p.numel() for p in m.parameters())
        return {
            "mpd":     count(self.mpd),
            "msstftd": count(self.msstftd),
            "total":   count(self),
        }

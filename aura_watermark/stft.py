"""
STFT front-end and iSTFT back-end for AURA.

Design decisions (all confirmed from paper unless noted):
  - n_fft=2048, hop=512, win=2048, hann window, center=True  [paper]
  - No extra pre-padding needed: center=True with 96000 samples
    gives T=188 naturally:
      total = 96000 + 2*(2048//2) = 98048
      T     = 1 + floor((98048-2048)/512) = 1 + 187 = 188    [derived]
  - Magnitude only modified by embedder; phase preserved      [paper]
  - Multiplicative mask: S_wm = S_mag × M                    [paper]
  - Output clipped to [-1, 1] after iSTFT                    [standard]

Shapes (B = batch size):
  Input waveform:  [B, 1, 96000]
  Magnitude:       [B, 1025, 188]
  Phase:           [B, 1025, 188]
  Output waveform: [B, 1, 96000]
"""

import torch
import torch.nn as nn

from .config import STFTConfig


class STFTProcessor(nn.Module):
    """
    Converts a raw mono waveform to magnitude and phase spectrograms.

    center=True (PyTorch default) pads the signal by n_fft//2=1024 on
    each side before computing frames, which gives exactly T=188 frames
    for a 96000-sample input:
        padded = 96000 + 2*1024 = 98048
        T      = 1 + floor((98048-2048)/512) = 1 + 187 = 188

    Args:
        cfg: STFTConfig

    Input:
        waveform: [B, 1, 96000]  mono audio, 2 seconds at 48 kHz

    Returns:
        magnitude: [B, 1025, 188]  |STFT|, always >= 0
        phase:     [B, 1025, 188]  angle(STFT), in [-pi, pi]
    """

    def __init__(self, cfg: STFTConfig = STFTConfig()):
        super().__init__()
        self.cfg = cfg

        # Register Hann window as a buffer so it:
        #   (a) moves to the correct device automatically with .to(device)
        #   (b) is saved/loaded with model state_dict
        #   (c) is not treated as a learnable parameter
        self.register_buffer(
            "window",
            torch.hann_window(cfg.win_length),
        )

    def forward(
        self, waveform: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            waveform: [B, 1, T]  where T must be cfg.segment_samples (96000)

        Returns:
            magnitude: [B, 1025, 188]
            phase:     [B, 1025, 188]
        """
        B, C, T = waveform.shape

        if C != 1:
            raise ValueError(
                f"STFTProcessor expects mono audio (C=1), got C={C}. "
                "Convert to mono before calling."
            )
        if T != self.cfg.segment_samples:
            raise ValueError(
                f"Expected waveform length {self.cfg.segment_samples}, got {T}. "
                "Ensure audio is exactly 2 seconds at 48 kHz."
            )

        # Remove channel dim: [B, 1, T] → [B, T]
        x = waveform.squeeze(1)

        # STFT with center=True
        # center=True pads by n_fft//2 on each side internally,
        # no manual pre-padding needed.
        # Output shape: [B, n_fft//2+1, num_frames] = [B, 1025, 188]
        stft_complex = torch.stft(
            x,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )   # [B, 1025, 188]

        # Stabilised magnitude: sqrt(re^2 + im^2 + eps) instead of torch.abs().
        # torch.abs() on a complex tensor has an INFINITE gradient at bins whose
        # magnitude is ~0 (d|z|/dz = z/|z| → 0/0). The detector backprops through
        # a fresh STFT of the *attacked* audio, so a single near-zero bin yields
        # an inf gradient that clip_grad_norm_ then smears across all params,
        # permanently NaN-poisoning the weights. The eps floor keeps it finite.
        magnitude = (stft_complex.real.pow(2) + stft_complex.imag.pow(2) + 1e-10).sqrt()
        phase = torch.angle(stft_complex)      # in [-pi, pi]

        return magnitude, phase


class ISTFTReconstructor(nn.Module):
    """
    Reconstructs a mono waveform from a (watermarked) magnitude and
    the original (frozen) phase.

    Reconstruction path:
        STFT_wm  = torch.polar(magnitude_wm, phase_original)
        waveform = torch.istft(STFT_wm, ..., length=96000)
        waveform = clamp(waveform, -1, 1)

    Args:
        cfg: STFTConfig (must match the STFTProcessor used for encoding).

    Input:
        magnitude: [B, 1025, 188]  watermarked magnitude  (S_mag × M)
        phase:     [B, 1025, 188]  original phase (never modified)

    Returns:
        waveform: [B, 1, 96000]  clipped to [-1, 1]
    """

    def __init__(self, cfg: STFTConfig = STFTConfig()):
        super().__init__()
        self.cfg = cfg

        self.register_buffer(
            "window",
            torch.hann_window(cfg.win_length),
        )

    def forward(
        self,
        magnitude: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            magnitude: [B, 1025, 188]
            phase:     [B, 1025, 188]

        Returns:
            waveform: [B, 1, 96000]
        """
        if magnitude.shape != phase.shape:
            raise ValueError(
                f"magnitude and phase must have the same shape. "
                f"Got {magnitude.shape} vs {phase.shape}."
            )

        # Reconstruct complex STFT from polar form:
        #   Z = |Z| * exp(j*angle) = magnitude * (cos(phase) + j*sin(phase))
        stft_complex = torch.polar(magnitude, phase)   # [B, 1025, 188]

        # iSTFT → [B, segment_samples]
        # length=segment_samples tells istft to trim output to exactly
        # 96000 samples, matching the original waveform length.
        # With center=True and Hann window at 75% overlap the round-trip
        # is near-lossless (SI-SNR > 80 dB).
        waveform = torch.istft(
            stft_complex,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=self.window,
            center=True,
            length=self.cfg.segment_samples,   # trim to 96000 exactly
        )   # [B, 96000]

        # Add channel dimension back: [B, 96000] → [B, 1, 96000]
        waveform = waveform.unsqueeze(1)

        # Hard clip to [-1, 1].
        # Normally M ≈ 1 keeps values in range; clipping handles rare
        # edge cases where large M values push samples out of bounds.
        waveform = torch.clamp(waveform, -1.0, 1.0)

        return waveform

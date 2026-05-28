"""
AURADecoder — watermark detector.

Architecture (confirmed from paper code):
    Input:  STFT magnitude spectrogram, treated as a single-channel image.
            Shape: [B, 1, 1025, 188]

    4 × (Conv2d → GroupNorm → LeakyReLU) strided blocks:
        ch:   1 →  64 → 128 → 256 → 512
        freq: 1025 → 513 → 257 → 129 →  65
        time:  188 →  94 →  47 →  24 →  12

    → AdaptiveAvgPool2d((1,1))  → [B, 512]
    → Linear(512 → 32)          → [B, 32]  raw logits

Key design decisions (from paper unless noted):
    - 2D Conv on spectrogram magnitude (NOT 1D on raw waveform)   [paper]
    - GroupNorm(num_groups=32)                                     [paper]
    - LeakyReLU(0.2)                                              [paper]
    - No sigmoid in forward — BCEWithLogitsLoss during training    [paper]
    - Sigmoid only at inference (.detect())                        [paper]

Usage:
    detector = AURADecoder()
    logits = detector(s_mag)          # [B, 32] — use with BCEWithLogitsLoss
    bits   = detector.detect(s_mag)   # [B, 32] — probabilities in [0,1]
"""

import torch
import torch.nn as nn

from .config import AURAConfig, DetectorConfig


def _conv_block(in_ch: int, out_ch: int, num_groups: int, negative_slope: float) -> nn.Sequential:
    """
    One strided Conv2d block: Conv → GroupNorm → LeakyReLU.

    Args:
        in_ch:          input channel count
        out_ch:         output channel count
        num_groups:     GroupNorm groups (must divide out_ch)
        negative_slope: LeakyReLU slope (0.2)

    Returns:
        nn.Sequential of the three layers.
    """
    assert out_ch % num_groups == 0, (
        f"out_ch ({out_ch}) must be divisible by num_groups ({num_groups})"
    )
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,    # GroupNorm has its own affine params
        ),
        nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
        nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
    )


class AURADecoder(nn.Module):
    """
    AURADecoder: detects / decodes the 32-bit watermark from STFT magnitude.

    The detector treats the spectrogram as a single-channel image and
    applies 4 strided Conv2d blocks to progressively downsample it.
    A global average pool collapses the spatial dims, and a final linear
    layer produces 32 raw logits — one per watermark bit.

    Args:
        cfg: AURAConfig (uses cfg.detector and cfg.message)

    Inputs:
        s_mag: [B, 1, n_freq_bins, n_time_frames]
               STFT magnitude of the audio to analyse.
               During training this is the watermarked magnitude (s_wm);
               at inference it can be any audio's magnitude.

    Outputs (forward):
        logits: [B, n_bits]  raw pre-sigmoid logits
                → feed directly to BCEWithLogitsLoss during training.

    Outputs (detect):
        probs:  [B, n_bits]  sigmoid(logits), values in (0, 1)
                → threshold at 0.5 to recover binary bits.
    """

    def __init__(self, cfg: AURAConfig = AURAConfig()):
        super().__init__()

        det = cfg.detector
        msg = cfg.message

        # ── Channel progression: 1 → 64 → 128 → 256 → 512 ───────────────
        ch_prog    = list(det.channel_progression)           # [64, 128, 256, 512]
        channels   = [det.in_channels] + ch_prog             # [1, 64, 128, 256, 512]
        num_groups = det.groupnorm_groups                    # 32
        neg_slope  = det.leaky_relu_slope                    # 0.2

        self.blocks = nn.ModuleList([
            _conv_block(channels[i], channels[i + 1], num_groups, neg_slope)
            for i in range(len(channels) - 1)
        ])

        # ── Global pooling: spatial dims → (1, 1) ─────────────────────────
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # ── Readout: 512 → n_bits (32) — raw logits ───────────────────────
        self.head = nn.Linear(ch_prog[-1], msg.n_bits)

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(self, s_mag: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s_mag: [B, 1, n_freq_bins, n_time_frames]
                   STFT magnitude spectrogram (single channel).

        Returns:
            logits: [B, n_bits]  raw pre-sigmoid logits.
                    Do NOT apply sigmoid here — use BCEWithLogitsLoss.
        """
        x = s_mag                                  # [B, 1, 1025, 188]

        for block in self.blocks:
            x = block(x)                           # progressive downsampling

        # x: [B, 512, ~65, ~12]
        x = self.pool(x)                           # [B, 512, 1, 1]
        x = x.flatten(start_dim=1)                 # [B, 512]
        logits = self.head(x)                      # [B, 32]

        return logits

    # ── Inference convenience ─────────────────────────────────────────────

    def detect(self, s_mag: torch.Tensor) -> torch.Tensor:
        """
        Run detection and return bit probabilities.

        Args:
            s_mag: [B, 1, n_freq_bins, n_time_frames]

        Returns:
            probs: [B, n_bits]  values in (0, 1).
                   Threshold at 0.5 to get binary bits:
                       bits = (probs > 0.5).long()
        """
        with torch.no_grad():
            logits = self.forward(s_mag)
        return torch.sigmoid(logits)

    def decode_bits(self, s_mag: torch.Tensor) -> torch.Tensor:
        """
        Return hard binary bit decisions (0 or 1).

        Args:
            s_mag: [B, 1, n_freq_bins, n_time_frames]

        Returns:
            bits: [B, n_bits]  dtype=torch.long, values in {0, 1}
        """
        probs = self.detect(s_mag)
        return (probs > 0.5).long()

    def count_parameters(self) -> dict:
        """Return parameter counts per sub-module."""
        def count(m):
            return sum(p.numel() for p in m.parameters())

        return {
            "blocks": count(self.blocks),
            "head":   count(self.head),
            "total":  count(self),
        }

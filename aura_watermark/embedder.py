"""
Stegaformer Embedder — full watermark embedding pipeline.

Full forward pass:
    waveform  [B, 1, 96000]
    → STFTProcessor          → S_mag [B, 1025, 188], phase [B, 1025, 188]
    → transpose              → [B, 188, 1025]
    → Linear(1025 → 512)    → [B, 188, 512]       input projection
    → StegaformerBackbone   → [B, 188, 512]       8x Conformer + FiLM
    → Linear(512 → 1025)    → [B, 188, 1025]      output projection
    → Softplus              → M [B, 188, 1025]     positive mask
    → transpose             → M [B, 1025, 188]
    → S_wm = S_mag × M      → [B, 1025, 188]      multiplicative application
    → ISTFTReconstructor    → [B, 1, 96000]        watermarked waveform

Key design decisions (confirmed from paper unless noted):
    - Single linear layer for input/output projections          [user confirmed]
    - Softplus activation on output (not tanh)                  [paper]
    - Multiplicative mask: S_wm = S_mag × M                    [paper]
    - Output bias initialised to 0.541 → Softplus ≈ 1.0 at init [us, paper silent]
    - Output weights initialised near zero (std=0.001)          [us, paper silent]
    - M ≈ 1.0 at init → S_wm ≈ S_mag (no watermark at step 0) [derived]
"""

import math
import torch
import torch.nn as nn

from .config import AURAConfig, STFTConfig, ConformerConfig, EmbedderConfig, MessageConfig
from .stft import STFTProcessor, ISTFTReconstructor
from .conformer import StegaformerBackbone


class StegaformerEmbedder(nn.Module):
    """
    Full Stegaformer Embedder.

    Takes a clean mono waveform and a binary message, returns a
    watermarked waveform of identical shape and duration.

    Args:
        cfg: AURAConfig (uses cfg.stft, cfg.conformer, cfg.embedder, cfg.message)

    Inputs:
        waveform: [B, 1, 96000]   clean mono audio, peak-normalised
        message:  [B, 32]         binary watermark bits in {0, 1}

    Outputs:
        watermarked: [B, 1, 96000]  watermarked audio, clipped to [-1, 1]
        mask:        [B, 1025, 188] the positive mask M (useful for NMR loss)
        s_mag:       [B, 1025, 188] original magnitude (useful for NMR loss)
    """

    def __init__(self, cfg: AURAConfig = AURAConfig()):
        super().__init__()

        self.stft_cfg = cfg.stft
        self.emb_cfg = cfg.embedder

        # ── Signal processing (no learnable params) ───────────────────────
        self.stft = STFTProcessor(cfg.stft)
        self.istft = ISTFTReconstructor(cfg.stft)

        # ── Input projection: n_freq_bins → d_model ───────────────────────
        # Single linear layer (confirmed by user from paper)
        self.input_proj = nn.Linear(
            cfg.stft.n_freq_bins,       # 1025
            cfg.conformer.d_model,      # 512
        )

        # ── Stegaformer backbone: 8 Conformer blocks + FiLM ──────────────
        self.backbone = StegaformerBackbone(cfg.conformer, cfg.message)

        # ── Output projection: d_model → n_freq_bins ─────────────────────
        # Linear layer followed by Softplus to guarantee M > 0.
        # Initialised so that Softplus(output) ≈ 1.0 at the start of
        # training (M=1 → no watermark → stable loss from step 1).
        self.output_proj = nn.Linear(
            cfg.conformer.d_model,      # 512
            cfg.stft.n_freq_bins,       # 1025
        )
        self.softplus = nn.Softplus()

        # Apply custom initialisations so M ≈ 1.0 at step 0
        self._init_output_proj()
        self._init_film_generator()

    # ── Initialisation ────────────────────────────────────────────────────

    def _init_output_proj(self) -> None:
        """
        Initialise the output projection so Softplus(output) ≈ 1.0 at step 0.

        Softplus(x) = log(1 + exp(x))
        Softplus(x) = 1.0  when x = log(e - 1) ≈ 0.541

        Strategy:
          - Bias    → 0.541   (dominates when weights are near zero)
          - Weights → N(0, 0.001)

        Paper does not mention this; we derived it from first principles.
        """
        target_bias = math.log(math.e - 1)   # ≈ 0.5413
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.output_proj.bias, target_bias)

    def _init_film_generator(self) -> None:
        """
        Initialise the FiLM projections so every gamma ≈ 1.0, every beta ≈ 0.0.

        With the corrected architecture, FiLMGenerator outputs
        [B, n_blocks, n_submodules, d_model] for both gamma and beta
        (32 unique vectors each).  The projection layers are:
            gamma_proj: Linear(512 → n_blocks * n_submodules * d_model)
            beta_proj:  Linear(512 → n_blocks * n_submodules * d_model)

        Strategy:
          - gamma_proj.bias  → 1.0  (all 32*512 = 16384 elements)
          - beta_proj.bias   → 0.0  (all 16384 elements)
          - Both weight matrices → N(0, 0.001)  (output ≈ bias at init)

        At init this gives FiLM(LN(x)) = 1.0 * LN(x) + 0.0 = LN(x) for every
        sub-module at every block — i.e., the backbone behaves as a plain
        Conformer with no message influence.  Combined with output_proj init,
        M ≈ 1.0 at step 0 → no watermark → stable loss from the first iteration.
        """
        film = self.backbone.film_generator

        # gamma_proj: bias=1.0 everywhere (identity scale for all 32 positions)
        nn.init.normal_(film.gamma_proj.weight, mean=0.0, std=0.001)
        nn.init.constant_(film.gamma_proj.bias, 1.0)

        # beta_proj: bias=0.0 everywhere (zero shift for all 32 positions)
        nn.init.normal_(film.beta_proj.weight, mean=0.0, std=0.001)
        nn.init.constant_(film.beta_proj.bias, 0.0)

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(
        self,
        waveform: torch.Tensor,
        message: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            waveform: [B, 1, 96000]  clean mono audio
            message:  [B, 32]        binary message, values in {0, 1}

        Returns:
            watermarked: [B, 1, 96000]   watermarked waveform
            mask:        [B, 1025, 188]  positive mask M
            s_mag:       [B, 1025, 188]  original magnitude (for NMR loss)
        """
        # ── Step 1: STFT ──────────────────────────────────────────────────
        # s_mag: [B, 1025, 188]  (always >= 0)
        # phase: [B, 1025, 188]  (frozen, never modified)
        s_mag, phase = self.stft(waveform)

        # ── Step 2: Prepare input for Conformer ───────────────────────────
        # Conformer expects [B, T, d_model] — time is the sequence dimension
        # Transpose: [B, 1025, 188] → [B, 188, 1025]
        x = s_mag.transpose(1, 2)                          # [B, 188, 1025]

        # Project frequency bins to d_model space
        x = self.input_proj(x)                             # [B, 188, 512]

        # ── Step 3: Stegaformer backbone ──────────────────────────────────
        # 8 Conformer blocks, each FiLM-conditioned on the message
        x = self.backbone(x, message)                      # [B, 188, 512]

        # ── Step 4: Output projection → positive mask ─────────────────────
        x = self.output_proj(x)                            # [B, 188, 1025]
        x = self.softplus(x)                               # M: all > 0
        # Transpose back: [B, 188, 1025] → [B, 1025, 188]
        mask = x.transpose(1, 2)                           # [B, 1025, 188]

        # ── Step 5: Apply mask multiplicatively ───────────────────────────
        # S_wm = S_mag × M
        # When M = 1.0: S_wm = S_mag (no change)
        # When M > 1.0: amplify that frequency bin
        # When M < 1.0: attenuate that frequency bin
        s_wm = s_mag * mask                                # [B, 1025, 188]

        # ── Step 6: iSTFT reconstruction ──────────────────────────────────
        # Combine watermarked magnitude with original phase
        watermarked = self.istft(s_wm, phase)              # [B, 1, 96000]

        # Guarantee finite audio leaves the embedder. The ISTFT already clamps
        # to [-1, 1], but a non-finite magnitude (possible early in training)
        # would propagate NaN/Inf into downstream attacks — notably the LAME
        # MP3 codec, which aborts the process on non-finite input.
        watermarked = torch.nan_to_num(
            watermarked, nan=0.0, posinf=1.0, neginf=-1.0
        )

        return watermarked, mask, s_mag

    # ── Convenience ───────────────────────────────────────────────────────

    def get_watermark_only(
        self,
        waveform: torch.Tensor,
        message: torch.Tensor,
    ) -> torch.Tensor:
        """Return only the watermarked waveform (for inference)."""
        watermarked, _, _ = self.forward(waveform, message)
        return watermarked

    def count_parameters(self) -> dict:
        """Return parameter counts per sub-module."""
        def count(module):
            return sum(p.numel() for p in module.parameters())

        return {
            "input_proj":  count(self.input_proj),
            "backbone":    count(self.backbone),
            "output_proj": count(self.output_proj),
            "total":       count(self),
        }

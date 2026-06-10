"""
AURA Step 6 — Loss Functions.

Five loss terms (paper Section 3.2):
  1. L_msg  : BCE on watermark bits               lambda = 1.0
  2. L_stft : Multi-resolution STFT loss          lambda = 1.0
  3. L_adv  : BigVGAN adversarial (generator)     lambda = 0.1
  4. L_fm   : Feature matching                    lambda = 2.0
  5. L_nmr  : NMR psychoacoustic                  lambda = 0.5

Discriminator loss (updates BigVGANDiscriminator, not the generator):
  discriminator_adversarial_loss(real_scores, fake_scores)

Training stages:
  Stage 1 (steps 0..70_000):   only L_msg active.
  Stage 2 (steps >70_000):     all five terms active.

The stage switch is passed explicitly to AURALoss.generator_step(stage=...).

Design decisions (paper silent unless noted):
  - LS-GAN for adversarial (matches BigVGAN v2 release)
  - Spectral convergence + log-magnitude L1 for multi-res STFT  [ParallelWaveGAN]
  - Simplified 24-band Bark psychoacoustic model for NMR  [ISO 11172-3 approx.]
  - Bark filterbank and spreading matrix registered as buffers (move with .to())
  - ATH added as a floor to the masking threshold
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import AURAConfig

Tensor = torch.Tensor


# ═════════════════════════════════════════════════════════════════════════════
# 1.  BCE Message Loss
# ═════════════════════════════════════════════════════════════════════════════

def message_loss(logits: Tensor, target: Tensor) -> Tensor:
    """
    Binary cross-entropy for watermark bit recovery.

    Args:
        logits: [B, n_bits]  raw pre-sigmoid logits from AURADecoder.forward()
        target: [B, n_bits]  ground-truth bits in {0, 1} (any dtype)

    Returns:
        Scalar BCE averaged over batch and bits.
    """
    return F.binary_cross_entropy_with_logits(logits, target.float())


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Multi-Resolution STFT Loss
# ═════════════════════════════════════════════════════════════════════════════

class MultiResSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT reconstruction loss (ParallelWaveGAN style).

    For each of the three STFT scales:
        spectral_convergence  = ||M_orig - M_wm||_F / (||M_orig||_F + eps)
        log_magnitude_L1      = mean |log(M_orig + eps) - log(M_wm + eps)|
        scale_loss            = spectral_convergence + log_magnitude_L1

    Total loss = mean over scales of scale_loss.

    Both terms are differentiable through x_wm and complement each other:
    SC targets large spectral errors; LML1 targets fine log-scale differences.

    Args:
        cfg: AURAConfig — uses cfg.multi_res_stft.scales
    """

    _EPS: float = 1e-5

    def __init__(self, cfg: AURAConfig = AURAConfig()):
        super().__init__()
        self.scales = cfg.multi_res_stft.scales   # list of {n_fft, hop_length, win_length}

        # Hann windows — one per scale, registered as buffers so they follow .to(device)
        for i, s in enumerate(self.scales):
            self.register_buffer(f"window_{i}", torch.hann_window(s["win_length"]))

    def _stft_mag(self, x: Tensor, idx: int) -> Tensor:
        """STFT magnitude for scale `idx`. x: [B, T] → [B, F, T']"""
        s = self.scales[idx]
        w = getattr(self, f"window_{idx}")
        X = torch.stft(
            x,
            n_fft=s["n_fft"],
            hop_length=s["hop_length"],
            win_length=s["win_length"],
            window=w,
            center=True,
            return_complex=True,
        )
        return X.abs()   # [B, F, T']

    def forward(self, x_orig: Tensor, x_wm: Tensor) -> Tensor:
        """
        Args:
            x_orig: [B, 1, T]  original waveform
            x_wm:   [B, 1, T]  watermarked waveform

        Returns:
            Scalar mean STFT loss over all scales.
        """
        # Force fp32: under AMP these inputs are float16, but torch.stft + the
        # log-magnitude term are precision-sensitive (and cuFFT is finicky in
        # fp16). Casting here keeps the perceptual loss accurate regardless of
        # autocast, at negligible cost.
        x0 = x_orig.squeeze(1).float()   # [B, T]
        xw = x_wm.squeeze(1).float()

        total = x0.new_zeros(())
        for i in range(len(self.scales)):
            M0 = self._stft_mag(x0, i)   # [B, F, T']
            Mw = self._stft_mag(xw, i)

            # Spectral convergence (Frobenius norm ratio)
            sc = torch.norm(M0 - Mw, p="fro") / (
                torch.norm(M0, p="fro") + self._EPS
            )

            # Log-magnitude L1
            lml1 = F.l1_loss(
                torch.log(M0 + self._EPS),
                torch.log(Mw + self._EPS),
            )

            total = total + sc + lml1

        return total / len(self.scales)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  NMR Psychoacoustic Loss
# ═════════════════════════════════════════════════════════════════════════════

def _hz_to_bark(f: Tensor) -> Tensor:
    """Traunmuller (1990) Hz → Bark conversion. Monotone, differentiable."""
    return 26.81 * f / (1960.0 + f) - 0.53


def _build_bark_filterbank(n_fft: int, sample_rate: int, n_bark: int = 24) -> Tensor:
    """
    Triangular Bark-band filterbank.

    Maps FFT-bin power spectra to n_bark critical bands using triangular
    windows in the Bark domain.  The 24 bands span 0–24 Bark (≈ 0–15.5 kHz).

    Returns:
        [n_bark, n_fft//2+1] float32 — each row is one Bark-band filter
    """
    n_freqs = n_fft // 2 + 1
    freqs   = torch.linspace(0.0, float(sample_rate) / 2.0, n_freqs)
    bark    = _hz_to_bark(freqs)                        # [n_freqs]

    edges   = torch.linspace(0.0, 24.0, n_bark + 1)    # [n_bark+1]

    fb = torch.zeros(n_bark, n_freqs)
    for b in range(n_bark):
        lo  = edges[b].item()
        hi  = edges[b + 1].item()
        mid = (lo + hi) * 0.5

        in_band = (bark >= lo) & (bark < hi)

        asc  = ((bark - lo)  / (mid - lo  + 1e-8)).clamp(0.0, 1.0)
        desc = ((hi  - bark) / (hi  - mid + 1e-8)).clamp(0.0, 1.0)

        tri = torch.where(bark < mid, asc, desc) * in_band.float()
        fb[b] = tri

    # Normalise each filter so it sums to 1 (prevents scale dependence on n_fft)
    row_sum = fb.sum(dim=1, keepdim=True).clamp(min=1e-8)
    fb = fb / row_sum

    return fb   # [n_bark, n_freqs]


def _build_spreading_matrix(n_bark: int = 24) -> Tensor:
    """
    Asymmetric simultaneous-masking spreading function (Schroeder 1979 approx.).

    spreading[i, j] = masking energy contributed by Bark band j to band i.

    Convention:
        delta = i - j  (positive if masker j is below band i)
        Below masker (delta > 0): -25 dB/Bark   (upward masking, steep)
        Above masker (delta < 0):  +3 dB/Bark   (downward masking, shallow)

    Returns:
        [n_bark, n_bark] float32 in linear scale
    """
    idx    = torch.arange(n_bark, dtype=torch.float32)
    delta  = idx.unsqueeze(1) - idx.unsqueeze(0)   # [n_bark, n_bark]  i - j

    db = torch.where(
        delta >= 0,
        -25.0 * delta,   # upward masking: steep decay
         3.0  * delta,   # downward masking: gentle decay (delta<0 → positive dB)
    )
    spread = 10.0 ** (db / 10.0)   # linear power scale

    # Row-normalise so masking threshold is in the same power units as the signal
    row_sum = spread.sum(dim=1, keepdim=True).clamp(min=1e-8)
    spread  = spread / row_sum

    return spread   # [n_bark, n_bark]


class NMRLoss(nn.Module):
    """
    Noise-to-Mask Ratio (NMR) psychoacoustic loss — fully differentiable.

    Measures whether the watermark distortion is perceptually masked by the
    original signal's simultaneous masking threshold in each Bark critical band.

    Algorithm:
      1. Compute short-time power spectra of original and noise (x_wm - x_orig).
      2. Map FFT-bin power to 24 Bark bands via triangular filterbank.
      3. Estimate masking threshold:
             T[b] = (spreading @ P_signal)[b] + ATH[b]
         where spreading is the asymmetric row-normalised matrix above.
      4. NMR[b] = P_noise[b] / (T[b] + eps)
      5. Loss   = mean_{b,sample} ReLU(log10(NMR[b] + eps))
                  = 0 when noise is fully masked; positive when audible.

    Gradients flow through steps 1–5 back to x_wm.

    Args:
        cfg:        AURAConfig — uses cfg.stft.sample_rate
        n_fft:      FFT size for psychoacoustic STFT (default 2048)
        hop_length: hop size  (default 512)
    """

    N_BARK = 24
    _EPS   = 1e-8

    # Absolute Threshold of Hearing per Bark band (dB SPL, simplified ISO 226)
    _ATH_DB: List[float] = [
        40., 30., 20., 15., 10., 5., 0., -3., -5., -6.,
        -7., -8., -9., -10., -10., -9., -8., -7., -6., -5.,
        -3., -1., 3., 10.,
    ]

    def __init__(
        self,
        cfg:        AURAConfig = AURAConfig(),
        n_fft:      int = 2048,
        hop_length: int = 512,
    ):
        super().__init__()
        self.n_fft      = n_fft
        self.hop_length = hop_length
        sr              = cfg.stft.sample_rate   # 48000

        # Bark filterbank: [24, n_fft//2+1]
        fb = _build_bark_filterbank(n_fft, sr, self.N_BARK)
        self.register_buffer("filterbank", fb)

        # Spreading matrix: [24, 24]
        spread = _build_spreading_matrix(self.N_BARK)
        self.register_buffer("spreading", spread)

        # ATH floor in linear power (arbitrary reference units)
        ath_db  = torch.tensor(self._ATH_DB, dtype=torch.float32)
        ath_lin = 10.0 ** (ath_db / 10.0)
        self.register_buffer("ath", ath_lin)   # [24]

        # Hann window for STFT
        self.register_buffer("window", torch.hann_window(n_fft))

    def _power_spectrum(self, x: Tensor) -> Tensor:
        """
        Mean short-time power spectrum.

        Args:
            x: [B, T]

        Returns:
            [B, n_fft//2+1] — mean power per FFT bin, averaged over time frames
        """
        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            return_complex=True,
        )   # [B, F, T']
        return X.abs().pow(2).mean(dim=-1)   # [B, F]

    def _to_bark(self, power: Tensor) -> Tensor:
        """
        Map FFT-bin power to Bark bands.

        Args:
            power: [B, F]

        Returns:
            [B, N_BARK]
        """
        # filterbank: [N_BARK, F]  →  power @ filterbank.T → [B, N_BARK]
        return power @ self.filterbank.t()

    def forward(self, x_orig: Tensor, x_wm: Tensor) -> Tensor:
        """
        Args:
            x_orig: [B, 1, T]  original waveform
            x_wm:   [B, 1, T]  watermarked waveform

        Returns:
            Scalar NMR loss (mean over batch and Bark bands).
        """
        # Force fp32 (see MultiResSTFTLoss): the Bark-band power, log10 and
        # masking-threshold math are precision-sensitive and use torch.stft.
        x0   = x_orig.squeeze(1).float()              # [B, T]
        noise = (x_wm - x_orig).squeeze(1).float()    # [B, T]

        # Power spectra
        P_signal = self._power_spectrum(x0)      # [B, F]
        P_noise  = self._power_spectrum(noise)   # [B, F]

        # Bark-band powers
        B_signal = self._to_bark(P_signal)   # [B, 24]
        B_noise  = self._to_bark(P_noise)    # [B, 24]

        # Masking threshold: spreading @ signal_power + ATH
        # spreading: [24, 24],  B_signal: [B, 24]
        # B_signal @ spreading.T → [B, 24]  (each row: how much masking band i gets)
        threshold = B_signal @ self.spreading.t() + self.ath.unsqueeze(0)  # [B, 24]

        # NMR in log-10 domain: positive means noise exceeds mask
        nmr_log = torch.log10(B_noise / (threshold + self._EPS) + self._EPS)

        # Only penalise audible noise (NMR > 0 dB)
        loss = F.relu(nmr_log)   # [B, 24]

        return loss.mean()


# ═════════════════════════════════════════════════════════════════════════════
# 4 & 5.  Adversarial and Feature-Matching Losses
# ═════════════════════════════════════════════════════════════════════════════

def generator_adversarial_loss(fake_scores: List[Tensor]) -> Tensor:
    """
    LS-GAN generator loss: drive each discriminator score towards 1.

        L_adv_G = (1/N) * sum_i mean((D_i(x_wm) - 1)^2)

    Args:
        fake_scores: list of N score tensors from discriminator(x_wm)

    Returns:
        Scalar generator adversarial loss.
    """
    total = sum(
        F.mse_loss(s, torch.ones_like(s))
        for s in fake_scores
    )
    return total / max(len(fake_scores), 1)


def discriminator_adversarial_loss(
    real_scores: List[Tensor],
    fake_scores: List[Tensor],
) -> Tensor:
    """
    LS-GAN discriminator loss.

        L_D = (1/N) * sum_i [mean((D_i(x_orig) - 1)^2) + mean(D_i(x_wm)^2)]

    IMPORTANT: fake_scores must be computed with x_wm.detach() before calling
    this function, so generator gradients do not flow into the discriminator.

    Args:
        real_scores: list of N score tensors for original audio
        fake_scores: list of N score tensors for watermarked audio (detached)

    Returns:
        Scalar discriminator loss.
    """
    total = sum(
        F.mse_loss(r, torch.ones_like(r)) + F.mse_loss(f, torch.zeros_like(f))
        for r, f in zip(real_scores, fake_scores)
    )
    return total / max(len(real_scores), 1)


def feature_matching_loss(
    real_features: List[List[Tensor]],
    fake_features: List[List[Tensor]],
) -> Tensor:
    """
    Feature-matching loss (L1 on discriminator intermediate activations).

        L_fm = (1/D*L) * sum_d sum_l mean |feats_real_dl.detach() - feats_fake_dl|_1

    Real features are detached — treated as fixed targets.
    Gradients flow only through fake_features (watermarked audio path).

    Args:
        real_features: [D][L]  D discriminators × L layers each, real audio
        fake_features: [D][L]  same structure, watermarked audio

    Returns:
        Scalar feature-matching loss.
    """
    total    = 0.0
    n_pairs  = 0
    for r_layers, f_layers in zip(real_features, fake_features):
        for r_feat, f_feat in zip(r_layers, f_layers):
            total   = total + F.l1_loss(r_feat.detach(), f_feat)
            n_pairs += 1
    return total / max(n_pairs, 1)


# ═════════════════════════════════════════════════════════════════════════════
# LossComponents — named container for loss reporting
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class LossComponents:
    """Per-term losses + weighted total, returned by AURALoss.generator_step()."""
    msg:   Tensor   # BCE message loss
    stft:  Tensor   # multi-resolution STFT loss
    adv:   Tensor   # adversarial generator loss
    fm:    Tensor   # feature matching loss
    nmr:   Tensor   # NMR psychoacoustic loss
    total: Tensor   # lambda-weighted sum

    def as_dict(self) -> Dict[str, float]:
        """Return {name: float} for logging. Detaches automatically."""
        return {k: float(v) for k, v in vars(self).items()}


# ═════════════════════════════════════════════════════════════════════════════
# AURALoss — combined generator + perceptual loss
# ═════════════════════════════════════════════════════════════════════════════

class AURALoss(nn.Module):
    """
    Combined AURA loss for the generator/embedder.

    Stage 1 (steps 0..70_000): only L_msg is active; perceptual terms are 0.
    Stage 2 (steps >70_000):   all five terms are active.

    Pass stage=1 or stage=2 to generator_step() — the training loop is
    responsible for switching at cfg.training.stage1_steps.

    Args:
        cfg: AURAConfig — reads LossConfig lambdas, STFT scales, sample rate.
    """

    def __init__(self, cfg: AURAConfig = AURAConfig()):
        super().__init__()
        lc = cfg.loss

        self.lambda_msg  = lc.lambda_msg    # 1.0
        self.lambda_stft = lc.lambda_stft   # 1.0
        self.lambda_adv  = lc.lambda_adv    # 0.1
        self.lambda_fm   = lc.lambda_fm     # 2.0
        self.lambda_nmr  = lc.lambda_nmr    # 0.5

        self.stft_loss = MultiResSTFTLoss(cfg)
        self.nmr_loss  = NMRLoss(cfg)

    def generator_step(
        self,
        *,
        x_orig:      Tensor,              # [B, 1, T]  original waveform
        x_wm:        Tensor,              # [B, 1, T]  watermarked waveform
        logits:      Tensor,              # [B, n_bits] from AURADecoder
        target_bits: Tensor,              # [B, n_bits] ground-truth bits {0, 1}
        fake_scores: List[Tensor],        # disc(x_wm) scores
        fake_feats:  List[List[Tensor]],  # disc(x_wm) features
        real_feats:  List[List[Tensor]],  # disc(x_orig) features (detached OK)
        stage: int = 2,
    ) -> LossComponents:
        """
        Compute the weighted generator loss.

        Stage 1 (stage=1):  only L_msg computed; stft/adv/fm/nmr are zero tensors.
        Stage 2 (stage=2):  all five terms computed.

        Args:
            x_orig:      original waveform
            x_wm:        watermarked waveform (graph attached)
            logits:      detector logits for x_wm (or attacked x_wm)
            target_bits: ground-truth binary message
            fake_scores: discriminator scores on x_wm
            fake_feats:  discriminator features on x_wm
            real_feats:  discriminator features on x_orig
            stage:       1 or 2

        Returns:
            LossComponents with all six fields populated.
        """
        l_msg = message_loss(logits, target_bits)

        if stage == 2:
            l_stft = self.stft_loss(x_orig, x_wm)
            l_adv  = generator_adversarial_loss(fake_scores)
            l_fm   = feature_matching_loss(real_feats, fake_feats)
            l_nmr  = self.nmr_loss(x_orig, x_wm)
        else:
            # Stage 1: perceptual terms are zero (no grad needed)
            zero = l_msg.new_zeros(())
            l_stft = l_fm = l_adv = l_nmr = zero

        total = (
            self.lambda_msg  * l_msg
            + self.lambda_stft * l_stft
            + self.lambda_adv  * l_adv
            + self.lambda_fm   * l_fm
            + self.lambda_nmr  * l_nmr
        )

        return LossComponents(
            msg=l_msg, stft=l_stft, adv=l_adv,
            fm=l_fm, nmr=l_nmr, total=total,
        )

    def discriminator_step(
        self,
        real_scores: List[Tensor],
        fake_scores: List[Tensor],
    ) -> Tensor:
        """
        LS-GAN discriminator loss (Stage 2 only).

        Call this with fake_scores computed from x_wm.detach() so the
        generator's computation graph is not retained.

        Args:
            real_scores: disc(x_orig) scores
            fake_scores: disc(x_wm.detach()) scores

        Returns:
            Scalar discriminator LS-GAN loss.
        """
        return discriminator_adversarial_loss(real_scores, fake_scores)

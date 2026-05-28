# AURA — Audio Watermarking via Stegaformer

> **Production implementation of the ICASSP 2026 paper:**  
> *"AURA: A Stegaformer-Based Scalable Deep Audio Watermark with Extreme Robustness"*

AURA embeds an imperceptible 32-bit binary watermark into audio by modulating the STFT magnitude spectrum through a FiLM-conditioned Conformer backbone. The watermark survives 20 signal-domain attacks (MP3 compression, resampling, pitch-shift, noise, etc.) and is recovered by a lightweight 2D-Conv detector with > 99% bit accuracy.

---

## Architecture Overview

```
 Raw Audio [B, 1, 96000]
       │
       ▼
 ┌─────────────┐
 │ STFTProcessor│  n_fft=2048, hop=512, center=True → T=188 frames
 └─────┬───────┘
       │  magnitude [B, 1025, 188]   phase [B, 1025, 188]
       ▼
 ┌──────────────────────────────────────────────────────────┐
 │              StegaformerEmbedder                         │
 │                                                          │
 │  Linear(1025→512)  →  8× ConformerBlock  →  Linear(512→1025) │
 │                                                          │
 │  FiLMGenerator(message[B,32]):                           │
 │    Embedding(64,512) → sum → Linear → [B, 8, 4, 512]    │
 │    Injects 32 unique (γ,β) pairs — one per sub-module    │
 │    inside each of 8 blocks (FF1, MHSA, Conv, FF2)        │
 └─────┬────────────────────────────────────────────────────┘
       │  watermarked magnitude [B, 1025, 188]
       │  (combined with original phase)
       ▼
 ┌──────────────────┐
 │ ISTFTReconstructor│  → watermarked audio [B, 1, 96000]
 └──────────────────┘
       │
       ▼  (attacked during training)
 ┌──────────────┐
 │  AttackLayer  │  20 signal-domain attacks + adaptive curriculum
 └──────┬───────┘
        │
        ▼
 ┌─────────────┐
 │ AURADecoder  │  4× Conv2d(stride=2) + GroupNorm(32) + LeakyReLU(0.2)
 │              │  → AdaptiveAvgPool → Linear(512→32) → logits [B, 32]
 └─────────────┘
```

---

## Key Design Decisions

| Component | Design | Source |
|-----------|---------|--------|
| STFT | n_fft=2048, hop=512, hann, center=True → T=188 | Paper |
| Backbone | 8 Conformer blocks, d_model=512, 8 heads | Paper |
| FiLM | 4 applications per block (FF1, MHSA, Conv, FF2) — **32 unique (γ,β) pairs** | Paper |
| FiLMGenerator | Embedding(64,512) → Linear(512→32×512) — **16.8M params** | Paper |
| Message bits | 32-bit binary payload | Paper |
| Detector | 2D Conv on STFT magnitude, BCEWithLogitsLoss | Paper |
| Gradient checkpointing | On each ConformerBlock — saves ~60% VRAM | Implementation |
| Attack robustness | 20 attacks, STE for non-differentiable ops | Paper |
| Curriculum | Adaptive: P_k ∝ max(L̄_k / Σ L̄_j, P_min) | Paper |
| Loss | BCE + Multi-Res STFT + Adversarial + FM + NMR (5 terms) | Paper |
| Discriminator | BigVGAN-style: MPD(5 periods) + MSSTFTD(3 scales) | Paper |

---

## Module Structure

```
aura_watermark/
├── config.py          — All hyperparameters (STFTConfig, ConformerConfig,
│                        AttackConfig, LossConfig, TrainingConfig, …)
├── stft.py            — STFTProcessor  [B,1,T] → (magnitude, phase)
│                        ISTFTReconstructor (magnitude, phase) → [B,1,T]
├── conformer.py       — FiLMGenerator, ConformerBlock, StegaformerBackbone
├── embedder.py        — StegaformerEmbedder (full encode pipeline)
├── detector.py        — AURADecoder (watermark detector / decoder)
├── attacks.py         — 20-attack AttackLayer + AdaptiveCurriculum
├── discriminator.py   — BigVGANDiscriminator (MPD + MSSTFTD)
└── losses.py          — AURALoss: BCE, MultiResSTFT, Adv, FM, NMR
```

```
tests/
├── test_stft.py       — STFT/iSTFT round-trip, shape, phase preservation
├── test_conformer.py  — FiLM uniqueness, gradient checkpointing, parameter count
├── test_embedder.py   — Full encode pipeline, mask range, parameter count
├── test_detector.py   — Detector shapes, BCEWithLogitsLoss, decode_bits
├── test_attacks.py    — All 20 attacks, STE gradients, curriculum
└── test_losses.py     — All 5 loss terms, discriminator shapes, gradients
```

---

## Loss Functions (Stage 2)

```
L_total = λ_msg · L_msg  +  λ_stft · L_stft  +  λ_adv · L_adv  +  λ_fm · L_fm  +  λ_nmr · L_nmr

          λ_msg  = 1.0   BCE on recovered bits
          λ_stft = 1.0   Multi-resolution STFT (spectral convergence + log-mag L1, 3 scales)
          λ_adv  = 0.1   LS-GAN adversarial (BigVGAN discriminator)
          λ_fm   = 2.0   Feature matching (L1 on discriminator intermediate activations)
          λ_nmr  = 0.5   NMR psychoacoustic (24-band Bark model with spreading function)
```

Stage 1 (steps 0–70k): only `L_msg` active — backbone learns to hide bits.  
Stage 2 (steps 70k–200k): all five terms active — audio quality optimised.

---

## Attacks (20 total)

| # | Name | Description |
|---|------|-------------|
| 1 | `noise` | Additive white Gaussian noise (SNR 10–40 dB) |
| 2 | `pink_noise` | Pink (1/f) noise via STFT phase randomisation |
| 3 | `lowpass` | Biquad lowpass (3–6 kHz cutoff) |
| 4 | `bandpass` | Biquad bandpass (300–400 Hz ↔ 7–9 kHz) |
| 5 | `mp3` | MP3 codec at 64/96/128/192 kbps (STE gradient) |
| 6 | `aac` | AAC codec at 32/64/96/128 kbps (STE gradient) |
| 7 | `opus` | Opus codec at 16/24/32/64 kbps (STE gradient) |
| 8 | `resample` | Downsample → upsample (44.1/24/22.05/16 kHz) |
| 9 | `suppress` | Zero 0.1% of samples randomly |
| 10 | `echo` | Add 100 ms delayed echo (decay 0.3) |
| 11 | `smooth` | Moving-average smoothing (window 2–10) |
| 12 | `speed` | Time-scale modification via phase vocoder (0.8–1.2×) |
| 13 | `pitch` | Pitch shift ±2 semitones (resample + STE) |
| 14 | `speed_pitch` | Speed change preserving pitch (0.8–1.2×) |
| 15 | `amplitude` | Random gain in [-6, +6] dB |
| 16 | `boost` | Fixed 1.2× gain |
| 17 | `duck` | Fixed 0.8× gain |
| 18 | `quantize` | Uniform quantisation (4–16 bits) |
| 19 | `phase_shift` | Global STFT phase rotation (energy preserved) |
| 20 | `spaug` | SpecAugment: time + frequency masking |

---

## Parameter Counts

| Module | Parameters |
|--------|-----------|
| FiLMGenerator | 16.84 M |
| StegaformerBackbone | 65.33 M |
| StegaformerEmbedder | 66.38 M |
| AURADecoder | 2.36 M |
| BigVGANDiscriminator | ~23 M |
| **Total (Embedder + Decoder)** | **~68.7 M** |

---

## Installation

```bash
git clone https://github.com/Sathwikkolli/aura_watermark.git
cd aura_watermark
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.1, torchaudio ≥ 2.1

---

## Quick Start

```python
import torch
from aura_watermark import (
    AURAConfig, StegaformerEmbedder, AURADecoder, BigVGANDiscriminator, AURALoss
)

cfg     = AURAConfig()
embedder = StegaformerEmbedder(cfg)
detector = AURADecoder(cfg)

# Embed a 32-bit message into 2 seconds of audio at 48 kHz
audio   = torch.randn(1, 1, 96_000)          # [B, 1, T]
message = torch.randint(0, 2, (1, 32))        # [B, n_bits]

watermarked = embedder(audio, message)         # [B, 1, 96000]

# Detect / decode from (potentially attacked) audio
stft_proc  = embedder.stft
mag, phase = stft_proc(watermarked)
s_mag      = mag.unsqueeze(1)                 # [B, 1, 1025, 188]
bits       = detector.decode_bits(s_mag)      # [B, 32]  {0 or 1}
```

---

## Running Tests

```bash
# Run all test suites
python tests/test_stft.py
python tests/test_conformer.py
python tests/test_embedder.py
python tests/test_detector.py
python tests/test_attacks.py
python tests/test_losses.py
```

Expected: **130+ tests, all passing.**

---

## Training (Planned — Step 7)

Two-stage training:

| Stage | Steps | Active Losses | Notes |
|-------|-------|---------------|-------|
| 1 | 0 – 70k | `L_msg` only | Backbone learns to embed bits |
| 2 | 70k – 200k | All 5 terms | Audio quality optimised via discriminator |

- Optimizer: Adam (lr=1e-4) — separate optimisers for generator and discriminator  
- Batch: 32 × 2 s clips, grad accumulation ×2 (virtual batch 64)  
- AMP FP16 + gradient clipping (max norm 1.0)  
- Double-encoding schedule: P_de ramps from 0 → 50% between steps 70k–90k  

---

## Citation

```bibtex
@inproceedings{aura2026,
  title     = {AURA: A Stegaformer-Based Scalable Deep Audio Watermark with Extreme Robustness},
  booktitle = {ICASSP},
  year      = {2026},
}
```

---

## License

MIT

"""
Central configuration for AURA.
All hyperparameters confirmed from the paper or explicitly decided
during implementation planning. See memory/project_aura.md for decisions.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class STFTConfig:
    # --- confirmed from paper ---
    sample_rate: int = 48_000
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048

    # --- derived ---
    n_freq_bins: int = 1025         # n_fft // 2 + 1
    segment_samples: int = 96_000   # 2 sec × 48 kHz
    n_time_frames: int = 188        # T = 1 + floor(96000/512) = 188 via center=True
    # NOTE: No extra padding needed. torch.stft with center=True pads by
    # n_fft//2=1024 on each side internally:
    #   total = 96000 + 2*1024 = 98048
    #   frames = 1 + floor((98048-2048)/512) = 1 + floor(187.5) = 188 ✓
    # Adding 256 extra samples would give T=189 (breaks detector downsampling).


@dataclass
class ConformerConfig:
    # --- confirmed from paper ---
    d_model: int = 512
    n_heads: int = 8
    n_blocks: int = 8
    conv_kernel_size: int = 31
    dropout: float = 0.1
    ff_expansion: int = 4           # inner dim = d_model × ff_expansion = 2048

    # --- derived ---
    head_dim: int = 64              # d_model // n_heads

    # --- FiLM conditioning ---
    # Paper key innovation: FiLM inserted inside each sub-module after its LayerNorm.
    # 4 sub-modules per block (FF1, MHSA, Conv, FF2) → 4 FiLM ops per block,
    # 32 total across 8 blocks.  Each position gets its own unique (gamma, beta).
    n_film_per_block: int = 4       # number of FiLM applications per Conformer block

    # --- memory ---
    # Recompute Conformer activations during backward instead of storing them.
    # Saves ~60% VRAM on V100 16 GB at the cost of ~30% extra compute.
    use_gradient_checkpointing: bool = False  # A40 has 48GB VRAM — no need to checkpoint


@dataclass
class EmbedderConfig:
    # Input/output projections
    input_proj_in: int = 1025       # n_freq_bins
    input_proj_out: int = 512       # d_model
    output_proj_in: int = 512       # d_model
    output_proj_out: int = 1025     # n_freq_bins

    # Output layer init: bias=0.541 so Softplus(bias) ≈ 1.0 at init
    # (decided by us — paper silent on this)
    output_bias_init: float = 0.541


@dataclass
class DetectorConfig:
    # --- confirmed from paper code ---
    in_channels: int = 1
    channel_progression: Tuple[int, ...] = (64, 128, 256, 512)
    conv_kernel: int = 3
    conv_stride: int = 2
    conv_padding: int = 1
    groupnorm_groups: int = 32
    leaky_relu_slope: float = 0.2
    fc_out: int = 32                # = message_bits


@dataclass
class MessageConfig:
    # --- confirmed from paper ---
    n_bits: int = 32


@dataclass
class MultiResSTFTConfig:
    # --- recommended for 48 kHz, paper silent on exact scales ---
    scales: List[dict] = field(default_factory=lambda: [
        {"n_fft": 512,  "hop_length": 50,  "win_length": 240},
        {"n_fft": 1024, "hop_length": 120, "win_length": 600},
        {"n_fft": 2048, "hop_length": 240, "win_length": 1200},
    ])


@dataclass
class LossConfig:
    # Stage 1: only lambda_msg active
    # Stage 2: all active
    # --- recommended weights (paper silent on exact values) ---
    lambda_msg: float = 1.0
    lambda_stft: float = 1.0
    lambda_adv: float = 0.1
    lambda_fm: float = 2.0
    lambda_nmr: float = 0.5


@dataclass
class DatasetConfig:
    """
    Training corpora (paper-scale): ~2 500 hr Emilia speech + ~2 500 hr FMA music.

    When both corpora are used, ``speech_ratio`` defaults to 0.5 so each
    optimizer step sees speech and music with equal probability.
    """

    emilia_hours: float = 2_500.0
    fma_hours: float = 2_500.0

    # Fraction of combined-training clips drawn from Emilia (speech).
    # Default 0.5 matches equal hour budgets above.
    speech_ratio: float = 0.5

    # FMA tree under ``--fma-root``: ``auto`` tries fma_full, then fma_large, then root.
    fma_subset: str = "auto"   # auto | fma_full | fma_large | root


@dataclass
class TrainingConfig:
    # --- confirmed from paper ---
    optimizer: str = "adam"
    learning_rate: float = 1e-4
    stage1_steps: int = 70_000

    # --- precision ---
    # Whole-graph AMP (float16) corrupts the small watermark signal and is the
    # source of the NaN/Inf that crashed the LAME codec.  The paper assumes full
    # precision; default to fp32.  113M params @ batch 32 fits A40 48 GB in fp32.
    use_amp: bool = False

    # --- cold-start curriculum warmup (paper-faithful bootstrap) ---
    # The detector cannot decode at init, so all-22-attacks-from-step-0 with the
    # adaptive curriculum up-weighting the hardest attacks deadlocks training.
    # Warm up on clean, then easy attacks, before enabling the full curriculum.
    clean_steps: int = 500                 # steps [0, clean_steps): identity (no attack)
    curriculum_warmup_steps: int = 2_000   # steps [clean_steps, this): easy subset only

    # --- recommended (paper silent) ---
    warmup_steps: int = 5_000
    total_steps: int = 200_000
    lr_min: float = 1e-6            # cosine annealing floor

    # Batch / accumulation (A40 48 GB settings)
    # batch 16×accum 4 = 64 effective. Halved from 32×2 because fp32 Stage 2
    # (BigVGAN discriminator: 3 fwd passes + its own backward/Adam on top of the
    # generator graph) OOMs the A40 at batch 32. Effective batch is unchanged.
    batch_size: int = 16            # local batch per step
    grad_accum_steps: int = 4       # virtual batch = 64

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Checkpointing
    save_every_n_steps: int = 5_000
    keep_last_n_checkpoints: int = 5

    # Double-encoding schedule (confirmed from paper)
    de_t_start: int = 70_000
    de_t_warmup: int = 20_000
    de_p_max: float = 0.50

    # Adaptive curriculum (recommended, paper silent)
    p_min: float = 0.01
    curriculum_window_batches: int = 1_000


@dataclass
class AttackConfig:
    """
    Hyperparameters for the 22 signal-domain attacks.
    Double-encoding is configured via TrainingConfig.de_* fields
    and handled inside the training loop.

    All attack parameter ranges are derived from the paper (Section 2.3)
    unless noted.  Adaptive curriculum parameters follow the paper formula:
        P_k_new = Normalize(max(L_k_bar / sum(L_k_bar), P_min))
    """

    # ── Adaptive curriculum ──────────────────────────────────────────────────
    p_min: float = 0.01           # minimum probability floor per attack
    window_size: int = 1_000      # rolling average window in batches

    # ── White noise ──────────────────────────────────────────────────────────
    noise_min_snr_db: float = 10.0
    noise_max_snr_db: float = 40.0

    # ── Pink noise ────────────────────────────────────────────────────────────
    # Amplitude of pink noise relative to signal RMS
    pink_min_scale: float = 0.01
    pink_max_scale: float = 0.10

    # ── Low-pass filter ───────────────────────────────────────────────────────
    lp_min_cutoff_hz: float = 3_000.0   # [paper: 3kHz–6kHz]
    lp_max_cutoff_hz: float = 6_000.0

    # ── High-pass filter ──────────────────────────────────────────────────────
    # Paper Table 1: HP attack. Removes low-frequency content.
    hp_min_cutoff_hz: float = 300.0     # gentle rumble filter
    hp_max_cutoff_hz: float = 3_000.0   # aggressive thinning

    # ── Band-pass filter ──────────────────────────────────────────────────────
    bp_low_min_hz:  float = 300.0       # [paper: 300–400 Hz]
    bp_low_max_hz:  float = 400.0
    bp_high_min_hz: float = 7_000.0     # [paper: 7–9 kHz]
    bp_high_max_hz: float = 9_000.0

    # ── Codec bitrates (kbps) ─────────────────────────────────────────────────
    mp3_bitrates:  Tuple[int, ...] = (64, 96, 128, 192)
    aac_bitrates:  Tuple[int, ...] = (32, 64, 96, 128)
    opus_bitrates: Tuple[int, ...] = (16, 24, 32, 64)

    # ── Resampling targets (Hz) ───────────────────────────────────────────────
    resample_rates: Tuple[int, ...] = (44_100, 24_000, 22_050, 16_000)

    # ── Suppression ───────────────────────────────────────────────────────────
    suppress_fraction: float = 0.001    # 0.1% of samples zeroed [paper]

    # ── Echo ──────────────────────────────────────────────────────────────────
    echo_delay_ms: float = 100.0        # [paper: 100ms]
    echo_decay:    float = 0.3          # [paper: 0.3]

    # ── Smooth ────────────────────────────────────────────────────────────────
    smooth_min_window: int = 2          # [paper: 2–10]
    smooth_max_window: int = 10

    # ── Speed / pitch ─────────────────────────────────────────────────────────
    speed_min: float = 0.8              # TSM rate range
    speed_max: float = 1.2
    pitch_min_semitones: float = -2.0   # semitones relative to original
    pitch_max_semitones: float = 2.0
    speed_pitch_min: float = 0.8        # standard resample rate range
    speed_pitch_max: float = 1.2

    # ── Quantization ──────────────────────────────────────────────────────────
    quantize_min_bits: int = 4
    quantize_max_bits: int = 16

    # ── Spectrogram augmentation ─────────────────────────────────────────────
    spaug_max_time_mask:  int = 20      # max consecutive time frames
    spaug_max_freq_mask:  int = 50      # max consecutive freq bins
    spaug_num_time_masks: int = 2
    spaug_num_freq_masks: int = 2

    # ── Reverb ────────────────────────────────────────────────────────────────
    # Synthetic RIR: exponentially-decaying white noise  [paper: "given RIR"]
    reverb_min_dur_ms: float = 100.0   # shortest RIR duration (ms)
    reverb_max_dur_ms: float = 500.0   # longest  RIR duration (ms)
    reverb_min_decay:  float = 4.0     # fastest decay (dry room)
    reverb_max_decay:  float = 10.0    # slowest decay (wet room)


@dataclass
class AURAConfig:
    stft: STFTConfig = field(default_factory=STFTConfig)
    conformer: ConformerConfig = field(default_factory=ConformerConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    message: MessageConfig = field(default_factory=MessageConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    multi_res_stft: MultiResSTFTConfig = field(default_factory=MultiResSTFTConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

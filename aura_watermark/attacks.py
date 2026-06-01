"""
Attack Layer — 22 signal-domain attacks for AURA robustness training.

Confirmed from paper Section 2.3 and Table 1.  Double-encoding is a
training-loop technique handled separately in the training loop.

Gradient strategy
─────────────────
All attacks that pass gradients allow the embedder to learn to embed the
watermark in a way that survives that transformation.

  Fully differentiable (grad flows through op):
      noise, pink_noise, lowpass, bandpass, resample, suppress, echo,
      smooth, speed, pitch, speed_pitch, amplitude, boost, duck,
      phase_shift, spaug

  Straight-through estimator (STE):
      mp3, aac, opus, quantize
      Forward  → real codec / rounded value  (no gradient through the op)
      Backward → as if the op were identity  (gradient passes through)
      Formula: y_ste = attacked.detach() + x - x.detach()

Usage
─────
    layer = AttackLayer(cfg.attack, sr=48_000)
    attacked, name = layer(watermarked)          # sample from curriculum
    attacked, name = layer(watermarked, "mp3")   # force specific attack

    # After computing decoder loss on attacked audio:
    layer.curriculum.record(name, loss_value)    # update adaptive probs
"""

import io
import math
import random
import warnings
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchaudio
    import torchaudio.functional as TAF
    _TORCHAUDIO_AVAILABLE = True
except ImportError:
    _TORCHAUDIO_AVAILABLE = False

from .config import AttackConfig


# ── Rational resampling ratios (avoid sinc-kernel OOM) ────────────────────────
# torchaudio.functional.resample builds a sinc filter of size ∝ lcm(orig, new).
# Passing raw floats (e.g. int(48000 * 1.123)) → huge lcm → OOM.
# Solution: keep ratios as small coprime integers; torchaudio internally reduces
# them further via their own GCD.

# speed_pitch (both speed AND pitch change via plain resampling)
# Format: (new_freq, orig_freq) so speed = new/orig
_SPEED_PITCH_RATIOS: List[Tuple[int, int]] = [
    (4,  5),   # 0.800× — (slower & lower)
    (5,  6),   # 0.833×
    (9,  10),  # 0.900×
    (10, 11),  # 0.909×
    (11, 10),  # 1.100×
    (10, 9),   # 1.111×
    (6,  5),   # 1.200×
    (5,  4),   # 1.250×
]

# pitch-only shift: 2^(n/12) approximated as small fractions
# Row: (num, den) so the output is num/den × pitch of input
_PITCH_RATIOS: List[Tuple[int, int]] = [
    (17, 19),  # ≈ 0.895  (−2 semitones)
    (15, 16),  # ≈ 0.938  (−1 semitone)
    (16, 15),  # ≈ 1.067  (+1 semitone)
    (17, 15),  # ≈ 1.133  (+2 semitones)
]


# ── Attack registry ───────────────────────────────────────────────────────────

ATTACK_NAMES: List[str] = [
    "noise",        # White noise at a random SNR
    "pink_noise",   # Pink (1/f) noise scaled to signal RMS
    "lowpass",      # Biquad LP filter, cutoff 3–6 kHz          [paper: LP]
    "highpass",     # Biquad HP filter, cutoff 300–3000 Hz      [paper: HP — Table 1]
    "bandpass",     # Biquad HP+LP chain, 300–400 Hz → 7–9 kHz [paper: BF]
    "mp3",          # MP3 encode/decode (STE)
    "aac",          # AAC encode/decode (STE)
    "opus",         # Opus encode/decode (STE)
    "resample",     # Downsample to {44.1,24,22.05,16} kHz, then back to 48 kHz
    "suppress",     # Zero exactly 0.1% of samples at random positions
    "echo",         # Additive echo: 100 ms delay, 0.3 decay
    "smooth",       # Moving-average filter, window 2–10 samples
    "speed",        # Phase-vocoder time-scale mod (preserves pitch)
    "pitch",        # Resample-based pitch shift, speed-preserved approx
    "speed_pitch",  # Standard resample (changes both speed and pitch)
    "amplitude",    # Multiply by random factor in [−1, 1]
    "boost",        # Multiply by 1.2
    "duck",         # Multiply by 0.8
    "quantize",     # Bit-depth reduction to 4–16 bits (STE)
    "phase_shift",  # Global phase rotation via FFT
    "spaug",        # SpecAugment: random time + freq masking on STFT magnitude
    "reverb",       # Convolutional reverberation with synthetic RIR [paper: RV]
]

N_ATTACKS: int = len(ATTACK_NAMES)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ste(x: torch.Tensor, attacked: torch.Tensor) -> torch.Tensor:
    """
    Straight-through estimator.
    Forward:  returns attacked  (real transformed value)
    Backward: gradient flows through x as if the op were identity

    Use for any non-differentiable attack (codecs, quantization).

    Fast path: when neither tensor requires gradient (eval / torch.no_grad
    context), the arithmetic `attacked + x - x` would equal `attacked`
    mathematically but introduces 1-ULP float32 noise via (a + b) - b ≠ a.
    Returning `attacked` directly avoids this.
    """
    if not x.requires_grad:
        return attacked
    return attacked.detach() + x - x.detach()


def _pad_or_crop(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Ensure last dimension equals target_len by zero-padding or cropping."""
    n = x.shape[-1]
    if n == target_len:
        return x
    if n < target_len:
        return F.pad(x, (0, target_len - n))
    return x[..., :target_len]


def _generate_pink_noise(
    B: int, C: int, T: int, device: torch.device
) -> torch.Tensor:
    """
    Generate unit-RMS pink (1/f) noise of shape [B, C, T].

    Method: random white spectrum scaled by 1/sqrt(f) amplitude envelope
    in the frequency domain, then inverse FFT.  DC component is zeroed.
    """
    n_freqs = T // 2 + 1
    freqs = torch.fft.rfftfreq(T, device=device)   # [n_freqs]
    freqs[0] = 1.0                                  # avoid /0 at DC
    amplitude = 1.0 / freqs.sqrt()                 # 1/sqrt(f) → 1/f power
    amplitude[0] = 0.0                              # zero DC

    # Random complex spectrum [B, C, n_freqs]
    phase = 2.0 * math.pi * torch.rand(B, C, n_freqs, device=device)
    spectrum = amplitude * torch.exp(1j * phase)

    pink = torch.fft.irfft(spectrum, n=T)           # [B, C, T]

    # Normalise to unit RMS
    rms = pink.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
    return pink / rms


def _codec_encode_decode(
    x: torch.Tensor,
    sr: int,
    fmt: str,
    bitrate_kbps: int,
) -> torch.Tensor:
    """
    Encode audio through a lossy codec and decode it back.

    Non-differentiable — caller must wrap with _ste().

    Tries, in order:
      1. torchaudio.functional.apply_codec   (fast, in-memory)
      2. torchaudio.save / torchaudio.load   (io.BytesIO fallback)
      3. Identity                            (graceful degradation)

    Args:
        x:            [B, 1, T]  on any device
        sr:           sample rate
        fmt:          "mp3" | "ogg" | "flac"
        bitrate_kbps: target bitrate
    Returns:
        [B, 1, T]  on same device as x (may be slightly different due to codec)
    """
    T = x.shape[-1]
    device = x.device
    x_cpu = x.detach().cpu()
    results = []

    for b in range(x_cpu.shape[0]):
        wav = x_cpu[b]  # [1, T]
        success = False

        # Method 1: apply_codec
        if _TORCHAUDIO_AVAILABLE:
            try:
                # compression parameter is format-specific:
                # mp3: quality 0-9 (lower = better), we map bitrate roughly
                # ogg/opus: quality -1 to 10
                comp = {
                    "mp3": max(0, min(9, int(9 - bitrate_kbps / 25))),
                    "ogg": max(-1, min(10, int(bitrate_kbps / 16 - 1))),
                    "flac": 8,
                }.get(fmt, None)

                y = TAF.apply_codec(wav, sample_rate=sr, format=fmt, compression=comp)
                y = _pad_or_crop(y, T)
                results.append(y.unsqueeze(0))
                success = True
            except Exception:
                pass

        # Method 2: io-based save/load
        if not success and _TORCHAUDIO_AVAILABLE:
            try:
                buf = io.BytesIO()
                torchaudio.save(buf, wav, sr, format=fmt)
                buf.seek(0)
                y, _ = torchaudio.load(buf, format=fmt)
                y = _pad_or_crop(y, T)
                results.append(y.unsqueeze(0))
                success = True
            except Exception:
                pass

        # Method 3: pseudo-codec fallback (always modifies the signal)
        # Simulates perceptual compression via 8-bit quantisation + 15 kHz
        # low-pass filter.  Not a real codec, but provides useful training
        # signal when ffmpeg / SoX are unavailable (e.g. Windows without extras).
        if not success:
            peak_val = wav.abs().max().clamp(min=1e-8)
            y_norm   = wav / peak_val
            # 8-bit quantisation
            y_q = torch.round(y_norm * 128.0) / 128.0
            # Gentle lowpass at 15 kHz to mimic codec bandwidth limiting
            if _TORCHAUDIO_AVAILABLE:
                try:
                    y_q = TAF.lowpass_biquad(y_q, sr, cutoff_freq=min(15_000.0, sr * 0.4))
                except Exception:
                    pass
            y = _pad_or_crop(y_q * peak_val, T)
            results.append(y.unsqueeze(0))

    return torch.cat(results, dim=0).to(device)


# ── Attack Layer ──────────────────────────────────────────────────────────────

class AttackLayer(nn.Module):
    """
    Applies one of 20 audio attacks to a watermarked waveform.

    The attack is either sampled according to the adaptive curriculum
    probabilities or specified explicitly (for evaluation).

    All attacks preserve the waveform shape [B, 1, T] and sample rate.

    Args:
        cfg: AttackConfig
        sr:  sample rate (default 48_000 Hz)
    """

    def __init__(self, cfg: AttackConfig = AttackConfig(), sr: int = 48_000):
        super().__init__()
        self.cfg = cfg
        self.sr  = sr
        self.curriculum = AdaptiveCurriculum(
            attack_names=ATTACK_NAMES,
            p_min=cfg.p_min,
            window_size=cfg.window_size,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def forward(
        self,
        x:           torch.Tensor,
        attack_name: Optional[str] = None,
    ) -> Tuple[torch.Tensor, str]:
        """
        Apply an attack to the waveform.

        Args:
            x:           [B, 1, T]  watermarked audio
            attack_name: if None, sample from adaptive curriculum

        Returns:
            attacked:    [B, 1, T]  transformed audio
            attack_name: name of the applied attack (for loss logging)
        """
        if attack_name is None:
            attack_name = self.curriculum.sample()

        attacked = self._dispatch(x, attack_name)
        return attacked, attack_name

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def _dispatch(self, x: torch.Tensor, name: str) -> torch.Tensor:
        cfg = self.cfg
        sr  = self.sr

        if name == "noise":
            return self._noise(x)
        elif name == "pink_noise":
            return self._pink_noise(x)
        elif name == "lowpass":
            return self._lowpass(x)
        elif name == "highpass":
            return self._highpass(x)
        elif name == "bandpass":
            return self._bandpass(x)
        elif name == "mp3":
            return self._mp3(x)
        elif name == "aac":
            return self._aac(x)
        elif name == "opus":
            return self._opus(x)
        elif name == "resample":
            return self._resample(x)
        elif name == "suppress":
            return self._suppress(x)
        elif name == "echo":
            return self._echo(x)
        elif name == "smooth":
            return self._smooth(x)
        elif name == "speed":
            return self._speed(x)
        elif name == "pitch":
            return self._pitch(x)
        elif name == "speed_pitch":
            return self._speed_pitch(x)
        elif name == "amplitude":
            return self._amplitude(x)
        elif name == "boost":
            return self._boost(x)
        elif name == "duck":
            return self._duck(x)
        elif name == "quantize":
            return self._quantize(x)
        elif name == "phase_shift":
            return self._phase_shift(x)
        elif name == "spaug":
            return self._spaug(x)
        elif name == "reverb":
            return self._reverb(x)
        else:
            raise ValueError(f"Unknown attack: {name!r}")

    # ── Individual attacks ────────────────────────────────────────────────────

    def _noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Additive white Gaussian noise at a random SNR.
        SNR uniformly sampled from [noise_min_snr_db, noise_max_snr_db].
        Fully differentiable.
        """
        snr_db = random.uniform(self.cfg.noise_min_snr_db, self.cfg.noise_max_snr_db)

        signal_rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        noise      = torch.randn_like(x)
        noise_rms  = noise.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)

        # Scale noise so SNR = 10 log10(signal_rms² / noise_rms²)
        target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
        noise = noise * (target_noise_rms / noise_rms)

        return x + noise

    def _pink_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add pink (1/f) noise scaled to a fraction of the signal RMS.
        Scale uniformly sampled from [pink_min_scale, pink_max_scale].
        Differentiable w.r.t. x (pink noise tensor has no grad).
        """
        B, C, T = x.shape
        scale = random.uniform(self.cfg.pink_min_scale, self.cfg.pink_max_scale)

        pink = _generate_pink_noise(B, C, T, x.device)  # [B, C, T], unit RMS

        # Scale to fraction of signal RMS
        signal_rms = x.detach().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        return x + pink * (signal_rms * scale)

    def _lowpass(self, x: torch.Tensor) -> torch.Tensor:
        """
        Biquad low-pass filter with random cutoff in [lp_min_cutoff_hz, lp_max_cutoff_hz].
        Uses torchaudio.functional.lowpass_biquad — fully differentiable.
        Falls back to identity if torchaudio unavailable.
        Note: biquad requires float32/float64 — cast from AMP float16 if needed.
        """
        if not _TORCHAUDIO_AVAILABLE:
            return x
        cutoff = random.uniform(self.cfg.lp_min_cutoff_hz, self.cfg.lp_max_cutoff_hz)
        B, C, T     = x.shape
        orig_dtype  = x.dtype
        orig_device = x.device
        # biquad requires float32 on CPU (no CUDA support in this torchaudio version)
        x_2d = x.view(B * C, T).float().cpu()
        y = TAF.lowpass_biquad(x_2d, self.sr, cutoff_freq=cutoff)
        return y.to(device=orig_device, dtype=orig_dtype).view(B, C, T)

    def _highpass(self, x: torch.Tensor) -> torch.Tensor:
        """
        Biquad high-pass filter with random cutoff.
        Paper Table 1: HP attack in the Filtering category.
        Cutoff sampled from [hp_min_cutoff_hz, hp_max_cutoff_hz].
        Simulates removal of low-frequency content (rumble, hum).
        Note: biquad requires float32 on CPU (no CUDA support in this torchaudio version).
        """
        if not _TORCHAUDIO_AVAILABLE:
            return x
        cutoff      = random.uniform(self.cfg.hp_min_cutoff_hz, self.cfg.hp_max_cutoff_hz)
        B, C, T     = x.shape
        orig_dtype  = x.dtype
        orig_device = x.device
        x_2d = x.view(B * C, T).float().cpu()
        y = TAF.highpass_biquad(x_2d, self.sr, cutoff_freq=cutoff)
        return y.to(device=orig_device, dtype=orig_dtype).view(B, C, T)

    def _bandpass(self, x: torch.Tensor) -> torch.Tensor:
        """
        Band-pass by chaining highpass(low_cutoff) + lowpass(high_cutoff).
        Cutoffs sampled from paper-specified ranges.
        Fully differentiable.
        Note: biquad requires float32/float64 — cast from AMP float16 if needed.
        """
        if not _TORCHAUDIO_AVAILABLE:
            return x
        low  = random.uniform(self.cfg.bp_low_min_hz,  self.cfg.bp_low_max_hz)
        high = random.uniform(self.cfg.bp_high_min_hz, self.cfg.bp_high_max_hz)
        B, C, T     = x.shape
        orig_dtype  = x.dtype
        orig_device = x.device
        # biquad requires float32 on CPU (no CUDA support in this torchaudio version)
        x_2d = x.view(B * C, T).float().cpu()
        y = TAF.highpass_biquad(x_2d, self.sr, cutoff_freq=low)
        y = TAF.lowpass_biquad(y,     self.sr, cutoff_freq=high)
        return y.to(device=orig_device, dtype=orig_dtype).view(B, C, T)

    def _mp3(self, x: torch.Tensor) -> torch.Tensor:
        """MP3 codec. STE: output is codec output, gradient is identity."""
        bitrate = random.choice(self.cfg.mp3_bitrates)
        attacked = _codec_encode_decode(x, self.sr, fmt="mp3", bitrate_kbps=bitrate)
        return _ste(x, attacked)

    def _aac(self, x: torch.Tensor) -> torch.Tensor:
        """
        AAC-like codec attack. STE.

        True AAC requires ffmpeg. We attempt OGG/Vorbis first (perceptually
        closer to AAC than MP3 — both use psychoacoustic masking with MDCT),
        then fall back to MP3 if unavailable. _codec_encode_decode handles
        the fallback chain internally.
        """
        bitrate = random.choice(self.cfg.aac_bitrates)
        attacked = _codec_encode_decode(x, self.sr, fmt="ogg", bitrate_kbps=bitrate)
        return _ste(x, attacked)

    def _opus(self, x: torch.Tensor) -> torch.Tensor:
        """Opus codec. STE."""
        bitrate = random.choice(self.cfg.opus_bitrates)
        attacked = _codec_encode_decode(x, self.sr, fmt="ogg", bitrate_kbps=bitrate)
        return _ste(x, attacked)

    def _resample(self, x: torch.Tensor) -> torch.Tensor:
        """
        Downsample to a random target rate, then upsample back to original rate.
        Target rates: {44100, 24000, 22050, 16000} Hz.
        Uses torchaudio.functional.resample (sinc interpolation) — differentiable.
        """
        if not _TORCHAUDIO_AVAILABLE:
            return x
        target_sr  = random.choice(self.cfg.resample_rates)
        orig_dtype = x.dtype
        T = x.shape[-1]
        y = TAF.resample(x.float(), orig_freq=self.sr, new_freq=target_sr)
        y = TAF.resample(y,         orig_freq=target_sr, new_freq=self.sr)
        return _pad_or_crop(y, T).to(orig_dtype)

    def _suppress(self, x: torch.Tensor) -> torch.Tensor:
        """
        Zero out exactly 0.1% of samples at random positions.
        Differentiable (multiplication by binary mask — gradient passes through
        un-masked positions).
        """
        B, C, T = x.shape
        n_zero = max(1, int(T * self.cfg.suppress_fraction))

        mask = torch.ones(B, 1, T, device=x.device)
        for b in range(B):
            idx = torch.randperm(T, device=x.device)[:n_zero]
            mask[b, 0, idx] = 0.0

        return x * mask

    def _echo(self, x: torch.Tensor) -> torch.Tensor:
        """
        Additive echo: y[t] = x[t] + decay * x[t - delay_samples].
        delay = 100 ms = 4800 samples at 48 kHz.  decay = 0.3.
        Fully differentiable.
        """
        delay   = int(self.cfg.echo_delay_ms / 1000.0 * self.sr)   # 4800
        decay   = self.cfg.echo_decay
        T       = x.shape[-1]

        # Pad start with `delay` zeros, then take first T samples → delayed x
        delayed = F.pad(x, (delay, 0))[..., :T]
        return x + decay * delayed

    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        """
        Moving-average filter with random window size in [smooth_min, smooth_max].
        Implemented as depthwise 1D conv with a uniform kernel.
        Fully differentiable.
        """
        B, C, T = x.shape
        win = random.randint(self.cfg.smooth_min_window, self.cfg.smooth_max_window)
        kernel  = torch.ones(C, 1, win, device=x.device) / win
        padding = win // 2
        y = F.conv1d(x, kernel, padding=padding, groups=C)
        return _pad_or_crop(y, T)

    def _speed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Time-scale modification (change speed, preserve pitch) using the
        phase vocoder.  Rate sampled from [speed_min, speed_max].
        Differentiable via STFT / phase_vocoder / ISTFT.
        Falls back to speed_pitch if torchaudio unavailable.
        """
        rate = random.uniform(self.cfg.speed_min, self.cfg.speed_max)

        if not _TORCHAUDIO_AVAILABLE:
            return self._speed_pitch(x)   # approximate fallback

        B, C, T = x.shape
        n_fft   = 512
        hop     = 128
        device  = x.device
        window  = torch.hann_window(n_fft, device=device)

        x_2d = x.view(B * C, T)

        # STFT → complex spectrogram [B*C, F, T_spec]
        spec = torch.stft(
            x_2d, n_fft=n_fft, hop_length=hop, win_length=n_fft,
            window=window, center=True, pad_mode="reflect",
            return_complex=True,
        )

        n_freqs = spec.shape[1]
        phase_advance = torch.linspace(
            0, math.pi * hop, n_freqs, device=device
        ).unsqueeze(1)  # [F, 1]

        # Phase-vocoder time stretching: rate>1 → faster (fewer frames)
        stretched = TAF.phase_vocoder(spec, rate=rate, phase_advance=phase_advance)

        # ISTFT → waveform
        # When rate > 1 the stretched spectrogram has fewer frames; istft
        # pads to length=T, which triggers a benign PyTorch warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            y = torch.istft(
                stretched, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                window=window, length=T, center=True,
            )
        return y.view(B, C, T)

    def _pitch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Approximate pitch shift (speed-preserved) via single resampling
        with a small rational ratio, followed by pad/crop to restore length.

        Uses _PITCH_RATIOS to avoid sinc-kernel OOM: passing raw floats
        (int(48000 * rate)) can create kernel sizes proportional to lcm(orig,new)
        which is enormous for non-dyadic floats.

        Fully differentiable via torchaudio.functional.resample.
        """
        if not _TORCHAUDIO_AVAILABLE:
            return x
        num, den   = random.choice(_PITCH_RATIOS)
        orig_dtype = x.dtype
        T          = x.shape[-1]
        y = TAF.resample(x.float(), orig_freq=den, new_freq=num)
        return _pad_or_crop(y, T).to(orig_dtype)

    def _speed_pitch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard resampling: changes both speed and pitch.
        Uses _SPEED_PITCH_RATIOS (small coprime ints) to avoid sinc-kernel OOM.
        Differentiable via torchaudio.functional.resample.
        """
        if not _TORCHAUDIO_AVAILABLE:
            return x
        num, den   = random.choice(_SPEED_PITCH_RATIOS)
        orig_dtype = x.dtype
        T          = x.shape[-1]
        y = TAF.resample(x.float(), orig_freq=den, new_freq=num)
        return _pad_or_crop(y, T).to(orig_dtype)

    def _amplitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multiply by a random scale factor in (−1, 1).
        Includes phase inversion when scale < 0.
        Fully differentiable.
        """
        # Avoid exactly 0 (would zero the signal completely)
        scale = random.uniform(-1.0, -0.01) if random.random() < 0.5 else random.uniform(0.01, 1.0)
        return x * scale

    def _boost(self, x: torch.Tensor) -> torch.Tensor:
        """Fixed +20% amplitude boost. Fully differentiable."""
        return x * 1.2

    def _duck(self, x: torch.Tensor) -> torch.Tensor:
        """Fixed −20% amplitude attenuation. Fully differentiable."""
        return x * 0.8

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Bit-depth reduction to a random number of bits in [quantize_min, quantize_max].
        Uses STE: forward = quantised value, backward = identity.

        Procedure:
          1. Normalise to [−1, 1] per sample using peak amplitude
          2. Map to signed integer in [−n_levels//2, n_levels//2 − 1]
             (clamp ensures exactly n_levels distinct values, not n_levels+1)
          3. Map back to [−1, 1] float
          4. De-normalise by peak
          5. Wrap with STE so gradients flow through

        For 4-bit: 16 unique output values {−1.0, −7/8, ..., 7/8} (not 17,
        because we exclude +1.0 the same way two's-complement signed integers
        exclude +n_levels//2).
        """
        n_bits   = random.randint(self.cfg.quantize_min_bits, self.cfg.quantize_max_bits)
        n_levels = 2 ** n_bits                      # total levels, e.g. 16 for 4-bit
        half     = n_levels // 2                    # e.g. 8

        peak   = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        x_norm = x / peak                           # ≈ [−1, 1]

        # Integer quantisation with clamping → exactly n_levels unique values
        x_int = torch.round(x_norm * half).clamp(-half, half - 1)
        x_q   = (x_int / half) * peak              # de-normalise

        return _ste(x, x_q)

    def _phase_shift(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a random global phase rotation via FFT.
        phase uniformly sampled from (0, 2π) — excludes 0 to guarantee modification.

        y = IFFT(FFT(x) * exp(i * phase))

        The phase rotation preserves the magnitude spectrum exactly in theory.
        In float32 there is O(1e-6) relative rounding error on large magnitudes,
        but the signal content is modified as intended.

        Fully differentiable.
        """
        # Sample a non-trivial phase (exclude near-0 and near-2pi)
        phase = random.uniform(0.1, 2.0 * math.pi - 0.1)
        X     = torch.fft.rfft(x, dim=-1)                            # complex64
        # Multiply by exp(i*phase) = cos + i*sin
        c = math.cos(phase)
        s = math.sin(phase)
        # Real part: Re(X)*c - Im(X)*s
        # Imag part: Re(X)*s + Im(X)*c
        X_r = X.real * c - X.imag * s
        X_i = X.real * s + X.imag * c
        X_shifted = torch.complex(X_r, X_i)
        return torch.fft.irfft(X_shifted, n=x.shape[-1], dim=-1)

    def _spaug(self, x: torch.Tensor) -> torch.Tensor:
        """
        SpecAugment — random time and frequency masking in the STFT domain.

        Procedure:
          1. STFT with the same n_fft/hop as the main pipeline
          2. Zero out spaug_num_time_masks random contiguous time blocks
          3. Zero out spaug_num_freq_masks random contiguous freq blocks
          4. ISTFT → waveform

        The mask is applied to the complex spectrum (zeros both magnitude
        and phase), which is the standard SpecAugment behaviour.
        Fully differentiable.
        """
        cfg = self.cfg
        B, C, T = x.shape
        device  = x.device

        n_fft  = 2048
        hop    = 512
        window = torch.hann_window(n_fft, device=device)

        x_2d = x.view(B * C, T)
        spec = torch.stft(
            x_2d, n_fft=n_fft, hop_length=hop, win_length=n_fft,
            window=window, center=True, pad_mode="reflect",
            return_complex=True,
        )  # [B*C, F, T_spec]

        F_bins, T_spec = spec.shape[1], spec.shape[2]

        # Build mask (ones everywhere, zero in masked regions)
        mask = torch.ones(B * C, F_bins, T_spec, device=device)

        for b in range(B * C):
            # Time masks
            for _ in range(cfg.spaug_num_time_masks):
                t_len   = random.randint(1, cfg.spaug_max_time_mask)
                t_start = random.randint(0, max(0, T_spec - t_len))
                mask[b, :, t_start : t_start + t_len] = 0.0

            # Freq masks
            for _ in range(cfg.spaug_num_freq_masks):
                f_len   = random.randint(1, cfg.spaug_max_freq_mask)
                f_start = random.randint(0, max(0, F_bins - f_len))
                mask[b, f_start : f_start + f_len, :] = 0.0

        spec = spec * mask

        y = torch.istft(
            spec, n_fft=n_fft, hop_length=hop, win_length=n_fft,
            window=window, length=T, center=True,
        )
        return y.view(B, C, T)


    def _reverb(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convolutional reverberation with a synthetic Room Impulse Response (RIR).

        Paper Section 2.3: "Reverb(RV): convolutional reverberation with a given RIR"

        Method:
          Generates a synthetic RIR by multiplying unit-variance white noise with
          an exponential decay envelope, simulating a small-to-medium sized room.
          The RIR is then convolved with the audio via FFT for efficiency.

          Parameters sampled per call:
            rir_dur  ~ Uniform[reverb_min_dur_ms, reverb_max_dur_ms]  (RIR length)
            decay    ~ Uniform[reverb_min_decay,  reverb_max_decay]   (room damping)

          Low decay  (4.0)  → short RT60 → dry room (office, corridor)
          High decay (10.0) → long  RT60 → wet room (church, hall)

        Fully differentiable (gradient flows through FFT convolution).
        """
        cfg = self.cfg
        B, C, T = x.shape
        device  = x.device
        dtype   = x.dtype

        # ── Synthetic RIR ────────────────────────────────────────────────────
        dur_ms  = random.uniform(cfg.reverb_min_dur_ms, cfg.reverb_max_dur_ms)
        decay   = random.uniform(cfg.reverb_min_decay,  cfg.reverb_max_decay)
        rir_len = max(1, int(dur_ms / 1000.0 * self.sr))

        # Exponentially-decaying white noise
        t   = torch.arange(rir_len, device=device, dtype=dtype)
        rir = torch.randn(rir_len, device=device, dtype=dtype)
        rir = rir * torch.exp(-decay * t / self.sr)

        # Normalise so peak = 1 (preserves signal loudness after convolution)
        rir = rir / (rir.abs().max().clamp(min=1e-8))

        # ── FFT convolution ──────────────────────────────────────────────────
        # Next power of 2 ≥ T + rir_len − 1 for linear (non-circular) convolution
        n_fft = 1
        while n_fft < T + rir_len - 1:
            n_fft <<= 1

        # [B, C, T] → [B, C, n_fft//2+1] complex
        X = torch.fft.rfft(x,   n=n_fft, dim=-1)
        # [rir_len] → [1, 1, n_fft//2+1] complex (broadcast over B, C)
        R = torch.fft.rfft(rir, n=n_fft, dim=-1).unsqueeze(0).unsqueeze(0)

        y = torch.fft.irfft(X * R, n=n_fft, dim=-1)[..., :T]

        # Clip to [-1, 1] to avoid trainer instability from occasional
        # large values when decay is slow and RIR is long
        return y.clamp(-1.0, 1.0)


# ── Adaptive Curriculum ───────────────────────────────────────────────────────

class AdaptiveCurriculum:
    """
    Tracks a rolling per-attack message-decoding loss and computes adaptive
    sampling probabilities.

    Paper formula (Section 2.3):
        L_k_bar = rolling average of BCE loss for attack k
        P_k_raw = max(L_k_bar / sum_k(L_k_bar),  P_min)
        P_k_new = P_k_raw / sum_k(P_k_raw)      (renormalise to sum=1)

    Interpretation:
        - Attacks the model struggles with → higher L_k_bar → sampled more
        - P_min ensures no attack is ever completely dropped from training
        - Probabilities are recomputed after every call to record()

    Args:
        attack_names: list of attack names (must match ATTACK_NAMES)
        p_min:        minimum probability floor (default 0.01)
        window_size:  rolling window length in batches (default 1000)
    """

    def __init__(
        self,
        attack_names: List[str] = ATTACK_NAMES,
        p_min:        float = 0.01,
        window_size:  int   = 1_000,
    ):
        self.attack_names = list(attack_names)
        self.n_attacks    = len(attack_names)
        self.p_min        = p_min
        self.window_size  = window_size

        # Rolling loss buffers (deque auto-trims old entries)
        self._buffers: Dict[str, deque] = {
            name: deque(maxlen=window_size) for name in attack_names
        }

        # Uniform initialisation — all attacks equally likely at the start
        uniform = 1.0 / self.n_attacks
        self._probs: Dict[str, float] = {name: uniform for name in attack_names}

    # ── Public API ────────────────────────────────────────────────────────────

    def record(self, attack_name: str, loss_value: float) -> None:
        """
        Record a new loss observation for an attack and update probabilities.

        Call this after computing the message decoding loss on the attacked
        audio for each training step.

        Args:
            attack_name: one of ATTACK_NAMES
            loss_value:  scalar BCE loss for this batch under this attack
        """
        if attack_name not in self._buffers:
            raise ValueError(f"Unknown attack: {attack_name!r}")
        self._buffers[attack_name].append(float(loss_value))
        self._recompute()

    def sample(self) -> str:
        """
        Sample an attack name according to current adaptive probabilities.
        Returns:
            attack_name: one of ATTACK_NAMES
        """
        names  = list(self._probs.keys())
        weights = [self._probs[n] for n in names]
        return random.choices(names, weights=weights, k=1)[0]

    def probabilities(self) -> Dict[str, float]:
        """Return a copy of the current sampling probabilities."""
        return dict(self._probs)

    def state_dict(self) -> dict:
        """Serialise curriculum state for checkpointing."""
        return {
            "buffers": {k: list(v) for k, v in self._buffers.items()},
            "probs":   dict(self._probs),
            "p_min":   self.p_min,
            "window_size": self.window_size,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore curriculum state from a checkpoint."""
        self.p_min        = state["p_min"]
        self.window_size  = state["window_size"]
        self._buffers = {
            k: deque(v, maxlen=self.window_size)
            for k, v in state["buffers"].items()
        }
        self._probs = dict(state["probs"])

    # ── Internal ──────────────────────────────────────────────────────────────

    def _recompute(self) -> None:
        """Recompute probabilities from current rolling averages."""
        # Rolling average per attack (default 1.0 if no data yet)
        l_bar: Dict[str, float] = {}
        for name in self.attack_names:
            buf = self._buffers[name]
            l_bar[name] = (sum(buf) / len(buf)) if buf else 1.0

        total = sum(l_bar.values())
        if total < 1e-8:
            total = 1e-8

        # Apply P_min floor
        raw = {
            name: max(l_bar[name] / total, self.p_min)
            for name in self.attack_names
        }

        # Renormalise to sum=1
        total_raw = sum(raw.values())
        self._probs = {
            name: raw[name] / total_raw
            for name in self.attack_names
        }

    def __repr__(self) -> str:
        top = sorted(self._probs.items(), key=lambda kv: -kv[1])[:5]
        top_str = ", ".join(f"{k}:{v:.3f}" for k, v in top)
        return f"AdaptiveCurriculum(top-5: {top_str})"

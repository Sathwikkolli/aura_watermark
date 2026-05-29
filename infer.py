#!/usr/bin/env python3
"""
AURA Inference — Step 10.

CLI for watermark embedding, detection, and batch evaluation using a
trained AURA model checkpoint.

Subcommands
-----------
embed   Embed a 32-bit watermark into one or more audio files.
detect  Detect and decode the watermark from audio files.
eval    Evaluate BER and SNR on a batch of audio files or a synthetic set.

Usage examples
--------------
  # Embed watermark (random bits)
  python infer.py embed \\
      --checkpoint checkpoints/step_0200000_final.pt \\
      --input  audio/track.wav \\
      --output audio/track_watermarked.wav

  # Embed with explicit bits  (32-char string of 0s and 1s)
  python infer.py embed \\
      --checkpoint checkpoints/step_0200000_final.pt \\
      --input  audio/track.wav \\
      --output audio/track_watermarked.wav \\
      --bits   "10110101001011010100101101010010"

  # Detect / decode watermark
  python infer.py detect \\
      --checkpoint checkpoints/step_0200000_final.pt \\
      --input  audio/track_watermarked.wav

  # Batch evaluate on a folder
  python infer.py eval \\
      --checkpoint checkpoints/step_0200000_final.pt \\
      --input-dir  audio/test_set/ \\
      --attacks    noise mp3 lowpass \\
      --n-files    100

  # Evaluate on synthetic data (no audio files needed)
  python infer.py eval \\
      --checkpoint checkpoints/step_0200000_final.pt \\
      --synthetic  --n-files 64
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
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
from aura_watermark.attacks import AttackLayer, ATTACK_NAMES

log = logging.getLogger("aura.infer")


# ═════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═════════════════════════════════════════════════════════════════════════════

def load_audio(
    path: str | Path,
    target_sr: int = 48_000,
    n_samples: int = 96_000,
) -> torch.Tensor:
    """
    Load an audio file, resample to ``target_sr``, convert to mono, crop/pad
    to exactly ``n_samples`` samples, and peak-normalise.

    Args:
        path:      Path to audio file (.wav, .flac, .mp3, …)
        target_sr: Target sample rate (default: 48 000 Hz)
        n_samples: Number of output samples (default: 96 000 = 2 s at 48 kHz)

    Returns:
        Tensor of shape [1, n_samples] on CPU.

    Raises:
        RuntimeError: if torchaudio cannot load the file.
    """
    try:
        import torchaudio
    except ImportError as e:
        raise ImportError("torchaudio is required for audio I/O: pip install torchaudio") from e

    waveform, sr = torchaudio.load(str(path))   # [C, T]

    # ── Mono ────────────────────────────────────────────────────────────────
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)   # [1, T]

    # ── Resample ────────────────────────────────────────────────────────────
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform  = resampler(waveform)

    # ── Crop / pad to exactly n_samples ────────────────────────────────────
    T = waveform.shape[-1]
    if T >= n_samples:
        start    = random.randint(0, T - n_samples)
        waveform = waveform[:, start : start + n_samples]
    else:
        reps     = math.ceil(n_samples / T)
        waveform = waveform.repeat(1, reps)[:, :n_samples]

    # ── Peak-normalise ──────────────────────────────────────────────────────
    peak = waveform.abs().max()
    if peak > 1e-6:
        waveform = waveform / peak

    return waveform   # [1, n_samples]


def save_audio(
    waveform: torch.Tensor,
    path: str | Path,
    sample_rate: int = 48_000,
) -> None:
    """
    Save a mono waveform tensor to an audio file.

    Args:
        waveform:    [1, T] or [T] tensor (CPU)
        path:        Output file path (extension determines format)
        sample_rate: Sample rate to write into the file header
    """
    try:
        import torchaudio
    except ImportError as e:
        raise ImportError("torchaudio is required for audio I/O: pip install torchaudio") from e

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    waveform = waveform.cpu().clamp(-1.0, 1.0)
    torchaudio.save(str(path), waveform, sample_rate)


def bits_to_str(bits: torch.Tensor) -> str:
    """Convert a [n_bits] binary tensor to a string of '0'/'1' characters."""
    return "".join(str(int(b)) for b in bits.flatten().tolist())


def str_to_bits(s: str, n_bits: int = 32) -> torch.Tensor:
    """
    Parse a string of '0'/'1' characters into a [n_bits] int64 tensor.

    Args:
        s:      Bit string, e.g. "10110101…" (must be exactly n_bits chars)
        n_bits: Expected length

    Returns:
        [n_bits] int64 tensor with values in {0, 1}

    Raises:
        ValueError: if the string length or characters are invalid
    """
    if len(s) != n_bits:
        raise ValueError(
            f"Bit string length {len(s)} != n_bits={n_bits}. "
            f"Provide exactly {n_bits} '0'/'1' characters."
        )
    if not all(c in "01" for c in s):
        raise ValueError("Bit string must contain only '0' and '1' characters.")
    return torch.tensor([int(c) for c in s], dtype=torch.long)


def compute_ber(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Bit Error Rate = fraction of mismatched bits (float in [0, 1])."""
    pred   = (predicted > 0).long().flatten()
    tgt    = target.long().flatten()
    return (pred != tgt).float().mean().item()


def compute_snr(original: torch.Tensor, watermarked: torch.Tensor) -> float:
    """Watermark SNR in dB (higher = less audible distortion)."""
    noise        = watermarked - original
    sig_power    = original.pow(2).mean().clamp(min=1e-10)
    noise_power  = noise.pow(2).mean().clamp(min=1e-10)
    return 10.0 * math.log10(sig_power.item() / noise_power.item())


# ═════════════════════════════════════════════════════════════════════════════
# Model loading
# ═════════════════════════════════════════════════════════════════════════════

def load_model(
    checkpoint: str | Path,
    device: torch.device,
    cfg: Optional[AURAConfig] = None,
) -> Tuple[StegaformerEmbedder, AURADecoder, AURAConfig]:
    """
    Load embedder and detector from a training checkpoint.

    Args:
        checkpoint: Path to ``.pt`` file saved by ``AURATrainer.save_checkpoint()``.
        device:     Target device for the models.
        cfg:        AURAConfig to use (if ``None``, a default config is created).

    Returns:
        (embedder, detector, cfg) — both models in eval mode on ``device``.

    Raises:
        FileNotFoundError: if the checkpoint path does not exist.
        KeyError:          if the checkpoint is missing required keys.
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if cfg is None:
        cfg = AURAConfig()

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    embedder = StegaformerEmbedder(cfg).to(device)
    detector = AURADecoder(cfg).to(device)

    embedder.load_state_dict(ckpt["gen_state"])
    detector.load_state_dict(ckpt["det_state"])

    embedder.eval()
    detector.eval()

    return embedder, detector, cfg


# ═════════════════════════════════════════════════════════════════════════════
# Core inference functions
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def embed_watermark(
    embedder:  StegaformerEmbedder,
    audio:     torch.Tensor,   # [1, T]  or  [B, 1, T]
    message:   torch.Tensor,   # [n_bits] or [B, n_bits]
) -> torch.Tensor:
    """
    Embed a watermark message into audio.

    Args:
        embedder: Trained StegaformerEmbedder (eval mode).
        audio:    [1, T] single clip or [B, 1, T] batch.
        message:  [n_bits] or [B, n_bits] binary tensor {0, 1}.

    Returns:
        Watermarked audio, same shape as input ``audio``.
    """
    if audio.dim() == 2:           # [1, T] → [1, 1, T]
        audio   = audio.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    if message.dim() == 1:         # [n_bits] → [1, n_bits]
        message = message.unsqueeze(0).expand(audio.shape[0], -1)

    x_wm, _, _ = embedder(audio, message.float())

    if squeeze:
        x_wm = x_wm.squeeze(0)    # [1, T]

    return x_wm


@torch.no_grad()
def detect_watermark(
    embedder: StegaformerEmbedder,
    detector: AURADecoder,
    audio:    torch.Tensor,   # [1, T] or [B, 1, T]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Detect and decode the watermark from audio.

    Args:
        embedder: Trained StegaformerEmbedder (used for STFT only).
        detector: Trained AURADecoder (eval mode).
        audio:    [1, T] single clip or [B, 1, T] batch.

    Returns:
        (logits, bits):
            logits — [B, n_bits]  raw detector outputs (before threshold)
            bits   — [B, n_bits]  decoded binary bits {0, 1} (logits > 0)
    """
    if audio.dim() == 2:           # [1, T] → [1, 1, T]
        audio   = audio.unsqueeze(0)

    mag, _ = embedder.stft(audio)      # [B, 1025, T_stft]
    s_mag  = mag.unsqueeze(1)          # [B, 1, 1025, T_stft]
    logits = detector(s_mag)           # [B, n_bits]
    bits   = (logits > 0).long()       # [B, n_bits]

    return logits, bits


# ═════════════════════════════════════════════════════════════════════════════
# Subcommand implementations
# ═════════════════════════════════════════════════════════════════════════════

def cmd_embed(args: argparse.Namespace) -> None:
    """
    Embed a watermark into one or more audio files.

    For each input file, saves a watermarked copy to the output path.
    Prints the embedded bit string and achieved SNR to stdout.
    """
    device = _resolve_device(args.device)
    log.info("Loading model from %s …", args.checkpoint)
    embedder, detector, cfg = load_model(args.checkpoint, device)

    # ── Resolve bit message ─────────────────────────────────────────────────
    n_bits = cfg.message.n_bits
    if args.bits:
        message = str_to_bits(args.bits, n_bits)
    else:
        message = torch.randint(0, 2, (n_bits,), dtype=torch.long)
        log.info("No --bits supplied — using random message: %s", bits_to_str(message))

    # ── Process files ───────────────────────────────────────────────────────
    input_paths  = _resolve_inputs(args.input)
    output_paths = _resolve_outputs(args.output, input_paths)

    for in_path, out_path in zip(input_paths, output_paths):
        log.info("Embedding into %s …", in_path)
        audio = load_audio(in_path, target_sr=cfg.stft.sample_rate,
                           n_samples=cfg.stft.segment_samples).to(device)

        x_wm = embed_watermark(embedder, audio, message.to(device))

        snr = compute_snr(audio, x_wm)
        log.info("  SNR = %.2f dB  →  saving to %s", snr, out_path)

        save_audio(x_wm, out_path, sample_rate=cfg.stft.sample_rate)

    print(json.dumps({
        "bits":    bits_to_str(message),
        "n_files": len(input_paths),
        "status":  "ok",
    }))


def cmd_detect(args: argparse.Namespace) -> None:
    """
    Detect and decode the watermark from one or more audio files.

    For each input file, prints the decoded bit string and confidence
    (fraction of bits above decision threshold) to stdout as JSON.
    """
    device = _resolve_device(args.device)
    log.info("Loading model from %s …", args.checkpoint)
    embedder, detector, cfg = load_model(args.checkpoint, device)

    input_paths = _resolve_inputs(args.input)
    results = []

    for in_path in input_paths:
        log.info("Detecting from %s …", in_path)
        audio = load_audio(in_path, target_sr=cfg.stft.sample_rate,
                           n_samples=cfg.stft.segment_samples).to(device)

        logits, bits = detect_watermark(embedder, detector, audio)

        bit_str    = bits_to_str(bits.squeeze(0))
        confidence = (logits.abs() > args.threshold).float().mean().item()

        log.info("  bits=%s  confidence=%.3f", bit_str, confidence)
        results.append({
            "file":       str(in_path),
            "bits":       bit_str,
            "confidence": round(confidence, 4),
        })

    print(json.dumps(results if len(results) > 1 else results[0], indent=2))


def cmd_eval(args: argparse.Namespace) -> None:
    """
    Evaluate BER and SNR on a batch of audio clips.

    Reports per-attack and aggregate BER (Bit Error Rate), plus mean
    watermark SNR.  Output is printed as JSON.
    """
    device = _resolve_device(args.device)
    log.info("Loading model from %s …", args.checkpoint)
    embedder, detector, cfg = load_model(args.checkpoint, device)

    attack_layer = AttackLayer(cfg.attack, sr=cfg.stft.sample_rate)

    # ── Select attacks to evaluate ──────────────────────────────────────────
    attacks_to_eval = args.attacks if args.attacks else ATTACK_NAMES

    # ── Collect audio clips ─────────────────────────────────────────────────
    if args.synthetic:
        clips   = _make_synthetic_clips(args.n_files, cfg)
        log.info("Evaluating on %d synthetic clips.", len(clips))
    else:
        paths   = _scan_audio(args.input_dir)[:args.n_files]
        clips   = []
        for p in paths:
            try:
                clips.append(load_audio(
                    p,
                    target_sr  = cfg.stft.sample_rate,
                    n_samples  = cfg.stft.segment_samples,
                ))
            except Exception as e:
                log.warning("Skipping %s: %s", p, e)
        log.info("Evaluating on %d audio clips.", len(clips))

    if not clips:
        log.error("No audio clips to evaluate.")
        sys.exit(1)

    # ── Run evaluation ──────────────────────────────────────────────────────
    n_bits  = cfg.message.n_bits
    per_attack_ber: Dict[str, List[float]] = {a: [] for a in attacks_to_eval}
    snr_list: List[float] = []

    for audio in clips:
        audio   = audio.to(device)
        message = torch.randint(0, 2, (n_bits,), dtype=torch.long, device=device)

        x_wm = embed_watermark(embedder, audio, message)
        snr_list.append(compute_snr(audio, x_wm))

        for attack_name in attacks_to_eval:
            try:
                x_att, _ = attack_layer(x_wm.unsqueeze(0), attack_name=attack_name)
                x_att    = x_att.squeeze(0)
            except Exception:
                continue

            logits, _ = detect_watermark(embedder, detector, x_att)
            ber = compute_ber(logits.squeeze(0), message)
            per_attack_ber[attack_name].append(ber)

    # ── Aggregate ───────────────────────────────────────────────────────────
    attack_results: Dict[str, float] = {}
    all_bers: List[float] = []

    for name, bers in per_attack_ber.items():
        if bers:
            m = sum(bers) / len(bers)
            attack_results[f"ber/{name}"] = round(m, 4)
            all_bers.extend(bers)

    mean_ber    = sum(all_bers) / len(all_bers) if all_bers else 0.0
    mean_bit_acc = 1.0 - mean_ber
    mean_snr    = sum(snr_list) / len(snr_list) if snr_list else 0.0

    summary = {
        "n_clips":    len(clips),
        "n_attacks":  len([a for a in attacks_to_eval if per_attack_ber[a]]),
        "ber":        round(mean_ber, 4),
        "bit_acc":    round(mean_bit_acc, 4),
        "snr_db":     round(mean_snr, 2),
        **attack_results,
    }

    print(json.dumps(summary, indent=2))
    log.info(
        "Eval complete — BER=%.4f  bit_acc=%.4f  SNR=%.2f dB",
        mean_ber, mean_bit_acc, mean_snr,
    )


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AURA watermark inference and evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = p.add_subparsers(dest="command", required=True)

    # ── embed ──────────────────────────────────────────────────────────────
    ep = sub.add_parser("embed", help="Embed watermark into audio file(s).")
    ep.add_argument("--checkpoint", type=str, required=True,
                    help="Path to trained .pt checkpoint.")
    ep.add_argument("--input",  type=str, nargs="+", required=True,
                    help="Input audio file(s).")
    ep.add_argument("--output", type=str, nargs="+", default=None,
                    help="Output path(s). Default: input_watermarked.<ext>.")
    ep.add_argument("--bits",   type=str, default=None,
                    help="32-char bit string (e.g. '10110101…'). "
                         "Random if omitted.")
    ep.add_argument("--device", type=str, default="auto")

    # ── detect ─────────────────────────────────────────────────────────────
    dp = sub.add_parser("detect", help="Decode watermark from audio file(s).")
    dp.add_argument("--checkpoint", type=str, required=True,
                    help="Path to trained .pt checkpoint.")
    dp.add_argument("--input",     type=str, nargs="+", required=True,
                    help="Input audio file(s).")
    dp.add_argument("--threshold", type=float, default=0.5,
                    help="Logit magnitude threshold for confidence reporting.")
    dp.add_argument("--device",    type=str,   default="auto")

    # ── eval ───────────────────────────────────────────────────────────────
    vp = sub.add_parser("eval", help="Evaluate BER and SNR on audio clips.")
    vp.add_argument("--checkpoint", type=str, required=True,
                    help="Path to trained .pt checkpoint.")
    vp.add_argument("--input-dir",  type=str, default=None,
                    help="Directory of audio files (required unless --synthetic).")
    vp.add_argument("--synthetic",  action="store_true",
                    help="Use randomly generated clips (no audio files needed).")
    vp.add_argument("--n-files",   type=int, default=100,
                    help="Max number of clips to evaluate.")
    vp.add_argument("--attacks",   type=str, nargs="*", default=None,
                    choices=ATTACK_NAMES,
                    help="Attacks to evaluate (default: all 20).")
    vp.add_argument("--device",    type=str, default="auto")

    return p.parse_args(argv)


# ═════════════════════════════════════════════════════════════════════════════
# Utilities
# ═════════════════════════════════════════════════════════════════════════════

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _resolve_inputs(inputs: List[str]) -> List[Path]:
    paths = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.wav")) + sorted(p.rglob("*.flac")))
        else:
            paths.append(p)
    return paths


def _resolve_outputs(
    outputs: Optional[List[str]],
    inputs:  List[Path],
) -> List[Path]:
    if outputs is not None:
        return [Path(o) for o in outputs]
    # Default: insert "_watermarked" before the suffix
    result = []
    for p in inputs:
        out = p.parent / f"{p.stem}_watermarked{p.suffix}"
        result.append(out)
    return result


def _scan_audio(directory: str | Path) -> List[Path]:
    d = Path(directory)
    files = sorted(d.rglob("*.wav")) + sorted(d.rglob("*.flac")) + sorted(d.rglob("*.mp3"))
    return files


def _make_synthetic_clips(
    n: int,
    cfg: AURAConfig,
) -> List[torch.Tensor]:
    """Generate ``n`` random synthetic audio clips [1, n_samples]."""
    n_samples = cfg.stft.segment_samples
    clips = []
    for _ in range(n):
        audio = torch.randn(1, n_samples) * 0.3
        peak  = audio.abs().max()
        if peak > 1e-6:
            audio = audio / peak
        clips.append(audio)
    return clips


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level   = getattr(logging, args.log_level),
        format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )
    if args.command == "embed":
        cmd_embed(args)
    elif args.command == "detect":
        cmd_detect(args)
    elif args.command == "eval":
        cmd_eval(args)

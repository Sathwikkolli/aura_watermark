"""
AURA - Step 8: Dataset Pipeline Tests

Tests (25 total):
  01. test_load_audio_basic            - load_audio returns [1, T] tensor
  02. test_load_audio_mono_conversion  - stereo -> mono [1, T]
  03. test_load_audio_resampling       - output is at target_sr
  04. test_load_audio_bad_path         - returns None for non-existent file
  05. test_random_segment_crop         - crops long audio to n_samples
  06. test_random_segment_pad          - tiles short audio to n_samples
  07. test_random_segment_exact        - no-op for exact-length audio
  08. test_random_segment_output_shape - always outputs [1, n_samples]
  09. test_peak_normalize_unit_peak    - max(|x|) == 1.0 after normalise
  10. test_peak_normalize_silent       - silent clip returned unchanged
  11. test_scan_audio_files_finds_wav  - scans a temp dir for .wav files
  12. test_scan_audio_files_recursive  - recurses into subdirectories
  13. test_scan_audio_files_filters    - ignores non-audio files
  14. test_synthetic_dataset_len       - __len__ returns n_clips
  15. test_synthetic_dataset_shapes    - waveform [1,96000], message [32]
  16. test_synthetic_dataset_deterministic - same idx = same output
  17. test_synthetic_dataset_message_binary - message values in {0,1}
  18. test_audio_segment_dataset_basic - loads from disk, correct shapes
  19. test_audio_segment_dataset_peak  - output waveform is peak-normalised
  20. test_audio_segment_dataset_retry - skips corrupt files gracefully
  21. test_train_val_split_sizes       - val_frac respected
  22. test_train_val_split_disjoint    - no overlap between train and val
  23. test_train_val_split_deterministic - same seed = same split
  24. test_synthetic_dataloader_batch  - DataLoader yields correct shapes
  25. test_synthetic_dataloader_iterations - iterates n_clips / batch_size
"""

import io
import sys
import struct
import tempfile
import traceback
from pathlib import Path
from typing import Callable, List

import torch

sys.path.insert(0, "C:/Users/Sathwik/aura_watermark")

from aura_watermark.config import AURAConfig
from aura_watermark.dataset import (
    AudioSegmentDataset,
    SyntheticAudioDataset,
    _train_val_split,
    build_synthetic_dataloaders,
    load_audio,
    peak_normalize,
    random_segment,
    scan_audio_files,
)

# ── test harness ─────────────────────────────────────────────────────────────

PASSED: List[str] = []
FAILED: List[str] = []


def run(name: str, fn: Callable) -> None:
    try:
        fn()
        PASSED.append(name)
    except Exception as exc:
        FAILED.append(name)
        print(f"  [FAIL] {exc}")
        traceback.print_exc()


# ── helpers ───────────────────────────────────────────────────────────────────

SR   = 48_000
T    = 96_000
BITS = 32
cfg  = AURAConfig()


def _write_wav(path: Path, n_samples: int = T, sr: int = SR, n_channels: int = 1) -> None:
    """Write a minimal valid PCM WAV file with random data."""
    import wave, array, os
    data = (torch.randn(n_channels, n_samples) * 16384).short()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(sr)
        # interleave channels
        interleaved = data.t().contiguous().numpy().tobytes()
        wf.writeframes(interleaved)


def _make_wav_dir(n_files: int = 4, sr: int = SR, n_samples: int = T) -> tempfile.TemporaryDirectory:
    """Return a TemporaryDirectory containing n_files WAV files."""
    tmpdir = tempfile.TemporaryDirectory()
    for i in range(n_files):
        _write_wav(Path(tmpdir.name) / f"clip_{i:03d}.wav", n_samples, sr)
    return tmpdir


# ═════════════════════════════════════════════════════════════════════════════
# 01-04. load_audio
# ═════════════════════════════════════════════════════════════════════════════

def test_load_audio_basic():
    tmpdir = _make_wav_dir(1)
    path   = next(Path(tmpdir.name).glob("*.wav"))
    wav    = load_audio(path, target_sr=SR)
    assert wav is not None
    assert wav.ndim == 2, f"Expected 2D, got {wav.ndim}D"
    assert wav.shape[0] == 1, f"Expected 1 channel, got {wav.shape[0]}"
    print(f"  load_audio basic: shape {tuple(wav.shape)}  [PASS]")
    tmpdir.cleanup()


def test_load_audio_mono_conversion():
    tmpdir = tempfile.TemporaryDirectory()
    path   = Path(tmpdir.name) / "stereo.wav"
    _write_wav(path, n_channels=2)
    wav = load_audio(path, target_sr=SR)
    assert wav is not None
    assert wav.shape[0] == 1, f"Stereo not converted to mono: {wav.shape}"
    print(f"  load_audio stereo->mono: shape {tuple(wav.shape)}  [PASS]")
    tmpdir.cleanup()


def test_load_audio_resampling():
    tmpdir = tempfile.TemporaryDirectory()
    path   = Path(tmpdir.name) / "22050.wav"
    _write_wav(path, n_samples=22_050, sr=22_050)
    wav = load_audio(path, target_sr=SR)
    assert wav is not None
    # After resampling 22050->48000 with 1s of audio: expect ~48000 samples
    assert abs(wav.shape[-1] - SR) < 1000, f"Unexpected length after resample: {wav.shape[-1]}"
    print(f"  load_audio resample 22050->48000: {wav.shape[-1]} samples  [PASS]")
    tmpdir.cleanup()


def test_load_audio_bad_path():
    result = load_audio("/nonexistent/path/to/audio.wav", target_sr=SR)
    assert result is None, f"Expected None for bad path, got {result}"
    print(f"  load_audio bad path -> None  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 05-08. random_segment
# ═════════════════════════════════════════════════════════════════════════════

def test_random_segment_crop():
    x   = torch.randn(1, T * 2)
    out = random_segment(x, T)
    assert out.shape == (1, T), f"Expected (1, {T}), got {out.shape}"
    print(f"  random_segment crop: {tuple(out.shape)}  [PASS]")


def test_random_segment_pad():
    x   = torch.randn(1, T // 4)   # quarter length
    out = random_segment(x, T)
    assert out.shape == (1, T), f"Expected (1, {T}), got {out.shape}"
    print(f"  random_segment pad (tile): {tuple(out.shape)}  [PASS]")


def test_random_segment_exact():
    x   = torch.randn(1, T)
    out = random_segment(x, T)
    assert out.shape == (1, T)
    assert torch.allclose(x, out), "No-op case: output should equal input"
    print(f"  random_segment exact: no-op  [PASS]")


def test_random_segment_output_shape():
    for length in [100, T // 2, T, T * 3]:
        x   = torch.randn(1, length)
        out = random_segment(x, T)
        assert out.shape == (1, T), f"length={length} -> {out.shape}"
    print(f"  random_segment output shape always (1, {T})  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 09-10. peak_normalize
# ═════════════════════════════════════════════════════════════════════════════

def test_peak_normalize_unit_peak():
    x    = torch.randn(1, T) * 5.0
    out  = peak_normalize(x)
    peak = out.abs().max().item()
    assert abs(peak - 1.0) < 1e-5, f"Peak should be 1.0, got {peak:.6f}"
    print(f"  peak_normalize: max(|x|) = {peak:.6f}  [PASS]")


def test_peak_normalize_silent():
    x   = torch.zeros(1, T)
    out = peak_normalize(x)
    assert torch.allclose(x, out), "Silent clip should be returned unchanged"
    print(f"  peak_normalize silent: no-op  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 11-13. scan_audio_files
# ═════════════════════════════════════════════════════════════════════════════

def test_scan_audio_files_finds_wav():
    tmpdir = _make_wav_dir(5)
    files  = scan_audio_files(tmpdir.name)
    assert len(files) == 5, f"Expected 5 files, got {len(files)}"
    print(f"  scan_audio_files: found {len(files)} wav files  [PASS]")
    tmpdir.cleanup()


def test_scan_audio_files_recursive():
    import os
    tmpdir = tempfile.TemporaryDirectory()
    sub    = Path(tmpdir.name) / "sub"
    sub.mkdir()
    _write_wav(Path(tmpdir.name) / "top.wav")
    _write_wav(sub / "deep.wav")
    files  = scan_audio_files(tmpdir.name, recursive=True)
    assert len(files) == 2, f"Expected 2 files (recursive), got {len(files)}"
    print(f"  scan_audio_files recursive: {len(files)} files  [PASS]")
    tmpdir.cleanup()


def test_scan_audio_files_filters():
    tmpdir = tempfile.TemporaryDirectory()
    _write_wav(Path(tmpdir.name) / "audio.wav")
    (Path(tmpdir.name) / "notes.txt").write_text("ignore me")
    (Path(tmpdir.name) / "image.png").write_bytes(b"\x89PNG")
    files = scan_audio_files(tmpdir.name)
    assert len(files) == 1, f"Expected 1 audio file, got {len(files)}"
    print(f"  scan_audio_files filters non-audio  [PASS]")
    tmpdir.cleanup()


# ═════════════════════════════════════════════════════════════════════════════
# 14-17. SyntheticAudioDataset
# ═════════════════════════════════════════════════════════════════════════════

def test_synthetic_dataset_len():
    ds = SyntheticAudioDataset(n_clips=128, cfg=cfg)
    assert len(ds) == 128, f"Expected 128, got {len(ds)}"
    print(f"  SyntheticAudioDataset len: {len(ds)}  [PASS]")


def test_synthetic_dataset_shapes():
    ds      = SyntheticAudioDataset(n_clips=4, cfg=cfg)
    wav, msg = ds[0]
    assert wav.shape == (1, T),    f"waveform shape {wav.shape} != (1, {T})"
    assert msg.shape == (BITS,),   f"message shape {msg.shape} != ({BITS},)"
    print(f"  SyntheticAudioDataset shapes: wav={tuple(wav.shape)}, msg={tuple(msg.shape)}  [PASS]")


def test_synthetic_dataset_deterministic():
    ds      = SyntheticAudioDataset(n_clips=4, cfg=cfg, seed=7)
    wav1, _ = ds[2]
    wav2, _ = ds[2]
    assert torch.allclose(wav1, wav2), "Same index should produce same output"
    print(f"  SyntheticAudioDataset deterministic  [PASS]")


def test_synthetic_dataset_message_binary():
    ds = SyntheticAudioDataset(n_clips=10, cfg=cfg)
    for i in range(10):
        _, msg = ds[i]
        assert set(msg.tolist()).issubset({0, 1}), f"Non-binary values in message: {msg}"
    print(f"  SyntheticAudioDataset message in {{0,1}}  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 18-20. AudioSegmentDataset (from real files)
# ═════════════════════════════════════════════════════════════════════════════

def test_audio_segment_dataset_basic():
    tmpdir = _make_wav_dir(4)
    paths  = scan_audio_files(tmpdir.name)
    ds     = AudioSegmentDataset(paths, cfg)
    assert len(ds) == 4
    wav, msg = ds[0]
    assert wav.shape == (1, T),   f"waveform shape {wav.shape}"
    assert msg.shape == (BITS,),  f"message shape {msg.shape}"
    print(f"  AudioSegmentDataset from disk: shapes OK  [PASS]")
    tmpdir.cleanup()


def test_audio_segment_dataset_peak():
    tmpdir = _make_wav_dir(2)
    paths  = scan_audio_files(tmpdir.name)
    ds     = AudioSegmentDataset(paths, cfg)
    wav, _ = ds[0]
    peak   = wav.abs().max().item()
    assert abs(peak - 1.0) < 1e-4 or peak == 0.0, f"Expected peak ~1.0, got {peak:.4f}"
    print(f"  AudioSegmentDataset waveform peak-normalised: {peak:.4f}  [PASS]")
    tmpdir.cleanup()


def test_audio_segment_dataset_retry():
    """Dataset should skip corrupt paths and fall back to valid ones."""
    tmpdir  = _make_wav_dir(4)
    paths   = scan_audio_files(tmpdir.name)
    # Prepend a broken path — dataset should retry and return a valid clip
    corrupt_path = Path(tmpdir.name) / "corrupt.wav"
    corrupt_path.write_bytes(b"THIS IS NOT A WAV FILE")
    bad_paths = [corrupt_path] + paths
    ds = AudioSegmentDataset(bad_paths, cfg, max_retries=10)
    wav, msg = ds[0]   # should not raise
    assert wav.shape == (1, T)
    print(f"  AudioSegmentDataset retry on corrupt: OK  [PASS]")
    tmpdir.cleanup()


# ═════════════════════════════════════════════════════════════════════════════
# 21-23. _train_val_split
# ═════════════════════════════════════════════════════════════════════════════

def test_train_val_split_sizes():
    files    = [Path(f"file_{i}.wav") for i in range(100)]
    train    = _train_val_split(files, "train", val_frac=0.1, seed=42)
    val      = _train_val_split(files, "val",   val_frac=0.1, seed=42)
    assert len(train) == 90, f"Expected 90 train files, got {len(train)}"
    assert len(val)   == 10, f"Expected 10 val files, got {len(val)}"
    print(f"  _train_val_split: train={len(train)}, val={len(val)}  [PASS]")


def test_train_val_split_disjoint():
    files = [Path(f"file_{i}.wav") for i in range(100)]
    train = set(_train_val_split(files, "train", val_frac=0.1, seed=42))
    val   = set(_train_val_split(files, "val",   val_frac=0.1, seed=42))
    overlap = train & val
    assert len(overlap) == 0, f"Train/val overlap: {overlap}"
    print(f"  _train_val_split: no overlap  [PASS]")


def test_train_val_split_deterministic():
    files  = [Path(f"file_{i}.wav") for i in range(50)]
    train1 = _train_val_split(files, "train", val_frac=0.1, seed=7)
    train2 = _train_val_split(files, "train", val_frac=0.1, seed=7)
    assert train1 == train2, "Same seed should produce same split"
    print(f"  _train_val_split deterministic (seed=7)  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# 24-25. build_synthetic_dataloaders
# ═════════════════════════════════════════════════════════════════════════════

def test_synthetic_dataloader_batch():
    cfg_small = AURAConfig()
    train_loader, val_loader = build_synthetic_dataloaders(
        cfg_small, n_train=8, n_val=4, batch_size=4, num_workers=0
    )
    batch_wav, batch_msg = next(iter(train_loader))
    assert batch_wav.shape == (4, 1, T),    f"waveform batch shape {batch_wav.shape}"
    assert batch_msg.shape == (4, BITS),    f"message batch shape {batch_msg.shape}"
    print(f"  DataLoader batch: wav={tuple(batch_wav.shape)}, msg={tuple(batch_msg.shape)}  [PASS]")


def test_synthetic_dataloader_iterations():
    cfg_small = AURAConfig()
    train_loader, _ = build_synthetic_dataloaders(
        cfg_small, n_train=16, n_val=4, batch_size=4, num_workers=0
    )
    n_batches = sum(1 for _ in train_loader)
    # 16 clips / 4 batch_size = 4 batches (drop_last=True)
    assert n_batches == 4, f"Expected 4 batches, got {n_batches}"
    print(f"  DataLoader iterations: {n_batches} batches  [PASS]")


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════

TESTS = [
    ("test_load_audio_basic",              test_load_audio_basic),
    ("test_load_audio_mono_conversion",    test_load_audio_mono_conversion),
    ("test_load_audio_resampling",         test_load_audio_resampling),
    ("test_load_audio_bad_path",           test_load_audio_bad_path),
    ("test_random_segment_crop",           test_random_segment_crop),
    ("test_random_segment_pad",            test_random_segment_pad),
    ("test_random_segment_exact",          test_random_segment_exact),
    ("test_random_segment_output_shape",   test_random_segment_output_shape),
    ("test_peak_normalize_unit_peak",      test_peak_normalize_unit_peak),
    ("test_peak_normalize_silent",         test_peak_normalize_silent),
    ("test_scan_audio_files_finds_wav",    test_scan_audio_files_finds_wav),
    ("test_scan_audio_files_recursive",    test_scan_audio_files_recursive),
    ("test_scan_audio_files_filters",      test_scan_audio_files_filters),
    ("test_synthetic_dataset_len",         test_synthetic_dataset_len),
    ("test_synthetic_dataset_shapes",      test_synthetic_dataset_shapes),
    ("test_synthetic_dataset_deterministic",test_synthetic_dataset_deterministic),
    ("test_synthetic_dataset_message_binary",test_synthetic_dataset_message_binary),
    ("test_audio_segment_dataset_basic",   test_audio_segment_dataset_basic),
    ("test_audio_segment_dataset_peak",    test_audio_segment_dataset_peak),
    ("test_audio_segment_dataset_retry",   test_audio_segment_dataset_retry),
    ("test_train_val_split_sizes",         test_train_val_split_sizes),
    ("test_train_val_split_disjoint",      test_train_val_split_disjoint),
    ("test_train_val_split_deterministic", test_train_val_split_deterministic),
    ("test_synthetic_dataloader_batch",    test_synthetic_dataloader_batch),
    ("test_synthetic_dataloader_iterations",test_synthetic_dataloader_iterations),
]

if __name__ == "__main__":
    print("=" * 60)
    print("AURA - Step 8: Dataset Pipeline Tests")
    print("=" * 60)

    for name, fn in TESTS:
        print(f"\n{name}")
        run(name, fn)

    print("\n" + "=" * 60)
    print(f"Results: {len(PASSED)} passed, {len(FAILED)} failed")
    print("=" * 60)
    if FAILED:
        print("FAILED tests:")
        for f in FAILED:
            print(f"  - {f}")
        sys.exit(1)

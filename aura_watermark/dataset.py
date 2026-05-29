"""
AURA Step 8 — Dataset Pipeline.

Two training corpora (~2 500 hr each):
  Emilia  ~2 500 hr multilingual speech  (amphion/Emilia-Dataset on HuggingFace)
  FMA     ~2 500 hr music                (FMA full or equivalent mirror — not fma_large alone)

Expected directory layouts after download:

  Emilia:
    <emilia_root>/
      EN/  ZH/  DE/  FR/  JA/  KO/
        *.mp3  (or *.wav / *.flac)

  FMA (~2 500 hr — use fma_full or a curated mirror, not only fma_large ~880 hr):
    <fma_root>/
      fma_full/          # preferred (mdeff/fma fma_full.zip)
        000/ 001/ ... /
          *.mp3
      # or fma_large/ for smaller smoke tests only

Both datasets are subclasses of AudioSegmentDataset which:
  1. Scans root directories for supported audio files (.wav/.mp3/.flac/.ogg)
  2. At __getitem__, loads a file via torchaudio, resamples to 48 kHz, mono
  3. Randomly segments to 2 s (96 000 samples) — repeats short clips
  4. Peak-normalises to [-1, 1]
  5. Generates a fresh random 32-bit message

AURACombinedDataset mixes the two corpora with a configurable
speech:music ratio (default 50:50 — equal hour budgets).

build_dataloaders() splits each corpus into train/val (fixed seed),
returns (train_loader, val_loader) ready for AURATrainer.

Usage:
    train_loader, val_loader = build_dataloaders(
        cfg,
        emilia_root = "/data/emilia",
        fma_root    = "/data/fma",
    )
    for audio, message in train_loader:
        result = trainer.train_step(audio.to(device), message.to(device))
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import torchaudio
    import torchaudio.functional as TAF
    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False

from .config import AURAConfig

Tensor = torch.Tensor
PathLike = Union[str, Path]

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


# ─────────────────────────────────────────────────────────────────────────────
# Low-level audio helpers (pure functions — easy to test in isolation)
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(
    path: PathLike,
    target_sr: int = 48_000,
) -> Optional[Tensor]:
    """
    Load an audio file, resample to ``target_sr``, convert to mono.

    Args:
        path:      path to any torchaudio-readable file
        target_sr: output sample rate (default 48 000 Hz)

    Returns:
        [1, T] float32 tensor, or ``None`` if the file cannot be loaded.
    """
    if not _TA_AVAILABLE:
        raise RuntimeError("torchaudio is required for dataset loading.")

    try:
        waveform, sr = torchaudio.load(str(path))
    except Exception as exc:
        warnings.warn(f"Failed to load {path}: {exc}")
        return None

    # Convert to mono (average channels)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to target_sr (integer ratio → no OOM)
    if sr != target_sr:
        waveform = TAF.resample(waveform, orig_freq=sr, new_freq=target_sr)

    return waveform.float()   # [1, T]


def random_segment(waveform: Tensor, n_samples: int, rng: Optional[random.Random] = None) -> Tensor:
    """
    Crop or pad a waveform to exactly ``n_samples``.

    Long clips:  random crop (uniform start offset).
    Short clips: tile then crop (avoids silent padding artefacts).

    Args:
        waveform:  [1, T] or [T]
        n_samples: target length
        rng:       optional ``random.Random`` instance (for reproducibility)

    Returns:
        [1, n_samples]
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    T = waveform.shape[-1]

    if T == n_samples:
        return waveform

    if T > n_samples:
        # Random crop
        max_start = T - n_samples
        start = (rng or random).randint(0, max_start)
        return waveform[:, start : start + n_samples]

    # Tile until long enough, then random crop
    repeats = math.ceil(n_samples / T)
    waveform = waveform.repeat(1, repeats)
    return random_segment(waveform, n_samples, rng)


def peak_normalize(waveform: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Normalise a waveform so its peak absolute value is 1.0.

    Clips that are completely silent (all zeros) are returned as-is.

    Args:
        waveform: [1, T]
        eps:      floor to prevent division by near-zero

    Returns:
        [1, T] with max(|x|) == 1.0
    """
    peak = waveform.abs().max()
    if peak < eps:
        return waveform
    return waveform / peak


def scan_audio_files(
    root: PathLike,
    extensions: Optional[set] = None,
    recursive: bool = True,
) -> List[Path]:
    """
    Walk ``root`` and collect all audio file paths.

    Args:
        root:       root directory to scan
        extensions: set of lowercase extensions to accept (default: WAV/MP3/FLAC/OGG/M4A/AAC)
        recursive:  if True, recurse into subdirectories

    Returns:
        Sorted list of Path objects.
    """
    exts  = extensions or _AUDIO_EXTENSIONS
    root  = Path(root)
    glob  = "**/*" if recursive else "*"
    files = [
        p for p in root.glob(glob)
        if p.is_file() and p.suffix.lower() in exts
    ]
    return sorted(files)


# math.ceil is used in random_segment
import math


# ─────────────────────────────────────────────────────────────────────────────
# Base dataset
# ─────────────────────────────────────────────────────────────────────────────

class AudioSegmentDataset(Dataset):
    """
    Generic audio dataset built from a list of file paths.

    Each ``__getitem__`` call:
      1. Loads the audio file (retries up to ``max_retries`` on failure).
      2. Resamples to 48 kHz, converts to mono.
      3. Randomly segments to 2 s (96 000 samples).
      4. Peak-normalises.
      5. Samples a fresh random 32-bit binary message.

    Args:
        paths:       list of audio file paths
        cfg:         AURAConfig — uses stft.sample_rate, stft.segment_samples,
                     message.n_bits
        transform:   optional callable applied to the waveform after preprocessing
        max_retries: how many consecutive load failures before raising an error
    """

    def __init__(
        self,
        paths:       List[Path],
        cfg:         AURAConfig,
        transform:   Optional[Callable[[Tensor], Tensor]] = None,
        max_retries: int = 10,
    ):
        if not paths:
            raise ValueError("AudioSegmentDataset received an empty file list.")

        self.paths       = paths
        self.target_sr   = cfg.stft.sample_rate       # 48 000
        self.n_samples   = cfg.stft.segment_samples   # 96 000
        self.n_bits      = cfg.message.n_bits          # 32
        self.transform   = transform
        self.max_retries = max_retries

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            waveform: [1, 96 000]  peak-normalised mono audio
            message:  [32]         random binary bits {0, 1}
        """
        for attempt in range(self.max_retries):
            try:
                waveform = load_audio(self.paths[idx], self.target_sr)
                if waveform is None or waveform.shape[-1] < self.target_sr // 4:
                    # Skip very short or unloadable clips
                    raise ValueError("clip too short or load failed")

                waveform = random_segment(waveform, self.n_samples)
                waveform = peak_normalize(waveform)

                if self.transform is not None:
                    waveform = self.transform(waveform)

                message = torch.randint(0, 2, (self.n_bits,), dtype=torch.long)
                return waveform, message

            except Exception:
                # Try a random different file
                idx = random.randint(0, len(self.paths) - 1)

        raise RuntimeError(
            f"Failed to load a valid audio clip after {self.max_retries} retries."
        )


# ─────────────────────────────────────────────────────────────────────────────
# FMA directory resolution
# ─────────────────────────────────────────────────────────────────────────────

FMA_SUBSET_CHOICES = ("auto", "fma_full", "fma_large", "root")


def resolve_fma_directory(root: PathLike, subset: str = "auto") -> Path:
    """
    Locate the FMA audio tree under ``root``.

    Args:
        root:   path passed as ``--fma-root``
        subset: ``auto`` | ``fma_full`` | ``fma_large`` | ``root``
                ``auto`` prefers ``fma_full`` (target ~2 500 hr), then ``fma_large``.

    Returns:
        Directory to scan recursively for audio files.

    Raises:
        FileNotFoundError: if the requested layout is missing.
        ValueError: invalid ``subset`` name.
    """
    root = Path(root)
    subset = subset.lower().strip()

    if subset not in FMA_SUBSET_CHOICES:
        raise ValueError(
            f"fma_subset must be one of {FMA_SUBSET_CHOICES}, got {subset!r}"
        )

    if subset == "root":
        return root

    if subset == "fma_large":
        d = root / "fma_large"
        return d if d.is_dir() else root

    if subset == "fma_full":
        d = root / "fma_full"
        if d.is_dir():
            return d
        raise FileNotFoundError(
            f"FMA full subset not found at {d}. "
            "Download fma_full from https://github.com/mdeff/fma (~2 500+ hr). "
            "For smoke tests only, use --fma-subset fma_large or auto."
        )

    # auto: prefer full corpus for paper-scale training
    for name in ("fma_full", "fma_large"):
        d = root / name
        if d.is_dir():
            return d
    return root


# ─────────────────────────────────────────────────────────────────────────────
# Emilia speech dataset
# ─────────────────────────────────────────────────────────────────────────────

class EmiliaDataset(AudioSegmentDataset):
    """
    Emilia multilingual speech corpus (~2 500 hr).

    Download: ``huggingface-cli download amphion/Emilia-Dataset``

    Expected directory layout::

        <root>/
          EN/   # English
          ZH/   # Mandarin
          DE/   # German
          FR/   # French
          JA/   # Japanese
          KO/   # Korean
            *.mp3 (or *.wav / *.flac)

    Args:
        root:      path to Emilia root directory
        cfg:       AURAConfig
        languages: list of language codes to include (default: all 6)
        split:     ``"train"`` or ``"val"``
        val_frac:  fraction of files reserved for validation
        seed:      random seed for the train/val split
    """

    ALL_LANGUAGES: List[str] = ["EN", "ZH", "DE", "FR", "JA", "KO"]

    def __init__(
        self,
        root:      PathLike,
        cfg:       AURAConfig,
        languages: Optional[List[str]] = None,
        split:     str = "train",
        val_frac:  float = 0.01,
        seed:      int = 42,
        **kwargs,
    ):
        langs = languages or self.ALL_LANGUAGES
        root  = Path(root)

        all_files: List[Path] = []
        for lang in langs:
            lang_dir = root / lang
            if lang_dir.is_dir():
                all_files.extend(scan_audio_files(lang_dir))
            else:
                warnings.warn(f"EmiliaDataset: language directory not found: {lang_dir}")

        if not all_files:
            raise FileNotFoundError(
                f"No audio files found under {root} for languages {langs}. "
                "Check that the Emilia dataset has been downloaded and extracted."
            )

        paths = _train_val_split(all_files, split, val_frac, seed)
        super().__init__(paths, cfg, **kwargs)

        self.root      = root
        self.languages = langs
        self.split     = split


# ─────────────────────────────────────────────────────────────────────────────
# FMA music dataset
# ─────────────────────────────────────────────────────────────────────────────

class FMADataset(AudioSegmentDataset):
    """
    Free Music Archive (FMA) music corpus (~2 500 hr target for AURA training).

    Use ``fma_full`` (or a ~2 500 hr mirror), not ``fma_large`` alone (~880 hr).

    Download: https://github.com/mdeff/fma
      - ``fma_full.zip`` — full corpus (~2 500+ hr; recommended)
      - ``fma_large.zip`` — smaller subset (~880 hr; smoke tests only)

    Expected directory layout::

        <root>/
          fma_full/          # preferred
            000/ 001/ ... /
              *.mp3
          # or fma_large/ for debugging

    Args:
        root:       path containing ``fma_full`` / ``fma_large`` or flat audio tree
        cfg:        AURAConfig (uses ``cfg.dataset.fma_subset`` when subset is None)
        subset:     ``auto`` | ``fma_full`` | ``fma_large`` | ``root``
        split:      ``"train"`` or ``"val"``
        val_frac:   fraction of files reserved for validation
        seed:       random seed for the train/val split
    """

    def __init__(
        self,
        root:     PathLike,
        cfg:      AURAConfig,
        subset:   Optional[str] = None,
        split:    str = "train",
        val_frac: float = 0.01,
        seed:     int = 42,
        **kwargs,
    ):
        root     = Path(root)
        subset   = subset if subset is not None else cfg.dataset.fma_subset
        fma_dir  = resolve_fma_directory(root, subset)

        all_files = scan_audio_files(fma_dir)

        if not all_files:
            raise FileNotFoundError(
                f"No audio files found under {fma_dir}. "
                "Extract fma_full (or fma_large for tests) under --fma-root. "
                f"subset={subset!r}, root={root}"
            )

        paths = _train_val_split(all_files, split, val_frac, seed)
        super().__init__(paths, cfg, **kwargs)

        self.root   = root
        self.subset = subset
        self.fma_dir = fma_dir
        self.split  = split


# ─────────────────────────────────────────────────────────────────────────────
# Combined dataset (speech + music mix)
# ─────────────────────────────────────────────────────────────────────────────

class AURACombinedDataset(Dataset):
    """
    Mixes Emilia speech and FMA music at a configurable ratio.

    At each ``__getitem__`` call, a Bernoulli draw determines whether to
    sample from speech or music.  The dataset length is set so that
    one epoch sees approximately ``total_clips`` clips.

    Args:
        speech_dataset: EmiliaDataset (or any AudioSegmentDataset)
        music_dataset:  FMADataset   (or any AudioSegmentDataset)
        speech_ratio:   fraction of clips drawn from the speech dataset
                        (default 0.5 — equal Emilia / FMA hour budgets)
        total_clips:    virtual epoch length (default: max of the two datasets)
    """

    def __init__(
        self,
        speech_dataset: AudioSegmentDataset,
        music_dataset:  AudioSegmentDataset,
        speech_ratio:   float = 0.5,
        total_clips:    Optional[int] = None,
    ):
        assert 0.0 < speech_ratio < 1.0, "speech_ratio must be in (0, 1)"

        self.speech  = speech_dataset
        self.music   = music_dataset
        self.p_speech = speech_ratio
        self._length = total_clips or max(len(speech_dataset), len(music_dataset))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, _idx: int) -> Tuple[Tensor, Tensor]:
        """
        Ignores the provided index; always draws uniformly from the
        chosen sub-dataset.  This is intentional: both datasets are
        large enough that index-based access would not cycle uniformly.
        """
        if random.random() < self.p_speech:
            inner_idx = random.randint(0, len(self.speech) - 1)
            return self.speech[inner_idx]
        else:
            inner_idx = random.randint(0, len(self.music) - 1)
            return self.music[inner_idx]


# ─────────────────────────────────────────────────────────────────────────────
# In-memory mock dataset (for testing and quick experiments)
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticAudioDataset(Dataset):
    """
    Generates random Gaussian audio clips in memory — no disk I/O.

    Useful for unit tests, debugging, and smoke-testing the training loop
    without downloading the full corpora.

    Args:
        n_clips:    number of clips in the dataset
        cfg:        AURAConfig
        seed:       random seed (for reproducibility across workers)
    """

    def __init__(
        self,
        n_clips: int,
        cfg:     AURAConfig,
        seed:    int = 0,
    ):
        self.n_clips   = n_clips
        self.n_samples = cfg.stft.segment_samples   # 96 000
        self.n_bits    = cfg.message.n_bits           # 32
        self.seed      = seed

    def __len__(self) -> int:
        return self.n_clips

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # Seeded per-clip so the dataset is deterministic
        gen = torch.Generator()
        gen.manual_seed(self.seed + idx)

        waveform = torch.randn(1, self.n_samples, generator=gen) * 0.3
        waveform = peak_normalize(waveform)
        message  = torch.randint(0, 2, (self.n_bits,), generator=gen, dtype=torch.long)
        return waveform, message


# ─────────────────────────────────────────────────────────────────────────────
# Train/val split helper
# ─────────────────────────────────────────────────────────────────────────────

def _train_val_split(
    files:    List[Path],
    split:    str,
    val_frac: float,
    seed:     int,
) -> List[Path]:
    """
    Deterministically split a file list into train and val subsets.

    Files are shuffled with ``seed`` then the last ``val_frac`` fraction
    is reserved for validation.
    """
    rng = random.Random(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_frac))

    if split == "val":
        return shuffled[-n_val:]
    elif split == "train":
        return shuffled[:-n_val]
    else:
        raise ValueError(f"split must be 'train' or 'val', got '{split}'")


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    cfg:          AURAConfig,
    emilia_root:  Optional[PathLike] = None,
    fma_root:     Optional[PathLike] = None,
    speech_ratio: Optional[float] = None,
    fma_subset:   Optional[str] = None,
    batch_size:   Optional[int] = None,
    num_workers:  int = 4,
    pin_memory:   bool = True,
    val_frac:     float = 0.01,
    seed:         int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders for AURA.

    At least one of ``emilia_root`` or ``fma_root`` must be provided.
    If both are provided, a combined speech:music dataset is used for training
    (default 50:50 — ~2 500 hr Emilia + ~2 500 hr FMA); validation concatenates
    both corpora.

    Args:
        cfg:          AURAConfig
        emilia_root:  path to Emilia dataset root (optional)
        fma_root:     path to FMA root with ``fma_full/`` or ``fma_large/`` (optional)
        speech_ratio: Emilia fraction (default: ``cfg.dataset.speech_ratio``, 0.5)
        fma_subset:   FMA tree selector (default: ``cfg.dataset.fma_subset``, ``auto``)
        batch_size:   clips per GPU per step (default: cfg.training.batch_size)
        num_workers:  DataLoader worker processes
        pin_memory:   pin CPU tensors to GPU memory for faster transfer
        val_frac:     fraction of each corpus reserved for validation
        seed:         split seed

    Returns:
        (train_loader, val_loader)
    """
    if emilia_root is None and fma_root is None:
        raise ValueError("At least one of emilia_root or fma_root must be provided.")

    speech_ratio = (
        speech_ratio if speech_ratio is not None else cfg.dataset.speech_ratio
    )
    fma_subset = fma_subset if fma_subset is not None else cfg.dataset.fma_subset

    bs = batch_size or cfg.training.batch_size

    train_datasets, val_datasets = [], []

    if emilia_root is not None:
        train_datasets.append(
            EmiliaDataset(emilia_root, cfg, split="train", val_frac=val_frac, seed=seed)
        )
        val_datasets.append(
            EmiliaDataset(emilia_root, cfg, split="val", val_frac=val_frac, seed=seed)
        )

    if fma_root is not None:
        train_datasets.append(
            FMADataset(
                fma_root, cfg, subset=fma_subset,
                split="train", val_frac=val_frac, seed=seed,
            )
        )
        val_datasets.append(
            FMADataset(
                fma_root, cfg, subset=fma_subset,
                split="val", val_frac=val_frac, seed=seed,
            )
        )

    # ── Training dataset ──────────────────────────────────────────────────
    if len(train_datasets) == 2:
        train_dataset = AURACombinedDataset(
            speech_dataset = train_datasets[0],
            music_dataset  = train_datasets[1],
            speech_ratio   = speech_ratio,
        )
    else:
        train_dataset = train_datasets[0]

    # ── Validation dataset ────────────────────────────────────────────────
    if len(val_datasets) == 2:
        # Simple concatenation for validation — no ratio mixing needed
        from torch.utils.data import ConcatDataset
        val_dataset = ConcatDataset(val_datasets)
    else:
        val_dataset = val_datasets[0]

    # ── DataLoaders ───────────────────────────────────────────────────────
    loader_kwargs = dict(
        batch_size  = bs,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = True,
        persistent_workers = num_workers > 0,
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle = True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle = False,
        **loader_kwargs,
    )

    return train_loader, val_loader


def build_synthetic_dataloaders(
    cfg:          AURAConfig,
    n_train:      int = 256,
    n_val:        int = 32,
    batch_size:   Optional[int] = None,
    num_workers:  int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build DataLoaders with synthetic in-memory data.

    Drop-in replacement for ``build_dataloaders`` when the real corpora
    are not available (CI, unit tests, quick sanity checks).

    Args:
        cfg:        AURAConfig
        n_train:    number of synthetic training clips
        n_val:      number of synthetic validation clips
        batch_size: batch size (default: cfg.training.batch_size)
        num_workers: DataLoader workers (default 0 for in-process)

    Returns:
        (train_loader, val_loader)
    """
    bs = batch_size or cfg.training.batch_size

    train_loader = DataLoader(
        SyntheticAudioDataset(n_train, cfg, seed=0),
        batch_size  = bs,
        shuffle     = True,
        num_workers = num_workers,
        drop_last   = True,
    )
    val_loader = DataLoader(
        SyntheticAudioDataset(n_val, cfg, seed=9999),
        batch_size  = bs,
        shuffle     = False,
        num_workers = num_workers,
        drop_last   = False,
    )
    return train_loader, val_loader

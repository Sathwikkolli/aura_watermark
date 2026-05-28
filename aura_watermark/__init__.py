# AURA: A Stegaformer-Based Scalable Deep Audio Watermark with Extreme Robustness
# Implementation following ICASSP 2026 paper

from .config import AURAConfig
from .stft import STFTProcessor, ISTFTReconstructor
from .conformer import StegaformerBackbone
from .embedder import StegaformerEmbedder
from .detector import AURADecoder
from .discriminator import BigVGANDiscriminator
from .losses import AURALoss, MultiResSTFTLoss, NMRLoss, LossComponents
from .trainer import AURATrainer, StepResult, compute_lr, compute_double_encode_prob
from .dataset import (
    AudioSegmentDataset, SyntheticAudioDataset,
    EmiliaDataset, FMADataset, AURACombinedDataset,
    build_dataloaders, build_synthetic_dataloaders,
    load_audio, random_segment, peak_normalize, scan_audio_files,
)

# AURA: A Stegaformer-Based Scalable Deep Audio Watermark with Extreme Robustness
# Implementation following ICASSP 2026 paper

from .config import AURAConfig
from .stft import STFTProcessor, ISTFTReconstructor
from .conformer import StegaformerBackbone
from .embedder import StegaformerEmbedder
from .detector import AURADecoder
from .discriminator import BigVGANDiscriminator
from .losses import AURALoss, MultiResSTFTLoss, NMRLoss, LossComponents

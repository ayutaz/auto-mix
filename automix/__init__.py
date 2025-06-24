"""
AutoMix - Auto mixing and video generation for vocal covers
"""

__version__ = "0.1.0"
__author__ = "ayutaz"
__email__ = "ka1357amnbpdr@gmail.com"

from .core.analyzer import AudioAnalyzer
from .core.audio_loader import AudioLoader
from .core.processor import MixProcessor

__all__ = [
    "AudioLoader",
    "AudioAnalyzer",
    "MixProcessor",
]

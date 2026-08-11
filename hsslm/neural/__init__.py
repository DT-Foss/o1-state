"""
HSSLM - Hierarchical State-Space Language Module

A minimal non-transformer language model with explicit hierarchical linguistic
processing, based on selective state space models (Mamba-style).

~7.3M parameters | O(n) complexity | Pure PyTorch
"""

from .config import HSSLMConfig
from .model import HSSLM

__version__ = "1.0.0"
__all__ = ["HSSLM", "HSSLMConfig"]

"""
Command line interface for RLHF library.
"""

from .train import train
from .evaluate import evaluate
from .collect import collect

__all__ = [
    "train",
    "evaluate", 
    "collect",
] 
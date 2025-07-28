"""
RLHF (Reinforcement Learning from Human Feedback) Library

A comprehensive library for training language models using human feedback.
"""

__version__ = "0.1.0"
__author__ = "RLHF Team"
__email__ = "team@rlhf.com"

# Core imports
from .trainer import RLHFTrainer
from .config import PPOConfig, DPOConfig, RLHFConfig
from .models import GPT2Policy, GPT2Value, LLMPolicy, LLMValue
from .data import PreferenceDataset, PreferenceDataLoader
from .algorithms import PPO, DPO, A2C
from .utils import set_seed, get_device, setup_logging

# CLI imports
from .cli import train, evaluate, collect

__all__ = [
    # Core
    "RLHFTrainer",
    "PPOConfig",
    "DPOConfig", 
    "RLHFConfig",
    "GPT2Policy",
    "GPT2Value",
    "LLMPolicy",
    "LLMValue",
    "PreferenceDataset",
    "PreferenceDataLoader",
    "PPO",
    "DPO",
    "A2C",
    # Utils
    "set_seed",
    "get_device",
    "setup_logging",
    # CLI
    "train",
    "evaluate",
    "collect",
] 
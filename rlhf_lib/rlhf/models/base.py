"""
Base model class for RLHF models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer


class BaseModel(ABC, nn.Module):
    """Base class for all RLHF models."""
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
    @abstractmethod
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass of the model."""
        pass
    
    @abstractmethod
    def generate(self, *args, **kwargs) -> torch.Tensor:
        """Generate text using the model."""
        pass
    
    def get_device(self) -> torch.device:
        """Get the device the model is on."""
        return self.device
    
    def to_device(self, device: torch.device) -> 'BaseModel':
        """Move model to specified device."""
        self.model = self.model.to(device)
        self.device = device
        return self
    
    def save_pretrained(self, path: str) -> None:
        """Save the model to a directory."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> 'BaseModel':
        """Load a model from a pretrained checkpoint."""
        raise NotImplementedError
    
    def get_trainable_parameters(self) -> nn.Parameter:
        """Get trainable parameters of the model."""
        return self.model.parameters()
    
    def get_model_size(self) -> int:
        """Get the number of parameters in the model."""
        return sum(p.numel() for p in self.model.parameters())
    
    def get_trainable_size(self) -> int:
        """Get the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad) 
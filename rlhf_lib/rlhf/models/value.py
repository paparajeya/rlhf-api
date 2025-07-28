"""
Value models for RLHF training.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer, AutoModel, AutoTokenizer
from .base import BaseModel


class GPT2Value(BaseModel):
    """GPT-2 based value model for RLHF."""
    
    def __init__(self, model_name: str = "gpt2"):
        model = GPT2Model.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Add value head
        self.value_head = nn.Linear(model.config.hidden_size, 1)
        
        super().__init__(model, tokenizer)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the value model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Get the last hidden state
        hidden_states = outputs.last_hidden_state
        
        # Apply value head
        values = self.value_head(hidden_states).squeeze(-1)
        
        return {
            "values": values,
            "hidden_states": outputs.hidden_states,
        }
    
    def get_value(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get the value for the given input."""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            values = outputs["values"]
            
        return values
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> 'GPT2Value':
        """Load a value model from a pretrained checkpoint."""
        model = GPT2Model.from_pretrained(path)
        tokenizer = GPT2Tokenizer.from_pretrained(path)
        
        instance = cls.__new__(cls)
        instance.model = model
        instance.tokenizer = tokenizer
        instance.device = next(model.parameters()).device
        
        # Initialize value head
        instance.value_head = nn.Linear(model.config.hidden_size, 1)
        
        return instance


class LLMValue(BaseModel):
    """Generic LLM value model for RLHF."""
    
    def __init__(self, model_name: str):
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Add value head
        self.value_head = nn.Linear(model.config.hidden_size, 1)
        
        super().__init__(model, tokenizer)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the value model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Get the last hidden state
        hidden_states = outputs.last_hidden_state
        
        # Apply value head
        values = self.value_head(hidden_states).squeeze(-1)
        
        return {
            "values": values,
            "hidden_states": outputs.hidden_states,
        }
    
    def get_value(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get the value for the given input."""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            values = outputs["values"]
            
        return values
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> 'LLMValue':
        """Load a value model from a pretrained checkpoint."""
        model = AutoModel.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        
        instance = cls.__new__(cls)
        instance.model = model
        instance.tokenizer = tokenizer
        instance.device = next(model.parameters()).device
        
        # Initialize value head
        instance.value_head = nn.Linear(model.config.hidden_size, 1)
        
        return instance 
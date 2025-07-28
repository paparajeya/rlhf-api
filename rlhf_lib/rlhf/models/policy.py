"""
Policy models for RLHF training.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
from .base import BaseModel


class GPT2Policy(BaseModel):
    """GPT-2 based policy model for RLHF."""
    
    def __init__(self, model_name: str = "gpt2"):
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        super().__init__(model, tokenizer)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the policy model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        return {
            "logits": outputs.logits,
            "loss": outputs.loss if labels is not None else None,
            "hidden_states": outputs.hidden_states,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """Generate text using the policy model."""
        return self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
    
    def get_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get log probabilities for the given input."""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs["logits"]
            logprobs = F.log_softmax(logits, dim=-1)
            
        return logprobs
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> 'GPT2Policy':
        """Load a policy model from a pretrained checkpoint."""
        model = GPT2LMHeadModel.from_pretrained(path)
        tokenizer = GPT2Tokenizer.from_pretrained(path)
        
        instance = cls.__new__(cls)
        instance.model = model
        instance.tokenizer = tokenizer
        instance.device = next(model.parameters()).device
        
        return instance


class LLMPolicy(BaseModel):
    """Generic LLM policy model for RLHF."""
    
    def __init__(self, model_name: str):
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        super().__init__(model, tokenizer)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the policy model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        return {
            "logits": outputs.logits,
            "loss": outputs.loss if labels is not None else None,
            "hidden_states": outputs.hidden_states,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """Generate text using the policy model."""
        return self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
    
    def get_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get log probabilities for the given input."""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs["logits"]
            logprobs = F.log_softmax(logits, dim=-1)
            
        return logprobs
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> 'LLMPolicy':
        """Load a policy model from a pretrained checkpoint."""
        model = AutoModelForCausalLM.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        
        instance = cls.__new__(cls)
        instance.model = model
        instance.tokenizer = tokenizer
        instance.device = next(model.parameters()).device
        
        return instance 
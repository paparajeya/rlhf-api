"""
Reward model for RLHF training.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer, AutoModel, AutoTokenizer
from .base import BaseModel


class RewardModel(BaseModel):
    """Reward model for RLHF training."""
    
    def __init__(self, model_name: str = "gpt2"):
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Add reward head
        self.reward_head = nn.Linear(model.config.hidden_size, 1)
        
        super().__init__(model, tokenizer)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the reward model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Get the last hidden state
        hidden_states = outputs.last_hidden_state
        
        # Apply reward head
        rewards = self.reward_head(hidden_states).squeeze(-1)
        
        return {
            "rewards": rewards,
            "hidden_states": outputs.hidden_states,
        }
    
    def get_reward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get the reward for the given input."""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            rewards = outputs["rewards"]
            
        return rewards
    
    def get_sequence_reward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get the reward for the entire sequence."""
        rewards = self.get_reward(input_ids, attention_mask)
        
        # Take the reward at the last token of each sequence
        if attention_mask is not None:
            # Find the last token position for each sequence
            last_token_positions = attention_mask.sum(dim=1) - 1
            batch_size = rewards.shape[0]
            sequence_rewards = rewards[torch.arange(batch_size), last_token_positions]
        else:
            # Take the last token reward
            sequence_rewards = rewards[:, -1]
            
        return sequence_rewards
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> 'RewardModel':
        """Load a reward model from a pretrained checkpoint."""
        model = AutoModel.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        
        instance = cls.__new__(cls)
        instance.model = model
        instance.tokenizer = tokenizer
        instance.device = next(model.parameters()).device
        
        # Initialize reward head
        instance.reward_head = nn.Linear(model.config.hidden_size, 1)
        
        return instance 
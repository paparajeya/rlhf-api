"""
A2C (Advantage Actor-Critic) implementation for RLHF.
"""

from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from ..models import BaseModel
from ..config import BaseConfig
from ..data import PreferenceDataset


class A2C:
    """A2C algorithm for RLHF training."""
    
    def __init__(
        self,
        policy_model: BaseModel,
        value_model: BaseModel,
        ref_model: BaseModel,
        reward_model: BaseModel,
        config: BaseConfig,
    ):
        self.policy_model = policy_model
        self.value_model = value_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.config = config
        
        # Setup optimizers
        self.policy_optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        self.value_optimizer = torch.optim.AdamW(
            self.value_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Setup schedulers
        self.policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.policy_optimizer, T_max=config.warmup_steps
        )
        
        self.value_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.value_optimizer, T_max=config.warmup_steps
        )
        
        # Move models to device
        self.device = torch.device(config.device)
        self.policy_model.to(self.device)
        self.value_model.to(self.device)
        self.ref_model.to(self.device)
        self.reward_model.to(self.device)
        
        # Set models to eval mode
        self.ref_model.eval()
        self.reward_model.eval()
        
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        """Compute advantages for A2C."""
        advantages = torch.zeros_like(rewards)
        
        # Compute advantages from the end
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = delta
            
        return advantages
    
    def compute_entropy_loss(
        self,
        logprobs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute entropy loss for exploration."""
        return -torch.mean(logprobs)
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single training step for A2C."""
        # Generate responses using policy model
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        # Generate responses
        with torch.no_grad():
            policy_outputs = self.policy_model.generate(
                input_ids=input_ids,
                max_length=self.config.max_length,
                temperature=1.0,
                do_sample=True,
                pad_token_id=self.policy_model.tokenizer.pad_token_id,
                eos_token_id=self.policy_model.tokenizer.eos_token_id,
            )
            
            ref_outputs = self.ref_model.generate(
                input_ids=input_ids,
                max_length=self.config.max_length,
                temperature=1.0,
                do_sample=True,
                pad_token_id=self.ref_model.tokenizer.pad_token_id,
                eos_token_id=self.ref_model.tokenizer.eos_token_id,
            )
        
        # Get log probabilities
        policy_logprobs = self.policy_model.get_logprobs(policy_outputs)
        ref_logprobs = self.ref_model.get_logprobs(ref_outputs)
        
        # Get rewards
        policy_rewards = self.reward_model.get_sequence_reward(policy_outputs)
        ref_rewards = self.reward_model.get_sequence_reward(ref_outputs)
        
        # Get values
        policy_values = self.value_model.get_value(policy_outputs)
        
        # Compute advantages
        advantages = self.compute_advantages(policy_rewards, policy_values)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute policy loss (actor loss)
        policy_loss = -(policy_logprobs * advantages).mean()
        
        # Compute value loss (critic loss)
        value_loss = F.mse_loss(policy_values, policy_rewards)
        
        # Compute entropy loss for exploration
        entropy_loss = self.compute_entropy_loss(policy_logprobs)
        
        # Total loss
        total_loss = (
            policy_loss +
            0.5 * value_loss +
            0.01 * entropy_loss
        )
        
        # Backward pass
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(), self.config.max_grad_norm
        )
        torch.nn.utils.clip_grad_norm_(
            self.value_model.parameters(), self.config.max_grad_norm
        )
        
        # Optimizer step
        self.policy_optimizer.step()
        self.value_optimizer.step()
        
        # Scheduler step
        self.policy_scheduler.step()
        self.value_scheduler.step()
        
        # Compute KL divergence
        kl_div = F.kl_div(
            F.log_softmax(policy_logprobs, dim=-1),
            F.softmax(ref_logprobs, dim=-1),
            reduction='batchmean'
        )
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
            "kl_div": kl_div.item(),
            "mean_reward": policy_rewards.mean().item(),
            "mean_advantage": advantages.mean().item(),
        }
    
    def train(
        self,
        dataset: PreferenceDataset,
        num_epochs: int = 1,
    ) -> List[Dict[str, float]]:
        """Train the model using A2C."""
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
        )
        
        history = []
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
                # Training step
                losses = self.train_step(batch)
                epoch_losses.append(losses)
                
                # Log progress
                if len(epoch_losses) % self.config.logging_steps == 0:
                    avg_losses = {
                        k: np.mean([l[k] for l in epoch_losses[-self.config.logging_steps:]])
                        for k in epoch_losses[0].keys()
                    }
                    print(f"Step {len(epoch_losses)}: {avg_losses}")
            
            history.extend(epoch_losses)
            
        return history
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "policy_model_state_dict": self.policy_model.state_dict(),
            "value_model_state_dict": self.value_model.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
            "config": self.config,
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_model.load_state_dict(checkpoint["policy_model_state_dict"])
        self.value_model.load_state_dict(checkpoint["value_model_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"]) 
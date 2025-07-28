"""
PPO (Proximal Policy Optimization) implementation for RLHF.
"""

from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from ..models import BaseModel
from ..config import PPOConfig
from ..data import PreferenceDataset


class PPO:
    """PPO algorithm for RLHF training."""
    
    def __init__(
        self,
        policy_model: BaseModel,
        value_model: BaseModel,
        ref_model: BaseModel,
        reward_model: BaseModel,
        config: PPOConfig,
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
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages using GAE."""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute advantages from the end
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            
        return advantages, returns
    
    def compute_kl_divergence(
        self,
        policy_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence between policy and reference model."""
        return F.kl_div(
            F.log_softmax(policy_logprobs, dim=-1),
            F.softmax(ref_logprobs, dim=-1),
            reduction='batchmean'
        )
    
    def clip_ratio(
        self,
        ratio: torch.Tensor,
        clip_ratio: float = 0.2,
    ) -> torch.Tensor:
        """Clip the ratio for PPO."""
        return torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single training step for PPO."""
        # Generate responses using policy model
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        # Generate responses
        with torch.no_grad():
            policy_outputs = self.policy_model.generate(
                input_ids=input_ids,
                max_length=self.config.max_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=self.config.do_sample,
                pad_token_id=self.policy_model.tokenizer.pad_token_id,
                eos_token_id=self.policy_model.tokenizer.eos_token_id,
            )
            
            ref_outputs = self.ref_model.generate(
                input_ids=input_ids,
                max_length=self.config.max_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=self.config.do_sample,
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
        advantages, returns = self.compute_advantages(
            policy_rewards, policy_values, self.config.gamma, self.config.gae_lambda
        )
        
        # Normalize advantages
        if self.config.use_score_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute ratio
        ratio = torch.exp(policy_logprobs - ref_logprobs)
        
        # Compute clipped ratio
        clipped_ratio = self.clip_ratio(ratio, self.config.clip_ratio)
        
        # Compute policy loss
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()
        
        # Compute value loss
        value_loss = F.mse_loss(policy_values, returns)
        if self.config.clip_value_loss:
            value_loss = torch.clamp(value_loss, 0, 1.0)
        
        # Compute entropy loss
        entropy_loss = -torch.mean(policy_logprobs)
        
        # Total loss
        total_loss = (
            policy_loss +
            self.config.value_loss_coef * value_loss +
            self.config.entropy_coef * entropy_loss
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
        kl_div = self.compute_kl_divergence(policy_logprobs, ref_logprobs)
        
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
        """Train the model using PPO."""
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
                # Check KL divergence for early stopping
                if self.config.target_kl_early_stopping:
                    with torch.no_grad():
                        # Quick KL check
                        policy_logprobs = self.policy_model.get_logprobs(
                            batch["input_ids"].to(self.device)
                        )
                        ref_logprobs = self.ref_model.get_logprobs(
                            batch["input_ids"].to(self.device)
                        )
                        kl_div = self.compute_kl_divergence(policy_logprobs, ref_logprobs)
                        
                        if kl_div > self.config.target_kl:
                            print(f"Early stopping due to KL divergence: {kl_div:.4f}")
                            break
                
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
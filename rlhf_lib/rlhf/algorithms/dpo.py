"""
DPO (Direct Preference Optimization) implementation for RLHF.
"""

from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from ..models import BaseModel
from ..config import DPOConfig
from ..data import PreferenceDataset


class DPO:
    """DPO algorithm for RLHF training."""
    
    def __init__(
        self,
        policy_model: BaseModel,
        ref_model: BaseModel,
        config: DPOConfig,
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.config = config
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.warmup_steps
        )
        
        # Move models to device
        self.device = torch.device(config.device)
        self.policy_model.to(self.device)
        self.ref_model.to(self.device)
        
        # Set reference model to eval mode
        self.ref_model.eval()
        
    def compute_dpo_loss(
        self,
        policy_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        rewards: torch.Tensor,
        beta: float = 0.1,
    ) -> torch.Tensor:
        """Compute DPO loss."""
        # Compute log ratios
        policy_log_ratio = policy_logprobs - ref_logprobs
        ref_log_ratio = ref_logprobs - policy_logprobs
        
        # Compute rewards
        policy_rewards = rewards
        ref_rewards = -rewards  # Opposite preference
        
        # Compute DPO loss
        losses = -F.logsigmoid(
            beta * (policy_log_ratio - ref_log_ratio) * (policy_rewards - ref_rewards)
        )
        
        return losses.mean()
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single training step for DPO."""
        # Get preferred and dispreferred responses
        preferred_ids = batch["preferred_ids"].to(self.device)
        dispreferred_ids = batch["dispreferred_ids"].to(self.device)
        preferred_mask = batch["preferred_mask"].to(self.device)
        dispreferred_mask = batch["dispreferred_mask"].to(self.device)
        
        # Get log probabilities from policy model
        policy_preferred_logprobs = self.policy_model.get_logprobs(
            preferred_ids, preferred_mask
        )
        policy_dispreferred_logprobs = self.policy_model.get_logprobs(
            dispreferred_ids, dispreferred_mask
        )
        
        # Get log probabilities from reference model
        with torch.no_grad():
            ref_preferred_logprobs = self.ref_model.get_logprobs(
                preferred_ids, preferred_mask
            )
            ref_dispreferred_logprobs = self.ref_model.get_logprobs(
                dispreferred_ids, dispreferred_mask
            )
        
        # Compute rewards (simple preference signal)
        preferred_rewards = torch.ones(preferred_ids.shape[0], device=self.device)
        dispreferred_rewards = torch.zeros(dispreferred_ids.shape[0], device=self.device)
        
        # Compute DPO loss
        loss = self.compute_dpo_loss(
            policy_preferred_logprobs,
            ref_preferred_logprobs,
            preferred_rewards,
            self.config.beta,
        ) + self.compute_dpo_loss(
            policy_dispreferred_logprobs,
            ref_dispreferred_logprobs,
            dispreferred_rewards,
            self.config.beta,
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(), self.config.max_grad_norm
        )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        # Compute additional metrics
        with torch.no_grad():
            # KL divergence between policy and reference
            kl_div = F.kl_div(
                F.log_softmax(policy_preferred_logprobs, dim=-1),
                F.softmax(ref_preferred_logprobs, dim=-1),
                reduction='batchmean'
            )
            
            # Preference accuracy (simplified)
            policy_pref_score = policy_preferred_logprobs.mean()
            policy_dispref_score = policy_dispreferred_logprobs.mean()
            preference_accuracy = (policy_pref_score > policy_dispref_score).float().mean()
        
        return {
            "loss": loss.item(),
            "kl_div": kl_div.item(),
            "preference_accuracy": preference_accuracy.item(),
            "policy_pref_score": policy_pref_score.item(),
            "policy_dispref_score": policy_dispref_score.item(),
        }
    
    def train(
        self,
        dataset: PreferenceDataset,
        num_epochs: int = 1,
    ) -> List[Dict[str, float]]:
        """Train the model using DPO."""
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
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_model.load_state_dict(checkpoint["policy_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) 
"""
Main RLHF trainer class.
"""

from typing import Dict, Any, Optional, List, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import os
from pathlib import Path

from .config import RLHFConfig, PPOConfig, DPOConfig
from .models import BaseModel, GPT2Policy, GPT2Value, LLMPolicy, LLMValue
from .algorithms import PPO, DPO, A2C
from .data import PreferenceDataset
from .utils import set_seed, get_device, setup_logging


class RLHFTrainer:
    """Main trainer class for RLHF training."""
    
    def __init__(
        self,
        policy_model: BaseModel,
        value_model: Optional[BaseModel] = None,
        ref_model: Optional[BaseModel] = None,
        reward_model: Optional[BaseModel] = None,
        config: RLHFConfig = None,
    ):
        self.policy_model = policy_model
        self.value_model = value_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.config = config or RLHFConfig()
        
        # Setup logging
        setup_logging()
        
        # Set seed for reproducibility
        set_seed(self.config.seed)
        
        # Setup algorithm
        self.algorithm = self._setup_algorithm()
        
        # Setup wandb if enabled
        if "wandb" in self.config.report_to:
            self._setup_wandb()
    
    def _setup_algorithm(self):
        """Setup the RLHF algorithm based on configuration."""
        if self.config.algorithm == "ppo":
            if not all([self.value_model, self.ref_model, self.reward_model]):
                raise ValueError("PPO requires value_model, ref_model, and reward_model")
            
            return PPO(
                policy_model=self.policy_model,
                value_model=self.value_model,
                ref_model=self.ref_model,
                reward_model=self.reward_model,
                config=self.config.ppo_config,
            )
        
        elif self.config.algorithm == "dpo":
            if not self.ref_model:
                raise ValueError("DPO requires ref_model")
            
            return DPO(
                policy_model=self.policy_model,
                ref_model=self.ref_model,
                config=self.config.dpo_config,
            )
        
        elif self.config.algorithm == "a2c":
            if not all([self.value_model, self.ref_model, self.reward_model]):
                raise ValueError("A2C requires value_model, ref_model, and reward_model")
            
            return A2C(
                policy_model=self.policy_model,
                value_model=self.value_model,
                ref_model=self.ref_model,
                reward_model=self.reward_model,
                config=self.config,
            )
        
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")
    
    def _setup_wandb(self):
        """Setup wandb logging."""
        wandb.init(
            project="rlhf-training",
            name=self.config.run_name,
            config=self.config.__dict__,
        )
    
    def train(
        self,
        train_dataset: PreferenceDataset,
        val_dataset: Optional[PreferenceDataset] = None,
        num_epochs: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Train the model using the selected algorithm."""
        if num_epochs is None:
            num_epochs = self.config.num_train_epochs
        
        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
        }
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.algorithm.train(train_dataset, num_epochs=1)
            history["train_metrics"].extend(train_metrics)
            
            # Average training loss
            avg_train_loss = np.mean([m.get("total_loss", m.get("loss", 0)) for m in train_metrics])
            history["train_loss"].append(avg_train_loss)
            
            # Validation
            if val_dataset is not None:
                val_metrics = self.evaluate(val_dataset)
                history["val_metrics"].extend(val_metrics)
                
                avg_val_loss = np.mean([m.get("total_loss", m.get("loss", 0)) for m in val_metrics])
                history["val_loss"].append(avg_val_loss)
                
                print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            else:
                print(f"Train Loss: {avg_train_loss:.4f}")
            
            # Log to wandb
            if "wandb" in self.config.report_to:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss if val_dataset else None,
                })
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_steps == 0:
                self.save_checkpoint(f"{self.config.output_dir}/checkpoint_epoch_{epoch + 1}.pt")
        
        # Save final model
        self.save_checkpoint(f"{self.config.output_dir}/final_model.pt")
        
        return history
    
    def evaluate(
        self,
        dataset: PreferenceDataset,
    ) -> List[Dict[str, float]]:
        """Evaluate the model on a dataset."""
        self.policy_model.eval()
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=4,
        )
        
        metrics = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Get model outputs
                input_ids = batch["input_ids"].to(self.policy_model.device)
                attention_mask = batch["attention_mask"].to(self.policy_model.device)
                
                # Generate responses
                outputs = self.policy_model.generate(
                    input_ids=input_ids,
                    max_length=self.config.max_length,
                    temperature=1.0,
                    do_sample=True,
                    pad_token_id=self.policy_model.tokenizer.pad_token_id,
                    eos_token_id=self.policy_model.tokenizer.eos_token_id,
                )
                
                # Compute metrics
                batch_metrics = self._compute_metrics(batch, outputs)
                metrics.append(batch_metrics)
        
        self.policy_model.train()
        return metrics
    
    def _compute_metrics(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        # Decode outputs
        decoded_outputs = self.policy_model.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        
        # Simple metrics (can be extended)
        metrics = {
            "output_length": np.mean([len(output) for output in decoded_outputs]),
            "num_outputs": len(decoded_outputs),
        }
        
        return metrics
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save using algorithm's save method
        self.algorithm.save_checkpoint(path)
        
        # Also save configuration
        config_path = path.replace(".pt", "_config.json")
        import json
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        self.algorithm.load_checkpoint(path)
    
    def generate(
        self,
        prompts: List[str],
        max_length: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> List[str]:
        """Generate responses for given prompts."""
        self.policy_model.eval()
        
        # Tokenize prompts
        encodings = self.policy_model.tokenizer(
            prompts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = encodings["input_ids"].to(self.policy_model.device)
        attention_mask = encodings["attention_mask"].to(self.policy_model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.policy_model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.policy_model.tokenizer.pad_token_id,
                eos_token_id=self.policy_model.tokenizer.eos_token_id,
            )
        
        # Decode outputs
        responses = self.policy_model.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        
        self.policy_model.train()
        return responses
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[RLHFConfig] = None,
    ) -> 'RLHFTrainer':
        """Load a trainer from a pretrained checkpoint."""
        # Load configuration
        config_path = model_path.replace(".pt", "_config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            if config is None:
                config = RLHFConfig(**config_dict)
        
        # Load models
        policy_model = GPT2Policy.from_pretrained(model_path)
        
        # Load other models if they exist
        value_model = None
        ref_model = None
        reward_model = None
        
        # This is a simplified version - in practice, you'd need to load all models
        # based on the algorithm and configuration
        
        return cls(
            policy_model=policy_model,
            value_model=value_model,
            ref_model=ref_model,
            reward_model=reward_model,
            config=config,
        ) 
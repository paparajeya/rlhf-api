"""
Configuration classes for RLHF training.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import torch


@dataclass
class BaseConfig:
    """Base configuration for RLHF training."""
    
    # Model configuration
    model_name: str = "gpt2"
    max_length: int = 512
    use_fast_tokenizer: bool = True
    
    # Training configuration
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # Device configuration
    device: str = "auto"  # auto, cpu, cuda, mps
    mixed_precision: bool = True
    bf16: bool = False
    
    # Logging and saving
    output_dir: str = "./outputs"
    save_steps: int = 1000
    logging_steps: int = 10
    eval_steps: int = 500
    
    # Evaluation
    eval_batch_size: int = 4
    num_eval_samples: int = 100
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class PPOConfig(BaseConfig):
    """Configuration for PPO training."""
    
    # PPO specific parameters
    target_kl: float = 0.1
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    clip_value_loss: bool = True
    
    # Reward model configuration
    reward_model_name: Optional[str] = None
    reward_model_path: Optional[str] = None
    
    # Reference model configuration
    ref_model_name: Optional[str] = None
    ref_model_path: Optional[str] = None
    
    # Training loop configuration
    ppo_epochs: int = 4
    target_kl_early_stopping: bool = True
    use_score_scaling: bool = True
    use_score_norm: bool = True
    
    # Generation configuration
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0


@dataclass
class DPOConfig(BaseConfig):
    """Configuration for DPO training."""
    
    # DPO specific parameters
    beta: float = 0.1
    max_prompt_length: int = 512
    max_length: int = 1024
    
    # Training loop configuration
    dpo_epochs: int = 1
    gradient_checkpointing: bool = True
    
    # Generation configuration
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0


@dataclass
class RLHFConfig(BaseConfig):
    """General RLHF configuration that can be used for any algorithm."""
    
    # Algorithm selection
    algorithm: str = "ppo"  # ppo, dpo, a2c
    
    # Algorithm specific configs
    ppo_config: Optional[PPOConfig] = None
    dpo_config: Optional[DPOConfig] = None
    
    # Data configuration
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    
    # Model paths
    base_model_path: Optional[str] = None
    reward_model_path: Optional[str] = None
    ref_model_path: Optional[str] = None
    
    # Training configuration
    num_train_epochs: int = 3
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    
    # Evaluation configuration
    evaluation_strategy: str = "steps"  # no, steps, epoch
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Logging configuration
    logging_dir: Optional[str] = None
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    run_name: Optional[str] = None
    
    # Seed for reproducibility
    seed: int = 42
    
    def __post_init__(self):
        super().__post_init__()
        
        if self.logging_dir is None:
            self.logging_dir = f"{self.output_dir}/logs"
        
        if self.run_name is None:
            self.run_name = f"rlhf_{self.algorithm}_{self.model_name}"
        
        # Set up algorithm specific configs
        if self.algorithm == "ppo" and self.ppo_config is None:
            self.ppo_config = PPOConfig()
        elif self.algorithm == "dpo" and self.dpo_config is None:
            self.dpo_config = DPOConfig()


@dataclass
class InferenceConfig:
    """Configuration for model inference."""
    
    model_path: str
    max_length: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True
    num_return_sequences: int = 1
    device: str = "auto"
    batch_size: int = 1
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    model_path: str
    test_file: str
    metrics: List[str] = field(default_factory=lambda: ["bleu", "rouge", "bertscore"])
    max_length: int = 512
    batch_size: int = 4
    device: str = "auto"
    output_file: Optional[str] = None
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu" 
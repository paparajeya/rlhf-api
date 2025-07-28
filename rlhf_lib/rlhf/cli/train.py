"""
CLI training function for RLHF.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from ..trainer import RLHFTrainer
from ..config import RLHFConfig, PPOConfig, DPOConfig
from ..models import GPT2Policy, GPT2Value, LLMPolicy, LLMValue
from ..data import PreferenceDataset


def train(
    config_path: Optional[str] = None,
    model_name: str = "gpt2",
    algorithm: str = "ppo",
    train_file: Optional[str] = None,
    val_file: Optional[str] = None,
    output_dir: str = "./outputs",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    **kwargs
) -> None:
    """Train an RLHF model."""
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = RLHFConfig(**config_dict)
    else:
        # Create default configuration
        config = RLHFConfig(
            algorithm=algorithm,
            model_name=model_name,
            train_file=train_file,
            validation_file=val_file,
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            **kwargs
        )
    
    # Initialize models
    print("Initializing models...")
    policy_model = GPT2Policy(model_name)
    value_model = GPT2Value(model_name)
    ref_model = GPT2Policy(model_name)
    reward_model = GPT2Value(model_name)  # Simplified reward model
    
    # Create trainer
    trainer = RLHFTrainer(
        policy_model=policy_model,
        value_model=value_model,
        ref_model=ref_model,
        reward_model=reward_model,
        config=config,
    )
    
    # Load dataset
    print("Loading dataset...")
    if train_file and os.path.exists(train_file):
        train_dataset = PreferenceDataset.from_json(
            train_file, policy_model.tokenizer, config.max_length
        )
    else:
        # Create dummy dataset for demonstration
        dummy_data = [
            {
                "prompt": "What is the capital of France?",
                "preferred": "The capital of France is Paris.",
                "dispreferred": "I don't know.",
            },
            {
                "prompt": "How do you make coffee?",
                "preferred": "To make coffee, you need to grind beans and brew them with hot water.",
                "dispreferred": "Just add water.",
            },
        ]
        train_dataset = PreferenceDataset(dummy_data, policy_model.tokenizer, config.max_length)
    
    # Load validation dataset
    val_dataset = None
    if val_file and os.path.exists(val_file):
        val_dataset = PreferenceDataset.from_json(
            val_file, policy_model.tokenizer, config.max_length
        )
    
    # Train
    print("Starting training...")
    history = trainer.train(train_dataset, val_dataset, num_epochs)
    
    print("Training completed!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if val_dataset:
        print(f"Final val loss: {history['val_loss'][-1]:.4f}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Train RLHF model")
    
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--model-name", type=str, default="gpt2", help="Model name")
    parser.add_argument("--algorithm", type=str, default="ppo", choices=["ppo", "dpo", "a2c"], help="RLHF algorithm")
    parser.add_argument("--train-file", type=str, help="Path to training data file")
    parser.add_argument("--val-file", type=str, help="Path to validation data file")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    train(
        config_path=args.config,
        model_name=args.model_name,
        algorithm=args.algorithm,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main() 
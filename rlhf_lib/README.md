# RLHF Library

A comprehensive Python library for Reinforcement Learning from Human Feedback (RLHF).

## Features

- **PPO Algorithm**: Proximal Policy Optimization implementation
- **DPO Algorithm**: Direct Preference Optimization implementation  
- **A2C Algorithm**: Advantage Actor-Critic implementation
- **Model Support**: GPT-2, GPT-3, and other transformer models
- **Data Processing**: Preference dataset handling
- **Training Utilities**: Comprehensive training pipeline

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from rlhf import RLHFTrainer, PPOConfig
from rlhf.models import GPT2Policy, GPT2Value
from rlhf.data import PreferenceDataset

# Initialize models
policy_model = GPT2Policy.from_pretrained("gpt2")
value_model = GPT2Value.from_pretrained("gpt2")

# Configure training
config = PPOConfig(
    learning_rate=1e-5,
    batch_size=4,
    max_grad_norm=1.0,
    target_kl=0.1
)

# Create trainer
trainer = RLHFTrainer(
    policy_model=policy_model,
    value_model=value_model,
    config=config
)

# Train the model
trainer.train(dataset, epochs=10)
```

## Documentation

See the main project README for complete documentation. 
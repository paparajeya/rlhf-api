"""
Utility functions for RLHF library.
"""

import torch
import random
import numpy as np
import logging
import os
from typing import Optional, Dict, Any
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior for CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for training."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> None:
    """Setup logging configuration."""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_by_layer(model: torch.nn.Module) -> Dict[str, int]:
    """Count parameters by layer name."""
    param_counts = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_counts[name] = param.numel()
    return param_counts


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def save_model_info(
    model: torch.nn.Module,
    save_path: str,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """Save model information to a file."""
    info = {
        "total_parameters": count_parameters(model),
        "parameters_by_layer": count_parameters_by_layer(model),
        "model_size_mb": get_model_size_mb(model),
    }
    
    if additional_info:
        info.update(additional_info)
    
    # Create directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=2)


def load_model_info(load_path: str) -> Dict[str, Any]:
    """Load model information from a file."""
    import json
    with open(load_path, 'r') as f:
        return json.load(f)


def create_output_directory(output_dir: str) -> Path:
    """Create output directory and return path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_config(config: Any, save_path: str) -> None:
    """Save configuration to a file."""
    # Create directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(save_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)


def load_config(load_path: str, config_class: type) -> Any:
    """Load configuration from a file."""
    import json
    with open(load_path, 'r') as f:
        config_dict = json.load(f)
    
    return config_class(**config_dict)


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


def format_number(num: float) -> str:
    """Format number to human readable string."""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}"


def compute_grad_norm(model: torch.nn.Module) -> float:
    """Compute gradient norm of a model."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def clip_gradients(
    model: torch.nn.Module,
    max_norm: float = 1.0,
    norm_type: float = 2.0
) -> float:
    """Clip gradients and return the total norm."""
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm, norm_type=norm_type
    )


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    num_training_steps: int = 1000,
    num_warmup_steps: int = 100,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler."""
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_training_steps, **kwargs
        )
    elif scheduler_type == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps
        )
    elif scheduler_type == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.95, **kwargs
        )
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=100, gamma=0.9, **kwargs
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    **kwargs
) -> torch.optim.Optimizer:
    """Get optimizer for a model."""
    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    metric_names: list = None
) -> Dict[str, float]:
    """Compute various metrics."""
    if metric_names is None:
        metric_names = ["accuracy", "precision", "recall", "f1"]
    
    metrics = {}
    
    if "accuracy" in metric_names:
        correct = (predictions == targets).sum().item()
        total = targets.numel()
        metrics["accuracy"] = correct / total if total > 0 else 0.0
    
    if "precision" in metric_names:
        # Binary precision
        tp = ((predictions == 1) & (targets == 1)).sum().item()
        fp = ((predictions == 1) & (targets == 0)).sum().item()
        metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    if "recall" in metric_names:
        # Binary recall
        tp = ((predictions == 1) & (targets == 1)).sum().item()
        fn = ((predictions == 0) & (targets == 1)).sum().item()
        metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if "f1" in metric_names:
        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        metrics["f1"] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return metrics 
"""
CLI evaluation function for RLHF.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from ..trainer import RLHFTrainer
from ..config import RLHFConfig
from ..models import GPT2Policy
from ..data import PreferenceDataset


def evaluate(
    model_path: str,
    test_file: str,
    output_file: Optional[str] = None,
    batch_size: int = 4,
    max_length: int = 512,
    **kwargs
) -> None:
    """Evaluate an RLHF model."""
    
    # Load model
    print(f"Loading model from {model_path}...")
    trainer = RLHFTrainer.from_pretrained(model_path)
    
    # Load test dataset
    print(f"Loading test dataset from {test_file}...")
    test_dataset = PreferenceDataset.from_json(
        test_file, trainer.policy_model.tokenizer, max_length
    )
    
    # Evaluate
    print("Evaluating model...")
    metrics = trainer.evaluate(test_dataset)
    
    # Aggregate metrics
    aggregated_metrics = {}
    for metric_name in metrics[0].keys():
        values = [m[metric_name] for m in metrics]
        aggregated_metrics[f"mean_{metric_name}"] = sum(values) / len(values)
        aggregated_metrics[f"std_{metric_name}"] = (sum((x - aggregated_metrics[f"mean_{metric_name}"]) ** 2 for x in values) / len(values)) ** 0.5
    
    # Print results
    print("\nEvaluation Results:")
    for metric_name, value in aggregated_metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(aggregated_metrics, f, indent=2)
        print(f"\nResults saved to {output_file}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Evaluate RLHF model")
    
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test-file", type=str, required=True, help="Path to test data file")
    parser.add_argument("--output-file", type=str, help="Path to save evaluation results")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model_path,
        test_file=args.test_file,
        output_file=args.output_file,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main() 
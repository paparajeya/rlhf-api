"""
CLI data collection function for RLHF.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

from ..models import GPT2Policy
from ..data import PreferenceDataset


def collect(
    output_file: str,
    prompts: Optional[List[str]] = None,
    num_samples: int = 100,
    model_name: str = "gpt2",
    max_length: int = 512,
    **kwargs
) -> None:
    """Collect preference data for RLHF training."""
    
    # Initialize model for generation
    print(f"Loading model {model_name}...")
    model = GPT2Policy(model_name)
    
    # Generate prompts if not provided
    if prompts is None:
        prompts = [
            "What is the capital of France?",
            "How do you make coffee?",
            "Explain quantum computing.",
            "What are the benefits of exercise?",
            "How does photosynthesis work?",
            "What is machine learning?",
            "Explain the water cycle.",
            "What causes climate change?",
            "How do vaccines work?",
            "What is artificial intelligence?",
        ]
    
    # Generate responses for each prompt
    print("Generating responses...")
    data = []
    
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        # Generate multiple responses for each prompt
        responses = model.generate(
            [prompt] * 3,  # Generate 3 responses per prompt
            max_length=max_length,
            temperature=0.8,
            do_sample=True,
        )
        
        # Create preference pairs (simplified - in practice, you'd collect human preferences)
        for j in range(len(responses) - 1):
            for k in range(j + 1, len(responses)):
                # Simple heuristic: longer responses are preferred (this is just for demonstration)
                if len(responses[j]) > len(responses[k]):
                    preferred = responses[j]
                    dispreferred = responses[k]
                else:
                    preferred = responses[k]
                    dispreferred = responses[j]
                
                data.append({
                    "prompt": prompt,
                    "preferred": preferred,
                    "dispreferred": dispreferred,
                })
    
    # Save data
    print(f"Saving {len(data)} preference pairs to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Data collection completed! Saved {len(data)} preference pairs.")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Collect preference data for RLHF")
    
    parser.add_argument("--output-file", type=str, required=True, help="Path to save collected data")
    parser.add_argument("--prompts-file", type=str, help="Path to file containing prompts")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--model-name", type=str, default="gpt2", help="Model name for generation")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Load prompts from file if provided
    prompts = None
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    
    collect(
        output_file=args.output_file,
        prompts=prompts,
        num_samples=args.num_samples,
        model_name=args.model_name,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main() 
"""
Utility functions for data processing in RLHF.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
from transformers import PreTrainedTokenizer
import numpy as np


def tokenize_text(
    text: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    truncation: bool = True,
    padding: bool = True,
    return_tensors: str = "pt",
) -> Dict[str, torch.Tensor]:
    """Tokenize text using a tokenizer."""
    return tokenizer(
        text,
        max_length=max_length,
        truncation=truncation,
        padding=padding,
        return_tensors=return_tensors,
    )


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching."""
    # Get all keys from the first item
    keys = batch[0].keys()
    
    # Stack tensors for each key
    collated = {}
    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([item[key] for item in batch])
        else:
            collated[key] = [item[key] for item in batch]
    
    return collated


def create_preference_data(
    prompts: List[str],
    preferred_responses: List[str],
    dispreferred_responses: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
) -> List[Dict[str, Any]]:
    """Create preference data from lists."""
    data = []
    
    for prompt, preferred, dispreferred in zip(prompts, preferred_responses, dispreferred_responses):
        data.append({
            "prompt": prompt,
            "preferred": preferred,
            "dispreferred": dispreferred,
        })
    
    return data


def create_single_response_data(
    prompts: List[str],
    responses: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
) -> List[Dict[str, Any]]:
    """Create single response data from lists."""
    data = []
    
    for prompt, response in zip(prompts, responses):
        data.append({
            "prompt": prompt,
            "response": response,
        })
    
    return data


def pad_sequences(
    sequences: List[torch.Tensor],
    padding_value: int = 0,
    max_length: Optional[int] = None,
) -> torch.Tensor:
    """Pad sequences to the same length."""
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            padding = torch.full((max_length - len(seq),), padding_value, dtype=seq.dtype)
            padded_seq = torch.cat([seq, padding])
        else:
            padded_seq = seq[:max_length]
        padded_sequences.append(padded_seq)
    
    return torch.stack(padded_sequences)


def create_attention_mask(
    input_ids: torch.Tensor,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """Create attention mask from input ids."""
    return (input_ids != pad_token_id).long()


def truncate_sequences(
    sequences: List[torch.Tensor],
    max_length: int,
) -> List[torch.Tensor]:
    """Truncate sequences to maximum length."""
    return [seq[:max_length] for seq in sequences]


def create_batch(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    padding: bool = True,
    truncation: bool = True,
) -> Dict[str, torch.Tensor]:
    """Create a batch from a list of texts."""
    encodings = tokenizer(
        texts,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors="pt",
    )
    
    return encodings


def create_preference_batch(
    prompts: List[str],
    preferred_responses: List[str],
    dispreferred_responses: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
) -> Dict[str, torch.Tensor]:
    """Create a batch for preference learning."""
    # Tokenize prompts
    prompt_encodings = tokenizer(
        prompts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    
    # Tokenize preferred responses
    preferred_encodings = tokenizer(
        preferred_responses,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    
    # Tokenize dispreferred responses
    dispreferred_encodings = tokenizer(
        dispreferred_responses,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    
    return {
        "input_ids": prompt_encodings["input_ids"],
        "attention_mask": prompt_encodings["attention_mask"],
        "preferred_ids": preferred_encodings["input_ids"],
        "preferred_mask": preferred_encodings["attention_mask"],
        "dispreferred_ids": dispreferred_encodings["input_ids"],
        "dispreferred_mask": dispreferred_encodings["attention_mask"],
    } 
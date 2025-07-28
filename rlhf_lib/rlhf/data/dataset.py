"""
Preference dataset for RLHF training.
"""

from typing import Dict, List, Optional, Tuple, Any
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
import numpy as np


class PreferenceDataset(Dataset):
    """Dataset for preference learning in RLHF."""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        truncation: bool = True,
        padding: bool = True,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Tokenize input
        input_text = item.get("prompt", "")
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors="pt",
        )
        
        # For preference data, we need preferred and dispreferred responses
        if "preferred" in item and "dispreferred" in item:
            preferred_text = item["preferred"]
            dispreferred_text = item["dispreferred"]
            
            # Tokenize preferred response
            preferred_encoding = self.tokenizer(
                preferred_text,
                max_length=self.max_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors="pt",
            )
            
            # Tokenize dispreferred response
            dispreferred_encoding = self.tokenizer(
                dispreferred_text,
                max_length=self.max_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors="pt",
            )
            
            return {
                "input_ids": input_encoding["input_ids"].squeeze(0),
                "attention_mask": input_encoding["attention_mask"].squeeze(0),
                "preferred_ids": preferred_encoding["input_ids"].squeeze(0),
                "preferred_mask": preferred_encoding["attention_mask"].squeeze(0),
                "dispreferred_ids": dispreferred_encoding["input_ids"].squeeze(0),
                "dispreferred_mask": dispreferred_encoding["attention_mask"].squeeze(0),
            }
        
        # For single response data
        else:
            response_text = item.get("response", "")
            response_encoding = self.tokenizer(
                response_text,
                max_length=self.max_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors="pt",
            )
            
            return {
                "input_ids": input_encoding["input_ids"].squeeze(0),
                "attention_mask": input_encoding["attention_mask"].squeeze(0),
                "response_ids": response_encoding["input_ids"].squeeze(0),
                "response_mask": response_encoding["attention_mask"].squeeze(0),
            }
    
    @classmethod
    def from_json(
        cls,
        file_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ) -> 'PreferenceDataset':
        """Load dataset from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(data, tokenizer, max_length)
    
    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        split: str = "train",
    ) -> 'PreferenceDataset':
        """Load dataset from HuggingFace datasets."""
        from datasets import load_dataset
        
        dataset = load_dataset(dataset_name, split=split)
        data = dataset.to_list()
        
        return cls(data, tokenizer, max_length)
    
    def save_to_json(self, file_path: str) -> None:
        """Save dataset to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
    ) -> Tuple['PreferenceDataset', 'PreferenceDataset', 'PreferenceDataset']:
        """Split dataset into train/val/test sets."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        np.random.seed(random_seed)
        indices = np.random.permutation(len(self.data))
        
        train_end = int(len(self.data) * train_ratio)
        val_end = train_end + int(len(self.data) * val_ratio)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        train_data = [self.data[i] for i in train_indices]
        val_data = [self.data[i] for i in val_indices]
        test_data = [self.data[i] for i in test_indices]
        
        train_dataset = PreferenceDataset(
            train_data, self.tokenizer, self.max_length
        )
        val_dataset = PreferenceDataset(
            val_data, self.tokenizer, self.max_length
        )
        test_dataset = PreferenceDataset(
            test_data, self.tokenizer, self.max_length
        )
        
        return train_dataset, val_dataset, test_dataset


class PreferenceDataLoader(DataLoader):
    """DataLoader for preference datasets."""
    
    def __init__(
        self,
        dataset: PreferenceDataset,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self._collate_fn,
        )
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for preference data."""
        # Check if this is preference data or single response data
        if "preferred_ids" in batch[0]:
            # Preference data
            return {
                "input_ids": torch.stack([item["input_ids"] for item in batch]),
                "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
                "preferred_ids": torch.stack([item["preferred_ids"] for item in batch]),
                "preferred_mask": torch.stack([item["preferred_mask"] for item in batch]),
                "dispreferred_ids": torch.stack([item["dispreferred_ids"] for item in batch]),
                "dispreferred_mask": torch.stack([item["dispreferred_mask"] for item in batch]),
            }
        else:
            # Single response data
            return {
                "input_ids": torch.stack([item["input_ids"] for item in batch]),
                "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
                "response_ids": torch.stack([item["response_ids"] for item in batch]),
                "response_mask": torch.stack([item["response_mask"] for item in batch]),
            } 
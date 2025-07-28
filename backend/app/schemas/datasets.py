"""
Pydantic schemas for datasets.
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime


class DatasetBase(BaseModel):
    """Base dataset schema."""
    name: str
    description: Optional[str] = None


class DatasetCreate(DatasetBase):
    """Schema for creating a dataset."""
    file_path: str
    file_size: int
    num_samples: int
    metadata: Optional[Dict[str, Any]] = None


class DatasetUpdate(BaseModel):
    """Schema for updating a dataset."""
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DatasetResponse(DatasetBase):
    """Schema for dataset response."""
    id: int
    file_size: int
    num_samples: int
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True 
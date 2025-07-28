"""
Pydantic schemas for models.
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime


class ModelBase(BaseModel):
    """Base model schema."""
    name: str
    description: Optional[str] = None
    model_type: str
    algorithm: str


class ModelCreate(ModelBase):
    """Schema for creating a model."""
    model_path: str
    config: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None


class ModelUpdate(BaseModel):
    """Schema for updating a model."""
    name: Optional[str] = None
    description: Optional[str] = None
    model_type: Optional[str] = None
    algorithm: Optional[str] = None
    is_active: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None


class ModelResponse(ModelBase):
    """Schema for model response."""
    id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    metrics: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class ModelListResponse(BaseModel):
    """Schema for model list response."""
    models: list[ModelResponse]
    total: int
    skip: int
    limit: int 
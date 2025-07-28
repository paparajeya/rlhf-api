"""
Pydantic schemas for training jobs.
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime


class TrainingJobBase(BaseModel):
    """Base training job schema."""
    name: str
    algorithm: str
    config: Dict[str, Any]


class TrainingJobCreate(TrainingJobBase):
    """Schema for creating a training job."""
    pass


class TrainingJobUpdate(BaseModel):
    """Schema for updating a training job."""
    name: Optional[str] = None
    algorithm: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    progress: Optional[float] = None
    logs: Optional[str] = None
    error_message: Optional[str] = None


class TrainingJobResponse(TrainingJobBase):
    """Schema for training job response."""
    id: int
    status: str
    progress: float
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    logs: Optional[str] = None

    class Config:
        from_attributes = True 
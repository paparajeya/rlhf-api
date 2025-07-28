"""
Pydantic schemas for evaluations.
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime


class EvaluationBase(BaseModel):
    """Base evaluation schema."""
    model_id: int
    dataset_id: int


class EvaluationCreate(EvaluationBase):
    """Schema for creating an evaluation."""
    pass


class EvaluationResponse(EvaluationBase):
    """Schema for evaluation response."""
    id: int
    model_name: str
    dataset_name: str
    metrics: Dict[str, Any]
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True 
"""
Pydantic schemas for inference.
"""

from pydantic import BaseModel, Field
from typing import Optional


class InferenceRequest(BaseModel):
    """Schema for inference request."""
    model_id: int
    prompt: str
    max_length: int = Field(default=100, ge=1, le=1000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class InferenceResponse(BaseModel):
    """Schema for inference response."""
    response: str
    model_name: str
    inference_time: float
    tokens_generated: int 
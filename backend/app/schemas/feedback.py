"""
Pydantic schemas for feedback.
"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class FeedbackItemResponse(BaseModel):
    """Schema for feedback item response."""
    id: int
    prompt: str
    response_a: str
    response_b: str
    created_at: datetime


class FeedbackSubmit(BaseModel):
    """Schema for submitting feedback."""
    item_id: int
    preferred_response: str  # 'A' or 'B'
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Schema for feedback response."""
    id: int
    prompt: str
    preferred_response: str
    dispreferred_response: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True 
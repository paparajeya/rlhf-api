"""
Feedback API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from datetime import datetime

from ...core.database import get_db, Feedback
from ...schemas.feedback import FeedbackItemResponse, FeedbackSubmit

router = APIRouter()


@router.get("/items", response_model=List[FeedbackItemResponse])
async def get_feedback_items(db: AsyncSession = Depends(get_db)):
    """Get feedback items for human evaluation."""
    # For now, return mock data. In a real implementation, this would
    # generate or retrieve actual feedback items from a dataset
    mock_items = [
        {
            "id": 1,
            "prompt": "Explain how photosynthesis works in simple terms.",
            "response_a": "Photosynthesis is the process where plants use sunlight to make food. They take in carbon dioxide and water, and with the help of sunlight, they create glucose (sugar) and oxygen.",
            "response_b": "Plants do this thing where they turn light into food. It's pretty cool because they can make their own food while we have to buy ours.",
            "created_at": datetime.utcnow()
        },
        {
            "id": 2,
            "prompt": "What are the benefits of regular exercise?",
            "response_a": "Regular exercise provides numerous health benefits including improved cardiovascular health, stronger muscles and bones, better mental health, weight management, and increased energy levels. It also helps reduce the risk of chronic diseases.",
            "response_b": "Exercise is good for you. It makes you stronger and helps you stay healthy. You should do it regularly.",
            "created_at": datetime.utcnow()
        },
        {
            "id": 3,
            "prompt": "How do you make a simple pasta dish?",
            "response_a": "To make a simple pasta dish, boil water in a large pot, add salt, and cook pasta according to package directions. Meanwhile, heat olive oil in a pan, add minced garlic, cook until fragrant, then add cooked pasta and toss with grated cheese and black pepper.",
            "response_b": "Just boil some pasta and put sauce on it. It's really easy and tastes good.",
            "created_at": datetime.utcnow()
        }
    ]
    
    return [
        FeedbackItemResponse(
            id=item["id"],
            prompt=item["prompt"],
            response_a=item["response_a"],
            response_b=item["response_b"],
            created_at=item["created_at"]
        )
        for item in mock_items
    ]


@router.post("/submit")
async def submit_feedback(
    feedback_data: FeedbackSubmit,
    db: AsyncSession = Depends(get_db)
):
    """Submit human feedback."""
    # Create feedback record
    feedback = Feedback(
        prompt="Mock prompt",  # In real implementation, get from feedback item
        preferred_response=feedback_data.preferred_response,
        dispreferred_response="Mock dispreferred response",  # In real implementation, get from feedback item
        user_id=feedback_data.user_id,
        session_id=feedback_data.session_id
    )
    
    db.add(feedback)
    await db.commit()
    await db.refresh(feedback)
    
    return {"message": "Feedback submitted successfully", "feedback_id": feedback.id}


@router.get("/stats")
async def get_feedback_stats(db: AsyncSession = Depends(get_db)):
    """Get feedback collection statistics."""
    # Get total feedback count
    result = await db.execute(select(Feedback))
    total_feedback = len(result.scalars().all())
    
    # Get feedback by session
    result = await db.execute(select(Feedback.session_id))
    sessions = result.scalars().all()
    unique_sessions = len(set(sessions))
    
    return {
        "total_feedback": total_feedback,
        "unique_sessions": unique_sessions,
        "average_feedback_per_session": total_feedback / unique_sessions if unique_sessions > 0 else 0
    } 
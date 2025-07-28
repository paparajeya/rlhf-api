"""
Inference API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import time
from typing import Dict, Any

from ...core.database import get_db, Model
from ...schemas.inference import InferenceRequest, InferenceResponse
from ...services.inference_service import InferenceService

router = APIRouter()


@router.post("/generate", response_model=InferenceResponse)
async def generate_response(
    request: InferenceRequest,
    db: AsyncSession = Depends(get_db)
):
    """Generate a response using a trained model."""
    # Check if model exists and is active
    result = await db.execute(select(Model).where(Model.id == request.model_id))
    model = result.scalar_one_or_none()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if not model.is_active:
        raise HTTPException(status_code=400, detail="Model is not active")
    
    # Start timing
    start_time = time.time()
    
    try:
        # Initialize inference service
        inference_service = InferenceService()
        
        # Generate response
        response_text = await inference_service.generate_response(
            model_path=model.model_path,
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        # Calculate timing and token count
        inference_time = time.time() - start_time
        tokens_generated = len(response_text.split())  # Simple token count
        
        return InferenceResponse(
            response=response_text,
            model_name=model.name,
            inference_time=inference_time,
            tokens_generated=tokens_generated
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.get("/models/active")
async def get_active_models(db: AsyncSession = Depends(get_db)):
    """Get all active models for inference."""
    result = await db.execute(select(Model).where(Model.is_active == True))
    models = result.scalars().all()
    
    return [
        {
            "id": model.id,
            "name": model.name,
            "model_type": model.model_type,
            "algorithm": model.algorithm
        }
        for model in models
    ] 
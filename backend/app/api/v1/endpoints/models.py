"""
Models API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
import os
import shutil
from datetime import datetime

from ...core.database import get_db, Model
from ...core.config import settings
from ...schemas.models import ModelCreate, ModelResponse, ModelUpdate

router = APIRouter()


@router.get("/", response_model=List[ModelResponse])
async def get_models(db: AsyncSession = Depends(get_db)):
    """Get all models."""
    result = await db.execute(select(Model))
    models = result.scalars().all()
    return [
        ModelResponse(
            id=model.id,
            name=model.name,
            description=model.description,
            model_type=model.model_type,
            algorithm=model.algorithm,
            is_active=model.is_active,
            created_at=model.created_at,
            metrics=model.metrics
        )
        for model in models
    ]


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model_id: int, db: AsyncSession = Depends(get_db)):
    """Get a specific model."""
    result = await db.execute(select(Model).where(Model.id == model_id))
    model = result.scalar_one_or_none()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return ModelResponse(
        id=model.id,
        name=model.name,
        description=model.description,
        model_type=model.model_type,
        algorithm=model.algorithm,
        is_active=model.is_active,
        created_at=model.created_at,
        metrics=model.metrics
    )


@router.post("/", response_model=ModelResponse)
async def create_model(
    name: str = Form(...),
    description: Optional[str] = Form(None),
    model_type: str = Form(...),
    algorithm: str = Form(...),
    model_file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """Upload and create a new model."""
    # Check if model with same name exists
    result = await db.execute(select(Model).where(Model.name == name))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Model with this name already exists")
    
    # Validate file type
    allowed_extensions = ['.pt', '.pth', '.bin', '.safetensors']
    file_extension = os.path.splitext(model_file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Create model directory
    model_dir = os.path.join(settings.MODEL_STORAGE_PATH, name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model file
    model_path = os.path.join(model_dir, model_file.filename)
    with open(model_path, "wb") as buffer:
        shutil.copyfileobj(model_file.file, buffer)
    
    # Create model record
    model = Model(
        name=name,
        description=description,
        model_path=model_path,
        model_type=model_type,
        algorithm=algorithm,
        is_active=True,
        config={},
        metrics={}
    )
    
    db.add(model)
    await db.commit()
    await db.refresh(model)
    
    return ModelResponse(
        id=model.id,
        name=model.name,
        description=model.description,
        model_type=model.model_type,
        algorithm=model.algorithm,
        is_active=model.is_active,
        created_at=model.created_at,
        metrics=model.metrics
    )


@router.put("/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: int,
    model_update: ModelUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update a model."""
    result = await db.execute(select(Model).where(Model.id == model_id))
    model = result.scalar_one_or_none()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Update fields
    for field, value in model_update.dict(exclude_unset=True).items():
        setattr(model, field, value)
    
    model.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(model)
    
    return ModelResponse(
        id=model.id,
        name=model.name,
        description=model.description,
        model_type=model.model_type,
        algorithm=model.algorithm,
        is_active=model.is_active,
        created_at=model.created_at,
        metrics=model.metrics
    )


@router.delete("/{model_id}")
async def delete_model(model_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a model."""
    result = await db.execute(select(Model).where(Model.id == model_id))
    model = result.scalar_one_or_none()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Delete model file
    if os.path.exists(model.model_path):
        os.remove(model.model_path)
    
    # Delete model directory if empty
    model_dir = os.path.dirname(model.model_path)
    if os.path.exists(model_dir) and not os.listdir(model_dir):
        os.rmdir(model_dir)
    
    await db.delete(model)
    await db.commit()
    
    return {"message": "Model deleted successfully"} 
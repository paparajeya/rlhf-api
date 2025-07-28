"""
Model service for handling model operations.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import os
import json
from pathlib import Path

from ..core.database import Model
from ..core.config import settings


class ModelService:
    """Service for model operations."""
    
    @staticmethod
    async def create_model(
        db: AsyncSession,
        name: str,
        model_path: str,
        model_type: str,
        algorithm: str,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Model:
        """Create a new model."""
        model = Model(
            name=name,
            description=description,
            model_path=model_path,
            model_type=model_type,
            algorithm=algorithm,
            config=config,
            metrics=metrics,
        )
        
        db.add(model)
        await db.commit()
        await db.refresh(model)
        
        return model
    
    @staticmethod
    async def get_model(db: AsyncSession, model_id: int) -> Optional[Model]:
        """Get a model by ID."""
        query = select(Model).where(Model.id == model_id)
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_models(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        model_type: Optional[str] = None,
        algorithm: Optional[str] = None,
    ) -> List[Model]:
        """Get models with optional filtering."""
        query = select(Model)
        
        if model_type:
            query = query.where(Model.model_type == model_type)
        
        if algorithm:
            query = query.where(Model.algorithm == algorithm)
        
        query = query.offset(skip).limit(limit)
        result = await db.execute(query)
        return result.scalars().all()
    
    @staticmethod
    async def update_model(
        db: AsyncSession,
        model_id: int,
        **kwargs
    ) -> Optional[Model]:
        """Update a model."""
        model = await ModelService.get_model(db, model_id)
        if not model:
            return None
        
        for key, value in kwargs.items():
            if hasattr(model, key):
                setattr(model, key, value)
        
        await db.commit()
        await db.refresh(model)
        
        return model
    
    @staticmethod
    async def delete_model(db: AsyncSession, model_id: int) -> bool:
        """Delete a model."""
        model = await ModelService.get_model(db, model_id)
        if not model:
            return False
        
        # Delete model file if it exists
        if os.path.exists(model.model_path):
            os.remove(model.model_path)
        
        await db.delete(model)
        await db.commit()
        
        return True
    
    @staticmethod
    async def upload_model_file(
        file_path: str,
        filename: str,
        model_type: str = "policy",
        algorithm: str = "ppo",
    ) -> str:
        """Upload a model file and return the stored path."""
        # Create model storage directory
        os.makedirs(settings.MODEL_STORAGE_PATH, exist_ok=True)
        
        # Generate unique filename
        import uuid
        unique_filename = f"{uuid.uuid4()}_{filename}"
        stored_path = os.path.join(settings.MODEL_STORAGE_PATH, unique_filename)
        
        # Move file to storage
        import shutil
        shutil.move(file_path, stored_path)
        
        return stored_path
    
    @staticmethod
    def validate_model_file(file_path: str) -> bool:
        """Validate model file."""
        if not os.path.exists(file_path):
            return False
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > settings.MAX_FILE_SIZE:
            return False
        
        # Check file extension
        file_ext = Path(file_path).suffix
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            return False
        
        return True
    
    @staticmethod
    def get_model_info(model_path: str) -> Dict[str, Any]:
        """Get model information."""
        if not os.path.exists(model_path):
            return {}
        
        file_size = os.path.getsize(model_path)
        file_stats = os.stat(model_path)
        
        return {
            "file_size": file_size,
            "created_time": file_stats.st_ctime,
            "modified_time": file_stats.st_mtime,
            "file_path": model_path,
        } 
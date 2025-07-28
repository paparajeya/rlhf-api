"""
Datasets API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
import os
import shutil
import json
from datetime import datetime

from ...core.database import get_db, Dataset
from ...core.config import settings
from ...schemas.datasets import DatasetCreate, DatasetResponse, DatasetUpdate

router = APIRouter()


@router.get("/", response_model=List[DatasetResponse])
async def get_datasets(db: AsyncSession = Depends(get_db)):
    """Get all datasets."""
    result = await db.execute(select(Dataset).order_by(Dataset.created_at.desc()))
    datasets = result.scalars().all()
    return [
        DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            file_size=dataset.file_size,
            num_samples=dataset.num_samples,
            created_at=dataset.created_at,
            metadata=dataset.metadata
        )
        for dataset in datasets
    ]


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: int, db: AsyncSession = Depends(get_db)):
    """Get a specific dataset."""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        file_size=dataset.file_size,
        num_samples=dataset.num_samples,
        created_at=dataset.created_at,
        metadata=dataset.metadata
    )


@router.post("/upload", response_model=DatasetResponse)
async def upload_dataset(
    name: str = Form(...),
    description: Optional[str] = Form(None),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """Upload a new dataset."""
    # Check if dataset with same name exists
    result = await db.execute(select(Dataset).where(Dataset.name == name))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Dataset with this name already exists")
    
    # Validate file type
    allowed_extensions = ['.json', '.csv', '.txt']
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Create dataset directory
    dataset_dir = os.path.join(settings.UPLOAD_DIR, "datasets")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Save dataset file
    file_path = os.path.join(dataset_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Get file size
    file_size = os.path.getsize(file_path)
    
    # Parse dataset to get number of samples
    num_samples = 0
    metadata = {}
    
    try:
        if file_extension == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    num_samples = len(data)
                elif isinstance(data, dict) and 'data' in data:
                    num_samples = len(data['data'])
                metadata = {'format': 'json', 'structure': type(data).__name__}
        elif file_extension == '.csv':
            import csv
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                num_samples = sum(1 for row in reader) - 1  # Subtract header
                metadata = {'format': 'csv'}
        elif file_extension == '.txt':
            with open(file_path, 'r') as f:
                lines = f.readlines()
                num_samples = len(lines)
                metadata = {'format': 'txt'}
    except Exception as e:
        # If parsing fails, still create the dataset but with 0 samples
        num_samples = 0
        metadata = {'error': str(e)}
    
    # Create dataset record
    dataset = Dataset(
        name=name,
        description=description,
        file_path=file_path,
        file_size=file_size,
        num_samples=num_samples,
        metadata=metadata
    )
    
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    
    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        file_size=dataset.file_size,
        num_samples=dataset.num_samples,
        created_at=dataset.created_at,
        metadata=dataset.metadata
    )


@router.put("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: int,
    dataset_update: DatasetUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update a dataset."""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Update fields
    for field, value in dataset_update.dict(exclude_unset=True).items():
        setattr(dataset, field, value)
    
    await db.commit()
    await db.refresh(dataset)
    
    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        file_size=dataset.file_size,
        num_samples=dataset.num_samples,
        created_at=dataset.created_at,
        metadata=dataset.metadata
    )


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a dataset."""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Delete dataset file
    if os.path.exists(dataset.file_path):
        os.remove(dataset.file_path)
    
    await db.delete(dataset)
    await db.commit()
    
    return {"message": "Dataset deleted successfully"} 
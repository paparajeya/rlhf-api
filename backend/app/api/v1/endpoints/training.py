"""
Training API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from datetime import datetime
import json

from ...core.database import get_db, TrainingJob
from ...schemas.training import TrainingJobCreate, TrainingJobResponse, TrainingJobUpdate
from ...services.training_service import TrainingService

router = APIRouter()


@router.get("/", response_model=List[TrainingJobResponse])
async def get_training_jobs(db: AsyncSession = Depends(get_db)):
    """Get all training jobs."""
    result = await db.execute(select(TrainingJob).order_by(TrainingJob.created_at.desc()))
    jobs = result.scalars().all()
    return [
        TrainingJobResponse(
            id=job.id,
            name=job.name,
            status=job.status,
            algorithm=job.algorithm,
            progress=job.progress,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
            logs=job.logs
        )
        for job in jobs
    ]


@router.get("/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(job_id: int, db: AsyncSession = Depends(get_db)):
    """Get a specific training job."""
    result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return TrainingJobResponse(
        id=job.id,
        name=job.name,
        status=job.status,
        algorithm=job.algorithm,
        progress=job.progress,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
        logs=job.logs
    )


@router.post("/", response_model=TrainingJobResponse)
async def create_training_job(
    job_data: TrainingJobCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Create a new training job."""
    # Check if job with same name exists
    result = await db.execute(select(TrainingJob).where(TrainingJob.name == job_data.name))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Training job with this name already exists")
    
    # Create training job
    job = TrainingJob(
        name=job_data.name,
        status="pending",
        algorithm=job_data.algorithm,
        config=job_data.config,
        progress=0.0,
        logs=""
    )
    
    db.add(job)
    await db.commit()
    await db.refresh(job)
    
    return TrainingJobResponse(
        id=job.id,
        name=job.name,
        status=job.status,
        algorithm=job.algorithm,
        progress=job.progress,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
        logs=job.logs
    )


@router.post("/{job_id}/start")
async def start_training_job(
    job_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Start a training job."""
    result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    if job.status != "pending":
        raise HTTPException(status_code=400, detail="Job is not in pending status")
    
    # Update job status
    job.status = "running"
    job.started_at = datetime.utcnow()
    await db.commit()
    
    # Start training in background
    training_service = TrainingService()
    background_tasks.add_task(training_service.run_training, job_id, job.config)
    
    return {"message": "Training job started successfully"}


@router.post("/{job_id}/stop")
async def stop_training_job(job_id: int, db: AsyncSession = Depends(get_db)):
    """Stop a training job."""
    result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    if job.status != "running":
        raise HTTPException(status_code=400, detail="Job is not running")
    
    # Update job status
    job.status = "failed"
    job.completed_at = datetime.utcnow()
    job.error_message = "Training stopped by user"
    await db.commit()
    
    return {"message": "Training job stopped successfully"}


@router.put("/{job_id}", response_model=TrainingJobResponse)
async def update_training_job(
    job_id: int,
    job_update: TrainingJobUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update a training job."""
    result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    # Update fields
    for field, value in job_update.dict(exclude_unset=True).items():
        setattr(job, field, value)
    
    await db.commit()
    await db.refresh(job)
    
    return TrainingJobResponse(
        id=job.id,
        name=job.name,
        status=job.status,
        algorithm=job.algorithm,
        progress=job.progress,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
        logs=job.logs
    )


@router.delete("/{job_id}")
async def delete_training_job(job_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a training job."""
    result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    await db.delete(job)
    await db.commit()
    
    return {"message": "Training job deleted successfully"} 
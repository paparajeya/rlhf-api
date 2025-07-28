"""
Training service for managing RLHF training jobs.
"""

import asyncio
import logging
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime

from ..core.database import get_db, TrainingJob
from ..core.config import settings

logger = logging.getLogger(__name__)


class TrainingService:
    """Service for managing training jobs."""
    
    def __init__(self):
        self.running_jobs = {}
    
    async def run_training(self, job_id: int, config: Dict[str, Any]):
        """Run a training job."""
        try:
            # Get database session
            async for db in get_db():
                # Update job status to running
                result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
                job = result.scalar_one_or_none()
                
                if not job:
                    logger.error(f"Training job {job_id} not found")
                    return
                
                job.status = "running"
                job.started_at = datetime.utcnow()
                job.progress = 0.0
                await db.commit()
                
                # Simulate training process
                logger.info(f"Starting training job {job_id} with config: {config}")
                
                # Simulate training epochs
                epochs = config.get('epochs', 3)
                for epoch in range(epochs):
                    if job.status != "running":
                        logger.info(f"Training job {job_id} stopped")
                        return
                    
                    # Simulate epoch training
                    await asyncio.sleep(2)  # Simulate training time
                    
                    # Update progress
                    progress = (epoch + 1) / epochs
                    job.progress = progress
                    job.logs = f"Epoch {epoch + 1}/{epochs} completed\n"
                    await db.commit()
                    
                    logger.info(f"Training job {job_id} - Epoch {epoch + 1}/{epochs} completed")
                
                # Mark job as completed
                job.status = "completed"
                job.progress = 1.0
                job.completed_at = datetime.utcnow()
                job.logs += "Training completed successfully\n"
                await db.commit()
                
                logger.info(f"Training job {job_id} completed successfully")
                
        except Exception as e:
            logger.error(f"Training job {job_id} failed: {str(e)}")
            
            # Update job status to failed
            async for db in get_db():
                result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
                job = result.scalar_one_or_none()
                
                if job:
                    job.status = "failed"
                    job.completed_at = datetime.utcnow()
                    job.error_message = str(e)
                    job.logs += f"Training failed: {str(e)}\n"
                    await db.commit()
    
    async def stop_training(self, job_id: int):
        """Stop a running training job."""
        async for db in get_db():
            result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
            job = result.scalar_one_or_none()
            
            if job and job.status == "running":
                job.status = "failed"
                job.completed_at = datetime.utcnow()
                job.error_message = "Training stopped by user"
                job.logs += "Training stopped by user\n"
                await db.commit()
                
                logger.info(f"Training job {job_id} stopped by user")
    
    async def get_training_logs(self, job_id: int) -> str:
        """Get training logs for a job."""
        async for db in get_db():
            result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
            job = result.scalar_one_or_none()
            
            if job:
                return job.logs or ""
            return "" 
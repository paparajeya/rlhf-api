"""
Evaluations API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from datetime import datetime

from ...core.database import get_db, Evaluation, Model, Dataset
from ...schemas.evaluations import EvaluationCreate, EvaluationResponse
from ...services.evaluation_service import EvaluationService

router = APIRouter()


@router.get("/", response_model=List[EvaluationResponse])
async def get_evaluations(db: AsyncSession = Depends(get_db)):
    """Get all evaluations."""
    result = await db.execute(
        select(Evaluation, Model.name.label('model_name'), Dataset.name.label('dataset_name'))
        .join(Model, Evaluation.model_id == Model.id)
        .join(Dataset, Evaluation.dataset_id == Dataset.id)
        .order_by(Evaluation.created_at.desc())
    )
    evaluations = result.all()
    
    return [
        EvaluationResponse(
            id=eval.id,
            model_id=eval.model_id,
            model_name=model_name,
            dataset_id=eval.dataset_id,
            dataset_name=dataset_name,
            metrics=eval.metrics,
            status=eval.status,
            created_at=eval.created_at,
            completed_at=eval.completed_at
        )
        for eval, model_name, dataset_name in evaluations
    ]


@router.get("/{evaluation_id}", response_model=EvaluationResponse)
async def get_evaluation(evaluation_id: int, db: AsyncSession = Depends(get_db)):
    """Get a specific evaluation."""
    result = await db.execute(
        select(Evaluation, Model.name.label('model_name'), Dataset.name.label('dataset_name'))
        .join(Model, Evaluation.model_id == Model.id)
        .join(Dataset, Evaluation.dataset_id == Dataset.id)
        .where(Evaluation.id == evaluation_id)
    )
    evaluation_data = result.first()
    
    if not evaluation_data:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    eval, model_name, dataset_name = evaluation_data
    
    return EvaluationResponse(
        id=eval.id,
        model_id=eval.model_id,
        model_name=model_name,
        dataset_id=eval.dataset_id,
        dataset_name=dataset_name,
        metrics=eval.metrics,
        status=eval.status,
        created_at=eval.created_at,
        completed_at=eval.completed_at
    )


@router.post("/", response_model=EvaluationResponse)
async def create_evaluation(
    evaluation_data: EvaluationCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Create a new evaluation."""
    # Check if model exists
    result = await db.execute(select(Model).where(Model.id == evaluation_data.model_id))
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Check if dataset exists
    result = await db.execute(select(Dataset).where(Dataset.id == evaluation_data.dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Create evaluation
    evaluation = Evaluation(
        model_id=evaluation_data.model_id,
        dataset_id=evaluation_data.dataset_id,
        metrics={},
        status="pending"
    )
    
    db.add(evaluation)
    await db.commit()
    await db.refresh(evaluation)
    
    # Start evaluation in background
    evaluation_service = EvaluationService()
    background_tasks.add_task(
        evaluation_service.run_evaluation,
        evaluation.id,
        evaluation_data.model_id,
        evaluation_data.dataset_id
    )
    
    return EvaluationResponse(
        id=evaluation.id,
        model_id=evaluation.model_id,
        model_name=model.name,
        dataset_id=evaluation.dataset_id,
        dataset_name=dataset.name,
        metrics=evaluation.metrics,
        status=evaluation.status,
        created_at=evaluation.created_at,
        completed_at=evaluation.completed_at
    )


@router.delete("/{evaluation_id}")
async def delete_evaluation(evaluation_id: int, db: AsyncSession = Depends(get_db)):
    """Delete an evaluation."""
    result = await db.execute(select(Evaluation).where(Evaluation.id == evaluation_id))
    evaluation = result.scalar_one_or_none()
    
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    await db.delete(evaluation)
    await db.commit()
    
    return {"message": "Evaluation deleted successfully"} 
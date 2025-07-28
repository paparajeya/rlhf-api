"""
Main API router for v1 endpoints.
"""

from fastapi import APIRouter
from .endpoints import models, training, datasets, feedback, evaluations, inference

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(training.router, prefix="/training", tags=["training"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(feedback.router, prefix="/feedback", tags=["feedback"])
api_router.include_router(evaluations.router, prefix="/evaluations", tags=["evaluations"])
api_router.include_router(inference.router, prefix="/inference", tags=["inference"]) 
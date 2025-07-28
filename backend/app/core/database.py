"""
Database configuration and models.
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from datetime import datetime
from typing import AsyncGenerator
import asyncio

from .config import settings

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
    echo=settings.DEBUG,
    pool_pre_ping=True,
)

# Create sync engine for migrations
sync_engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_pre_ping=True,
)

# Create session factories
AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

SessionLocal = sessionmaker(
    sync_engine, class_=Session, expire_on_commit=False
)

# Create base class
Base = declarative_base()


class Model(Base):
    """Model table for storing trained models."""
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text, nullable=True)
    model_path = Column(String, nullable=False)
    model_type = Column(String, nullable=False)  # policy, value, reward, etc.
    algorithm = Column(String, nullable=False)  # ppo, dpo, a2c
    config = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TrainingJob(Base):
    """Training job table for tracking training progress."""
    __tablename__ = "training_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    status = Column(String, default="pending")  # pending, running, completed, failed
    algorithm = Column(String, nullable=False)
    config = Column(JSON, nullable=False)
    progress = Column(Float, default=0.0)
    metrics = Column(JSON, nullable=True)
    logs = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)


class Dataset(Base):
    """Dataset table for storing training datasets."""
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text, nullable=True)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    num_samples = Column(Integer, nullable=False)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Feedback(Base):
    """Feedback table for storing human feedback."""
    __tablename__ = "feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(Text, nullable=False)
    preferred_response = Column(Text, nullable=False)
    dispreferred_response = Column(Text, nullable=False)
    user_id = Column(String, nullable=True)
    session_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Evaluation(Base):
    """Evaluation table for storing model evaluations."""
    __tablename__ = "evaluations"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, nullable=False)
    dataset_id = Column(Integer, nullable=False)
    metrics = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# Database dependency
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


def get_sync_db() -> Session:
    """Get synchronous database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()


async def init_db():
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def init_sync_db():
    """Initialize database tables synchronously."""
    Base.metadata.create_all(bind=sync_engine) 
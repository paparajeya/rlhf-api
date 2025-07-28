"""
Configuration settings for the RLHF API.
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "RLHF API"
    VERSION: str = "0.1.0"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # CORS Configuration
    ALLOWED_HOSTS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    
    # Database Configuration
    DATABASE_URL: str = "postgresql://user:password@localhost/rlhf_db"
    DATABASE_TEST_URL: str = "postgresql://user:password@localhost/rlhf_test_db"
    
    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_DB: int = 0
    
    # Security Configuration
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # File Upload Configuration
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: List[str] = [".json", ".txt", ".csv", ".pt", ".pth"]
    
    # Model Configuration
    MODEL_STORAGE_PATH: str = "./models"
    DEFAULT_MODEL: str = "gpt2"
    MAX_LENGTH: int = 512
    BATCH_SIZE: int = 4
    
    # Training Configuration
    DEFAULT_LEARNING_RATE: float = 1e-5
    DEFAULT_EPOCHS: int = 3
    DEFAULT_ALGORITHM: str = "ppo"
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    
    # Monitoring Configuration
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # Celery Configuration
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # External APIs
    OPENAI_API_KEY: Optional[str] = None
    HUGGINGFACE_TOKEN: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()

# Create directories
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.MODEL_STORAGE_PATH, exist_ok=True) 
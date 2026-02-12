from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "healthXher RA Prediction"
    API_V1_STR: str = "/api/v1"
    
    # Security: Use environment variables for these!
    SECRET_KEY: str = "REQUIRED_SECRET_KEY_ENV_VAR" 
    DB_PASSPHRASE: str = "REQUIRED_DB_PASSPHRASE_ENV_VAR"
    
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 1 week

    # Standard SQLite for MVP if SQLCipher build fails in environment
    DATABASE_URL: str = "sqlite:///./ra_prediction.db"
    # For SQLCipher, use: "sqlite+pysqlcipher://:passphrase@/./ra_encrypted.db"

    MODEL_CACHE_PATH: str = "app/core/model_cache.pkl"
    PIPELINE_CACHE_PATH: str = "python/model_cache.pkl"
    
    # Path to the dataset used for one-time training/seeding
    TRAINING_DATA_PATH: str = "data/The_final_data_after_screening.csv"

    # Expert Weights - Can be overridden via environment variables for research scaling
    # Format: {"FeatureName": Weight}
    CUSTOM_FEATURE_WEIGHTS: Optional[dict] = None

    class Config:
        case_sensitive = True

settings = Settings()

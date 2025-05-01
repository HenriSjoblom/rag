from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator, field_validator, computed_field # Import computed_field
from typing import Literal, Optional, Dict, Any # Import Dict, Any
from pathlib import Path

class Settings(BaseSettings):
    """Loads and validates application settings."""
    EMBEDDING_MODEL_NAME: str = Field("all-MiniLM-L6-v2", validation_alias='EMBEDDING_MODEL_NAME')
    TOP_K_RESULTS: int = Field(5, gt=0, validation_alias='TOP_K_RESULTS') # Must be > 0

    # ChromaDB Settings
    CHROMA_PATH: Optional[str] = Field("./data/chroma_db", validation_alias='CHROMA_PATH')
    CHROMA_COLLECTION_NAME: str = Field("support_docs", validation_alias='CHROMA_COLLECTION_NAME')

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

settings = Settings()
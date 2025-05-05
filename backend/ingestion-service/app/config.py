from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, DirectoryPath, FilePath, validator, field_validator
from typing import Literal, Optional
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Loads and validates application settings for the Ingestion Service."""
    # Document Source
    SOURCE_DIRECTORY: str = Field("./documents_to_ingest", validation_alias='SOURCE_DIRECTORY')

    # Embedding Model (Must match retrieval service)
    EMBEDDING_MODEL_NAME: str = Field("all-MiniLM-L6-v2", validation_alias='EMBEDDING_MODEL_NAME')

    CHROMA_LOCAL_PATH: Optional[str] = Field("./data/chroma_db", validation_alias='CHROMA_LOCAL_PATH')
    CHROMA_COLLECTION_NAME: str = Field("support_docs", validation_alias='CHROMA_COLLECTION_NAME')

    # Text Splitting
    CHUNK_SIZE: int = Field(1000, gt=0, validation_alias='CHUNK_SIZE')
    CHUNK_OVERLAP: int = Field(150, ge=0, validation_alias='CHUNK_OVERLAP')

    # Ingestion Options
    CLEAN_COLLECTION_BEFORE_INGEST: bool = Field(False, validation_alias='CLEAN_COLLECTION_BEFORE_INGEST')

    # Validators
    @field_validator('SOURCE_DIRECTORY', mode='before')
    @classmethod
    def validate_source_directory(cls, v):
        if not v:
            raise ValueError("SOURCE_DIRECTORY cannot be empty.")
        # Resolve path relative to project root (assuming .env is in root)
        path = Path(v)
        if not path.is_absolute():
            # Assumes config.py is in app/core, go up two levels for root
            base_dir = Path(__file__).resolve().parent.parent.parent
            path = base_dir / v
        if not path.exists() or not path.is_dir():
            raise ValueError(f"SOURCE_DIRECTORY does not exist or is not a directory: {path.resolve()}")
        return str(path.resolve())

    @field_validator('CHUNK_OVERLAP', mode='before')
    @classmethod
    def validate_chunk_overlap(cls, v, info):
        chunk_size = info.data.get('CHUNK_SIZE')
        # Ensure overlap is not larger than chunk size
        if chunk_size is not None and v is not None and v >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return v

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

settings = Settings()

# Log the resolved source directory path for verification
logger.debug(f"Resolved source directory: {settings.SOURCE_DIRECTORY}")
logger.debug(f"Resolved ChromaDB local path: {settings.CHROMA_PATH}")


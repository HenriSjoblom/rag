import logging
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

SERVICE_ROOT_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Loads and validates application settings for the Ingestion Service."""

    # Document Source
    SOURCE_DIRECTORY: str = Field(
        default="/app/documents", validation_alias="SOURCE_DIRECTORY"
    )

    # File Upload Limits
    MAX_FILE_SIZE_MB: int = Field(default=50, ge=1, le=500)

    # Embedding Model (Must match retrieval service)
    EMBEDDING_MODEL_NAME: str = Field(
        "all-MiniLM-L6-v2", validation_alias="EMBEDDING_MODEL_NAME"
    )

    # ChromaDB Settings
    CHROMA_MODE: Literal["local", "docker"] = Field(
        "docker", validation_alias="CHROMA_MODE"
    )
    CHROMA_PATH: Optional[str] = Field(
        "./data/chroma_db", validation_alias="CHROMA_PATH"
    )
    CHROMA_HOST: Optional[str] = Field("chromadb", validation_alias="CHROMA_HOST")
    CHROMA_PORT: Optional[int] = Field(8000, validation_alias="CHROMA_PORT")
    CHROMA_COLLECTION_NAME: str = Field(
        "support_docs", validation_alias="CHROMA_COLLECTION_NAME"
    )

    # Text Splitting
    CHUNK_SIZE: int = Field(1000, gt=100, le=4000, validation_alias="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(150, ge=0, validation_alias="CHUNK_OVERLAP")

    # Processing Options
    CLEAN_COLLECTION_BEFORE_INGEST: bool = Field(
        False, validation_alias="CLEAN_COLLECTION_BEFORE_INGEST"
    )

    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def validate_chunk_overlap(cls, v, info):
        chunk_size = info.data.get("CHUNK_SIZE", 1000)
        if v >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return v

    @field_validator("CHROMA_HOST")
    @classmethod
    def validate_chroma_host(cls, v, info):
        mode = info.data.get("CHROMA_MODE")
        if mode == "docker" and not v:
            raise ValueError("CHROMA_HOST is required for docker mode")
        return v

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=True
    )


settings = Settings()

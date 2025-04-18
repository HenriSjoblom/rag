# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, DirectoryPath, FilePath, validator
from typing import Literal, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """Loads and validates application settings."""
    EMBEDDING_MODEL_NAME: str = Field("all-MiniLM-L6-v2", validation_alias='EMBEDDING_MODEL_NAME')
    TOP_K_RESULTS: int = Field(5, gt=0, validation_alias='TOP_K_RESULTS') # Must be > 0

    # ChromaDB Settings
    CHROMA_MODE: Literal['local', 'http', 'https'] = Field("local", validation_alias='CHROMA_MODE')
    CHROMA_LOCAL_PATH: Optional[str] = Field("./data/chroma_db", validation_alias='CHROMA_LOCAL_PATH')
    CHROMA_COLLECTION_NAME: str = Field("support_docs", validation_alias='CHROMA_COLLECTION_NAME')
    CHROMA_HOST: Optional[str] = Field(None, validation_alias='CHROMA_HOST')
    CHROMA_PORT: Optional[int] = Field(None, validation_alias='CHROMA_PORT')

    @validator('CHROMA_LOCAL_PATH', pre=True, always=True)
    def validate_local_path(cls, v, values):
        mode = values.get('CHROMA_MODE')
        if mode == 'local' and not v:
            raise ValueError("CHROMA_LOCAL_PATH must be set when CHROMA_MODE is 'local'")
        if mode == 'local' and v:
            # Ensure the path exists or can be created relative to the project root
            # Assuming .env is in the root, this path is relative to root
            path = Path(v)
            if not path.is_absolute():
                base_dir = Path(__file__).resolve().parent.parent.parent
                path = base_dir / v
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            return str(path.resolve()) # Return resolved absolute path string
        return v # Return None if not local mode

    @validator('CHROMA_HOST', 'CHROMA_PORT', pre=True, always=True)
    def validate_server_config(cls, v, values, field):
        mode = values.get('CHROMA_MODE')
        if mode in ('http', 'https') and not v:
            raise ValueError(f"{field.name} must be set when CHROMA_MODE is '{mode}'")
        return v

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

settings = Settings()
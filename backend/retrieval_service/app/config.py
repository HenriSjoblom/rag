from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_FILE_PATH = Path(__file__).resolve()
SERVICE_ROOT_DIR = CONFIG_FILE_PATH.parent.parent
ENV_FILE_PATH = SERVICE_ROOT_DIR / ".env"


class Settings(BaseSettings):
    """Loads and validates application settings."""

    EMBEDDING_MODEL_NAME: str = Field(
        "all-MiniLM-L6-v2", validation_alias="EMBEDDING_MODEL_NAME"
    )
    TOP_K_RESULTS: int = Field(5, gt=0, validation_alias="TOP_K_RESULTS")  # Must be > 0

    # ChromaDB Settings
    # Mode: local or docker
    CHROMA_MODE: Literal["local", "docker"] = Field(
        "local", validation_alias="CHROMA_MODE"
    )
    # For local mode
    CHROMA_PATH: Optional[str] = Field(
        "./data/chroma_db", validation_alias="CHROMA_PATH"
    )
    # For docker mode
    CHROMA_HOST: Optional[str] = Field(
        "http://localhost", validation_alias="CHROMA_HOST"
    )
    CHROMA_PORT: Optional[int] = Field(8010, validation_alias="CHROMA_PORT")
    CHROMA_COLLECTION_NAME: str = Field(
        "support_docs", validation_alias="CHROMA_COLLECTION_NAME"
    )

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH, env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()

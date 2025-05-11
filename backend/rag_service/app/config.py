from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyHttpUrl, Field
from pathlib import Path


CONFIG_FILE_PATH = Path(__file__).resolve()
SERVICE_ROOT_DIR = CONFIG_FILE_PATH.parent.parent
ENV_FILE_PATH = SERVICE_ROOT_DIR / ".env"


class Settings(BaseSettings):
    """Loads and validates application settings from environment variables."""
    RETRIEVAL_SERVICE_URL: AnyHttpUrl = Field(..., validation_alias='RETRIEVAL_SERVICE_URL')
    GENERATION_SERVICE_URL: AnyHttpUrl = Field(..., validation_alias='GENERATION_SERVICE_URL')
    INGESTION_SERVICE_URL: AnyHttpUrl = Field(..., validation_alias='INGESTION_SERVICE_URL')
    HTTP_CLIENT_TIMEOUT: float = 10.0

    # Configure Pydantic settings to load from a .env file
    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH, # Load from .env file in the root
        env_file_encoding='utf-8',
        extra='ignore' # Ignore extra fields from .env
    )

# Create a single instance of settings to be used across the application
settings = Settings()
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyHttpUrl, Field
import os
from pathlib import Path


class Settings(BaseSettings):
    """Loads and validates application settings from environment variables."""
    RETRIEVAL_SERVICE_URL: AnyHttpUrl = Field(..., validation_alias='RETRIEVAL_SERVICE_URL')
    GENERATION_SERVICE_URL: AnyHttpUrl = Field(..., validation_alias='GENERATION_SERVICE_URL')
    HTTP_CLIENT_TIMEOUT: float = 10.0

    # Configure Pydantic settings to load from a .env file
    model_config = SettingsConfigDict(
        env_file='.env', # Load from .env file in the root
        env_file_encoding='utf-8',
        extra='ignore' # Ignore extra fields from .env
    )

# Create a single instance of settings to be used across the application
settings = Settings()
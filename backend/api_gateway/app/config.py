from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyHttpUrl, Field
import os
from pathlib import Path


class Settings(BaseSettings):
    """Loads and validates application settings from environment variables."""
    RAG_SERVICE_URL: AnyHttpUrl = Field(..., validation_alias='RAG_SERVICE_URL')

    # Configure Pydantic settings to load from a .env file
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

# Create a single instance of settings to be used across the application
settings = Settings()
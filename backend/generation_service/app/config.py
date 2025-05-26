import logging
from typing import Optional

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings with validation."""

    # LLM Configuration
    LLM_PROVIDER: str = Field(
        default="openai", description="LLM provider (openai, etc.)"
    )
    LLM_MODEL_NAME: str = Field(default="gpt-3.5-turbo", description="LLM model name")
    LLM_TEMPERATURE: float = Field(
        default=0.7, ge=0.0, le=2.0, description="LLM temperature"
    )
    LLM_MAX_TOKENS: int = Field(
        default=500, ge=1, le=4000, description="Maximum tokens for LLM response"
    )

    # API Keys
    OPENAI_API_KEY: Optional[SecretStr] = Field(
        default=None, description="OpenAI API key"
    )

    # Service Configuration
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    @field_validator("LLM_PROVIDER")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        """Validate LLM provider."""
        supported_providers = ["openai"]
        if v not in supported_providers:
            raise ValueError(
                f"Unsupported LLM provider: {v}. Supported: {supported_providers}"
            )
        return v

    @model_validator(mode="after")
    def validate_openai_configuration(self) -> "Settings":
        """Validate OpenAI configuration requirements."""
        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI provider")
        return self

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

# Create global settings instance
try:
    settings = Settings()
    logger.info(
        f"Settings loaded successfully with LLM provider: {settings.LLM_PROVIDER}"
    )
except Exception as e:
    logger.error(f"Failed to load settings: {e}")
    raise

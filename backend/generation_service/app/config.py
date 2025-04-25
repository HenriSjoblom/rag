from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr
from typing import Literal, Optional

class Settings(BaseSettings):
    """Loads and validates application settings."""
    LLM_PROVIDER: Literal["openai"] = Field("openai", validation_alias='LLM_PROVIDER') # Extend Literal if supporting more
    LLM_MODEL_NAME: str = Field("gpt-3.5-turbo", validation_alias='LLM_MODEL_NAME')
    LLM_TEMPERATURE: float = Field(0.3, ge=0.0, le=2.0, validation_alias='LLM_TEMPERATURE')
    LLM_MAX_TOKENS: int = Field(500, gt=0, validation_alias='LLM_MAX_TOKENS')

    # --- Provider Specific API Keys ---
    # Use SecretStr to prevent accidental logging/exposure
    OPENAI_API_KEY: Optional[SecretStr] = Field(None, validation_alias='OPENAI_API_KEY')

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

settings = Settings()

# Security check: Ensure API key is loaded if provider is OpenAI
if settings.LLM_PROVIDER == "openai" and not settings.OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing in environment variables or .env file.")
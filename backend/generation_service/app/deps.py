from fastapi import Depends
from functools import lru_cache

from app.config import Settings, settings as global_settings
from app.services.generation import GenerationService

# Dependency to get application settings
def get_settings() -> Settings:
    return global_settings

# Use lru_cache to create a singleton instance of the GenerationService
# This avoids re-initializing the LLM model on every request.
# maxsize=1 effectively makes it a singleton within the running worker process.
@lru_cache(maxsize=1)
def get_generation_service() -> GenerationService:
    """
    Provides a cached singleton instance of the GenerationService.
    Initializes the LLM client once per worker process.
    """
    print("Creating GenerationService instance (or retrieving from cache)...")
    print(f"Settings in creating generation service used: {global_settings.LLM_PROVIDER}, {global_settings.LLM_MODEL_NAME}")
    return GenerationService(settings=global_settings)

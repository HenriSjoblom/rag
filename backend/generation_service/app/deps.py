import logging
from functools import lru_cache

from app.config import Settings
from app.config import settings as global_settings
from app.services.generation import GenerationService

logger = logging.getLogger(__name__)


def get_settings() -> Settings:
    """
    Get application settings instance.

    Returns:
        Settings: Global application settings
    """
    return global_settings


@lru_cache(maxsize=1)
def get_generation_service() -> GenerationService:
    """
    Provide a cached singleton instance of the GenerationService.

    This function uses LRU cache to ensure the LLM model is initialized
    only once per worker process, improving performance and resource usage.

    Returns:
        GenerationService: Cached service instance

    Raises:
        RuntimeError: If service initialization fails
    """
    try:
        logger.info("Creating or retrieving GenerationService from cache")
        logger.debug(
            f"Using configuration: {global_settings.LLM_PROVIDER}/{global_settings.LLM_MODEL_NAME}"
        )

        # Create service instance
        service = GenerationService(settings=global_settings)

        # Verify service health after creation
        if not service.is_healthy():
            logger.error("GenerationService failed health check after initialization")
            raise RuntimeError(
                "GenerationService failed health check after initialization"
            )

        logger.info("GenerationService is ready and healthy")
        return service

    except Exception as e:
        logger.error(f"GenerationService initialization failed: {e}", exc_info=True)

        # Clear cache to allow retry on next request
        get_generation_service.cache_clear()
        logger.debug("Cleared GenerationService cache due to initialization failure")

        raise RuntimeError(f"Failed to initialize GenerationService: {e}") from e

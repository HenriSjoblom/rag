import httpx
from fastapi import Depends, HTTPException, status

from app.config import Settings
from app.config import settings as app_settings
from app.services.chat_processor import ChatProcessorService

# Import the getter for the globally managed client
from app.services.http_client import get_global_http_client


def get_http_client() -> (
    httpx.AsyncClient
):  # Removed settings dependency as timeout is handled at lifespan
    """
    Provides the globally managed httpx.AsyncClient instance
    as a FastAPI dependency.
    """
    # This now calls the getter from http_client.py, which returns the instance
    # initialized by the lifespan_http_client.
    return get_global_http_client()


# Cache for ChatProcessor
_chat_processor_instance: ChatProcessorService | None = None


def get_chat_processor_service(
    settings: Settings = Depends(lambda: app_settings),
    http_client: httpx.AsyncClient = Depends(get_http_client),
) -> ChatProcessorService:
    """
    Provides a singleton instance of the ChatProcessorService.
    Initializes it with necessary service URLs from settings and the HTTP client.
    """
    global _chat_processor_instance
    if _chat_processor_instance is None:
        if not settings.RETRIEVAL_SERVICE_URL or not settings.GENERATION_SERVICE_URL:
            logger.error(
                "RETRIEVAL_SERVICE_URL or GENERATION_SERVICE_URL is not configured."
            )  # Added logger
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="RETRIEVAL_SERVICE_URL or GENERATION_SERVICE_URL is not configured.",
            )
        _chat_processor_instance = ChatProcessorService(
            retrieval_service_url=str(settings.RETRIEVAL_SERVICE_URL),
            generation_service_url=str(settings.GENERATION_SERVICE_URL),
            http_client=http_client,
        )
    return _chat_processor_instance


def get_settings() -> Settings:
    return app_settings

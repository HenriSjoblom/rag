import httpx
from fastapi import Depends, HTTPException, status

from app.config import Settings
from app.config import settings as app_settings
from app.services.chat_processor import ChatProcessorService


def get_http_client() -> httpx.AsyncClient:
    """Dependency to provide HTTP client."""
    return httpx.AsyncClient(timeout=30.0)


def get_chat_processor_service(
    settings: Settings = Depends(lambda: app_settings),
    http_client: httpx.AsyncClient = Depends(get_http_client),
) -> ChatProcessorService:
    """
    Provides a ChatProcessorService instance.
    Creates a new instance per request with the injected HTTP client.
    """
    if not settings.RETRIEVAL_SERVICE_URL or not settings.GENERATION_SERVICE_URL:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RETRIEVAL_SERVICE_URL or GENERATION_SERVICE_URL is not configured.",
        )
    return ChatProcessorService(
        retrieval_service_url=str(settings.RETRIEVAL_SERVICE_URL),
        generation_service_url=str(settings.GENERATION_SERVICE_URL),
        http_client=http_client,
    )


def get_settings() -> Settings:
    return app_settings

import httpx
from fastapi import Depends, HTTPException, Request, status

from app.config import Settings
from app.config import settings as app_settings
from app.services.chat_processor import ChatProcessorService

# Import the getter for the globally managed client


def get_http_client(request: Request) -> httpx.AsyncClient:
    """Dependency to get the global HTTP client from application state."""
    if (
        not hasattr(request.app.state, "http_client")
        or request.app.state.http_client is None
    ):
        raise RuntimeError(
            "Global HTTP client is not initialized. Ensure the application lifespan manager has run."
        )
    return request.app.state.http_client


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

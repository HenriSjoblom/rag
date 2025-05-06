from fastapi import Depends
import httpx

from app.config import Settings, settings as global_settings
from app.services.chat_processor import ChatProcessorService
from app.services.http_client import get_http_client

# Dependency to get application settings
def get_settings() -> Settings:
    return global_settings

# Dependency to get an instance of the ChatProcessorService
def get_chat_processor_service(
    settings: Settings = Depends(get_settings),
    http_client: httpx.AsyncClient = Depends(get_http_client) # Get shared client
) -> ChatProcessorService:
    """Provides an instance of the ChatProcessorService."""
    return ChatProcessorService(settings=settings, http_client=http_client)
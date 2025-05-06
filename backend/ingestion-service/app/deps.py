from fastapi import Depends

from app.config import Settings, settings as global_settings
from app.services.ingestion_processor import IngestionProcessorService

# Dependency to get application settings
def get_settings() -> Settings:
    return global_settings

def get_ingestion_processor_service(
    settings: Settings = Depends(get_settings)
) -> IngestionProcessorService:
    """Provides a cached instance of the IngestionProcessorService."""
    return IngestionProcessorService(settings=settings)
from fastapi import Depends, Request

from app.config import Settings
from app.config import settings as global_settings
from app.services.collection_manager import CollectionManagerService
from app.services.file_management import FileManagementService
from app.services.ingestion_processor import IngestionProcessorService
from app.services.ingestion_state import IngestionStateService


# Dependency to get application settings
def get_settings() -> Settings:
    return global_settings


def get_ingestion_processor_service(
    settings: Settings = Depends(get_settings),
) -> IngestionProcessorService:
    """Provides an instance of the IngestionProcessorService."""
    # This service now directly uses get_vector_store which uses global _vector_store
    return IngestionProcessorService(settings=settings)


def get_file_management_service(
    settings: Settings = Depends(get_settings),
) -> FileManagementService:
    """Dependency to get FileManagementService instance."""
    return FileManagementService(settings)


def get_collection_manager_service(
    settings: Settings = Depends(get_settings),
) -> CollectionManagerService:
    """Dependency to get CollectionManagerService instance."""
    return CollectionManagerService(settings)


def get_ingestion_state_service(request: Request) -> IngestionStateService:
    """Dependency to get IngestionStateService from application state."""
    if not hasattr(request.app.state, "ingestion_state_service"):
        # Fallback initialization if not properly set up in lifespan
        request.app.state.ingestion_state_service = IngestionStateService()
    return request.app.state.ingestion_state_service


# Keep backward compatibility aliases if needed elsewhere
def get_document_service(
    settings: Settings = Depends(get_settings),
) -> FileManagementService:
    """Alias for backward compatibility - use get_file_management_service instead."""
    return FileManagementService(settings)


def get_file_upload_service(
    settings: Settings = Depends(get_settings),
) -> FileManagementService:
    """Alias for backward compatibility - use get_file_management_service instead."""
    return FileManagementService(settings)

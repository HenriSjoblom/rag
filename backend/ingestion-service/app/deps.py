from fastapi import Depends

from app.config import Settings
from app.config import settings as global_settings
from app.services.collection_manager import CollectionManagerService
from app.services.document_service import DocumentService
from app.services.file_uploader import FileUploadService
from app.services.ingestion_processor import IngestionProcessorService


# Dependency to get application settings
def get_settings() -> Settings:
    return global_settings


def get_ingestion_processor_service(
    settings: Settings = Depends(get_settings),
) -> IngestionProcessorService:
    """Provides an instance of the IngestionProcessorService."""
    # This service now directly uses get_vector_store which uses global _vector_store
    return IngestionProcessorService(settings=settings)


def get_file_upload_service(
    settings: Settings = Depends(get_settings),
) -> FileUploadService:
    """Provides an instance of the FileUploadService."""
    return FileUploadService(source_directory_str=settings.SOURCE_DIRECTORY)


def get_collection_manager_service(
    settings: Settings = Depends(get_settings),
) -> CollectionManagerService:
    """Dependency to get CollectionManagerService instance."""
    return CollectionManagerService(settings)


def get_document_service(settings: Settings = Depends(get_settings)) -> DocumentService:
    """Dependency to get DocumentService instance."""
    return DocumentService(settings)

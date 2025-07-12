from fastapi import Depends, Request

from app.config import Settings
from app.config import settings as global_settings
from app.services.chroma_manager import (
    ChromaClientManager,
    EmbeddingModelManager,
    VectorStoreManager,
)
from app.services.collection_manager import CollectionManagerService
from app.services.file_management import FileManagementService
from app.services.ingestion_processor import IngestionProcessorService
from app.services.ingestion_state import IngestionStateService


# Dependency to get application settings
def get_settings() -> Settings:
    return global_settings


def get_chroma_client_manager(request: Request) -> ChromaClientManager:
    """Get ChromaDB client manager from application state."""
    return request.app.state.chroma_manager


def get_embedding_model_manager(request: Request) -> EmbeddingModelManager:
    """Get embedding model manager from application state."""
    return request.app.state.embedding_manager


def get_vector_store_manager(request: Request) -> VectorStoreManager:
    """Get vector store manager from application state."""
    return request.app.state.vector_store_manager


def get_ingestion_processor_service(
    settings: Settings = Depends(get_settings),
    request: Request = None,
) -> IngestionProcessorService:
    """Provides an instance of the IngestionProcessorService."""
    return IngestionProcessorService(
        settings,
        request.app.state.chroma_manager,
        request.app.state.embedding_manager,
        request.app.state.vector_store_manager,
    )


def get_file_management_service(
    settings: Settings = Depends(get_settings),
) -> FileManagementService:
    """Dependency to get FileManagementService instance."""
    return FileManagementService(settings)


def get_ingestion_state_service(request: Request) -> IngestionStateService:
    """Dependency to get IngestionStateService from application state."""
    return request.app.state.ingestion_state_service


def get_file_upload_service(
    settings: Settings = Depends(get_settings),
) -> FileManagementService:
    """Alias for backward compatibility - use get_file_management_service instead."""
    return FileManagementService(settings)

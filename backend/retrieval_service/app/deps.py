import logging

from fastapi import Depends, Request

from app.config import Settings
from app.config import settings as global_settings
from app.services.chroma_manager import ChromaClientManager
from app.services.embedding_manager import EmbeddingModelManager
from app.services.vector_search import VectorSearchService
from app.services.vector_store_manager import VectorStoreManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Dependency to get application settings
def get_settings() -> Settings:
    """Dependency to get application settings."""
    return global_settings


def get_chroma_client_manager(request: Request) -> ChromaClientManager:
    """Get ChromaDB client manager from application state."""
    try:
        return request.app.state.chroma_manager
    except AttributeError as e:
        logger.error(f"ChromaDB manager not found in app state: {e}", exc_info=True)
        raise RuntimeError(
            "Application not properly initialized - ChromaDB manager missing"
        ) from e


def get_embedding_model_manager(request: Request) -> EmbeddingModelManager:
    """Get embedding model manager from application state."""
    try:
        return request.app.state.embedding_manager
    except AttributeError as e:
        logger.error(f"Embedding manager not found in app state: {e}", exc_info=True)
        raise RuntimeError(
            "Application not properly initialized - embedding manager missing"
        ) from e


def get_vector_store_manager(request: Request) -> VectorStoreManager:
    """Get vector store manager from application state."""
    try:
        return request.app.state.vector_store_manager
    except AttributeError as e:
        logger.error(f"Vector store manager not found in app state: {e}", exc_info=True)
        raise RuntimeError(
            "Application not properly initialized - vector store manager missing"
        ) from e


# Dependency to get the core VectorSearchService instance
def get_vector_search_service(
    settings: Settings = Depends(get_settings),
    request: Request = None,
) -> VectorSearchService:
    """Dependency to get VectorSearchService instance."""
    if request is None:
        raise ValueError("Request parameter is required")

    try:
        return VectorSearchService(
            settings=settings,
            chroma_manager=request.app.state.chroma_manager,
            embedding_manager=request.app.state.embedding_manager,
            vector_store_manager=request.app.state.vector_store_manager,
        )
    except AttributeError as e:
        logger.error(f"Required managers not found in app state: {e}", exc_info=True)
        raise RuntimeError(
            "Application not properly initialized - required managers missing"
        ) from e
    except Exception as e:
        logger.error(f"Failed to create VectorSearchService: {e}", exc_info=True)
        raise RuntimeError(
            f"Failed to initialize vector search service: {str(e)}"
        ) from e

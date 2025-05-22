import chromadb
from fastapi import Depends
from sentence_transformers import SentenceTransformer
import logging

from app.config import Settings
from app.config import settings as global_settings
from app.services.vector_search import (
    VectorSearchService,
    get_chroma_client,
    get_embedding_model,
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dependency to get application settings
def get_settings() -> Settings:
    return global_settings


# Dependency to get the core VectorSearchService instance
def get_vector_search_service(
    settings: Settings = Depends(get_settings),
    embedding_model: SentenceTransformer = Depends(get_embedding_model),
    chroma_client: chromadb.ClientAPI = Depends(get_chroma_client),  # Use client
) -> VectorSearchService:
    """Provides an instance of the VectorSearchService."""
    logger.debug(
        f"Creating VectorSearchService with collection '{settings.CHROMA_COLLECTION_NAME}', "
        f"top_k '{settings.TOP_K_RESULTS}', threshold '{settings.DISTANCE_THRESHOLD}'"
    )
    return VectorSearchService(
        embedding_model=embedding_model,
        chroma_client=chroma_client,  # Pass client
        collection_name=settings.CHROMA_COLLECTION_NAME,  # Pass collection name from settings
        top_k=settings.TOP_K_RESULTS,
        distance_threshold=settings.DISTANCE_THRESHOLD,  # Make sure this is in your Settings model
    )

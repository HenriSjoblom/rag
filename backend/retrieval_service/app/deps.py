from fastapi import Depends
import chromadb
from sentence_transformers import SentenceTransformer

from app.config import Settings, settings as global_settings
from app.services.vector_search import (
    VectorSearchService,
    get_embedding_model,
    get_chroma_collection
)

# Dependency to get application settings
def get_settings() -> Settings:
    return global_settings

# Dependency to get the core VectorSearchService instance
def get_vector_search_service(
    settings: Settings = Depends(get_settings),
    embedding_model: SentenceTransformer = Depends(get_embedding_model), # Get from lifespan
    chroma_collection: chromadb.Collection = Depends(get_chroma_collection) # Get from lifespan
) -> VectorSearchService:
    """Provides an instance of the VectorSearchService."""
    print("Creating VectorSearchService instance...")
    print(f"Settings in creating vector search service used: {settings.CHROMA_PATH}, {settings.CHROMA_COLLECTION_NAME}")
    return VectorSearchService(
        embedding_model=embedding_model,
        chroma_collection=chroma_collection,
        top_k=settings.TOP_K_RESULTS
    )
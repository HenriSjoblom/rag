import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import HTTPException, status
import numpy as np
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global instances managed by lifespan
_embedding_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.ClientAPI] = None
_chroma_collection: Optional[chromadb.Collection] = None

async def get_embedding_model() -> SentenceTransformer:
    """Dependency to get the loaded embedding model."""
    if _embedding_model is None:
        raise RuntimeError("Embedding model not initialized.")
    return _embedding_model

async def get_chroma_collection() -> chromadb.Collection:
    """Dependency to get the ChromaDB collection object."""
    if _chroma_collection is None:
        raise RuntimeError("ChromaDB collection not initialized.")
    return _chroma_collection

@asynccontextmanager
async def lifespan_retrieval_service(app, model_name: str, chroma_path: str, collection_name: str):
    """Manages the lifespan of embedding model and ChromaDB client."""
    global _embedding_model, _chroma_client, _chroma_collection
    # Load Model
    print("Loading embedding model...")
    print(f"Collection name: {collection_name}")
    print(f"ChromaDB path: {chroma_path}")
    logger.info(f"Loading embedding model: {model_name}...")
    try:
        _embedding_model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load embedding model '{model_name}': {e}", exc_info=True)
        # Optionally re-raise or handle critical failure
        raise RuntimeError(f"Failed to load embedding model: {e}") from e

    # Connect to ChromaDB
    try:
        print("Trying to connect to ChromaDB...")
        _chroma_client = chromadb.PersistentClient(path=chroma_path, settings=ChromaSettings(allow_reset=True))

        print("ChromaDB connection successful.")
        logger.info(f"Getting or creating ChromaDB collection: {collection_name}...")
        _chroma_collection = _chroma_client.get_or_create_collection(name=collection_name)
        logger.info(f"ChromaDB collection '{collection_name}' ready.")

    except Exception as e:
        logger.error(f"Failed to connect to or setup ChromaDB: {e}", exc_info=True)
        _chroma_client = None
        _chroma_collection = None
        raise RuntimeError(f"Failed to initialize ChromaDB: {e}") from e

    try:
        yield # Application runs
    finally:
        # Cleanup
        logger.info("Shutting down retrieval service resources...")
        if _chroma_client:
            try:
                print("Resetting ChromaDB client...")
                _chroma_client.reset() # This should release file locks
                print("ChromaDB client reset successfully.")
                logger.info("ChromaDB client reset.")
                time.sleep(5.5)
            except Exception as e:
                print(f"Error resetting ChromaDB client: {e}")
                logger.error(f"Error resetting ChromaDB client: {e}", exc_info=True)

        _embedding_model = None # Allow garbage collection
        _chroma_client = None
        _chroma_collection = None
        logger.info("Retrieval service resources released.")


class VectorSearchService:
    def __init__(
        self,
        embedding_model: SentenceTransformer,
        chroma_collection: chromadb.Collection,
        top_k: int,
    ):
        self.embedding_model = embedding_model
        self.chroma_collection = chroma_collection
        self.top_k = top_k

    def _embed_query(self, query: str) -> List[float]:
        """Generates embedding for the given query text."""
        # Note: encode() returns a numpy array, convert to list for ChromaDB
        try:
            embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            # Some models might output multi-dimensional arrays, ensure it's 1D
            if embedding.ndim > 1:
                 embedding = embedding.flatten() # Or handle appropriately based on model output
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to embed query: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate query embedding."
            ) from e

    async def search(self, query: str) -> List[str]:
        """
        Embeds the query and performs similarity search in ChromaDB.

        Returns:
            A list of relevant document chunk texts.
        """
        logger.info(f"Search service using collection: '{self.chroma_collection.name}'")
        logger.info(f"Embedding query: '{query[:50]}...'")
        query_embedding = self._embed_query(query)

        logger.info(f"Querying ChromaDB collection '{self.chroma_collection.name}' for top {self.top_k} results...")
        try:
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding], # Chroma expects a list of embeddings
                n_results=self.top_k,
                include=['documents'] # We only need the document text content
            )
            logger.info(f"ChromaDB query successful. Found results: {results}")

            # Extract the document texts from the results
            # Results should now be a dictionary
            retrieved_docs = results.get('documents', [[]])[0]
            if not retrieved_docs:
                 logger.warning("No documents found in ChromaDB for the query.")
                 return []

            logger.info(f"Retrieved {len(retrieved_docs)} document chunks.")
            return retrieved_docs # Return the list of chunk texts

        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}", exc_info=True)
            # Check for specific ChromaDB errors if possible
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to query vector database: {e}",
            ) from e
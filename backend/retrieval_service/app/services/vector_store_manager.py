import logging
from typing import Optional

import chromadb
from app.config import Settings
from app.services.chroma_manager import ChromaClientManager
from app.services.embedding_manager import EmbeddingModelManager
from chromadb.errors import ChromaError
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages vector store operations with ChromaDB."""

    def __init__(
        self,
        settings: Settings,
        chroma_manager: ChromaClientManager,
        embedding_manager: EmbeddingModelManager,
    ):
        self.settings = settings
        self.chroma_manager = chroma_manager
        self.embedding_manager = embedding_manager
        self._collection: Optional[chromadb.Collection] = None
        logger.info("VectorStoreManager initialized.")

    def get_collection(self) -> chromadb.Collection:
        """Get or create the ChromaDB collection."""
        try:
            client = self.chroma_manager.get_client()

            # Use the managed embedding model for consistency
            chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.settings.EMBEDDING_MODEL_NAME
            )

            collection_name = self.settings.CHROMA_COLLECTION_NAME
            if not collection_name or not collection_name.strip():
                raise ValueError("Collection name cannot be empty")

            logger.info(
                f"Getting or creating ChromaDB collection '{collection_name}' with EF for model '{self.settings.EMBEDDING_MODEL_NAME}'..."
            )

            collection = client.get_or_create_collection(
                name=collection_name,
                embedding_function=chroma_ef,
            )

            logger.info(
                f"ChromaDB collection '{collection.name}' (ID: {collection.id}) is available."
            )
            return collection

        except ValueError as e:
            logger.error(f"Invalid collection configuration: {e}", exc_info=True)
            raise RuntimeError(f"Collection configuration error: {str(e)}") from e
        except ChromaError as e:
            logger.error(f"ChromaDB error getting collection: {e}", exc_info=True)
            raise RuntimeError(f"ChromaDB collection error: {str(e)}") from e
        except ConnectionError as e:
            logger.error(f"Connection error getting collection: {e}", exc_info=True)
            raise RuntimeError(
                f"Cannot connect to ChromaDB for collection operations: {str(e)}"
            ) from e
        except Exception as e:
            logger.error(
                f"Failed to get ChromaDB collection '{self.settings.CHROMA_COLLECTION_NAME}': {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to get ChromaDB collection '{self.settings.CHROMA_COLLECTION_NAME}': {str(e)}"
            ) from e

    def get_embedding_model(self) -> SentenceTransformer:
        """Get the managed embedding model instance."""
        try:
            return self.embedding_manager.get_model()
        except Exception as e:
            logger.error(f"Failed to get embedding model: {e}", exc_info=True)
            raise RuntimeError(f"Cannot access embedding model: {str(e)}") from e

    def reset(self):
        """Reset the collection reference."""
        try:
            self._collection = None
            logger.info("Vector store manager reset.")
        except Exception as e:
            logger.error(f"Error during vector store manager reset: {e}", exc_info=True)
            self._collection = None  # Force cleanup
            raise RuntimeError(f"Failed to reset vector store manager: {str(e)}") from e

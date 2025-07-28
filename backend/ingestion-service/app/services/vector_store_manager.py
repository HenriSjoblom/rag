import logging
from typing import Optional

from app.config import Settings
from chromadb.errors import InvalidCollectionException
from langchain_chroma import Chroma
from app.services.chroma_manager import ChromaClientManager
from app.services.embedding_manager import EmbeddingModelManager

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages vector store instances."""

    def __init__(
        self,
        settings: Settings,
        chroma_manager: ChromaClientManager,
        embedding_manager: EmbeddingModelManager,
    ):
        self.settings = settings
        self.chroma_manager = chroma_manager
        self.embedding_manager = embedding_manager
        self._vector_store: Optional[Chroma] = None

    def get_vector_store(self) -> Chroma:
        if self._vector_store is None:
            self._vector_store = self._create_vector_store()
        return self._vector_store

    def _create_vector_store(self) -> Chroma:
        logger.info("Initializing LangChain Chroma vector store...")
        client = self.chroma_manager.get_client()
        embedding_function = self.embedding_manager.get_model()

        try:
            # Ensure the collection exists
            logger.info(
                f"Ensuring collection '{self.settings.CHROMA_COLLECTION_NAME}' exists..."
            )
            try:
                collection = client.get_collection(
                    name=self.settings.CHROMA_COLLECTION_NAME
                )
                logger.info(
                    f"Collection '{self.settings.CHROMA_COLLECTION_NAME}' exists with {collection.count()} documents."
                )
            except InvalidCollectionException:
                logger.info(
                    f"Creating new collection '{self.settings.CHROMA_COLLECTION_NAME}'..."
                )
                collection = client.create_collection(
                    name=self.settings.CHROMA_COLLECTION_NAME
                )
                logger.info(
                    f"Collection '{self.settings.CHROMA_COLLECTION_NAME}' created successfully."
                )

            vector_store = Chroma(
                client=client,
                collection_name=self.settings.CHROMA_COLLECTION_NAME,
                embedding_function=embedding_function,
            )
            logger.info(
                f"Vector store connected to collection '{self.settings.CHROMA_COLLECTION_NAME}'."
            )
            return vector_store
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize vector store: {e}") from e

    def reset(self):
        """Reset the vector store instance."""
        self._vector_store = None
import logging
from typing import Optional

import chromadb
from app.config import Settings
from chromadb.errors import InvalidCollectionException
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

logger = logging.getLogger(__name__)


class ChromaClientManager:
    """Manages ChromaDB client connections."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: Optional[chromadb.ClientAPI] = None

    def get_client(self) -> chromadb.ClientAPI:
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> chromadb.ClientAPI:
        chroma_mode = self.settings.CHROMA_MODE.lower()

        if chroma_mode == "local":
            if not self.settings.CHROMA_PATH:
                raise ValueError("CHROMA_PATH is required for local mode.")
            logger.info(
                f"Connecting to local ChromaDB at path: {self.settings.CHROMA_PATH}"
            )
            return chromadb.PersistentClient(path=self.settings.CHROMA_PATH)

        elif chroma_mode == "docker":
            if not self.settings.CHROMA_HOST or not self.settings.CHROMA_PORT:
                raise ValueError(
                    "CHROMA_HOST and CHROMA_PORT are required for docker mode."
                )
            logger.info(
                f"Connecting to ChromaDB at {self.settings.CHROMA_HOST}:{self.settings.CHROMA_PORT}"
            )
            return chromadb.HttpClient(
                host=self.settings.CHROMA_HOST, port=self.settings.CHROMA_PORT
            )

        else:
            raise ValueError(
                f"Invalid CHROMA_MODE: {chroma_mode}. Must be 'local' or 'docker'."
            )

    def reset(self):
        """Reset the client connection."""
        self._client = None


class EmbeddingModelManager:
    """Manages embedding model instances."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: Optional[SentenceTransformerEmbeddings] = None

    def get_model(self) -> SentenceTransformerEmbeddings:
        if self._model is None:
            self._model = self._create_model()
        return self._model

    def _create_model(self) -> SentenceTransformerEmbeddings:
        logger.info(f"Loading embedding model: {self.settings.EMBEDDING_MODEL_NAME}")
        try:
            model = SentenceTransformerEmbeddings(
                model_name=self.settings.EMBEDDING_MODEL_NAME
            )
            logger.info("Embedding model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load embedding model: {e}") from e


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

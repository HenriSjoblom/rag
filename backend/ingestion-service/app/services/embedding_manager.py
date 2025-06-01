import logging
from typing import Optional

from app.config import Settings

from langchain_community.embeddings import SentenceTransformerEmbeddings

logger = logging.getLogger(__name__)

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


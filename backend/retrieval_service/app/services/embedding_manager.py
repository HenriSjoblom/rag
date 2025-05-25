import logging
from typing import Optional

from app.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModelManager:
    """Manages SentenceTransformer embedding models."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: Optional[SentenceTransformer] = None
        logger.info("EmbeddingModelManager initialized.")

    def get_model(self) -> SentenceTransformer:
        """Get or load the embedding model."""
        if self._model is None:
            logger.info(
                f"Loading embedding model: {self.settings.EMBEDDING_MODEL_NAME}"
            )
            try:
                self._model = SentenceTransformer(self.settings.EMBEDDING_MODEL_NAME)
                logger.info("Embedding model loaded successfully.")
            except FileNotFoundError as e:
                logger.error(f"Embedding model file not found: {e}", exc_info=True)
                raise RuntimeError(
                    f"Embedding model '{self.settings.EMBEDDING_MODEL_NAME}' not found. Please check the model name."
                ) from e
            except MemoryError as e:
                logger.error(
                    f"Insufficient memory to load embedding model: {e}", exc_info=True
                )
                raise RuntimeError(
                    f"Insufficient memory to load embedding model '{self.settings.EMBEDDING_MODEL_NAME}'"
                ) from e
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}", exc_info=True)
                raise RuntimeError(
                    f"Failed to load embedding model '{self.settings.EMBEDDING_MODEL_NAME}': {str(e)}"
                ) from e
        return self._model

    def reset(self):
        """Reset the model instance."""
        try:
            self._model = None
            logger.info("Embedding model manager reset.")
        except Exception as e:
            logger.error(
                f"Error during embedding model manager reset: {e}", exc_info=True
            )
            raise RuntimeError(
                f"Failed to reset embedding model manager: {str(e)}"
            ) from e

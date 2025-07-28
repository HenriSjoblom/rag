import logging
from typing import Optional

import chromadb
from app.config import Settings

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



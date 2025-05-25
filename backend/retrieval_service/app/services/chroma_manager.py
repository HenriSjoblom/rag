import logging
import os
import socket
from typing import Optional

import chromadb
from app.config import Settings
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import ChromaError

logger = logging.getLogger(__name__)


class ChromaClientManager:
    """Manages ChromaDB client connections."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: Optional[chromadb.ClientAPI] = None
        logger.info("ChromaClientManager initialized.")

    def get_client(self) -> chromadb.ClientAPI:
        """Get or create ChromaDB client."""
        if self._client is None:
            logger.info(
                f"Connecting to ChromaDB in '{self.settings.CHROMA_MODE}' mode..."
            )
            try:
                if self.settings.CHROMA_MODE == "local":
                    self._connect_local()
                elif self.settings.CHROMA_MODE == "docker":
                    self._connect_docker()
                else:
                    raise ValueError(
                        f"Invalid CHROMA_MODE: {self.settings.CHROMA_MODE}. Must be 'local' or 'docker'."
                    )
                logger.info("ChromaDB client connected successfully.")
            except ValueError as e:
                logger.error(f"Configuration error: {e}", exc_info=True)
                raise RuntimeError(f"ChromaDB configuration error: {str(e)}") from e
            except ChromaError as e:
                logger.error(f"ChromaDB specific error: {e}", exc_info=True)
                raise RuntimeError(f"ChromaDB connection error: {str(e)}") from e
            except ConnectionError as e:
                logger.error(f"Network connection error: {e}", exc_info=True)
                raise RuntimeError(f"Failed to connect to ChromaDB: {str(e)}") from e
            except Exception as e:
                logger.error(
                    f"Unexpected error connecting to ChromaDB: {e}", exc_info=True
                )
                raise RuntimeError(f"Failed to connect to ChromaDB: {str(e)}") from e
        return self._client

    def _connect_local(self):
        """Connect to local ChromaDB instance."""
        if not self.settings.CHROMA_PATH:
            raise ValueError("CHROMA_PATH is required for local mode.")

        chroma_path = self.settings.CHROMA_PATH
        logger.info(f"ChromaDB local path: {chroma_path}")

        try:
            # Check if directory exists and is writable
            os.makedirs(chroma_path, exist_ok=True)
            if not os.access(chroma_path, os.W_OK):
                raise PermissionError(
                    f"No write permission for ChromaDB path: {chroma_path}"
                )

            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=ChromaSettings(allow_reset=True),
            )
        except PermissionError as e:
            logger.error(f"Permission error for ChromaDB path: {e}", exc_info=True)
            raise RuntimeError(
                f"Permission denied for ChromaDB path '{chroma_path}': {str(e)}"
            ) from e
        except OSError as e:
            logger.error(f"File system error for ChromaDB path: {e}", exc_info=True)
            raise RuntimeError(
                f"File system error for ChromaDB path '{chroma_path}': {str(e)}"
            ) from e

    def _connect_docker(self):
        """Connect to Docker ChromaDB instance."""
        if not self.settings.CHROMA_HOST:
            raise ValueError("CHROMA_HOST is required for docker mode.")

        port = self.settings.CHROMA_PORT or 8000
        host = self.settings.CHROMA_HOST
        logger.info(f"ChromaDB Docker host: {host}, port: {port}")

        try:
            # Test network connectivity first
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)  # 5 second timeout
            result = sock.connect_ex(
                (host.replace("http://", "").replace("https://", ""), port)
            )
            sock.close()

            if result != 0:
                raise ConnectionError(f"Cannot reach ChromaDB at {host}:{port}")

            self._client = chromadb.HttpClient(host=host, port=port)
        except socket.gaierror as e:
            logger.error(f"DNS resolution error for ChromaDB host: {e}", exc_info=True)
            raise RuntimeError(
                f"Cannot resolve ChromaDB host '{host}': {str(e)}"
            ) from e
        except socket.timeout as e:
            logger.error(f"Connection timeout to ChromaDB: {e}", exc_info=True)
            raise RuntimeError(
                f"Connection timeout to ChromaDB at {host}:{port}"
            ) from e
        except ConnectionError as e:
            logger.error(f"Connection error to ChromaDB: {e}", exc_info=True)
            raise RuntimeError(
                f"Cannot connect to ChromaDB at {host}:{port}: {str(e)}"
            ) from e

    def reset(self):
        """Reset the client connection."""
        try:
            if self._client and hasattr(self._client, "reset"):
                try:
                    logger.info("Resetting ChromaDB client...")
                    self._client.reset()
                    logger.info("ChromaDB client reset successfully.")
                except ChromaError as e:
                    logger.error(f"ChromaDB error during reset: {e}", exc_info=True)
                    # Continue with cleanup even if reset fails
                except Exception as e:
                    logger.error(f"Error resetting ChromaDB client: {e}", exc_info=True)
                    # Continue with cleanup even if reset fails
            self._client = None
            logger.info("ChromaDB client manager reset.")
        except Exception as e:
            logger.error(
                f"Unexpected error during ChromaDB client reset: {e}", exc_info=True
            )
            self._client = None  # Force cleanup
            raise RuntimeError(
                f"Failed to reset ChromaDB client manager: {str(e)}"
            ) from e

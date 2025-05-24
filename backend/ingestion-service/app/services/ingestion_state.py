import asyncio
import logging

logger = logging.getLogger(__name__)


class IngestionStateService:
    """Manages ingestion state and concurrency control."""

    def __init__(self):
        self._is_ingesting = False
        self._lock = asyncio.Lock()

    async def is_ingesting(self) -> bool:
        """Check if ingestion is currently running."""
        async with self._lock:
            return self._is_ingesting

    async def start_ingestion(self) -> bool:
        """
        Attempt to start ingestion.

        Returns:
            True if ingestion was started, False if already running
        """
        async with self._lock:
            if self._is_ingesting:
                return False
            self._is_ingesting = True
            logger.info("Ingestion state set to running.")
            return True

    async def stop_ingestion(self):
        """Mark ingestion as completed."""
        async with self._lock:
            self._is_ingesting = False
            logger.info("Ingestion state set to stopped.")

    def reset_state(self):
        """Reset the ingestion state (for cleanup/testing)."""
        self._is_ingesting = False
        logger.info("Ingestion state reset.")

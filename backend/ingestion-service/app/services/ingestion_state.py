import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.models import IngestionStatus

logger = logging.getLogger(__name__)


class IngestionStateService:
    """Manages ingestion state and concurrency control."""

    def __init__(self):
        self._is_ingesting = False
        self._lock = asyncio.Lock()
        self._last_status = "idle"
        self._last_completed: Optional[str] = None
        self._last_result: Optional[IngestionStatus] = None
        self._errors: List[str] = []

    async def is_ingesting(self) -> bool:
        """Check if ingestion is currently running."""
        async with self._lock:
            return self._is_ingesting

    # Add alias for backward compatibility with tests
    async def is_processing(self) -> bool:
        """Alias for is_ingesting for backward compatibility."""
        return await self.is_ingesting()

    async def start_ingestion(self) -> bool:
        """Attempt to start ingestion."""
        async with self._lock:
            if self._is_ingesting:
                return False

            self._is_ingesting = True
            self._last_status = "processing"
            self._errors = []
            logger.info("Ingestion state set to running.")
            return True

    async def stop_ingestion(
        self,
        result: Optional[IngestionStatus] = None,
        errors: Optional[List[str]] = None,
    ):
        """Mark ingestion as completed."""
        async with self._lock:
            self._is_ingesting = False
            self._last_completed = datetime.utcnow().isoformat()
            self._last_result = result
            self._errors = errors or []
            self._last_status = (
                "completed" if not self._errors else "completed_with_errors"
            )
            logger.info("Ingestion state set to stopped.")

    async def get_status(self) -> Dict[str, Any]:
        """Get current ingestion status."""
        async with self._lock:
            return {
                "is_processing": self._is_ingesting,
                "status": self._last_status,
                "last_completed": self._last_completed,
                "documents_processed": self._last_result.documents_processed
                if self._last_result
                else None,
                "chunks_added": self._last_result.chunks_added
                if self._last_result
                else None,
                "errors": self._errors,
            }

    def reset_state(self):
        """Reset the ingestion state."""
        self._is_ingesting = False
        self._last_status = "idle"
        self._last_completed = None
        self._last_result = None
        self._errors = []
        logger.info("Ingestion state reset.")

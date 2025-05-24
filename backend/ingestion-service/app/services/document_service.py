import logging
from pathlib import Path
from typing import List

from app.config import Settings
from app.models import DocumentDetail, DocumentListResponse

logger = logging.getLogger(__name__)


class DocumentService:
    """Handles document listing and management operations."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.source_directory = Path(settings.SOURCE_DIRECTORY)

    def list_documents(self) -> DocumentListResponse:
        """
        Lists all PDF documents in the source directory.

        Returns:
            DocumentListResponse with count, documents list, and source directory path
        """
        logger.info(
            f"Listing PDF documents from source directory: '{self.source_directory}'"
        )

        if not self.source_directory.exists() or not self.source_directory.is_dir():
            logger.warning(
                f"Source directory '{self.source_directory}' not found or is not a directory."
            )
            return DocumentListResponse(
                count=0, documents=[], source_directory=str(self.source_directory)
            )

        document_details: List[DocumentDetail] = []

        try:
            # Recursively find all .pdf files
            pdf_files = list(self.source_directory.rglob("*.pdf"))

            for pdf_file in pdf_files:
                if pdf_file.is_file():  # Ensure it's a file
                    document_details.append(DocumentDetail(name=pdf_file.name))

            logger.info(
                f"Found {len(document_details)} PDF documents in '{self.source_directory}'."
            )

            return DocumentListResponse(
                count=len(document_details),
                documents=document_details,
                source_directory=str(self.source_directory),
            )

        except Exception as e:
            logger.error(
                f"Error listing documents in '{self.source_directory}': {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Failed to list documents: {str(e)}") from e

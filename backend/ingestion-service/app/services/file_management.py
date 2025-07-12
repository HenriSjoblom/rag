import logging
import shutil
from pathlib import Path
from typing import List

from app.config import Settings
from app.models import DocumentDetail, DocumentListResponse
from fastapi import HTTPException, UploadFile, status

logger = logging.getLogger(__name__)


class FileManagementService:
    """Handles all file operations including document listing, validation, and file uploads."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.source_directory = Path(settings.SOURCE_DIRECTORY)
        # Set default max file size if not configured (500MB default)
        self.max_file_size_mb = getattr(settings, "MAX_FILE_SIZE_MB", 500)
        self._ensure_source_directory()
        logger.info(
            f"FileManagementService initialized with source directory: {self.source_directory}"
        )

    def _ensure_source_directory(self) -> None:
        """Ensure source directory exists."""
        try:
            self.source_directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create source directory: {e}")
            raise RuntimeError(f"Failed to create source directory: {e}") from e

    def list_documents(self) -> DocumentListResponse:
        """
        Lists all PDF documents in the source directory.

        Returns:
            DocumentListResponse with count and documents list
        """
        logger.info(
            f"Listing PDF documents from source directory: '{self.source_directory}'"
        )

        if not self.source_directory.exists() or not self.source_directory.is_dir():
            logger.warning(
                f"Source directory '{self.source_directory}' not found or is not a directory."
            )
            return DocumentListResponse(count=0, documents=[])

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
                count=len(document_details), documents=document_details
            )

        except Exception as e:
            logger.error(
                f"Error listing documents in '{self.source_directory}': {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Failed to list documents: {str(e)}") from e

    async def save_uploaded_file(self, file: UploadFile) -> tuple[Path, bool]:
        """
        Validates and saves uploaded file with improved error handling and duplicate detection.

        Args:
            file: The UploadFile object from FastAPI.

        Returns:
            Tuple of (file_location, was_overwritten)

        Raises:
            HTTPException: If validation fails or an error occurs during saving.
        """
        # Validation
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided."
            )

        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are allowed.",
            )

        # Check file size if available
        if hasattr(file, "size") and file.size:
            max_size_bytes = self.max_file_size_mb * 1024 * 1024
            if file.size > max_size_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File too large. Maximum size: {self.max_file_size_mb}MB",
                )

        file_location = self.source_directory / file.filename

        # Check if file already exists
        was_overwritten = file_location.exists()
        if was_overwritten:
            logger.info(f"File {file.filename} already exists, will be overwritten.")

        try:
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            action = "overwritten" if was_overwritten else "saved"
            logger.info(f"File {action}: {file.filename}")
            return file_location, was_overwritten
        except Exception as e:
            # Cleanup on failure
            if file_location.exists():
                try:
                    file_location.unlink()
                except Exception:
                    pass
            logger.error(f"Failed to save file {file.filename}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save file.",
            )

    def count_documents(self) -> int:
        """
        Counts the number of PDF documents in the source directory.

        Returns:
            Number of PDF files found
        """
        try:
            if self.source_directory.exists() and self.source_directory.is_dir():
                pdf_files = list(self.source_directory.rglob("*.pdf"))
                return len([f for f in pdf_files if f.is_file()])
            return 0
        except Exception as e:
            logger.warning(f"Could not count documents in source directory: {e}")
            return 0

    def count_all_files(self) -> int:
        """
        Counts all files in the source directory (not just PDFs).

        Returns:
            Number of files found
        """
        try:
            if self.source_directory.exists() and self.source_directory.is_dir():
                all_files = list(self.source_directory.rglob("*.*"))
                return len([f for f in all_files if f.is_file()])
            return 0
        except Exception as e:
            logger.warning(f"Could not count all files in source directory: {e}")
            return 0

    def clear_all_files(self) -> int:
        """
        Deletes all files in the source directory.

        Returns:
            Number of files deleted
        """
        deleted_count = 0
        try:
            if self.source_directory.exists() and self.source_directory.is_dir():
                all_files = list(self.source_directory.rglob("*"))
                for file_path in all_files:
                    if file_path.is_file():
                        try:
                            file_path.unlink()
                            deleted_count += 1
                            logger.debug(f"Deleted file: {file_path}")
                        except Exception as e:
                            logger.warning(f"Failed to delete file {file_path}: {e}")

                logger.info(f"Deleted {deleted_count} files from source directory.")
            return deleted_count
        except Exception as e:
            logger.error(
                f"Error clearing files from source directory: {e}", exc_info=True
            )
            raise RuntimeError(f"Failed to clear files: {e}") from e

import logging
import os
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
        logger.info(
            f"FileManagementService initialized with source directory: {self.source_directory}"
        )

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

    async def save_uploaded_file(self, file: UploadFile) -> Path:
        """
        Validates the uploaded file and saves it to the source directory.

        Args:
            file: The UploadFile object from FastAPI.

        Returns:
            The Path object to the saved file location.

        Raises:
            HTTPException: If validation fails or an error occurs during saving.
        """
        if not file.filename:
            logger.warning("No filename provided with the uploaded file.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No filename provided with the uploaded file.",
            )

        if not file.filename.lower().endswith(".pdf"):
            logger.warning(
                f"Invalid file type for '{file.filename}'. Only PDF documents are allowed."
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Only PDF documents are allowed.",
            )

        # Ensure the directory exists
        try:
            self.source_directory.mkdir(parents=True, exist_ok=True)
        except Exception as e_mkdir:
            logger.error(
                f"Failed to create source directory '{self.source_directory}': {e_mkdir}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to prepare source directory for uploads.",
            )

        file_location = self.source_directory / file.filename

        try:
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(
                f"Successfully saved file: '{file.filename}' to '{file_location}'"
            )
            return file_location
        except Exception as e_save:
            logger.error(
                f"Failed to save uploaded file '{file.filename}' to '{file_location}': {e_save}",
                exc_info=True,
            )
            # Attempt to remove partially written file if save failed
            if file_location.exists():
                try:
                    os.remove(file_location)
                    logger.info(f"Removed partially written file: '{file_location}'")
                except Exception as e_remove:
                    logger.error(
                        f"Failed to remove partially written file '{file_location}': {e_remove}"
                    )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save uploaded file: {str(e_save)}",
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

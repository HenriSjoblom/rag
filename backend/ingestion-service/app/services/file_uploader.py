import logging
import shutil
from pathlib import Path
import os

from fastapi import HTTPException, UploadFile, status

logger = logging.getLogger(__name__)

class FileUploadService:
    """Handles validation and saving of uploaded files."""

    def __init__(self, source_directory_str: str):
        self.source_directory = Path(source_directory_str)
        logger.info(f"FileUploadService initialized with source directory: {self.source_directory}")

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
            logger.warning(f"Invalid file type for '{file.filename}'. Only PDF documents are allowed.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Only PDF documents are allowed.",
            )

        if not self.source_directory.exists() or not self.source_directory.is_dir():
            logger.error(
                f"Source directory '{self.source_directory}' does not exist or is not a directory. Cannot save uploaded file."
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server configuration error: Source directory for uploads not found.",
            )

        # Ensure the directory exists (it should, but defensive check)
        try:
            self.source_directory.mkdir(parents=True, exist_ok=True)
        except Exception as e_mkdir:
            logger.error(f"Failed to create source directory '{self.source_directory}': {e_mkdir}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to prepare source directory for uploads.",
            )


        file_location = self.source_directory / file.filename

        try:
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"Successfully saved file: '{file.filename}' to '{file_location}'")
            return file_location
        except Exception as e_save:
            logger.error(
                f"Failed to save uploaded file '{file.filename}' to '{file_location}': {e_save}", exc_info=True
            )
            # Attempt to remove partially written file if save failed
            if file_location.exists():
                try:
                    os.remove(file_location)
                    logger.info(f"Removed partially written file: '{file_location}'")
                except Exception as e_remove:
                    logger.error(f"Failed to remove partially written file '{file_location}': {e_remove}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save uploaded file: {str(e_save)}",
            )
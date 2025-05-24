import logging
import os
from pathlib import Path
from typing import List, Tuple

import app.services.ingestion_processor as ingestion_processor_module
from app.config import Settings
from app.services.ingestion_processor import get_chroma_client

logger = logging.getLogger(__name__)


class CollectionManagerService:
    """Handles ChromaDB collection and source file management operations."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.source_directory = Path(settings.SOURCE_DIRECTORY)
        self.collection_name = settings.CHROMA_COLLECTION_NAME

    def clear_source_files(self) -> Tuple[bool, int, List[str]]:
        """
        Deletes all files from the source directory.

        Returns:
            Tuple of (success, deleted_count, error_messages)
        """
        deleted_files_count = 0
        error_messages = []

        logger.info(
            f"Attempting to delete all files from source directory: '{self.source_directory}'"
        )

        if not self.source_directory.exists() or not self.source_directory.is_dir():
            message = f"Source directory '{self.source_directory}' not found or is not a directory. No files deleted."
            logger.warning(message)
            return (
                True,
                0,
                [message],
            )  # Consider this success as there's nothing to delete

        try:
            for item in self.source_directory.iterdir():
                if item.is_file():
                    try:
                        os.remove(item)
                        deleted_files_count += 1
                        logger.debug(f"Deleted file: {item}")
                    except Exception as e:
                        err_msg = f"Failed to delete file {item}: {e}"
                        logger.error(err_msg, exc_info=True)
                        error_messages.append(err_msg)

            success = len(error_messages) == 0
            if success:
                log_msg = f"Successfully deleted {deleted_files_count} file(s) from '{self.source_directory}'."
                logger.info(log_msg)

            return success, deleted_files_count, error_messages

        except Exception as e:
            err_msg = f"An error occurred while deleting files from '{self.source_directory}': {e}"
            logger.error(err_msg, exc_info=True)
            return False, deleted_files_count, [err_msg]

    def clear_chroma_collection(self) -> Tuple[bool, List[str]]:
        """
        Deletes the ChromaDB collection and resets cached instances.

        Returns:
            Tuple of (success, messages)
        """
        messages = []

        logger.info(
            f"Attempting to delete ChromaDB collection: '{self.collection_name}'"
        )

        try:
            client = get_chroma_client(self.settings)
            client.delete_collection(name=self.collection_name)

            msg = f"Successfully deleted ChromaDB collection: '{self.collection_name}'"
            logger.info(msg)
            messages.append(msg)

            # Reset cached vector store instances
            self._reset_vector_store_cache()

            return True, messages

        except Exception as e:
            error_str = str(e).lower()
            if any(
                phrase in error_str
                for phrase in ["not found", "does not exist", "collection"]
            ):
                # Collection doesn't exist - this is the desired state
                msg = f"Collection '{self.collection_name}' not found. No deletion performed."
                logger.info(msg)
                messages.append(msg)

                # Still reset cache in case it was stale
                self._reset_vector_store_cache()

                return True, messages
            else:
                # Actual error occurred
                err_msg = f"Failed to delete collection '{self.collection_name}': {e}"
                logger.error(err_msg, exc_info=True)
                return False, [err_msg]

    def _reset_vector_store_cache(self):
        """Resets all cached vector store instances."""
        logger.info("Resetting cached LangChain Chroma vector store instances.")

        # Reset the ingestion processor module cache
        ingestion_processor_module._vector_store = None

    def clear_all(self) -> dict:
        """
        Clears both ChromaDB collection and source files.

        Returns:
            Dictionary with operation results and details
        """
        # Clear source files
        files_success, deleted_count, file_errors = self.clear_source_files()

        # Clear ChromaDB collection
        collection_success, collection_messages = self.clear_chroma_collection()

        # Combine all messages
        all_messages = collection_messages + file_errors
        if files_success and deleted_count > 0:
            all_messages.append(
                f"Successfully deleted {deleted_count} file(s) from source directory."
            )

        # Determine overall success
        overall_success = files_success and collection_success

        return {
            "overall_success": overall_success,
            "collection_deleted": collection_success,
            "source_files_cleared": files_success,
            "files_deleted_count": deleted_count,
            "messages": all_messages,
            "has_errors": len(file_errors) > 0 or not collection_success,
        }

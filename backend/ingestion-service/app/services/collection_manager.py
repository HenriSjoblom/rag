import logging
from typing import Dict

from app.config import Settings
from app.services.file_management import FileManagementService
from app.services.ingestion_processor import (
    ChromaClientManager,
    VectorStoreManager,
)

logger = logging.getLogger(__name__)


class CollectionManagerService:
    """Handles ChromaDB collection management operations."""

    def __init__(
        self,
        settings: Settings,
        chroma_manager: ChromaClientManager = None,
        vector_store_manager: VectorStoreManager = None,
    ):
        self.settings = settings
        self.file_service = FileManagementService(settings)
        self.chroma_manager = chroma_manager or ChromaClientManager(settings)
        self.vector_store_manager = vector_store_manager
        logger.info("CollectionManagerService initialized.")

    def clear_all(self) -> Dict:
        """
        Clears the ChromaDB collection and all source files.
        Recreates the collection after deletion to ensure it exists.
        """
        result = {
            "collection_deleted": False,
            "source_files_cleared": False,
            "files_deleted_count": 0,
            "messages": [],
            "overall_success": False,
        }

        # Clear ChromaDB collection
        try:
            chroma_client = self.chroma_manager.get_client()
            try:
                chroma_client.delete_collection(self.settings.CHROMA_COLLECTION_NAME)
                result["collection_deleted"] = True
                result["messages"].append(
                    f"ChromaDB collection '{self.settings.CHROMA_COLLECTION_NAME}' deleted successfully."
                )
                logger.info(
                    f"Collection '{self.settings.CHROMA_COLLECTION_NAME}' deleted successfully."
                )

                # Recreate the collection to ensure it exists
                chroma_client.create_collection(self.settings.CHROMA_COLLECTION_NAME)
                result["messages"].append(
                    f"ChromaDB collection '{self.settings.CHROMA_COLLECTION_NAME}' recreated successfully."
                )
                logger.info(
                    f"Collection '{self.settings.CHROMA_COLLECTION_NAME}' recreated successfully."
                )

            except Exception as e:
                if "does not exist" in str(e).lower() or "not found" in str(e).lower():
                    # Collection doesn't exist, create it
                    chroma_client.create_collection(
                        self.settings.CHROMA_COLLECTION_NAME
                    )
                    result["collection_deleted"] = True
                    result["messages"].append(
                        f"Collection '{self.settings.CHROMA_COLLECTION_NAME}' not found. Created new collection."
                    )
                    logger.info(
                        f"Collection '{self.settings.CHROMA_COLLECTION_NAME}' not found. Created new collection."
                    )
                else:
                    raise e

            # Reset vector store manager if available
            if self.vector_store_manager:
                self.vector_store_manager.reset()
                logger.info("Vector store manager reset after collection recreation.")

        except Exception as e:
            error_msg = f"Failed to manage ChromaDB collection: {str(e)}"
            result["messages"].append(error_msg)
            logger.error(error_msg, exc_info=True)

        # Clear source files
        try:
            files_deleted = self.file_service.clear_all_files()
            result["source_files_cleared"] = True
            result["files_deleted_count"] = files_deleted
            result["messages"].append(
                f"Cleared {files_deleted} files from source directory."
            )
            logger.info(f"Cleared {files_deleted} files from source directory.")
        except Exception as e:
            result["source_files_cleared"] = False
            result["files_deleted_count"] = 0
            error_msg = f"Failed to clear source files: {str(e)}"
            result["messages"].append(error_msg)
            logger.error(error_msg, exc_info=True)

        # Determine overall success
        result["overall_success"] = (
            result["collection_deleted"] and result["source_files_cleared"]
        )

        return result

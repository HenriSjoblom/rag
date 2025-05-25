import logging

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from app.deps import get_collection_manager_service
from app.services.collection_manager import CollectionManagerService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/collection", tags=["collection_management"])


@router.delete(
    "/",
    status_code=status.HTTP_200_OK,
    summary="Clear ChromaDB Collection and Source Documents",
    description="Deletes the entire ChromaDB collection specified in the service settings AND all files in the source documents directory. This action is irreversible.",
)
async def clear_chroma_collection_and_documents(
    collection_service: CollectionManagerService = Depends(
        get_collection_manager_service
    ),
):
    """
    Deletes the configured ChromaDB collection and all files in the source documents directory.
    Resets the cached vector store instance.
    """
    logger.info("Starting collection and source files cleanup operation.")

    result = collection_service.clear_all()

    if result["overall_success"]:
        final_status_code = status.HTTP_200_OK
        final_message = "ChromaDB collection and source documents cleared successfully."
    elif result["collection_deleted"] or result["source_files_cleared"]:
        final_status_code = status.HTTP_207_MULTI_STATUS
        final_message = "Partial success in clearing resources."
    else:
        final_status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        final_message = "Failed to clear ChromaDB collection and/or source documents."

    return JSONResponse(
        status_code=final_status_code,
        content={
            "message": final_message,
            "files_deleted_count": result["files_deleted_count"],
            "collection_deleted": result["collection_deleted"],
            "source_files_cleared": result["source_files_cleared"],
        },
    )

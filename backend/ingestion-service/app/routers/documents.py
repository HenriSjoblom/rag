import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.deps import get_file_management_service
from app.models import DocumentListResponse
from app.services.file_management import FileManagementService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents_management"])


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List PDF documents in the source directory",
    description="Retrieves a list of all PDF documents found in the configured source directory.",
)
async def list_source_documents(
    file_management_service: FileManagementService = Depends(
        get_file_management_service
    ),
):
    """Lists all PDF documents in the configured source directory."""
    try:
        logger.info("Processing request to list documents")
        if not file_management_service:
            logger.error("FileManagementService dependency is None")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Service dependency not available",
            )

        result = file_management_service.list_documents()
        logger.info(f"Successfully listed {result.count} documents")
        return result
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except RuntimeError as e:
        logger.error(f"Service error while listing documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error while listing documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while listing documents.",
        )

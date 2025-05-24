import logging

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse

from app.deps import (
    get_collection_manager_service,
    get_file_management_service,
    get_ingestion_processor_service,
    get_ingestion_state_service,
)
from app.models import (
    DocumentListResponse,
    IngestionResponse,
    IngestionStatus,
    IngestionStatusResponse,
)
from app.services.collection_manager import CollectionManagerService
from app.services.file_management import FileManagementService
from app.services.ingestion_processor import (
    IngestionProcessorService,
)
from app.services.ingestion_state import IngestionStateService

logger = logging.getLogger(__name__)
router = APIRouter()


async def run_ingestion_background(
    ingestion_service: IngestionProcessorService, state_service: IngestionStateService
):
    """Wrapper function to run ingestion and handle the state."""
    result = None
    errors = []
    try:
        logger.info("Background ingestion task started.")
        ingestion_status: IngestionStatus = ingestion_service.run_ingestion()
        result = ingestion_status
        if ingestion_status.errors:
            errors = ingestion_status.errors
            logger.error(
                f"Background ingestion task finished with errors: {ingestion_status.errors}"
            )
        else:
            logger.info(
                f"Background ingestion task finished successfully. Added {ingestion_status.chunks_added} chunks."
            )
    except Exception as e:
        logger.error(f"Exception during background ingestion task: {e}", exc_info=True)
        errors = [str(e)]
    finally:
        await state_service.stop_ingestion(result=result, errors=errors)
        logger.info("Ingestion task completed and state released.")


@router.post(
    "/upload",
    response_model=IngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a PDF document and trigger ingestion",
    description="Uploads a PDF document to the source directory and then triggers the ingestion process in the background.",
    tags=["ingestion"],
)
async def upload_document_and_ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF document to upload."),
    ingestion_service: IngestionProcessorService = Depends(
        get_ingestion_processor_service
    ),
    file_management_service: FileManagementService = Depends(
        get_file_management_service
    ),
    state_service: IngestionStateService = Depends(get_ingestion_state_service),
):
    if await state_service.is_ingesting():
        logger.warning("Upload attempt while ingestion task is already running.")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An ingestion process is already running. Please wait for it to complete before uploading and triggering a new one.",
        )

    try:
        # Use the FileManagementService to save the file
        file_location = await file_management_service.save_uploaded_file(file)
        logger.info(
            f"File '{file.filename}' processed by FileManagementService and saved to '{file_location}'"
        )
    except HTTPException:
        # Re-raise HTTPException from the service to be returned to the client
        raise
    except Exception as e:
        # Catch any other unexpected errors from the service
        logger.error(
            f"Unexpected error during file save operation via service: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while saving the file.",
        )
    finally:
        await file.close()  # Ensure the uploaded file is closed

    # Start ingestion
    if not await state_service.start_ingestion():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Failed to start ingestion - another process may have started.",
        )

    logger.info("Starting background ingestion task after file upload.")
    background_tasks.add_task(
        run_ingestion_background, ingestion_service, state_service
    )

    docs_found_count = file_management_service.count_documents()

    return IngestionResponse(
        status="File uploaded and ingestion task started.",
        documents_found=docs_found_count,
        message=f"File '{file.filename}' uploaded. Processing documents in the background.",
    )


@router.post(
    "/ingest",
    response_model=IngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger document ingestion process",
    description="Scans the configured source directory, processes documents, and stores them in the vector database. Runs as a background task.",
    tags=["ingestion"],
)
async def trigger_ingestion(
    background_tasks: BackgroundTasks,
    ingestion_service: IngestionProcessorService = Depends(
        get_ingestion_processor_service
    ),
    file_management_service: FileManagementService = Depends(
        get_file_management_service
    ),
    state_service: IngestionStateService = Depends(get_ingestion_state_service),
):
    """
    Triggers the document ingestion pipeline to run in the background.
    Prevents concurrent runs using proper state management.
    """
    if await state_service.is_ingesting():
        logger.warning("Ingestion task is already running.")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An ingestion process is already running. Please wait for it to complete.",
        )

    if not await state_service.start_ingestion():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Failed to start ingestion - another process may have started.",
        )

    logger.info("Starting background ingestion task.")
    background_tasks.add_task(
        run_ingestion_background, ingestion_service, state_service
    )

    # Use FileManagementService to count documents
    docs_found_count = file_management_service.count_documents()

    return IngestionResponse(
        status="Ingestion task started.",
        documents_found=docs_found_count,
        message="Processing documents in the background. Check logs for progress.",
    )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List PDF documents in the source directory",
    description="Retrieves a list of all PDF documents found in the configured source directory.",
    tags=["documents_management"],
)
async def list_source_documents(
    file_management_service: FileManagementService = Depends(
        get_file_management_service
    ),
):
    """Lists all PDF documents in the configured source directory."""
    try:
        return file_management_service.list_documents()
    except RuntimeError as e:
        logger.error(f"Service error while listing documents: {e}")
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


@router.delete(
    "/collection",
    status_code=status.HTTP_200_OK,
    summary="Clear ChromaDB Collection and Source Documents",
    description="Deletes the entire ChromaDB collection specified in the service settings AND all files in the source documents directory. This action is irreversible.",
    tags=["collection_management"],
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

    # Determine HTTP status code based on results
    if result["overall_success"]:
        final_status_code = status.HTTP_200_OK
        final_message = "ChromaDB collection and source documents cleared successfully."
    elif result["collection_deleted"] or result["source_files_cleared"]:
        final_status_code = status.HTTP_207_MULTI_STATUS
        final_message = "Partial success in clearing resources. Check details."
    else:
        final_status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        final_message = "Failed to clear ChromaDB collection and/or source documents."

    return JSONResponse(
        status_code=final_status_code,
        content={
            "message": final_message,
            "details": result["messages"],
            "files_deleted_count": result["files_deleted_count"],
            "collection_deleted": result["collection_deleted"],
            "source_files_cleared": result["source_files_cleared"],
        },
    )


@router.get(
    "/status",
    response_model=IngestionStatusResponse,
    summary="Get ingestion status",
    description="Returns the current status of ingestion process including completion details.",
    tags=["ingestion"],
)
async def get_ingestion_status(
    state_service: IngestionStateService = Depends(get_ingestion_state_service),
):
    """Get the current ingestion status."""
    status_info = await state_service.get_status()
    return IngestionStatusResponse(**status_info)

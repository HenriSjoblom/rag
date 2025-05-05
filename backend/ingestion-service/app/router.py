import logging
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks

from app.models import IngestionResponse, IngestionStatus
from app.services.ingestion_processor import IngestionProcessorService
from app.deps import get_ingestion_processor_service
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory flag to prevent concurrent ingestions
is_ingesting = False

def run_ingestion_background(service: IngestionProcessorService):
    """Wrapper function to run ingestion and handle the lock."""
    global is_ingesting
    try:
        logger.info("Background ingestion task started.")
        ingestion_status: IngestionStatus = service.run_ingestion()
        if ingestion_status.errors:
             logger.error(f"Background ingestion task finished with errors: {ingestion_status.errors}")
        else:
             logger.info(f"Background ingestion task finished successfully. Added {ingestion_status.chunks_added} chunks.")
    except Exception as e:
        logger.error(f"Exception during background ingestion task: {e}", exc_info=True)
    finally:
        is_ingesting = False # Release the lock
        logger.info("Ingestion lock released.")


@router.post(
    "/ingest",
    response_model=IngestionResponse,
    status_code=status.HTTP_202_ACCEPTED, # Use 202 Accepted for background tasks
    summary="Trigger document ingestion process",
    description="Scans the configured source directory, processes documents, and stores them in the vector database. Runs as a background task.",
    tags=["Ingestion"]
)
async def trigger_ingestion(
    background_tasks: BackgroundTasks, # Inject BackgroundTasks
    service: IngestionProcessorService = Depends(get_ingestion_processor_service) # Inject service
):
    """
    Triggers the document ingestion pipeline to run in the background.
    Prevents concurrent runs using a simple in-memory flag.
    """
    global is_ingesting
    if is_ingesting:
        logger.warning("Ingestion task is already running.")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An ingestion process is already running. Please wait for it to complete.",
        )

    # Check if source directory exists before starting background task
    source_path = Path(settings.SOURCE_DIRECTORY)
    if not source_path.exists() or not source_path.is_dir():
         logger.error(f"Ingestion trigger failed: Source directory '{source_path}' not found.")
         raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source directory '{source_path}' not found or is not accessible.",
         )

    # Check for documents (optional, provides better immediate feedback)
    try:
        # Simple check if any files exist directly within the source dir or subdirs
        # This doesn't guarantee they are loadable documents, just that it's not empty.
        doc_files = list(source_path.rglob('*.*')) # Find any file recursively
        docs_found = len([f for f in doc_files if f.is_file()])
        if docs_found == 0:
            logger.warning(f"No documents found in source directory '{source_path}'. Ingestion not started.")
            # Return 200 OK instead of 202 as nothing will run in background
            return IngestionResponse(
                status="No documents found",
                documents_found=0,
                message=f"No files found in the source directory: {source_path}"
            )
    except Exception as e:
        logger.error(f"Error checking source directory contents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check source directory contents: {e}"
        )


    # Set the lock and add the task to the background
    is_ingesting = True
    logger.info("Setting ingestion lock and adding task to background.")
    background_tasks.add_task(run_ingestion_background, service)

    return IngestionResponse(
        status="Ingestion task started",
        documents_found=docs_found, # Include the count found during the check
        message=f"Processing documents from {settings.SOURCE_DIRECTORY} in the background."
    )
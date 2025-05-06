import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse

from app.models import IngestionResponse, IngestionStatus
from app.services.ingestion_processor import IngestionProcessorService
from app.deps import get_ingestion_processor_service

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
    tags=["ingestion"]
)
async def trigger_ingestion(
    background_tasks: BackgroundTasks,
    service: IngestionProcessorService = Depends(get_ingestion_processor_service)
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

    source_path = Path(service.settings.SOURCE_DIRECTORY)

    try:
        # Simple check if any files exist directly within the source dir or subdirs
        # This doesn't guarantee they are loadable documents, just that it's not empty.
        doc_files = list(source_path.rglob('*.*')) # Find any file recursively
        docs_found = len([f for f in doc_files if f.is_file()])
        if docs_found == 0:
            logger.warning(f"No documents found in source directory '{source_path}'. Ingestion not started.")
            # Return 200 OK instead of 202 as nothing will run in background
            response_data = IngestionResponse(
                status="No documents found",
                documents_found=0,
                message="No files found in the source directory"
            )
            simple_content = {
                "status": "No documents found",
                "documents_found": 0,
                "message": "No files found in the source directory"
            }
            print(f"response_data: {response_data}")
            return JSONResponse(
                #content=response_data.model_dump(),
                content=simple_content,
                status_code=status.HTTP_200_OK
            )
    except Exception as e:
        logger.error(f"Error checking source directory contents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check source directory contents: {e}"
        )

    print(f"--- DEBUG [ENDPOINT]: id(background_tasks) = {id(background_tasks)}")

    # Set the lock and add the task to the background
    is_ingesting = True
    logger.info("Setting ingestion lock and adding task to background.")
    background_tasks.add_task(run_ingestion_background, service)
    logger.info(f"Background ingestion task added. Found {docs_found} documents to process.")

    return IngestionResponse(
        status="Ingestion task started",
        documents_found=docs_found,
        message=f"Processing documents in the background."
    )
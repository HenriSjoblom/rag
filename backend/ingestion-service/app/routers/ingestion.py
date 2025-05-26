import logging

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)

from app.deps import (
    get_file_management_service,
    get_ingestion_processor_service,
    get_ingestion_state_service,
)
from app.models import (
    IngestionResponse,
    IngestionStatus,
)
from app.services.file_management import FileManagementService
from app.services.ingestion_processor import IngestionProcessorService
from app.services.ingestion_state import IngestionStateService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ingestion"])


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
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload PDF file for ingestion",
    description="Uploads a PDF file to the server and automatically triggers the ingestion process. ",
)
async def upload_file(
    file: UploadFile = File(...),
    auto_ingest: bool = Query(
        True, description="Automatically trigger ingestion after upload"
    ),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    file_service: FileManagementService = Depends(get_file_management_service),
    ingestion_service: IngestionProcessorService = Depends(
        get_ingestion_processor_service
    ),
    state_service: IngestionStateService = Depends(get_ingestion_state_service),
):
    """Upload a PDF file and optionally trigger ingestion."""

    # Check if file already exists and if it's been processed
    if file.filename:
        processed_files = ingestion_service._get_processed_files()
        if file.filename in processed_files:
            logger.warning(f"File '{file.filename}' already exists. Upload rejected.")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"File '{file.filename}' has already been processed. Upload rejected to prevent duplicates.",
            )

    # Save the uploaded file
    try:
        file_location, was_overwritten = await file_service.save_uploaded_file(file)
        action = "overwritten" if was_overwritten else "uploaded"
        logger.info(
            f"File '{file.filename}' {action} by FileManagementService and saved to '{file_location}'"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during file upload: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file.",  # Generic message loses context
        )

    # Trigger ingestion if requested and not already running
    if auto_ingest:
        if await state_service.is_ingesting():
            logger.info(
                "Ingestion already running, file uploaded but not ingested automatically."
            )
            return {
                "message": f"File '{file.filename}' uploaded successfully. Ingestion is already running.",
                "filename": file.filename,
                "auto_ingest": False,
            }

        if await state_service.start_ingestion():
            logger.info("Starting background ingestion task after file upload.")
            background_tasks.add_task(
                run_ingestion_background, ingestion_service, state_service
            )
            logger.info("Background ingestion task started.")

            return {
                "message": f"File '{file.filename}' uploaded and ingestion started.",
                "filename": file.filename,
                "auto_ingest": True,
            }
        else:
            logger.warning("Failed to start ingestion after upload.")
            return {
                "message": f"File '{file.filename}' uploaded but failed to start ingestion.",
                "filename": file.filename,
                "auto_ingest": False,
            }

    return {
        "message": f"File '{file.filename}' uploaded successfully.",
        "filename": file.filename,
        "auto_ingest": False,
    }


@router.post(
    "/ingest",
    response_model=IngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger document ingestion process",
    description="Scans the configured source directory, processes documents, and stores them in the vector database. Runs as a background task.",
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
    if await state_service.is_ingesting():
        logger.warning("Ingestion task is already running.")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An ingestion process is already running. Please wait for it to complete.",
        )

    # Check for new files before starting ingestion
    processed_files = ingestion_service._get_processed_files()
    docs_found_count = file_management_service.count_documents()

    # Get list of all PDF files
    pdf_files = list(file_management_service.source_directory.rglob("*.pdf"))
    new_files = [f for f in pdf_files if f.name not in processed_files]

    if not new_files and docs_found_count > 0:
        logger.info("No new files to process. All files have already been ingested.")
        return IngestionResponse(
            status="No new files to process.",
            documents_found=docs_found_count,
            message="All documents have already been processed. No ingestion needed.",
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

    if new_files:
        message = f"Processing {len(new_files)} new documents in the background. Check logs for progress."
    else:
        message = "Processing documents in the background. Check logs for progress."

    return IngestionResponse(
        status="Ingestion task started.",
        documents_found=docs_found_count,
        message=message,
    )

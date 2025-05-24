import logging
from pathlib import Path
from typing import List

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

from app.config import Settings
from app.deps import (
    get_collection_manager_service,
    get_file_upload_service,
    get_ingestion_processor_service,
    get_settings,
)
from app.models import (
    DocumentDetail,
    DocumentListResponse,
    IngestionResponse,
    IngestionStatus,
)
from app.services.collection_manager import CollectionManagerService
from app.services.file_uploader import FileUploadService
from app.services.ingestion_processor import (
    IngestionProcessorService,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Global cache for vector store instance
global_vector_store_cache = None
is_ingesting = False


def run_ingestion_background(service: IngestionProcessorService):
    """Wrapper function to run ingestion and handle the lock."""
    global is_ingesting
    try:
        logger.info("Background ingestion task started.")
        ingestion_status: IngestionStatus = service.run_ingestion()
        if ingestion_status.errors:
            logger.error(
                f"Background ingestion task finished with errors: {ingestion_status.errors}"
            )
        else:
            logger.info(
                f"Background ingestion task finished successfully. Added {ingestion_status.chunks_added} chunks."
            )
    except Exception as e:
        logger.error(f"Exception during background ingestion task: {e}", exc_info=True)
    finally:
        is_ingesting = False  # Release the lock
        logger.info("Ingestion lock released.")


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
    file_upload_service: FileUploadService = Depends(
        get_file_upload_service
    ),  # Inject new service
):
    global is_ingesting  # Assuming is_ingesting is defined globally in this file
    if is_ingesting:
        logger.warning("Upload attempt while ingestion task is already running.")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An ingestion process is already running. Please wait for it to complete before uploading and triggering a new one.",
        )

    try:
        # Use the FileUploadService to save the file
        file_location = await file_upload_service.save_uploaded_file(file)
        logger.info(
            f"File '{file.filename}' processed by FileUploadService and saved to '{file_location}'"
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

    # Set the lock and add the task to the background
    is_ingesting = True
    logger.info(
        "Setting ingestion lock and adding task to background after file upload."
    )
    # Pass the IngestionProcessorService instance to the background task
    background_tasks.add_task(run_ingestion_background, ingestion_service)

    docs_found_count = 0
    try:
        # Use the source_directory from the file_upload_service or ingestion_service settings
        source_path = Path(ingestion_service.settings.SOURCE_DIRECTORY)
        if source_path.exists() and source_path.is_dir():
            doc_files = list(
                source_path.rglob("*.*")
            )  # Consider specific file types if needed
            docs_found_count = len([f for f in doc_files if f.is_file()])
        else:
            docs_found_count = 1  # If path doesn't exist after successful save, count the one just saved.
    except Exception as e:
        logger.warning(
            f"Could not count documents in source directory after upload: {e}. Reporting based on upload."
        )
        docs_found_count = 1  # At least the uploaded file is there

    return IngestionResponse(
        status="File uploaded and ingestion task started.",
        documents_found=docs_found_count,
        message=f"File '{file.filename}' uploaded. Processing documents in the background.",
    )


@router.post(
    "/ingest",
    response_model=IngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,  # Use 202 Accepted for background tasks
    summary="Trigger document ingestion process",
    description="Scans the configured source directory, processes documents, and stores them in the vector database. Runs as a background task.",
    tags=["ingestion"],
)
async def trigger_ingestion(
    background_tasks: BackgroundTasks,
    service: IngestionProcessorService = Depends(get_ingestion_processor_service),
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

    is_ingesting = True  # Set the lock
    logger.info("Setting ingestion lock and adding task to background.")
    background_tasks.add_task(run_ingestion_background, service)

    # Attempt to count documents in the source directory for a more accurate response
    docs_found_count = 0
    try:
        source_path = Path(service.settings.SOURCE_DIRECTORY)
        if source_path.exists() and source_path.is_dir():
            doc_files = list(
                source_path.rglob("*.*")
            )  # Consider only PDFs if that's what you process
            docs_found_count = len([f for f in doc_files if f.is_file()])
    except Exception as e:
        logger.warning(f"Could not count documents in source directory: {e}")
        # Fallback or leave as 0 if preferred when count fails

    return IngestionResponse(
        status="Ingestion task started.",
        documents_found=docs_found_count,  # Reflects count before ingestion starts
        message="Processing documents in the background. Check logs for progress.",
    )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List PDF documents in the source directory",
    description="Retrieves a list of all PDF documents found in the configured source directory.",
    tags=["documents_management"],  # New tag or use existing like "ingestion"
)
async def list_source_documents(settings: Settings = Depends(get_settings)):
    source_directory = Path(settings.SOURCE_DIRECTORY)
    logger.info(f"Listing PDF documents from source directory: '{source_directory}'")

    if not source_directory.exists() or not source_directory.is_dir():
        logger.warning(
            f"Source directory '{source_directory}' not found or is not a directory."
        )
        # Return empty list
        return DocumentListResponse(
            count=0, documents=[], source_directory=str(source_directory)
        )

    document_details: List[DocumentDetail] = []
    try:
        # Recursively find all .pdf files
        pdf_files = list(source_directory.rglob("*.pdf"))
        for pdf_file in pdf_files:
            if pdf_file.is_file():  # Ensure it's a file
                document_details.append(DocumentDetail(name=pdf_file.name))

        logger.info(
            f"Found {len(document_details)} PDF documents in '{source_directory}'."
        )
        return DocumentListResponse(
            count=len(document_details),
            documents=document_details,
            source_directory=str(source_directory),
        )
    except Exception as e:
        logger.error(
            f"Error listing documents in '{source_directory}': {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}",
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

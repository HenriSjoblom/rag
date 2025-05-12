import logging
import os
import shutil
from pathlib import Path
from typing import List

import chromadb
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
    JSONResponse
)
from fastapi.responses import JSONResponse

from app.config import Settings
from app.deps import get_ingestion_processor_service, get_settings
from app.models import (
    DocumentDetail,
    DocumentListResponse,
    IngestionResponse,
    IngestionStatus,
)
from app.services.ingestion_processor import (
    IngestionProcessorService,
    get_chroma_client,
)
from app.services.ingestion_processor import _vector_store as global_vector_store_cache

logger = logging.getLogger(__name__)
router = APIRouter()

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
    service: IngestionProcessorService = Depends(get_ingestion_processor_service),
):
    global is_ingesting
    if is_ingesting:
        logger.warning("Upload attempt while ingestion task is already running.")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An ingestion process is already running. Please wait for it to complete before uploading and triggering a new one.",
        )

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided with the uploaded file.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF documents are allowed.",
        )

    source_path = Path(service.settings.SOURCE_DIRECTORY)
    if not source_path.exists() or not source_path.is_dir():
        logger.error(
            f"Source directory {source_path} does not exist or is not a directory. Cannot save uploaded file."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: Source directory for uploads not found.",
        )

    # Ensure the directory exists (it should, but defensive check)
    source_path.mkdir(parents=True, exist_ok=True)

    file_location = source_path / file.filename

    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Successfully uploaded file: {file.filename} to {file_location}")
    except Exception as e:
        logger.error(
            f"Failed to save uploaded file {file.filename}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {e}",
        )
    finally:
        await file.close()  # Ensure the uploaded file is closed

    # Set the lock and add the task to the background
    is_ingesting = True
    logger.info(
        "Setting ingestion lock and adding task to background after file upload."
    )
    background_tasks.add_task(run_ingestion_background, service)

    docs_found_count = 0
    try:
        doc_files = list(source_path.rglob("*.*"))
        docs_found_count = len([f for f in doc_files if f.is_file()])
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
    settings: Settings = Depends(get_settings),
):
    """
    Deletes the configured ChromaDB collection and all files in the source documents directory.
    Resets the cached vector store instance.
    """
    global global_vector_store_cache

    collection_name = settings.CHROMA_COLLECTION_NAME
    source_directory = Path(settings.SOURCE_DIRECTORY)
    deleted_files_count = 0
    collection_deleted_successfully = False
    files_deleted_successfully = False
    messages = []

    #Delete files from the source directory
    logger.info(
        f"Attempting to delete all files from source directory: '{source_directory}'"
    )
    if not source_directory.exists() or not source_directory.is_dir():
        message = f"Source directory '{source_directory}' not found or is not a directory. No files deleted."
        logger.warning(message)
        messages.append(message)
        files_deleted_successfully = (
            True  # Considered success as there's nothing to delete
        )
    else:
        try:
            for item in source_directory.iterdir():
                if item.is_file():
                    try:
                        os.remove(item)
                        deleted_files_count += 1
                        logger.debug(f"Deleted file: {item}")
                    except Exception as e:
                        err_msg = f"Failed to delete file {item}: {e}"
                        logger.error(err_msg, exc_info=True)
                        messages.append(err_msg)


            if not messages:  # If no errors during file deletion
                files_deleted_successfully = True

            log_msg = f"Successfully deleted {deleted_files_count} file(s) from '{source_directory}'."
            logger.info(log_msg)
            messages.append(log_msg)

        except Exception as e:
            err_msg = (
                f"An error occurred while deleting files from '{source_directory}': {e}"
            )
            logger.error(err_msg, exc_info=True)
            messages.append(err_msg)
            # Proceed to attempt collection deletion even if file deletion fails partially or fully

    # Delete ChromaDB collection
    logger.info(f"Attempting to delete ChromaDB collection: '{collection_name}'")
    try:
        client = get_chroma_client(settings)
        client.delete_collection(name=collection_name)
        collection_deleted_successfully = True
        msg = f"Successfully deleted ChromaDB collection: '{collection_name}'"
        logger.info(msg)
        messages.append(msg)

        if global_vector_store_cache is not None:
            logger.info("Resetting cached LangChain Chroma vector store instance.")
            global_vector_store_cache = None

    except chromadb.errors.NotACollectionError:
        collection_deleted_successfully = (
            True  # Desired state (collection doesn't exist)
        )
        msg = f"Collection '{collection_name}' not found. No deletion performed."
        logger.info(msg)
        messages.append(msg)
        if global_vector_store_cache is not None:
            logger.info(
                "Resetting cached LangChain Chroma vector store instance (collection not found)."
            )
            global_vector_store_cache = None
    except Exception as e:
        err_msg = f"Failed to delete collection '{collection_name}': {e}"
        logger.error(err_msg, exc_info=True)
        messages.append(err_msg)

    # Determine overall status
    if collection_deleted_successfully and files_deleted_successfully:
        final_status_code = status.HTTP_200_OK
        final_message = "ChromaDB collection and source documents cleared successfully."
    elif (
        collection_deleted_successfully or files_deleted_successfully
    ):  # Partial success
        final_status_code = status.HTTP_207_MULTI_STATUS
        final_message = "Partial success in clearing resources. Check details."
    else:  # Both failed or had significant errors
        final_status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        final_message = "Failed to clear ChromaDB collection and/or source documents."

    return JSONResponse(
        status_code=final_status_code,
        content={
            "message": final_message,
            "details": messages,
            "files_deleted_count": deleted_files_count,
            "collection_deleted": collection_deleted_successfully,
            "source_files_cleared": files_deleted_successfully,
        },
    )

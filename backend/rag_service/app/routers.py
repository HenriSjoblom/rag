import logging

import httpx
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)

from app.config import Settings
from app.config import settings as app_settings
from app.deps import (
    get_chat_processor_service,
    get_http_client,
)
from app.models import (
    ChatRequest,
    ChatResponse,
    IngestionDeleteResponse,
    IngestionUploadResponse,
    RagDocumentDetail,  # New
    RagDocumentListResponse,  # New
    ServiceErrorResponse,
)
from app.services.chat_processor import ChatProcessorService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Process a user's chat message",
    description="Receives a user message, orchestrates retrieval and generation, and returns an AI response.",
    tags=["chat"],
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ServiceErrorResponse,
            "description": "Downstream service is unavailable",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ServiceErrorResponse,
            "description": "Internal server error",
        },
    },
)
async def process_chat_message(
    chat_request: ChatRequest,
    chat_processor: ChatProcessorService = Depends(get_chat_processor_service),
):
    try:
        logger.info(
            f"Received chat request for user_id: {chat_request.user_id}, message: '{chat_request.message[:50]}...'"
        )
        ai_response_text = await chat_processor.process(
            user_id=chat_request.user_id, query=chat_request.message
        )
        logger.info(f"Successfully processed chat for user_id: {chat_request.user_id}")
        return ChatResponse(
            user_id=chat_request.user_id,
            query=chat_request.message,
            response=ai_response_text,
        )
    except HTTPException as http_exc:  # Re-raise known HTTP exceptions
        logger.error(
            f"HTTPException during chat processing for user {chat_request.user_id}: {http_exc.detail}"
        )
        raise http_exc
    except Exception as e:
        logger.exception(
            f"Unexpected error processing chat for user_id {chat_request.user_id}: {e}",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )


# --- Ingestion Service Proxy Routes ---


@router.post(
    "/documents/upload",
    response_model=IngestionUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a PDF document for ingestion via Ingestion Service",
    description="Forwards a PDF file to the Ingestion Service to be saved and trigger background ingestion.",
    tags=["documents"],
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ServiceErrorResponse,
            "description": "Ingestion service is unavailable",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ServiceErrorResponse,
            "description": "Bad request (e.g., invalid file type from ingestion service)",
        },
        status.HTTP_409_CONFLICT: {
            "model": ServiceErrorResponse,
            "description": "Conflict (e.g., ingestion already in progress by ingestion service)",
        },
    },
)
async def upload_document_for_ingestion(
    file: UploadFile = File(..., description="PDF document to upload."),
    http_client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(lambda: app_settings),
):
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided with the uploaded file.",
        )
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF documents are allowed for upload to RAG service.",
        )

    ingestion_service_upload_url = f"{settings.INGESTION_SERVICE_URL}api/v1/upload"
    logger.info(
        f"Forwarding file '{file.filename}' to Ingestion Service at {ingestion_service_upload_url}"
    )

    try:
        # Prepare files for httpx.post
        # The content of UploadFile needs to be read.
        # httpx expects a tuple: (filename, file_object, content_type)
        files_data = {"file": (file.filename, await file.read(), file.content_type)}

        response = await http_client.post(
            ingestion_service_upload_url,
            files=files_data,  # Use files parameter for multipart/form-data
        )
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses

        ingestion_response_data = response.json()
        logger.info(
            f"Successfully forwarded file to Ingestion Service. Response: {ingestion_response_data}"
        )
        # Map to our IngestionUploadResponse model
        return IngestionUploadResponse(
            status=ingestion_response_data.get(
                "status", "Unknown status from ingestion service"
            ),
            documents_found=ingestion_response_data.get("documents_found"),
            message=ingestion_response_data.get(
                "message", "No message from ingestion service"
            ),
        )

    except (
        HTTPException
    ) as http_exc:  # Re-raise if already an HTTPException (e.g. from raise_for_status)
        logger.error(
            f"HTTP error while forwarding to Ingestion Service: {http_exc.status_code} - {http_exc.detail}"
        )
        raise http_exc
    except Exception as e:
        logger.exception(f"Error forwarding file to Ingestion Service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to connect or communicate with Ingestion Service: {str(e)}",
        )
    finally:
        await file.close()


@router.get(
    "/documents",
    response_model=RagDocumentListResponse,
    summary="List documents managed by the Ingestion Service",
    description="Retrieves a list of documents by querying the Ingestion Service.",
    tags=["documents"],
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ServiceErrorResponse,
            "description": "Ingestion service is unavailable",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ServiceErrorResponse,
            "description": "Error retrieving documents from ingestion service",
        },
    },
)
async def list_documents_via_ingestion_service(
    http_client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(lambda: app_settings),
):
    ingestion_service_docs_url = f"{settings.INGESTION_SERVICE_URL}api/v1/documents"
    logger.info(
        f"Requesting document list from Ingestion Service at {ingestion_service_docs_url}"
    )

    try:
        response = await http_client.get(ingestion_service_docs_url)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses

        ingestion_response_data = response.json()
        logger.info(
            f"Successfully retrieved document list from Ingestion Service. Response: {ingestion_response_data}"
        )

        # Map to our RagDocumentListResponse model
        # Assuming ingestion_response_data matches DocumentListResponse from ingestion-service
        doc_details = [
            RagDocumentDetail(name=doc.get("name"))
            for doc in ingestion_response_data.get("documents", [])
            if doc.get("name")  # Ensure name exists
        ]

        return RagDocumentListResponse(
            count=ingestion_response_data.get("count", 0),
            documents=doc_details,
            source_directory=ingestion_response_data.get("source_directory"),
        )
    except HTTPException as http_exc:
        logger.error(
            f"HTTP error while requesting document list from Ingestion Service: {http_exc.status_code} - {http_exc.detail}"
        )
        raise http_exc
    except Exception as e:
        logger.exception(f"Error requesting document list from Ingestion Service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to connect or communicate with Ingestion Service: {str(e)}",
        )


@router.delete(
    "/documents",
    response_model=IngestionDeleteResponse,  # Use the new response model
    summary="Clear all documents and ingested data via Ingestion Service",
    description="Requests the Ingestion Service to delete its ChromaDB collection and source documents.",
    tags=["documents"],
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ServiceErrorResponse,
            "description": "Ingestion service is unavailable",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ServiceErrorResponse,
            "description": "Error during deletion in ingestion service",
        },
    },
)
async def delete_all_documents_and_ingested_data(
    http_client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(lambda: app_settings),
):
    ingestion_service_delete_url = f"{settings.INGESTION_SERVICE_URL}api/v1/collection"
    logger.info(
        f"Requesting Ingestion Service to delete data at {ingestion_service_delete_url}"
    )

    try:
        response = await http_client.delete(ingestion_service_delete_url)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses

        ingestion_response_data = response.json()
        logger.info(
            f"Successfully requested data deletion from Ingestion Service. Response: {ingestion_response_data}"
        )
        # Map to our IngestionDeleteResponse model
        return IngestionDeleteResponse(
            message=ingestion_response_data.get(
                "message", "No message from ingestion service"
            ),
            details=ingestion_response_data.get("details"),
            files_deleted_count=ingestion_response_data.get("files_deleted_count"),
            collection_deleted=ingestion_response_data.get("collection_deleted"),
            source_files_cleared=ingestion_response_data.get("source_files_cleared"),
        )
    except HTTPException as http_exc:
        logger.error(
            f"HTTP error during delete request to Ingestion Service: {http_exc.status_code} - {http_exc.detail}"
        )
        raise http_exc
    except Exception as e:
        logger.exception(f"Error requesting data deletion from Ingestion Service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to connect or communicate with Ingestion Service: {str(e)}",
        )

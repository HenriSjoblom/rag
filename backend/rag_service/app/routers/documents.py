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
from app.deps import get_http_client
from app.models import (
    IngestionDeleteResponse,
    IngestionUploadResponse,
    RagDocumentDetail,
    RagDocumentListResponse,
    ServiceErrorResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["documents"])


@router.post(
    "/documents/upload",
    response_model=IngestionUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a PDF document for ingestion via Ingestion Service",
    description="Forwards a PDF file to the Ingestion Service to be saved and trigger background ingestion.",
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
        f"RAG Service configuration - INGESTION_SERVICE_URL: {settings.INGESTION_SERVICE_URL}"
    )
    logger.info(
        f"Attempting to connect to ingestion service at: {ingestion_service_upload_url}"
    )

    # First, try to ping the ingestion service health endpoint
    try:
        health_url = f"{settings.INGESTION_SERVICE_URL}health"
        logger.info(f"Checking ingestion service health at: {health_url}")
        health_response = await http_client.get(health_url, timeout=10.0)
        if health_response.status_code == 200:
            logger.info("Ingestion service health check passed")
        else:
            logger.warning(
                f"Ingestion service health check returned: {health_response.status_code}"
            )
    except httpx.ConnectError as health_connect_error:
        logger.error(
            f"Cannot connect to ingestion service health endpoint: {health_connect_error}"
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cannot connect to Ingestion Service. Configured URL: {settings.INGESTION_SERVICE_URL}. In Docker, ensure services can reach each other by service name, not localhost.",
        )
    except Exception as health_error:
        logger.error(f"Ingestion service health check failed: {health_error}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ingestion service health check failed. Please verify the service is running at {settings.INGESTION_SERVICE_URL}",
        )

    try:
        # Prepare files for httpx.post
        files_data = {"file": (file.filename, await file.read(), file.content_type)}

        logger.info(f"Sending POST request to: {ingestion_service_upload_url}")
        response = await http_client.post(
            ingestion_service_upload_url,
            files=files_data,
            timeout=60.0,  # Increase timeout to 60 seconds
        )

        # Log the response details for debugging
        logger.info(f"Ingestion service response status: {response.status_code}")
        logger.info(f"Ingestion service response headers: {dict(response.headers)}")

        # Check if the response is successful before trying to parse
        if response.status_code not in [200, 202]:
            logger.error(f"Ingestion service returned status {response.status_code}")
            response.raise_for_status()

        # Check if response has content and is JSON
        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type:
            logger.warning(
                f"Non-JSON response from ingestion service. Content-Type: {content_type}"
            )
            # For 202 responses, the ingestion service might not return JSON
            return IngestionUploadResponse(
                status="Upload accepted",
                documents_found=None,
                message="File upload accepted by ingestion service",
                filename=file.filename,
            )

        try:
            ingestion_response_data = response.json()
            logger.info(f"Successfully parsed JSON response: {ingestion_response_data}")
        except Exception as json_error:
            logger.warning(f"Failed to parse JSON response: {json_error}")
            # Still return success since we got a good status code
            return IngestionUploadResponse(
                status="Upload accepted",
                documents_found=None,
                message="File upload accepted by ingestion service",
                filename=file.filename,
            )

        # Map to our IngestionUploadResponse model
        return IngestionUploadResponse(
            status=ingestion_response_data.get("status", "Upload accepted"),
            documents_found=ingestion_response_data.get("documents_found"),
            message=ingestion_response_data.get(
                "message", "File uploaded successfully"
            ),
            filename=ingestion_response_data.get("filename", file.filename),
        )

    except httpx.ConnectError as connect_error:
        # Handle connection errors specifically
        logger.error(
            f"Connection error details: {type(connect_error).__name__}: {connect_error}"
        )
        logger.error(f"Failed to connect to: {ingestion_service_upload_url}")
        logger.error(
            f"Configured INGESTION_SERVICE_URL: {settings.INGESTION_SERVICE_URL}"
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cannot connect to Ingestion Service. Configured URL: {settings.INGESTION_SERVICE_URL}. In Docker, use service names instead of localhost. Error: {str(connect_error)}",
        )
    except httpx.TimeoutException as timeout_error:
        # Handle timeout errors
        logger.error(f"Timeout error after 60 seconds: {timeout_error}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ingestion Service is not responding within 60 seconds. The service may be overloaded or stuck.",
        )
    except httpx.HTTPStatusError as http_status_error:
        # Handle HTTP status errors from the ingestion service
        logger.error(
            f"HTTP status error from Ingestion Service: {http_status_error.response.status_code}"
        )
        try:
            error_detail = http_status_error.response.json().get(
                "detail", str(http_status_error)
            )
        except Exception:
            error_detail = f"HTTP {http_status_error.response.status_code}: {http_status_error.response.text}"

        # Map the status code from ingestion service to appropriate RAG service status
        if http_status_error.response.status_code == 409:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=error_detail,
            )
        elif http_status_error.response.status_code == 400:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_detail,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Ingestion service error: {error_detail}",
            )
    except httpx.RequestError as request_error:
        # Handle other request errors
        logger.error(
            f"Request error while connecting to Ingestion Service: {request_error}"
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to connect to Ingestion Service: {str(request_error)}",
        )
    except Exception as e:
        logger.exception(f"Unexpected error forwarding file to Ingestion Service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to communicate with Ingestion Service: {str(e)}",
        )
    finally:
        await file.close()


@router.get(
    "/documents",
    response_model=RagDocumentListResponse,
    summary="List documents managed by the Ingestion Service",
    description="Retrieves a list of documents by querying the Ingestion Service.",
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
    ingestion_service_docs_url = f"{settings.INGESTION_SERVICE_URL}api/v1/documents/"
    logger.info(
        f"Requesting document list from Ingestion Service at {ingestion_service_docs_url}"
    )

    try:
        response = await http_client.get(ingestion_service_docs_url, timeout=30.0)
        response.raise_for_status()

        ingestion_response_data = response.json()
        logger.info(
            f"Successfully retrieved document list from Ingestion Service. Response: {ingestion_response_data}"
        )

        # Map to our RagDocumentListResponse model
        doc_details = [
            RagDocumentDetail(name=doc.get("name"))
            for doc in ingestion_response_data.get("documents", [])
            if doc.get("name")
        ]

        return RagDocumentListResponse(
            count=ingestion_response_data.get("count", 0),
            documents=doc_details,
        )
    except httpx.ConnectError as connect_error:
        logger.error(f"Connection error to Ingestion Service: {connect_error}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cannot connect to Ingestion Service. Please check if the service is running.",
        )
    except httpx.TimeoutException as timeout_error:
        logger.error(f"Timeout error connecting to Ingestion Service: {timeout_error}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ingestion Service is not responding. Please try again later.",
        )
    except httpx.HTTPStatusError as http_exc:
        logger.error(
            f"HTTP error while requesting document list from Ingestion Service: {http_exc.response.status_code}"
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ingestion Service returned error: {http_exc.response.status_code}",
        )
    except Exception as e:
        logger.exception(f"Error requesting document list from Ingestion Service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to connect or communicate with Ingestion Service: {str(e)}",
        )


@router.delete(
    "/documents",
    response_model=IngestionDeleteResponse,
    summary="Clear all documents and ingested data via Ingestion Service",
    description="Requests the Ingestion Service to delete its ChromaDB collection and source documents.",
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
    ingestion_service_delete_url = f"{settings.INGESTION_SERVICE_URL}api/v1/collection/"
    logger.info(
        f"Requesting Ingestion Service to delete data at {ingestion_service_delete_url}"
    )

    try:
        response = await http_client.delete(ingestion_service_delete_url)
        response.raise_for_status()

        ingestion_response_data = response.json()
        logger.info(
            f"Successfully requested data deletion from Ingestion Service. Response: {ingestion_response_data}"
        )
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

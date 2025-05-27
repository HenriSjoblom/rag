import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings
from app.config import settings as app_settings
from app.deps import get_http_client
from app.models import IngestionStatusResponse, ServiceErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ingestion"])


@router.get(
    "/ingestion/status",
    response_model=IngestionStatusResponse,
    summary="Get ingestion status from Ingestion Service",
    description="Returns the current status of the ingestion process.",
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ServiceErrorResponse,
            "description": "Ingestion service is unavailable",
        },
    },
)
async def get_ingestion_status(
    http_client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(lambda: app_settings),
):
    ingestion_service_status_url = f"{settings.INGESTION_SERVICE_URL}api/v1/status"
    logger.info(f"Requesting ingestion status from {ingestion_service_status_url}")

    try:
        response = await http_client.get(ingestion_service_status_url, timeout=30.0)
        response.raise_for_status()

        status_data = response.json()
        return IngestionStatusResponse(**status_data)
    except httpx.ConnectError as connect_error:
        logger.error(f"Connection error to Ingestion Service: {connect_error}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cannot connect to Ingestion Service for status check",
        )
    except httpx.TimeoutException as timeout_error:
        logger.error(f"Timeout error connecting to Ingestion Service: {timeout_error}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ingestion Service status check timed out",
        )
    except httpx.HTTPStatusError as http_status_error:
        logger.error(
            f"HTTP status error from Ingestion Service: {http_status_error.response.status_code}"
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to get status from Ingestion Service",
        )
    except Exception as e:
        logger.exception(f"Error getting status from Ingestion Service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to communicate with Ingestion Service: {str(e)}",
        )

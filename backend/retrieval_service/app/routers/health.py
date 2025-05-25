import logging

from fastapi import APIRouter

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get(
    "/health",
    summary="Health check",
    description="Basic health check endpoint to verify service availability.",
)
async def health_check():
    """Basic health check endpoint."""
    logger.debug("Health check requested")
    return {"status": "ok", "service": "retrieval"}

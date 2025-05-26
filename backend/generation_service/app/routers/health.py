import logging

from fastapi import APIRouter


router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get(
    "/health",
    summary="Generation service health check",
    description="Check if the generation service and its dependencies are healthy.",
)
async def health_check():
    """Basic health check endpoint."""
    logger.debug("Basic health check requested")
    return {"status": "ok", "service": "generation"}
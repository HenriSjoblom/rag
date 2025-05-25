"""
Main router module for backward compatibility.

This module re-exports routers from the organized router modules.
"""

# For backward compatibility, create a combined router
from fastapi import APIRouter

from .routers.health import router as health_router
from .routers.retrieval import router as retrieval_router

router = APIRouter()
router.include_router(retrieval_router)

__all__ = ["router", "health_router", "retrieval_router"]

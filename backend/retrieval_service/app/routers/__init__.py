"""
API routers for the retrieval service.

This package contains all the API route definitions organized by functionality.
"""

from .health import router as health_router
from .retrieval import router as retrieval_router

__all__ = ["health_router", "retrieval_router"]

from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

from app.routers import router as retrieve_router
from app.config import settings
from app.services.vector_search import lifespan_retrieval_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager to load model and connect to DB on startup,
    and clean up on shutdown.
    """
    logger.info("Retrieval service lifespan startup...")
    # Prepare ChromaDB settings based on mode
    chroma_config = {
        "chroma_db_impl": "duckdb+parquet", # Default, suitable for local
        "persist_directory": settings.CHROMA_LOCAL_PATH if settings.CHROMA_MODE == 'local' else None,
        # For server mode:
        "chroma_api_impl": "rest" if settings.CHROMA_MODE in ('http', 'https') else None,
        "chroma_server_host": settings.CHROMA_HOST if settings.CHROMA_MODE in ('http', 'https') else None,
        "chroma_server_http_port": settings.CHROMA_PORT if settings.CHROMA_MODE == 'http' else None,
        "chroma_server_https_port": settings.CHROMA_PORT if settings.CHROMA_MODE == 'https' else None,
        # Add other relevant ChromaSettings here if needed (e.g., auth, ssl)
    }
    # Filter out None values from chroma_config before passing to ChromaSettings
    chroma_settings_dict = {k: v for k, v in chroma_config.items() if v is not None}

    # Use the async context manager from vector_search service
    async with lifespan_retrieval_service(
        app,
        model_name=settings.EMBEDDING_MODEL_NAME,
        chroma_settings=chroma_settings_dict,
        collection_name=settings.CHROMA_COLLECTION_NAME
    ):
        logger.info("Retrieval service startup complete. Model and DB connection ready.")
        yield # Application runs
    # Cleanup happens automatically when exiting the 'async with' block
    logger.info("Retrieval service lifespan shutdown complete.")


app = FastAPI(
    title="Retrieval Service",
    description="Embeds queries and retrieves relevant documents from a vector database.",
    version="1.0.0",
    lifespan=lifespan # Register the lifespan manager
)

# Include API routers
app.include_router(retrieve_router, prefix="/api/v1") # Add a version prefix

@app.get("/health", tags=["health"])
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok"}

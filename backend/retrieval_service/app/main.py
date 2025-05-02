from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

from app.routers import router as retrieve_router
from app.deps import get_settings
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
    settings = get_settings()
    print(f"ChromaDB path: {settings.CHROMA_PATH}")
    logger.info("Retrieval service lifespan startup...")
    # Use the async context manager from vector_search service
    async with lifespan_retrieval_service(
        app=app,
        model_name=settings.EMBEDDING_MODEL_NAME,
        chroma_path=settings.CHROMA_PATH,
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

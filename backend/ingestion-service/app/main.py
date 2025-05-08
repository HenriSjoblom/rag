from fastapi import FastAPI
import logging
from contextlib import asynccontextmanager

from app.router import router as ingest_router
from app.deps import get_settings
from app.services.ingestion_processor import get_embedding_model, get_chroma_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Ingestion Service starting up...")
    # Pre-load embedding model and Chroma client on startup
    try:
        logger.info("Pre-loading embedding model...")
        get_embedding_model(settings) # Loads and caches if not already loaded
        logger.info("Pre-connecting to ChromaDB...")
        get_chroma_client(settings) # Connects and caches if not already connected
        logger.info("Resources pre-loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to pre-load resources during startup: {e}", exc_info=True)
        raise RuntimeError(f"Failed to pre-load resources: {e}") from e
    yield
    logger.info("Ingestion Service shutting down...")


app = FastAPI(
    title="Ingestion Service",
    description="Loads, processes, and stores documents in a vector database.",
    version="1.0.0",
    lifespan=lifespan # Register the lifespan manager
)

# Include API routers
app.include_router(ingest_router, prefix="/api/v1")

@app.get("/health", tags=["health"])
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok"}

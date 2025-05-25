import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from app.deps import get_ingestion_state_service, get_settings
from app.models import IngestionStatusResponse
from app.routers import collection, documents, ingestion
from app.services.chroma_manager import (
    ChromaClientManager,
    EmbeddingModelManager,
    VectorStoreManager,
)
from app.services.ingestion_state import IngestionStateService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Ingestion Service starting up...")

    # Initialize managers as singletons in app state
    app.state.chroma_manager = ChromaClientManager(settings)
    app.state.embedding_manager = EmbeddingModelManager(settings)
    app.state.vector_store_manager = VectorStoreManager(
        settings, app.state.chroma_manager, app.state.embedding_manager
    )
    app.state.ingestion_state_service = IngestionStateService()

    # Pre-load resources on startup
    try:
        logger.info("Pre-loading embedding model...")
        app.state.embedding_manager.get_model()
        logger.info("Pre-connecting to ChromaDB...")
        app.state.chroma_manager.get_client()
        logger.info("Resources pre-loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to pre-load resources during startup: {e}", exc_info=True)
        raise RuntimeError(f"Failed to pre-load resources: {e}") from e

    yield

    # Cleanup on shutdown
    logger.info("Ingestion Service shutting down...")


app = FastAPI(
    title="Ingestion Service",
    description="Loads, processes, and stores documents in a vector database.",
    version="1.0.0",
    lifespan=lifespan,
)

# Include API routers
api_prefix = "/api/v1"

app.include_router(ingestion.router, prefix=api_prefix)
app.include_router(documents.router, prefix=api_prefix)
app.include_router(collection.router, prefix=api_prefix)


@app.get("/health", summary="Health check", tags=["health"])
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok", "service": "ingestion"}


@app.get(
    f"{api_prefix}/status",
    response_model=IngestionStatusResponse,
    summary="Get ingestion status",
    description="Returns the current status of ingestion process including completion details.",
    tags=["ingestion"],
)
async def get_ingestion_status(
    state_service: IngestionStateService = Depends(get_ingestion_state_service),
):
    """Get the current ingestion status."""
    status_info = await state_service.get_status()
    return IngestionStatusResponse(**status_info)

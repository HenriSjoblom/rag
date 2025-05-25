import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.deps import get_settings
from app.routers import health_router, retrieval_router
from app.services.chroma_manager import ChromaClientManager
from app.services.embedding_manager import EmbeddingModelManager
from app.services.vector_store_manager import VectorStoreManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Retrieval Service starting up...")

    # Initialize managers as singletons in app state
    app.state.chroma_manager = ChromaClientManager(settings)
    app.state.embedding_manager = EmbeddingModelManager(settings)
    app.state.vector_store_manager = VectorStoreManager(
        settings, app.state.chroma_manager, app.state.embedding_manager
    )

    # Pre-load resources on startup
    try:
        logger.info("Pre-loading embedding model...")
        app.state.embedding_manager.get_model()
        logger.info("Pre-connecting to ChromaDB...")
        app.state.chroma_manager.get_client()
        logger.info("Pre-loading collection...")
        app.state.vector_store_manager.get_collection()
        logger.info("Resources pre-loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to pre-load resources during startup: {e}", exc_info=True)
        raise RuntimeError(f"Failed to pre-load resources: {e}") from e

    yield

    # Cleanup on shutdown
    logger.info("Retrieval Service shutting down...")


app = FastAPI(
    title="Retrieval Service",
    description="Retrieves relevant document chunks from a vector database.",
    version="1.0.0",
    lifespan=lifespan,
)

# Include health router at root level
app.include_router(health_router)

# Include API routers with prefix
api_prefix = "/api/v1"
app.include_router(retrieval_router, prefix=api_prefix)

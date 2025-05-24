import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings
from app.deps import get_settings
from app.routers import router as rag_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings: Settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("RAG Service starting up...")
    logger.info(f"Configured ingestion service URL: {settings.INGESTION_SERVICE_URL}")

    # Initialize global HTTP client
    app.state.http_client = httpx.AsyncClient(timeout=30.0)
    logger.info("Global HTTP client initialized.")

    yield

    # Cleanup on shutdown
    logger.info("RAG Service shutting down...")
    if hasattr(app.state, "http_client"):
        await app.state.http_client.aclose()
        logger.info("Global HTTP client closed.")


app = FastAPI(
    title="Chat API Service",
    description="Orchestrates RAG pipeline for the customer support chatbot.",
    version="1.0.0",
    lifespan=lifespan,
)

# Include API routers
app.include_router(rag_router, prefix="/api/v1")

# Add cors middleware
origins = ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok"}

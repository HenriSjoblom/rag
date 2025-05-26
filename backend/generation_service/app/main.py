import logging

from fastapi import FastAPI

from app.config import settings
from app.routers import generation, health

# Configure logging with better format and level handling
logging.basicConfig(
    level=getattr(logging, getattr(settings, "LOG_LEVEL", "INFO"), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Generation Service",
    description="Generates text responses using a configured Large Language Model based on provided context.",
    version="1.0.0",
)

# Include API routers with proper organization
api_prefix = "/api/v1"
app.include_router(generation.router, prefix=api_prefix)
app.include_router(health.router)

logger.info("Generation Service initialized successfully")

from fastapi import FastAPI
import logging

from app.routers import generate as generate_router
from app.config import settings

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Generation Service",
    description="Generates text responses using a configured Large Language Model based on provided context.",
    version="1.0.0",
)

# Include API routers
app.include_router(generate_router.router, prefix="/api/v1")

@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok", "llm_provider": settings.LLM_PROVIDER, "llm_model": settings.LLM_MODEL_NAME}

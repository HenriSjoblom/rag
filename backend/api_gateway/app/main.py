from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routers import router as gateway_router
from app.config import Settings
from app.services.http_client import lifespan_http_client
from app.deps import get_settings


settings: Settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with lifespan_http_client(app, timeout=settings.HTTP_CLIENT_TIMEOUT):
        yield


app = FastAPI(
    title="API Gateway Service",
    description="Orchestrates RAG pipeline.",
    version="1.0.0"
)

# Include API routers
app.include_router(gateway_router, prefix="/api/v1")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}
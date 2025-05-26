import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings
from app.deps import get_settings
from app.routers.chat import router as chat_router
from app.routers.documents import router as documents_router
from app.routers.health import router as health_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings: Settings = get_settings()

app = FastAPI(
    title="RAG Service",
    description="Orchestrates RAG pipeline for the user manual assistant chatbot.",
    version="1.0.0",
)

# Include routers directly
app.include_router(chat_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")
app.include_router(health_router)

# Add cors middleware
origins = ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

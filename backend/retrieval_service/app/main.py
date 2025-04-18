# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import router as retriever_router
from app.config import settings
from app.services.http_client import lifespan_http_client

app = FastAPI(
    title="Chat API Service",
    description="Orchestrates RAG pipeline for the customer support chatbot.",
    version="1.0.0",
)

# Include API routers
app.include_router(retriever_router, prefix="/api/v1")

# Add cors middleware
origins = [
     "http://localhost",
     "http://localhost:3000"
]
app.add_middleware(
    CORSMiddleware,
     allow_origins=origins,
     allow_credentials=True,
     allow_methods=["*"],
     allow_headers=["*"],
)


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
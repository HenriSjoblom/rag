# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routers import router as gateway_router
from app.config import settings

app = FastAPI(
    title="API Gateway Service",
    description="Orchestrates RAG pipeline.",
    version="1.0.0"
)

# Include API routers
app.include_router(gateway_router, prefix="/api/v0")

# Add cors middleware
origins = [
     "http://localhost",
     "http://localhost:3000",
]
app.add_middleware(
     CORSMiddleware,
     allow_origins=origins,
     allow_credentials=True,
     allow_methods=["*"],
     allow_headers=["*"]
)

@app.get("/health", tags=["health"])
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok"}
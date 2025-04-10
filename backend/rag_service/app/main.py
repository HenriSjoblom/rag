# app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api.routers import chat as chat_router
from app.core.config import settings
from app.services.http_client import lifespan_http_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager. Manages resources like the HTTP client pool.
    """
    # Startup: Initialize resources
    async with lifespan_http_client(app, timeout=settings.EXTERNAL_SERVICE_TIMEOUT):
        print(f"Started Chat API. Configured services:")
        print(f" - Retrieval: {settings.RETRIEVAL_SERVICE_URL}")
        print(f" - Generation: {settings.GENERATION_SERVICE_URL}")
        yield # Application runs
    # Shutdown: Cleanup resources (handled by lifespan_http_client context manager)
    print("Chat API shut down.")


app = FastAPI(
    title="Chat API Service",
    description="Orchestrates RAG pipeline for the customer support chatbot.",
    version="1.0.0",
    lifespan=lifespan # Register the lifespan manager
)

# Include API routers
app.include_router(chat_router.router, prefix="/api/v1") # Add a version prefix

# Optional: Add CORS middleware if your frontend is on a different domain
# from fastapi.middleware.cors import CORSMiddleware
# origins = [
#     "http://localhost",
#     "http://localhost:3000", # Example frontend dev server
#     # Add your frontend deployment URL here
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok"}

# If running directly using `python app/main.py` (not recommended for production)
# Use `uvicorn app.main:app --reload` instead
if __name__ == "__main__":
    import uvicorn
    # Note: Uvicorn CLI is preferred for running the app
    uvicorn.run(app, host="0.0.0.0", port=8000)
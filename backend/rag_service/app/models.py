from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    """Request model for initiating a chat interaction."""
    user_id: str = Field(..., description="Unique identifier for the user session.")
    message: str = Field(..., min_length=1, description="The user's message.")

class RetrievalRequest(BaseModel):
    """Data sent TO the retrieval service."""
    query: str

class RetrievalResponse(BaseModel):
    """Data received FROM the retrieval service."""
    chunks: List[str]

class GenerationRequest(BaseModel):
    """Data sent TO the generation service."""
    query: str
    context_chunks: List[str]

class GenerationResponse(BaseModel):
    """Data received FROM the generation service."""
    answer: str

class ChatResponse(BaseModel):
    """Response model sent back to the client."""
    response: str

from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class RetrievalRequest(BaseModel):
    """Request model expected from the chat-api-service."""
    query: str = Field(..., min_length=1, description="The user query to search for.")


class RetrievalResponse(BaseModel):
    """Response model sent back to the chat-api-service."""
    # List of relevant text chunks retrieved from the vector database
    chunks: List[str] = Field(..., description="List of relevant text chunks.")
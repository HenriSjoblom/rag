from typing import List, Optional

from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    """Request model for document retrieval."""

    query: str = Field(..., min_length=1, description="Search query text")


class RetrievalResponse(BaseModel):
    """Response model for document retrieval."""

    chunks: List[str] = Field(
        default_factory=list, description="Retrieved document chunks"
    )
    collection_name: Optional[str] = Field(
        None, description="Name of the collection searched"
    )
    query: Optional[str] = Field(None, description="Original query")

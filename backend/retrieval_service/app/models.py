from pydantic import BaseModel, Field
from typing import List

class RetrievalRequest(BaseModel):
    """Request model for retrieving document chunks."""
    query: str = Field(
        ...,
        description="The user query to search for relevant documents.",
        min_length=1,
        json_schema_extra={'example': "Tell me about apples"}
    )

class RetrievalResponse(BaseModel):
    """Response model containing the retrieved document chunks."""
    chunks: List[str] = Field(
        ...,
        description="List of relevant document text chunks.",
        json_schema_extra={'example': ["This is the first test document about apples.", "A final document discussing apples and oranges together."]}
    )
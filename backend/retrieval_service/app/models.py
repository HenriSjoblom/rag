from pydantic import BaseModel, Field
from typing import List, Dict

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

class AddDataRequest(BaseModel):
    """Request model for adding new documents."""
    documents: Dict[str, str] = Field(
        ...,
        description="A dictionary where keys are unique document IDs and values are the document texts.",
        json_schema_extra={'example': {"doc_id_4": "This is a new document about bananas.", "doc_id_5": "Another document about grapes."}}
    )

class AddDataResponse(BaseModel):
    """Response model after adding documents."""
    message: str = Field(
        default="Documents added successfully.", # Add default for consistency
        description="Status message indicating the result of the operation."
    )
    added_count: int = Field(..., description="Number of documents successfully added.")
    collection_name: str = Field(..., description="Name of the collection documents were added to.")
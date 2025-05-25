import pytest
from app.services.ingestion_processor import IngestionProcessorService

@pytest.fixture(scope="session")
def real_chroma_db(test_data_root):
    """Real ChromaDB instance for integration tests."""
    # Setup real ChromaDB with test data
    pass

@pytest.fixture
def integration_ingestion_service(integration_settings, real_chroma_db):
    """IngestionProcessorService with real dependencies."""
    # Create service with real ChromaDB, real embeddings, etc.
    pass
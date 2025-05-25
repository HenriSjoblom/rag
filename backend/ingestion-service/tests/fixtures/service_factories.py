from unittest.mock import Mock
from app.services.ingestion_processor import IngestionProcessorService
from app.services.ingestion_state import IngestionStateService

class ServiceMockFactory:
    """Factory for creating consistently mocked services."""

    @staticmethod
    def create_ingestion_processor(mocker, settings=None, **overrides):
        """Create a mocked IngestionProcessorService."""
        default_mocks = {
            'chroma_manager': mocker.Mock(),
            'embedding_manager': mocker.Mock(),
            'vector_store_manager': mocker.Mock(),
        }
        default_mocks.update(overrides)

        return IngestionProcessorService(
            settings=settings or mocker.Mock(),
            **default_mocks
        )

    @staticmethod
    def create_ingestion_state(initial_state="idle"):
        """Create an IngestionStateService with specific initial state."""
        service = IngestionStateService()
        service._last_status = initial_state
        return service

class ServiceTestBuilder:
    """Builder pattern for creating services in specific states."""

    def __init__(self, mocker):
        self.mocker = mocker
        self.mocks = {}

    def with_chroma_manager(self, **behaviors):
        """Add ChromaManager with specific behaviors."""
        mock = self.mocker.Mock()
        for method, return_value in behaviors.items():
            getattr(mock, method).return_value = return_value
        self.mocks['chroma_manager'] = mock
        return self

    def with_vector_store(self, **behaviors):
        """Add VectorStore with specific behaviors."""
        mock = self.mocker.Mock()
        for method, return_value in behaviors.items():
            getattr(mock, method).return_value = return_value
        self.mocks['vector_store_manager'] = mock
        return self

    def build_ingestion_processor(self, settings):
        """Build the IngestionProcessorService."""
        return IngestionProcessorService(settings=settings, **self.mocks)
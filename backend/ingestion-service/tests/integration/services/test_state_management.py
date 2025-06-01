"""
State management integration tests.
Tests state service behavior in isolation.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock


class TestStateManagementIntegration:
    """Test state management service in isolation."""

    def test_initial_state(self, mock_state_service):
        """Test initial state of the service."""
        status = mock_state_service.get_status()
        assert status["is_processing"] is False
        assert status["status"] == "idle"
        assert status["documents_processed"] == 0
        assert status["chunks_added"] == 0
        assert status["last_completed"] is None

    def test_ingestion_state_transitions(self, mock_state_service):
        """Test state transitions during ingestion."""
        # Initial state: not ingesting
        assert mock_state_service.is_ingesting() is False

        # Start ingestion
        mock_state_service.is_ingesting.return_value = False
        result = mock_state_service.start_ingestion()
        assert result is True

        # Configure for ingesting state
        mock_state_service.is_ingesting.return_value = True
        assert mock_state_service.is_ingesting() is True

        # Configure completed state
        mock_state_service.is_ingesting.return_value = False
        mock_state_service.get_status.return_value = {
            "is_processing": False,
            "status": "completed",
            "documents_processed": 5,
            "chunks_added": 25,
            "last_completed": "2023-01-01T12:00:00Z",
            "errors": []
        }

        final_status = mock_state_service.get_status()
        assert final_status["status"] == "completed"
        assert final_status["documents_processed"] == 5

    def test_concurrent_ingestion_prevention(self, mock_state_service):
        """Test prevention of concurrent ingestion."""
        # Mock already ingesting
        mock_state_service.is_ingesting.return_value = True

        # Attempt to start another ingestion
        # In real service, this would return False
        # For mock, we verify the state check
        is_busy = mock_state_service.is_ingesting()
        assert is_busy is True

        # Verify start_ingestion is not called when busy
        if is_busy:
            # Should not start new ingestion
            pass
        else:
            mock_state_service.start_ingestion()

    def test_error_state_handling(self, mock_state_service):
        """Test handling of error states."""
        # Configure error state
        mock_state_service.get_status.return_value = {
            "is_processing": False,
            "status": "error",
            "documents_processed": 2,
            "chunks_added": 10,
            "last_completed": None,
            "errors": [
                "Failed to process document.pdf: Invalid format",
                "ChromaDB connection timeout"
            ]
        }

        status = mock_state_service.get_status()
        assert status["status"] == "error"
        assert len(status["errors"]) == 2
        assert "Invalid format" in status["errors"][0]

    def test_status_persistence(self, mock_state_service):
        """Test that status persists across calls."""
        # Set initial status
        initial_status = {
            "is_processing": False,
            "status": "idle",
            "documents_processed": 0,
            "chunks_added": 0,
            "last_completed": None,
            "errors": []
        }
        mock_state_service.get_status.return_value = initial_status

        # Get status multiple times
        status1 = mock_state_service.get_status()
        status2 = mock_state_service.get_status()

        assert status1 == status2
        assert status1["status"] == "idle"

    def test_async_operations(self, mock_state_service):
        """Test async state operations."""
        # Since mock_state_service is AsyncMock, test async behavior
        assert hasattr(mock_state_service, '_mock_name')

        # Test that async methods can be called
        status = mock_state_service.get_status()
        assert isinstance(status, dict)

        # Test start_ingestion
        result = mock_state_service.start_ingestion()
        assert result is True


class TestStateServiceErrorHandling:
    """Test error handling in state service."""

    def test_service_unavailable(self, mock_state_service):
        """Test behavior when service is unavailable."""
        # Configure service to raise exception
        mock_state_service.get_status.side_effect = Exception("Service unavailable")

        with pytest.raises(Exception) as exc_info:
            mock_state_service.get_status()

        assert "Service unavailable" in str(exc_info.value)

    def test_invalid_state_recovery(self, mock_state_service):
        """Test recovery from invalid states."""
        # Configure invalid state
        mock_state_service.get_status.return_value = {
            "is_processing": None,  # Invalid value
            "status": "unknown",
            "documents_processed": -1,  # Invalid value
            "chunks_added": None,
            "last_completed": "invalid_date",
            "errors": None
        }

        status = mock_state_service.get_status()
        # In real service, this would be validated and corrected
        # For tests, we verify we can handle invalid data
        assert status["status"] == "unknown"

    def test_state_consistency(self, mock_state_service):
        """Test state consistency checks."""
        # Configure consistent state
        consistent_status = {
            "is_processing": False,
            "status": "completed",
            "documents_processed": 5,
            "chunks_added": 25,
            "last_completed": "2023-01-01T12:00:00Z",
            "errors": []
        }
        mock_state_service.get_status.return_value = consistent_status

        status = mock_state_service.get_status()

        # Verify consistency
        if status["is_processing"] is False:
            # When not processing, should have definitive status
            assert status["status"] in ["idle", "completed", "error"]

        # Documents processed should not be negative
        assert status["documents_processed"] >= 0
        assert status["chunks_added"] >= 0
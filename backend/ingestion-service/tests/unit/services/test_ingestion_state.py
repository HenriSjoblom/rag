"""
Unit tests for the IngestionStateService.
"""

import asyncio
from datetime import datetime, timezone

import pytest
from app.models import IngestionStatus
from app.services.ingestion_state import IngestionStateService


class TestIngestionStateService:
    """Test cases for IngestionStateService."""

    @pytest.fixture
    def state_service(self):
        """Create IngestionStateService instance."""
        return IngestionStateService()

    @pytest.mark.asyncio
    async def test_initial_state(self, state_service):
        """Test that service starts with correct initial state."""
        assert await state_service.is_ingesting() is False

        status = await state_service.get_status()
        assert status["is_processing"] is False
        assert status["status"] == "idle"
        assert status["last_completed"] is None
        assert status["documents_processed"] is None
        assert status["chunks_added"] is None
        assert status["errors"] == []

    @pytest.mark.asyncio
    async def test_start_ingestion_success(self, state_service):
        """Test successfully starting ingestion."""
        result = await state_service.start_ingestion()
        assert result is True
        assert await state_service.is_ingesting() is True

        status = await state_service.get_status()
        assert status["is_processing"] is True
        assert status["status"] == "processing"

    @pytest.mark.asyncio
    async def test_start_ingestion_already_processing(self, state_service):
        """Test that starting ingestion fails when already processing."""
        # Start first ingestion
        await state_service.start_ingestion()
        assert await state_service.is_ingesting() is True

        # Try to start second ingestion
        result = await state_service.start_ingestion()
        assert result is False
        assert await state_service.is_ingesting() is True

    @pytest.mark.asyncio
    async def test_stop_ingestion_success(self, state_service, mocker):
        """Test successfully stopping ingestion."""
        # Start ingestion first
        await state_service.start_ingestion()
        assert await state_service.is_ingesting() is True

        # Create mock result
        mock_result = IngestionStatus(
            documents_processed=5, chunks_added=100, errors=[]
        )

        # Mock datetime.utcnow instead of datetime.now
        mock_datetime = mocker.patch("app.services.ingestion_state.datetime")
        mock_now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.utcnow.return_value = mock_now

        await state_service.stop_ingestion(result=mock_result, errors=[])

        assert await state_service.is_ingesting() is False

        status = await state_service.get_status()
        assert status["is_processing"] is False
        assert status["status"] == "completed"
        assert status["documents_processed"] == 5
        assert status["chunks_added"] == 100
        assert status["errors"] == []
        # Verify that datetime.utcnow was called
        mock_datetime.utcnow.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_ingestion_with_errors(self, state_service, mocker):
        """Test stopping ingestion with errors."""
        await state_service.start_ingestion()

        errors = ["Error 1", "Error 2"]

        mock_datetime = mocker.patch("app.services.ingestion_state.datetime")
        mock_now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.utcnow.return_value = mock_now

        await state_service.stop_ingestion(result=None, errors=errors)

        status = await state_service.get_status()
        assert status["is_processing"] is False
        assert status["status"] == "completed_with_errors"
        assert status["errors"] == errors

    @pytest.mark.asyncio
    async def test_stop_ingestion_not_processing(self, state_service):
        """Test stopping ingestion when not processing."""
        assert await state_service.is_ingesting() is False

        # Stopping when not processing should not raise error
        await state_service.stop_ingestion(result=None, errors=[])

        assert await state_service.is_ingesting() is False

    @pytest.mark.asyncio
    async def test_stop_ingestion_with_result_and_additional_errors(
        self, state_service, mocker
    ):
        """Test stopping ingestion with both result errors and additional errors."""
        await state_service.start_ingestion()

        mock_result = IngestionStatus(
            documents_processed=3, chunks_added=50, errors=["Result error"]
        )
        additional_errors = ["Additional error"]

        mock_datetime = mocker.patch("app.services.ingestion_state.datetime")
        mock_now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.utcnow.return_value = mock_now

        await state_service.stop_ingestion(result=mock_result, errors=additional_errors)

        status = await state_service.get_status()
        # Additional errors take precedence
        assert status["errors"] == additional_errors
        assert status["status"] == "completed_with_errors"

    @pytest.mark.asyncio
    async def test_concurrent_start_attempts(self, state_service):
        """Test that concurrent start attempts are handled correctly."""
        # Start multiple tasks that try to start ingestion
        tasks = []
        for _ in range(5):
            task = asyncio.create_task(state_service.start_ingestion())
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Only one should succeed
        successful_starts = sum(1 for result in results if result is True)
        assert successful_starts == 1
        assert await state_service.is_ingesting() is True

    @pytest.mark.asyncio
    async def test_get_status_formatting(self, state_service, mocker):
        """Test that get_status returns properly formatted datetime."""
        await state_service.start_ingestion()

        mock_datetime = mocker.patch("app.services.ingestion_state.datetime")
        # Use a specific datetime for testing
        test_time = datetime(2025, 5, 24, 14, 30, 45, tzinfo=timezone.utc)
        mock_datetime.utcnow.return_value = test_time

        await state_service.stop_ingestion(result=None, errors=[])

        status = await state_service.get_status()
        # Verify that datetime.utcnow was called
        mock_datetime.utcnow.assert_called_once()
        assert status["last_completed"] is not None

    @pytest.mark.asyncio
    async def test_state_persistence_across_operations(self, state_service):
        """Test that state persists correctly across multiple operations."""
        # Initial state
        assert await state_service.is_ingesting() is False

        # Start ingestion
        await state_service.start_ingestion()
        assert await state_service.is_ingesting() is True

        # Check status while processing
        status = await state_service.get_status()
        assert status["is_processing"] is True
        assert status["status"] == "processing"

        # Stop ingestion
        mock_result = IngestionStatus(documents_processed=2, chunks_added=40)
        await state_service.stop_ingestion(result=mock_result, errors=[])

        # Check final state
        assert await state_service.is_ingesting() is False
        status = await state_service.get_status()
        assert status["is_processing"] is False
        assert status["status"] == "completed"
        assert status["documents_processed"] == 2
        assert status["chunks_added"] == 40

    @pytest.mark.asyncio
    async def test_error_handling_in_stop_ingestion(self, state_service):
        """Test error handling in stop_ingestion method."""
        await state_service.start_ingestion()

        # Test with None result and empty errors
        await state_service.stop_ingestion(result=None, errors=[])
        status = await state_service.get_status()
        assert status["status"] == "completed"
        assert status["errors"] == []

    @pytest.mark.asyncio
    async def test_lock_behavior(self, state_service):
        """Test that the async lock works correctly."""
        # This test ensures that the lock prevents race conditions
        lock_acquired_count = 0

        async def mock_operation():
            nonlocal lock_acquired_count
            async with state_service._lock:
                lock_acquired_count += 1
                # Simulate some work
                await asyncio.sleep(0.01)
                return lock_acquired_count

        # Run multiple operations concurrently
        tasks = [mock_operation() for _ in range(3)]
        results = await asyncio.gather(*tasks)

        # Each operation should see the lock count increment properly
        assert results == [1, 2, 3]

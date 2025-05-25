"""
Unit tests for ChromaClientManager in the retrieval service.
"""

import os
import socket
from unittest.mock import MagicMock

import pytest
from app.services.chroma_manager import ChromaClientManager
from chromadb.errors import ChromaError


class TestChromaClientManagerInit:
    """Test ChromaClientManager initialization."""

    def test_init_with_settings(self, mock_settings):
        """Test successful initialization with settings."""
        manager = ChromaClientManager(mock_settings)

        assert manager.settings == mock_settings
        assert manager._client is None

    def test_init_stores_settings_reference(self, mock_settings):
        """Test that initialization stores the settings reference correctly."""
        manager = ChromaClientManager(mock_settings)

        # Verify settings are stored and accessible
        assert manager.settings.CHROMA_MODE == "local"
        assert manager.settings.CHROMA_PATH == "/tmp/test"
        assert manager.settings.CHROMA_COLLECTION_NAME == "test_collection"


class TestChromaClientManagerLocalMode:
    """Test ChromaClientManager in local mode."""

    def test_get_client_local_success(self, mock_settings, mocker):
        """Test successful local client creation."""
        # Mock chromadb
        mock_persistent_client = MagicMock()
        mock_chroma = mocker.patch("app.services.chroma_manager.chromadb")
        mock_chroma.PersistentClient.return_value = mock_persistent_client

        # Mock os operations
        mock_makedirs = mocker.patch("app.services.chroma_manager.os.makedirs")
        mock_access = mocker.patch(
            "app.services.chroma_manager.os.access", return_value=True
        )

        manager = ChromaClientManager(mock_settings)
        client = manager.get_client()

        assert client == mock_persistent_client
        assert manager._client == mock_persistent_client
        mock_makedirs.assert_called_once_with("/tmp/test", exist_ok=True)
        mock_access.assert_called_once_with("/tmp/test", os.W_OK)
        mock_chroma.PersistentClient.assert_called_once()

    def test_get_client_local_cached(self, mock_settings, mocker):
        """Test that subsequent calls return cached client."""
        mock_persistent_client = MagicMock()
        mock_chroma = mocker.patch("app.services.chroma_manager.chromadb")
        mock_chroma.PersistentClient.return_value = mock_persistent_client

        mocker.patch("app.services.chroma_manager.os.makedirs")
        mocker.patch("app.services.chroma_manager.os.access", return_value=True)

        manager = ChromaClientManager(mock_settings)

        # First call creates client
        client1 = manager.get_client()
        # Second call returns cached client
        client2 = manager.get_client()

        assert client1 == client2 == mock_persistent_client
        # ChromaDB client should only be created once
        mock_chroma.PersistentClient.assert_called_once()

    def test_get_client_local_no_path(self):
        """Test local mode with missing CHROMA_PATH."""
        settings = MagicMock()
        settings.CHROMA_MODE = "local"
        settings.CHROMA_PATH = None

        manager = ChromaClientManager(settings)

        with pytest.raises(RuntimeError, match="ChromaDB configuration error"):
            manager.get_client()

    def test_get_client_local_permission_error(self, mock_settings, mocker):
        """Test local mode with permission error."""
        mocker.patch("app.services.chroma_manager.os.makedirs")
        mocker.patch("app.services.chroma_manager.os.access", return_value=False)

        manager = ChromaClientManager(mock_settings)

        with pytest.raises(RuntimeError, match="Permission denied for ChromaDB path"):
            manager.get_client()

    def test_get_client_local_os_error(self, mock_settings, mocker):
        """Test local mode with OS error during directory creation."""
        mock_makedirs = mocker.patch("app.services.chroma_manager.os.makedirs")
        mock_makedirs.side_effect = OSError("Directory creation failed")

        manager = ChromaClientManager(mock_settings)

        with pytest.raises(RuntimeError, match="File system error for ChromaDB path"):
            manager.get_client()

    def test_get_client_local_chroma_error(self, mock_settings, mocker):
        """Test local mode with ChromaDB error."""
        mocker.patch("app.services.chroma_manager.os.makedirs")
        mocker.patch("app.services.chroma_manager.os.access", return_value=True)

        mock_chroma = mocker.patch("app.services.chroma_manager.chromadb")
        mock_chroma.PersistentClient.side_effect = ChromaError("ChromaDB error")

        manager = ChromaClientManager(mock_settings)

        with pytest.raises(RuntimeError, match="ChromaDB connection error"):
            manager.get_client()


class TestChromaClientManagerDockerMode:
    """Test ChromaClientManager in docker mode."""

    def test_get_client_docker_success(self, mock_docker_settings, mocker):
        """Test successful docker client creation."""
        # Mock chromadb
        mock_http_client = MagicMock()
        mock_chroma = mocker.patch("app.services.chroma_manager.chromadb")
        mock_chroma.HttpClient.return_value = mock_http_client

        # Mock socket operations for connectivity test
        mock_socket = mocker.patch("app.services.chroma_manager.socket.socket")
        mock_sock_instance = MagicMock()
        mock_socket.return_value = mock_sock_instance
        mock_sock_instance.connect_ex.return_value = 0  # Success

        manager = ChromaClientManager(mock_docker_settings)
        client = manager.get_client()

        assert client == mock_http_client
        assert manager._client == mock_http_client
        mock_chroma.HttpClient.assert_called_once_with(
            host="http://localhost", port=8000
        )
        mock_sock_instance.settimeout.assert_called_once_with(5)
        mock_sock_instance.connect_ex.assert_called_once_with(("localhost", 8000))
        mock_sock_instance.close.assert_called_once()

    def test_get_client_docker_no_host(self):
        """Test docker mode with missing CHROMA_HOST."""
        settings = MagicMock()
        settings.CHROMA_MODE = "docker"
        settings.CHROMA_HOST = None

        manager = ChromaClientManager(settings)

        with pytest.raises(RuntimeError, match="ChromaDB configuration error"):
            manager.get_client()

    def test_get_client_docker_connection_failed(self, mock_docker_settings, mocker):
        """Test docker mode with connection failure."""
        # Mock socket operations for connectivity test failure
        mock_socket = mocker.patch("app.services.chroma_manager.socket.socket")
        mock_sock_instance = MagicMock()
        mock_socket.return_value = mock_sock_instance
        mock_sock_instance.connect_ex.return_value = 1  # Connection failed

        manager = ChromaClientManager(mock_docker_settings)

        with pytest.raises(RuntimeError, match="Cannot connect to ChromaDB"):
            manager.get_client()

    def test_get_client_docker_dns_error(self, mock_docker_settings, mocker):
        """Test docker mode with DNS resolution error."""
        mock_socket = mocker.patch("app.services.chroma_manager.socket.socket")
        mock_sock_instance = MagicMock()
        mock_socket.return_value = mock_sock_instance
        mock_sock_instance.connect_ex.side_effect = socket.gaierror(
            "DNS resolution failed"
        )

        manager = ChromaClientManager(mock_docker_settings)

        with pytest.raises(RuntimeError, match="Cannot resolve ChromaDB host"):
            manager.get_client()

    def test_get_client_docker_timeout(self, mock_docker_settings, mocker):
        """Test docker mode with connection timeout."""
        mock_socket = mocker.patch("app.services.chroma_manager.socket.socket")
        mock_sock_instance = MagicMock()
        mock_socket.return_value = mock_sock_instance
        mock_sock_instance.connect_ex.side_effect = socket.timeout("Connection timeout")

        manager = ChromaClientManager(mock_docker_settings)

        with pytest.raises(RuntimeError, match="Connection timeout to ChromaDB"):
            manager.get_client()

    def test_get_client_docker_default_port(self, mocker):
        """Test docker mode uses default port when not specified."""
        settings = MagicMock()
        settings.CHROMA_MODE = "docker"
        settings.CHROMA_HOST = "http://test-host"
        settings.CHROMA_PORT = None

        # Mock chromadb
        mock_http_client = MagicMock()
        mock_chroma = mocker.patch("app.services.chroma_manager.chromadb")
        mock_chroma.HttpClient.return_value = mock_http_client

        # Mock socket operations
        mock_socket = mocker.patch("app.services.chroma_manager.socket.socket")
        mock_sock_instance = MagicMock()
        mock_socket.return_value = mock_sock_instance
        mock_sock_instance.connect_ex.return_value = 0

        manager = ChromaClientManager(settings)
        manager.get_client()

        # Should use default port 8000
        mock_chroma.HttpClient.assert_called_once_with(
            host="http://test-host", port=8000
        )
        mock_sock_instance.connect_ex.assert_called_once_with(("test-host", 8000))


class TestChromaClientManagerInvalidMode:
    """Test ChromaClientManager with invalid configuration."""

    def test_get_client_invalid_mode(self):
        """Test with invalid CHROMA_MODE."""
        settings = MagicMock()
        settings.CHROMA_MODE = "invalid_mode"

        manager = ChromaClientManager(settings)

        with pytest.raises(RuntimeError, match="ChromaDB configuration error"):
            manager.get_client()


class TestChromaClientManagerReset:
    """Test ChromaClientManager reset functionality."""

    def test_reset_with_resetable_client(self, mock_settings, mocker):
        """Test reset with a client that supports reset."""
        mock_client = MagicMock()
        mock_client.reset = MagicMock()

        manager = ChromaClientManager(mock_settings)
        manager._client = mock_client

        manager.reset()

        mock_client.reset.assert_called_once()
        assert manager._client is None

    def test_reset_with_non_resetable_client(self, mock_settings):
        """Test reset with a client that doesn't support reset."""
        mock_client = MagicMock()
        # Remove reset method
        del mock_client.reset

        manager = ChromaClientManager(mock_settings)
        manager._client = mock_client

        # Should not raise error, just clear the client
        manager.reset()
        assert manager._client is None

    def test_reset_with_no_client(self, mock_settings):
        """Test reset when no client exists."""
        manager = ChromaClientManager(mock_settings)

        # Should not raise error
        manager.reset()
        assert manager._client is None

    def test_reset_with_chroma_error(self, mock_settings, mocker):
        """Test reset when ChromaDB reset raises an error."""
        mock_client = MagicMock()
        mock_client.reset.side_effect = ChromaError("Reset failed")

        manager = ChromaClientManager(mock_settings)
        manager._client = mock_client

        # Should continue with cleanup despite error
        manager.reset()
        assert manager._client is None

    def test_reset_with_unexpected_error(self, mock_settings):
        """Test reset when reset operation raises unexpected error."""
        mock_client = MagicMock()
        mock_client.reset.side_effect = Exception("Unexpected error")

        manager = ChromaClientManager(mock_settings)
        manager._client = mock_client

        # Should continue with cleanup despite error
        manager.reset()
        assert manager._client is None

    def test_reset_cleanup_error(self, mock_settings, mocker):
        """Test reset when cleanup itself fails."""
        mock_client = MagicMock()
        manager = ChromaClientManager(mock_settings)
        manager._client = mock_client

        # Mock hasattr to raise exception, which would trigger the outer except block
        original_hasattr = hasattr

        def mock_hasattr(obj, name):
            if obj is mock_client and name == "reset":
                raise Exception("hasattr failed")
            return original_hasattr(obj, name)

        mocker.patch("builtins.hasattr", side_effect=mock_hasattr)

        with pytest.raises(
            RuntimeError, match="Failed to reset ChromaDB client manager"
        ):
            manager.reset()

        # Should still force cleanup
        assert manager._client is None


class TestChromaClientManagerEdgeCases:
    """Test edge cases and error scenarios."""

    def test_multiple_get_client_calls_after_error(self, mock_settings, mocker):
        """Test that errors don't prevent subsequent successful connections."""
        mock_chroma = mocker.patch("app.services.chroma_manager.chromadb")
        mocker.patch("app.services.chroma_manager.os.makedirs")
        mocker.patch("app.services.chroma_manager.os.access", return_value=True)

        manager = ChromaClientManager(mock_settings)

        # First call fails
        mock_chroma.PersistentClient.side_effect = ChromaError("First error")
        with pytest.raises(RuntimeError):
            manager.get_client()

        # Second call succeeds
        mock_client = MagicMock()
        mock_chroma.PersistentClient.side_effect = None
        mock_chroma.PersistentClient.return_value = mock_client

        client = manager.get_client()
        assert client == mock_client

    def test_get_client_generic_exception(self, mock_settings, mocker):
        """Test handling of unexpected exceptions during client creation."""
        mocker.patch("app.services.chroma_manager.os.makedirs")
        mocker.patch("app.services.chroma_manager.os.access", return_value=True)

        mock_chroma = mocker.patch("app.services.chroma_manager.chromadb")
        mock_chroma.PersistentClient.side_effect = RuntimeError("Unexpected error")

        manager = ChromaClientManager(mock_settings)

        with pytest.raises(RuntimeError, match="Failed to connect to ChromaDB"):
            manager.get_client()

    def test_docker_mode_host_url_parsing(self, mocker):
        """Test that docker mode correctly parses different host URL formats."""
        test_cases = [
            ("http://localhost", "localhost"),
            ("https://localhost", "localhost"),
            ("localhost", "localhost"),
            ("http://chroma-service", "chroma-service"),
        ]

        for host_url, expected_host in test_cases:
            settings = MagicMock()
            settings.CHROMA_MODE = "docker"
            settings.CHROMA_HOST = host_url
            settings.CHROMA_PORT = 8000

            mock_http_client = MagicMock()
            mock_chroma = mocker.patch("app.services.chroma_manager.chromadb")
            mock_chroma.HttpClient.return_value = mock_http_client

            mock_socket = mocker.patch("app.services.chroma_manager.socket.socket")
            mock_sock_instance = MagicMock()
            mock_socket.return_value = mock_sock_instance
            mock_sock_instance.connect_ex.return_value = 0

            manager = ChromaClientManager(settings)
            manager.get_client()

            # Verify socket connection uses parsed host
            mock_sock_instance.connect_ex.assert_called_with((expected_host, 8000))
            # Reset for next iteration
            manager._client = None

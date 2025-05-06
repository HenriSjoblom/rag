from fastapi.testclient import TestClient
from app.config import Settings
# Import the renamed helper function from conftest
from .conftest import get_injected_settings

def test_client_uses_override_settings(
    client: TestClient,
    override_settings: Settings
):
    """
    Verifies that the TestClient's underlying app instance is configured
    to use the settings provided by the override_settings fixture.
    """
    settings_used_by_client_app = get_injected_settings(client)

    # Assert that the settings object retrieved is the exact same object
    # provided by the override_settings fixture.
    assert settings_used_by_client_app is override_settings

    assert settings_used_by_client_app.CHROMA_PATH == override_settings.CHROMA_PATH
    assert settings_used_by_client_app.CHROMA_COLLECTION_NAME == override_settings.CHROMA_COLLECTION_NAME

    print(f"\nTest PASSED: Client is correctly using override settings.")
    print(f"  -> CHROMA_PATH: {settings_used_by_client_app.CHROMA_PATH}")
    print(f"  -> COLLECTION_NAME: {settings_used_by_client_app.CHROMA_COLLECTION_NAME}")

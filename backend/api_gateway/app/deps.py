import httpx

from app.config import Settings, settings as global_settings

# Dependency to get application settings
def get_settings() -> Settings:
    return global_settings
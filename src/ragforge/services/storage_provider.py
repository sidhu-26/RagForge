"""Storage service dependencies and provider factory."""

from fastapi import Depends

from ragforge.core.config import Settings, get_settings
from ragforge.services.storage import LocalFilesystemStorage, StorageEngine

_storage_instance: StorageEngine | None = None


def get_storage_engine(settings: Settings = Depends(get_settings)) -> StorageEngine:
    """FastAPI dependency providing the configured storage engine.

    Returns the storage engine instance based on application settings.
    Currently defaults to LocalFilesystemStorage, but provides the extension point
    to inject AzureBlobStorage or S3Storage seamlessly without altering router logic.
    """
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = LocalFilesystemStorage(settings.STORAGE_LOCAL_ROOT)
    return _storage_instance

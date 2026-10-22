"""Storage service dependencies and provider factory."""

from fastapi import Depends

from ragforge.core.config import Settings, StorageBackendType, get_settings
from ragforge.services.storage import LocalFilesystemStorage, S3Storage, StorageEngine

_storage_instance: StorageEngine | None = None


def create_storage_engine(settings: Settings) -> StorageEngine:
    """Factory creating the configured StorageEngine instance based on settings."""
    if settings.STORAGE_BACKEND == StorageBackendType.S3:
        aws_access_key = (
            settings.AWS_ACCESS_KEY_ID.get_secret_value() if settings.AWS_ACCESS_KEY_ID else None
        )
        aws_secret_key = (
            settings.AWS_SECRET_ACCESS_KEY.get_secret_value()
            if settings.AWS_SECRET_ACCESS_KEY
            else None
        )
        aws_session_token = (
            settings.AWS_SESSION_TOKEN.get_secret_value() if settings.AWS_SESSION_TOKEN else None
        )
        return S3Storage(
            bucket_name=settings.S3_BUCKET,
            region_name=settings.S3_REGION,
            prefix=settings.S3_PREFIX,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            aws_session_token=aws_session_token,
            endpoint_url=settings.S3_ENDPOINT_URL,
        )

    return LocalFilesystemStorage(settings.STORAGE_LOCAL_ROOT)


def get_storage_engine(settings: Settings = Depends(get_settings)) -> StorageEngine:
    """FastAPI dependency providing the configured storage engine."""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = create_storage_engine(settings)
    return _storage_instance

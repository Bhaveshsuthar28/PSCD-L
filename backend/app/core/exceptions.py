"""Typed exception hierarchy and error codes for TRINETRA backend."""

from __future__ import annotations

from typing import Any


class TrinetraBackendError(Exception):
    """Base exception for controlled TRINETRA backend errors."""

    status_code: int = 500
    error_code: str = "INTERNAL_SERVER_ERROR"
    default_message: str = "An unexpected server error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        error_code: str | None = None,
        status_code: int | None = None,
        details: Any = None,
    ) -> None:
        self.message = message or self.default_message
        if error_code is not None:
            self.error_code = error_code
        if status_code is not None:
            self.status_code = status_code
        self.details = details
        super().__init__(self.message)


class ValidationError(TrinetraBackendError):
    status_code = 400
    error_code = "VALIDATION_ERROR"
    default_message = "The request contains invalid parameters."


class InvalidIdentifierError(ValidationError):
    status_code = 400
    error_code = "INVALID_IDENTIFIER"
    default_message = "The provided identifier is invalid."


class UnsupportedMediaError(TrinetraBackendError):
    status_code = 415
    error_code = "UNSUPPORTED_MEDIA_TYPE"
    default_message = "The uploaded media format is not supported."


class PayloadTooLargeError(TrinetraBackendError):
    status_code = 413
    error_code = "UPLOAD_TOO_LARGE"
    default_message = "The uploaded file exceeds the configured maximum size."


class VideoMetadataInvalidError(TrinetraBackendError):
    status_code = 422
    error_code = "VIDEO_METADATA_INVALID"
    default_message = "The uploaded video could not be decoded or has invalid metadata."


class ResourceNotFoundError(TrinetraBackendError):
    status_code = 404
    error_code = "RESOURCE_NOT_FOUND"
    default_message = "The requested resource was not found."


class ConflictError(TrinetraBackendError):
    status_code = 409
    error_code = "CONFLICT"
    default_message = "The requested operation conflicts with current resource state."


class DuplicateVideoError(ConflictError):
    error_code = "DUPLICATE_VIDEO"
    default_message = "An identical video with the same SHA-256 hash already exists."


class ForbiddenPathError(TrinetraBackendError):
    status_code = 403
    error_code = "FORBIDDEN_PATH"
    default_message = "Access outside the allowed storage boundary is forbidden."


class QueueFullError(TrinetraBackendError):
    status_code = 429
    error_code = "PROCESSING_QUEUE_FULL"
    default_message = "The processing queue is currently full. Please retry later."


class StorageExhaustedError(TrinetraBackendError):
    status_code = 507
    error_code = "INSUFFICIENT_STORAGE"
    default_message = "Insufficient disk storage space for this operation."


class ServiceUnavailableError(TrinetraBackendError):
    status_code = 503
    error_code = "SERVICE_UNAVAILABLE"
    default_message = "A required backend service or worker is unavailable."


class DatabaseOperationError(TrinetraBackendError):
    status_code = 500
    error_code = "DATABASE_ERROR"
    default_message = "A database operation failed."


class ModelInitializationError(TrinetraBackendError):
    status_code = 503
    error_code = "MODEL_INITIALIZATION_FAILED"
    default_message = "The TRINETRA inference model could not be initialized."

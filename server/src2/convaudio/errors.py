"""Custom exceptions for convaudio."""


class ConvaudioError(Exception):
    """Base exception for all convaudio errors."""


class UnsupportedAudioFormat(ConvaudioError):
    """Raised when an audio file has an unsupported format or cannot be decoded without ffmpeg."""

    def __init__(self, detected_format: str, message: str | None = None) -> None:
        self.detected_format = detected_format
        msg = message or f"Unsupported audio format detected: {detected_format}. convaudio does not depend on ffmpeg and only supports formats supported natively by soundfile/torchaudio."
        super().__init__(msg)


class LicenceViolation(ConvaudioError):
    """Raised when a software distribution or model weight violates licence policy."""


class TimelineValidationError(ConvaudioError):
    """Raised when a timeline JSON fails schema validation or semantic invariants."""


class StageExecutionError(ConvaudioError):
    """Raised when an error occurs during stage execution."""


class DownloadNotAllowedError(ConvaudioError):
    """Raised when downloading a remote model weight is disallowed or missing credentials."""

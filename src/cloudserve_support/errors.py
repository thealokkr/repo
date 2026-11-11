"""Application errors with stable, safe-to-display codes."""


class CloudServeError(Exception):
    """Base error that can be reported without exposing secret values."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)

    def as_dict(self) -> dict[str, str]:
        """Return the public error contract for logs and command output."""
        return {"code": self.code, "message": self.message}


class ConfigurationError(CloudServeError):
    """Raised when application settings are invalid or incomplete."""

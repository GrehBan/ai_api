"""Errors raised when no valid HTTP response could be obtained."""

from ai_api.exceptions import AIAPIError


class APITransportError(AIAPIError):
    """The request failed below the HTTP layer.

    Attributes:
        transient: Whether retrying the request may succeed.
    """

    def __init__(
        self,
        message: str,
        *,
        transient: bool = True,
    ) -> None:
        """Store the message and the retry hint.

        Args:
            message: Description of the failure.
            transient: Whether the request may be retried.
        """
        super().__init__(message)
        self.transient = transient


class APIConnectionError(APITransportError):
    """The connection could not be established or was lost."""

    def __init__(
        self,
        message: str = "Failed to connect to the API.",
    ) -> None:
        """Create a retryable connection error.

        Args:
            message: Description of the failure.
        """
        super().__init__(message, transient=True)


class APITimeoutError(APITransportError):
    """The request or a read from the stream exceeded its timeout."""

    def __init__(
        self,
        message: str = "API request timed out.",
    ) -> None:
        """Create a retryable timeout error.

        Args:
            message: Description of the failure.
        """
        super().__init__(message, transient=True)


class APITLSError(APITransportError):
    """TLS verification failed; retrying will not help."""

    def __init__(
        self,
        message: str = "TLS connection verification failed.",
    ) -> None:
        """Create a non-retryable TLS error.

        Args:
            message: Description of the failure.
        """
        super().__init__(message, transient=False)


class APIRedirectError(APITransportError):
    """A redirect was not followed; retrying will not help.

    Raised when redirects are off (the default), lead to another origin,
    lack a ``Location`` header or loop.
    """

    def __init__(
        self,
        message: str = "Too many redirects.",
    ) -> None:
        """Create a non-retryable redirect error.

        Args:
            message: Description of the failure.
        """
        super().__init__(message, transient=False)


class APIRequestError(APITransportError):
    """The request itself cannot be sent; retrying will not help.

    Raised when an upload source fails (a missing file, a stream that
    raises) or the HTTP client rejects a header or the URL.
    """

    def __init__(
        self,
        message: str = "Failed to send the request.",
    ) -> None:
        """Create a non-retryable request error.

        Args:
            message: Description of the failure.
        """
        super().__init__(message, transient=False)


class APIResponseError(AIAPIError):
    """A response was received but its body could not be used.

    Raised when the body cannot be read completely, decoded or validated
    against the method's response model. It is never retried.

    Attributes:
        status: HTTP status code of the response, if known.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
    ) -> None:
        """Store the message and the response status.

        Args:
            message: Description of the failure.
            status: HTTP status code of the response.
        """
        self.status = status
        super().__init__(message)

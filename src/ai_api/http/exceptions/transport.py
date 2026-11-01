"""Errors raised when no valid HTTP response could be obtained."""

from __future__ import annotations

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
        message: str = "Redirect was not followed.",
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


class APIStreamInterruptedError(APITransportError):
    """An open stream broke or ended before it was complete.

    Never retried: the request has already started a generation, and
    sending it again would start (and bill) a new one. Raised while
    iterating a stream when the transport fails mid-way (the original
    error, such as an ``APIConnectionError`` or the ``APITimeoutError``
    of a silent stream, is the ``__cause__``), and when the body ends
    before the method's terminal event (no ``__cause__``).

    Attributes:
        items_received: Items the stream had delivered before it broke.
    """

    def __init__(
        self,
        message: str = "The stream was interrupted.",
        *,
        items_received: int,
    ) -> None:
        """Create a non-retryable stream error.

        Args:
            message: Description of the failure.
            items_received: Items the stream had delivered before it
                broke.
        """
        super().__init__(message, transient=False)
        self.items_received = items_received

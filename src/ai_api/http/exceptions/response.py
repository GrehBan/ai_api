"""Errors for responses that arrived but cannot be used."""

from __future__ import annotations

from ai_api.exceptions import AIAPIError


class APIResponseError(AIAPIError):
    """A response was received but its body could not be used.

    Raised when the response is not valid HTTP, its body is larger than
    the session accepts, or the body cannot be decoded or validated
    against the method's response model, or a stream breaks its protocol
    (an item after its terminal event). It is never retried; a body cut
    off by a lost connection is an ``APIConnectionError`` instead, and an
    ``APIStreamInterruptedError`` while a stream is iterated.

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

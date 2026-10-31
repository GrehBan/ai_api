"""Errors for HTTP responses with an error status code."""

from __future__ import annotations

from typing import Any, Final

from ai_api.exceptions import AIAPIError


class APIStatusError(AIAPIError):
    """The API answered with a status code of 400 or higher.

    Instances are created by an error resolver from the response status,
    headers and body; subclasses in ``HTTP_ERRORS`` narrow it down by
    status code.

    Attributes:
        status: HTTP status code of the response.
        code: Provider-specific error code, if the body carried one.
        error_type: Provider-specific error type, if the body carried one.
        provider: Provider that produced the error.
        request_id: Request id from the response headers or body, useful
            when contacting the provider's support.
        retry_after: Seconds to wait before retrying, from the
            ``Retry-After`` header.
        transient: Whether retrying the same request may succeed. This is
            the only flag the retry strategy looks at.
        raw: The parsed result that reported the failure, when a
            successful response or a stream event did: OpenAI's
            ``Response`` with status ``failed``, or its
            ``response.failed`` event. ``None`` for an error status and
            for events that do not fit the item type.
    """

    def __init__(
        self,
        *,
        status: int,
        message: str | None = None,
        code: str | None = None,
        error_type: str | None = None,
        provider: str | None = None,
        request_id: str | None = None,
        retry_after: float | None = None,
        transient: bool = False,
        raw: Any = None,
    ) -> None:
        """Store the error details.

        Args:
            status: HTTP status code of the response.
            message: Error message reported by the API.
            code: Provider-specific error code.
            error_type: Provider-specific error type.
            provider: Provider that produced the error.
            request_id: Request id of the failed request.
            retry_after: Seconds to wait before retrying.
            transient: Whether the request may be retried.
            raw: The parsed result that reported the failure, if any.
        """
        self.status = status
        self.code = code
        self.error_type = error_type
        self.provider = provider
        self.request_id = request_id
        self.retry_after = retry_after
        self.transient = transient
        self.raw = raw

        super().__init__(message)


class BadRequestError(APIStatusError):
    """HTTP 400: the request parameters are invalid."""

    default_message = (
        "Bad request: Invalid request parameters or malformed data."
    )


class AuthenticationError(APIStatusError):
    """HTTP 401: the credentials are missing or invalid."""

    default_message = "Authentication failed: Invalid or missing credentials."


class BillingError(APIStatusError):
    """HTTP 402: payment is required."""

    default_message = (
        "Billing error: Payment is required to access this resource."
    )


class PermissionDeniedError(APIStatusError):
    """HTTP 403: the credentials lack the required permissions."""

    default_message = (
        "Permission denied: Insufficient permissions to access this resource."
    )


class NotFoundError(APIStatusError):
    """HTTP 404: the resource does not exist."""

    default_message = (
        "Not found: The requested resource does not exist or is unavailable."
    )


class RequestTimeoutError(APIStatusError):
    """HTTP 408: the server timed out waiting for the request."""

    default_message = (
        "Request timeout: The server did not respond within the allowed time."
    )


class ConflictError(APIStatusError):
    """HTTP 409: the request conflicts with the resource state."""

    default_message = (
        "Conflict: The request conflicts with the current resource state."
    )


class UnprocessableEntityError(APIStatusError):
    """HTTP 422: the request is well-formed but semantically invalid."""

    default_message = (
        "Unprocessable entity: The request contains "
        "invalid or unsupported data."
    )


class RateLimitError(APIStatusError):
    """HTTP 429: too many requests or tokens."""

    default_message = "Rate limit exceeded: Too many requests or tokens."


class QuotaExceededError(APIStatusError):
    """The account's credits or usage limits are exhausted.

    Not mapped to a status code by default; providers that report quota
    errors with a distinct code can return it from their resolver.
    """

    default_message = (
        "Quota exceeded: Available credits "
        "or usage limits have been exhausted."
    )


class InternalServerError(APIStatusError):
    """HTTP 500: the API failed unexpectedly."""

    default_message = (
        "Internal server error: The API encountered an unexpected error."
    )


class BadGatewayError(APIStatusError):
    """HTTP 502: an upstream server returned an invalid response."""

    default_message = (
        "Bad gateway: The upstream server returned an invalid response."
    )


class ServiceUnavailableError(APIStatusError):
    """HTTP 503: the API is temporarily unavailable."""

    default_message = (
        "Service unavailable: The API is temporarily unavailable."
    )


class GatewayTimeoutError(APIStatusError):
    """HTTP 504: an upstream server did not respond in time."""

    default_message = (
        "Gateway timeout: The upstream server did not respond in time."
    )


HTTP_ERRORS: Final[dict[int, type[APIStatusError]]] = {
    400: BadRequestError,
    401: AuthenticationError,
    402: BillingError,
    403: PermissionDeniedError,
    404: NotFoundError,
    408: RequestTimeoutError,
    409: ConflictError,
    422: UnprocessableEntityError,
    429: RateLimitError,
    500: InternalServerError,
    502: BadGatewayError,
    503: ServiceUnavailableError,
    504: GatewayTimeoutError,
}
"""Error class to raise for each known HTTP status code."""

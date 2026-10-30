"""Default mapping of HTTP error responses to ``APIStatusError`` classes."""

from __future__ import annotations

from collections.abc import Mapping
from http import HTTPStatus
from typing import Any, ClassVar, Final

from ai_api.enums import Provider
from ai_api.http.exceptions.status import HTTP_ERRORS, APIStatusError
from ai_api.http.headers import (
    REQUEST_ID,
    X_REQUEST_ID,
    get_header,
    parse_retry_after,
)

TRANSIENT_STATUSES: Final[frozenset[int]] = frozenset(
    {
        HTTPStatus.REQUEST_TIMEOUT,
        HTTPStatus.INTERNAL_SERVER_ERROR,
        HTTPStatus.BAD_GATEWAY,
        HTTPStatus.SERVICE_UNAVAILABLE,
        HTTPStatus.GATEWAY_TIMEOUT,
        529,  # Overloaded (Anthropic).
    }
)
"""Statuses whose errors are worth retrying."""


def _optional_str(value: Any) -> str | None:
    """Return ``value`` if it is a string, otherwise ``None``.

    Args:
        value: Any value taken from a decoded JSON body.

    Returns:
        The value itself when it is a ``str``, else ``None``.
    """
    return value if isinstance(value, str) else None


class BaseErrorResolver:
    """Turns an HTTP error response into an ``APIStatusError``.

    The body is expected in the common ``{"error": {"code", "type",
    "message"}}`` shape; a string ``error`` (as Ollama sends) is taken as
    the message. Provider-specific resolvers subclass this and override
    ``provider``, ``get_error_code`` or ``classify``.

    Attributes:
        provider: Provider recorded on the created errors.
    """

    provider: ClassVar[Provider] = Provider.GENERIC

    def resolve(
        self,
        *,
        status: int,
        body: Any,
        headers: Mapping[str, str],
    ) -> APIStatusError:
        """Build the error for a failed response.

        Args:
            status: HTTP status code of the response.
            body: Decoded JSON body, or ``None`` if it was not valid JSON.
            headers: Response headers.

        Returns:
            The error to raise; it is returned, not raised, so the caller
            controls the traceback.
        """
        payload: Mapping[str, Any] = body if isinstance(body, Mapping) else {}

        raw_error = payload.get("error")
        error: Mapping[str, Any] = (
            raw_error if isinstance(raw_error, Mapping) else {}
        )

        code = self.get_error_code(error)
        error_type = _optional_str(error.get("type"))
        message = _optional_str(error.get("message"))

        if not message:
            message = _optional_str(raw_error)

        request_id = (
            get_header(headers, X_REQUEST_ID)
            or get_header(headers, REQUEST_ID)
            or _optional_str(payload.get("request_id"))
        )

        retry_after = parse_retry_after(headers)

        error_class, transient = self.classify(
            status=status,
            code=code,
            error_type=error_type,
            message=message,
            retry_after=retry_after,
        )

        return error_class(
            status=status,
            message=message,
            code=code,
            error_type=error_type,
            provider=self.provider,
            request_id=request_id,
            retry_after=retry_after,
            transient=transient,
        )

    def get_error_code(
        self,
        error: Mapping[str, Any],
    ) -> str | None:
        """Extract the provider error code from the ``error`` object.

        Args:
            error: The ``error`` object of the response body.

        Returns:
            The error code, or ``None`` if there is none.
        """
        return _optional_str(error.get("code"))

    def classify(
        self,
        *,
        status: int,
        code: str | None,
        error_type: str | None,
        message: str | None,
        retry_after: float | None,
    ) -> tuple[type[APIStatusError], bool]:
        """Choose the error class and whether the error is transient.

        The default maps the status through ``HTTP_ERRORS`` and treats
        timeouts, server errors and 529 (overloaded) as transient, and
        429 (rate limit) when the API said when to retry.

        Args:
            status: HTTP status code of the response.
            code: Provider error code.
            error_type: Provider error type.
            message: Provider error message.
            retry_after: Seconds from the ``Retry-After`` header.

        Returns:
            The ``APIStatusError`` subclass to instantiate and the
            transient flag.
        """
        error_class = HTTP_ERRORS.get(status, APIStatusError)

        transient = status in TRANSIENT_STATUSES or (
            status == HTTPStatus.TOO_MANY_REQUESTS and retry_after is not None
        )

        return error_class, transient

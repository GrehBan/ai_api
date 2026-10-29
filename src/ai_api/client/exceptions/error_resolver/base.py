"""Default mapping of HTTP error responses to ``APIError`` subclasses."""

import math
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from http import HTTPStatus
from typing import Any, ClassVar

from ai_api.client.exceptions.http import HTTP_ERRORS, APIError
from ai_api.enums import Provider


def optional_str(value: Any) -> str | None:
    """Return ``value`` if it is a string, otherwise ``None``.

    Args:
        value: Any value taken from a decoded JSON body.

    Returns:
        The value itself when it is a ``str``, else ``None``.
    """
    return value if isinstance(value, str) else None


def get_header(
    headers: Mapping[str, str],
    name: str,
) -> str | None:
    """Look up a header case-insensitively.

    Args:
        headers: Response headers, in any mapping type.
        name: Header name to look up.

    Returns:
        The first matching header value, or ``None`` if it is absent.
    """
    name = name.lower()

    for key, value in headers.items():
        if key.lower() == name:
            return value

    return None


def parse_retry_after(
    headers: Mapping[str, str],
) -> float | None:
    """Read the ``Retry-After`` header as a number of seconds.

    Both forms allowed by RFC 9110 are supported: a number of seconds and
    an HTTP date. Dates in the past yield ``0.0``.

    Args:
        headers: Response headers.

    Returns:
        Seconds to wait, or ``None`` if the header is absent or invalid.
    """
    value = get_header(headers, "retry-after")

    if value is None:
        return None

    try:
        seconds = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
        except ValueError, TypeError, IndexError:
            return None

        if date.tzinfo is None:
            date = date.replace(tzinfo=UTC)

        seconds = (date - datetime.now(UTC)).total_seconds()

    if not math.isfinite(seconds):
        return None

    return max(0.0, seconds)


class BaseErrorResolver:
    """Turns an HTTP error response into an ``APIError``.

    The body is expected in the common ``{"error": {"code", "type",
    "message"}}`` shape; a string ``error`` (as Ollama sends) is taken as
    the message. Provider-specific resolvers subclass this and override
    ``provider``, ``get_error_code`` or ``classify``.

    Attributes:
        provider: Provider recorded on the created errors.
    """

    provider: ClassVar[Provider] = Provider.NONE

    def resolve(
        self,
        *,
        status: int,
        body: Any,
        headers: Mapping[str, str],
    ) -> APIError:
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
        error_type = optional_str(error.get("type"))
        message = optional_str(error.get("message")) or optional_str(raw_error)

        request_id = (
            get_header(headers, "x-request-id")
            or get_header(headers, "request-id")
            or optional_str(payload.get("request_id"))
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
        return optional_str(error.get("code"))

    def classify(
        self,
        *,
        status: int,
        code: str | None,
        error_type: str | None,
        message: str | None,
        retry_after: float | None,
    ) -> tuple[type[APIError], bool]:
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
            The ``APIError`` subclass to instantiate and the transient flag.
        """
        error_class = HTTP_ERRORS.get(status, APIError)

        transient = status in {
            408,
            500,
            502,
            503,
            504,
            529,
        } or (
            status == HTTPStatus.TOO_MANY_REQUESTS and retry_after is not None
        )

        return error_class, transient

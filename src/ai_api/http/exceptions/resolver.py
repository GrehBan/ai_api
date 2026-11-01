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
    }
)
"""Statuses whose errors are worth retrying.

Only standard statuses; a provider's own, such as Anthropic's 529
(overloaded), is classified by its resolver.
"""


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
    the message. Provider-specific resolvers subclass this and set
    ``provider`` and ``error_statuses``, and override ``get_error_code``
    or ``classify_status`` where the provider differs.

    A failure reported with a status below 400 (a stream's error event,
    a result that says it failed) is in-band: it is classified by the
    status ``error_statuses`` gives its code, else its type, and it is
    never transient, whatever its class. The generation ran and failed;
    running it again is the caller's choice, as in the official SDKs,
    which retry by the HTTP status only.

    Attributes:
        provider: Provider recorded on the created errors.
        error_statuses: HTTP status that each in-band error code or type
            stands for; empty by default.
    """

    provider: ClassVar[Provider] = Provider.GENERIC
    error_statuses: ClassVar[Mapping[str, int]] = {}

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

        An in-band failure (a status below 400) is first given the status
        ``error_statuses`` lists for its code, else for its type, and is
        never transient; then ``classify_status`` decides.

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
        in_band = status < HTTPStatus.BAD_REQUEST

        if in_band:
            status = self.error_statuses.get(
                code or "", self.error_statuses.get(error_type or "", status)
            )

        error_class, transient = self.classify_status(
            status=status,
            code=code,
            error_type=error_type,
            message=message,
            retry_after=retry_after,
        )
        return error_class, transient and not in_band

    def classify_status(
        self,
        *,
        status: int,
        code: str | None,
        error_type: str | None,
        message: str | None,
        retry_after: float | None,
    ) -> tuple[type[APIStatusError], bool]:
        """Choose the error class and transience of an HTTP status.

        The default maps the status through ``HTTP_ERRORS`` and treats
        timeouts and server errors (``TRANSIENT_STATUSES``) as transient,
        and 429 (rate limit) when the API said when to retry. A provider
        overrides it for its own codes; ``classify`` keeps an in-band
        failure non-transient whatever it returns.

        Args:
            status: HTTP status code, for an in-band failure the one its
                code or type stands for.
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

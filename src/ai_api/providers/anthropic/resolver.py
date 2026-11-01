"""Error mapping of the Anthropic API."""

from __future__ import annotations

from collections.abc import Mapping
from http import HTTPStatus
from typing import Any, Final, cast

from ai_api.http.exceptions.resolver import BaseErrorResolver
from ai_api.http.exceptions.status import (
    APIStatusError,
    QuotaExceededError,
    ServiceUnavailableError,
)
from ai_api.providers.anthropic.api import PROVIDER

OVERLOADED: Final = 529
"""Status of ``overloaded_error``: the API is temporarily overloaded."""

ERROR_STATUSES: Final[Mapping[str, int]] = {
    "invalid_request_error": HTTPStatus.BAD_REQUEST,
    "authentication_error": HTTPStatus.UNAUTHORIZED,
    "billing_error": HTTPStatus.PAYMENT_REQUIRED,
    "permission_error": HTTPStatus.FORBIDDEN,
    "not_found_error": HTTPStatus.NOT_FOUND,
    "conflict_error": HTTPStatus.CONFLICT,
    "request_too_large": HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
    "rate_limit_error": HTTPStatus.TOO_MANY_REQUESTS,
    "api_error": HTTPStatus.INTERNAL_SERVER_ERROR,
    "timeout_error": HTTPStatus.GATEWAY_TIMEOUT,
    "overloaded_error": OVERLOADED,
}
"""HTTP status of each documented error type.

A stream reports an error after its 200 status, as an ``error`` event;
its type tells which error it is.
"""

SPEND_CAP_CODE: Final = "enforced_spend_limit_reached"
"""``error.details.error_code`` of a 429 for a reached tier spend cap."""

SPEND_LIMIT_MESSAGE: Final = "You have reached your specified"
"""Start of the message of a 400 for a reached spend limit you set.

Anthropic gives this 400 no code of its own: it is an
``invalid_request_error`` whose message starts with ``You have reached
your specified API usage limits`` or ``... workspace API usage limits``
(docs: api/rate-limits, "Setting your own spend limit"). The prefix is
all there is to recognize it by; the documented messages are pinned by a
test, so a change shows up when the snapshot is refreshed.
"""


class AnthropicErrorResolver(BaseErrorResolver):
    """Error resolver with Anthropic's specifics.

    Anthropic's ``error`` object has a ``type`` (``rate_limit_error``,
    ``overloaded_error``, ...) and no ``code``; ``details.error_code``,
    when present, is read as the code. A reached spend cap (429 with
    ``enforced_spend_limit_reached``) or spend limit (400 whose message
    says so) is a ``QuotaExceededError`` and is not retried. The 529
    ``overloaded_error`` is a transient ``ServiceUnavailableError``, as
    OpenAI's 503. An ``error`` event of a stream arrives with the
    stream's 200 status, so its class is picked by its type; as for
    every resolver, it is never transient.
    """

    provider = PROVIDER
    error_statuses = ERROR_STATUSES

    def get_error_code(
        self,
        error: Mapping[str, Any],
    ) -> str | None:
        """Read ``details.error_code`` of the ``error`` object.

        Args:
            error: The ``error`` object of the response body.

        Returns:
            The error code, or ``None`` if there is none.
        """
        details: object = error.get("details")
        if not isinstance(details, Mapping):
            return None
        code: object = cast(Mapping[str, Any], details).get("error_code")
        return code if isinstance(code, str) else None

    def classify_status(
        self,
        *,
        status: int,
        code: str | None,
        error_type: str | None,
        message: str | None,
        retry_after: float | None,
    ) -> tuple[type[APIStatusError], bool]:
        """Pick the error class and transience of Anthropic's statuses.

        Args:
            status: HTTP status the error is classified by.
            code: Provider error code, if any.
            error_type: Provider error type, if any.
            message: Error message, if any.
            retry_after: Seconds from ``Retry-After``, if any.

        Returns:
            ``QuotaExceededError`` (never transient) for a reached spend
            cap or limit, a transient ``ServiceUnavailableError`` for 529,
            else the default classification.
        """
        if self._is_spend_limit(status, code, message):
            return QuotaExceededError, False
        if status == OVERLOADED:
            return ServiceUnavailableError, True
        return super().classify_status(
            status=status,
            code=code,
            error_type=error_type,
            message=message,
            retry_after=retry_after,
        )

    @staticmethod
    def _is_spend_limit(
        status: int,
        code: str | None,
        message: str | None,
    ) -> bool:
        """Tell whether an error means a reached spend cap or limit.

        Args:
            status: HTTP status the error is classified by.
            code: Provider error code, if any.
            message: Error message, if any.

        Returns:
            ``True`` for a reached tier spend cap or a spend limit set by
            the organization.
        """
        if status == HTTPStatus.TOO_MANY_REQUESTS:
            return code == SPEND_CAP_CODE
        if status == HTTPStatus.BAD_REQUEST:
            return message is not None and message.startswith(
                SPEND_LIMIT_MESSAGE
            )
        return False

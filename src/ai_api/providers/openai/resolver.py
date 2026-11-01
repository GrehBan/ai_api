"""Error mapping of the OpenAI API."""

from __future__ import annotations

from collections.abc import Mapping
from http import HTTPStatus
from typing import Any, Final, cast

from ai_api.enums import Provider
from ai_api.http.exceptions.resolver import BaseErrorResolver
from ai_api.http.exceptions.status import APIStatusError, QuotaExceededError

QUOTA_CODES: Final = frozenset(
    {
        "insufficient_quota",
        "credit_balance_exhausted",
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
        "organization_usage_limit_exceeded",
    }
)
"""Codes (or the type) of a 429 that retrying does not fix.

The credits, a spend limit or a usage limit are exhausted; only a change
to the account restores access.
"""

ERROR_STATUSES: Final[Mapping[str, int]] = {
    "invalid_request_error": HTTPStatus.BAD_REQUEST,
    "rate_limit_exceeded": HTTPStatus.TOO_MANY_REQUESTS,
    "rate_limit_error": HTTPStatus.TOO_MANY_REQUESTS,
    "slow_down": HTTPStatus.TOO_MANY_REQUESTS,
    "server_error": HTTPStatus.INTERNAL_SERVER_ERROR,
    "service_unavailable_error": HTTPStatus.SERVICE_UNAVAILABLE,
    "server_is_overloaded": HTTPStatus.SERVICE_UNAVAILABLE,
    "server_overloaded": HTTPStatus.SERVICE_UNAVAILABLE,
}
"""HTTP status of each documented error code or type.

The ``error`` event of a stream arrives after the 200 status, so its
code, or else its type, tells which error it is.
"""


def _failed_response(response: Mapping[str, Any]) -> dict[str, Any]:
    """Read the ``error`` of a ``Response`` with status ``failed``.

    Args:
        response: The decoded ``Response``.

    Returns:
        A body with its ``error`` object, or with a message saying that
        OpenAI sent none.
    """
    error = response.get("error")

    if isinstance(error, Mapping):
        return {"error": error}

    return {
        "error": {
            "message": f"Response {response.get('id')} failed without an error"
        }
    }


class OpenAIErrorResolver(BaseErrorResolver):
    """Error resolver with OpenAI's specifics.

    A 429 whose code or type is in ``QUOTA_CODES`` (no credit left, a
    spend or usage limit reached) is a ``QuotaExceededError`` and is not
    retried. The ``error`` event of a response stream is flat
    (``{"type": "error", "code", "message", "param"}``) and is read like
    an ``error`` object; a ``response.failed`` event and a ``Response``
    with status ``failed`` are read by their ``response.error``. They
    come with a 200 status, so their class is picked by their code or
    type through ``ERROR_STATUSES``; a code it does not list, such as
    ``invalid_prompt``, gives ``APIStatusError`` itself. Such an in-band
    failure is never transient, whatever its class: the generation ran
    and failed, and running it again is the caller's choice, as in the
    official SDK, which retries by the HTTP status only.
    """

    provider = Provider.OPENAI

    def resolve(
        self,
        *,
        status: int,
        body: Any,
        headers: Mapping[str, str],
    ) -> APIStatusError:
        """Build the error, reading OpenAI's in-band failures too.

        Args:
            status: HTTP status code of the response.
            body: Decoded JSON body or event data, or ``None``.
            headers: Response headers.

        Returns:
            The error to raise.
        """
        if isinstance(body, Mapping):
            event = cast(Mapping[str, Any], body)
            kind = event.get("type")
            response = event.get("response")

            if kind == "error" and "error" not in event:
                fields = {k: v for k, v in event.items() if k != "type"}
                body = {"error": fields}
            elif kind == "response.failed" and isinstance(response, Mapping):
                body = _failed_response(cast(Mapping[str, Any], response))
            elif (
                event.get("object") == "response"
                and event.get("status") == "failed"
            ):
                body = _failed_response(event)

        return super().resolve(status=status, body=body, headers=headers)

    def classify(
        self,
        *,
        status: int,
        code: str | None,
        error_type: str | None,
        message: str | None,
        retry_after: float | None,
    ) -> tuple[type[APIStatusError], bool]:
        """Pick the error class and whether the error is transient.

        Args:
            status: HTTP status of the response.
            code: Provider error code, if any.
            error_type: Provider error type, if any.
            message: Error message, if any.
            retry_after: Seconds from ``Retry-After``, if any.

        Returns:
            The error class and the ``transient`` flag, always ``False``
            for a failure reported after a status below 400.
        """
        in_band = status < HTTPStatus.BAD_REQUEST

        if in_band:
            status = ERROR_STATUSES.get(
                code or "", ERROR_STATUSES.get(error_type or "", status)
            )
        if status == HTTPStatus.TOO_MANY_REQUESTS and (
            code in QUOTA_CODES or error_type in QUOTA_CODES
        ):
            return QuotaExceededError, False

        error_class, transient = super().classify(
            status=status,
            code=code,
            error_type=error_type,
            message=message,
            retry_after=retry_after,
        )
        return error_class, transient and not in_band

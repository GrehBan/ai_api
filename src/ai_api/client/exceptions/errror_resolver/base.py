
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, ClassVar

from ai_api.client.exceptions.http import HTTP_ERRORS, APIError
from ai_api.providers.enums import Provider


def optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def get_header(
    headers: Mapping[str, str],
    name: str,
) -> str | None:
    name = name.lower()

    for key, value in headers.items():
        if key.lower() == name:
            return value

    return None


def parse_retry_after(
    headers: Mapping[str, str],
) -> float | None:
    value = get_header(headers, "retry-after")

    if value is None:
        return None

    try:
        seconds = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
        except (ValueError, TypeError, IndexError):
            return None

        if date.tzinfo is None:
            date = date.replace(tzinfo=UTC)

        seconds = (
            date - datetime.now(UTC)
        ).total_seconds()

    if not math.isfinite(seconds):
        return None

    return max(0.0, seconds)


class BaseErrorResolver:
    provider: ClassVar[Provider] = Provider.NONE

    def resolve(
        self,
        *,
        status: int,
        body: Any,
        headers: Mapping[str, str],
    ) -> APIError:
        payload = body if isinstance(body, dict) else {}

        error = payload.get("error")

        if not isinstance(error, dict):
            error = {}

        code = self.get_error_code(error)
        error_type = optional_str(error.get("type"))
        message = optional_str(error.get("message"))

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
        error_class = HTTP_ERRORS.get(status, APIError)

        transient = status in {
            408, 500, 502, 503, 504, 529,
        }

        return error_class, transient

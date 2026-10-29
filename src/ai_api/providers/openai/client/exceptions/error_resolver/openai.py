"""Error mapping of the OpenAI API."""

from http import HTTPStatus

from ai_api.client.exceptions.error_resolver.base import BaseErrorResolver
from ai_api.client.exceptions.http import APIError, QuotaExceededError
from ai_api.enums import Provider


class OpenAIErrorResolver(BaseErrorResolver):
    """Error resolver with OpenAI's specifics.

    A 429 with the code ``insufficient_quota`` means the account has no
    credit left: it is a ``QuotaExceededError`` and is not retried.
    """

    provider = Provider.OPENAI

    def classify(
        self,
        *,
        status: int,
        code: str | None,
        error_type: str | None,
        message: str | None,
        retry_after: float | None,
    ) -> tuple[type[APIError], bool]:
        """Pick the error class and whether the error is transient.

        Args:
            status: HTTP status of the response.
            code: Provider error code, if any.
            error_type: Provider error type, if any.
            message: Error message, if any.
            retry_after: Seconds from ``Retry-After``, if any.

        Returns:
            The error class and the ``transient`` flag.
        """
        if (
            status == HTTPStatus.TOO_MANY_REQUESTS
            and code == "insufficient_quota"
        ):
            return QuotaExceededError, False

        return super().classify(
            status=status,
            code=code,
            error_type=error_type,
            message=message,
            retry_after=retry_after,
        )

"""Interface for turning HTTP error responses into exceptions."""

from collections.abc import Mapping
from typing import Any, Protocol

from ai_api.client.exceptions.http import APIError


class IErrorResolver(Protocol):
    """Creates the ``APIError`` for a response with an error status."""

    def resolve(
        self,
        *,
        status: int,
        body: Any,
        headers: Mapping[str, str],
    ) -> APIError:
        """Build the error for a failed response.

        Args:
            status: HTTP status code, 400 or higher.
            body: Decoded JSON body, or ``None`` if it was not valid JSON.
            headers: Response headers.

        Returns:
            The error for the session to raise.
        """
        ...

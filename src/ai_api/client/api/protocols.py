"""Interface describing where an API lives."""

from collections.abc import Mapping
from typing import Protocol

from yarl import URL


class IAPI(Protocol):
    """Builds request URLs for one API.

    Implemented once per provider; the session asks it for the full URL
    of every request.

    Attributes:
        base_url: Root URL of the API, e.g. ``https://api.example.com``.
        version: API version segment, e.g. ``"v1"``.
    """

    base_url: URL
    version: str

    def build_url(
        self,
        method: str,
        query: Mapping[str, str] | None = None,
    ) -> str:
        """Build the absolute URL for a request.

        Args:
            method: Request path of the method, e.g. ``"/chat"``.
            query: Query parameters to append, already converted to
                strings.

        Returns:
            The absolute URL to request.
        """
        ...

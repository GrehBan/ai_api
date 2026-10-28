from collections.abc import Mapping
from typing import Any, Protocol

from ai_api.client.exceptions import APIError


class IErrorResolver(Protocol):
    def resolve(
        self,
        *,
        status: int,
        body: Any,
        headers: Mapping[str, str],
    ) -> APIError:
        ...
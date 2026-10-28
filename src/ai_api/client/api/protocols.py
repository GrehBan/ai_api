from collections.abc import Mapping
from typing import Protocol

from yarl import URL


class IAPI(Protocol):
    base_url: URL
    version: str

    def build_url(
        self,
        method: str,
        query: Mapping[str, str] | None = None,
    ) -> str:
        ...

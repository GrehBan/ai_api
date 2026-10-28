from collections.abc import Mapping
from typing import Any, Protocol

from multidict import CIMultiDict

type HeadersDict = CIMultiDict[Any]
type HeadersMapping = Mapping[str, Any]


class IHeaders(Protocol):
    @property
    def headers(self) -> HeadersDict:
        ...

    def add(self, headers: HeadersMapping) -> None:
        ...

    def copy(self) -> HeadersDict:
        ...

    def build(self, extra: HeadersMapping) -> HeadersDict:
        ...
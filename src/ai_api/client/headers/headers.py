
from multidict import CIMultiDict

from ai_api.client.headers.protocols import HeadersDict, HeadersMapping


class Headers:
    def __init__(self, headers: HeadersMapping | None = None) -> None:
        if headers is None:
            headers = {}

        self.headers = CIMultiDict(headers)

    def add(self, headers: HeadersMapping) -> None:
        self.headers.update(headers)

    def copy(self) -> HeadersDict:
        return self.headers.copy()
    
    def build(self, extra: HeadersMapping) -> HeadersDict:
        headers = self.headers.copy()
        headers.update(extra)
        return headers

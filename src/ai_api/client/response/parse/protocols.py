from typing import Protocol, TypeVar

from ai_api.methods.protocols import IAPIMethod
from ai_api.client.response.base import BufferedResponse


class IResponseParser(Protocol):
    def parse[T](
        self,
        method: IAPIMethod[T],
        buffered: BufferedResponse,
    ) -> T: ...
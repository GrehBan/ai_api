from typing import TYPE_CHECKING, Protocol, Self

from ai_api.methods.protocols import IAPIMethod
from ai_api.types.typevars import ResponseT_co

if TYPE_CHECKING:
    from types import TracebackType


class ISession(Protocol):
    async def close(self) -> None:
        ...

    async def __aenter__(self) -> Self:
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        ...

    async def send_request(
        self,
        method: IAPIMethod[ResponseT_co]
    ) -> ResponseT_co:
        ...
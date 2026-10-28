from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

from tenacity import AsyncRetrying, stop_after_attempt

from ai_api.client.api.protocols import IAPI
from ai_api.client.auth.protocols import IAuth
from ai_api.client.exceptions.errror_resolver.protocols import IErrorResolver
from ai_api.client.headers.headers import Headers
from ai_api.client.headers.protocols import IHeaders
from ai_api.client.retry.startegy import RETRY_CONDITION, calculate_wait
from ai_api.client.serialization.json import JsonDumps, JsonLoads, dumps, loads
from ai_api.methods.protocols import IAPIMethod
from ai_api.types.typevars import ResponseT_co

if TYPE_CHECKING:
    from types import TracebackType


class BaseSession(ABC):
    def __init__(
        self,
        api: IAPI,
        auth: IAuth,
        error_resolver: IErrorResolver,
        headers: IHeaders | None = None,
        json_loads: JsonLoads = loads,
        json_dumps: JsonDumps = dumps
    ) -> None:

        self.api = api
        self.auth = auth
        self.error_resolver = error_resolver
        self.headers = headers or Headers()
        self.json_loads = json_loads
        self.json_dumps = json_dumps

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def send_request(
        self,
        method: IAPIMethod[ResponseT_co]
    ) -> ResponseT_co:

        if not method.__retryable__:
            return await self._send_once(method)

        async for attempt in AsyncRetrying(
            retry=RETRY_CONDITION,
            wait=calculate_wait,
            stop=stop_after_attempt(4),
            reraise=True,
        ):
            with attempt:
                return await self._send_once(method)

        raise RuntimeError("Unexpected retry state")

    @abstractmethod
    async def _send_once(
        self,
        method: IAPIMethod[ResponseT_co],
    ) -> ResponseT_co:
        raise NotImplementedError

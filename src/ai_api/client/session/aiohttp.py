
import asyncio
import ssl
from http import HTTPStatus
from typing import Any, Self

import certifi
from aiohttp import (
    ClientSession,
    TCPConnector,
)

from ai_api.client.api.protocols import IAPI
from ai_api.client.auth.protocols import IAuth
from ai_api.client.exceptions.errror_resolver.protocols import IErrorResolver
from ai_api.client.exceptions.transport import APIResponseError
from ai_api.client.headers.names import (
    AUTHORIZATION,
    SERVER_SOFTWARE,
    USER_AGENT,
)
from ai_api.client.headers.protocols import IHeaders
from ai_api.client.response.base import BufferedResponse
from ai_api.client.serialization.json import JsonDumps, JsonLoads, dumps, loads
from ai_api.client.session.base import BaseSession
from ai_api.client.session.builders.aiohttp import AiohttpBodyBuilder
from ai_api.methods.protocols import IAPIMethod
from ai_api.types.typevars import ResponseT


class AiohttpSession(BaseSession):
    def __init__(
        self,
        api: IAPI,
        auth: IAuth,
        error_resolver: IErrorResolver,
        headers: IHeaders | None = None,
        json_loads: JsonLoads = loads,
        json_dumps: JsonDumps = dumps,
        limit: int = 100,
    ) -> None:
        super().__init__(
            api,
            auth,
            error_resolver,
            headers,
            json_loads,
            json_dumps,
        )

        self._connector_type: type[TCPConnector] = TCPConnector
        self._connector_init: dict[str, Any] = {
            "ssl": ssl.create_default_context(
                cafile=certifi.where()
            ),
            "limit": limit,
            "ttl_dns_cache": 3600,
        }

        self._session: ClientSession | None = None
        self._body_builder = AiohttpBodyBuilder(
            json_dumps=self.json_dumps
        )

    async def __aenter__(self) -> Self:
        await self.create_session()
        return self

    async def close(self) -> None:
        session = self._session
        self._session = None

        if session is not None and not session.closed:
            await session.close()
            await asyncio.sleep(0.25)

    async def create_session(self) -> ClientSession:
        if self._session is None or self._session.closed:
            self._session = ClientSession(
                connector=self._connector_type(
                    **self._connector_init
                ),
                json_serialize_bytes=self.json_dumps,
                headers=self.headers.build(
                    {
                        USER_AGENT: SERVER_SOFTWARE,
                        AUTHORIZATION: self.auth.encode(),
                    }
                ),
            )

        return self._session

    async def _send_once(
        self,
        method: IAPIMethod[ResponseT],
    ) -> ResponseT:
        session = await self.create_session()

        url = self.api.build_url(
            method.__request_path__,
            method.get_query_map(),
        )

        async with session.request(
            method=method.__request_method__,
            url=url,
            **self._body_builder.build(
                method.get_body_map(),
                method.__request_body_format__,
            ),
        ) as response:
            raw = await response.read()

            if response.status >= HTTPStatus.BAD_REQUEST:
                try:
                    body = self.json_loads(raw)
                except (ValueError, TypeError):
                    body = None

                raise self.error_resolver.resolve(
                    status=response.status,
                    body=body,
                    headers=response.headers,
                )

            buffered = BufferedResponse(
                status=response.status,
                headers=response.headers.copy(),
                body=raw,
                charset=response.charset,
            )

        try:
            return method.response_parser.parse(method, buffered)
        except (ValueError, UnicodeError) as exc:
            raise APIResponseError(
                "Failed to parse API response",
                status=buffered.status,
            ) from exc
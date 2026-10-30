# pyright: reportUnnecessaryTypeIgnoreComment=true
"""Static checks of the public API.

Never run: mypy (through ``pytest --mypy``) and pyright check it. Every
``# type: ignore[...]`` marks a line that must stay a type error; mypy's
``warn_unused_ignores`` and pyright's ``reportUnnecessaryTypeIgnoreComment``
fail the check if that error disappears.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal, assert_type

from pydantic import Field

from ai_api import (
    BaseAPI,
    BaseAPIMethod,
    BaseAPIType,
    BaseClient,
    BaseStreamMethod,
    Headers,
    ResponseFormat,
    ServerSentEvent,
    Stream,
)
from ai_api.client import IClient
from ai_api.http import IAPI
from ai_api.methods import IAPIMethod, IStreamMethod
from ai_api.providers.openai import OpenAIClient
from ai_api.providers.openai.client import session_factory
from ai_api.resources import BaseResource, BaseUseCase, IUseCase
from ai_api.response import map_stream
from ai_api.session import ISession
from ai_api.session.aiohttp import AiohttpSession
from ai_api.types.requests import GenerationRequest
from ai_api.types.responses import GenerationResponse


class Model(BaseAPIType):
    id: str


class GetModel(BaseAPIMethod[Model]):
    __request_path__ = "/models/{model_id}"
    __request_method__ = "GET"

    model_id: str


class GetText(BaseAPIMethod[str]):
    __request_path__ = "/text"
    __request_method__ = "GET"


class GetBytes(BaseAPIMethod[bytes]):
    __request_path__ = "/bytes"
    __request_method__ = "GET"


class Delete(BaseAPIMethod[None]):
    __request_path__ = "/items/1"
    __request_method__ = "DELETE"


class Download(BaseStreamMethod[bytes]):
    __request_path__ = "/files/content"
    __request_method__ = "GET"


class Generate(BaseStreamMethod[str]):
    __request_path__ = "/generate"
    __request_method__ = "POST"


class Start(BaseAPIType):
    type: Literal["start"]


class Stop(BaseAPIType):
    type: Literal["stop"]


type Event = Annotated[Start | Stop, Field(discriminator="type")]


class Deltas(BaseStreamMethod[Model]):
    __request_path__ = "/deltas"
    __request_method__ = "POST"
    __response_format__ = ResponseFormat.NDJSON


class Events(BaseStreamMethod[Event]):
    __request_path__ = "/events"
    __request_method__ = "POST"
    __response_format__ = ResponseFormat.SSE


class RawEvents(BaseStreamMethod[ServerSentEvent]):
    __request_path__ = "/events"
    __request_method__ = "GET"


class ListPage(BaseAPIMethod[bytes]):
    __request_path__ = "/items"
    __request_method__ = "GET"
    __request_query__ = {"page_size"}

    page_size: int = Field(
        validation_alias="pageSize",
        serialization_alias="pageSize",
    )


async def send_request_infers_result(session: AiohttpSession) -> None:
    assert_type(await session.send_request(GetModel(model_id="m")), Model)
    assert_type(await session.send_request(GetText()), str)
    assert_type(await session.send_request(GetBytes()), bytes)
    # mypy rejects using the result of a call that returns None, so the
    # call and the await are split.
    deleting = session.send_request(Delete())
    assert_type(await deleting, None)


async def stream_infers_item(session: AiohttpSession) -> None:
    async with session.stream(Download()) as chunks:
        assert_type(chunks, Stream[bytes])
        assert_type(await chunks.save("out.bin"), int)

    async with session.stream(Generate()) as pieces:
        assert_type(pieces, Stream[str])


async def stream_infers_events(session: AiohttpSession) -> None:
    async with session.stream(Deltas()) as deltas:
        assert_type(deltas, Stream[Model])

    async with session.stream(Events()) as events:
        async for event in events:
            assert_type(event, Start | Stop)

    async with session.stream(RawEvents()) as raw:
        assert_type(raw, Stream[ServerSentEvent])


async def wrong_kind_of_method_is_rejected(session: AiohttpSession) -> None:
    await session.send_request(Download())  # type: ignore[arg-type]
    session.stream(GetText())  # type: ignore[arg-type]

    async with session.stream(Generate()) as pieces:
        await pieces.save("out.txt")  # type: ignore[misc]


async def call_options_are_keyword_only(session: AiohttpSession) -> None:
    await session.send_request(GetText(), headers={"X-A": "1"}, timeout=2)
    await session.send_request(GetText(), headers=Headers(), timeout=0.5)
    await session.send_request(GetText(), {"X-A": "1"})  # type: ignore[call-arg]
    await session.send_request(GetText(), timeout="2")  # type: ignore[arg-type]

    async with session.stream(Download(), timeout=30) as chunks:
        assert_type(chunks, Stream[bytes])


def aliased_fields_are_set_by_name() -> None:
    # With validation_alias/serialization_alias, mypy and pyright both
    # take the field name (with alias=, they disagree).
    ListPage(page_size=1)
    ListPage(pageSize=1)  # type: ignore[call-arg]


def session_implements_protocol(session: AiohttpSession) -> ISession:
    return session


def base_api_implements_protocol() -> IAPI:
    # Regression: a base_url typed str | URL did not match IAPI.base_url.
    return BaseAPI("https://api.example.com", "v1")


def methods_implement_protocols() -> None:
    # Regression: pyright once rejected method classes whose dunder
    # attributes are not annotated as ClassVar.
    buffered: IAPIMethod[Model] = GetModel(model_id="m")
    streamed: IStreamMethod[bytes] = Download()
    assert_type(buffered.response_type, type[Model] | None)
    assert_type(streamed.response_type, type[bytes] | None)


async def client_infers_result(client: OpenAIClient) -> IClient:
    assert_type(await client.send_request(GetModel(model_id="m")), Model)

    async with client.stream(Download()) as chunks:
        assert_type(chunks, Stream[bytes])

    return client


def client_keeps_the_session_type(session: AiohttpSession) -> IClient:
    client = BaseClient(session)
    assert_type(client.session, AiohttpSession)
    assert_type(client.session.loop, asyncio.AbstractEventLoop | None)
    BaseClient(object())  # type: ignore[type-var]

    return client


def session_factory_checks_session_arguments() -> None:
    assert_type(session_factory("sk", max_body_size=1), AiohttpSession)
    session_factory("sk", max_body_sizee=1)  # type: ignore[call-arg]


async def map_stream_infers_the_item(session: AiohttpSession) -> None:
    async with map_stream(session.stream(Download()), len) as sizes:
        assert_type(sizes, Stream[int])


class Echo(BaseUseCase[GenerationRequest, GenerationResponse[str]]):
    """Use case without a session type: it defaults to ``ISession``."""

    async def __call__(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse[str]:
        assert_type(self.session, ISession)
        return GenerationResponse[str].model_validate(
            {"provider": "openai", "model": request.model, "raw": "x"}
        )


def use_case_implements_protocol(
    session: AiohttpSession,
) -> IUseCase[GenerationRequest, GenerationResponse[str]]:
    return Echo(session)


def resource_keeps_the_session_type(session: AiohttpSession) -> None:
    resource = BaseResource(session)
    assert_type(resource.session, AiohttpSession)
    BaseResource(object())  # type: ignore[type-var]


class WrongRequest(BaseUseCase[Model, GenerationResponse[str]]):  # type: ignore[type-var]
    """A use case request must be a ``BaseRequest``."""

    async def __call__(self, request: Model) -> GenerationResponse[str]:
        raise NotImplementedError

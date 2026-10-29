"""Block K: buffered requests of ``AiohttpSession`` against a real server."""

import socket
import ssl
from typing import Any

import pytest
from aiohttp import (
    ClientConnectorCertificateError,
    ClientSession,
    ClientTimeout,
    ServerFingerprintMismatch,
    web,
)
from aiohttp.test_utils import TestServer
from pytest_aiohttp import AiohttpServer
from pytest_mock import MockerFixture
from yarl import URL

from ai_api.client.auth.bearer import BearerAuth
from ai_api.client.exceptions.error_resolver.base import BaseErrorResolver
from ai_api.client.exceptions.http import (
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
)
from ai_api.client.exceptions.transport import (
    APIConnectionError,
    APIResponseError,
    APITimeoutError,
    APITLSError,
)
from ai_api.client.headers.headers import Headers
from ai_api.client.headers.names import SERVER_SOFTWARE
from ai_api.client.session.aiohttp import AiohttpSession
from ai_api.methods.base import BaseAPIMethod
from ai_api.methods.enums import BodyFormat
from ai_api.types.base import BaseAPIType
from tests.conftest import TEST_TOKEN, LocalAPI, Recorder, Reply

pytestmark = pytest.mark.integration


class Model(BaseAPIType):
    id: str
    count: int


class Ping(BaseAPIMethod[bytes]):
    __request_path__ = "/ping"
    __request_method__ = "GET"


class RetryablePing(Ping):
    __retryable__ = True


class UpdateModel(BaseAPIMethod[Model]):
    __request_path__ = "/models/{model_id}"
    __request_method__ = "PATCH"
    __request_query__ = {"limit", "dry_run"}
    __request_body_format__ = BodyFormat.JSON

    model_id: str
    limit: int | None = None
    dry_run: bool | None = None
    name: str | None = None
    tags: list[str] | None = None


class SubmitForm(BaseAPIMethod[None]):
    __request_path__ = "/forms"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.FORM

    name: str
    tags: list[str]
    active: bool


class GetText(BaseAPIMethod[str]):
    __request_path__ = "/text"
    __request_method__ = "GET"


class Delete(BaseAPIMethod[None]):
    __request_path__ = "/items/1"
    __request_method__ = "DELETE"


def make_session(api: LocalAPI, **kwargs: Any) -> AiohttpSession:
    return AiohttpSession(
        api,
        BearerAuth(TEST_TOKEN),
        BaseErrorResolver(),
        **kwargs,
    )


async def test_default_headers(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    await session.send_request(Ping())

    headers = recorder.last.headers
    assert headers["Authorization"] == f"Bearer {TEST_TOKEN}"
    assert headers["User-Agent"] == SERVER_SOFTWARE


async def test_extra_headers(api: LocalAPI, recorder: Recorder) -> None:
    extra = Headers({"X-Trace": "abc", "User-Agent": "overridden"})

    async with make_session(api, headers=extra) as session:
        await session.send_request(Ping())

    headers = recorder.last.headers
    assert headers["X-Trace"] == "abc"
    # The library's own headers win over the defaults.
    assert headers["User-Agent"] == SERVER_SOFTWARE


async def test_path_query_and_json_body_in_one_request(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply(json={"id": "m", "count": 2}))

    result = await session.send_request(
        UpdateModel(
            model_id="org/m 1",
            limit=5,
            dry_run=True,
            name="new",
            tags=["a", "b"],
        )
    )

    request = recorder.last
    assert result == Model(id="m", count=2)
    assert request.method == "PATCH"
    assert request.raw_path == "/v1/models/org%2Fm%201"
    assert dict(request.query) == {"limit": "5", "dry_run": "true"}
    assert request.headers["Content-Type"] == "application/json"
    assert request.body == b'{"name":"new","tags":["a","b"]}'


@pytest.mark.parametrize(
    ("model_id", "raw"),
    [
        ("../admin", "..%2Fadmin"),
        ("модель", "%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D1%8C"),
        ("50%", "50%25"),
    ],
    ids=["traversal", "cyrillic", "percent"],
)
async def test_regression_path_is_not_encoded_twice(
    session: AiohttpSession,
    recorder: Recorder,
    model_id: str,
    raw: str,
) -> None:
    recorder.reply(Reply(json={"id": "m", "count": 0}))

    await session.send_request(UpdateModel(model_id=model_id))

    assert recorder.last.raw_path == f"/v1/models/{raw}"


async def test_form_body(session: AiohttpSession, recorder: Recorder) -> None:
    await session.send_request(
        SubmitForm(name="n", tags=["a", "b"], active=False)
    )

    request = recorder.last
    assert request.headers["Content-Type"] == (
        "application/x-www-form-urlencoded"
    )
    assert request.body == b"name=n&tags=a&tags=b&active=false"


async def test_no_body(session: AiohttpSession, recorder: Recorder) -> None:
    await session.send_request(Ping())

    assert recorder.last.method == "GET"
    assert recorder.last.body == b""
    assert recorder.last.query_string == ""


async def test_text_response_in_declared_charset(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    recorder.reply(
        Reply(
            body="привет".encode("cp1251"),
            content_type="text/plain; charset=windows-1251",
        )
    )

    assert await session.send_request(GetText()) == "привет"


async def test_bytes_response(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply(body=b"\x00\x01binary"))

    assert await session.send_request(Ping()) == b"\x00\x01binary"


async def test_empty_204_response(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply(status=204))

    result = await session.send_request(Delete())  # type: ignore[func-returns-value]

    assert result is None


async def test_provider_error(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    recorder.reply(
        Reply(
            status=404,
            json={"error": {"code": "model_not_found", "message": "No"}},
            headers={"x-request-id": "req-42"},
        )
    )

    with pytest.raises(NotFoundError) as info:
        await session.send_request(Ping())

    assert info.value.code == "model_not_found"
    assert info.value.request_id == "req-42"


async def test_transient_errors_are_retried(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply(status=503), Reply(status=503), Reply(body=b"ok"))

    assert await session.send_request(RetryablePing()) == b"ok"
    assert len(recorder.requests) == 3


async def test_retries_give_up(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    recorder.default = Reply(status=503)

    with pytest.raises(ServiceUnavailableError):
        await session.send_request(RetryablePing())

    assert len(recorder.requests) == 4


async def test_rate_limit_with_retry_after_is_retried(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    recorder.reply(
        Reply(status=429, headers={"Retry-After": "12"}),
        Reply(body=b"pong"),
    )

    assert await session.send_request(RetryablePing()) == b"pong"
    assert len(recorder.requests) == 2


@pytest.mark.parametrize(
    "headers",
    [{}, {"Retry-After": "61"}],
    ids=["no-retry-after", "above-the-cap"],
)
async def test_rate_limit_is_raised_at_once(
    session: AiohttpSession,
    recorder: Recorder,
    headers: dict[str, str],
) -> None:
    recorder.reply(Reply(status=429, headers=headers))

    with pytest.raises(RateLimitError):
        await session.send_request(RetryablePing())

    assert len(recorder.requests) == 1


async def test_timeout(api: LocalAPI, recorder: Recorder) -> None:
    recorder.reply(Reply(delay=0.6))

    async with make_session(api, timeout=ClientTimeout(total=0.2)) as session:
        with pytest.raises(APITimeoutError) as info:
            await session.send_request(Ping())

    assert isinstance(info.value.__cause__, TimeoutError)


async def test_closed_port() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    api = LocalAPI(URL(f"http://127.0.0.1:{port}"))

    async with make_session(api) as session:
        with pytest.raises(APIConnectionError):
            await session.send_request(Ping())


async def test_body_cut_short(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply(chunks=[b"partial"], content_length=100, abort=True))

    with pytest.raises(APIResponseError, match="Failed to read") as info:
        await session.send_request(Ping())

    assert info.value.status == 200


@pytest.mark.parametrize(
    "error",
    [
        ClientConnectorCertificateError(
            None,  # type: ignore[arg-type]
            ssl.SSLCertVerificationError("self-signed certificate"),
        ),
        ServerFingerprintMismatch(b"expected", b"got", "example.com", 443),
    ],
    ids=["certificate", "fingerprint"],
)
async def test_tls_errors_are_mapped(
    session: AiohttpSession,
    mocker: MockerFixture,
    error: Exception,
) -> None:
    # A real handshake cannot produce a fingerprint mismatch.
    mocker.patch.object(ClientSession, "request", side_effect=error)

    with pytest.raises(APITLSError) as info:
        await session.send_request(Ping())

    assert info.value.__cause__ is error
    assert info.value.transient is False


async def test_self_signed_certificate_is_rejected(
    aiohttp_server: AiohttpServer,
) -> None:
    trustme = pytest.importorskip("trustme")
    authority = trustme.CA()
    server_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    authority.issue_cert("127.0.0.1").configure_cert(server_context)
    app = web.Application()
    server: TestServer = await aiohttp_server(app, ssl=server_context)

    async with make_session(LocalAPI(server.make_url("/"))) as session:
        with pytest.raises(APITLSError):
            await session.send_request(Ping())


async def test_client_session_is_created_lazily_and_recreated(
    api: LocalAPI,
    recorder: Recorder,
    mocker: MockerFixture,
) -> None:
    created = mocker.patch(
        "ai_api.client.session.aiohttp.ClientSession",
        wraps=ClientSession,
    )
    session = make_session(api)

    assert created.call_count == 0

    await session.send_request(Ping())
    await session.send_request(Ping())
    assert created.call_count == 1

    await session.close()
    await session.close()

    await session.send_request(Ping())
    assert created.call_count == 2
    assert len(recorder.requests) == 3

    await session.close()


async def test_closed_client_session_is_replaced(
    session: AiohttpSession,
) -> None:
    client = await session.create_session()
    await client.close()

    replacement = await session.create_session()

    assert replacement is not client
    assert not replacement.closed
    await session.close()
    await session.close()


async def test_context_manager_opens_and_closes(api: LocalAPI) -> None:
    async with make_session(api) as session:
        client = await session.create_session()
        assert await session.create_session() is client

    assert client.closed

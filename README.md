# ai-api

An async-first toolkit for building typed HTTP API clients, aimed at AI providers.

You describe each API call as a small declarative class. The library sends it over `aiohttp` and returns a validated, statically typed result. It also handles retries, errors, file uploads and streaming.

> **Status: pre-alpha (0.0.4).** The core abstractions are in place, but the public API may still change without notice. `OpenAIClient` sets up the connection to OpenAI, but there are no ready-made OpenAI methods yet. See [Limitations](#limitations-and-roadmap).

## Features

- **Typed methods.** A method is a Pydantic model whose generic argument is its result type. `send_request(method)` returns that type, and both mypy and pyright infer it.
- **Buffered and streamed responses.** Responses can be JSON models, text, bytes or empty. Streams yield byte chunks, text pieces, or typed events from server-sent events (SSE) and newline-delimited JSON (NDJSON) through `async with session.stream(...)`.
- **File uploads.** Upload from bytes, a path on disk, an open binary stream or an async iterator. Each retry resends the full content.
- **Retries.** Transient failures are retried with exponential backoff and jitter, and `Retry-After` is honored up to a cap.
- **Structured errors.** One exception class per HTTP status, plus transport errors. Each error carries the provider error code, request id and a `transient` flag.
- **Provider clients.** `OpenAIClient("sk-...")` builds a session with OpenAI's URL, authentication and error mapping; `BaseClient` and `SessionFactory` are the base for other providers.
- **Pluggable parts.** Sessions depend on protocols, so the API, authentication, error mapping, headers and JSON codec can each be replaced.

## Installation

The library requires Python 3.15 or newer. It is not on PyPI yet, so install it from source:

```bash
git clone https://github.com/GrehBan/ai_api.git
cd ai_api
uv sync
```

Or install it with pip:

```bash
pip install -e .
```

## Quick start

```python
import asyncio

from yarl import URL

from ai_api.client.api.protocols import Query
from ai_api.client.auth.bearer import BearerAuth
from ai_api.client.exceptions.error_resolver.base import BaseErrorResolver
from ai_api.client.session.aiohttp import AiohttpSession
from ai_api.methods.base import BaseAPIMethod
from ai_api.methods.enums import BodyFormat
from ai_api.types.base import BaseAPIType


class ExampleAPI:
    """Where the API lives and how request URLs are built."""

    base_url = URL("https://api.example.com")
    version = "v1"

    def build_url(
        self,
        method: str,
        query: Query | None = None,
    ) -> str:
        # `method` is already percent-encoded: do not encode it again.
        url = self.base_url.joinpath(
            self.version,
            method.lstrip("/"),
            encoded=True,
        )
        return str(url.with_query(query or {}))


class Message(BaseAPIType):
    role: str
    content: str


class Completion(BaseAPIType):
    id: str
    text: str


class CreateCompletion(BaseAPIMethod[Completion]):
    __request_path__ = "/completions"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON
    __retryable__ = True

    model: str
    messages: list[Message]
    temperature: float | None = None


async def main() -> None:
    session = AiohttpSession(
        api=ExampleAPI(),
        auth=BearerAuth("sk-..."),
        error_resolver=BaseErrorResolver(),
    )

    async with session:
        completion = await session.send_request(
            CreateCompletion(
                model="example-1",
                messages=[Message(role="user", content="Hello!")],
            )
        )

    print(completion.text)  # completion: Completion


asyncio.run(main())
```

## Defining methods

A method class declares the HTTP request in its dunder class attributes. Its fields are the request parameters. Fields set to `None` are not sent, and unknown fields raise `ValidationError`, so a misspelled parameter is not sent silently.

| Attribute | Default | Meaning |
|---|---|---|
| `__request_path__` | required | Path appended to the API base URL; may contain `{field}` placeholders |
| `__path_safe__` | `""` | Characters left unescaped in path parameters, e.g. `"/"` |
| `__request_method__` | required | HTTP method, e.g. `"POST"` |
| `__request_query__` | `None` | Names of fields sent as query parameters; fields that are neither path nor query parameters go to the body. Unknown names and mapping or model fields raise `TypeError` at class definition |
| `__request_body_format__` | `BodyFormat.NONE` | `JSON`, `FORM`, `MULTIPART` or `NONE` |
| `__retryable__` | `False` | Retry transient failures (up to 4 attempts in total) |
| `__returning__` | inferred | Result class, taken from the generic argument |
| `__response_format__` | inferred | How the response body is parsed |

The generic argument sets both the result type and the response format:

| Declaration | `send_request` returns | Response format |
|---|---|---|
| `BaseAPIMethod[SomeModel]` | `SomeModel` | JSON, validated into the model |
| `BaseAPIMethod[str]` | `str` | Text, decoded with the response charset |
| `BaseAPIMethod[bytes]` | `bytes` | Raw body |
| `BaseAPIMethod[None]` | `None` | Empty body expected |

Setting `__response_format__` explicitly, in the method or in an intermediate base class, overrides the inferred value. It must still produce the declared type: `TEXT` gives `str`, `BYTES` gives `bytes`, `NONE` gives `None` and `JSON` gives a model, while `SSE` and `NDJSON` are for stream methods only. A format that contradicts the generic argument, such as `BaseAPIMethod[SomeModel]` with `BYTES`, raises `TypeError` at class definition, unless the method sets its own `__response_parser__`.

Query parameters are listed in `__request_query__`:

```python
class ListModels(BaseAPIMethod[ModelList]):
    __request_path__ = "/models"
    __request_method__ = "GET"
    __request_query__ = {"limit"}

    limit: int | None = None          # sent as ?limit=...
```

A list or tuple is sent as a repeated key: `ids=["a", "b"]` becomes `?ids=a&ids=b`.

To send a field under another name, use `validation_alias` and `serialization_alias` rather than `alias`. The constructor then takes the field name in both mypy and pyright; with `alias`, mypy wants the field name and pyright the alias. At runtime both names are accepted.

```python
from pydantic import Field


class ListFiles(BaseAPIMethod[ModelList]):
    __request_path__ = "/files"
    __request_method__ = "GET"
    __request_query__ = {"page_size"}

    page_size: int | None = Field(
        default=None,
        validation_alias="pageSize",
        serialization_alias="pageSize",
    )


ListFiles(page_size=10)   # sent as ?pageSize=10
```

### Path parameters

Placeholders in `__request_path__` are filled from the fields of the same name. Those fields are not sent in the body:

```python
class RetrieveModel(BaseAPIMethod[Model]):
    __request_path__ = "/models/{model_id}"
    __request_method__ = "GET"

    model_id: str


RetrieveModel(model_id="example-1").request_path   # "/models/example-1"
```

- **Escaping.** Values are percent-encoded, including `/`, `?` and `#`, so `model_id="../admin"` cannot change the request path. Enums give their value, as in query parameters.
- **`build_url` gets an encoded path.** Join it without encoding it again, e.g. with `URL.joinpath(..., encoded=True)` as in the quick start. yarl's `/` operator would turn `%2F` into `%252F`, and the server would see a different identifier.
- **Rejected values.** `None`, an empty string and the `.` / `..` segments raise `ValueError` when the request is built.
- **Identifiers with slashes.** For values such as `meta-llama/Llama-3`, set `__path_safe__ = "/"` on that method. Each segment is still checked for `.` and `..`.
- **Checks at class definition.** A misspelled or missing field, a placeholder that is not a plain name (`{}`, `{a.b}`, `{a:>5}`, `{a!r}`) and a field that is also in `__request_query__` raise `TypeError` when the class is defined, not when the request is sent. Write literal braces as `{{` and `}}`.

## Uploading files

Annotate a field as `InputFile` and use a multipart body:

```python
from pathlib import Path

from ai_api.types.files import BytesFile, InputFile, StreamFile


class Transcribe(BaseAPIMethod[Transcription]):
    __request_path__ = "/audio/transcriptions"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.MULTIPART
    __retryable__ = True

    file: InputFile
    model: str


Transcribe(file=Path("voice.mp3"), model="whisper")      # PathFile, audio/mpeg
Transcribe(file=BytesFile(data, "image.png"), model="x")  # in memory
Transcribe(file=StreamFile(chunks(), "s.wav"), model="x") # async iterator
```

| Class | Source | Can be resent on retry |
|---|---|---|
| `BytesFile` | `bytes` in memory; sent with a known length | yes |
| `PathFile` | a file on disk, read in chunks with `aiofiles` | yes |
| `IOFile` | an open binary stream, read from its position at creation | if the stream is seekable |
| `StreamFile` | an async iterator of `bytes` | no |

An `InputFile` field also accepts plain `bytes`, a `Path` or a binary stream and converts it to the matching class. The content type is guessed from the file name when you don't give one.

A request whose files can be read only once is never retried, even if `__retryable__` is set. Text streams are rejected when the method is created, and files in a JSON or URL-encoded body raise `TypeError`. A missing `PathFile` (or one that points to a directory), or a source that fails while it is sent, raises `APIRequestError` and is not retried.

Files other than `BytesFile` are sent with chunked transfer encoding. If a server requires `Content-Length`, as some presigned upload URLs do, use `BytesFile`.

## Streaming responses

Subclass `BaseStreamMethod`. The generic argument is the type of one streamed item: `bytes`, `str`, or an event type (see [Events](#events-sse-and-ndjson)).

```python
from ai_api.methods.base import BaseStreamMethod


class Download(BaseStreamMethod[bytes]):      # raw chunks
    __request_path__ = "/files/content"
    __request_method__ = "GET"
    __request_query__ = {"file_id"}
    __retryable__ = True

    file_id: str


class Generate(BaseStreamMethod[str]):        # decoded text pieces
    __request_path__ = "/generate"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON

    prompt: str


async with session.stream(Download(file_id="f-1")) as stream:
    await stream.save("result.bin")            # only on Stream[bytes]

async with session.stream(Generate(prompt="Hi")) as stream:
    async for piece in stream:                 # piece: str
        print(piece, end="")
```

- **Errors before the stream starts.** An error status raises the matching `APIError` as you enter the `async with` block.
- **Retries.** Only opening the stream is retried. A failure while iterating reaches your code, so it never receives duplicated data.
- **Cleanup.** The connection is released when the block exits, including on `break` or an exception. Iterating the stream after the block, even through an iterator taken inside it, raises `RuntimeError`.
- **Text decoding.** It is incremental, so a multi-byte character split between two chunks still decodes correctly. A charset that is not a text encoding (`base64`, `zlib`) raises `APIResponseError`.
- **Type safety.** Passing a stream method to `send_request`, or a buffered method to `stream`, is a type error.
- **Format and item type.** `BYTES` streams yield `bytes` and `TEXT` streams yield `str`, so an explicit `__response_format__` that contradicts the generic argument, such as `BaseStreamMethod[str]` with `BYTES`, raises `TypeError` at class definition, unless the method sets its own `__stream_parser__`.

### Events: SSE and NDJSON

For a stream of events, set `__response_format__` to `ResponseFormat.SSE` (`text/event-stream`) or `ResponseFormat.NDJSON` (one JSON value per line). The generic argument can be a model or a discriminated union of models, and each event is validated into it.

```python
from typing import Annotated, Literal

from pydantic import Field

from ai_api.client.response.parse.sse import ServerSentEvent
from ai_api.methods.enums import ResponseFormat


class TextDelta(BaseAPIType):
    type: Literal["text_delta"]
    text: str


class MessageStop(BaseAPIType):
    type: Literal["message_stop"]


type MessageEvent = Annotated[
    TextDelta | MessageStop,
    Field(discriminator="type"),
]


class StreamMessage(BaseStreamMethod[MessageEvent]):
    __request_path__ = "/messages"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON
    __response_format__ = ResponseFormat.SSE

    prompt: str
    stream: bool = True


class RawEvents(BaseStreamMethod[ServerSentEvent]):  # SSE, not validated
    __request_path__ = "/events"
    __request_method__ = "GET"


async with session.stream(StreamMessage(prompt="Hi")) as events:
    async for event in events:                 # TextDelta | MessageStop
        if isinstance(event, TextDelta):
            print(event.text, end="")
```

- **Content type.** An SSE response must have `Content-Type: text/event-stream`; anything else (such as a proxy's HTML login page with status 200) raises `APIResponseError` instead of giving an empty stream. For a server that sends another type, use `StreamResponseParser(require_event_stream=False)` as the method's `__stream_parser__`.
- **Format.** It must be set for models and unions: the item type alone doesn't tell SSE from NDJSON. Without it the class definition raises `TypeError`. `ServerSentEvent` items imply SSE.
- **Parsing.** SSE follows the HTML standard: multi-line `data`, comments, `event`, `id` and `retry`, and `\n`, `\r\n` or `\r` line endings. An event split across chunks is held back until it is complete, and an unfinished event at the end of the body is dropped. NDJSON skips blank lines.
- **End of stream.** `data: [DONE]` (OpenAI) ends the stream and is not yielded.
- **Errors inside the stream.** An SSE event of type `error` (Anthropic) raises the `APIError` built by the session's error resolver, so a provider's `classify` can mark it transient. So does JSON data whose `type` is `"error"`, or that has no `type` and a top-level `error` that is `true` or a non-empty object or string (OpenAI, Ollama). An event with another `type` may carry an `error` field as data (a failed tool call, for example). Override `StreamResponseParser.is_error_event` for other rules. An event that doesn't match the item type raises `APIResponseError`.
- **Raw events.** `BaseStreamMethod[ServerSentEvent]` yields events as they are. `[DONE]` and error events are handled the same way.

## Errors and retries

```text
AIAPIError
├── APIError                    HTTP status ≥ 400
│   ├── BadRequestError         400
│   ├── AuthenticationError     401
│   ├── BillingError            402
│   ├── PermissionDeniedError   403
│   ├── NotFoundError           404
│   ├── RequestTimeoutError     408
│   ├── ConflictError           409
│   ├── UnprocessableEntityError 422
│   ├── RateLimitError          429
│   ├── QuotaExceededError      (returned by custom resolvers)
│   ├── InternalServerError     500
│   ├── BadGatewayError         502
│   ├── ServiceUnavailableError 503
│   └── GatewayTimeoutError     504
├── APITransportError           no valid HTTP response
│   ├── APIConnectionError
│   ├── APITimeoutError
│   ├── APITLSError
│   ├── APIRedirectError        redirect not followed
│   └── APIRequestError         failed upload source, invalid header or URL
└── APIResponseError            body could not be read, decoded or validated
```

An `APIError` carries:
- `status`, `code`, `error_type` and `request_id`;
- `retry_after`, from the `Retry-After` header;
- `transient`.

A retry happens only when `transient` is true. By default that means:
- statuses 408, 500, 502, 503, 504 and 529;
- status 429 when the response has `Retry-After`;
- connection errors and timeouts.

Waits use random exponential backoff of up to 30 s, but never less than `Retry-After`. When the API asks to wait longer than `max_retry_after` (60 s by default), the error is raised at once instead. No retry starts after `retry_deadline` (15 minutes from the first attempt by default), and a wait that would end later is not started; an attempt in progress is limited by its own timeout.

`BaseErrorResolver` also takes a string `error` (`{"error": "model not found"}`) as the message. Library errors can be pickled and copied, for example to pass them between processes.

Error responses are mapped by an error resolver. `BaseErrorResolver` reads the common `{"error": {"code", "type", "message"}}` body shape. Subclass it to handle a provider's specifics:

```python
from ai_api.client.exceptions.http import APIError, QuotaExceededError
from ai_api.enums import Provider


class ExampleErrorResolver(BaseErrorResolver):
    provider = Provider.OPENAI

    def classify(
        self,
        *,
        status: int,
        code: str | None,
        error_type: str | None,
        message: str | None,
        retry_after: float | None,
    ) -> tuple[type[APIError], bool]:
        if status == 429 and code == "insufficient_quota":
            return QuotaExceededError, False      # not worth retrying

        return super().classify(
            status=status,
            code=code,
            error_type=error_type,
            message=message,
            retry_after=retry_after,
        )
```

## Session options

`AiohttpSession` takes these arguments:

| Argument | Default | Meaning |
|---|---|---|
| `api` | required | Builds request URLs (`IAPI`) |
| `auth` | required | Produces the authentication header (`IAuth`): `BearerAuth` for `Authorization: Bearer ...`, `APIKeyAuth` for a key in its own header (`x-api-key` by default) |
| `error_resolver` | required | Maps error responses to exceptions (`IErrorResolver`) |
| `headers` | `None` | Extra default headers (`Headers({...})`) |
| `json_loads` / `json_dumps` | `orjson` | JSON codec |
| `limit` | `100` | Maximum number of simultaneous connections |
| `timeout` | 600 s total, 30 s connect | Timeout of buffered requests |
| `stream_timeout` | no total limit, 30 s connect, 600 s between chunks | Timeout of streamed requests |
| `max_retry_after` | `60` | Longest `Retry-After` to wait for, in seconds |
| `retry_deadline` | `900` | Seconds after the first attempt after which no retry starts; `None` for no limit |
| `max_body_size` | `None` | Largest buffered response or error body, in bytes; streamed bodies are not limited |
| `trust_env` | `False` | Use proxy settings and `.netrc` from the environment |
| `follow_redirects` | `False` | Follow redirects within the origin of the request |
| `loop` | `None` | Event loop the session is bound to; `None` binds the loop of its first use |

The authentication header is produced on every request, so an `IAuth` that rotates its token takes effect at once.

Redirects are not followed by default: API providers do not redirect, and aiohttp strips only `Authorization` on a redirect to another host, so a key in `x-api-key` or in the default headers would leak there. A redirect raises `APIRedirectError`. With `follow_redirects=True`, redirects within the same origin (scheme, host and port) are followed; one to another origin still raises `APIRedirectError` before anything is sent there.

A session is bound to one event loop: the one passed as `loop=`, or else the loop of its first request (or `async with`). It can be created anywhere, even at module level, but using or closing it from another loop (such as a second `asyncio.run`) raises `RuntimeError`. The `session.loop` property only reports the bound loop (`None` before first use) and never binds one. Close the session before its loop ends, ideally with `async with`.

Long SSE or NDJSON lines can be limited with `StreamResponseParser(max_line_size=...)`, and the data of one SSE event with `max_event_size=...`, set as a method's `__stream_parser__`.

Timeouts are `aiohttp.ClientTimeout` objects. The default `timeout` is longer than aiohttp's 5 minutes, because long model generations exceed that.

## Provider clients

`OpenAIClient` builds an `AiohttpSession` for OpenAI from a token: the `https://api.openai.com/v1` URL, `BearerAuth` and `OpenAIErrorResolver`, which turns a 429 with the code `insufficient_quota` into a `QuotaExceededError` that is not retried. The client owns its session, so close it, ideally with `async with`:

```python
from ai_api.providers.openai.client.openai import OpenAIClient


async def list_models() -> ModelList:
    async with OpenAIClient("sk-...") as client:
        return await client.send_request(ListModels())
```

- **Custom session settings.** Pass a ready session with `OpenAIClient(session=...)` instead of a token. A `BaseSession` with the generic `BaseErrorResolver`, or with authentication outside the `Authorization` header, gets a `UserWarning`. `session_factory(token, max_body_size=...)` from the same module builds a session with OpenAI's settings and your own options; type checkers check those options against `AiohttpSession`.
- **Other providers.** Subclass `BaseClient`, which delegates `send_request`, `stream`, `close` and `async with` to its session, and build the session with a `SessionFactory` (`client/factory/`). `BaseClient` is generic in the session type: `BaseClient(AiohttpSession(...)).session` is typed as `AiohttpSession`, while `OpenAIClient`, which accepts any session, is a `BaseClient[ISession]`.

## Project layout

```text
src/ai_api/
├── client/
│   ├── base.py         BaseClient; IClient in protocols.py
│   ├── factory/        SessionFactory; validators.py checks client arguments
│   ├── resources/      resources and use cases over shared types
│   ├── api/            IAPI: base URL, URL building, path templates
│   ├── auth/           BearerAuth, APIKeyAuth
│   ├── exceptions/     HTTP and transport errors, error resolvers
│   ├── headers/        default headers
│   ├── response/       BufferedResponse, Stream, map_stream, parsers
│   ├── retry/          retry condition and backoff
│   ├── serialization/  JSON codec, body value normalization
│   └── session/        BaseSession, AiohttpSession, body builders
├── methods/            BaseAPIMethod, BaseStreamMethod, protocols, enums
├── providers/
│   └── openai/         OpenAIClient, OPENAI_API, OpenAIErrorResolver
├── types/              BaseAPIType, InputFile; requests/ and responses/
│                       hold the shared GenerationRequest/Response, events
└── enums.py            Provider
```

`BaseSession` holds the logic that doesn't depend on the HTTP client: retries, error mapping and parsing. A transport such as `AiohttpSession` implements only `_execute` (send the request and read the whole body), `_open_stream` (send the request and leave the body unread) and `close`.

## Development

```bash
uv sync --all-groups                 # runtime, dev and test dependencies
uv run ruff check src tests          # lint
uv run ruff format src tests         # format
uv run mypy src                      # strict type checking
uv run pyright                       # strict type checking of src and tests
uv run pytest --cov                  # tests with branch coverage
uv run pytest --mypy -m mypy tests/typing   # type inference of the public API
uv run pre-commit install            # git hooks
```

The code uses Google-style docstrings and PEP 695 generics. pyright is configured in `pyproject.toml` too; the few rules it disables are explained there.

## Limitations and roadmap

- Streams are not resumed: SSE `id` and `retry` are parsed, but a dropped stream is not reconnected with `Last-Event-ID`.
- There is no raw binary request body (an upload without multipart).
- `OpenAIClient` only sets up the connection: there are no OpenAI method classes yet, and no Anthropic client.
- Resources and use cases (`client/resources/`) and the shared request, response and event types (`types/requests/`, `types/responses/`) are the base of a provider-neutral API; no provider implements them yet.

## License

MIT. See [LICENSE](LICENSE).

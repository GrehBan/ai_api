# ai-api

An async-first toolkit for building typed HTTP API clients, aimed at AI providers.

You describe each API call as a small declarative class. The library sends it over `aiohttp` and returns a validated, statically typed result. It also handles retries, errors, file uploads and streaming.

> **Status: pre-alpha (0.0.2).** The core abstractions are in place, but the public API may still change without notice. There are no ready-made provider clients yet. See [Limitations](#limitations-and-roadmap).

## Features

- **Typed methods.** A method is a Pydantic model whose generic argument is its result type. `send_request(method)` returns that type, and both mypy and pyright infer it.
- **Buffered and streamed responses.** Responses can be JSON models, text, bytes or empty. Streams yield byte chunks or text pieces through `async with session.stream(...)`.
- **File uploads.** Upload from bytes, a path on disk, an open binary stream or an async iterator. Each retry resends the full content.
- **Retries.** Transient failures are retried with exponential backoff and jitter, and `Retry-After` is honored.
- **Structured errors.** One exception class per HTTP status, plus transport errors. Each error carries the provider error code, request id and a `transient` flag.
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
from collections.abc import Mapping

from yarl import URL

from ai_api.client.auth.bearer import BearerAuth
from ai_api.client.exceptions.errror_resolver.base import BaseErrorResolver
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
        query: Mapping[str, str] | None = None,
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

A method class declares the HTTP request in its dunder class attributes. Its fields are the request parameters. Fields set to `None` are not sent.

| Attribute | Default | Meaning |
|---|---|---|
| `__request_path__` | required | Path appended to the API base URL; may contain `{field}` placeholders |
| `__path_safe__` | `""` | Characters left unescaped in path parameters, e.g. `"/"` |
| `__request_method__` | required | HTTP method, e.g. `"POST"` |
| `__request_query__` | `None` | Names of fields sent as query parameters; fields that are neither path nor query parameters go to the body |
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

Setting `__response_format__` explicitly, in the method or in an intermediate base class, overrides the inferred value.

Query parameters are listed in `__request_query__`:

```python
class ListModels(BaseAPIMethod[ModelList]):
    __request_path__ = "/models"
    __request_method__ = "GET"
    __request_query__ = {"limit"}

    limit: int | None = None          # sent as ?limit=...
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

A request whose files can be read only once is never retried, even if `__retryable__` is set. Text streams are rejected when the method is created, and files in a JSON or URL-encoded body raise `TypeError`.

Files other than `BytesFile` are sent with chunked transfer encoding. If a server requires `Content-Length`, as some presigned upload URLs do, use `BytesFile`.

## Streaming responses

Subclass `BaseStreamMethod`. The generic argument is the type of one streamed item:

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
- **Cleanup.** The connection is released when the block exits, including on `break` or an exception.
- **Text decoding.** It is incremental, so a multi-byte character split between two chunks still decodes correctly.
- **Type safety.** Passing a stream method to `send_request`, or a buffered method to `stream`, is a type error.

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
│   └── APITLSError
└── APIResponseError            body could not be read, decoded or validated
```

An `APIError` carries:
- `status`, `code`, `error_type` and `request_id`;
- `retry_after`, from the `Retry-After` header;
- `transient`.

A retry happens only when `transient` is true. By default that means:
- statuses 408, 500, 502, 503, 504 and 529;
- connection errors and timeouts.

Waits use random exponential backoff of up to 30 s, but never less than `Retry-After`.

Error responses are mapped by an error resolver. `BaseErrorResolver` reads the common `{"error": {"code", "type", "message"}}` body shape. Subclass it to handle a provider's specifics:

```python
from ai_api.client.exceptions.http import APIError, QuotaExceededError
from ai_api.providers.enums import Provider


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
| `auth` | required | Produces the `Authorization` header (`IAuth`), e.g. `BearerAuth` |
| `error_resolver` | required | Maps error responses to exceptions (`IErrorResolver`) |
| `headers` | `None` | Extra default headers (`Headers({...})`) |
| `json_loads` / `json_dumps` | `orjson` | JSON codec |
| `limit` | `100` | Maximum number of simultaneous connections |
| `timeout` | 600 s total, 30 s connect | Timeout of buffered requests |
| `stream_timeout` | no total limit, 30 s connect, 600 s between chunks | Timeout of streamed requests |

Timeouts are `aiohttp.ClientTimeout` objects. The default `timeout` is longer than aiohttp's 5 minutes, because long model generations exceed that.

## Project layout

```text
src/ai_api/
├── client/
│   ├── api/            IAPI: base URL and URL building
│   ├── auth/           BearerAuth
│   ├── exceptions/     HTTP and transport errors, error resolvers
│   ├── headers/        default headers
│   ├── response/       BufferedResponse, Stream and the parsers
│   ├── retry/          retry condition and backoff
│   ├── serialization/  JSON codec, body value normalization
│   └── session/        BaseSession, AiohttpSession, body builders
├── methods/            BaseAPIMethod, BaseStreamMethod, protocols, enums
├── providers/          provider identifiers
└── types/              BaseAPIType, InputFile and its subclasses
```

`BaseSession` holds the logic that doesn't depend on the HTTP client: retries, error mapping and parsing. A transport such as `AiohttpSession` implements only `_execute` (send the request and read the whole body), `_open_stream` (send the request and leave the body unread) and `close`.

## Development

```bash
uv sync --all-groups                 # runtime, dev and test dependencies
uv run ruff check src                # lint
uv run ruff format src               # format
uv run mypy src                      # strict type checking
uv run pytest                        # tests (tests/ does not exist yet)
```

The code uses Google-style docstrings and PEP 695 generics. pyright is configured in `pyproject.toml` too; the few rules it disables are explained there.

## Limitations and roadmap

- Server-sent events (`SSE`) and newline-delimited JSON (`NDJSON`) streams are declared but not parsed yet. Event models for streams, including discriminated unions, will come with them.
- There is no raw binary request body (an upload without multipart).
- There are no provider clients (OpenAI, Anthropic) yet, only the building blocks.
- There is no test suite yet.

## License

MIT. See [LICENSE](LICENSE).

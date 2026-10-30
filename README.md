# ai-api

An async-first toolkit for building typed HTTP API clients, aimed at AI providers.

You describe each API call as a small declarative class. The library sends it over `aiohttp` and returns a validated, statically typed result. It also handles retries, errors, file uploads and streaming.

> **Status: pre-alpha (0.1.0).** The core abstractions are in place, but the public API may still change without notice. `OpenAIClient` sets up the connection to OpenAI, but there are no ready-made OpenAI methods yet. See [Limitations](#limitations-and-roadmap).

## Contents

- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [How a request flows](#how-a-request-flows)
- [Defining methods](#defining-methods): attributes, request body, query and path parameters, responses
- [Uploading files](#uploading-files)
- [Streaming responses](#streaming-responses): events, interrupted streams
- [Errors and retries](#errors-and-retries)
- [Sessions](#sessions): options, authentication, headers, timeouts, redirects, event loop
- [Describing an API](#describing-an-api)
- [Provider clients](#provider-clients), [adding a provider](#adding-a-provider)
- [Resources and use cases](#resources-and-use-cases)
- [Writing a transport](#writing-a-transport)
- [Type safety](#type-safety)
- [Security](#security)
- [Design decisions](#design-decisions)
- [Project layout](#project-layout)
- [Development](#development)
- [Limitations and roadmap](#limitations-and-roadmap)

## Features

- **Typed methods.** A method is a Pydantic model whose generic argument is its result type. `send_request(method)` returns that type, and both mypy and pyright infer it.
- **Buffered and streamed responses.** Responses can be JSON models, text, bytes or empty. Streams yield byte chunks, text pieces, or typed events from server-sent events (SSE) and newline-delimited JSON (NDJSON) through `async with session.stream(...)`.
- **File uploads.** Upload from bytes, a path on disk, an open binary stream or an async iterator. Each retry resends the full content.
- **Retries.** Transient failures are retried with exponential backoff and jitter, and `Retry-After` is honored up to a cap.
- **Structured errors.** One exception class per HTTP status, plus transport errors. Each error carries the provider error code, request id and a `transient` flag.
- **Provider clients.** `OpenAIClient("sk-...")` builds a session with OpenAI's URL, authentication and error mapping; `BaseClient` and `SessionFactory` are the base for other providers.
- **Pluggable parts.** Sessions depend on protocols, so the API, authentication, error mapping, headers and JSON codec can each be replaced.
- **Safe defaults.** Path parameters are escaped, redirects to other hosts are refused, proxy settings from the environment are ignored unless enabled, and secrets are kept out of `repr`.

## Installation

The library requires Python 3.13 or newer. It is not on PyPI yet, so install it from source:

```bash
git clone https://github.com/GrehBan/ai_api.git
cd ai_api
uv sync
```

Or install it with pip:

```bash
pip install -e .
```

Runtime dependencies:

| Package | Used for |
|---|---|
| `aiohttp` | the HTTP transport |
| `aiodns`, `brotli` | asynchronous DNS resolution and `br`-compressed responses in aiohttp |
| `pydantic` | methods, response models and validation |
| `orjson` | the default JSON codec |
| `tenacity` | the retry loop |
| `yarl`, `multidict` | URLs and case-insensitive headers |
| `aiofiles` | reading uploads from disk and `Stream.save` |
| `certifi` | the CA bundle for TLS |

## Quick start

```python
import asyncio

from ai_api import (
    BaseAPI,
    BaseAPIMethod,
    BaseAPIType,
    BaseErrorResolver,
    BearerAuth,
    BodyFormat,
)
from ai_api.session.aiohttp import AiohttpSession

API = BaseAPI("https://api.example.com", "v1")


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
        api=API,
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

This sends `POST https://api.example.com/v1/completions` with the body `{"model": "example-1", "messages": [{"role": "user", "content": "Hello!"}]}`. `temperature` is `None`, so it is not sent at all. The response is validated into `Completion`, and type checkers know that `completion` is a `Completion`.

## How a request flows

A buffered request, `await session.send_request(method)`:

```text
send_request(method)
└── retry loop: only if method.retryable, up to 4 attempts
    └── one attempt
        ├── URL:   api.build_url(method.request_path, method.get_query_map())
        ├── body:  method.get_body_map() → normalized → JSON / form / multipart
        ├── send:  the transport sends the request and reads the whole body
        ├── check: status ≥ 400 → error resolver → APIStatusError(transient=…)
        └── parse: method.response_parser → the typed result
```

A streamed request, `async with session.stream(method) as stream`:

```text
stream(method)
├── retry loop around opening only
│   ├── send:  the transport sends the request and leaves the body unread
│   └── check: status ≥ 400 → the error body is read → APIStatusError
├── yield Stream(method.stream_parser.parse(...))   items are parsed as you iterate
└── on exit: the parser and the connection are closed (also on break and errors)
```

Two things follow from this design:

- **The status check is inside the retried attempt.** Whether to retry is decided by the `transient` flag of the error raised there, so the error resolver alone decides which statuses are retried.
- **Only opening a stream is retried.** Once your code has received items, the library cannot resend the request without duplicating them. See [Interrupted streams](#interrupted-streams).

## Defining methods

A method class declares the HTTP request in its dunder class attributes. Its fields are the request parameters. Fields set to `None` are not sent, and unknown fields raise `ValidationError`, so a misspelled parameter is not sent silently.

| Attribute | Default | Meaning |
|---|---|---|
| `__request_path__` | required | Path appended to the API base URL; may contain `{field}` placeholders |
| `__request_path_safe__` | `""` | Characters left unescaped in path parameters, e.g. `"/"` |
| `__request_method__` | required | HTTP method, e.g. `"POST"` |
| `__request_query__` | empty | Names of fields sent as query parameters; fields that are neither path nor query parameters go to the body. Unknown names and mapping or model fields raise `TypeError` at class definition |
| `__request_body_format__` | `BodyFormat.NONE` | `JSON`, `FORM`, `MULTIPART` or `NONE` |
| `__retryable__` | `False` | Retry transient failures (up to 4 attempts in total) |
| `__returning__` | inferred | Result class, taken from the generic argument |
| `__response_format__` | inferred | How the response body is parsed |
| `__response_parser__` | `BufferedResponseParser()` | Parser of buffered responses (`BaseAPIMethod`) |
| `__stream_parser__` | `StreamResponseParser()` | Parser of streamed responses (`BaseStreamMethod`) |

The attributes are inherited, so an intermediate base class can hold what a group of methods shares:

```python
class JSONPost[T](BaseAPIMethod[T]):          # shared by a group of endpoints
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON


class Summarize(JSONPost[Completion]):        # POST /summaries, JSON body
    __request_path__ = "/summaries"

    text: str
```

Why dunder class attributes rather than fields or a decorator:

- **They never collide with request parameters.** An API may well have a parameter named `path`, `method` or `format`.
- **They describe the class, not an instance.** The configuration is checked once, when the class is defined: a bad path template, an unknown query field or a response format that contradicts the result type raises `TypeError` at import time, not at the first request.
- **The design follows aiogram's `TelegramMethod`.** A request is data that knows its own endpoint and result type, so one generic `send_request` serves every endpoint, and adding an endpoint means adding one small class.

Inside the library, these attributes are read only through regular properties (`request_path`, `request_method`, `request_body_format`, `retryable`, `response_format`, `response_type`, `response_parser`, `stream_parser`), which is what the session and parsers see.

### Validation of parameters

Methods are validated by Pydantic when they are created, before anything is sent:

- `extra="forbid"`: an unknown field raises `ValidationError`. A misspelled optional parameter would otherwise be dropped silently, and the API would use its default without telling you.
- Fields can be given by name or by alias.
- Any Pydantic constraint works, e.g. `temperature: float = Field(ge=0, le=2)`.

Response models are the opposite on purpose. `BaseAPIType` uses `extra="ignore"`, so a field the provider adds tomorrow does not break today's client.

### Result type and response format

The generic argument sets both the result type and the response format:

| Declaration | `send_request` returns | Response format |
|---|---|---|
| `BaseAPIMethod[SomeModel]` | `SomeModel` | JSON, validated into the model |
| `BaseAPIMethod[str]` | `str` | Text, decoded with the response charset |
| `BaseAPIMethod[bytes]` | `bytes` | Raw body |
| `BaseAPIMethod[None]` | `None` | Empty body expected |

Setting `__response_format__` explicitly, in the method or in an intermediate base class, overrides the inferred value. It must still produce the declared type: `TEXT` gives `str`, `BYTES` gives `bytes`, `NONE` gives `None` and `JSON` gives a model, while `SSE` and `NDJSON` are for stream methods only. A format that contradicts the generic argument, such as `BaseAPIMethod[SomeModel]` with `BYTES`, raises `TypeError` at class definition, unless the method sets its own `__response_parser__`. The check exists because the parser casts its result to the declared type: without it, a contradicting format would make the static type lie.

How `BufferedResponseParser` reads each format:

- **JSON** is validated with `model_validate_json`, directly from the bytes, in Pydantic's JSON mode. With a custom `json_loads` the body is decoded first and validated in Python mode, where strict fields reject JSON forms such as a date string.
- **TEXT** is decoded with the charset of the response, or UTF-8 when there is none.
- **BYTES** is returned as it is.
- **NONE** expects an empty body; anything else raises `APIResponseError`.

A body that cannot be decoded or validated raises `APIResponseError`, which is never retried: sending the same request again would get the same answer.

### Request body

`__request_body_format__` decides how the body fields are encoded:

| Format | Sent as | Notes |
|---|---|---|
| `NONE` | no body | the default, for `GET` and `DELETE` |
| `JSON` | `application/json` | NaN and infinity raise `ValueError`; files and `bytes` raise `TypeError` |
| `FORM` | `application/x-www-form-urlencoded` | files raise `TypeError` |
| `MULTIPART` | `multipart/form-data` | the only format that carries files |

Before encoding, the values are normalized:

- **`None`** fields are dropped, at the top level and in nested models, but kept as `null` inside lists and dicts, where the position or key matters.
- **Nested models** (`BaseAPIType`) are dumped to dicts.
- **Enums** are sent as their values.
- **Datetimes** become Unix timestamps in seconds, as strings. A naive datetime raises `ValueError`, because its moment is ambiguous.
- **Timedeltas** become the timestamp of now plus the delta, for parameters such as an expiry time.
- **Circular references** and nesting deeper than 256 levels raise `ValueError` instead of hitting Python's recursion limit.

In `FORM` and `MULTIPART` bodies a list sends its field once per item, a dict is sent as a JSON string and a boolean as `"true"` or `"false"`.

### Query parameters

Query parameters are listed in `__request_query__`:

```python
class ListModels(BaseAPIMethod[ModelList]):
    __request_path__ = "/models"
    __request_method__ = "GET"
    __request_query__ = {"limit"}

    limit: int | None = None          # sent as ?limit=...
```

A list or tuple is sent as a repeated key: `ids=["a", "b"]` becomes `?ids=a&ids=b`. Booleans become `true` and `false`, and enums their values. A mapping or a model cannot be a query parameter, since there is no standard way to put it in a URL; such a field in `__request_query__` raises `TypeError` at class definition.

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
- **`build_url` gets an encoded path.** Join it without encoding it again, e.g. with `URL.joinpath(..., encoded=True)` as `BaseAPI` does. yarl's `/` operator would turn `%2F` into `%252F`, and the server would see a different identifier.
- **Rejected values.** `None`, an empty string and the `.` / `..` segments raise `ValueError` when the request is built.
- **Identifiers with slashes.** For values such as `meta-llama/Llama-3`, set `__request_path_safe__ = "/"` on that method. Each segment is still checked for `.` and `..`.
- **Checks at class definition.** A misspelled or missing field, a placeholder that is not a plain name (`{}`, `{a.b}`, `{a:>5}`, `{a!r}`) and a field that is also in `__request_query__` raise `TypeError` when the class is defined, not when the request is sent. Write literal braces as `{{` and `}}`.

## Uploading files

Annotate a field as `InputFile` and use a multipart body:

```python
from pathlib import Path

from ai_api import BytesFile, InputFile, StreamFile


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

How files behave on retry, and why:

- **Every attempt opens the file again.** `InputFile.open()` returns a fresh iterator each time, so a retry sends the whole content, not the rest of a half-read stream.
- **A one-shot source disables retries.** A request whose files can be read only once (`StreamFile`, a non-seekable `IOFile`) is never retried, even if `__retryable__` is set: a retry would send an empty or truncated file.
- **A failing source is not a network error.** A missing `PathFile` (or one that points to a directory), or a source that fails while it is sent, raises `APIRequestError`, which is not retried. Retrying would fail the same way.

Text streams are rejected when the method is created, and files in a JSON or URL-encoded body raise `TypeError`.

Files other than `BytesFile` are sent with chunked transfer encoding, so they are never read into memory as a whole. If a server requires `Content-Length`, as some presigned upload URLs do, use `BytesFile`.

## Streaming responses

Subclass `BaseStreamMethod`. The generic argument is the type of one streamed item: `bytes`, `str`, or an event type (see [Events](#events-sse-and-ndjson)).

```python
from ai_api import BaseStreamMethod


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

- **Errors before the stream starts.** An error status raises the matching `APIStatusError` as you enter the `async with` block.
- **Retries.** Only opening the stream is retried. A failure while iterating reaches your code, so it never receives duplicated data.
- **Cleanup.** The connection is released when the block exits, including on `break` or an exception. Iterating the stream after the block, even through an iterator taken inside it, raises `RuntimeError`, instead of quietly yielding nothing from a closed connection.
- **Text decoding.** It is incremental, so a multi-byte character split between two chunks still decodes correctly. A charset that is not a text encoding (`base64`, `zlib`) raises `APIResponseError`.
- **Saving to a file.** `Stream.save(path)` writes chunks as they arrive, without holding the body in memory, and returns the number of bytes written.
- **Timeouts.** Streams have no total time limit, only 600 s between two chunks, so a long generation is not cut off. See [Timeouts](#timeouts).
- **Type safety.** Passing a stream method to `send_request`, or a buffered method to `stream`, is a type error.
- **Format and item type.** `BYTES` streams yield `bytes` and `TEXT` streams yield `str`, so an explicit `__response_format__` that contradicts the generic argument, such as `BaseStreamMethod[str]` with `BYTES`, raises `TypeError` at class definition, unless the method sets its own `__stream_parser__`.

### Events: SSE and NDJSON

For a stream of events, set `__response_format__` to `ResponseFormat.SSE` (`text/event-stream`) or `ResponseFormat.NDJSON` (one JSON value per line). The generic argument can be a model or a discriminated union of models, and each event is validated into it.

```python
from typing import Annotated, Literal

from pydantic import Field

from ai_api import ResponseFormat, ServerSentEvent


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
- **Validation.** Each event is validated through a `TypeAdapter` built once, when the class is defined, from the generic argument as written. That is why discriminated unions and `type` aliases work as item types. An event that doesn't match raises `APIResponseError`.
- **End of stream.** `data: [DONE]` (OpenAI) ends the stream and is not yielded.
- **Errors inside the stream.** An SSE event of type `error` (Anthropic) raises the `APIStatusError` built by the session's error resolver, so a provider's `classify` can mark it transient. So does JSON data whose `type` is `"error"`, or that has no `type` and a top-level `error` that is `true` or a non-empty object or string (OpenAI, Ollama). An event with another `type` may carry an `error` field as data (a failed tool call, for example). Override `StreamResponseParser.is_error_event` for other rules.
- **Raw events.** `BaseStreamMethod[ServerSentEvent]` yields events as they are, with their `data`, `event`, `id` and `retry`. `[DONE]` and error events are handled the same way.
- **Limits.** `StreamResponseParser(max_line_size=...)` limits one SSE or NDJSON line and `max_event_size=...` the data of one SSE event, in characters. Both are off by default; set them for an untrusted server, since a line without an end would otherwise grow in memory.

### Interrupted streams

A connection can break at any point of a long generation. What happens depends on when it breaks:

| When | What you get | Retried |
|---|---|---|
| Before the response arrives: connection refused, DNS, TLS, connect timeout | `APIConnectionError`, `APITimeoutError` or `APITLSError` on entering the block | yes, if the method is retryable and the error is transient |
| The server answers with an error status | the `APIStatusError` for it, on entering the block | yes, if the error is transient |
| The connection drops after some items | `APIConnectionError` from `async for` | no |
| No data for 600 s (`sock_read`) | `APITimeoutError` from `async for` | no |
| The server sends an error event | the `APIStatusError` for it, from `async for` | no |

Items you received before the failure stay yours, and the connection is released as the block exits. The error's `transient` flag still says whether trying again makes sense, but the library does not try again on its own.

**Why a broken stream is not retried automatically.** Opening the stream again would send the request again. For a generation endpoint that is a new, separately billed generation with different output. The library cannot know what your code has already done with the items it received, such as printing them or forwarding them to a user, so it cannot splice the two streams into one without duplicating or losing text.

**Why the SSE resume protocol is not used.** SSE has its own way to continue a stream. The server marks events with `id:`, the client reconnects after `retry:` milliseconds and sends the last id in the `Last-Event-ID` header, and the server replays what came after it. The library parses `id` and `retry`; you can read them on `ServerSentEvent` items. It does not reconnect, for three reasons:

- **It only works if the server keeps the events and replays them.** AI providers' generation streams generally don't. Where a provider supports resuming, it does so with its own parameters (OpenAI's background responses, for example, are resumed from an event's `sequence_number` with `starting_after`). That is a provider method, not a transport feature.
- **Methods cannot set per-request headers yet,** so there is no place to put `Last-Event-ID`. See [Limitations](#limitations-and-roadmap).
- **Silent reconnection hides failures.** A client that reconnects on its own can loop for a long time against a broken server, while the caller sees a stream that is merely slow.

**What to do instead.** Decide in your code what a broken stream means. For short answers the simplest choice is to start over and drop the partial text:

```python
from ai_api import APIConnectionError, APITimeoutError


async def generate_text(prompt: str, attempts: int = 3) -> str:
    for attempt in range(1, attempts + 1):
        pieces: list[str] = []

        try:
            async with session.stream(Generate(prompt=prompt)) as stream:
                async for piece in stream:
                    pieces.append(piece)
        except (APIConnectionError, APITimeoutError):
            if attempt == attempts:
                raise
            continue                        # drop the partial text, ask again

        return "".join(pieces)

    raise ValueError("attempts must be at least 1")
```

For a provider that can resume, keep the position from the last event you received (its `id`, or a sequence number in its data) and call the provider's resume method with it.

An opt-in reconnection with `Last-Event-ID` for servers that support it is on the [roadmap](#limitations-and-roadmap): limited attempts, the `retry` delay, and the same `Stream` object continuing with the replayed events.

## Errors and retries

```text
AIAPIError
├── APIStatusError              HTTP status ≥ 400
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
│   ├── APIConnectionError      also a body cut off by a lost connection
│   ├── APITimeoutError
│   ├── APITLSError
│   ├── APIRedirectError        redirect not followed
│   └── APIRequestError         failed upload source, invalid header or URL
└── APIResponseError            invalid HTTP, body too large, not decodable or valid
```

The three branches answer the question "what came back?":

- **`APIStatusError`**: the server answered, with an error status. Other statuses without their own class are raised as `APIStatusError` itself.
- **`APITransportError`**: no usable answer came back, because the request failed on the way.
- **`APIResponseError`**: an answer came back, but it cannot be used. It is not valid HTTP, it is larger than `max_body_size`, or it does not decode or validate.

An `APIStatusError` carries:
- `status`, `code`, `error_type` and `request_id`;
- `provider`, from the error resolver;
- `retry_after`, from the `Retry-After` header;
- `transient`.

Every error message starts with the class's fixed description and ends with `detail: <the API's message>`, so logs have a stable prefix and the provider's own explanation.

```python
from ai_api import APIStatusError, APITransportError, RateLimitError


async def ask(prompt: str) -> str | None:
    method = CreateCompletion(
        model="example-1",
        messages=[Message(role="user", content=prompt)],
    )

    try:
        completion = await session.send_request(method)
    except RateLimitError as error:            # retries did not help
        print(f"Rate limited, request {error.request_id}")
        return None
    except APIStatusError as error:
        print(f"API error {error.status}, code {error.code}")
        raise
    except APITransportError as error:
        print(f"Network error, transient: {error.transient}")
        raise

    return completion.text
```

### When a request is retried

A request is retried only when the method sets `__retryable__ = True`, does not carry a one-shot file, and the error's `transient` flag is true. By default that flag is true for:

- statuses 408, 500, 502, 503, 504 and 529 (Anthropic's "overloaded");
- status 429 when the response has `Retry-After`;
- connection errors, including a body cut off before its end, and timeouts.

It is false for TLS errors, redirects that are not followed, invalid requests and failing upload sources, since repeating them gives the same result, and `APIResponseError` is never retried.

Waits use random exponential backoff of up to 30 s, but never less than `Retry-After`. When the API asks to wait longer than `max_retry_after` (60 s by default), the error is raised at once instead. No retry starts after `retry_deadline` (15 minutes from the first attempt by default), and a wait that would end later is not started; an attempt in progress is limited by its own timeout.

The reasons behind these rules:

- **Retries are opt-in per method.** Most AI endpoints are `POST`s that are not idempotent. A request that timed out may still have run on the server, and sending it again can start a second billed generation. The author of a method knows whether that is acceptable; the library does not.
- **A 429 is retried only with `Retry-After`.** Without it there is no telling a short rate limit from an exhausted quota, and retrying a quota error only delays the failure. With it, the server has said when to come back.
- **`Retry-After` is capped.** A server that asks for ten minutes would otherwise block the caller silently. Raising the error lets the caller decide, for example to queue the work.
- **The deadline bounds the total time.** Four attempts with long waits and long timeouts could otherwise take far longer than any caller expects.
- **Jitter spreads the retries.** Many clients failing at the same moment don't all come back at the same moment.

### Error resolvers

Error responses are mapped by an error resolver. `BaseErrorResolver` reads:

- the common `{"error": {"code", "type", "message"}}` body shape, and a string `error` (`{"error": "model not found"}`, as Ollama sends) as the message;
- the request id from the `X-Request-Id` or `Request-Id` header, or from a `request_id` field of the body;
- `Retry-After`, in seconds or as an HTTP date.

A body that is not JSON gives an error without a code or message, so an HTML error page from a proxy still becomes the right `APIStatusError` for its status. Subclass the resolver to handle a provider's specifics, overriding `provider`, `get_error_code` or `classify`:

```python
from ai_api import APIStatusError, Provider, QuotaExceededError


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
    ) -> tuple[type[APIStatusError], bool]:
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

The resolver returns the error instead of raising it, so the session raises it where the traceback is most useful. Error events inside streams go through the same resolver, so one `classify` covers both.

Library errors can be pickled and copied, for example to pass them between processes. They are rebuilt from their attributes without calling `__init__` again, since `__init__` composes the message and would add the prefix twice.

## Sessions

### Session options

`AiohttpSession` takes these arguments; all after `error_resolver` are keyword-only:

| Argument | Default | Meaning |
|---|---|---|
| `api` | required | Builds request URLs (`IAPI`) |
| `auth` | required | Produces the authentication header (`IAuth`): `BearerAuth` for `Authorization: Bearer ...`, `APIKeyAuth` for a key in its own header (`x-api-key` by default) |
| `error_resolver` | required | Maps error responses to exceptions (`IErrorResolver`) |
| `headers` | `None` | Extra default headers (`Headers({...})`) |
| `json_loads` / `json_dumps` | `orjson` | JSON codec |
| `max_connections` | `100` | Maximum number of simultaneous connections |
| `timeout` | 600 s total, 30 s connect | Timeout of buffered requests |
| `stream_timeout` | no total limit, 30 s connect, 600 s between chunks | Timeout of streamed requests |
| `max_retry_after` | `60` | Longest `Retry-After` to wait for, in seconds |
| `retry_deadline` | `900` | Seconds after the first attempt after which no retry starts; `None` for no limit |
| `max_body_size` | `None` | Largest buffered response or error body, in bytes; streamed bodies are not limited |
| `trust_env` | `False` | Use proxy settings and `.netrc` from the environment |
| `follow_redirects` | `False` | Follow redirects within the origin of the request |
| `loop` | `None` | Event loop the session is bound to; `None` binds the loop of its first use |

The connection pool is created lazily, on the first request or on `async with`. It uses the `certifi` CA bundle, caches DNS answers for an hour and sends `User-Agent: Python/3.x ai_api/<version>`.

### Lifecycle

Close a session when you are done with it, ideally with `async with session:`. Closing releases the pooled connections and waits a quarter of a second so aiohttp can shut down the TLS connections cleanly; without it, Python warns about unclosed sessions at exit. A closed session opens a new pool if it is used again.

One session can serve many concurrent requests: `asyncio.gather` over `send_request` calls shares its pool, up to `max_connections` at a time.

### Authentication

The authentication header is produced on every request, by calling `auth.encode()`, so an `IAuth` that rotates its token takes effect at once:

```python
class RotatingAuth:
    """Sends the current token of a token store."""

    header_name = "Authorization"

    def __init__(self, token: str) -> None:
        self.token = token

    def encode(self) -> str:      # called for every request
        return f"Bearer {self.token}"
```

Any object with a `header_name` and an `encode()` method is an `IAuth`; it does not have to inherit from anything. `BearerAuth` and `APIKeyAuth` reject an empty token and whitespace, which would otherwise produce a broken or injected header, and keep the token out of `repr`, so it does not end up in logs or tracebacks.

### Default headers

`headers=Headers({...})` adds headers to every request, for example an API version header:

```text
AiohttpSession(api, auth, resolver, headers=Headers({"anthropic-version": "2023-06-01"}))
```

They are copied into the connection pool when it is created, so set them before the first request; changes made after that are not sent until the session is closed and opened again. The `User-Agent` set by the library takes precedence over one in `headers`, and the authentication header over both.

### JSON codec

`json_loads` and `json_dumps` replace the default `orjson` codec, for example to use the standard `json` module or to encode custom types. `json_dumps` may return `str` or `bytes`. NaN and infinity are rejected before encoding, because orjson would silently write them as `null`.

With a custom `json_loads`, response models are validated in Python mode instead of directly from JSON; see [Result type and response format](#result-type-and-response-format).

### Timeouts

Timeouts are `aiohttp.ClientTimeout` objects, set per session:

- **`timeout`**, for buffered requests: 600 s in total, 30 s to connect. It is longer than aiohttp's default of 5 minutes, because a long model generation that is not streamed exceeds that.
- **`stream_timeout`**, for streams: no total limit, 30 s to connect and 600 s of silence between two chunks. A stream may legitimately run for a long time, but a stream that sends nothing for ten minutes is dead.

A timeout raises `APITimeoutError`, which is transient. When the method is retryable, a buffered request that timed out is sent again, so keep `__retryable__` off for requests that must not run twice.

### Body size limits

`max_body_size` limits buffered response bodies and error bodies, in bytes; a larger body raises `APIResponseError` without being read to the end. It is off by default, since a legitimate response such as a file download can be large. Set it when the server is not trusted. Streamed bodies are not limited by it, because they are not held in memory; limit the lines and events of SSE and NDJSON with the stream parser's [limits](#events-sse-and-ndjson) instead.

### Redirects

Redirects are not followed by default: API providers do not redirect, and aiohttp strips only `Authorization` on a redirect to another host, so a key in `x-api-key` or in the default headers would leak there. A redirect raises `APIRedirectError`. With `follow_redirects=True`, redirects within the same origin (scheme, host and port) are followed; one to another origin still raises `APIRedirectError` before anything is sent there. A redirect that needs the request body again (307, 308) after an upload stream was consumed raises `APIRequestError`.

### Proxies and the environment

`trust_env` is off by default, so `HTTP(S)_PROXY` variables and `.netrc` credentials are ignored. A proxy picked up from the environment would silently route API keys through another host, and `.netrc` could add credentials nobody asked for. Turn it on with `trust_env=True` when you do want the environment's proxy.

### Event loop

A session is bound to one event loop: the one passed as `loop=`, or else the loop of its first request (or `async with`). It can be created anywhere, even at module level, but using or closing it from another loop (such as a second `asyncio.run`) raises `RuntimeError`. The `session.loop` property only reports the bound loop (`None` before first use) and never binds one. Close the session before its loop ends, ideally with `async with`.

aiohttp connections belong to the loop that created them, and using them from another loop fails with unclear errors such as `Event loop is closed` or a hang. Binding the session gives a clear error at the first wrong call instead.

## Describing an API

An `IAPI` says where the API lives and builds the URL of each request. `BaseAPI` covers the common case of a root URL and a version segment:

```text
BaseAPI("https://api.example.com", "v1").build_url("/models/m%2F1", {"limit": "5"})
# 'https://api.example.com/v1/models/m%2F1?limit=5'
```

- **`base_url`** must be an absolute HTTP(S) URL without a query or fragment. It may end with a path prefix, e.g. `https://gateway.example.com/openai`.
- **`version`** must be one path segment of unreserved characters (`v1`, `2024-01`), or empty for none. `.` and `..` are rejected.
- Both are checked when `BaseAPI` is created, because they are joined into URLs without encoding.

For a different URL scheme, write your own class with `base_url`, `version` and `build_url(method, query)`. The `method` argument is the method's request path, already percent-encoded: join it with `URL.joinpath(..., encoded=True)`, never with yarl's `/` operator, which encodes it a second time.

## Provider clients

`OpenAIClient` builds an `AiohttpSession` for OpenAI from a token: the `https://api.openai.com/v1` URL, `BearerAuth` and `OpenAIErrorResolver`, which turns a 429 with the code `insufficient_quota` into a `QuotaExceededError` that is not retried. The client owns its session, so close it, ideally with `async with`:

```python
from ai_api.providers.openai import OpenAIClient


async def list_models() -> ModelList:
    async with OpenAIClient("sk-...") as client:
        return await client.send_request(ListModels())
```

- **Custom session settings.** Pass a ready session with `OpenAIClient(session=...)` instead of a token. A `BaseSession` with the generic `BaseErrorResolver`, or with authentication outside the `Authorization` header, gets a `UserWarning`, since both are almost certainly mistakes for OpenAI. `session_factory(token, max_body_size=...)` from `ai_api.providers.openai.client` builds a session with OpenAI's settings and your own options; type checkers check those options against `AiohttpSession`.
- **Ownership.** A client closes its session when it is closed, also a session you passed in. One owner per session means no connection pool is left open by accident; create one session per client.
- **Other providers.** Subclass `BaseClient`, which delegates `send_request`, `stream`, `close` and `async with` to its session, and build the session with a `SessionFactory` (`client/factory.py`). `BaseClient` is generic in the session type: `BaseClient(AiohttpSession(...)).session` is typed as `AiohttpSession`, while `OpenAIClient`, which accepts any session, is a `BaseClient[ISession]`.

### Adding a provider

A provider is a flat package under `providers/<name>/`, built from the same parts as `providers/openai/`:

1. **`api.py`**: its `IAPI`, usually `BaseAPI("https://api.provider.com", "v1")`.
2. **`resolver.py`**: a `BaseErrorResolver` subclass with the provider's `provider` value, and `get_error_code` or `classify` for its error codes, such as which 429s are quota errors and which statuses are transient.
3. **`client.py`**: a module-level `SessionFactory(AiohttpSession, auth_factory=..., api=..., error_resolver=...)` and a `BaseClient` subclass that accepts a token or a ready session.
4. **`types/`, `methods/` and `resources/`** subpackages with its first endpoints: response models, method classes, and use cases that map them to the shared types.

The package `__init__.py` re-exports the client, which makes it the one `__init__` allowed to import a transport.

## Resources and use cases

Methods are specific to one provider. Resources and use cases are the layer above them that offers one API for all providers, in shared types:

- **Shared requests** (`ai_api.types.requests`): `BaseRequest`, which forbids unknown fields, and `GenerationRequest` (`model`, `input`, `temperature`, `max_output_tokens`, `metadata`).
- **Shared responses** (`ai_api.types.responses`): `GenerationResponse[RawT]` with `id`, `provider`, `model`, `text`, `usage` and the provider's own response in `raw`, so nothing the shared type lacks is lost.
- **Shared events**: `GenerationEvent[RawT]`, a discriminated union of `TextDeltaEvent`, `CompletedEvent`, `FailedEvent` and `OtherEvent`. `FailedEvent` is for a failure the provider reports as a regular event; protocol errors stay exceptions.
- **Use cases** (`ai_api.resources`): `BaseUseCase` turns a shared request into a provider method and its result into a shared response, `BaseStreamUseCase` does the same for streams, and `BaseCompositeUseCase` does both.
- **Resources**: `BaseResource` groups the use cases of one area, such as `client.responses.generate(...)`.

```python
from contextlib import AbstractAsyncContextManager

from ai_api import Provider, Stream
from ai_api.resources import BaseCompositeUseCase, BaseResource
from ai_api.response import map_stream
from ai_api.session import ISession
from ai_api.types.requests import GenerationRequest
from ai_api.types.responses import (
    GenerationEvent,
    GenerationResponse,
    OtherEvent,
    TextDeltaEvent,
)


def to_event(event: MessageEvent) -> GenerationEvent[MessageEvent]:
    if isinstance(event, TextDelta):
        return TextDeltaEvent[MessageEvent](text=event.text, raw=event)
    return OtherEvent[MessageEvent](raw=event)


class Complete(
    BaseCompositeUseCase[
        GenerationRequest,
        GenerationResponse[Completion],
        GenerationEvent[MessageEvent],
    ]
):
    async def __call__(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse[Completion]:
        completion = await self.session.send_request(
            CreateCompletion(
                model=request.model,
                messages=[Message(role="user", content=request.input)],
                temperature=request.temperature,
            )
        )
        return GenerationResponse[Completion](
            id=completion.id,
            provider=Provider.GENERIC,
            model=request.model,
            text=completion.text,
            raw=completion,
        )

    def stream(
        self,
        request: GenerationRequest,
    ) -> AbstractAsyncContextManager[Stream[GenerationEvent[MessageEvent]]]:
        method = StreamMessage(prompt=request.input)
        return map_stream(self.session.stream(method), to_event)


class Completions(BaseResource):
    def __init__(self, session: ISession) -> None:
        super().__init__(session)
        self.complete = Complete(session)
```

- **`map_stream`** opens the provider's stream and yields a `Stream` of converted items. On exit it closes the converted stream first and the source after it. A hand-written `async for ... yield` wrapper would leave its generator suspended until the event loop ends.
- **The session type is optional.** The last type parameter, `SessionT`, defaults to `ISession`. Give a concrete session, such as `BaseResource[AiohttpSession]`, to reach its settings through `self.session` without a cast.
- **`RawT` has no bound.** With a `BaseAPIType` bound, an unparametrized `GenerationResponse` would turn a dict in `raw` into an empty model, because unknown fields are ignored.

No provider implements these yet; they are the base for the provider-neutral API.

## Writing a transport

`BaseSession` holds everything that doesn't depend on the HTTP client: retries, error mapping, parsing and stream cleanup. A transport subclasses it and implements three methods:

- **`_execute(method) -> BufferedResponse`**: send the request and read the whole body.
- **`_open_stream(method) -> IStreamResponse`**: send the request and leave the body unread; the response has `status`, `headers` and `charset`, and offers `iter_chunks()`, `read()` and `aclose()`.
- **`close()`**: release the connections.

Both request methods return error statuses as responses, without raising, so the status check stays in `BaseSession`. I/O failures are raised as `APITransportError` subclasses with the right `transient` flag; that mapping is what makes retries behave the same on every transport. Build the URL with `api.build_url(method.request_path, method.get_query_map())` and the body from `method.get_body_map()` and `method.request_body_format`, and send the authentication header from `auth` on every request.

Transport code lives only in `<layer>/<transport>.py` modules (today `session/aiohttp.py`, `session/body/aiohttp.py`, `response/aiohttp.py` and `http/exceptions/aiohttp.py`), so a second transport, such as httpx, adds `*/httpx.py` files and changes nothing else. `bind_loop` in `session/loop.py` is transport-neutral and can be reused.

## Type safety

The library is checked with strict mypy and strict pyright, ships `py.typed`, and is designed so that type checkers catch mistakes in your code too:

- **Result types are inferred.** `await session.send_request(CreateCompletion(...))` is a `Completion`, and `async with session.stream(StreamMessage(...)) as events` gives `Stream[TextDelta | MessageStop]`.
- **Buffered and stream methods don't mix.** `send_request(stream_method)` and `stream(buffered_method)` are type errors.
- **`Stream.save` exists only on `Stream[bytes]`.** Calling it on a text or event stream is a type error.
- **Method constructors are typed.** With `validation_alias` and `serialization_alias`, both checkers accept the field name and reject unknown parameters.
- **Session options are checked through factories.** `session_factory(token, max_body_sizee=1)` is a type error, because the factory takes `AiohttpSession`'s own signature.

`tests/typing/` checks these promises: every line marked `# type: ignore[...]` there must stay a type error.

## Security

Defaults that protect the API key and the request:

- **Path injection.** Path parameters are percent-encoded, including `/`, and `.`, `..` and empty segments are rejected, so a parameter value cannot reach another endpoint. `base_url` and `version` are validated for the same reason.
- **Leaking keys on redirects.** Redirects are off; when enabled, they never leave the origin of the request.
- **Environment proxies.** Ignored unless `trust_env=True`.
- **TLS.** Certificates are verified against the `certifi` CA bundle, the same on every operating system. A TLS failure raises `APITLSError`, which is not retried.
- **Secrets in logs.** `BearerAuth` and `APIKeyAuth` keep the token out of `repr`, and reject whitespace that could inject another header.
- **Untrusted servers.** `max_body_size`, `max_line_size` and `max_event_size` limit what is held in memory, and an SSE stream with another `Content-Type` is rejected instead of being read as empty.
- **Invalid JSON values.** NaN and infinity are rejected rather than sent as `null`.

## Design decisions

The main choices, and the reasons for them. Details are in the sections above.

- **One class per API call.** A request is data that knows its endpoint and result type, as in aiogram's `TelegramMethod`. One generic `send_request` serves all endpoints, an endpoint is one small class, Pydantic validates parameters before anything is sent, and type checkers infer the result.
- **Strict requests, lenient responses.** A typo in a request should fail loudly; a field a provider adds to its responses should not break anyone.
- **Async only.** AI calls take seconds to minutes and are often streamed, so they are run concurrently. A synchronous wrapper can be built on top of an async core; the opposite works poorly.
- **Protocols between the parts.** The session depends on `IAPI`, `IAuth`, `IErrorResolver`, `IHeaders` and the parser protocols, not on concrete classes. Each part can be replaced without inheritance, and tests use simple fakes.
- **The transport is isolated.** All logic is in `BaseSession`, and aiohttp is imported only by `*/aiohttp.py` modules. Another HTTP client behaves the same, and `import ai_api` does not import aiohttp.
- **One flag decides retries.** Errors carry `transient`, set by the error resolver or the transport mapping. The retry loop knows no statuses, and a provider changes retry behavior by overriding `classify`.
- **Retries are opt-in.** Repeating a non-idempotent generation can cost money; only the author of a method can say it is safe.
- **Streams are not retried once started**, and not resumed: see [Interrupted streams](#interrupted-streams).
- **Errors by what came back.** Status, transport and response errors are separate branches, because they call for different handling: an API error, a network problem, a broken answer.
- **Clients own their sessions.** Exactly one object closes each connection pool.
- **Sessions are bound to one loop**, so a misuse fails with a clear error rather than aiohttp's.
- **Safe network defaults.** No redirects, no environment proxies, verified TLS; each can be turned on explicitly.
- **Files are not models.** `InputFile` is not a `BaseAPIType`, because dumping a model would read and copy the file; files are opened again for every attempt instead.
- **Layers in one direction.** Packages are ordered core → data and HTTP primitives → methods → response → session → client → resources → providers, and a module imports only from layers below it, which keeps the core free of providers and transports.

## Project layout

```text
src/ai_api/
├── enums.py            Provider, BodyFormat, ResponseFormat
├── exceptions.py       AIAPIError, the root of all library errors
├── types/              BaseAPIType, InputFile; requests/ and responses/
│                       hold the shared GenerationRequest/Response, events
├── serialization/      JSON codec, body value normalization
├── http/               IAPI and BaseAPI, BearerAuth, APIKeyAuth, headers
│   └── exceptions/     status, transport and response errors, resolver
├── methods/            BaseAPIMethod, BaseStreamMethod, path templates
├── response/           BufferedResponse, Stream, map_stream; parse/ parsers
├── session/            BaseSession, AiohttpSession, retries, body/ builders
├── client/             BaseClient, SessionFactory
├── resources/          resources and use cases over shared types
└── providers/
    └── openai/         OpenAIClient, OPENAI_API, OpenAIErrorResolver
```

The names most code needs are re-exported from `ai_api` (`from ai_api import BaseAPIMethod, RateLimitError`), and each layer package re-exports its own, e.g. `from ai_api.http import Query`. Transport modules such as `ai_api.session.aiohttp` and provider packages such as `ai_api.providers.openai` are imported on their own, so `import ai_api` does not import aiohttp.

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

The tests are in `tests/`: `unit/`, `session/` (the retry and error logic on a fake transport), `integration/` (aiohttp against a local server), `providers/`, `typing/` and `docs/`, which runs every Python example of this README. Coverage must stay at 98 % or more, for lines and branches. CI runs the same commands, with pytest on Ubuntu and Windows.

The code uses Google-style docstrings, which ruff enforces, and PEP 695 generics. Every module starts with `from __future__ import annotations`. pyright is configured in `pyproject.toml` too; the few rules it disables are explained there.

## Limitations and roadmap

- **Streams are not resumed.** SSE `id` and `retry` are parsed and available on `ServerSentEvent`, but a dropped stream is not reconnected with `Last-Event-ID`; the error reaches your code. Planned: an opt-in reconnection for servers that replay events. See [Interrupted streams](#interrupted-streams).
- **No per-request headers.** Methods cannot set headers, and default headers are fixed when the connection pool is created. Headers such as `Last-Event-ID`, an idempotency key or a beta flag for one request need a session of their own for now.
- **No per-request timeouts.** Timeouts are set per session, one for buffered requests and one for streams.
- **No raw binary request body.** A file can only be uploaded as a multipart part, not as the whole body.
- **No logging or hooks.** The library logs nothing and has no request or response hooks for metrics and tracing yet.
- **No synchronous API.**
- **Providers.** `OpenAIClient` only sets up the connection: there are no OpenAI method classes yet, and no Anthropic client.
- **Shared API.** Resources and use cases (`resources/`) and the shared request, response and event types (`types/requests/`, `types/responses/`) are the base of a provider-neutral API; no provider implements them yet.

## License

MIT. See [LICENSE](LICENSE).

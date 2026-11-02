# ai-api

An async-first toolkit for building typed HTTP API clients, aimed at AI providers.

<!-- --8<-- [start:intro] -->
You describe each API call as a small declarative class: a Pydantic model whose generic argument is its result type. The library sends it over `aiohttp` and returns a validated, statically typed result. It also handles retries, errors, file uploads and streaming, and ships clients for OpenAI and Anthropic with one shared text generation API.

> **Status: pre-alpha (0.6.0).** The core abstractions are in place, but the public API may still change: before 1.0 a breaking change bumps the minor version, see [Versioning](https://grehban.github.io/ai_api/guide/design/#versioning).

## Features

- **Typed methods.** `send_request(method)` returns the method's result type, and mypy and pyright both infer it.
- **Buffered and streamed responses.** JSON models, text, bytes or nothing; streams of bytes, text, or typed events from server-sent events (SSE) and newline-delimited JSON (NDJSON).
- **Checked streams.** A method can declare the protocol's last event, so a stream cut between two events raises instead of looking complete.
- **File uploads.** From bytes, a path, an open binary stream or an async iterator; a retry resends the whole file.
- **Opt-in retries.** Transient failures are retried with exponential backoff and jitter, honouring `Retry-After` up to a cap.
- **Structured errors.** One exception class per common HTTP status, plus transport and response errors, each with the provider's code, the request id and a `transient` flag.
- **Provider clients.** `OpenAIClient` (Responses API) and `AnthropicClient` (Messages API), both with the provider-neutral `client.generation`.
- **Replaceable parts.** Sessions depend on protocols, so the API, authentication, error mapping, headers and JSON codec can each be swapped.
- **Safe defaults.** Path parameters are escaped, redirects are off, proxies from the environment are ignored, and tokens stay out of `repr`.

## Installation

The library requires Python 3.13 or newer. It is not on PyPI yet, so install it from source:

```bash
git clone https://github.com/GrehBan/ai_api.git
cd ai_api
uv sync
```

Or with pip: `pip install -e .`

## Quick start

```python title="quick_start.py"
import asyncio

from pydantic import MISSING

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
    temperature: float | MISSING = MISSING   # left out until given


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
<!-- --8<-- [end:intro] -->

## Documentation

The guide and the API reference are at **[grehban.github.io/ai_api](https://grehban.github.io/ai_api/)**:

- [Methods](https://grehban.github.io/ai_api/guide/methods/): parameters, body, query and path, result types
- [Uploads](https://grehban.github.io/ai_api/guide/uploads/) and [streaming](https://grehban.github.io/ai_api/guide/streaming/)
- [Errors and retries](https://grehban.github.io/ai_api/guide/errors/) and [sessions](https://grehban.github.io/ai_api/guide/sessions/)
- [Provider clients](https://grehban.github.io/ai_api/guide/providers/) and [shared generation](https://grehban.github.io/ai_api/guide/generation/)
- [Reference](https://grehban.github.io/ai_api/reference/), generated from the code

## License

MIT. See [LICENSE](LICENSE).

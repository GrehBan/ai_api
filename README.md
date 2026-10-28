# ai-api

Pre-alpha release: 0.0.1

This project is a small, async-first API client foundation for building typed integrations against HTTP-based services. At this stage it is intentionally a low-level toolkit rather than a complete, production-ready SDK. The library is centered on `aiohttp`, typed request methods, Pydantic models, and provider-agnostic request/response plumbing.

## Project status

Current version: `0.0.1`

This is a pre-alpha state. The public surface is still evolving and may change without notice. The goal of the current release is to establish the core abstractions for request execution, response parsing, serialization, retries, and structured error handling.

## What is implemented

The current codebase includes the following building blocks:

- Async HTTP session layer built on `aiohttp`
- Request abstraction through typed method classes
- Query/body serialization helpers
- Support for JSON, form, multipart, and empty-body request payloads
- Response parsing for JSON, text, bytes, and blank responses
- Retry handling using `tenacity`
- Bearer token authentication helper
- Structured exception classes for HTTP and transport failures
- Header composition and request metadata support
- Pydantic-based response and API models

## Core design

The package is organized around a few strong abstractions:

- `BaseAPIMethod` defines the request contract for a concrete API call: path, method, query params, body format, retries, response format, and response model.
- `BaseSession` encapsulates the shared lifecycle of a session, request dispatch, retry behavior, and context manager support.
- `AiohttpSession` implements the concrete HTTP transport using `aiohttp`.
- `BufferedResponse` represents the raw HTTP response before parsing.
- `BaseAPIType` provides a common Pydantic base for typed API payloads.
- `BaseErrorResolver` centralizes HTTP status handling and provider-specific error mapping.

## Quick example

The library is designed to be extended by implementing your own API object and request method classes.

```python
from pydantic import Field

from ai_api.client.auth.bearer import BearerAuth
from ai_api.client.session.aiohttp import AiohttpSession
from ai_api.methods.base import BaseAPIMethod
from ai_api.methods.enums import ResponseFormat
from ai_api.types.base import BaseAPIType


class MyAPI:
    base_url = "https://api.example.com"
    version = "v1"

    def build_url(self, method: str, query=None) -> str:
        url = f"{self.base_url}/{self.version}{method}"
        if query:
            params = "&".join(f"{k}={v}" for k, v in query.items())
            url = f"{url}?{params}"
        return url


class PingResponse(BaseAPIType):
    ok: bool = Field(default=False)
    message: str | None = None


class PingMethod(BaseAPIMethod[PingResponse]):
    __request_path__ = "/ping"
    __request_method__ = "GET"
    __returning__ = PingResponse
    __response_format__ = ResponseFormat.JSON


async def main() -> None:
    api = MyAPI()
    session = AiohttpSession(
        api=api,
        auth=BearerAuth("demo-token"),
        error_resolver=MyErrorResolver(),
    )

    async with session:
        response = await session.send_request(PingMethod())
        print(response)
```

The example above shows the intended usage pattern: define a typed request model, send it through a session, and receive a validated Pydantic object.

## Installation

This project uses `pyproject.toml` and expects Python 3.15+.

```bash
pip install ai-api
```

Or install from source:

```bash
git clone https://github.com/your-org/ai-api.git
cd ai-api
pip install -e .
```

## Dependencies

The project currently relies on:

- `aiohttp` for HTTP transport
- `pydantic` for typed payload validation
- `orjson` for efficient JSON serialization
- `tenacity` for retry logic
- `yarl` for URL handling
- `multidict` for header storage

## Project structure

```text
src/ai_api/
├── client/
│   ├── api/
│   ├── auth/
│   ├── exceptions/
│   ├── headers/
│   ├── response/
│   ├── retry/
│   ├── serialization/
│   └── session/
├── methods/
├── providers/
├── types/
├── __init__.py
├── __meta__.py
├── exceptions.py
└── ...
```

## Current limitations

This release is not yet a complete third-party API SDK. The package currently focuses on infrastructure and abstractions, not provider-specific operations. In practical terms:

- there are no ready-made client calls for a specific AI provider yet
- method naming and API ergonomics may evolve
- the library is still missing broader documentation and examples beyond the core building blocks
- compatibility guarantees are not yet in place for public interfaces

## Roadmap

The next likely milestones are:

- finalize the public request/method API
- add provider-specific client implementations
- improve example usage and documentation
- add tests for retry, serialization, and validation paths
- stabilize error and response conventions across integrations

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

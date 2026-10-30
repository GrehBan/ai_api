# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-30

This release reorganizes the package into layers and defines the public API.
Almost every import path changes; see [Migrating from 0.0.4](#migrating-from-004).

### Added

- Public re-exports. Every package `__init__.py` has a sorted `__all__`, and
  `ai_api` re-exports the names most code needs: method and type base classes,
  files, enums, auth, `BaseAPI`, `Headers`, `Stream`, `ServerSentEvent`,
  `BaseSession`, `BaseClient`, `BaseErrorResolver` and the whole error tree.
  `import ai_api` does not import aiohttp.
- `ai_api.http` package for the HTTP primitives: `BaseAPI`, `BearerAuth`,
  `APIKeyAuth`, `Headers`, the `IAPI`/`IAuth`/`IHeaders` protocols and `Query`.
- Header and media type constants in `ai_api.http.headers`: `RETRY_AFTER`,
  `REQUEST_ID`, `X_REQUEST_ID`, `JSON_CONTENT_TYPE`, `EVENT_STREAM_CONTENT_TYPE`.
- `Provider.GENERIC`, the provider of `BaseErrorResolver`.
- This changelog.
- README sections on the request flow, request body, response parsing,
  interrupted streams, sessions, adding a provider, resources, writing a
  transport, type safety, security and design decisions.

### Changed

- Python 3.13 or newer is required (was 3.15). Every module uses
  `from __future__ import annotations`.
- `APIError` is renamed to `APIStatusError`.
- A response body cut off by a lost connection raises the transient
  `APIConnectionError` instead of `APIResponseError`, so buffered requests are
  retried. A stream cut off mid-way still reaches the caller.
- `APIResponseError` moved to `ai_api.http.exceptions.response`.
- `AiohttpSession(limit=...)` is renamed to `max_connections=...`, and
  `create_session()` to `get_client()`.
- Session arguments after `error_resolver` are keyword-only, in `BaseSession`
  and `AiohttpSession`.
- `__path_safe__` is renamed to `__request_path_safe__`.
- `__request_query__` defaults to an empty set and is stored as a `frozenset`.
- The method property `body_format` is renamed to `request_body_format`.
- `StreamResponseParser` options are keyword-only:
  `StreamResponseParser(*, max_line_size, max_event_size, require_event_stream)`.
  It no longer takes `json_loads` or `error_resolver`.
- `BufferedResponseParser` takes no arguments.
- `IResponseParser.parse(method, response, json_loads)` and
  `IStreamParser.parse(method, response, json_loads, error_resolver)` take all
  their arguments as required. The response argument is named `response`, not
  `buffered`.
- `Stream(events)` is renamed to `Stream(items)`.
- `SSEDecoder.feed()` is renamed to `decode()`, and `SSE_DONE` moved to
  `ai_api.response.parse.sse`.
- `get_header` and `parse_retry_after` moved from the error resolver to
  `ai_api.http.headers`.
- `ResponseEvent` is renamed to `GenerationEvent`.
- `Provider.NONE` is renamed to `Provider.GENERIC`.
- `check_session_creds` is renamed to `require_token_or_session`, and the
  OpenAI `validate_session` to `warn_about_session`.
- `BodyFormat` and `ResponseFormat` moved from `ai_api.methods.enums` to
  `ai_api.enums`.

### Removed

- `ResponseMode` and `__response_mode__`: buffered and stream methods are told
  apart by their base class.
- `RETRY_CONDITION`; use `retry_condition(max_retry_after)`.
- The `ai_api.client.*` sub-packages; their modules moved to the layers below.

### Migrating from 0.0.4

Most code only needs `from ai_api import ...`. Old module paths and their
new locations:

| 0.0.4 | 0.1.0 |
|---|---|
| `ai_api.client.api.base` | `ai_api.http.api` |
| `ai_api.client.api.protocols`, `client.auth.protocols`, `client.headers.protocols` | `ai_api.http.protocols` |
| `ai_api.client.api.url.parse` | `ai_api.methods.path` |
| `ai_api.client.auth.bearer`, `client.auth.api_key` | `ai_api.http.auth` |
| `ai_api.client.headers.headers`, `client.headers.names` | `ai_api.http.headers` |
| `ai_api.client.exceptions.http` | `ai_api.http.exceptions.status` |
| `ai_api.client.exceptions.transport` | `ai_api.http.exceptions.transport` (`APIResponseError`: `ai_api.http.exceptions.response`) |
| `ai_api.client.exceptions.aiohttp` | `ai_api.http.exceptions.aiohttp` |
| `ai_api.client.exceptions.error_resolver.base` | `ai_api.http.exceptions.resolver` |
| `ai_api.client.exceptions.error_resolver.protocols` | `ai_api.http.exceptions.protocols` |
| `ai_api.methods.enums` | `ai_api.enums` |
| `ai_api.methods.resolve` | `ai_api.methods.fields` |
| `ai_api.client.response.base` | `ai_api.response.buffered` |
| `ai_api.client.response.{aiohttp,protocols,stream}` | `ai_api.response.{aiohttp,protocols,stream}` |
| `ai_api.client.response.parse.*` | `ai_api.response.parse.*` |
| `ai_api.client.serialization.json` | `ai_api.serialization.json` |
| `ai_api.client.serialization.body` | `ai_api.serialization.normalize` |
| `ai_api.client.session.{base,aiohttp,loop,protocols}` | `ai_api.session.{base,aiohttp,loop,protocols}` |
| `ai_api.client.session.builders.*` | `ai_api.session.body.*` |
| `ai_api.client.retry.startegy` | `ai_api.session.retry` |
| `ai_api.client.factory.factory` | `ai_api.client.factory` |
| `ai_api.client.resources.base` | `ai_api.resources.base` |
| `ai_api.client.resources.usecases.base` | `ai_api.resources.usecases` |
| `ai_api.client.resources.usecases.protocols` | `ai_api.resources.protocols` |
| `ai_api.providers.openai.client.openai` | `ai_api.providers.openai.client` |
| `ai_api.providers.openai.client.api.openai` | `ai_api.providers.openai.api` |
| `ai_api.providers.openai.client.exceptions.error_resolver.openai` | `ai_api.providers.openai.resolver` |

[0.1.0]: https://github.com/GrehBan/ai_api/compare/2b58dfc...v0.1.0

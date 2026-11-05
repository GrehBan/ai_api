# Design

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
- **TLS.** Certificates are verified against the `certifi` CA bundle, the same on every operating system. A certificate that fails verification raises `APITLSError`, which is not retried; a handshake that breaks off is an `APIConnectionError`, retried like other connection failures.
- **Secrets in logs.** `BearerAuth` and `APIKeyAuth` keep the token out of `repr`, and reject whitespace that could inject another header. The library's own logs carry no token, header, body, query string, path parameter value, error message or note, or file name, at any level (see [Logging](sessions.md#logging)).
- **Untrusted servers.** `max_body_size` (off by default), `max_line_size` and `max_event_size` (on by default) limit what is held in memory, and an SSE stream with another `Content-Type` is rejected instead of being read as empty.
- **Invalid values.** NaN and infinity are rejected in every part of the request, rather than sent as `null`, `nan` or `None`.

## Design decisions

The main choices, and the reasons for them. Details are in the sections above.

- **One class per API call.** A request is data that knows its endpoint and result type, as in aiogram's `TelegramMethod`. One generic `send_request` serves all endpoints, an endpoint is one small class, Pydantic validates parameters before anything is sent, and type checkers infer the result.
- **Strict requests, lenient responses.** A typo in a request should fail loudly; a field a provider adds to its responses should not break anyone.
- **Async only.** AI calls take seconds to minutes and are often streamed, so they are run concurrently. A synchronous wrapper can be built on top of an async core; the opposite works poorly.
- **Protocols between the parts.** The session depends on `IAPI`, `IAuth`, `IErrorResolver`, `IHeaders` and the parser protocols, not on concrete classes. Each part can be replaced without inheritance, and tests use simple fakes.
- **The transport is isolated.** All logic is in `BaseSession`, and aiohttp is imported only by `*/aiohttp.py` modules. Another HTTP client behaves the same, and `import aiollm` does not import aiohttp.
- **One flag decides retries.** Errors carry `transient`, set by the error resolver or the transport mapping. The retry loop knows no statuses, and a provider changes retry behavior by overriding the resolver's `classify_status` (the base `classify` keeps the rule that an in-band failure is never transient).
- **Retries are opt-in.** Repeating a non-idempotent generation can cost money; only the author of a method can say it is safe.
- **Streams are not retried once started**, and not resumed: see [Interrupted streams](streaming.md#interrupted-streams).
- **Errors by what came back.** Status, transport and response errors are separate branches, because they call for different handling: an API error, a network problem, a broken answer.
- **Clients own their sessions.** Exactly one object closes each connection pool.
- **Sessions are bound to one loop**, so a misuse fails with a clear error rather than aiohttp's.
- **Safe network defaults.** No redirects, no environment proxies, verified TLS; each can be turned on explicitly.
- **Files are not models.** `InputFile` is not a `BaseAPIType`, because dumping a model would read and copy the file; files are opened again for every attempt instead.
- **Layers in one direction.** Packages are ordered core → data and HTTP primitives → methods → response → session → client → adapters → resources → providers, and a module imports only from layers below it, which keeps the core free of providers and transports.

## Versioning

The version follows [Semantic Versioning](https://semver.org/). Before 1.0 the public API is not stable yet, and the rule is:

- a **breaking** change (a public name removed or renamed, a signature, a default, an error type, a wire format or a behaviour that code may rely on) bumps the **minor** version: 0.5.x → 0.6.0;
- anything else (a fix, an addition, documentation) bumps the **patch**: 0.5.0 → 0.5.1.

Pin the minor version (`aiollm ~= 0.5.0`) to get fixes without breaking changes. Every breaking change is marked **Breaking** in the [changelog](../changelog.md), with how to migrate. From 1.0, breaking changes bump the major version and are announced by a deprecation warning one release before.

A version is not a decimal and not a countdown: after 0.9 comes 0.10, then 0.11, so breaking changes do not bring 1.0 closer. 1.0 is a promise that the public API stays as it is, not a mark of a finished library, and it is released when these hold:

- two releases in a row without a breaking change;
- the core (methods, sessions, HTTP primitives, errors) and the shared generation layer are both declared stable;
- the library has users outside the project, and their feedback is in.

Before it, `1.0.0a1`, `1.0.0b1` and `1.0.0rc1` pre-releases may try the final API out; pip installs them only when asked with `--pre`. Features added after 1.0 bump the minor version (1.1, 1.2).

The public API is what `aiollm` and the `__all__` of its packages export. Everything else (modules and helpers whose names start with `_`, the internal helper modules such as `aiollm.methods.fields`, log messages) may change in any release. The provider types follow the providers' own schemas, so a field can appear or change when the provider changes it.

## Limitations and roadmap

- **Streams are not resumed.** SSE `id` and `retry` are parsed and available on `ServerSentEvent` (both are the last value the stream set, as the HTML standard applies them), but a dropped stream is not reconnected on its own; the error reaches your code, which can open it again with `headers={"Last-Event-ID": ...}`. Planned: an opt-in reconnection for servers that replay events. See [Interrupted streams](streaming.md#interrupted-streams).
- **No raw binary request body.** A file can only be uploaded as a multipart part, not as the whole body.
- **No hooks.** The library logs (see [Logging](sessions.md#logging)), but has no request or response hooks for metrics and tracing yet.
- **No synchronous API.**
- **Providers.** OpenAI has only the Responses API and Anthropic only the Messages API, each with text in and text out. Request parameters the methods do not list, such as `tools` or `thinking`, cannot be sent (methods forbid unknown fields); images and other content parts go in the dict forms of `input`, `content` and `system`. Response fields the library does not model are kept in `raw`.
- **Shared API.** Resources and use cases (`resources/`) and the shared request, response and event types (`types/requests/`, `types/responses/`) are the base of a provider-neutral API; `OpenAIClient.generation` and `AnthropicClient.generation` implement them.

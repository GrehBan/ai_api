# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `IValueFormatter` and `BaseValueFormatter` (`ai_api`,
  `ai_api.serialization`): a method's `__value_formatter__` (property
  `value_formatter`, also on `IBaseMethod`) writes dates, times and
  durations in every part of the request. The default is ISO 8601; an API
  that wants another format overrides it on its base method, and one field
  can use Pydantic's `PlainSerializer`.
- `BaseProviderClient` (`ai_api`, `ai_api.client`): what a provider
  client shares, from its `provider`, `session_factory`, `api_key_env`,
  `auth_headers` and `adapter_types` (every adapter class it builds, one
  per shared resource); `OpenAIClient` and `AnthropicClient` are built on
  it. The client, its resolver and every adapter must name one provider
  (`check_same_provider`), checked when the class is defined;
  `client.adapter(AdapterClass, ...)` builds an adapter, typed as a call
  of the class, and refuses a class not in `adapter_types`.
- `SessionMismatchWarning` (`ai_api`), a `UserWarning`: a session given
  to a provider client whose error resolver is not the provider's (a
  subclass of it is fine; another provider's resolver was not warned
  about before) or whose key goes in a header the provider does not take.
  Turn it off with the standard warning filters.
- `ISessionFactory`, with `error_resolver`, and `SessionFactory` are
  exported from `ai_api.session`; **breaking:** `SessionFactory` moved
  from `ai_api.client.factory` to `ai_api.session.factory`.
- `RETRY_ATTEMPT_LIMIT` (`ai_api.session.retry`): the 4 attempts of a
  retried request, as a constant.
- **Breaking:** matchers compare values as JSON tells them apart
  (`FieldValues("done", True)` no longer matches `1`), read a model's
  fields but not its properties (a property that raised broke the
  stream), and `EventTypes("x")` equals `FieldValues("type", "x")`;
  `EventTypes.types` is a view of `values`. A single matcher in `AnyOf`
  shows as `AnyOf(x)`, and a protocol-only matcher may stand on the left
  of `|`.
- **Breaking:** a `ServerSentEvent` item whose data is JSON goes through
  `error_events`, so an error in it raises instead of being yielded, and
  `ServerSentEvent.retry` keeps the last value the stream set, as `id`
  does (the HTML standard applies both at once).
- **Breaking:** `GenerationRequest` refuses an empty `model` or text
  `input` before anything is sent; `BaseAPI` refuses credentials in
  `base_url`; a payload model (`BaseAPIType`) with a field of a type
  Pydantic cannot validate fails at class definition (only the `raw` of
  shared results takes any type).
- Raw binary streams nested in a model inside a method field are wrapped
  as well (the model is copied); a stream in a container that cannot be
  rebuilt, such as a `dict` subclass, raises `TypeError` instead of being
  dumped lazily. `Annotated` members of a union are unwrapped by the
  path and query checks, so `Annotated[dict, ...] | None` is refused as a
  query field.
- A path without placeholders unescapes `{{`/`}}` as one with them does;
  `APIKeyAuth` defaults to the `X_API_KEY` constant; an `InputFile`
  field explains that a `str` is not taken, and a path-like that `Path()`
  rejects is a validation error.
- `typing-extensions`, `aresponses` and `isort` are no longer
  dependencies.
- Tests: the local `Recorder` fails a test on any request it was not
  scripted for, mirrored provider client tests run once per provider
  (`tests/providers/test_provider_clients.py`), the backoff is checked by
  properties of the real jitter, and documented responses are compared
  field by field with their JSON. The retry loop awaits tenacity's call
  directly, without an unreachable guard.
- `BaseErrorResolver.error_statuses` and `classify_status`: a provider
  sets its table of in-band codes and types and overrides the
  classification of a status; `classify` keeps an in-band failure
  non-transient for every resolver.
- `ContentTooLargeError` for HTTP 413 (Anthropic's `request_too_large`
  was a plain `APIStatusError`).
- `CompletedEvent.provider`, as on `GenerationResponse`; every adapter has
  a `provider` (`PROVIDER` of its package, `Provider.GENERIC` outside the
  library).
- `Stream.items_received`: how many items a stream has given.
  `transform_stream(..., stream_type=...)` yields a `Stream` subclass built
  on the transformed items, which `GenerationResource` uses for its
  `GenerationStream`.

- Conversions: `ai_api.adapters.IConversion[SourceT, TargetT]` (a protocol
  of `__call__(source, /)`, so a plain function fits) and its ABC
  `BaseConversion`, for pure value conversions between a provider's types
  and the shared ones; `ai_api.adapters.Stop` holds what a stop conversion
  returns (`incomplete_reason`, `raw_incomplete_reason`, `refusal`).
- `OpenAIGenerationAdapter` and `AnthropicGenerationAdapter` take
  `input_conversion=`, `usage_conversion=` and `stop_conversion=` keyword
  arguments (defaults: each provider's `InputConversion`,
  `UsageConversion` and `StopConversion`), so one can be swapped without
  subclassing the adapter. A response and a stream go through the same
  conversions.
- `tests/docs/test_unification.py` checks that every field of
  `docs/unification.md` is filled by a conversion that every adapter has,
  or is listed as taken as is.
- `generation.create(...)` and `generation.stream(...)` take the fields of
  `GenerationRequest` as keyword arguments, as the providers' SDKs do
  (`create(model="...", input="Hi")`), or a ready `GenerationRequest`;
  passing both raises `TypeError`. `IGenerationResource` has the same
  overloads.
- `GenerationStream` (`ai_api.resources`): `generation.stream(...)` yields
  it. Besides iterating every event, `events.text()` gives the text deltas
  and `await events.final()` reads to the end and returns the
  `CompletedEvent`; whatever read the stream, the `CompletedEvent` is kept.
- `CompletedEvent` carries the response's `id`, `model` (required) and whole
  `text`, from OpenAI's terminal event and from Anthropic's `message_start`
  and text deltas.
- `IGenerationClient[RawT = Any, RawItemT = Any]` (`ai_api.resources`): a
  client with a read-only `generation`; `OpenAIClient` and
  `AnthropicClient` match it.
- `OpenAIClient(token, **session_kwargs)` and
  `AnthropicClient(token, **session_kwargs)` pass every keyword argument of
  `AiohttpSession` to the session built from the token, typed by
  `ai_api.session.aiohttp.AiohttpSessionOptions`; with `session=` they raise
  `ValueError`.
- `BaseAPIType.__upstream_schema__`: the name of the upstream schema
  definition a provider model mirrors (several names, or `()` for none).
  `tests/providers/test_wire_contracts.py` requires it on every provider
  model and checks each modeled field against that definition: it exists
  upstream, accepts `null` where upstream does and is optional where
  upstream is.
- The root package re-exports the shared generation layer:
  `GenerationRequest`, `Turn`, `GenerationResponse`, `TokenUsage`,
  `TextDeltaEvent`, `CompletedEvent`, `OtherEvent`, `GenerationEvent`,
  `GenerationStream`, `IGenerationResource` and `IGenerationClient`.

### Changed

- **Breaking:** a concrete adapter must set `provider`, and
  `CompletedEvent` requires it.
- **Breaking:** OpenAI's `ERROR_EVENTS` is `RESPONSE_ERROR_EVENTS`, so it
  is not mistaken for the parser's default of the same name; the
  module-level `warn_about_session` functions are now the classmethod
  `BaseProviderClient.warn_about_session`, whose second warning names the
  headers the provider takes.
- Leaving a client's `async with` calls the session's `__aexit__`, as
  entering calls its `__aenter__`, instead of `close()`.
- **Breaking:** 529 is no longer transient in `BaseErrorResolver`: it is
  Anthropic's own status, and `AnthropicErrorResolver` keeps classifying it
  as a transient `ServiceUnavailableError`.
- **Breaking:** only a certificate that fails verification
  (`ClientConnectorCertificateError`, `ServerFingerprintMismatch`) raises
  `APITLSError`; another SSL error, such as a handshake that breaks off, is
  a transient `APIConnectionError`, retried as the official SDKs do.
- **Breaking:** `Retry-After` is read as RFC 9110 defines it, whole seconds
  in ASCII digits or an HTTP date. `1.5`, `-5` (which meant an immediate
  retry) or `1e3` count as no header.
- **Breaking:** `APIStatusError.provider` is a `Provider`,
  `Provider.GENERIC` by default (it was typed `str | None` while the
  resolvers passed the enum).
- The session no longer turns a parser's `ValueError` into
  `APIResponseError`: parsers raise `APIResponseError` themselves, so a
  custom decoder's failure is `Failed to decode JSON response` and other
  parser errors surface as they are. Invalid JSON is `Failed to decode`,
  a body that does not match the model `Failed to validate`, in buffered
  and streamed responses alike.
- `RequestValidationError` lists every failing field of the provider's
  method under its shared name (`max_output_tokens`, not `max_tokens`);
  `field` stays the first one.
- When a stream use case's source method declares `__stream_terminal__`
  and the adapter gives no final event, the error is a `RuntimeError`
  naming the adapter instead of `APIStreamInterruptedError`: the parser
  proved the stream complete, so it was not cut.
- `APIRedirectError`'s default message is `Redirect was not followed.`,
  since it is also raised when redirects are off or lead to another origin.
- **Breaking:** a datetime in a JSON or form body is sent in ISO 8601
  (`"2026-01-02T03:04:05Z"`) instead of a Unix timestamp string, and a
  timedelta as an ISO 8601 duration (`"PT30S"`) instead of the timestamp of
  now plus the delta. Query and path values were already ISO 8601; a date
  or time in the body goes through the same formatter. To keep Unix
  seconds, set a `BaseValueFormatter` subclass as `__value_formatter__`.
- **Breaking:** `IBodyBuilder.build` and `BodyNormalizer.normalize` /
  `normalize_mapping` take the method's formatter; `AiohttpSession` passes
  `method.value_formatter`.
- Query, path and form values follow one rule (`scalar_text`): enums as
  their values, booleans as `true`/`false`, dates through the formatter.
  A naive datetime now raises `ValueError` in a query or path too, and a
  value with no text form (such as a class) raises `TypeError`.
- `BodyNormalizer.MAX_DEPTH` is 253, orjson's limit for a body field: a
  field nested 254 to 256 levels deep raised the encoder's `TypeError`
  instead of the documented `ValueError`.
- A method without a body (`BodyFormat.NONE`, the default) whose field is
  neither a path nor a query parameter raises `TypeError` at class
  definition, instead of dropping the field from every request; the aiohttp
  body builder refuses a non-empty body for `NONE` with `ValueError`.
- OpenAI's `CreateResponse` and `StreamResponse` and Anthropic's
  `CreateMessage` and `StreamMessage` set `__retryable__ = True`: transient
  failures are retried (a stream only while it opens), as in the official
  SDKs. Before, `client.generation` never retried.
- `OpenAIErrorResolver` and `AnthropicErrorResolver`: a failure reported
  after a 200 (OpenAI's `failed` response, `response.failed` or `error`
  stream event, Anthropic's `error` stream event) is never transient; its
  class is still picked by its code or type.
- Values the request cannot express raise instead of being dropped or
  mangled: `None` in a query list (`ValueError`), `None` in a form list
  (`ValueError`) and a list inside a form list (`TypeError`, it was
  flattened); a path parameter annotated as a list, set, dict or model
  raises `TypeError` at class definition, and such a value in an `Any`
  field when the request is built (it was sent as its Python `repr`).
- Checks moved to class definition: a buffered method whose generic
  argument is not a class (`list[Model]`, a union) and a `BYTES`/`TEXT`
  stream with `__stream_terminal__` raise `TypeError` unless the method
  sets its own parser. Before, the first failed after the request was sent
  and the second was ignored. A `TEXT` stream checks its charset when it
  opens, like the `Content-Type` of SSE, instead of on the first item.
- `IOFile` over a non-blocking stream with no data ready raises
  `BlockingIOError` (sent as `APIRequestError`) instead of sending a
  truncated file.
- `AiohttpSession` builds the session's headers for every request instead
  of copying them into the connection pool once, so a header added after
  the first request is sent.
- `generation.stream(...)` yields a `GenerationStream`, a `Stream` subclass,
  instead of a plain `Stream`.
- `CompletedEvent.model` is required: an adapter that builds the event
  passes the model.

- An Anthropic stream builds a real `Usage` (from `message_start` and the
  cumulative `message_delta` counts) and `MessageDelta` for the final
  `CompletedEvent`. A stream without `message_start`, or with counts before
  it, now raises `APIResponseError` instead of reporting partial counts.

### Fixed

- An OpenAI quota error reported inside a stream (`insufficient_quota`
  and the other quota codes) is a `QuotaExceededError`, as over HTTP 429,
  instead of a plain `APIStatusError`.
- `OpenAIErrorResolver` recognizes a failed `Response` by its status
  alone, as the parser does, so one without `object` keeps its error.
- After a failed read, `GenerationStream.final()` raises the original
  error again instead of `APIStreamInterruptedError: ended without a
  CompletedEvent`; later reads and `text()` do the same instead of ending
  quietly.
- NaN and infinity raise `ValueError` in every part of the request: a
  `float` path parameter was sent as `nan`, a NaN in an `Any` query field
  as the string `None`, and a form field as `nan`.
- A nested model passed to the normalizer is dumped by alias, so its wire
  names are kept, and enum dict keys are sent as their values (orjson
  rejected them). Keys that give the same value raise `ValueError`.
- A text stream in a body raises a clear `TypeError` in the normalizer
  instead of a message about binary data from the body builder.
- `OutputText.logprobs` accepts `null`, which OpenAI sends when log
  probabilities were not requested; such a response failed validation.

### Removed

- The internal conversion functions they replace: OpenAI's `to_usage`,
  `refusal`, `incomplete_reason`, `raw_incomplete_reason` and `to_event`,
  Anthropic's `to_usage`, `message_usage` and `to_stop` (its `Stop` moved
  to `ai_api.adapters`).

## [0.2.0] - 2026-10-31

### Added

- Method headers: `__request_headers__` (a mapping or `Headers`, `None` by
  default) is sent with every request of the method; it is exposed as the
  `request_headers` property, a new `IHeaders` collection on every access.
- Per-call options: `send_request` and `stream` on `ISession`, `BaseSession`,
  `IClient` and `BaseClient` take keyword-only `headers=` and `timeout=`.
  Both apply to every attempt of the call, retries included.
  - Headers are merged in the order session < method < call < authentication;
    a call cannot replace the authentication header.
  - `timeout` is in seconds. For a buffered request it replaces the total of
    the session's `timeout`; for a stream, the silence between chunks
    (`sock_read`) of `stream_timeout`. Other limits are kept, and a stream
    still has no total limit.
- `ai_api.http.HeadersType` (a mapping or an `IHeaders`) and
  `ai_api.session.Timeout`.
- `Headers`, `Headers.add` and `Headers.build` accept another `IHeaders`
  collection as well as a mapping. `IHeaders` is `runtime_checkable`.
- `ai_api.adapters`: adapters convert between shared types and a provider's
  methods and nothing else. `IAdapter` / `BaseAdapter` have `to_method(request)`
  and `to_response(raw, request)`; `IStreamAdapter` / `BaseStreamAdapter` have
  `to_stream_method(request)` and `to_events(items, request)`, which sees the
  whole stream; `ICompositeAdapter` / `BaseCompositeAdapter` have all four.
- `AdapterUseCase`, `AdapterStreamUseCase` and `AdapterCompositeUseCase` in
  `ai_api.resources`: use cases that only orchestrate a session and an
  adapter. The abstract `BaseUseCase` classes stay for calls that are not one
  request and one response.
- `ai_api.response.transform_stream(source, transform)`: opens a stream and
  passes its items through a function of the whole stream, closing the
  transformed stream before the source.
- `GenerationResource(session, adapter)` in `ai_api.resources`: the shared
  resource of text generation, one class for every provider, with
  `create(request)` and `stream(request)`; `IGenerationResource` is its
  protocol.
- `GenerationRequest.instructions`: instructions given apart from the input,
  such as a system prompt.
- `OpenAIClient()` without a token or a session reads the key from
  `OPENAI_API_KEY`; an explicit token wins over it and an empty variable
  counts as unset.

- OpenAI Responses API: `CreateResponse` and `StreamResponse` in
  `ai_api.providers.openai.methods` (`POST /v1/responses`, with OpenAI's
  limits on `max_output_tokens`, `temperature`, `top_p` and metadata checked
  before sending), and `Response`, its output items, usage and the stream
  events in `ai_api.providers.openai.types`. `Response.output_text` joins the
  text of the output messages. Items, content parts and events of a type the
  library does not model yet are kept whole as `UnknownOutputItem`,
  `UnknownContent` and `UnknownEvent`.
- `OpenAIErrorResolver` reads the flat `error` event of a response stream,
  so its code and message reach the raised `APIStatusError`, and picks its
  class by its code or type (`server_is_overloaded` → transient
  `ServiceUnavailableError`, `server_error` → `InternalServerError`, ...),
  since it arrives after the stream's 200.
- `OpenAIErrorResolver` treats every documented exhausted-limit 429
  (`credit_balance_exhausted`, organization and project spend limits, the
  usage limit, and the `insufficient_quota` type) as a non-transient
  `QuotaExceededError`, not only the `insufficient_quota` code.
- Cross-provider contract tests (`tests/providers/test_generation_contract.py`):
  the same scenarios for `OpenAIClient.generation` and
  `AnthropicClient.generation`.
- `OpenAIGenerationAdapter` in `ai_api.providers.openai.adapters`: shared
  generation over the Responses API, for `GenerationResource`; OpenAI's
  limits on the request raise `RequestValidationError`. A buffered
  response that did not finish (`cancelled`, `in_progress`, ...) raises
  `APIResponseError`; a failed one is raised by `CreateResponse` and
  `StreamResponse` themselves (see below).
- `incomplete_reason` on `GenerationResponse` and `CompletedEvent`: why the
  provider stopped early, an `IncompleteReason` (`MAX_TOKENS`, `REFUSAL` or
  `OTHER`); `None` if it finished. The provider's own reason is in
  `raw_incomplete_reason`.
- Anthropic provider (`ai_api.providers.anthropic`): `AnthropicClient` (a
  token, a ready session or `ANTHROPIC_API_KEY`; `BearerAuth`), `ANTHROPIC_API`,
  `API_VERSION` (`2026-10-21`) and `AnthropicErrorResolver`: spend caps and
  spend limits are non-transient `QuotaExceededError`s, 529
  `overloaded_error` is a transient `ServiceUnavailableError`, and a stream's
  `error` event is classified by its type.
- `X_API_KEY` and `ANTHROPIC_VERSION` header names in `ai_api.http.headers`.
- Anthropic Messages API: `CreateMessage` and `StreamMessage` in
  `ai_api.providers.anthropic.methods` (`POST /v1/messages`; both send
  `anthropic-version: 2026-10-21` through `VERSION_HEADERS`; `max_tokens` is
  required, turns are `MessageParam` `TypedDict`s and `metadata` takes only
  `user_id`, all checked before sending), and `Message` with its blocks, usage
  and the stream events in `ai_api.providers.anthropic.types`. `Message.text`
  joins the text blocks; unknown blocks, deltas and events are kept whole as
  `UnknownBlock`, `UnknownDelta` and `UnknownEvent`.
- `AnthropicGenerationAdapter` in `ai_api.providers.anthropic.adapters` and
  `AnthropicClient.generation` (with `default_max_tokens=1024` for requests
  without `max_output_tokens`): shared generation over the Messages API.
  Metadata other than `user_id` raises `RequestValidationError`; a
  `stop_reason` other than `end_turn` or `stop_sequence` sets
  `incomplete_reason`; the stream's
  `CompletedEvent` comes on `message_stop` with the counts of `message_start`
  and the cumulative `message_delta`; a message without `stop_reason` raises
  `APIResponseError`.
- `BaseUnknown`, `by_type` and `UNKNOWN` in `ai_api.types` (moved from the
  OpenAI types, now shared by both providers): a pydantic discriminator that
  keeps a value of an unknown `type` whole instead of rejecting it.
- `OpenAIClient.generation`: a `GenerationResource` with
  `OpenAIGenerationAdapter`, so `await client.generation.create(request)` and
  `client.generation.stream(request)` work with shared types.
- Wire-contract tests (`tests/providers/test_wire_contracts.py`): requests of
  `CreateResponse`/`StreamResponse` and `CreateMessage`/`StreamMessage` are
  validated against the providers' SDK-derived JSON schemas (closed, so an
  unknown field fails), and the documented response bodies and stream events
  are parsed by the library's types. The schemas and examples are a snapshot
  in `tests/providers/contracts/` (OpenAI SDK 3.41.0, Anthropic SDK 2.3.0),
  rebuilt from the offline docs dump by
  `python -m tests.providers.contracts.update`. New test dependency
  `jsonschema`, new dev dependency `types-jsonschema`.

- Event matchers in `ai_api.types`: the `IEventMatcher` protocol
  (`matches(item) -> bool`, for models and decoded JSON alike), `BaseMatcher`,
  `EventTypes(*types)`, `FieldValues(name, *values)`, `TopLevelError()`,
  `DONE` (matches `DONE_MARKER`, the `data: [DONE]` line) and `AnyOf`, built
  with `a | b`. Matchers compare by value and show what they match.
- Stream terminals: `BaseStreamMethod.__stream_terminal__` (an
  `IEventMatcher`, `None` by default; the `stream_terminal` property, also on
  `IStreamMethod`) names the protocol's last event. The SSE and NDJSON
  parsers then read the body to its end: a body that ends without the
  terminal raises `APIStreamInterruptedError`, an item after it raises
  `APIResponseError`, and `[DONE]` is matched as `DONE_MARKER` instead of
  ending the stream. `StreamResponse` declares `response.completed`,
  `response.incomplete` and `response.failed`; `StreamMessage` declares
  `message_stop`. Without a terminal nothing changes. A contract test checks
  that the declared terminal and error event types exist upstream.
- `StreamResponseParser(error_events=...)`: the matcher of decoded events
  that are raised through the error resolver, `ERROR_EVENTS`
  (`EventTypes("error") | TopLevelError()`) by default, and the
  `error_events` property.
- `BufferedResponseParser(error_responses=...)`: a matcher of parsed JSON
  results that are failures; a match is raised through the error resolver
  with the result in `raw`.
- `APIStreamInterruptedError` (an `APITransportError`, not transient, with
  `items_received`): an open stream broke or ended before its terminal.
  `Stream` wraps every transport error raised while iterating in it, with
  the original as `__cause__`; nested streams pass it through unchanged.
- `APIStatusError.raw`: the parsed object that reported the failure, such as
  OpenAI's failed `Response`, or a stream's error event when it fits the item
  type; `None` otherwise.
- Failed OpenAI responses are raised: `CreateResponse` raises a response with
  `status: "failed"`, and `StreamResponse` the `response.failed` event, as the
  `APIStatusError` that `OpenAIErrorResolver` builds from `response.error`
  (with the `Response` or the event in `raw`). This is stricter than the
  official SDK, which returns a failed response as a value.
- The shared stream contract: `AdapterStreamUseCase` and
  `AdapterCompositeUseCase` take keyword-only `terminal=`, and then require
  exactly one event that matches it, as the last one (none:
  `APIStreamInterruptedError`; an event after it: `APIResponseError`).
  `GenerationResource` passes `GENERATION_TERMINAL` (`ai_api.types.responses`),
  which matches `CompletedEvent`, so every provider's `generation.stream` ends
  with exactly one `CompletedEvent`.
- Tests: cancellation (`task.cancel()` while opening and while iterating
  releases the connection, for raw and shared streams of both providers),
  every two-chunk split of the documented provider streams, hypothesis
  property tests of the line and SSE decoders, cuts and silence mid-stream,
  and a close-delimited HTTP/1.1 body cut mid-event. New test dependency
  `hypothesis`.
- Conversations in shared generation: `GenerationRequest.input` takes a list
  of `Turn`s (`ai_api.types.requests`; frozen, unknown fields rejected,
  `Turn.user(...)` and `Turn.assistant(...)`) as well as a text, and
  `GenerationResponse.to_turn()` turns an answer into the next request's
  `assistant` turn. The list must not be empty and must end with a `user`
  turn. OpenAI gets the turns as `EasyInputMessage` items, Anthropic as
  `messages`.
- `Role`, `IncompleteReason` and `GenerationEventType` (`StrEnum`s in
  `ai_api.enums`, re-exported from `ai_api`).
- `RequestValidationError` (`ai_api`, a `ValueError`, with `provider` and the
  shared `field`): the generation adapters raise it, before anything is sent,
  for a shared request that a provider cannot take; a pydantic error behind it
  is its `__cause__`.
- `raw_incomplete_reason` and `refusal` on `GenerationResponse` and
  `CompletedEvent`. A refusal is an answer, as in both SDKs: it sets
  `incomplete_reason` to `IncompleteReason.REFUSAL` and `refusal` to OpenAI's
  refusal text, or Anthropic's `stop_details.explanation` (else its
  `category`).
- `TokenUsage.cache_read_tokens`, `cache_write_tokens` and `reasoning_tokens`
  (parts of the input and output counts; `None` if not reported), from
  OpenAI's token details and Anthropic's cache counts and thinking tokens.
- `OutputTokensDetails` (`thinking_tokens`) and `output_tokens_details` on
  Anthropic's `Usage` and `MessageDeltaUsage`.
- `docs/unification.md`: what every shared field becomes for each provider,
  what is lost, when a request or response raises, and what is not unified;
  `tests/docs/test_unification.py` checks it against the shared types and
  raises every error it lists.

### Changed

- Transports implement `_execute(method, *, headers=None, timeout=None)` and
  `_open_stream(method, *, headers=None, timeout=None)`.
- `BaseSession(headers=...)` keeps a given `IHeaders` object as it is instead
  of copying it into a new `Headers`; a mapping is still copied.
- `map_stream` is built on `transform_stream` and returns its context
  manager; its behavior is unchanged.
- `require_token_or_session` in `ai_api.client.validators` is replaced by
  `resolve_token(token, session, env)`, which also reads the environment.
  Without a token, a session and the variable, the error now names the
  variable.
- A transport error while iterating a stream (a dropped connection, a read
  timeout) is raised as `APIStreamInterruptedError`, with the original error
  as its cause, instead of as that error.
- `StreamResponseParser.is_error_event` is replaced by the `error_events`
  matcher; the default accepts the same events as before.
- `IResponseParser.parse` takes `error_resolver` after `json_loads`, and
  `BaseSession` passes its own.
- Shared streams report failures only as exceptions: the generation adapters
  no longer check the end of a stream themselves; the provider method's
  terminal and the use case's `terminal` do.
- The `type` of the shared events is a `GenerationEventType`, and
  `GENERATION_TERMINAL` matches `GenerationEventType.COMPLETED`; the values
  are the same strings as before.

### Removed

- `GenerationRequest.temperature`. Sampling parameters are not common to all
  providers: Anthropic removed `temperature`, `top_p` and `top_k` in API
  version `2026-10-21`. A provider's own method can still declare them.
- `FailedEvent` from the shared events: a failure is an exception at every
  level, so `GenerationEvent` is `TextDeltaEvent | CompletedEvent |
  OtherEvent`.

### Fixed

- `LineDecoder` no longer adds an empty line at the end of a body whose last
  `\r\n` is split between two chunks, which could complete an unfinished
  server-sent event.

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

[Unreleased]: https://github.com/GrehBan/ai_api/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/GrehBan/ai_api/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/GrehBan/ai_api/compare/2b58dfc...v0.1.0

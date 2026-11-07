# Provider clients

`OpenAIClient` builds an `AiohttpSession` for OpenAI from a token: the `https://api.openai.com/v1` URL, `BearerAuth` and `OpenAIErrorResolver`, which turns a 429 for exhausted credits or a reached spend or usage limit (`insufficient_quota`, `credit_balance_exhausted`, `organization_spend_limit_exceeded`, ...) into a `QuotaExceededError` that is not retried, also when a stream's `error` event reports it. A stream's `error` event and a `failed` response come after a 200, so their class is picked by the code or type, such as `ServiceUnavailableError` for `server_is_overloaded` and `QuotaExceededError` for `insufficient_quota`. Such a failure is never transient: the generation ran and failed, and running it again is your choice, as in the official SDK, which retries by the HTTP status only. The client owns its session, so close it, ideally with `async with`:

```python title="client.py"
from aiollm.providers.openai import OpenAIClient


async def list_models() -> ModelList:
    async with OpenAIClient("sk-...") as client:
        return await client.send_request(ListModels())
```

- **Key from the environment.** `OpenAIClient()` without a token or a session reads the key from `OPENAI_API_KEY`. An explicit token always wins over the variable, a session is never combined with it, and an empty variable counts as unset.
- **Custom session settings.** Pass a ready session with `OpenAIClient(session=...)` instead of a token. A `BaseSession` whose error resolver is not an `OpenAIErrorResolver` (a subclass of it is fine), or with authentication outside the `Authorization` header, gets a `SessionMismatchWarning` (a `UserWarning`), since both are almost certainly mistakes for OpenAI. When the session is deliberate, turn the warning off with the standard filters: `warnings.filterwarnings("ignore", category=SessionMismatchWarning)`. To change only some options of the session built from a token, pass them to the client: `OpenAIClient(token, max_body_size=..., timeout=...)` takes every keyword argument of `AiohttpSession` (`AiohttpSessionOptions`), and type checkers check them; with a ready `session=` they raise `ValueError`, since they would be ignored. `session_factory(token, max_body_size=...)` from `aiollm.providers.openai.client` builds such a session on its own.
- **Ownership.** A client closes its session when it is closed, also a session you passed in. One owner per session means no connection pool is left open by accident; create one session per client.
- **Anthropic.** `AnthropicClient` (`aiollm.providers.anthropic`) works the same way: a token, a ready `session=` or `ANTHROPIC_API_KEY`, with the `https://api.anthropic.com/v1` URL, `BearerAuth` (Anthropic also accepts `x-api-key`, so `APIKeyAuth` in a given session gets no warning) and `AnthropicErrorResolver`. The resolver reads Anthropic's `error.type` as `error_type` and `details.error_code` as `code`; a reached spend cap (429 `enforced_spend_limit_reached`) or a spend limit you set (400) is a `QuotaExceededError` that is not retried, and the 529 `overloaded_error` is a transient `ServiceUnavailableError`, like OpenAI's 503. A stream's `error` event comes after a 200, so its class is picked by its type, and it is never transient, as for OpenAI. Anthropic also requires an `anthropic-version` header; `API_VERSION` (`2026-10-21`) is the version the library's Anthropic types follow.
- **Other providers.** Subclass `BaseClient`, which delegates `send_request`, `stream`, `close` and `async with` to its session, and build the session with a `SessionFactory` (`aiollm.session`). `BaseClient` is generic in the session type: `BaseClient(AiohttpSession(...)).session` is typed as `AiohttpSession`, while `OpenAIClient`, which accepts any session, is a `BaseClient[ISession]`.

## OpenAI Responses API

`aiollm.providers.openai.methods` has `CreateResponse` and `StreamResponse` for `POST /v1/responses`. Their request fields come from `aiollm.providers.openai.methods.params` and OpenAI's response and event types from `aiollm.providers.openai.types`; both are generated from OpenAI's official OpenAPI file (see [Provider types](#provider-types)):

```python title="openai_responses.py"
from aiollm.providers.openai.adapters import output_text
from aiollm.providers.openai.methods import CreateResponse, StreamResponse
from aiollm.providers.openai.types import ResponseTextDeltaEvent


async def ask(client: OpenAIClient) -> str | None:
    response = await client.send_request(
        CreateResponse(model="gpt-6-astra", input="Hi", instructions="Be brief")
    )
    return output_text(response)


async def ask_streamed(client: OpenAIClient) -> str:
    text = ""
    method = StreamResponse(model="gpt-6-astra", input="Hi")

    async with client.stream(method) as events:
        async for event in events:
            if isinstance(event, ResponseTextDeltaEvent):
                text += event.delta

    return text
```

- **Text is joined by `output_text`.** The API sends no text field: OpenAI's schema names `Response.output_text` as a property the SDKs compute, so it reads as `None`. `output_text(response)` (`aiollm.providers.openai.adapters`) joins the `output_text` parts of the `message` items, as the SDKs do, and `reasoning_summary(response)` the reasoning summaries.
- **Nothing unknown is dropped or rejected.** An output item, content part or stream event of a type the schema does not name yet arrives as `UnknownVariant` (`aiollm.providers.openai.types.shared`), with every field the provider sent kept.
- **Failures are raised.** A `failed` response, buffered (`CreateResponse`) or as the stream's `response.failed` event, raises the `APIStatusError` that `OpenAIErrorResolver` builds from `response.error`, with the `Response` or the event in `raw`; so does the separate `error` event. `server_error` is an `InternalServerError`, `rate_limit_exceeded` a `RateLimitError`; a code without its own class, such as `invalid_prompt`, gives a plain `APIStatusError`. None of them is transient (see [Provider clients](#provider-clients)). This is stricter than the official SDK, which returns a failed response as a value.
- **Retried like the SDK.** `CreateResponse` and `StreamResponse` set `__retryable__ = True`: a lost connection, a timeout, a 5xx or a 429 with `Retry-After` is retried, a stream only while it opens. `response.incomplete` stays a regular `ResponseIncompleteEvent` with `response.incomplete_details`.
- **Checked before sending.** Every field follows the schema, and nested data is a model that refuses unknown keys. Dicts are taken and checked too, though type checkers want the models. On top of the schema, `metadata` takes up to 16 pairs with keys up to 64 and values up to 512 characters, limits OpenAI writes only in its description.
- **Only what is given is sent.** Every optional field is left out until it is given (it defaults to `MISSING`), and takes `None`, sent as `null`, where OpenAI's schema allows `null`. See [Optional fields](methods.md#optional-fields-missing-and-none).

## Anthropic Messages API

`aiollm.providers.anthropic.methods` has `CreateMessage` and `StreamMessage` for `POST /v1/messages`, with request fields in `aiollm.providers.anthropic.methods.params` and Anthropic's own types in `aiollm.providers.anthropic.types`, generated from the schemas of the offline documentation snapshot:

```python title="anthropic_messages.py"
from aiollm.providers.anthropic import AnthropicClient
from aiollm.providers.anthropic.adapters import message_text
from aiollm.providers.anthropic.methods import CreateMessage, StreamMessage
from aiollm.providers.anthropic.methods.params import UserMessageParam
from aiollm.providers.anthropic.types import RawContentBlockDeltaEvent, TextDelta


async def ask_claude(client: AnthropicClient) -> str | None:
    message = await client.send_request(
        CreateMessage(
            model="claude-opus-5-5",
            max_tokens=1024,
            system="Be brief",
            messages=[UserMessageParam(role="user", content="Hi")],
        )
    )
    return message_text(message)


async def ask_claude_streamed(client: AnthropicClient) -> str:
    text = ""
    method = StreamMessage(
        model="claude-opus-5-5",
        max_tokens=1024,
        messages=[UserMessageParam(role="user", content="Hi")],
    )

    async with client.stream(method) as events:
        async for event in events:
            if isinstance(event, RawContentBlockDeltaEvent) and isinstance(
                event.delta, TextDelta
            ):
                text += event.delta.text

    return text
```

- **Retried like the SDK.** `CreateMessage` and `StreamMessage` set `__retryable__ = True`: a lost connection, a timeout, a 5xx, the 529 `overloaded_error` or a 429 with `Retry-After` is retried, a stream only while it opens.
- **The version is part of the method.** Both methods send `anthropic-version: 2026-10-21` (`VERSION_HEADERS`), the version their types follow, whatever session sends them. A method of your own for Anthropic sets `__request_headers__ = VERSION_HEADERS` too.
- **Text is joined by `message_text`.** `message_text(message)` (`aiollm.providers.anthropic.adapters`) joins the `text` blocks of the content and is `None` without them; `message_thinking(message)` joins the thinking.
- **Nothing unknown is dropped or rejected.** Content blocks, deltas and stream events of a type the schema does not name (such as `ping`) arrive as `UnknownVariant` (`aiollm.providers.anthropic.types.shared`), with every field kept. The `error` event is raised as an `APIStatusError` chosen by its type, such as `ServiceUnavailableError` for `overloaded_error`; like OpenAI's in-band failures, it is never transient.
- **Checked before sending.** Every field follows the schema, and nested data is a model that refuses unknown keys. Beyond the schema, which names no limits here, `messages` needs at least one turn, `max_tokens` at least 0 and `metadata.user_id` at most 512 characters (`MessageLimits`). There are no `temperature`, `top_p` and `top_k`: API version 2026-10-21 rejects them.
- **Only what is given is sent.** An optional field is left out until it is given; it takes `None` only where Anthropic's schema allows `null`.

## Provider types

The provider types and request parameters are generated with aiollm-codegen by `scripts/generate_provider_types.py`: OpenAI's from its official OpenAPI file, Anthropic's from the snapshot of the documentation. Every class names the upstream schema it mirrors in `__upstream_schema__`, and `tests/providers/test_wire_contracts.py` checks each field against it. The method classes in `methods/` stay hand-written: they add what the schema does not say, such as retries, the parsing of a `failed` response and the limits above.

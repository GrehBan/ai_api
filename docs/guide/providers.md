# Provider clients

`OpenAIClient` builds an `AiohttpSession` for OpenAI from a token: the `https://api.openai.com/v1` URL, `BearerAuth` and `OpenAIErrorResolver`, which turns a 429 for exhausted credits or a reached spend or usage limit (`insufficient_quota`, `credit_balance_exhausted`, `organization_spend_limit_exceeded`, ...) into a `QuotaExceededError` that is not retried, also when a stream's `error` event reports it. A stream's `error` event and a `failed` response come after a 200, so their class is picked by the code or type, such as `ServiceUnavailableError` for `server_is_overloaded` and `QuotaExceededError` for `insufficient_quota`. Such a failure is never transient: the generation ran and failed, and running it again is your choice, as in the official SDK, which retries by the HTTP status only. The client owns its session, so close it, ideally with `async with`:

```python title="client.py"
from ai_api.providers.openai import OpenAIClient


async def list_models() -> ModelList:
    async with OpenAIClient("sk-...") as client:
        return await client.send_request(ListModels())
```

- **Key from the environment.** `OpenAIClient()` without a token or a session reads the key from `OPENAI_API_KEY`. An explicit token always wins over the variable, a session is never combined with it, and an empty variable counts as unset.
- **Custom session settings.** Pass a ready session with `OpenAIClient(session=...)` instead of a token. A `BaseSession` whose error resolver is not an `OpenAIErrorResolver` (a subclass of it is fine), or with authentication outside the `Authorization` header, gets a `SessionMismatchWarning` (a `UserWarning`), since both are almost certainly mistakes for OpenAI. When the session is deliberate, turn the warning off with the standard filters: `warnings.filterwarnings("ignore", category=SessionMismatchWarning)`. To change only some options of the session built from a token, pass them to the client: `OpenAIClient(token, max_body_size=..., timeout=...)` takes every keyword argument of `AiohttpSession` (`AiohttpSessionOptions`), and type checkers check them; with a ready `session=` they raise `ValueError`, since they would be ignored. `session_factory(token, max_body_size=...)` from `ai_api.providers.openai.client` builds such a session on its own.
- **Ownership.** A client closes its session when it is closed, also a session you passed in. One owner per session means no connection pool is left open by accident; create one session per client.
- **Anthropic.** `AnthropicClient` (`ai_api.providers.anthropic`) works the same way: a token, a ready `session=` or `ANTHROPIC_API_KEY`, with the `https://api.anthropic.com/v1` URL, `BearerAuth` (Anthropic also accepts `x-api-key`, so `APIKeyAuth` in a given session gets no warning) and `AnthropicErrorResolver`. The resolver reads Anthropic's `error.type` as `error_type` and `details.error_code` as `code`; a reached spend cap (429 `enforced_spend_limit_reached`) or a spend limit you set (400) is a `QuotaExceededError` that is not retried, and the 529 `overloaded_error` is a transient `ServiceUnavailableError`, like OpenAI's 503. A stream's `error` event comes after a 200, so its class is picked by its type, and it is never transient, as for OpenAI. Anthropic also requires an `anthropic-version` header; `API_VERSION` (`2026-10-21`) is the version the library's Anthropic types follow.
- **Other providers.** Subclass `BaseClient`, which delegates `send_request`, `stream`, `close` and `async with` to its session, and build the session with a `SessionFactory` (`ai_api.session`). `BaseClient` is generic in the session type: `BaseClient(AiohttpSession(...)).session` is typed as `AiohttpSession`, while `OpenAIClient`, which accepts any session, is a `BaseClient[ISession]`.

## OpenAI Responses API

`ai_api.providers.openai.methods` has `CreateResponse` and `StreamResponse` for `POST /v1/responses`, with OpenAI's own types in `ai_api.providers.openai.types`:

```python title="openai_responses.py"
from ai_api.providers.openai.methods import CreateResponse, StreamResponse
from ai_api.providers.openai.types import OutputTextDeltaEvent


async def ask(client: OpenAIClient) -> str | None:
    response = await client.send_request(
        CreateResponse(model="gpt-6-astra", input="Hi", instructions="Be brief")
    )
    return response.output_text


async def ask_streamed(client: OpenAIClient) -> str:
    text = ""
    method = StreamResponse(model="gpt-6-astra", input="Hi")

    async with client.stream(method) as events:
        async for event in events:
            if isinstance(event, OutputTextDeltaEvent):
                text += event.delta

    return text
```

- **`output_text` is computed.** The API sends no such field; like the official SDKs, `Response.output_text` joins the `output_text` parts of the `message` items, and is `None` when there are none.
- **Nothing unknown is dropped or rejected.** Output items, content parts and stream events that the library does not model yet (reasoning, tool calls, new event types) arrive as `UnknownOutputItem`, `UnknownContent` and `UnknownEvent`, with every field the provider sent kept as an extra field.
- **Failures are raised.** A `failed` response, buffered (`CreateResponse`) or as the stream's `response.failed` event, raises the `APIStatusError` that `OpenAIErrorResolver` builds from `response.error`, with the `Response` or the event in `raw`; so does the separate `error` event. `server_error` is an `InternalServerError`, `rate_limit_exceeded` a `RateLimitError`; a code without its own class, such as `invalid_prompt`, gives a plain `APIStatusError`. None of them is transient (see [Provider clients](#provider-clients)). This is stricter than the official SDK, which returns a failed response as a value.
- **Retried like the SDK.** `CreateResponse` and `StreamResponse` set `__retryable__ = True`: a lost connection, a timeout, a 5xx or a 429 with `Retry-After` is retried, a stream only while it opens. `response.incomplete` stays a regular `ResponseStateEvent` with `response.incomplete_details`.
- **Checked before sending.** `max_output_tokens` of at least 16, `temperature` from 0 to 2, and metadata of up to 16 pairs with keys up to 64 and values up to 512 characters, as OpenAI documents them.

## Anthropic Messages API

`ai_api.providers.anthropic.methods` has `CreateMessage` and `StreamMessage` for `POST /v1/messages`, with Anthropic's own types in `ai_api.providers.anthropic.types`:

```python title="anthropic_messages.py"
from ai_api.providers.anthropic import AnthropicClient
from ai_api.providers.anthropic.methods import CreateMessage, StreamMessage
from ai_api.providers.anthropic.types import ContentBlockDeltaEvent, TextDelta


async def ask_claude(client: AnthropicClient) -> str | None:
    message = await client.send_request(
        CreateMessage(
            model="claude-opus-5-5",
            max_tokens=1024,
            system="Be brief",
            messages=[{"role": "user", "content": "Hi"}],
        )
    )
    return message.text


async def ask_claude_streamed(client: AnthropicClient) -> str:
    text = ""
    method = StreamMessage(
        model="claude-opus-5-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hi"}],
    )

    async with client.stream(method) as events:
        async for event in events:
            if isinstance(event, ContentBlockDeltaEvent) and isinstance(
                event.delta, TextDelta
            ):
                text += event.delta.text

    return text
```

- **Retried like the SDK.** `CreateMessage` and `StreamMessage` set `__retryable__ = True`: a lost connection, a timeout, a 5xx, the 529 `overloaded_error` or a 429 with `Retry-After` is retried, a stream only while it opens.
- **The version is part of the method.** Both methods send `anthropic-version: 2026-10-21` (`VERSION_HEADERS`), the version their types follow, whatever session sends them. A method of your own for Anthropic sets `__request_headers__ = VERSION_HEADERS` too.
- **`text` is computed.** `Message.text` joins the `text` blocks of the content, and is `None` when there are none.
- **Nothing unknown is dropped or rejected.** Content blocks, deltas and stream events the library does not model yet (thinking, tool use, new event types) arrive as `UnknownBlock`, `UnknownDelta` and `UnknownEvent`, with every field kept. The `error` event is raised as an `APIStatusError` chosen by its type, such as `ServiceUnavailableError` for `overloaded_error`; like OpenAI's in-band failures, it is never transient.
- **Checked before sending.** `max_tokens` is required; `messages` needs at least one turn, and a turn (a `MessageParam`, a `TypedDict`, so dict literals type-check) takes only `role` and `content`; `metadata` takes only `user_id`, up to 512 characters. There are no `temperature`, `top_p` and `top_k`: API version 2026-10-21 rejects them.

# Shared generation

`client.generation` is the provider-neutral way to generate text: a `GenerationResource` that takes the fields of a shared `GenerationRequest` as keyword arguments, as the providers' SDKs do (or a ready `GenerationRequest`, not both), and returns shared types, with the provider's own response in `raw`. `OpenAIClient` and `AnthropicClient` both have it, and both are an `IGenerationClient`, so the code below works with either; with an `AnthropicClient`, `raw` is a `Message`. All of it is imported from `aiollm`.

```python title="shared_generation.py"
from typing import Any

from aiollm import (
    CompletedEvent,
    GenerationRequest,
    GenerationResponse,
    IGenerationClient,
    Turn,
)


async def generate(client: IGenerationClient) -> GenerationResponse[Any]:
    return await client.generation.create(
        model="gpt-6-astra",
        input="Hi",
        instructions="Be brief",
        max_output_tokens=64,
    )


async def generate_streamed(
    client: IGenerationClient,
) -> tuple[str, CompletedEvent[Any]]:
    text = ""
    async with client.generation.stream(
        model="gpt-6-astra", input="Hi"
    ) as events:
        async for delta in events.text():
            text += delta
        completed = await events.final()

    return text, completed


async def chat(client: IGenerationClient) -> GenerationResponse[Any]:
    first = await client.generation.create(model="gpt-6-astra", input="Hi")
    request = GenerationRequest(
        model="gpt-6-astra",
        input=[Turn.user("Hi"), first.to_turn(), Turn.user("Tell me more")],
    )
    return await client.generation.create(request)
```

What each shared field becomes for each provider, what is lost and when a request raises is listed in [docs/unification.md](../unification.md).

Provider options are typed by the concrete client:

```python title="provider_options.py"
from aiollm import GenerationResponse
from aiollm.providers.openai import OpenAIClient
from aiollm.providers.openai.types import Response


async def generate_unstored(
    client: OpenAIClient,
) -> GenerationResponse[Response]:
    return await client.generation.create(
        model="gpt-6-astra",
        input="Hi",
        provider_options={"store": False, "temperature": 0.2},
    )
```

- **Keyword arguments or a request.** `create(model=..., input=...)` and `stream(...)` take exactly the fields of `GenerationRequest`; a misspelled field is a type error and, at run time, a pydantic `ValidationError`. Passing a `GenerationRequest` and fields together raises `TypeError`.
- **Effort.** `effort=` (an `Effort` from `aiollm`: `LOW`, `MEDIUM`, `HIGH`, `XHIGH`, `MAX`) sets how much work the model puts into the response, OpenAI's `reasoning.effort` and Anthropic's `output_config.effort`. Left out, it is not sent and the model's default applies; a model that does not take a level answers 400. OpenAI's `none` and `minimal` are not shared, since Anthropic has nothing like them.
- **Provider options.** A field that only one provider has, which the shared request does not hold, goes in `provider_options=` of `create(...)` or `stream(...)`, for that call only. The options are a `TypedDict` of the provider (`OpenAIGenerationOptions`: `store`, `previous_response_id`, `temperature`, `top_p`; `AnthropicGenerationOptions`: `stop_sequences`), so a wrong key or value is a type error; at run time an unknown key, a shared field among them, or an invalid value raises `RequestValidationError` before anything is sent. A key left out is not sent. Code typed as `IGenerationClient` cannot pass them, since it works with any provider: its `create` has no such parameter (example below).
- **Call options.** `create(...)` and `stream(...)` also take `headers=`, `timeout=` and `retry_policy=`, as `send_request` does (see [Per-request headers and timeout](sessions.md#per-request-headers-and-timeout)), and pass them to the session unchanged, so they work the same for every provider. They are options of the call, not fields of the request: no field of `GenerationRequest` is named like one of them.
- **Stream helpers.** `stream(...)` yields a `GenerationStream`: iterate it for every shared event, or use `events.text()` for the text deltas only and `await events.final()` for the `CompletedEvent`, which carries the response's `id`, `model`, whole `text`, `usage` and why it stopped. `final()` reads the rest of the stream, and after `text()` or a loop it returns the `CompletedEvent` already read. When a read failed (a provider's error event, a cut connection), every later read, `text()` and `final()` raise that same error again, never a misleading "ended without a CompletedEvent".
- **Any provider.** `IGenerationClient[RawT = Any, RawItemT = Any]` (from `aiollm`) is a client with `generation`; every provider client matches it, so code typed with it takes any of them.
- **A conversation is a list of turns.** `input` is a text, taken as one `user` turn, or a list of `Turn`s (`Turn.user(...)`, `Turn.assistant(...)`, or `response.to_turn()` for an earlier answer); the list must not be empty, and its last turn must be a `user` one. System prompts go in `instructions`, not in turns.
- **Optional fields are left out until given.** `instructions`, `max_output_tokens` and `metadata` default to `MISSING`, and the adapters pass them on as they are, so a provider gets only what was given. They take no `None`, which would mean `null`: Anthropic's schema has no `null` for them, so `create(..., instructions=None)` is a type error and a `ValidationError`. A field not given reads as `MISSING`, which is truthy: test it with `request.instructions is MISSING`, not `if request.instructions`. Setting a field later (`request.instructions = None`) is validated too; `request.model_copy(update=...)` is not, as everywhere in Pydantic, so a value it lets in reaches the adapter: a `None` is sent as `null` to OpenAI, whose schema takes it, and raises `RequestValidationError` for Anthropic, whose schema does not.
- **A request a provider cannot take raises `RequestValidationError`** (from `aiollm`, a `ValueError`) before anything is sent, with the `provider` and the shared `field` at fault; a pydantic error behind it is its `__cause__`. Errors of `GenerationRequest` itself stay pydantic `ValidationError`s.
- **Why it stopped.** `incomplete_reason` is an `IncompleteReason` (`MAX_TOKENS`, `REFUSAL` or `OTHER`; `None` if the generation finished), and `raw_incomplete_reason` keeps the provider's own reason. A refusal is an answer, not an error, as in both SDKs: it sets `incomplete_reason` to `REFUSAL` and `refusal` to the provider's explanation.
- **OpenAI mapping.** `OpenAIGenerationAdapter` (`aiollm.providers.openai.adapters`) builds `CreateResponse` or `StreamResponse` from the request, turns as `EasyInputMessage` items, so OpenAI's limits, such as `max_output_tokens` of at least 16, are checked before sending. Parameters the shared request lacks reach it through `provider_options` (`OpenAIGenerationOptions`), or stay on the methods above.
- **OpenAI stopped early.** An `incomplete` response, cut off by `max_output_tokens` for example, keeps its `text` and gets `incomplete_reason` (`MAX_TOKENS` for `max_output_tokens`, `OTHER` for any other reason) with OpenAI's reason in `raw_incomplete_reason`; in a stream, `response.incomplete` is a `CompletedEvent` with them. A `refusal` content part makes the response a refusal, with the part's text in `refusal`.
- **OpenAI failures are raised.** A `failed` response raises the `APIStatusError` described under the Responses API, before the adapter sees it, in `create` and in `stream` alike. Any other buffered status that is not `completed` or `incomplete` (`cancelled`, `in_progress`, ...) raises `APIResponseError`, since there is no generated text to return.
- **Anthropic mapping.** `AnthropicGenerationAdapter` (`aiollm.providers.anthropic.adapters`) sends a text `input` as one `user` turn, a list of turns as `messages`, and `instructions` as `system`. Anthropic requires `max_tokens`, so a request without `max_output_tokens` gets the client's `default_max_tokens` (`AnthropicClient(..., default_max_tokens=16000)`; thinking, always on for the 5.x models, counts toward it, so a lower value can end a reply before its text). Metadata other than `user_id` raises `RequestValidationError` instead of being dropped.
- **Anthropic counts and stops.** `input_tokens` is the sum of the input outside the cache and the input read from and written to it (`cache_read_tokens`, `cache_write_tokens`), and `reasoning_tokens` are the thinking tokens. A `stop_reason` other than `end_turn` or `stop_sequence` stops the generation early, with the text kept: `max_tokens` is `MAX_TOKENS`, `refusal` is `REFUSAL` (with `stop_details.explanation`, else its `category`, in `refusal`) and any other is `OTHER`, with the `stop_reason` in `raw_incomplete_reason`. In a stream, `CompletedEvent` comes on `message_stop`, with the input counts of `message_start` and the last cumulative counts of `message_delta`; a message without a `stop_reason` raises `APIResponseError`. Anthropic reports failures as `error` events, which are raised as `APIStatusError`.
- **The end is checked for every provider.** `stream` yields exactly one `CompletedEvent`, as the last event: a stream that ends without it raises `APIStreamInterruptedError` (the provider method's terminal check usually raises it first), and an event after it raises `APIResponseError`. When the provider's terminal proved the stream complete and the adapter still gave no `CompletedEvent`, the adapter lost it: that raises `RuntimeError` naming the adapter, not an interruption. A failure is always an exception, never an event.

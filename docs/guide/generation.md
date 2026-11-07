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

A `user` turn can hold images and PDF documents next to its text: its `content` is then a list of parts, `TextPart`, `ImagePart` and `DocumentPart`, all from `aiollm`.

```python title="generation_parts.py"
from pathlib import Path

from aiollm import DocumentPart, ImagePart, TextPart


async def describe(
    client: IGenerationClient, image: Path, report: Path
) -> GenerationResponse[Any]:
    question = Turn.user(
        [
            TextPart(text="Does the chart match the report?"),
            await ImagePart.load(image),
            await DocumentPart.load(report),
            ImagePart(url="https://example.com/logo.png"),
        ]
    )
    return await client.generation.create(
        model="gpt-6-astra", input=[question]
    )
```

- **Bytes or a URL.** An `ImagePart` or a `DocumentPart` holds the file's bytes, sent inline in base64, or a URL the provider downloads it from, exactly one of them. `ImagePart(data=..., media_type="image/png")` takes bytes already in memory; JPEG, PNG, GIF and WebP are the formats both providers take. A document is a PDF; `DocumentPart(data=..., filename=...)` names it: OpenAI's `filename` (`document.pdf` otherwise) and Anthropic's `title`.
- **Reading a file.** `await ImagePart.load(path)` and `await DocumentPart.load(path)` read the file in a worker thread, so the event loop goes on; an image's format comes from the file's extension (`media_type=` gives it otherwise, and an unknown extension raises `ValueError`), a document's name from the file. The bytes stay in the part, so a request sent again, by a retry or in the next turn of a conversation, does not read the file again. The format is taken on trust: the extension is not checked against the file's content (a JPEG named `photo.png` is sent as `image/png`), and `DocumentPart.load` reads any file as a PDF; a wrong one is refused by the provider.
- **Only the user's turns.** Parts go in `user` turns; an `assistant` turn is text, and parts in it raise `ValidationError`. The providers limit the size of an image, a document and a request, and answer 400 past them; the parts are not checked against them. A file is read whole and its bytes stay in the part; while a request is built, their base64 (a third larger) and the JSON body are in memory too. The default JSON encoder, orjson, keeps a buffer much larger than the long string it encodes, so the peak was measured at about 15 times the file's size (files of 1 to 30 MB, orjson 3.11 to 3.13); a client made with `json_dumps=lambda value: json.dumps(value).encode()` (the standard `json` module, which makes a string and then its bytes) brings it to about five times. Check a file's size before loading it where that matters. What each part becomes is listed under [Content parts](../unification.md#content-parts).

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
- **Structured output.** `output_schema=` takes a JSON Schema (a `dict`; the `JSONSchema` type is in `aiollm.types.requests`), and the response `text` is then JSON that follows it: OpenAI's `text.format` with `strict: true`, Anthropic's `output_config.format`. OpenAI requires a name for the format: `OpenAIGenerationAdapter(schema_name=...)` sets it, `"output"` by default. The schema is sent as it is; each provider supports its own subset of JSON Schema and answers 400 to the rest. A reply cut off by `max_output_tokens` is unfinished JSON, with `incomplete_reason` `MAX_TOKENS`, so check it before parsing.
- **Reasoning.** `response.reasoning` and `CompletedEvent.reasoning` hold a summary of the model's reasoning, and a stream yields it piece by piece as `ReasoningDeltaEvent`s (`events.text()` skips them). A provider sends a summary only when asked, through `provider_options`: OpenAI's `{"reasoning": {"summary": "auto"}}`, which the adapter merges with the shared `effort` into one `reasoning` (setting `effort` there too raises `RequestValidationError`), and Anthropic's `{"thinking": {"type": "adaptive", "display": "summarized"}}`. Without one, `reasoning` is `None` for both. OpenAI's raw reasoning text and encrypted reasoning, and Anthropic's signatures, stay in `raw`.
- **End user.** `end_user_id=` is an opaque id of the end user the request is made for, such as a hash of an account id, never personal data: OpenAI's `safety_identifier` (up to 128 characters) and Anthropic's `metadata.user_id` (up to 512); both use it to detect abuse. OpenAI's free-form `metadata` is a provider option.
- **Tools.** `tools=`, `tool_choice=` and `parallel_tool_calls=` let the model call functions of your code, `response.tool_calls` holds its calls, and `response.to_turn()` and `ToolResult`s carry the loop on; see [Tools](tools.md).
- **Provider options.** A field that only one provider has, which the shared request does not hold, goes in `provider_options=` of `create(...)` or `stream(...)`, for that call only. The options are a `TypedDict` of the provider (`OpenAIGenerationOptions`: `metadata`, `reasoning`, `store`, `previous_response_id`, `temperature`, `top_p`, `include`, `max_tool_calls`; `AnthropicGenerationOptions`: `stop_sequences`, `thinking`), so a wrong key or value is a type error; at run time an unknown key, a shared field among them, or an invalid value raises `RequestValidationError` before anything is sent. A key left out is not sent. Code typed as `IGenerationClient` cannot pass them, since it works with any provider: its `create` has no such parameter (example below).
- **Call options.** `create(...)` and `stream(...)` also take `headers=`, `timeout=` and `retry_policy=`, as `send_request` does (see [Per-request headers and timeout](sessions.md#per-request-headers-and-timeout)), and pass them to the session unchanged, so they work the same for every provider. They are options of the call, not fields of the request: no field of `GenerationRequest` is named like one of them.
- **Stream helpers.** `stream(...)` yields a `GenerationStream`: iterate it for every shared event, or use `events.text()` for the text deltas only and `await events.final()` for the `CompletedEvent`, which carries the response's `id`, `model`, whole `text`, `usage`, `tool_calls` and why it stopped. Tool calls stream as `ToolCallStartEvent`, `ToolCallDeltaEvent` and `ToolCallDoneEvent` ([Tools](tools.md)). `final()` reads the rest of the stream, and after `text()` or a loop it returns the `CompletedEvent` already read. When a read failed (a provider's error event, a cut connection), every later read, `text()` and `final()` raise that same error again, never a misleading "ended without a CompletedEvent".
- **Any provider.** `IGenerationClient[RawT = Any, RawItemT = Any]` (from `aiollm`) is a client with `generation`; every provider client matches it, so code typed with it takes any of them.
- **A conversation is a list of items.** `input` is a text, taken as one `user` turn, or a list of `Turn`s (`Turn.user(...)`, with a text or parts, `Turn.assistant(...)`, or `response.to_turn()` for an earlier answer), with `ToolCallTurn`s and `ToolResult`s in a tool loop (see [Tools](tools.md)); the list must not be empty, and it ends with a `user` turn or a tool result. System prompts go in `instructions`, not in turns.
- **Optional fields are left out until given.** `instructions`, `max_output_tokens`, `effort`, `output_schema`, `end_user_id`, `tools`, `tool_choice` and `parallel_tool_calls` default to `MISSING`, and the adapters pass them on as they are, so a provider gets only what was given. They take no `None`, which would mean `null`: Anthropic's schema has no `null` for them, so `create(..., instructions=None)` is a type error and a `ValidationError`. A field not given reads as `MISSING`, which is truthy: test it with `request.instructions is MISSING`, not `if request.instructions`. Setting a field later (`request.instructions = None`) is validated too; `request.model_copy(update=...)` is not, as everywhere in Pydantic, so a value it lets in reaches the adapter: a `None` is sent as `null` to OpenAI, whose schema takes it, and raises `RequestValidationError` for Anthropic, whose schema does not.
- **A request a provider cannot take raises `RequestValidationError`** (from `aiollm`, a `ValueError`) before anything is sent, with the `provider` and the shared `field` at fault; a pydantic error behind it is its `__cause__`. Errors of `GenerationRequest` itself stay pydantic `ValidationError`s.
- **Why it stopped.** `incomplete_reason` is an `IncompleteReason` (`MAX_TOKENS`, `REFUSAL` or `OTHER`; `None` if the generation finished), and `raw_incomplete_reason` keeps the provider's own reason. A refusal is an answer, not an error, as in both SDKs: it sets `incomplete_reason` to `REFUSAL` and `refusal` to the provider's explanation.
- **OpenAI mapping.** `OpenAIGenerationAdapter` (`aiollm.providers.openai.adapters`) builds `CreateResponse` or `StreamResponse` from the request, turns as `EasyInputMessage` items, so OpenAI's limits, such as `max_output_tokens` of at least 16, are checked before sending. Parameters the shared request lacks reach it through `provider_options` (`OpenAIGenerationOptions`), or stay on the methods above.
- **OpenAI stopped early.** An `incomplete` response, cut off by `max_output_tokens` for example, keeps its `text` and gets `incomplete_reason` (`MAX_TOKENS` for `max_output_tokens`, `OTHER` for any other reason) with OpenAI's reason in `raw_incomplete_reason`; in a stream, `response.incomplete` is a `CompletedEvent` with them. A `refusal` content part makes the response a refusal, with the part's text in `refusal`.
- **OpenAI failures are raised.** A `failed` response raises the `APIStatusError` described under the Responses API, before the adapter sees it, in `create` and in `stream` alike. Any other buffered status that is not `completed` or `incomplete` (`cancelled`, `in_progress`, ...) raises `APIResponseError`, since there is no generated text to return.
- **Anthropic mapping.** `AnthropicGenerationAdapter` (`aiollm.providers.anthropic.adapters`) sends a text `input` as one `user` turn, a list of turns as `messages`, and `instructions` as `system`. Anthropic requires `max_tokens`, so a request without `max_output_tokens` gets the client's `default_max_tokens` (`AnthropicClient(..., default_max_tokens=16000)`; thinking, always on for the 5.x models, counts toward it, so a lower value can end a reply before its text). `end_user_id` becomes `metadata.user_id`.
- **Anthropic counts and stops.** `input_tokens` is the sum of the input outside the cache and the input read from and written to it (`cache_read_tokens`, `cache_write_tokens`), and `reasoning_tokens` are the thinking tokens. A `stop_reason` other than `end_turn`, `stop_sequence` or `tool_use` (the model waits for tool results) stops the generation early, with the text kept: `max_tokens` is `MAX_TOKENS`, `refusal` is `REFUSAL` (with `stop_details.explanation`, else its `category`, in `refusal`) and any other is `OTHER`, with the `stop_reason` in `raw_incomplete_reason`. In a stream, `CompletedEvent` comes on `message_stop`, with the input counts of `message_start` and the last cumulative counts of `message_delta`; a message without a `stop_reason` raises `APIResponseError`. Anthropic reports failures as `error` events, which are raised as `APIStatusError`.
- **The end is checked for every provider.** `stream` yields exactly one `CompletedEvent`, as the last event: a stream that ends without it raises `APIStreamInterruptedError` (the provider method's terminal check usually raises it first), and an event after it raises `APIResponseError`. When the provider's terminal proved the stream complete and the adapter still gave no `CompletedEvent`, the adapter lost it: that raises `RuntimeError` naming the adapter, not an interruption. A failure is always an exception, never an event.

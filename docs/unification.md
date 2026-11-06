# Unification: what the shared types promise

`client.generation` speaks shared types (`GenerationRequest`, `GenerationResponse`, `GenerationEvent`) for every provider. This page lists, field by field, what each shared field becomes for each provider, what is lost on the way, and when a request or a response raises instead.

Two rules decide what is shared:

- A shared type holds only what every provider has. Provider-only data stays in the provider's own types and is reached through `raw`, which always holds the provider's whole response or event.
- Nothing is dropped silently. A shared request that a provider cannot take raises `RequestValidationError` before anything is sent.
- Nothing is invented either. An optional field that is not given is not sent to the provider at all, not even as `null`; the shared fields take no `None`.

`tests/docs/test_unification.py` checks this page against the code: every field of the shared types has a row here, and every case in [Errors](#errors) raises the error it names.

## Request

`GenerationRequest` maps to OpenAI's `POST /responses` (`CreateResponse`, `StreamResponse`) and Anthropic's `POST /messages` (`CreateMessage`, `StreamMessage`).

| Shared field | OpenAI | Anthropic | Lost, or when it raises |
| --- | --- | --- | --- |
| `GenerationRequest.model` | `model` | `model` | Sent as given; the provider rejects an unknown model with an `APIStatusError`. |
| `GenerationRequest.input` | `input`: a text as is; a list of turns as `EasyInputMessage` items (`role`, `content`) | `messages`: a text as one `user` turn; a list of turns as they are | A list must not be empty and must end with a `user` turn: Anthropic rejects an assistant prefill on current models, and OpenAI gives a last assistant turn another meaning. Images, files, tool calls and results are not shared. |
| `GenerationRequest.instructions` | `instructions` | `system` | System and developer turns are not shared: instructions go here. Not sent when not given. |
| `GenerationRequest.max_output_tokens` | `max_output_tokens`, at least 16, reasoning included; without it, the model's limit | `max_tokens`, thinking included; required, so without it the adapter's `default_max_tokens` (16000, room for thinking) | Must be positive. Below 16 raises for OpenAI. Not sent to OpenAI when not given. |
| `GenerationRequest.end_user_id` | `safety_identifier`, up to 128 characters | `metadata.user_id`, up to 512 characters | An opaque id (a hash), never personal data; both providers use it to detect abuse. A longer value raises before sending. Not sent when not given. OpenAI's `metadata`, which has no Anthropic counterpart, is a provider option. |
| `GenerationRequest.effort` | `reasoning.effort`: how deeply a reasoning model reasons | `output_config.effort`: all output tokens, thinking and reply | An `Effort`: `low`, `medium`, `high`, `xhigh` or `max`, the levels both have. OpenAI's `none` and `minimal` have no Anthropic counterpart and are not shared (`CreateResponse` takes them). A model that does not take a level answers 400, an `APIStatusError`. Not sent when not given, so the model's own default applies. |
| `GenerationRequest.output_schema` | `text.format`: `json_schema` with the adapter's `schema_name` (`"output"` unless set) and `strict: true` | `output_config.format`: `json_schema`, next to `effort` in the same object | The schema is sent as it is: each provider takes its own subset of JSON Schema and answers 400 to the rest, an `APIStatusError`. The JSON comes back in `text`; a reply cut by the token limit is invalid JSON, with `incomplete_reason` `MAX_TOKENS`. Not sent when not given. |

## Response

| Shared field | OpenAI | Anthropic | Lost, or when it raises |
| --- | --- | --- | --- |
| `GenerationResponse.id` | `id` | `id` | — |
| `GenerationResponse.provider` | `Provider.OPENAI` | `Provider.ANTHROPIC` | — |
| `GenerationResponse.model` | `model` | `model` | — |
| `GenerationResponse.text` | The `output_text` parts of the `message` items, joined (as the SDKs' `output_text`) | The `text` blocks, joined | `None` without text. Annotations, citations, encrypted or raw reasoning and tool calls stay in `raw`. |
| `GenerationResponse.usage` | `usage`, see [Token usage](#token-usage) | `usage`, see [Token usage](#token-usage) | `None` if OpenAI sends no usage. |
| `GenerationResponse.incomplete_reason` | `REFUSAL` if the output has a `refusal` part; else, for `status: "incomplete"`, `MAX_TOKENS` for `max_output_tokens` and `OTHER` for any other reason (`content_filter`, `max_messages`, `steered`, none given) | `MAX_TOKENS` for `stop_reason: "max_tokens"`, `REFUSAL` for `"refusal"`, `OTHER` for any other early stop (`model_context_window_exceeded`, `pause_turn`, a new reason); `None` for `end_turn`, `stop_sequence` and `tool_use` (the model waits for tool results, as a `completed` OpenAI response with function calls) | Only the values both providers have are shared; the provider's reason is in `raw_incomplete_reason`. A `failed` OpenAI response raises an `APIStatusError`, any status other than `completed` or `incomplete` raises `APIResponseError`, and an Anthropic message without a `stop_reason` raises `APIResponseError`. |
| `GenerationResponse.raw_incomplete_reason` | `incomplete_details.reason` of an incomplete response | `stop_reason` of an early stop | `None` when the provider gives no reason, such as a completed OpenAI response with a refusal part. |
| `GenerationResponse.refusal` | The `refusal` parts, joined | `stop_details.explanation`, else `stop_details.category`, of a `refusal` stop | A refusal is an answer, not an error, as in both SDKs. `None` if the provider gives no explanation. |
| `GenerationResponse.reasoning` | The `summary_text` parts of the `reasoning` items, joined; there only when the request asks for them with `provider_options={"reasoning": {"summary": "auto"}}` | The `thinking` blocks, joined; a summary on the 5.x models, there only with `provider_options={"thinking": {"type": "adaptive", "display": "summarized"}}` | `None` without a summary, the same for both. OpenAI's raw `reasoning_text` and `encrypted_content`, and Anthropic's `signature` and `redacted_thinking`, stay in `raw`. |
| `GenerationResponse.raw` | `Response` | `Message` | Holds everything the shared fields leave out. |

`response.to_turn()` turns a response into an `assistant` turn for the next request; a response without text raises `ValueError`.

## Token usage

Cache and reasoning counts are parts of the input and output counts, not added to them. `None` means the provider did not report the count.

| Shared field | OpenAI | Anthropic | Lost, or when it raises |
| --- | --- | --- | --- |
| `TokenUsage.input_tokens` | `input_tokens` | `input_tokens` + `cache_creation_input_tokens` + `cache_read_input_tokens` | — |
| `TokenUsage.output_tokens` | `output_tokens` | `output_tokens` | — |
| `TokenUsage.total_tokens` | `total_tokens` | The input above plus `output_tokens` | — |
| `TokenUsage.cache_read_tokens` | `input_tokens_details.cached_tokens` | `cache_read_input_tokens` | — |
| `TokenUsage.cache_write_tokens` | `input_tokens_details.cache_write_tokens` | `cache_creation_input_tokens` | Anthropic's split by cache lifetime (`cache_creation`) stays in `raw`. |
| `TokenUsage.reasoning_tokens` | `output_tokens_details.reasoning_tokens` | `output_tokens_details.thinking_tokens` | — |

Server tool use, service tier, speed and inference region (Anthropic) and tool usage (OpenAI) stay in `raw`.

## Stream events

`generation.stream()` yields one shared event per provider event. Its `type` is a `GenerationEventType`.

| Shared event | OpenAI | Anthropic | Lost, or when it raises |
| --- | --- | --- | --- |
| `TextDeltaEvent` | `response.output_text.delta` | `content_block_delta` with a `text_delta` | — |
| `ReasoningDeltaEvent` | `response.reasoning_summary_text.delta` | `content_block_delta` with a non-empty `thinking_delta` | Only when a summary was asked for, as `GenerationResponse.reasoning`; `GenerationStream.text()` skips it. |
| `CompletedEvent` | `response.completed`, `response.incomplete` | `message_stop`, with the counts of `message_start` and the last (cumulative) `message_delta` | Exactly one, as the last event: a stream that ends without it raises `APIStreamInterruptedError`, an event after it raises `APIResponseError`. `response.failed` and `error` events raise an `APIStatusError`. |
| `OtherEvent` | Every other event, refusal deltas included | Every other event, `signature_delta` included | The provider's event is in `raw`. |

`CompletedEvent` carries the fields of the whole response that a stream has:

| Shared field | OpenAI | Anthropic | Lost, or when it raises |
| --- | --- | --- | --- |
| `CompletedEvent.type` | `GenerationEventType.COMPLETED` | `GenerationEventType.COMPLETED` | — |
| `CompletedEvent.id` | `response.id` of the terminal event | `message.id` of `message_start` | — |
| `CompletedEvent.provider` | `Provider.OPENAI` | `Provider.ANTHROPIC` | The adapter's `provider`, as for `GenerationResponse.provider`. |
| `CompletedEvent.model` | `response.model` of the terminal event | `message.model` of `message_start` | A stream without `message_start` raises `APIResponseError`. |
| `CompletedEvent.text` | As `GenerationResponse.text`, from the event's `response` | The `text_delta`s joined | `None` without text. |
| `CompletedEvent.usage` | As `GenerationResponse.usage`, from the event's `response` | As `GenerationResponse.usage` | — |
| `CompletedEvent.incomplete_reason` | As `GenerationResponse.incomplete_reason` | As `GenerationResponse.incomplete_reason`, from the last `message_delta` | — |
| `CompletedEvent.raw_incomplete_reason` | As `GenerationResponse.raw_incomplete_reason` | As `GenerationResponse.raw_incomplete_reason` | — |
| `CompletedEvent.refusal` | As `GenerationResponse.refusal` | As `GenerationResponse.refusal`, from `message_delta.delta.stop_details` | — |
| `CompletedEvent.reasoning` | As `GenerationResponse.reasoning`, from the event's `response` | The `thinking_delta`s joined | `None` without a summary. |
| `CompletedEvent.raw` | The terminal event | `message_stop` | — |

## Errors

Every case below is raised before the request is sent (requests) or before the shared response is returned (responses). `RequestValidationError` is a `ValueError` with `provider` and the shared `field`; a pydantic error behind it is its `__cause__`. Errors of `GenerationRequest` itself stay pydantic `ValidationError`s, for every provider.

| Case | Provider | Shared field | Error |
| --- | --- | --- | --- |
| `empty-turn-list` | any | `input` | `ValidationError` |
| `zero-max-output-tokens` | any | `max_output_tokens` | `ValidationError` |
| `unknown-request-field` | any | — | `ValidationError` |
| `openai-last-assistant-turn` | OpenAI | `input` | `RequestValidationError` |
| `anthropic-last-assistant-turn` | Anthropic | `input` | `RequestValidationError` |
| `openai-max-output-tokens-below-16` | OpenAI | `max_output_tokens` | `RequestValidationError` |
| `openai-too-long-end-user-id` | OpenAI | `end_user_id` | `RequestValidationError` |
| `anthropic-too-long-end-user-id` | Anthropic | `end_user_id` | `RequestValidationError` |
| `openai-failed-response` | OpenAI | — | `APIStatusError` |
| `openai-unfinished-response` | OpenAI | — | `APIResponseError` |
| `anthropic-message-without-stop-reason` | Anthropic | — | `APIResponseError` |
| `stream-without-completion` | any | — | `APIStreamInterruptedError` |
| `turn-from-response-without-text` | any | — | `ValueError` |

## Provider options

Not shared, but reachable through `client.generation`: `create(..., provider_options={...})` and `stream(...)` send a provider's own fields with that call. Each key is a field of the provider's method with its type, and none is a field the shared request fills, which `tests/providers/test_provider_options.py` checks.

| Provider | Options type | Keys |
|---|---|---|
| OpenAI | `OpenAIGenerationOptions` | `metadata`, `reasoning` (`summary`, `context`, `mode`; its `effort` is the shared field, and the two are merged), `store`, `previous_response_id`, `temperature`, `top_p` |
| Anthropic | `AnthropicGenerationOptions` | `stop_sequences`, `thinking` |

## Not unified

These are reached only through the provider's own methods and types, or, where listed under [Provider options](#provider-options), through `provider_options`:

- sampling parameters (`temperature`, `top_p`, `top_k`): Anthropic API version `2026-10-21` removed them; OpenAI's are provider options;
- system or developer turns inside `input` (use `instructions`), images, audio and files;
- tools, tool calls and tool results, server tools;
- reasoning and thinking settings and their content;
- structured output (JSON schemas);
- continuing a stored response (`previous_response_id`, `store`) and stop sequences, which are provider options;
- background responses, cancellation and polling.

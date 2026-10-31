# Unification: what the shared types promise

`client.generation` speaks shared types (`GenerationRequest`, `GenerationResponse`, `GenerationEvent`) for every provider. This page lists, field by field, what each shared field becomes for each provider, what is lost on the way, and when a request or a response raises instead.

Two rules decide what is shared:

- A shared type holds only what every provider has. Provider-only data stays in the provider's own types and is reached through `raw`, which always holds the provider's whole response or event.
- Nothing is dropped silently. A shared request that a provider cannot take raises `RequestValidationError` before anything is sent.

`tests/docs/test_unification.py` checks this page against the code: every field of the shared types has a row here, and every case in [Errors](#errors) raises the error it names.

## Request

`GenerationRequest` maps to OpenAI's `POST /responses` (`CreateResponse`, `StreamResponse`) and Anthropic's `POST /messages` (`CreateMessage`, `StreamMessage`).

| Shared field | OpenAI | Anthropic | Lost, or when it raises |
| --- | --- | --- | --- |
| `GenerationRequest.model` | `model` | `model` | Sent as given; the provider rejects an unknown model with an `APIStatusError`. |
| `GenerationRequest.input` | `input`: a text as is; a list of turns as `EasyInputMessage` items (`role`, `content`) | `messages`: a text as one `user` turn; a list of turns as they are | A list must not be empty and must end with a `user` turn: Anthropic rejects an assistant prefill on current models, and OpenAI gives a last assistant turn another meaning. Images, files, tool calls and results are not shared. |
| `GenerationRequest.instructions` | `instructions` | `system` | System and developer turns are not shared: instructions go here. |
| `GenerationRequest.max_output_tokens` | `max_output_tokens`, at least 16, reasoning included; without it, the model's limit | `max_tokens`, thinking included; required, so without it the adapter's `default_max_tokens` (1024) | Must be positive. Below 16 raises for OpenAI. |
| `GenerationRequest.metadata` | `metadata`: up to 16 pairs, keys up to 64 and values up to 512 characters | `metadata.user_id` only, up to 512 characters | A key other than `user_id` raises for Anthropic instead of being dropped; OpenAI's limits raise too. |

## Response

| Shared field | OpenAI | Anthropic | Lost, or when it raises |
| --- | --- | --- | --- |
| `GenerationResponse.id` | `id` | `id` | — |
| `GenerationResponse.provider` | `Provider.OPENAI` | `Provider.ANTHROPIC` | — |
| `GenerationResponse.model` | `model` | `model` | — |
| `GenerationResponse.text` | The `output_text` parts of the `message` items, joined (as the SDKs' `output_text`) | The `text` blocks, joined | `None` without text. Annotations, citations, reasoning, thinking and tool calls stay in `raw`. |
| `GenerationResponse.usage` | `usage`, see [Token usage](#token-usage) | `usage`, see [Token usage](#token-usage) | `None` if OpenAI sends no usage. |
| `GenerationResponse.incomplete_reason` | `REFUSAL` if the output has a `refusal` part; else, for `status: "incomplete"`, `MAX_TOKENS` for `max_output_tokens` and `OTHER` for any other reason (`content_filter`, `max_messages`, `steered`, none given) | `MAX_TOKENS` for `stop_reason: "max_tokens"`, `REFUSAL` for `"refusal"`, `OTHER` for any other early stop (`model_context_window_exceeded`, `pause_turn`, `tool_use`, a new reason); `None` for `end_turn` and `stop_sequence` | Only the values both providers have are shared; the provider's reason is in `raw_incomplete_reason`. A `failed` OpenAI response raises an `APIStatusError`, any status other than `completed` or `incomplete` raises `APIResponseError`, and an Anthropic message without a `stop_reason` raises `APIResponseError`. |
| `GenerationResponse.raw_incomplete_reason` | `incomplete_details.reason` of an incomplete response | `stop_reason` of an early stop | `None` when the provider gives no reason, such as a completed OpenAI response with a refusal part. |
| `GenerationResponse.refusal` | The `refusal` parts, joined | `stop_details.explanation`, else `stop_details.category`, of a `refusal` stop | A refusal is an answer, not an error, as in both SDKs. `None` if the provider gives no explanation. |
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
| `CompletedEvent` | `response.completed`, `response.incomplete` | `message_stop`, with the counts of `message_start` and the last (cumulative) `message_delta` | Exactly one, as the last event: a stream that ends without it raises `APIStreamInterruptedError`, an event after it raises `APIResponseError`. `response.failed` and `error` events raise an `APIStatusError`. |
| `OtherEvent` | Every other event, refusal deltas included | Every other event | The provider's event is in `raw`. |

`CompletedEvent` carries the fields of the whole response that a stream has:

| Shared field | OpenAI | Anthropic | Lost, or when it raises |
| --- | --- | --- | --- |
| `CompletedEvent.type` | `GenerationEventType.COMPLETED` | `GenerationEventType.COMPLETED` | — |
| `CompletedEvent.usage` | As `GenerationResponse.usage`, from the event's `response` | As `GenerationResponse.usage` | — |
| `CompletedEvent.incomplete_reason` | As `GenerationResponse.incomplete_reason` | As `GenerationResponse.incomplete_reason`, from the last `message_delta` | — |
| `CompletedEvent.raw_incomplete_reason` | As `GenerationResponse.raw_incomplete_reason` | As `GenerationResponse.raw_incomplete_reason` | — |
| `CompletedEvent.refusal` | As `GenerationResponse.refusal` | As `GenerationResponse.refusal`, from `message_delta.delta.stop_details` | — |
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
| `openai-too-many-metadata-pairs` | OpenAI | `metadata` | `RequestValidationError` |
| `openai-too-long-metadata-key` | OpenAI | `metadata` | `RequestValidationError` |
| `anthropic-metadata-key-other-than-user-id` | Anthropic | `metadata` | `RequestValidationError` |
| `anthropic-too-long-user-id` | Anthropic | `metadata` | `RequestValidationError` |
| `openai-failed-response` | OpenAI | — | `APIStatusError` |
| `openai-unfinished-response` | OpenAI | — | `APIResponseError` |
| `anthropic-message-without-stop-reason` | Anthropic | — | `APIResponseError` |
| `stream-without-completion` | any | — | `APIStreamInterruptedError` |
| `turn-from-response-without-text` | any | — | `ValueError` |

## Not unified

These are reached only through the provider's own methods and types:

- sampling parameters (`temperature`, `top_p`, `top_k`): Anthropic API version `2026-10-21` removed them;
- system or developer turns inside `input` (use `instructions`), images, audio and files;
- tools, tool calls and tool results, server tools;
- reasoning and thinking settings and their content;
- structured output (JSON schemas);
- continuing a stored response (`previous_response_id`, `store`) and stop sequences;
- background responses, cancellation and polling.

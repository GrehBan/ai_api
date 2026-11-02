# Resources and use cases

Methods are specific to one provider. Resources and use cases are the layer above them that offers one API for all providers, in shared types:

- **Shared requests** (`aiollm.types.requests`): `BaseRequest`, which forbids unknown fields, and `GenerationRequest` (`model`, `input`, a text or a list of `Turn`s, neither of them empty, `instructions`, `max_output_tokens`, `metadata`). `instructions` are given apart from the input, like a system prompt. It has no sampling parameters such as `temperature`: Anthropic removed them in API version `2026-10-21`, so they are not common to all providers.
- **Shared responses** (`aiollm.types.responses`): `GenerationResponse[RawT]` with `id`, `provider`, `model`, `text`, `usage` (`TokenUsage`: input, output and total tokens, with the cached and reasoning tokens among them), `incomplete_reason` (why the provider stopped early, an `IncompleteReason`; `None` if it finished), `raw_incomplete_reason`, `refusal` and the provider's own response in `raw`, so nothing the shared type lacks is lost.
- **Shared events**: `GenerationEvent[RawT]`, a discriminated union of `TextDeltaEvent`, `CompletedEvent` and `OtherEvent`. `CompletedEvent`, the stream's final event, carries `id`, `provider`, `model`, the whole `text`, `usage`, `incomplete_reason`, `raw_incomplete_reason` and `refusal` too. Their `type` is a `GenerationEventType`. A failure is an exception at every level, never an event.
- **Adapters** (`aiollm.adapters`): convert models and nothing else. `BaseAdapter` builds the provider's method from a shared request (`to_method`) and turns its result into a shared response (`to_response`); its `provider` names the provider in the results and in the errors of the requests it refuses. `BaseStreamAdapter` does the same for streams (`to_stream_method`, `to_events`), and `BaseCompositeAdapter` does both.
- **Use cases** (`aiollm.resources`): only orchestrate. `AdapterUseCase(session, adapter)` sends what the adapter builds and returns what it converts; `AdapterStreamUseCase` and `AdapterCompositeUseCase` do the same for streams and for both. Given `terminal=` (an event matcher, such as `GENERATION_TERMINAL` from `aiollm.types.responses`, which matches `CompletedEvent`), they check that exactly one event matches it, as the last one: a stream without it raises `APIStreamInterruptedError` (or `RuntimeError` when the provider method's `__stream_terminal__` proved the stream complete), an event after it `APIResponseError`.
- **Resources**: `GenerationResource(session, adapter)` is the shared resource of text generation, the same class for every provider: `create(...)` and `stream(...)`, with keyword arguments or a request, typed by the adapter's raw types; `stream` yields a `GenerationStream` with `text()` and `final()`. `BaseResource` groups the use cases of any other area.

```python title="resources.py"
from collections.abc import AsyncGenerator, AsyncIterator

from aiollm import (
    BaseClient,
    CompletedEvent,
    GenerationEvent,
    GenerationRequest,
    GenerationResponse,
    Provider,
    TextDeltaEvent,
)
from aiollm.adapters import BaseCompositeAdapter
from aiollm.resources import GenerationResource
from aiollm.session import ISession


class CompletionAdapter(
    BaseCompositeAdapter[
        GenerationRequest,
        Completion,                       # what CreateCompletion returns
        GenerationResponse[Completion],
        MessageEvent,                     # what StreamMessage yields
        GenerationEvent[MessageEvent],
    ]
):
    provider = Provider.GENERIC           # a provider the library does not list

    def to_method(self, request: GenerationRequest) -> CreateCompletion:
        return CreateCompletion(
            model=request.model,
            messages=[Message(role="user", content=request.input)],
        )

    def to_response(
        self,
        raw: Completion,
        request: GenerationRequest,
    ) -> GenerationResponse[Completion]:
        return GenerationResponse[Completion](
            id=raw.id,
            provider=self.provider,
            model=request.model,          # the API does not send it back
            text=raw.text,
            raw=raw,
        )

    def to_stream_method(self, request: GenerationRequest) -> StreamMessage:
        return StreamMessage(prompt=request.input)

    async def to_events(
        self,
        items: AsyncIterator[MessageEvent],
        request: GenerationRequest,
    ) -> AsyncGenerator[GenerationEvent[MessageEvent]]:
        async for event in items:
            if isinstance(event, TextDelta):
                yield TextDeltaEvent[MessageEvent](text=event.text, raw=event)
            else:
                yield CompletedEvent[MessageEvent](
                    provider=self.provider, model=request.model, raw=event
                )


class ExampleClient(BaseClient[ISession]):
    def __init__(self, session: ISession) -> None:
        super().__init__(session)
        self.generation = GenerationResource(session, CompletionAdapter())
```

`await client.generation.create(model=..., input=...)` returns a `GenerationResponse[Completion]`, and `async with client.generation.stream(model=..., input=...) as events` yields a `GenerationStream[MessageEvent]` of `GenerationEvent[MessageEvent]`s. Another provider's client builds the same `GenerationResource` with its own adapter, so the calling code does not change.

- **Why three parts.** The session moves bytes, the adapter converts models, and the use case connects them. An adapter has no session and does no I/O, so it is tested with plain objects; a use case has no provider knowledge, so one class serves every provider.
- **Grouped by mode, not by direction.** One adapter class holds both halves of a buffered call, so a type checker sees at the class itself whether the method it builds returns the type its `to_response` expects.
- **The request reaches the conversions.** `to_response` and `to_events` get the shared request too, for what the provider does not send back, such as the model or an output schema.
- **`to_events` sees the whole stream.** It may keep state, drop items and turn one item into several, for example collect deltas into a final event. The use case runs it through `transform_stream` (`aiollm.response`), which closes the converted stream first and the provider's stream after it, also on `break` and errors. `map_stream` is its one-item-to-one-item case.
- **Resources hold use cases.** `GenerationResource` runs an `AdapterCompositeUseCase`; a resource of another area builds `AdapterUseCase`s the same way, and `IGenerationResource` is its protocol.
- **Calls that are not one request and one response,** such as pagination, several dependent requests or polling, subclass `BaseUseCase`, `BaseStreamUseCase` or `BaseCompositeUseCase` directly and use adapters inside.
- **The session type is optional.** The last type parameter, `SessionT`, defaults to `ISession`. Give a concrete session, such as `BaseResource[AiohttpSession]`, to reach its settings through `self.session` without a cast.
- **`RawT` has no bound.** With a `BaseAPIType` bound, an unparametrized `GenerationResponse` would turn a dict in `raw` into an empty model, because unknown fields are ignored.

`OpenAIClient.generation` and `AnthropicClient.generation` are `GenerationResource`s with `OpenAIGenerationAdapter` and `AnthropicGenerationAdapter` (see [Shared generation](generation.md#shared-generation)).

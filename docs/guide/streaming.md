# Streaming responses

Subclass `BaseStreamMethod`. The generic argument is the type of one streamed item: `bytes`, `str`, or an event type (see [Events](#events-sse-and-ndjson)).

```python title="streaming.py"
from ai_api import BaseStreamMethod


class Download(BaseStreamMethod[bytes]):      # raw chunks
    __request_path__ = "/files/content"
    __request_method__ = "GET"
    __request_query__ = {"file_id"}
    __retryable__ = True

    file_id: str


class Generate(BaseStreamMethod[str]):        # decoded text pieces
    __request_path__ = "/generate"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON

    prompt: str


async with session.stream(Download(file_id="f-1")) as stream:
    await stream.save("result.bin")            # only on Stream[bytes]

async with session.stream(Generate(prompt="Hi")) as stream:
    async for piece in stream:                 # piece: str
        print(piece, end="")
```

- **Errors before the stream starts.** An error status raises the matching `APIStatusError` as you enter the `async with` block.
- **Retries.** Only opening the stream is retried. A failure while iterating reaches your code, so it never receives duplicated data.
- **Cleanup.** The connection is released when the block exits, including on `break` or an exception. Iterating the stream after the block, even through an iterator taken inside it, raises `RuntimeError`, instead of quietly yielding nothing from a closed connection.
- **Text decoding.** It is incremental, so a multi-byte character split between two chunks still decodes correctly. A charset that is not a text encoding (`base64`, `zlib`) raises `APIResponseError`.
- **Saving to a file.** `Stream.save(path)` writes chunks as they arrive, without holding the body in memory, and returns the number of bytes written. It opens, writes and closes the file in a worker thread; a cancelled save closes the file once the write running there is over.
- **Timeouts.** Streams have no total time limit, only 600 s between two chunks, so a long generation is not cut off. See [Timeouts](sessions.md#timeouts).
- **Type safety.** Passing a stream method to `send_request`, or a buffered method to `stream`, is a type error.
- **Format and item type.** `BYTES` streams yield `bytes` and `TEXT` streams yield `str`, so an explicit `__response_format__` that contradicts the generic argument, such as `BaseStreamMethod[str]` with `BYTES`, raises `TypeError` at class definition, unless the method sets its own `__stream_parser__`.

## Events: SSE and NDJSON

For a stream of events, set `__response_format__` to `ResponseFormat.SSE` (`text/event-stream`) or `ResponseFormat.NDJSON` (one JSON value per line). The generic argument can be a model or a discriminated union of models, and each event is validated into it.

```python title="events.py"
from typing import Annotated, Literal

from pydantic import Field

from ai_api import ResponseFormat, ServerSentEvent


class TextDelta(BaseAPIType):
    type: Literal["text_delta"]
    text: str


class MessageStop(BaseAPIType):
    type: Literal["message_stop"]


type MessageEvent = Annotated[
    TextDelta | MessageStop,
    Field(discriminator="type"),
]


class StreamMessage(BaseStreamMethod[MessageEvent]):
    __request_path__ = "/messages"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON
    __response_format__ = ResponseFormat.SSE

    prompt: str
    stream: bool = True


class RawEvents(BaseStreamMethod[ServerSentEvent]):  # SSE, not validated
    __request_path__ = "/events"
    __request_method__ = "GET"


async with session.stream(StreamMessage(prompt="Hi")) as events:
    async for event in events:                 # TextDelta | MessageStop
        if isinstance(event, TextDelta):
            print(event.text, end="")
```

- **Content type.** An SSE response must have `Content-Type: text/event-stream`; anything else (such as a proxy's HTML login page with status 200) raises `APIResponseError` instead of giving an empty stream. For a server that sends another type, use `StreamResponseParser(require_event_stream=False)` as the method's `__stream_parser__`.
- **Format.** It must be set for models and unions: the item type alone doesn't tell SSE from NDJSON. Without it the class definition raises `TypeError`, unless the method sets its own `__stream_parser__`. `ServerSentEvent` items imply SSE.
- **Parsing.** SSE follows the HTML standard: multi-line `data`, comments, `event`, `id` and `retry`, and `\n`, `\r\n` or `\r` line endings. An event split across chunks is held back until it is complete, and an unfinished event at the end of the body is dropped. NDJSON skips blank lines.
- **Validation.** Each event is validated through a `TypeAdapter` built once, when the class is defined, from the generic argument as written. That is why discriminated unions and `type` aliases work as item types. An event that doesn't match raises `APIResponseError`.
- **End of stream.** By the HTTP and SSE standards, the end of the body is a normal end, so a stream cut between two events looks complete. Only the protocol's own last event proves that it is. Declare it as `__stream_terminal__` and the parser checks it: a body that ends without it raises `APIStreamInterruptedError`, and an item after it raises `APIResponseError` (the parser reads the body to its end to see that), an error event included: once the terminal has come, the stream is complete, and anything after it breaks the protocol. Matchers live in `ai_api.types`: `EventTypes("message_stop")` matches a `type`, `FieldValues(name, *values)` any field, `DONE` the `data: [DONE]` line, and `a | b` either. They compare values as JSON tells them apart (`FieldValues("done", True)` does not match `{"done": 1}`), read a model's fields but never its properties, and are equal when they match the same items (`EventTypes("x") == FieldValues("type", "x")`). Without a declared terminal (the default), `data: [DONE]` ends the stream and nothing is checked. Only SSE and NDJSON have events to check, so a terminal on a `BYTES` or `TEXT` stream raises `TypeError` at class definition, unless the method sets its own `__stream_parser__`. `[DONE]` and SSE comments are never items. The provider methods declare theirs: `response.completed`, `response.incomplete` or `response.failed` for OpenAI's `StreamResponse`, `message_stop` for Anthropic's `StreamMessage`.
- **Errors inside the stream.** An SSE event whose `event:` field is `error` (Anthropic) raises, whatever its data, the `APIStatusError` built by the session's error resolver, and it is never transient: the generation already ran. So does decoded data that the parser's `error_events` matcher accepts, also for a `ServerSentEvent` item whose data is JSON (data that is not JSON is yielded as it is). The default, `EventTypes("error") | TopLevelError()`, accepts JSON whose `type` is `"error"`, or that has no `type` and a top-level `error` that is `true` or a non-empty object or string (OpenAI, Ollama); an event with another `type` may carry an `error` field as data (a failed tool call, for example). For other rules, give the method a `__stream_parser__ = StreamResponseParser(error_events=...)`, as OpenAI's `StreamResponse` does to raise `response.failed`. When the error event also fits the item type, the validated event is the error's `raw`.
- **Raw events.** `BaseStreamMethod[ServerSentEvent]` yields events as they are, with their `data`, `event`, `id` and `retry`. `[DONE]` and error events are handled the same way.
- **Limits.** `StreamResponseParser(max_line_size=...)` limits one SSE or NDJSON line and `max_event_size=...` the data of one SSE event, in characters; a longer one raises `APIResponseError`. Both are on by default, 32 Mi characters each (`MAX_LINE_SIZE`, `MAX_EVENT_SIZE`), since a server that never ends a line would otherwise make the stream grow in memory without bound. They are large because a normal event can be: OpenAI sends a generated image as base64 in one line, and the final event repeats the whole response. `None` lifts a limit; that is your decision to make. A `retry` field with more digits than Python turns into a number (4300 by default) raises `APIResponseError` as well.

## Interrupted streams

A connection can break at any point of a long generation. What happens depends on when it breaks:

| When | What you get | Retried |
|---|---|---|
| Before the response arrives: connection refused, DNS, a TLS handshake that breaks off, connect timeout, a certificate that fails verification | `APIConnectionError`, `APITimeoutError` or (for the certificate) `APITLSError` on entering the block | yes, if the method is retryable and the error is transient |
| The server answers with an error status | the `APIStatusError` for it, on entering the block | yes, if the error is transient |
| The connection drops after some items | `APIStreamInterruptedError` from `async for`, caused by the `APIConnectionError` | no |
| No data for 600 s (`sock_read`) | `APIStreamInterruptedError` from `async for`, caused by the `APITimeoutError` | no |
| The body ends before the method's `__stream_terminal__` | `APIStreamInterruptedError` from `async for`, without a cause | no |
| The server sends an error event | the `APIStatusError` for it, from `async for` | no |

Items you received before the failure stay yours (`items_received` counts them), and the connection is released as the block exits, also when the task is cancelled. `APIStreamInterruptedError` is never `transient`: trying again means a new request, and the library does not do that on its own.

A body that simply stops between two events, without an error, is the hard case: HTTP/1.1 ends a body without `Content-Length` or chunked encoding by closing the connection (RFC 9112, section 6.3), so such a cut looks like a normal end. Only the terminal event tells a complete stream from a cut one, which is why the provider methods declare it and `generation` checks its own `CompletedEvent`.

**Why a broken stream is not retried automatically.** Opening the stream again would send the request again. For a generation endpoint that is a new, separately billed generation with different output. The library cannot know what your code has already done with the items it received, such as printing them or forwarding them to a user, so it cannot splice the two streams into one without duplicating or losing text.

**Why the SSE resume protocol is not used.** SSE has its own way to continue a stream. The server marks events with `id:`, the client reconnects after `retry:` milliseconds and sends the last id in the `Last-Event-ID` header, and the server replays what came after it. The library parses `id` and `retry`; you can read them on `ServerSentEvent` items. It does not reconnect, for three reasons:

- **It only works if the server keeps the events and replays them.** AI providers' generation streams generally don't. Where a provider supports resuming, it does so with its own parameters (OpenAI's background responses, for example, are resumed from an event's `sequence_number` with `starting_after`). That is a provider method, not a transport feature.
- **Resuming is a policy, not a request.** It needs a number of attempts, a delay, the last id and a way to continue the same `Stream`, and each of these depends on the server. A single call can already send `Last-Event-ID` with `headers=`; deciding when to send it again is the caller's part for now.
- **Silent reconnection hides failures.** A client that reconnects on its own can loop for a long time against a broken server, while the caller sees a stream that is merely slow.

**What to do instead.** Decide in your code what a broken stream means. For short answers the simplest choice is to start over and drop the partial text:

```python title="restart.py"
from ai_api import (
    APIConnectionError,
    APIStreamInterruptedError,
    APITimeoutError,
)


async def generate_text(prompt: str, attempts: int = 3) -> str:
    for attempt in range(1, attempts + 1):
        pieces: list[str] = []

        try:
            async with session.stream(Generate(prompt=prompt)) as stream:
                async for piece in stream:
                    pieces.append(piece)
        except (
            APIConnectionError,  # opening failed
            APITimeoutError,
            APIStreamInterruptedError,  # broke after it was opened
        ):
            if attempt == attempts:
                raise
            continue                        # drop the partial text, ask again

        return "".join(pieces)

    raise ValueError("attempts must be at least 1")
```

For a provider that can resume, keep the position from the last event you received (its `id`, or a sequence number in its data) and call the provider's resume method with it. For a server that replays SSE events, open the stream again with the last id: `session.stream(method, headers={"Last-Event-ID": last_id})`.

An opt-in reconnection with `Last-Event-ID` for servers that support it is on the [roadmap](design.md#limitations-and-roadmap): limited attempts, the `retry` delay, and the same `Stream` object continuing with the replayed events.

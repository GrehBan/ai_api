# Sessions

## Session options

`AiohttpSession` takes these arguments; all after `error_resolver` are keyword-only:

| Argument | Default | Meaning |
|---|---|---|
| `api` | required | Builds request URLs (`IAPI`) |
| `auth` | required | Produces the authentication header (`IAuth`): `BearerAuth` for `Authorization: Bearer ...`, `APIKeyAuth` for a key in its own header (`X-Api-Key` by default; header names are case-insensitive) |
| `error_resolver` | required | Maps error responses to exceptions (`IErrorResolver`) |
| `headers` | `None` | Extra default headers, as a mapping or `Headers`; an `IHeaders` object is kept as it is |
| `json_loads` / `json_dumps` | `orjson` | JSON codec |
| `max_connections` | `100` | Maximum number of simultaneous connections |
| `timeout` | 600 s total, 30 s connect | Timeout of buffered requests |
| `stream_timeout` | no total limit, 30 s connect, 600 s between chunks | Timeout of streamed requests |
| `max_retry_after` | `60` | Longest `Retry-After` to wait for, in seconds |
| `retry_deadline` | `900` | Seconds after the first attempt after which no retry starts; `None` for no limit |
| `max_body_size` | `None` | Largest buffered response or error body, in bytes; streamed bodies are not limited |
| `trust_env` | `False` | Use proxy settings and `.netrc` from the environment |
| `follow_redirects` | `False` | Follow redirects within the origin of the request |
| `loop` | `None` | Event loop the session is bound to; `None` binds the loop of its first use |

The connection pool is created lazily, on the first request or on `async with`. It uses the `certifi` CA bundle, caches DNS answers for an hour and sends `User-Agent: Python/3.x ai_api/<version>`.

## Lifecycle

Close a session when you are done with it, ideally with `async with session:`. Closing releases the pooled connections and waits a quarter of a second so aiohttp can shut down the TLS connections cleanly; without it, Python warns about unclosed sessions at exit. A closed session opens a new pool if it is used again.

One session can serve many concurrent requests: `asyncio.gather` over `send_request` calls shares its pool, up to `max_connections` at a time.

## Authentication

The authentication header is produced on every request, by calling `auth.encode()`, so an `IAuth` that rotates its token takes effect at once:

```python title="auth.py"
class RotatingAuth:
    """Sends the current token of a token store."""

    header_name = "Authorization"

    def __init__(self, token: str) -> None:
        self.token = token

    def encode(self) -> str:      # called for every request
        return f"Bearer {self.token}"
```

Any object with a `header_name` and an `encode()` method is an `IAuth`; it does not have to inherit from anything. `BearerAuth` and `APIKeyAuth` reject an empty token and whitespace, which would otherwise produce a broken or injected header, and keep the token out of `repr`, so it does not end up in logs or tracebacks.

## Default headers

`headers=Headers({...})` adds headers to every request, for example an API version header:

```text
AiohttpSession(api, auth, resolver, headers=Headers({"anthropic-version": "2023-06-01"}))
```

They are read on every request, so a header added later with `session.headers.add(...)` is sent from the next request on.

A request gets headers from four places. Names are case-insensitive, and a later place replaces a header of an earlier one:

1. the session's `headers`, with the library's `User-Agent` over them;
2. the method's `__request_headers__`;
3. the call's `headers=`;
4. the authentication header, which neither a method nor a call can replace.

## Per-request headers and timeout

`send_request` and `stream`, on a session or a client, take two keyword-only arguments for one call:

```python title="call_options.py"
class BetaCompletion(CreateCompletion):
    __request_headers__ = {"x-beta-features": "long-context"}   # every request of the method


async def complete(prompt: str, key: str) -> Completion:
    return await session.send_request(
        BetaCompletion(
            model="example-1",
            messages=[Message(role="user", content=prompt)],
        ),
        headers={"Idempotency-Key": key},   # this call only, every attempt
        timeout=30,                         # seconds for the whole request
    )
```

- **`headers=`** is a mapping or `Headers`. It is sent with every attempt of the call, retries included, and is not kept for later calls: an idempotency key, `Last-Event-ID`, a trace id.
- **`timeout=`** is a number of seconds. It changes one limit of the session's timeout for this call and keeps the others; see [Timeouts](#timeouts).

Both stay out of the method, because they describe one call, not the API endpoint: the same `CreateCompletion` can be sent with different keys and limits.

## JSON codec

`json_loads` and `json_dumps` replace the default `orjson` codec, for example to use the standard `json` module or to encode custom types. `json_dumps` may return `str` or `bytes`. NaN and infinity are rejected before encoding, because orjson would silently write them as `null`.

With a custom `json_loads`, response models are validated in Python mode instead of directly from JSON; see [Result type and response format](methods.md#result-type-and-response-format).

## Timeouts

Timeouts are `aiohttp.ClientTimeout` objects, set per session:

- **`timeout`**, for buffered requests: 600 s in total, 30 s to connect. It is longer than aiohttp's default of 5 minutes, because a long model generation that is not streamed exceeds that.
- **`stream_timeout`**, for streams: no total limit, 30 s to connect and 600 s of silence between two chunks. A stream may legitimately run for a long time, but a stream that sends nothing for ten minutes is dead.

`timeout=` on a call replaces one of these limits, in seconds, and keeps the connect limits:

- for `send_request`, the total of `timeout`, so a quick call can fail sooner and a slow one can wait longer than 600 s;
- for `stream`, the silence between two chunks (`sock_read`) of `stream_timeout`. A stream still has no total limit, so a long generation is not cut off however small the number is.

A timeout raises `APITimeoutError`, which is transient. When the method is retryable, a buffered request that timed out is sent again, so keep `__retryable__` off for requests that must not run twice.

## Body size limits

`max_body_size` limits buffered response bodies and error bodies, in bytes; a larger body raises `APIResponseError` without being read to the end. It is off by default, since a legitimate response such as a file download can be large. Set it when the server is not trusted. Streamed bodies are not limited by it, because they are not held in memory; limit the lines and events of SSE and NDJSON with the stream parser's [limits](streaming.md#events-sse-and-ndjson) instead.

## Redirects

Redirects are not followed by default: API providers do not redirect, and aiohttp strips only `Authorization` on a redirect to another host, so a key in `x-api-key` or in the default headers would leak there. A redirect raises `APIRedirectError`. With `follow_redirects=True`, redirects within the same origin (scheme, host and port) are followed; one to another origin still raises `APIRedirectError` before anything is sent there. A redirect that needs the request body again (307, 308) after an upload stream was consumed raises `APIRequestError`.

## Proxies and the environment

`trust_env` is off by default, so `HTTP(S)_PROXY` variables and `.netrc` credentials are ignored. A proxy picked up from the environment would silently route API keys through another host, and `.netrc` could add credentials nobody asked for. Turn it on with `trust_env=True` when you do want the environment's proxy.

## Event loop

A session is bound to one event loop: the one passed as `loop=`, or else the loop of its first request (or `async with`). It can be created anywhere, even at module level, but using or closing it from another loop (such as a second `asyncio.run`) raises `RuntimeError`. The `session.loop` property only reports the bound loop (`None` before first use) and never binds one. Close the session before its loop ends, ideally with `async with`.

aiohttp connections belong to the loop that created them, and using them from another loop fails with unclear errors such as `Event loop is closed` or a hang. Binding the session gives a clear error at the first wrong call instead.

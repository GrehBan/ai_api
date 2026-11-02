# Errors and retries

```text
AIAPIError
├── APIStatusError              the API reported a failure
│   ├── BadRequestError         400
│   ├── AuthenticationError     401
│   ├── BillingError            402
│   ├── PermissionDeniedError   403
│   ├── NotFoundError           404
│   ├── RequestTimeoutError     408
│   ├── ConflictError           409
│   ├── ContentTooLargeError    413
│   ├── UnprocessableEntityError 422
│   ├── RateLimitError          429
│   ├── QuotaExceededError      no status of its own: a quota error
│   ├── InternalServerError     500
│   ├── BadGatewayError         502
│   ├── ServiceUnavailableError 503
│   └── GatewayTimeoutError     504
├── APITransportError           no valid HTTP response
│   ├── APIConnectionError      also a body cut off by a lost connection
│   ├── APITimeoutError
│   ├── APITLSError
│   ├── APIRedirectError        redirect not followed
│   ├── APIRequestError         failed upload source, invalid header or URL
│   └── APIStreamInterruptedError  an open stream broke or ended early
└── APIResponseError            invalid HTTP, body too large, not decodable or valid
```

The three branches answer the question "what came back?":

- **`APIStatusError`**: the API reported a failure. Usually that is an error status (400 or more); a status without its own class is raised as `APIStatusError` itself. It is also a failure reported in-band after a successful status, such as a stream's error event or a result that says it failed; its `status` is then that successful one (see [Error resolvers](#error-resolvers)).
- **`APITransportError`**: no usable answer came back, because the request failed on the way.
- **`APIResponseError`**: an answer came back, but it cannot be used. It is not valid HTTP, it is larger than `max_body_size`, or it does not decode or validate.

An `APIStatusError` carries:

- `status`, `code`, `error_type` and `request_id`;
- `provider`, from the error resolver;
- `retry_after`, from the `Retry-After` header;
- `transient`;
- `raw`, the parsed object that reported the failure, when there is one: OpenAI's `Response` with `status: "failed"`, or a stream's error event when it fits the item type.

A status error's message starts with the class's fixed description and ends with `detail: <the API's message>` when the API sent one, so logs have a stable prefix and the provider's own explanation. Transport and response errors carry their own message.

One more library error is raised before anything is sent: `RequestValidationError` (a `ValueError`), for a shared request that a provider cannot take; see [Shared generation](generation.md).

```python title="handling.py"
from aiollm import APIStatusError, APITransportError, RateLimitError


async def ask(prompt: str) -> str | None:
    method = CreateCompletion(
        model="example-1",
        messages=[Message(role="user", content=prompt)],
    )

    try:
        completion = await session.send_request(method)
    except RateLimitError as error:            # retries did not help
        print(f"Rate limited, request {error.request_id}")
        return None
    except APIStatusError as error:
        print(f"API error {error.status}, code {error.code}")
        raise
    except APITransportError as error:
        print(f"Network error, transient: {error.transient}")
        raise

    return completion.text
```

## When a request is retried

A request is retried only when the method sets `__retryable__ = True`, does not carry a one-shot file, and the error's `transient` flag is true. By default that flag is true for:

- statuses 408, 500, 502, 503, 504, and a provider's own transient statuses, which its resolver classifies (Anthropic's 529 "overloaded");
- status 429 when the response has `Retry-After`;
- connection errors, including a body cut off before its end, and timeouts.

It is false for certificate errors, redirects that are not followed, invalid requests and failing upload sources, since repeating them gives the same result, and `APIResponseError` is never retried.

Waits use random exponential backoff of up to 30 s, but never less than `Retry-After`. When the API asks to wait longer than `max_retry_after` (60 s by default), the error is raised at once instead. No retry starts after `retry_deadline` (15 minutes from the first attempt by default), and a wait that would end later is not started; an attempt in progress is limited by its own timeout.

The reasons behind these rules:

- **Retries are opt-in per method.** Most AI endpoints are `POST`s that are not idempotent. A request that timed out may still have run on the server, and sending it again can start a second billed generation. The author of a method knows whether that is acceptable; the library does not. The library's provider methods opt in, as the official SDKs do, and accept the same risk.
- **A 429 is retried only with `Retry-After`.** Without it there is no telling a short rate limit from an exhausted quota, and retrying a quota error only delays the failure. With it, the server has said when to come back.
- **`Retry-After` is capped.** A server that asks for ten minutes would otherwise block the caller silently. Raising the error lets the caller decide, for example to queue the work.
- **The deadline bounds the total time.** Four attempts with long waits and long timeouts could otherwise take far longer than any caller expects.
- **Jitter spreads the retries.** Many clients failing at the same moment don't all come back at the same moment.

## Error resolvers

Error responses are mapped by an error resolver. `BaseErrorResolver` reads:

- the common `{"error": {"code", "type", "message"}}` body shape, and a string `error` (`{"error": "model not found"}`, as Ollama sends) as the message;
- the request id from the `X-Request-Id` or `Request-Id` header, or from a `request_id` field of the body;
- `Retry-After`, in the two forms of RFC 9110: whole seconds in ASCII digits or an HTTP date. Any other form (`1.5`, `-5`, `1e3`) counts as no header, so a negative number cannot mean an immediate retry.

A body that is not JSON gives an error without a code or message, so an HTML error page from a proxy still becomes the right `APIStatusError` for its status.

A failure the API reports after a successful status (a stream's error event, a result that says it failed) is in-band. It is classified by the status that the resolver's `error_statuses` table gives its code, else its type, and it is never transient, whatever its class: the generation ran and failed, and running it again is your choice, as in the official SDKs, which retry by the HTTP status only.

Subclass the resolver to handle a provider's specifics: set `provider` and `error_statuses`, and override `get_error_code` or `classify_status`, which picks the class and transience of a status (the in-band rule stays the base class's):

```python title="resolver.py"
from aiollm import APIStatusError, Provider, QuotaExceededError


class ExampleErrorResolver(BaseErrorResolver):
    provider = Provider.GENERIC                    # Provider is closed
    error_statuses = {"insufficient_quota": 429}   # in a stream, too

    def classify_status(
        self,
        *,
        status: int,
        code: str | None,
        error_type: str | None,
        message: str | None,
        retry_after: float | None,
    ) -> tuple[type[APIStatusError], bool]:
        if status == 429 and code == "insufficient_quota":
            return QuotaExceededError, False      # not worth retrying

        return super().classify_status(
            status=status,
            code=code,
            error_type=error_type,
            message=message,
            retry_after=retry_after,
        )
```

The resolver returns the error instead of raising it, so the session raises it where the traceback is most useful. Error events inside streams go through the same resolver, so one `classify_status` covers both.

Library errors can be pickled and copied, for example to pass them between processes. They are rebuilt from their attributes without calling `__init__` again, since `__init__` composes the message and would add the prefix twice.

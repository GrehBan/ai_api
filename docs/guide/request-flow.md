# How a request flows

A buffered request, `await session.send_request(method)`:

```text
send_request(method)
└── retry loop: only if method.retryable, up to 4 attempts
    └── one attempt
        ├── URL:   api.build_url(method.request_path, method.get_query_map())
        ├── body:  method.get_body_map() → normalized → JSON / form / multipart
        ├── headers: session < method.request_headers < headers= < auth
        ├── send:  the transport sends the request and reads the whole body
        ├── check: status ≥ 400 → error resolver → APIStatusError(transient=…)
        └── parse: method.response_parser → the typed result
```

A streamed request, `async with session.stream(method) as stream`:

```text
stream(method)
├── retry loop around opening only
│   ├── send:  the transport sends the request and leaves the body unread
│   └── check: status ≥ 400 → the error body is read → APIStatusError
├── yield Stream(method.stream_parser.parse(...))   items are parsed as you iterate
└── on exit: the parser and the connection are closed (also on break and errors)
```

Two things follow from this design:

- **The status check is inside the retried attempt.** Whether to retry is decided by the `transient` flag of the error raised there, so the error resolver alone decides which statuses are retried.
- **A success status can still be a failure.** A body or a stream event that reports a failure after a 2xx (OpenAI's `failed` response, an `error` event) raises an `APIStatusError` too, from the parser; such an in-band failure is never transient, so it is not retried. See [Error resolvers](errors.md#error-resolvers).
- **Only opening a stream is retried.** Once your code has received items, the library cannot resend the request without duplicating them. See [Interrupted streams](streaming.md#interrupted-streams).

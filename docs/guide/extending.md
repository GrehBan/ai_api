# Extending

## Adding a provider

A provider is a flat package under `providers/<name>/`, built from the same parts as `providers/openai/`:

1. **`api.py`**: its `IAPI`, usually `BaseAPI("https://api.provider.com", "v1")`, and `PROVIDER: Final = Provider.<NAME>`, the one place the package names its provider. `Provider` is a closed enum, extended only inside the library together with a provider package; code outside the library uses `Provider.GENERIC`.
2. **`resolver.py`**: a `BaseErrorResolver` subclass with `provider = PROVIDER`, the provider's `error_statuses` table (the status each in-band code or type stands for), and `get_error_code` or `classify_status` for its error codes, such as which 429s are quota errors and which statuses are transient.
3. **`client.py`**: a module-level `SessionFactory(AiohttpSession, auth_factory=..., api=..., error_resolver=...)` and a `BaseProviderClient` subclass that sets `provider = PROVIDER`, `session_factory`, `api_key_env`, `auth_headers` (the headers the provider takes the key in) and `adapter_types`, every adapter class it builds (one per shared resource). Its `__init__` types `**session_kwargs`, builds each adapter with `self.adapter(AdapterClass, ...)` (type checkers check its arguments as a call of the class; a class missing from `adapter_types` raises `TypeError`) and its resources after `super().__init__()`. The base class reads the token, the ready session or the environment variable, builds the session and warns about a given session that does not fit. The client, the factory's resolver and every adapter must name the same provider: a mismatch raises `TypeError` when the class is defined.
4. **`types/`, `methods/` and `adapters/`** subpackages with its first endpoints: response models, method classes, and adapters that map them to the shared types (an adapter sets `provider = PROVIDER`; a concrete adapter without it raises `TypeError`); its resources build `AdapterUseCase`s from those adapters. An optional field of a shared request that was not given is `MISSING`: pass it to the method as it is, and test it with `is MISSING`, never by truth (`bool(MISSING)` is `True`). Write the method's fields after the provider's request schema (see [Optional fields](methods.md#optional-fields-missing-and-none)), and add the endpoint to the wire-contract tests, which check them.

The package `__init__.py` re-exports the client, which makes it the one `__init__` allowed to import a transport.

## Writing a transport

`BaseSession` holds everything that doesn't depend on the HTTP client: retries, error mapping, parsing and stream cleanup. A transport subclasses it and implements three methods:

- **`_execute(method, *, headers=None, timeout=None) -> BufferedResponse`**: send the request and read the whole body.
- **`_open_stream(method, *, headers=None, timeout=None) -> IStreamResponse`**: send the request and leave the body unread; the response has `status`, `headers` and `charset`, and offers `iter_chunks()`, `read()` and `aclose()`.
- **`close()`**: release the connections.

Both request methods return error statuses as responses, without raising, so the status check stays in `BaseSession`. I/O failures are raised as `APITransportError` subclasses with the right `transient` flag; that mapping is what makes retries behave the same on every transport. Build the URL with `api.build_url(method.request_path, method.get_query_map())` and the body from `method.get_body_map()` and `method.request_body_format`, and the headers from `method.request_headers.build(headers)` with the authentication header from `auth` set last, on every request. `timeout` is the call's `timeout=`; apply it as described in [Timeouts](sessions.md#timeouts), and use the session's own timeouts when it is `None`. Log an attempt the way the aiohttp transport does: the method and `method.request_path_template` when it is sent, the status and duration when it ends, never `request_path` (its values may name a user's data), the query, headers or body.

Transport code lives only in `<layer>/<transport>.py` modules (today `session/aiohttp.py`, `session/body/aiohttp.py`, `response/aiohttp.py` and `http/exceptions/aiohttp.py`), so a second transport, such as httpx, adds `*/httpx.py` files and changes nothing else. `bind_loop` in `session/loop.py` is transport-neutral and can be reused.

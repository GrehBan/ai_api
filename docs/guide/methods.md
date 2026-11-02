# Methods

A method class declares the HTTP request in its dunder class attributes. Its fields are the request parameters. Fields set to `None` are not sent, and unknown fields raise `ValidationError`, so a misspelled parameter is not sent silently.

| Attribute | Default | Meaning |
|---|---|---|
| `__request_path__` | required | Path appended to the API base URL; may contain `{field}` placeholders |
| `__request_path_safe__` | `""` | Characters left unescaped in path parameters, e.g. `"/"` |
| `__request_method__` | required | HTTP method, e.g. `"POST"` |
| `__request_query__` | empty | Names of fields sent as query parameters; fields that are neither path nor query parameters go to the body. Unknown names and mapping or model fields raise `TypeError` at class definition |
| `__request_body_format__` | `BodyFormat.NONE` | `JSON`, `FORM`, `MULTIPART` or `NONE` |
| `__request_headers__` | `None` | Headers sent with every request of the method, as a mapping or `Headers`; see [Default headers](sessions.md#default-headers) for which ones win |
| `__value_formatter__` | `BaseValueFormatter()` | Writes dates, times and durations (ISO 8601); see [Dates, times and durations](#dates-times-and-durations) |
| `__retryable__` | `False` | Retry transient failures (up to 4 attempts in total) |
| `__returning__` | inferred | Result class, taken from the generic argument |
| `__response_format__` | inferred | How the response body is parsed |
| `__response_parser__` | `BufferedResponseParser()` | Parser of buffered responses (`BaseAPIMethod`) |
| `__stream_parser__` | `StreamResponseParser()` | Parser of streamed responses (`BaseStreamMethod`) |

The attributes are inherited, so an intermediate base class can hold what a group of methods shares:

```python title="inheritance.py"
class JSONPost[T](BaseAPIMethod[T]):          # shared by a group of endpoints
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON


class Summarize(JSONPost[Completion]):        # POST /summaries, JSON body
    __request_path__ = "/summaries"

    text: str
```

Why dunder class attributes rather than fields or a decorator:

- **They never collide with request parameters.** An API may well have a parameter named `path`, `method` or `format`.
- **They describe the class, not an instance.** The configuration is checked once, when the class is defined: a bad path template, an unknown query field or a response format that contradicts the result type raises `TypeError` at import time, not at the first request.
- **The design follows aiogram's `TelegramMethod`.** A request is data that knows its own endpoint and result type, so one generic `send_request` serves every endpoint, and adding an endpoint means adding one small class.

Inside the library, these attributes are read only through regular properties (`request_path`, `request_method`, `request_headers`, `request_body_format`, `retryable`, `response_format`, `response_type`, `response_parser`, `stream_parser`), which is what the session and parsers see.

## Validation of parameters

Methods are validated by Pydantic when they are created, before anything is sent:

- `extra="forbid"`: an unknown field raises `ValidationError`. A misspelled optional parameter would otherwise be dropped silently, and the API would use its default without telling you.
- Fields can be given by name or by alias.
- Any Pydantic constraint works, e.g. `temperature: float = Field(ge=0, le=2)`.

## Optional fields: `MISSING` and `None`

A field is sent as it is declared. JSON tells a field that is absent from one that is `null`, and an API may treat them differently (`null` can mean "clear the value"), so a method has a separate way to say each:

| Declaration | Not given | `None` given |
|---|---|---|
| `x: int` | required | `ValidationError` |
| `x: int \| None` | required | sent as `"x": null` |
| `x: int \| MISSING = MISSING` | left out | `ValidationError` |
| `x: int \| None \| MISSING = MISSING` | left out | sent as `"x": null` |

`MISSING` is Pydantic's sentinel (`from pydantic import MISSING`): a field that holds it is left out of the request, in the body and in the query. Write the type after the provider's schema: `None` in the type where the schema allows `null`, a `MISSING` default where the field is optional. `MISSING` goes in a type only together with the `MISSING` default, and the other way round: a required field that takes it could be left out of a request it is required in, so either raises `TypeError` at class definition. `MISSING` leaves a whole field out; as an item of a list or a value of a dict it raises `ValueError` when the request is built.

A default of `None` raises `TypeError` at class definition, also one that a default factory gives (the factory is called once to see). It would send `null` for a field nobody set, which some APIs reject and others take as a request to clear a value; `= MISSING` is what leaves a field out. Fields with `exclude=True` are never sent and are not checked.

The same goes for nested request data: a model, a dataclass (standard or Pydantic's) or a `NamedTuple` whose field defaults to `None`, anywhere inside a field's type (also behind a `NewType`), raises `TypeError` too. Types are looked into once the class is complete, so a recursive type alias such as `type JSON = dict[str, JSON] | list[JSON] | str | None` is fine, and a model declared further down than the method is checked at the method's first use: from then on every instance raises. Describe nested data with a `TypedDict`, as the provider SDKs do: a key marked `NotRequired` is left out until given, and a plain dict literal type-checks against it:

```python title="optional_fields.py"
from typing import NotRequired, TypedDict

from pydantic import MISSING


class Filter(TypedDict):
    field: str
    value: NotRequired[str | None]    # left out, a string or null


class Search(BaseAPIMethod[ModelList]):
    __request_path__ = "/search"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON

    query: str
    limit: int | MISSING = MISSING             # left out until given
    cursor: str | None | MISSING = MISSING     # left out, a string or null
    filters: list[Filter] | MISSING = MISSING


Search(query="cats").get_body_map()   # {"query": "cats"}
Search(query="cats", cursor=None).get_body_map()
# {"query": "cats", "cursor": None}, sent as "cursor": null
```

A model of this library that a field typed `Any` holds at run time, such as a response item sent back in the next request, keeps the shape it came in: its fields that hold `None` by default (absent in the response) are left out, and the rest are sent as they are.

Response models are the opposite on purpose: a field the provider adds tomorrow does not break today's client. `BaseAPIType` uses `extra="ignore"` and drops such a field; `BaseAPITypeExtra` uses `extra="allow"` and keeps it in `model_extra`. The provider types derive from `BaseAPITypeExtra`, so whatever the provider sent and the library does not model is still in `raw`, e.g. `response.raw.model_extra["service_tier"]`.

## Result type and response format

The generic argument sets both the result type and the response format:

| Declaration | `send_request` returns | Response format |
|---|---|---|
| `BaseAPIMethod[SomeModel]` | `SomeModel` | JSON, validated into the model |
| `BaseAPIMethod[str]` | `str` | Text, decoded with the response charset |
| `BaseAPIMethod[bytes]` | `bytes` | Raw body |
| `BaseAPIMethod[None]` | `None` | Empty body expected |

Setting `__response_format__` explicitly, in the method or in an intermediate base class, overrides the inferred value. It must still produce the declared type: `TEXT` gives `str`, `BYTES` gives `bytes`, `NONE` gives `None` and `JSON` gives a model, while `SSE` and `NDJSON` are for stream methods only. A format that contradicts the generic argument, such as `BaseAPIMethod[SomeModel]` with `BYTES`, raises `TypeError` at class definition, unless the method sets its own `__response_parser__`. The check exists because the parser casts its result to the declared type: without it, a contradicting format would make the static type lie. For the same reason a generic argument that is not a class, such as `list[Model]`, `dict[str, Any]` or a union, raises `TypeError` at class definition unless the method sets its own `__response_parser__`: the default parser could only fail on the response, after the request was sent.

How `BufferedResponseParser` reads each format:

- **JSON** is validated with `model_validate_json`, directly from the bytes, in Pydantic's JSON mode. With a custom `json_loads` the body is decoded first and validated in Python mode, where strict fields reject JSON forms such as a date string.
- **TEXT** is decoded with the charset of the response, or UTF-8 when there is none.
- **BYTES** is returned as it is.
- **NONE** expects an empty body; anything else raises `APIResponseError`.

A body that cannot be decoded or validated raises `APIResponseError`, which is never retried: sending the same request again would get the same answer.

## Request body

`__request_body_format__` decides how the body fields are encoded:

| Format | Sent as | Notes |
|---|---|---|
| `NONE` | no body | the default, for `GET` and `DELETE` |
| `JSON` | `application/json` | NaN and infinity raise `ValueError`; files and `bytes` raise `TypeError` |
| `FORM` | `application/x-www-form-urlencoded` | files raise `TypeError` |
| `MULTIPART` | `multipart/form-data` | the only format that carries files |

A method without a body sends every field as a path or query parameter. A field that is neither raises `TypeError` when the class is defined, since it would never be sent: list it in `__request_query__` or set a body format. Fields with `exclude=True` are never sent, so they are allowed, and classes without `__request_path__` (bases of other methods) are not checked.

Before encoding, the values are normalized:

- **`None`** is sent as `null`, at any depth; a field left `MISSING` is not in the body at all (see [Optional fields](#optional-fields-missing-and-none)).
- **Nested models** (`BaseAPIType`) are dumped to dicts, by their wire names.
- **Enums** are sent as their values, also as dict keys. Two keys that give the same value, such as `Color.RED` and `"red"`, raise `ValueError`, since one of the values would be lost.
- **Dates, times and durations** are written by the method's formatter, in ISO 8601 by default (see [Dates, times and durations](#dates-times-and-durations)).
- **Text streams** are refused: uploads are bytes, so open the file in binary mode. A method given one raises `ValidationError` when it is created.
- **Circular references** and nesting deeper than 253 levels raise `ValueError`, instead of an encoder error: orjson refuses deeper data.

In `FORM` and `MULTIPART` bodies a list sends its field once per item, a dict is sent as a JSON string and a boolean as `"true"` or `"false"`. A form can express neither `null` nor nesting, so a field that is `None`, or `None` in a list, raises `ValueError` (leave the field `MISSING` to leave it out), and a list inside a list raises `TypeError`.

## Dates, times and durations

A value is written the same way in every part of the request (JSON body, form, query and path), by the method's `__value_formatter__`. The default `BaseValueFormatter` uses ISO 8601, the `date-time`, `date` and `time` formats of JSON Schema and OpenAPI:

| Value | Sent as |
|---|---|
| `datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)` | `"2026-01-02T03:04:05Z"`; another offset as `+03:00` |
| `date(2026, 1, 2)` | `"2026-01-02"` |
| `time(12, 30)` | `"12:30:00"` |
| `timedelta(seconds=30)` | `"PT30S"` |

A naive datetime raises `ValueError`, because the moment it names depends on the reader's clock. NaN and infinity raise `ValueError` everywhere, since no API can read them.

An API that wants another format gets a subclass that overrides what differs, set on the base class of its methods. A single field takes its own format through Pydantic's `PlainSerializer`, which runs before the formatter:

```python title="dates.py"
import datetime
from typing import Annotated

from pydantic import PlainSerializer

from ai_api import BaseValueFormatter


class UnixSeconds(BaseValueFormatter):        # the API wants Unix seconds
    def format_datetime(self, value: datetime.datetime, /) -> str:
        return str(round(value.timestamp()))


class LegacyMethod[T](BaseAPIMethod[T]):      # base of the API's methods
    __value_formatter__ = UnixSeconds()


class CreateJob(LegacyMethod[Model]):
    __request_path__ = "/jobs"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON

    run_at: datetime.datetime                 # "1767323045"
    timeout: datetime.timedelta               # "PT30S": not overridden
    expires_at: Annotated[                    # 1767323045, a JSON number
        datetime.datetime,
        PlainSerializer(lambda value: round(value.timestamp())),
    ]
```

## Query parameters

Query parameters are listed in `__request_query__`:

```python title="query.py"
class ListModels(BaseAPIMethod[ModelList]):
    __request_path__ = "/models"
    __request_method__ = "GET"
    __request_query__ = {"limit"}

    limit: int | MISSING = MISSING    # sent as ?limit=... when given
```

A field left `MISSING` is not sent. A query has no `null`, so `None` raises `ValueError` when the request is built: skipping it would send another request than the one asked for. A list, tuple or set is sent as a repeated key: `ids=["a", "b"]` becomes `?ids=a&ids=b`; `None` in the list raises `ValueError` too, since dropping it would shift the other items. Booleans become `true` and `false`, enums their values, and dates as in [Dates, times and durations](#dates-times-and-durations); NaN and infinity raise `ValueError`. A value with no text form, such as a class, raises `TypeError`. A mapping or a model cannot be a query parameter, since there is no standard way to put it in a URL; such a field in `__request_query__` raises `TypeError` at class definition.

To send a field under another name, use `validation_alias` and `serialization_alias` rather than `alias`. The constructor then takes the field name in both mypy and pyright; with `alias`, mypy wants the field name and pyright the alias. At runtime both names are accepted.

```python title="aliases.py"
from pydantic import MISSING, Field


class ListFiles(BaseAPIMethod[ModelList]):
    __request_path__ = "/files"
    __request_method__ = "GET"
    __request_query__ = {"page_size"}

    page_size: int | MISSING = Field(
        default=MISSING,
        validation_alias="pageSize",
        serialization_alias="pageSize",
    )


ListFiles(page_size=10)   # sent as ?pageSize=10
```

## Path parameters

Placeholders in `__request_path__` are filled from the fields of the same name. Those fields are not sent in the body:

```python title="path_params.py"
class RetrieveModel(BaseAPIMethod[Model]):
    __request_path__ = "/models/{model_id}"
    __request_method__ = "GET"

    model_id: str


RetrieveModel(model_id="example-1").request_path   # "/models/example-1"
```

- **Escaping.** Values are percent-encoded, including `/`, `?` and `#`, so `model_id="../admin"` cannot change the request path. Values are written as in query parameters: enums give their value, booleans `true`/`false`, dates ISO 8601.
- **`build_url` gets an encoded path.** Join it without encoding it again, e.g. with `URL.joinpath(..., encoded=True)` as `BaseAPI` does. yarl's `/` operator would turn `%2F` into `%252F`, and the server would see a different identifier.
- **Rejected values.** `None`, NaN, infinity, an empty string and the `.` / `..` segments raise `ValueError` when the request is built.
- **Identifiers with slashes.** For values such as `meta-llama/Llama-3`, set `__request_path_safe__ = "/"` on that method. Each segment is still checked for `.` and `..`.
- **Checks at class definition.** A misspelled or missing field, a placeholder that is not a plain name (`{}`, `{a.b}`, `{a:>5}`, `{a!r}`), a field that is also in `__request_query__` a field annotated as a list, set, dict or model (it has no single segment) and a field that defaults to `MISSING` (a path cannot leave a segment out) raise `TypeError` when the class is defined, not when the request is sent. Such a value in a field typed `Any` raises `TypeError` when the request is built. Write literal braces as `{{` and `}}`.

## Describing an API

An `IAPI` says where the API lives and builds the URL of each request. `BaseAPI` covers the common case of a root URL and a version segment:

```text
BaseAPI("https://api.example.com", "v1").build_url("/models/m%2F1", {"limit": "5"})
# 'https://api.example.com/v1/models/m%2F1?limit=5'
```

- **`base_url`** must be an absolute HTTP(S) URL without credentials (`https://user:pw@host` would be sent as Basic auth next to the session's own; give them as the auth), a query or a fragment. It may end with a path prefix, e.g. `https://gateway.example.com/openai`.
- **`version`** must be one path segment of unreserved characters (`v1`, `2024-01`), or empty for none. `.` and `..` are rejected.
- Both are checked when `BaseAPI` is created, because they are joined into URLs without encoding.

For a different URL scheme, write your own class with `base_url`, `version` and `build_url(method, query)`. The `method` argument is the method's request path, already percent-encoded: join it with `URL.joinpath(..., encoded=True)`, never with yarl's `/` operator, which encodes it a second time.

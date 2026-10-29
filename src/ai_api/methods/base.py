"""Base classes for declaring API methods."""

import io
import string
import types
from abc import ABC
from collections.abc import Mapping
from typing import (
    Any,
    ClassVar,
    Final,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, PydanticUserError, TypeAdapter

from ai_api.client.api.protocols import Query
from ai_api.client.response.parse.buffered import BufferedResponseParser
from ai_api.client.response.parse.protocols import (
    IResponseParser,
    IStreamParser,
)
from ai_api.client.response.parse.sse import ServerSentEvent
from ai_api.client.response.parse.stream import StreamResponseParser
from ai_api.methods.enums import BodyFormat, ResponseFormat, ResponseMode
from ai_api.types.files import InputFile, IOFile

_FORMAT_BY_TYPE: Final[dict[type[Any], ResponseFormat]] = {
    str: ResponseFormat.TEXT,
    bytes: ResponseFormat.BYTES,
    type(None): ResponseFormat.NONE,
}

_STREAM_FORMAT_BY_TYPE: Final[dict[Any, ResponseFormat]] = {
    bytes: ResponseFormat.BYTES,
    str: ResponseFormat.TEXT,
    ServerSentEvent: ResponseFormat.SSE,
}

_EVENT_FORMATS: Final[frozenset[ResponseFormat]] = frozenset(
    {ResponseFormat.SSE, ResponseFormat.NDJSON}
)

_DOT_SEGMENTS: Final[frozenset[str]] = frozenset({".", ".."})


def _parse_path_params(template: str) -> frozenset[str]:
    """Extract the placeholder names of a request path template.

    Args:
        template: A path such as ``"/models/{model_id}"``. Literal braces
            are written as ``{{`` and ``}}``.

    Returns:
        The placeholder names.

    Raises:
        ValueError: If the template is malformed or a placeholder is not
            a plain name (``{}``, ``{a.b}``, ``{a[0]}``, ``{a:>5}`` and
            ``{a!r}`` are rejected).
    """
    names: set[str] = set()

    for _, name, spec, conversion in string.Formatter().parse(template):
        if name is None:
            continue

        if not name.isidentifier() or spec or conversion:
            raise ValueError(
                f"Path placeholder {{{name}}} must be a plain field name"
            )

        names.add(name)

    return frozenset(names)


def _encode_path_value(name: str, value: Any, safe: str, raw: Any) -> str:
    """Convert a path parameter to an escaped path segment.

    Args:
        name: The parameter name, for error messages.
        value: The JSON-mode dump of the field value.
        safe: Characters left unescaped, e.g. ``"/"``.
        raw: The field value itself, for error messages.

    Returns:
        The percent-encoded value.

    Raises:
        ValueError: If the value is ``None``, dumps to ``null`` (such as
            NaN), is empty, or is a ``.``/``..`` segment, which would
            change the request path.
    """
    if value is None:
        if raw is not None:
            raise ValueError(
                f"Path parameter {name!r} has an unsafe value: {raw!r}"
            )

        raise ValueError(f"Path parameter {name!r} is not set")

    text = str(value).lower() if isinstance(value, bool) else str(value)
    segments = text.split("/") if "/" in safe else [text]

    if any(not segment or segment in _DOT_SEGMENTS for segment in segments):
        raise ValueError(
            f"Path parameter {name!r} has an unsafe value: {text!r}"
        )

    return quote(text, safe=safe)


def _is_mapping_type(annotation: Any) -> bool:
    """Check whether a field annotation allows mappings or models.

    Args:
        annotation: The annotation of a field, unions included.

    Returns:
        ``True`` if the annotation or a member of its union is a mapping
        or a Pydantic model.
    """
    origin = get_origin(annotation)

    if origin in {Union, types.UnionType}:
        return any(_is_mapping_type(arg) for arg in get_args(annotation))

    target = origin or annotation

    return isinstance(target, type) and issubclass(
        target, (Mapping, BaseModel)
    )


def _query_item(name: str, value: Any) -> str:
    """Convert one dumped query value to a string.

    Args:
        name: The parameter name, for error messages.
        value: A JSON-mode dump of the value.

    Returns:
        The value as a string; booleans as ``"true"``/``"false"``.

    Raises:
        TypeError: If the value is an object or a nested list.
    """
    if isinstance(value, (dict, list)):
        raise TypeError(
            f"Query parameter {name!r} must be a scalar or a list of scalars"
        )

    return str(value).lower() if isinstance(value, bool) else str(value)


def _query_value(name: str, value: Any) -> str | list[str]:
    """Convert a dumped query field to a string or a list of strings.

    Args:
        name: The parameter name, for error messages.
        value: A JSON-mode dump of the field.

    Returns:
        A string, or a list of strings sent as repeated keys; ``None``
        items are dropped.

    Raises:
        TypeError: If the value is an object or contains one.
    """
    if isinstance(value, list):
        return [_query_item(name, item) for item in value if item is not None]

    return _query_item(name, value)


def _resolve_response_type(arg: Any) -> type[Any] | None:
    """Turn a generic argument into a runtime class.

    Args:
        arg: The argument of ``BaseMethod[...]``.

    Returns:
        ``NoneType`` for ``None``, the class itself for classes and
        ``None`` for anything else (type variables, unions, aliases).
    """
    if arg is None:
        return type(None)

    if isinstance(arg, type):
        return arg

    return None


def _wrap_streams(value: Any) -> Any:
    """Wrap raw binary streams into ``IOFile``, recursing into containers.

    Args:
        value: A field value of a method.

    Returns:
        The value with every binary stream replaced by an ``IOFile``;
        the same object if it contains no streams.

    Raises:
        ValueError: If a text stream is found.
    """
    if isinstance(value, (io.BufferedIOBase, io.RawIOBase)):
        return IOFile(value)

    if isinstance(value, io.IOBase):
        raise ValueError("Only binary streams can be uploaded")

    if type(value) in {list, tuple}:
        items = [_wrap_streams(item) for item in value]

        if all(new is old for new, old in zip(items, value, strict=True)):
            return value

        return type(value)(items)

    if type(value) is dict:
        wrapped = {key: _wrap_streams(item) for key, item in value.items()}

        if all(wrapped[key] is item for key, item in value.items()):
            return value

        return wrapped

    return value


def _has_one_shot_file(value: Any) -> bool:
    """Check whether a value contains a file that can be read only once.

    Args:
        value: A method, model, container or field value.

    Returns:
        ``True`` if any nested ``InputFile`` is not reusable.
    """
    if isinstance(value, InputFile):
        return not value.reusable

    if isinstance(value, BaseModel):
        return any(_has_one_shot_file(item) for _, item in value)

    if isinstance(value, Mapping):
        return any(_has_one_shot_file(item) for item in value.values())

    if isinstance(value, (list, tuple)):
        return any(_has_one_shot_file(item) for item in value)

    return False


class BaseMethod[ResponseT](BaseModel, ABC):
    """Request contract shared by buffered and stream methods.

    A method is a Pydantic model: its fields are the request parameters,
    and dunder class attributes describe the HTTP request. Subclass
    ``BaseAPIMethod`` or ``BaseStreamMethod`` rather than this class.

    The generic argument determines ``__returning__`` and
    ``__response_format__``, unless a subclass sets them explicitly.

    Attributes:
        __request_path__: Path appended to the API's base URL. It may
            contain ``{field}`` placeholders, filled from the fields of
            the same name and percent-encoded.
        __request_path_params__: Names of the placeholders; derived from
            ``__request_path__``.
        __path_safe__: Characters left unescaped in path parameters.
            Empty by default; set it to ``"/"`` for identifiers such as
            ``"org/model"``.
        __request_method__: HTTP method, e.g. ``"POST"``.
        __request_query__: Names of fields sent as query parameters; all
            other fields go to the body. A list or tuple is sent as a
            repeated key (``ids=a&ids=b``); mappings and models are
            rejected.
        __returning__: Runtime class of the result; inferred from the
            generic argument unless a subclass sets it explicitly.
        __request_body_format__: Encoding of the request body.
        __retryable__: Whether transient failures may be retried. Ignored
            when the request carries a file that can be read only once.
        __response_format__: How the response body is parsed.
        __response_mode__: Whether the response is buffered or streamed.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_by_name=True,
        validate_by_alias=True,
        arbitrary_types_allowed=True,
    )

    __request_path__: ClassVar[str]
    __request_path_params__: ClassVar[frozenset[str]] = frozenset()
    __path_safe__: ClassVar[str] = ""
    __request_method__: ClassVar[str]
    __request_query__: ClassVar[set[str] | None] = None
    __returning__: ClassVar[type[Any] | None] = None
    __request_body_format__: ClassVar[BodyFormat] = BodyFormat.NONE
    __retryable__: ClassVar[bool] = False
    __response_format__: ClassVar[ResponseFormat]
    __response_mode__: ClassVar[ResponseMode]

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Validate the path template and infer the response type.

        Args:
            **kwargs: Class keyword arguments, forwarded to Pydantic.
        """
        super().__pydantic_init_subclass__(**kwargs)
        cls._init_path_params()
        cls._init_query_params()
        cls._init_response_type()

        # A parametrized class such as BaseStreamMethod[Event] is created
        # before the subclass body that may set the format or parser, so
        # only the classes users declare are validated.
        if cls.__pydantic_generic_metadata__["origin"] is None:
            cls._validate_response_type()

    @classmethod
    def _init_path_params(cls) -> None:
        """Derive ``__request_path_params__`` from the class's own path.

        Classes that do not set ``__request_path__`` themselves keep the
        inherited parameters. A placeholder's field must be declared in
        the same class as the path or above it.

        Raises:
            TypeError: If the template is malformed, names a missing
                field, or a placeholder is also a query parameter.
        """
        template = cls.__dict__.get("__request_path__")

        if template is not None:
            try:
                params = _parse_path_params(template)
            except ValueError as exc:
                raise TypeError(
                    f"{cls.__name__}: invalid __request_path__ "
                    f"{template!r}: {exc}"
                ) from exc

            if missing := params - cls.model_fields.keys():
                raise TypeError(
                    f"{cls.__name__}: __request_path__ placeholders "
                    f"{sorted(missing)} have no matching fields"
                )

            cls.__request_path_params__ = params

        # Checked on every subclass: either side may be set further down.
        query = cls.__request_query__ or set()

        if overlap := cls.__request_path_params__ & query:
            raise TypeError(
                f"{cls.__name__}: fields {sorted(overlap)} cannot be both "
                "path and query parameters"
            )

    @classmethod
    def _init_query_params(cls) -> None:
        """Check that every query parameter is a field that fits a URL.

        Raises:
            TypeError: If a name in ``__request_query__`` has no field, or
                the field is annotated as a mapping or a model.
        """
        query = cls.__request_query__ or set()

        if unknown := query - cls.model_fields.keys():
            raise TypeError(
                f"{cls.__name__}: __request_query__ names {sorted(unknown)} "
                "have no matching fields"
            )

        for name in sorted(query):
            if _is_mapping_type(cls.model_fields[name].annotation):
                raise TypeError(
                    f"{cls.__name__}: query parameter {name!r} cannot be "
                    "a mapping or a model"
                )

    @classmethod
    def _init_response_type(cls) -> None:
        """Infer ``__returning__`` and the response format."""
        args = cls.__pydantic_generic_metadata__["args"]

        if not args:
            return

        response_type = _resolve_response_type(args[0])

        if response_type is None:
            return

        if not cls._defines("__returning__"):
            cls.__returning__ = response_type

        if not cls._defines("__response_format__"):
            response_format = cls._infer_response_format(response_type)

            if response_format is not None:
                cls.__response_format__ = response_format

    @classmethod
    def _validate_response_type(cls) -> None:
        """Check that the result type can be produced by the parser.

        The base implementation accepts everything; subclasses narrow it.
        """

    def model_post_init(self, context: Any, /) -> None:
        """Wrap raw binary streams in any field into ``IOFile``.

        This happens once, at construction: ``model_dump()`` would turn
        raw streams into lazy iterators, and the wrapper records the start
        position needed to resend the content on retry.

        Args:
            context: Pydantic validation context.

        Raises:
            ValueError: If a field holds a text stream.
        """
        for name, value in self:
            wrapped = _wrap_streams(value)

            if wrapped is not value:
                setattr(self, name, wrapped)

    @classmethod
    def _infer_response_format(
        cls,
        response_type: type[Any],
    ) -> ResponseFormat | None:
        """Pick the response format for a result class.

        Args:
            response_type: The class from the generic argument.

        Returns:
            The format, or ``None`` to keep the inherited default.
        """
        return _FORMAT_BY_TYPE.get(response_type)

    @classmethod
    def _defines(cls, name: str) -> bool:
        """Check whether a user class in the MRO sets an attribute.

        Defaults set by the library's base classes do not count, so the
        generic argument can still pick the format.

        Args:
            name: Attribute name.

        Returns:
            ``True`` if a non-library class defines it.
        """
        return any(
            name in klass.__dict__
            for klass in cls.__mro__
            if klass not in _BASE_METHODS
        )

    @property
    def request_path(self) -> str:
        """Path appended to the API's base URL, placeholders filled in.

        Each parameter is dumped in JSON mode (so enums give their value)
        and percent-encoded; only ``__path_safe__`` characters are kept.

        Raises:
            ValueError: If a parameter is ``None``, empty, or a ``.``/``..``
                segment.
        """
        params = self.__request_path_params__

        if not params:
            return self.__request_path__

        values = self.model_dump(include=set(params), mode="json")

        return self.__request_path__.format_map(
            {
                name: _encode_path_value(
                    name,
                    values.get(name),
                    self.__path_safe__,
                    getattr(self, name),
                )
                for name in params
            }
        )

    @property
    def request_method(self) -> str:
        """HTTP method."""
        return self.__request_method__

    @property
    def body_format(self) -> BodyFormat:
        """Encoding of the request body."""
        return self.__request_body_format__

    @property
    def retryable(self) -> bool:
        """Whether this request may be retried.

        ``False`` when ``__retryable__`` is not set or the request carries
        a file that can be read only once.
        """
        return self.__retryable__ and not _has_one_shot_file(self)

    @property
    def response_format(self) -> ResponseFormat:
        """How the response body is parsed."""
        return self.__response_format__

    @property
    def response_mode(self) -> ResponseMode:
        """Whether the response is buffered or streamed."""
        return self.__response_mode__

    @property
    def response_type(self) -> type[ResponseT] | None:
        """Runtime class of the result, if known."""
        return cast(type[ResponseT] | None, self.__returning__)

    def get_query_map(self) -> Query | None:
        """Return the query parameters.

        Returns:
            Fields listed in ``__request_query__`` converted to strings
            (booleans as ``"true"``/``"false"``) or, for lists and tuples,
            to lists of strings; ``None`` if the method has no query
            parameters.

        Raises:
            TypeError: If a value is an object or contains one.
        """
        if self.__request_query__ is None:
            return None

        data = self.model_dump(
            include=self.__request_query__,
            exclude_none=True,
            by_alias=True,
            mode="json",
        )

        return {key: _query_value(key, value) for key, value in data.items()}

    def get_body_map(self) -> Mapping[str, Any]:
        """Return the body fields.

        Returns:
            All fields except path and query parameters and ``None``
            values, keyed by alias.
        """
        return self.model_dump(
            exclude=(
                (self.__request_query__ or set())
                | self.__request_path_params__
            ),
            exclude_none=True,
            by_alias=True,
            mode="python",
        )


class BaseAPIMethod[ResponseT](BaseMethod[ResponseT]):
    """A method whose whole response is read and parsed at once.

    ``ResponseT`` is the type of the parsed response: a ``BaseAPIType``
    model (parsed as JSON), ``str``, ``bytes`` or ``None``.

    Attributes:
        __response_parser__: Parser of the response.

    Example:
        >>> class ListModels(BaseAPIMethod[ModelList]):
        ...     __request_path__ = "/models"
        ...     __request_method__ = "GET"
    """

    __response_format__ = ResponseFormat.JSON
    __response_mode__ = ResponseMode.BUFFERED
    __response_parser__: ClassVar[IResponseParser] = BufferedResponseParser()

    @classmethod
    def _infer_response_format(
        cls,
        response_type: type[Any],
    ) -> ResponseFormat | None:
        """Pick the response format, mapping models to JSON.

        Args:
            response_type: The class from the generic argument.

        Returns:
            The format, or ``None`` to keep the inherited default.
        """
        if issubclass(response_type, BaseModel):
            return ResponseFormat.JSON

        return super()._infer_response_format(response_type)

    @property
    def response_parser(self) -> IResponseParser:
        """Parser of the response."""
        return self.__response_parser__


class BaseStreamMethod[ResponseT](BaseMethod[ResponseT]):
    """A method whose response is consumed as a stream.

    ``ResponseT`` is the type of one streamed item: ``bytes`` (raw
    chunks, the default), ``str`` (decoded text) or ``ServerSentEvent``
    (server-sent events as they are). Send it with ``session.stream()``.

    Other item types, such as models and discriminated unions, need
    ``__response_format__`` set to SSE or NDJSON: each event is then
    validated into the item type. Without an explicit format or
    ``__stream_parser__`` they raise ``TypeError`` at class definition;
    otherwise the stream would yield ``bytes`` typed as the model.

    Attributes:
        __stream_parser__: Parser of the stream.
        __stream_item__: The generic argument as written, including
            unions and type aliases.
        __response_adapter__: Validator of the item type for SSE and
            NDJSON; ``None`` for ``ServerSentEvent`` items and other
            formats.

    Example:
        >>> class Download(BaseStreamMethod[bytes]):
        ...     __request_path__ = "/files/content"
        ...     __request_method__ = "GET"
        ...     __request_query__ = {"file_id"}
        ...     file_id: str
    """

    __response_format__ = ResponseFormat.BYTES
    __response_mode__ = ResponseMode.STREAM
    __stream_parser__: ClassVar[IStreamParser] = StreamResponseParser()
    __stream_item__: ClassVar[Any] = None
    __response_adapter__: ClassVar[TypeAdapter[Any] | None] = None

    @classmethod
    def _init_response_type(cls) -> None:
        """Infer the response type and keep the item type as written."""
        super()._init_response_type()
        args = cls.__pydantic_generic_metadata__["args"]

        if args:
            item = args[0]
            cls.__stream_item__ = type(None) if item is None else item

    @classmethod
    def _infer_response_format(
        cls,
        response_type: type[Any],
    ) -> ResponseFormat | None:
        """Pick the stream format from the item type.

        Args:
            response_type: The class from the generic argument.

        Returns:
            BYTES for ``bytes``, TEXT for ``str``, SSE for
            ``ServerSentEvent``, or ``None`` for any other item type.
        """
        return _STREAM_FORMAT_BY_TYPE.get(response_type)

    @classmethod
    def _validate_response_type(cls) -> None:
        """Check the item type and build its validator.

        Raises:
            TypeError: If the default parser cannot produce the item type
                and neither the format nor the parser is set explicitly,
                or if Pydantic cannot validate the item type of an SSE or
                NDJSON stream.
        """
        item = cls.__stream_item__

        if item is None or isinstance(item, TypeVar):
            return

        # Identity, not ``in``: Annotated arguments may be unhashable.
        known = any(item is cls_ for cls_ in _STREAM_FORMAT_BY_TYPE)

        if not (
            known
            or cls._defines("__response_format__")
            or cls._defines("__stream_parser__")
        ):
            name = item.__name__ if isinstance(item, type) else item
            raise TypeError(
                f"{cls.__name__}: stream items of type {name} are not "
                "supported by the default parser; use bytes, str or "
                "ServerSentEvent, or set __response_format__ (SSE or "
                "NDJSON) or __stream_parser__ explicitly"
            )

        if cls.__response_format__ not in _EVENT_FORMATS or (
            item is ServerSentEvent
            and cls.__response_format__ is ResponseFormat.SSE
        ):
            cls.__response_adapter__ = None
            return

        try:
            cls.__response_adapter__ = TypeAdapter(item)
        except PydanticUserError as exc:
            raise TypeError(
                f"{cls.__name__}: cannot validate stream items of type "
                f"{item!r}: {exc}"
            ) from exc

    @property
    def stream_parser(self) -> IStreamParser:
        """Parser of the stream."""
        return self.__stream_parser__

    @property
    def response_adapter(self) -> TypeAdapter[ResponseT] | None:
        """Validator of one SSE or NDJSON item, if events are validated."""
        return cast(
            TypeAdapter[ResponseT] | None,
            self.__response_adapter__,
        )


_BASE_METHODS: Final[frozenset[type[Any]]] = frozenset(
    {BaseMethod, BaseAPIMethod, BaseStreamMethod}
)

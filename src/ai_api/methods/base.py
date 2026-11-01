"""Base classes for declaring API methods."""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from typing import (
    Any,
    ClassVar,
    Final,
    TypeVar,
    cast,
)

from pydantic import BaseModel, ConfigDict, PydanticUserError, TypeAdapter

from ai_api.enums import BodyFormat, ResponseFormat
from ai_api.http.headers import Headers
from ai_api.http.protocols import HeadersType, IHeaders, Query
from ai_api.methods.fields import (
    has_one_shot_file,
    is_container_type,
    is_mapping_type,
    query_value,
    wrap_streams,
)
from ai_api.methods.path import encode_path_value, parse_path_params
from ai_api.response.parse.buffered import BufferedResponseParser
from ai_api.response.parse.protocols import (
    IResponseParser,
    IStreamParser,
)
from ai_api.response.parse.sse import ServerSentEvent
from ai_api.response.parse.stream import StreamResponseParser
from ai_api.serialization.protocols import IValueFormatter
from ai_api.serialization.values import BaseValueFormatter
from ai_api.types.protocols import IEventMatcher

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

_TYPE_BY_FORMAT: Final[dict[ResponseFormat, type[Any]]] = {
    response_format: response_type
    for response_type, response_format in _FORMAT_BY_TYPE.items()
}

_STREAM_TYPE_BY_FORMAT: Final[dict[ResponseFormat, type[Any]]] = {
    ResponseFormat.BYTES: bytes,
    ResponseFormat.TEXT: str,
}


def _type_name(response_type: Any) -> str:
    """Name a result or item type for error messages.

    Args:
        response_type: A class or another generic argument.

    Returns:
        The class name, ``None`` for ``NoneType`` and the ``repr`` of
        anything else.
    """
    if response_type is type(None):
        return "None"

    if isinstance(response_type, type):
        return response_type.__name__

    return repr(response_type)


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
        __request_path_safe__: Characters left unescaped in path
            parameters.
            Empty by default; set it to ``"/"`` for identifiers such as
            ``"org/model"``.
        __request_method__: HTTP method, e.g. ``"POST"``.
        __request_headers__: Headers sent with every request of the
            method, as a mapping or an ``IHeaders`` collection; ``None``
            for none. They override the session's default headers and
            are overridden by the headers of a call.
        __request_query__: Names of fields sent as query parameters; all
            other fields go to the body. Any set may be given; it is
            stored as a ``frozenset``. A list or tuple is sent as a
            repeated key (``ids=a&ids=b``); mappings and models are
            rejected.
        __returning__: Runtime class of the result; inferred from the
            generic argument unless a subclass sets it explicitly.
        __request_body_format__: Encoding of the request body.
        __value_formatter__: Writes dates, times and durations in every
            part of the request (body, form, query and path). ISO 8601 by
            default; a provider whose API wants another format sets its
            own on its base method.
        __retryable__: Whether transient failures may be retried. Ignored
            when the request carries a file that can be read only once.
        __response_format__: How the response body is parsed.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_by_name=True,
        validate_by_alias=True,
        arbitrary_types_allowed=True,
    )

    __request_path__: ClassVar[str]
    __request_path_params__: ClassVar[frozenset[str]] = frozenset()
    __request_path_safe__: ClassVar[str] = ""
    __request_method__: ClassVar[str]
    __request_query__: ClassVar[AbstractSet[str]] = frozenset()
    __returning__: ClassVar[type[Any] | None] = None
    __request_body_format__: ClassVar[BodyFormat] = BodyFormat.NONE
    __value_formatter__: ClassVar[IValueFormatter] = BaseValueFormatter()
    __retryable__: ClassVar[bool] = False
    __response_format__: ClassVar[ResponseFormat]
    __request_headers__: ClassVar[HeadersType | None] = None

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Validate the path template and infer the response type.

        Args:
            **kwargs: Class keyword arguments, forwarded to Pydantic.
        """
        super().__pydantic_init_subclass__(**kwargs)
        cls._init_query_params()
        cls._init_path_params()
        cls._check_body_fields()
        cls._init_response_type()

        # A parametrized class such as BaseStreamMethod[Event] is created
        # before the subclass body that may set the format or parser, so
        # only the classes users declare are validated.
        if cls.__pydantic_generic_metadata__["origin"] is None:
            cls._validate_response_type()
            cls._init_response_adapter()

    @classmethod
    def _init_path_params(cls) -> None:
        """Derive ``__request_path_params__`` from the class's own path.

        Classes that do not set ``__request_path__`` themselves keep the
        inherited parameters. A placeholder's field must be declared in
        the same class as the path or above it.

        Raises:
            TypeError: If the template is malformed, names a missing
                field, a placeholder is also a query parameter, or its
                field is annotated as a mapping, a model or a collection,
                which has no single path segment.
        """
        template = cls.__dict__.get("__request_path__")

        if template is not None:
            try:
                params = parse_path_params(template)
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
        if overlap := cls.__request_path_params__ & cls.__request_query__:
            raise TypeError(
                f"{cls.__name__}: fields {sorted(overlap)} cannot be both "
                "path and query parameters"
            )

        for name in sorted(cls.__request_path_params__):
            if is_container_type(cls.model_fields[name].annotation):
                raise TypeError(
                    f"{cls.__name__}: path parameter {name!r} must be a "
                    "scalar, not a mapping, a model or a collection"
                )

    @classmethod
    def _init_query_params(cls) -> None:
        """Freeze ``__request_query__`` and check every query parameter.

        Raises:
            TypeError: If a name in ``__request_query__`` has no field, or
                the field is annotated as a mapping or a model.
        """
        query = frozenset(cls.__request_query__)
        cls.__request_query__ = query

        if unknown := query - cls.model_fields.keys():
            raise TypeError(
                f"{cls.__name__}: __request_query__ names {sorted(unknown)} "
                "have no matching fields"
            )

        for name in sorted(query):
            if is_mapping_type(cls.model_fields[name].annotation):
                raise TypeError(
                    f"{cls.__name__}: query parameter {name!r} cannot be "
                    "a mapping or a model"
                )

    @classmethod
    def _check_body_fields(cls) -> None:
        """Reject fields that a method without a body would never send.

        A field that is neither a path nor a query parameter goes to the
        body, and ``BodyFormat.NONE`` sends no body, so such a field
        would be dropped silently (a ``GET`` whose ``__request_query__``
        was forgotten, for one). Classes without a ``__request_path__``
        are bases that a subclass completes, so they are not checked.

        Raises:
            TypeError: If the body format is ``NONE`` and a field, other
                than one excluded from dumps, is neither a path nor a
                query parameter.
        """
        if (
            cls.__request_body_format__ is not BodyFormat.NONE
            or getattr(cls, "__request_path__", None) is None
        ):
            return

        params = cls.__request_query__ | cls.__request_path_params__

        if body := sorted(
            name
            for name, field in cls.model_fields.items()
            if name not in params and not field.exclude
        ):
            raise TypeError(
                f"{cls.__name__}: fields {body} would never be sent: they "
                "are not path or query parameters, and the method has no "
                "body; add them to __request_query__ or set "
                "__request_body_format__"
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

    @classmethod
    def _init_response_adapter(cls) -> None:
        """Build what the parser needs to validate the result.

        The base implementation builds nothing; stream methods build the
        validator of their items.
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
            wrapped = wrap_streams(value)

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
    def request_headers(self) -> IHeaders:
        """Headers of the method, as a new collection on every access."""
        return Headers(self.__request_headers__)

    @property
    def request_path(self) -> str:
        """Path appended to the API's base URL, placeholders filled in.

        Each parameter is written as text (enums as their values,
        booleans as ``true``/``false``, dates through ``value_formatter``)
        and percent-encoded; only ``__request_path_safe__`` characters are
        kept.

        Raises:
            TypeError: If a parameter (of a field typed ``Any``) holds a
                mapping, a model or a collection.
            ValueError: If a parameter is ``None``, NaN or infinite,
                empty, or a ``.``/``..`` segment.
        """
        params = self.__request_path_params__
        # Always formatted, also without placeholders, so "{{" and "}}"
        # become literal braces in every path alike.
        values = (
            self.model_dump(include=set(params), mode="python")
            if params
            else {}
        )

        return self.__request_path__.format_map(
            {
                name: encode_path_value(
                    name,
                    values.get(name),
                    self.__request_path_safe__,
                    self.value_formatter,
                )
                for name in params
            }
        )

    @property
    def request_method(self) -> str:
        """HTTP method."""
        return self.__request_method__

    @property
    def request_body_format(self) -> BodyFormat:
        """Encoding of the request body."""
        return self.__request_body_format__

    @property
    def value_formatter(self) -> IValueFormatter:
        """Writes the dates, times and durations of the request."""
        return self.__value_formatter__

    @property
    def retryable(self) -> bool:
        """Whether this request may be retried.

        ``False`` when ``__retryable__`` is not set or the request carries
        a file that can be read only once.
        """
        return self.__retryable__ and not has_one_shot_file(self)

    @property
    def response_format(self) -> ResponseFormat:
        """How the response body is parsed."""
        return self.__response_format__

    @property
    def response_type(self) -> type[ResponseT] | None:
        """Runtime class of the result, if known."""
        return cast(type[ResponseT] | None, self.__returning__)

    def get_query_map(self) -> Query | None:
        """Return the query parameters.

        Returns:
            Fields listed in ``__request_query__`` converted to strings
            (booleans as ``"true"``/``"false"``, dates through
            ``value_formatter``) or, for lists, tuples and sets, to lists
            of strings; ``None`` if the method has no query parameters.

        Raises:
            TypeError: If a value is an object or contains one.
            ValueError: If a list contains ``None``, or a value is NaN or
                infinite or rejected by the formatter.
        """
        if not self.__request_query__:
            return None

        data = self.model_dump(
            include=set(self.__request_query__),
            exclude_none=True,
            by_alias=True,
            mode="python",
        )

        return {
            key: query_value(key, value, self.value_formatter)
            for key, value in data.items()
        }

    def get_body_map(self) -> Mapping[str, Any]:
        """Return the body fields.

        Returns:
            All fields except path and query parameters and ``None``
            values, keyed by alias.
        """
        return self.model_dump(
            exclude=set(self.__request_query__ | self.__request_path_params__),
            exclude_none=True,
            by_alias=True,
            mode="python",
        )


class BaseAPIMethod[ResponseT](BaseMethod[ResponseT]):
    """A method whose whole response is read and parsed at once.

    ``ResponseT`` is the type of the parsed response: a ``BaseAPIType``
    model (parsed as JSON), ``str``, ``bytes`` or ``None``.

    Other arguments (``list[Model]``, ``dict``, unions, type aliases)
    raise ``TypeError`` at class definition unless the method sets its
    own ``__response_parser__``: the default parser could only fail on
    the response, after the request was sent.

    Attributes:
        __response_parser__: Parser of the response.
        __response_argument__: The generic argument as written.

    Example:
        >>> class ListModels(BaseAPIMethod[ModelList]):
        ...     __request_path__ = "/models"
        ...     __request_method__ = "GET"
    """

    __response_format__ = ResponseFormat.JSON
    __response_parser__: ClassVar[IResponseParser] = BufferedResponseParser()
    __response_argument__: ClassVar[Any] = None

    @classmethod
    def _init_response_type(cls) -> None:
        """Infer the response type and keep the argument as written."""
        super()._init_response_type()
        args = cls.__pydantic_generic_metadata__["args"]

        if args:
            cls.__response_argument__ = args[0]

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

    @classmethod
    def _validate_response_type(cls) -> None:
        """Check that the default parser produces the result class.

        TEXT gives ``str``, BYTES ``bytes``, NONE ``None`` and JSON a
        model, so an explicit format that contradicts the generic argument
        would return a value of another type than the one declared.

        Raises:
            TypeError: If ``__response_parser__`` is not set explicitly
                and the generic argument is not a class (such as
                ``list[Model]`` or a union), or the format is a stream
                format or produces another type.
        """
        returning = cls.__returning__

        if returning is None:
            argument = cls.__response_argument__

            # Unset on BaseAPIMethod itself (defined before _BASE_METHODS);
            # a type variable is filled in by a subclass.
            if (
                argument is None
                or isinstance(argument, TypeVar)
                or cls._defines("__response_parser__")
            ):
                return

            raise TypeError(
                f"{cls.__name__}: responses of type {_type_name(argument)} "
                "are not supported by the default parser; use a model, "
                "str, bytes or None, or set __response_parser__ explicitly"
            )

        if cls._defines("__response_parser__"):
            return

        response_format = cls.__response_format__

        if response_format in _EVENT_FORMATS:
            raise TypeError(
                f"{cls.__name__}: {response_format.name} responses can "
                "only be streamed; subclass BaseStreamMethod instead"
            )

        if response_format is ResponseFormat.JSON:
            produced = "a model"
            valid = issubclass(returning, BaseModel)
        else:
            expected = _TYPE_BY_FORMAT[response_format]
            produced = _type_name(expected)
            valid = returning is expected

        if not valid:
            raise TypeError(
                f"{cls.__name__}: {response_format.name} responses are "
                f"parsed into {produced}, not {_type_name(returning)}; "
                "change the generic argument or the format, or set "
                "__response_parser__ explicitly"
            )

    @property
    def response_parser(self) -> IResponseParser:
        """Parser of the response."""
        return self.__response_parser__


class BaseStreamMethod[ItemT](BaseMethod[ItemT]):
    """A method whose response is consumed as a stream.

    ``ItemT`` is the type of one streamed item: ``bytes`` (raw
    chunks, the default), ``str`` (decoded text) or ``ServerSentEvent``
    (server-sent events as they are). Send it with ``session.stream()``.

    Other item types, such as models and discriminated unions, need
    ``__response_format__`` set to SSE or NDJSON: each event is then
    validated into the item type. Without an explicit format or
    ``__stream_parser__`` they raise ``TypeError`` at class definition;
    otherwise the stream would yield ``bytes`` typed as the model.

    Attributes:
        __stream_parser__: Parser of the stream.
        __stream_terminal__: Recognizes the item that ends a complete
            SSE or NDJSON stream, e.g. ``EventTypes("message_stop")`` or
            ``DONE``. When it is set, a body that ends without it raises
            ``APIStreamInterruptedError`` and an item after it raises
            ``APIResponseError`` (the body is read to its end). ``None``,
            the default, checks nothing: by the HTTP and SSE standards
            the end of the body is a normal end.
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
    __stream_parser__: ClassVar[IStreamParser] = StreamResponseParser()
    __stream_terminal__: ClassVar[IEventMatcher | None] = None
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
        """Check that the parser can produce the item type.

        Raises:
            TypeError: If the default parser cannot produce the item type
                and neither the format nor the parser is set explicitly;
                or if, without an explicit parser, the format is not a
                stream format or BYTES/TEXT is set for items other than
                ``bytes``/``str``.
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
            raise TypeError(
                f"{cls.__name__}: stream items of type {_type_name(item)} "
                "are not supported by the default parser; use bytes, str "
                "or ServerSentEvent, or set __response_format__ (SSE or "
                "NDJSON) or __stream_parser__ explicitly"
            )

        if not cls._defines("__stream_parser__"):
            cls._validate_stream_format(item)

    @classmethod
    def _init_response_adapter(cls) -> None:
        """Build ``__response_adapter__`` for SSE and NDJSON items.

        ``ServerSentEvent`` items of an SSE stream are yielded raw, and
        other formats do not validate items, so they get ``None``.

        Raises:
            TypeError: If Pydantic cannot validate the item type.
        """
        item = cls.__stream_item__

        if item is None or isinstance(item, TypeVar):
            return

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

    @classmethod
    def _validate_stream_format(cls, item: Any) -> None:
        """Check that the default parser yields items of the item type.

        SSE and NDJSON validate each event into any item type; BYTES
        yields only ``bytes`` and TEXT only ``str``.

        Args:
            item: The generic argument as written.

        Raises:
            TypeError: If the format is not a stream format, BYTES or
                TEXT is set for another item type, or a BYTES or TEXT
                stream sets ``__stream_terminal__``, which only SSE and
                NDJSON check.
        """
        response_format = cls.__response_format__

        if response_format in _EVENT_FORMATS:
            return

        if cls.__stream_terminal__ is not None:
            raise TypeError(
                f"{cls.__name__}: __stream_terminal__ is checked only for "
                f"SSE and NDJSON streams, not {response_format.name}; "
                "remove it or set __stream_parser__ explicitly"
            )

        expected = _STREAM_TYPE_BY_FORMAT.get(response_format)

        if expected is None:
            raise TypeError(
                f"{cls.__name__}: {response_format.name} is not a stream "
                "format; use BYTES, TEXT, SSE or NDJSON"
            )

        if item is not expected:
            raise TypeError(
                f"{cls.__name__}: {response_format.name} streams yield "
                f"{_type_name(expected)} items, not {_type_name(item)}; "
                "change the generic argument or the format, or set "
                "__stream_parser__ explicitly"
            )

    @property
    def stream_parser(self) -> IStreamParser:
        """Parser of the stream."""
        return self.__stream_parser__

    @property
    def stream_terminal(self) -> IEventMatcher | None:
        """Recognizes the item that ends a complete stream, if declared."""
        return self.__stream_terminal__

    @property
    def response_adapter(self) -> TypeAdapter[ItemT] | None:
        """Validator of one SSE or NDJSON item, if events are validated."""
        return cast(
            TypeAdapter[ItemT] | None,
            self.__response_adapter__,
        )


_BASE_METHODS: Final[frozenset[type[Any]]] = frozenset(
    {BaseMethod, BaseAPIMethod, BaseStreamMethod}
)

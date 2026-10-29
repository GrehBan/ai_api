"""Base classes for declaring API methods."""

import io
from abc import ABC
from collections.abc import Mapping
from typing import Any, ClassVar, Final, cast

from pydantic import BaseModel, ConfigDict

from ai_api.client.response.parse.buffered import BufferedResponseParser
from ai_api.client.response.parse.protocols import (
    IResponseParser,
    IStreamParser,
)
from ai_api.client.response.parse.stream import StreamResponseParser
from ai_api.methods.enums import BodyFormat, ResponseFormat, ResponseMode
from ai_api.types.files import InputFile, IOFile

_FORMAT_BY_TYPE: Final[dict[type[Any], ResponseFormat]] = {
    str: ResponseFormat.TEXT,
    bytes: ResponseFormat.BYTES,
    type(None): ResponseFormat.NONE,
}


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
        The value with every binary stream replaced by an ``IOFile``.

    Raises:
        ValueError: If a text stream is found.
    """
    if isinstance(value, (io.BufferedIOBase, io.RawIOBase)):
        return IOFile(value)

    if isinstance(value, io.IOBase):
        raise ValueError("Only binary streams can be uploaded")

    if type(value) is list:
        return [_wrap_streams(item) for item in value]

    if type(value) is tuple:
        return tuple(_wrap_streams(item) for item in value)

    if type(value) is dict:
        return {key: _wrap_streams(item) for key, item in value.items()}

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

    The generic argument determines ``__returning__`` and, unless a
    subclass sets it explicitly, ``__response_format__``.

    Attributes:
        __request_path__: Path appended to the API's base URL.
        __request_method__: HTTP method, e.g. ``"POST"``.
        __request_query__: Names of fields sent as query parameters; all
            other fields go to the body.
        __returning__: Runtime class of the result; inferred from the
            generic argument.
        __request_body_format__: Encoding of the request body.
        __retryable__: Whether transient failures may be retried. Ignored
            when the request carries a file that can be read only once.
        __response_format__: How the response body is parsed.
        __response_mode__: Whether the response is buffered or streamed.
    """

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    __request_path__: ClassVar[str]
    __request_method__: ClassVar[str]
    __request_query__: ClassVar[set[str] | None] = None
    __returning__: ClassVar[type[Any] | None] = None
    __request_body_format__: ClassVar[BodyFormat] = BodyFormat.NONE
    __retryable__: ClassVar[bool] = False
    __response_format__: ClassVar[ResponseFormat]
    __response_mode__: ClassVar[ResponseMode]

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Infer ``__returning__`` and the response format.

        Args:
            **kwargs: Class keyword arguments, forwarded to Pydantic.
        """
        super().__pydantic_init_subclass__(**kwargs)

        args = cls.__pydantic_generic_metadata__["args"]

        if not args:
            return

        response_type = _resolve_response_type(args[0])

        if response_type is None:
            return

        if "__returning__" not in cls.__dict__:
            cls.__returning__ = response_type

        if not cls._defines("__response_format__"):
            response_format = cls._infer_response_format(response_type)

            if response_format is not None:
                cls.__response_format__ = response_format

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
        """Path appended to the API's base URL."""
        return self.__request_path__

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

    def get_query_map(self) -> Mapping[str, str] | None:
        """Return the query parameters.

        Returns:
            Fields listed in ``__request_query__`` converted to strings
            (booleans as ``"true"``/``"false"``), or ``None`` if the
            method has no query parameters.
        """
        if self.__request_query__ is None:
            return None

        data = self.model_dump(
            include=self.__request_query__,
            exclude_none=True,
            by_alias=True,
            mode="json",
        )

        return {
            key: str(value).lower() if isinstance(value, bool) else str(value)
            for key, value in data.items()
        }

    def get_body_map(self) -> Mapping[str, Any]:
        """Return the body fields.

        Returns:
            All fields except query parameters and ``None`` values, keyed
            by alias.
        """
        return self.model_dump(
            exclude=self.__request_query__ or set(),
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
    chunks, the default) or ``str`` (decoded text). Send it with
    ``session.stream()``.

    Attributes:
        __stream_parser__: Parser of the stream.

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

    @property
    def stream_parser(self) -> IStreamParser:
        """Parser of the stream."""
        return self.__stream_parser__


_BASE_METHODS: Final[frozenset[type[Any]]] = frozenset(
    {BaseMethod, BaseAPIMethod, BaseStreamMethod}
)

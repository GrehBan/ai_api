"""Matchers that recognize stream items by their content.

They describe a provider's stream protocol declaratively: the event that
ends a stream (``BaseStreamMethod.__stream_terminal__``), the events
that report an error (``StreamResponseParser(error_events=...)``) and
the responses that report a failure
(``BufferedResponseParser(error_responses=...)``). Matchers combine with
``|``.

Example:
    >>> terminal = EventTypes("message_stop")
    >>> terminal.matches({"type": "message_stop"})
    True
    >>> errors = EventTypes("error") | TopLevelError()
    >>> errors.matches({"error": "overloaded"})
    True
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Final, cast

from pydantic import BaseModel

from ai_api.types.protocols import IEventMatcher

_MISSING: Final = object()
"""Marks a field the item does not have."""


def _key(value: object) -> object:
    """Key a value so that JSON's kinds stay apart.

    Python takes ``True == 1`` and ``hash(True) == hash(1)``, so a set
    of values would match a JSON ``1`` against ``true``. Booleans get a
    key of their own; numbers share one, since JSON writes ``1`` and
    ``1.0`` for the same number; anything else is itself, so a
    ``StrEnum`` member still matches its string.

    Args:
        value: A field value or an accepted value.

    Returns:
        The value to compare and hash.
    """
    if isinstance(value, bool):
        return (bool, value)

    if isinstance(value, int | float):
        return (float, value)

    return value


def _field(item: object, name: str) -> object:
    """Read a field of an item, without running any code of it.

    Args:
        item: A model, a decoded JSON value or ``DONE_MARKER``.
        name: The field's name.

    Returns:
        A key of a mapping, a field (or an extra field) of a Pydantic
        model, an attribute of any other object, or ``_MISSING``. A
        model's properties are not read: a matcher must not fail on an
        item, and a property may raise.
    """
    if isinstance(item, Mapping):
        return cast(Mapping[Any, object], item).get(name, _MISSING)

    if isinstance(item, BaseModel):
        if name in type(item).model_fields:
            return getattr(item, name)

        extra = item.model_extra or {}
        return extra.get(name, _MISSING)

    return getattr(item, name, _MISSING)


class BaseMatcher(ABC):
    """Base class of matchers; gives every matcher ``a | b``."""

    @abstractmethod
    def matches(self, item: object) -> bool:
        """Check one item.

        Args:
            item: A model, a decoded JSON value or ``DONE_MARKER``.

        Returns:
            Whether the item is one the matcher looks for.
        """
        raise NotImplementedError

    def __or__(self, other: IEventMatcher) -> AnyOf:
        """Match what either matcher matches.

        Args:
            other: Another matcher.

        Returns:
            A matcher of both.
        """
        return AnyOf(self, other)

    def __ror__(self, other: IEventMatcher) -> AnyOf:
        """Match what either matcher matches, with this one on the right.

        Lets a matcher that only follows ``IEventMatcher``, without this
        base class, stand on the left of ``|``.

        Args:
            other: Another matcher.

        Returns:
            A matcher of both, ``other`` first.
        """
        return AnyOf(other, self)


class AnyOf(BaseMatcher):
    """Matches an item that any of its matchers matches.

    Attributes:
        matchers: The combined matchers; nested ``AnyOf`` are flattened.
    """

    def __init__(self, *matchers: IEventMatcher) -> None:
        """Combine matchers.

        Args:
            *matchers: The matchers to combine.

        Raises:
            ValueError: If no matcher is given, since the result would
                never match.
        """
        if not matchers:
            raise ValueError("AnyOf needs at least one matcher")

        flat: list[IEventMatcher] = []

        for matcher in matchers:
            if isinstance(matcher, AnyOf):
                flat.extend(matcher.matchers)
            else:
                flat.append(matcher)

        self.matchers: tuple[IEventMatcher, ...] = tuple(flat)

    def matches(self, item: object) -> bool:
        """Check one item against every matcher.

        Args:
            item: A model, a decoded JSON value or ``DONE_MARKER``.

        Returns:
            Whether any matcher matches it.
        """
        return any(matcher.matches(item) for matcher in self.matchers)

    def __eq__(self, other: object) -> bool:
        """Compare by the combined matchers.

        Args:
            other: Any object.

        Returns:
            Whether ``other`` is an ``AnyOf`` of equal matchers.
        """
        if not isinstance(other, AnyOf):
            return NotImplemented

        return self.matchers == other.matchers

    def __hash__(self) -> int:
        """Hash by the combined matchers.

        Returns:
            The hash.
        """
        return hash((AnyOf, self.matchers))

    def __repr__(self) -> str:
        """Show the combined matchers.

        Returns:
            The matchers joined with ``|``; a single one as
            ``AnyOf(...)``, since it is not equal to that matcher alone.
        """
        if len(self.matchers) == 1:
            return f"AnyOf({self.matchers[0]!r})"

        return " | ".join(map(repr, self.matchers))


class FieldValues(BaseMatcher):
    """Matches an item whose field has one of the given values.

    The field is read as a key of a mapping (decoded JSON), a field of a
    Pydantic model (never one of its properties) or an attribute of any
    other object. Values are compared as JSON tells them apart: ``true``
    is not ``1``. Matchers of the same field and values are equal, as
    ``EventTypes("x")`` and ``FieldValues("type", "x")`` are.

    Attributes:
        name: The field's name.
        values: The accepted values.

    Example:
        >>> FieldValues("status", "failed").matches({"status": "failed"})
        True
    """

    def __init__(self, name: str, *values: object) -> None:
        """Set the field and its values.

        Args:
            name: The field's name.
            *values: The accepted values; they must be hashable.

        Raises:
            ValueError: If no value is given, since the matcher would
                never match.
        """
        if not values:
            raise ValueError(f"FieldValues({name!r}) needs at least one value")

        self.name = name
        self.values: frozenset[object] = frozenset(values)
        self._keys: frozenset[object] = frozenset(map(_key, values))

    def matches(self, item: object) -> bool:
        """Check the field of one item.

        Args:
            item: A model, a decoded JSON value or ``DONE_MARKER``.

        Returns:
            Whether the item has the field and its value is accepted; an
            unhashable value is never accepted.
        """
        value = _field(item, self.name)

        try:
            return _key(value) in self._keys
        except TypeError:  # an unhashable value, such as an object
            return False

    def __eq__(self, other: object) -> bool:
        """Compare by field and values, whatever the class.

        Args:
            other: Any object.

        Returns:
            Whether ``other`` matches the same field and values, so the
            same items.
        """
        if not isinstance(other, FieldValues):
            return NotImplemented

        return (self.name, self._keys) == (other.name, other._keys)

    def __hash__(self) -> int:
        """Hash by field and values.

        Returns:
            The hash.
        """
        return hash((FieldValues, self.name, self._keys))

    def __repr__(self) -> str:
        """Show the field and the values.

        Returns:
            The matcher as a constructor call, values sorted.
        """
        values = ", ".join(sorted(map(repr, self.values)))
        return f"FieldValues({self.name!r}, {values})"


class EventTypes(FieldValues):
    """Matches an event by its ``type``.

    It is ``FieldValues("type", ...)``, and equal to it.

    Example:
        >>> EventTypes("message_stop").matches({"type": "message_stop"})
        True
    """

    def __init__(self, *types: str) -> None:
        """Set the accepted types.

        Args:
            *types: The accepted values of ``type``.

        Raises:
            ValueError: If no type is given.
        """
        super().__init__("type", *types)

    @property
    def types(self) -> frozenset[str]:
        """The accepted types.

        A contract test checks that each one exists in the provider's
        upstream stream schema.
        """
        return cast(frozenset[str], self.values)

    def __repr__(self) -> str:
        """Show the types.

        Returns:
            The matcher as a constructor call, types sorted.
        """
        return f"EventTypes({', '.join(sorted(map(repr, self.types)))})"


class _DoneMarker:
    """Type of ``DONE_MARKER``."""

    def __repr__(self) -> str:
        """Name the marker.

        Returns:
            ``"DONE_MARKER"``.
        """
        return "DONE_MARKER"


DONE_MARKER: Final = _DoneMarker()
"""What the stream parser passes to a matcher for ``data: [DONE]``.

``[DONE]`` is not a stream item and is never yielded.
"""


class _Done(BaseMatcher):
    """Type of ``DONE``."""

    def matches(self, item: object) -> bool:
        """Check whether the item is ``data: [DONE]``.

        Args:
            item: A model, a decoded JSON value or ``DONE_MARKER``.

        Returns:
            Whether it is ``DONE_MARKER``.
        """
        return item is DONE_MARKER

    def __repr__(self) -> str:
        """Name the matcher.

        Returns:
            ``"DONE"``.
        """
        return "DONE"


DONE: Final = _Done()
"""Matches ``data: [DONE]``, the SSE convention for the end of a stream.

Some streaming APIs send a ``[DONE]`` line instead of a typed last event;
declaring ``DONE`` as ``__stream_terminal__`` makes a stream that ends
without it an interrupted one. Neither the Responses nor the Messages API
uses it.
"""


def _is_error(value: object) -> bool:
    """Check whether the ``error`` key of an object reports an error.

    Args:
        value: The value of the top-level ``error`` key.

    Returns:
        ``True`` for ``true`` and a non-empty object or string. ``null``,
        ``false`` and empty values mean "no error", as in OpenAI batch
        output lines.
    """
    return value is True or (isinstance(value, (Mapping, str)) and bool(value))


class TopLevelError(BaseMatcher):
    """Matches a JSON object without ``type`` that reports an error.

    Its ``error`` must be ``true`` or a non-empty object or string, as
    OpenAI batch output and Ollama send it; ``null``, ``false`` and empty
    values mean "no error". An object with a ``type`` never matches, so
    a regular event may carry an ``error`` field (a failed tool call, for
    example).
    """

    def matches(self, item: object) -> bool:
        """Check one item.

        Args:
            item: A model, a decoded JSON value or ``DONE_MARKER``.

        Returns:
            Whether it is a mapping without ``type`` whose ``error``
            reports an error.
        """
        if not isinstance(item, Mapping):
            return False

        body = cast(Mapping[Any, object], item)
        return "type" not in body and _is_error(body.get("error"))

    def __eq__(self, other: object) -> bool:
        """Compare by class: every instance matches the same items.

        Args:
            other: Any object.

        Returns:
            Whether ``other`` is a ``TopLevelError``.
        """
        if not isinstance(other, TopLevelError):
            return NotImplemented

        return True

    def __hash__(self) -> int:
        """Hash by class.

        Returns:
            The hash.
        """
        return hash(TopLevelError)

    def __repr__(self) -> str:
        """Show the matcher.

        Returns:
            ``"TopLevelError()"``.
        """
        return "TopLevelError()"

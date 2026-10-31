"""Matchers of stream items: terminals, error events, failed results."""

from __future__ import annotations

from typing import Any

import pytest

from ai_api.types.base import BaseAPIType
from ai_api.types.matchers import (
    DONE,
    DONE_MARKER,
    AnyOf,
    EventTypes,
    FieldValues,
    TopLevelError,
)


class Event(BaseAPIType):
    type: str
    status: str | None = None


def test_event_types_match_mappings_and_models() -> None:
    matcher = EventTypes("message_stop", "done")

    assert matcher.types == frozenset({"message_stop", "done"})
    assert matcher.matches({"type": "message_stop"})
    assert matcher.matches(Event(type="done"))
    assert not matcher.matches({"type": "ping"})
    assert not matcher.matches(Event(type="ping"))


@pytest.mark.parametrize(
    "item",
    [{}, {"kind": "x"}, object(), None, "message_stop", DONE_MARKER, [1]],
    ids=["empty", "other-key", "object", "none", "str", "done", "list"],
)
def test_items_without_the_field_do_not_match(item: object) -> None:
    assert not EventTypes("message_stop").matches(item)


def test_unhashable_values_do_not_match() -> None:
    matcher = FieldValues("status", "failed")

    assert not matcher.matches({"status": {"code": "x"}})
    assert not matcher.matches({"status": ["failed"]})


def test_field_values_match_any_field() -> None:
    matcher = FieldValues("status", "failed", "cancelled")

    assert matcher.matches({"status": "failed"})
    assert matcher.matches(Event(type="x", status="cancelled"))
    assert not matcher.matches(Event(type="x", status="completed"))
    assert not matcher.matches(Event(type="x"))


@pytest.mark.parametrize(
    ("build", "args"),
    [(FieldValues, ("status",)), (EventTypes, ()), (AnyOf, ())],
    ids=["field-values", "event-types", "any-of"],
)
def test_a_matcher_that_never_matches_is_rejected(
    build: Any, args: tuple[Any, ...]
) -> None:
    with pytest.raises(ValueError, match="at least one"):
        build(*args)


def test_matchers_combine_with_or() -> None:
    matcher = EventTypes("error") | TopLevelError() | DONE

    assert isinstance(matcher, AnyOf)
    assert len(matcher.matchers) == 3
    assert matcher.matches({"type": "error"})
    assert matcher.matches({"error": "overloaded"})
    assert matcher.matches(DONE_MARKER)
    assert not matcher.matches({"type": "delta"})


def test_matchers_compare_by_value() -> None:
    assert EventTypes("a", "b") == EventTypes("b", "a")
    assert hash(EventTypes("a")) == hash(EventTypes("a"))
    assert EventTypes("a") != FieldValues("type", "a")
    assert FieldValues("s", "a") == FieldValues("s", "a")
    assert FieldValues("s", "a") != FieldValues("t", "a")
    assert TopLevelError() == TopLevelError()
    assert hash(TopLevelError()) == hash(TopLevelError())
    assert EventTypes("a") | DONE == EventTypes("a") | DONE
    assert hash(EventTypes("a") | DONE) == hash(EventTypes("a") | DONE)
    assert EventTypes("a") != "a"
    assert TopLevelError() != "a"
    assert AnyOf(DONE) != DONE


def test_matchers_show_what_they_match() -> None:
    matcher = EventTypes("b", "a") | FieldValues("status", "failed")

    assert repr(matcher) == (
        "EventTypes('a', 'b') | FieldValues('status', 'failed')"
    )
    assert repr(TopLevelError() | DONE) == "TopLevelError() | DONE"
    assert repr(DONE_MARKER) == "DONE_MARKER"


def test_done_matches_only_the_marker() -> None:
    assert DONE.matches(DONE_MARKER)
    assert not DONE.matches("[DONE]")
    assert not DONE.matches({"type": "done"})


@pytest.mark.parametrize(
    "item",
    [
        {"error": True},
        {"error": "overloaded"},
        {"error": {"message": "boom"}},
    ],
    ids=["true", "string", "object"],
)
def test_top_level_error_matches_an_error(item: object) -> None:
    assert TopLevelError().matches(item)


@pytest.mark.parametrize(
    "item",
    [
        {"error": None},
        {"error": False},
        {"error": ""},
        {"error": {}},
        {"error": 1},
        {"error": ["x"]},
        {"id": "batch_1"},
        {"type": "tool_result", "error": "file not found"},
        {"type": "x", "error": True},
        ["error"],
        Event(type="error"),
    ],
    ids=[
        "null",
        "false",
        "empty-string",
        "empty-object",
        "number",
        "list",
        "no-error",
        "typed-string",
        "typed-true",
        "not-a-mapping",
        "model",
    ],
)
def test_top_level_error_ignores_everything_else(item: object) -> None:
    assert not TopLevelError().matches(item)

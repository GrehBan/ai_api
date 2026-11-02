"""One guard against cycles and deep nesting for every value traversal.

The body normalizer and the file helpers of methods walk values that a
caller built, and Pydantic accepts a value that contains itself in a
field typed ``Any``. Without a guard a cycle ends in ``RecursionError``,
and the walks would disagree on how deep a value may be.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Final

MAX_DEPTH: Final = 253
"""Deepest nesting of containers accepted in a body field.

orjson refuses more than 254 nested containers, and the body object is
one of them; a deeper field fails with a clear ``ValueError`` instead
of an encoder ``TypeError``. The file helpers of methods use the same
limit, so a value is accepted or refused the same way at construction
and when it is sent.
"""


@contextmanager
def guard_traversal(value: object, ancestors: set[int]) -> Generator[None]:
    """Enter a container during a recursive walk, refusing cycles.

    The identities of the containers on the current path are kept, not
    of every container seen, so one value in two places is not a cycle.

    Args:
        value: The container being entered.
        ancestors: Identities of the containers on the current path; the
            container is added for the time of the block.

    Yields:
        Nothing; the walk goes into the container inside the block.

    Raises:
        ValueError: If the container is one of its own ancestors, or it
            would be nested deeper than ``MAX_DEPTH``.
    """
    identity = id(value)

    if identity in ancestors:
        raise ValueError("Circular reference detected")

    if len(ancestors) >= MAX_DEPTH:
        raise ValueError(f"Container is nested deeper than {MAX_DEPTH} levels")

    ancestors.add(identity)

    try:
        yield
    finally:
        ancestors.remove(identity)

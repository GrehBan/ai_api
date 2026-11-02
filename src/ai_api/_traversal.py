from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Final

MAX_DEPTH: Final = 253
"""Maximum number of enclosing containers allowed in a body value."""


@contextmanager
def guard_traversal(
    value: object,
    ancestors: set[int],
    *,
    max_depth: int = MAX_DEPTH,
) -> Generator[None]:
    """Guard recursive traversal against cycles and excessive depth.

    Args:
        value: The container being traversed.
        ancestors: Identities of containers in the current recursion path.
        max_depth: Maximum number of enclosing containers.

    Raises:
        ValueError: If a circular reference or excessive depth is found.
    """
    identity = id(value)

    if identity in ancestors:
        raise ValueError("Circular reference detected")

    if len(ancestors) >= max_depth:
        raise ValueError(f"Container is nested deeper than {max_depth} levels")

    ancestors.add(identity)

    try:
        yield
    finally:
        ancestors.remove(identity)

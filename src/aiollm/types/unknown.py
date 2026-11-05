"""Values of a ``type`` the library does not model yet."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, Final

from pydantic import ConfigDict

from aiollm.types.base import BaseAPIType

logger = logging.getLogger(__name__)

UNKNOWN: Final = "unknown"
"""Tag of a value whose ``type`` the library does not know yet."""


class BaseUnknown(BaseAPIType):
    """A value of a ``type`` the library does not model yet.

    Providers add output items, content blocks and stream events over
    time. Rejecting them would break a working stream on the first new
    event, and dropping them would lose data, so they are kept whole:
    every field the provider sent stays available as an extra field.

    Attributes:
        type: The provider's type of the value.
    """

    model_config = ConfigDict(extra="allow")

    type: str


def by_type(tags: Mapping[str, str]) -> Callable[[Any], str]:
    """Build a pydantic discriminator that falls back to ``UNKNOWN``.

    Args:
        tags: Tag of each known ``type`` value; several values may share
            one tag.

    Returns:
        A function that returns the tag of a raw mapping or a model, and
        ``UNKNOWN`` for a missing or unknown ``type``.
    """

    def discriminate(value: Any) -> str:
        """Tag a value by its ``type``.

        Args:
            value: A raw mapping during validation, or a model.

        Returns:
            The tag of the value's ``type``, or ``UNKNOWN``.
        """
        kind: object = (
            value.get("type")
            if isinstance(value, Mapping)
            else getattr(value, "type", None)
        )
        if not isinstance(kind, str):
            return UNKNOWN

        tag = tags.get(kind)

        if tag is None:
            # A type the library does not model yet, such as a new event.
            logger.debug("Unknown type %r is kept as an unknown item", kind)
            return UNKNOWN

        return tag

    return discriminate

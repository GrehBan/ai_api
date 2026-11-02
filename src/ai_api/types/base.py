"""Base classes for API data models."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Final

from pydantic import (
    BaseModel,
    ConfigDict,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    model_serializer,
)

REQUEST_CONTEXT: Final[Mapping[str, bool]] = MappingProxyType(
    {"ai_api.request": True}
)
"""Serialization context of a request body.

A method dumps its body with it, so a model of this library put into
the body at run time (a response item sent back, say) leaves out the
fields that hold ``None`` by default: they were absent in the response.
"""


class BaseAPIType(BaseModel):
    """Base class for request and response payload models.

    Unknown fields are ignored, so new fields added by the provider do
    not break validation. Fields can be populated by name or alias. Give
    a wire name with ``validation_alias`` and ``serialization_alias``
    rather than ``alias``: then the constructor takes the field name in
    both mypy and pyright (with ``alias`` they disagree). A field of a
    type Pydantic cannot validate fails at class definition: payloads
    are JSON, so such a field is a mistake (only the ``raw`` of shared
    results allows any type, see ``BaseResponse``).

    Attributes:
        __upstream_schema__: Name of the definition in the provider's
            upstream JSON schema that the model mirrors, several names
            for a model that covers several definitions, or ``()`` for a
            model without one. ``None`` (the default) means not set; a
            provider's contract test requires it on every provider model
            and checks that the model accepts what that schema allows.
    """

    __upstream_schema__: ClassVar[str | tuple[str, ...] | None] = None

    model_config = ConfigDict(
        extra="ignore",
        validate_by_name=True,
        validate_by_alias=True,
    )

    @model_serializer(mode="wrap")
    def _serialize(
        self,
        handler: SerializerFunctionWrapHandler,
        info: SerializationInfo,
    ) -> Any:
        """Dump the model; in a request body, without unset defaults.

        A field that holds ``None`` and defaults to ``None`` is left out
        of a request body (``REQUEST_CONTEXT``): that is how a response
        model says a field was absent, so a model sent back keeps the
        shape it came in. Every other dump is Pydantic's own.

        Args:
            handler: Pydantic's serializer of the model.
            info: The options of the dump.

        Returns:
            What Pydantic dumps, less those fields in a request body.
        """
        data = handler(self)
        context = info.context

        if not (
            isinstance(context, Mapping)
            and context.get("ai_api.request")
            and isinstance(data, dict)
        ):
            return data

        for name, field in type(self).model_fields.items():
            if field.default is None and getattr(self, name) is None:
                key = (
                    field.serialization_alias or field.alias or name
                    if info.by_alias
                    else name
                )
                data.pop(key, None)

        return data


class BaseAPITypeExtra(BaseAPIType):
    """Base class for response models that keep the fields they do not model.

    A field the model does not declare is validated as nothing and kept
    in ``model_extra``, so whatever the provider sent is still there:
    the provider types model only what the library reads, and the rest
    is reached through ``raw`` of a shared result. Everything else is as
    for ``BaseAPIType``.

    Example:
        >>> class Reply(BaseAPITypeExtra):
        ...     text: str
        >>> Reply.model_validate({"text": "hi", "seed": 7}).model_extra
        {'seed': 7}
    """

    model_config = ConfigDict(
        extra="allow",
        validate_by_name=True,
        validate_by_alias=True,
    )

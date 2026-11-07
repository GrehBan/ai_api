"""Parts of a turn's content: text, images and documents."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Final, Literal, Self, get_args

from pydantic import (
    MISSING,
    ConfigDict,
    Discriminator,
    Field,
    model_validator,
)

from aiollm.types.base import BaseAPIType
from aiollm.types.files import guess_content_type

type ImageMediaType = Literal[
    "image/jpeg", "image/png", "image/gif", "image/webp"
]
"""The image formats both providers take."""

type DocumentMediaType = Literal["application/pdf"]
"""The document formats both providers take: PDF."""

IMAGE_MEDIA_TYPES: Final[frozenset[str]] = frozenset(
    get_args(ImageMediaType.__value__)
)
"""The values of ``ImageMediaType``."""

URL_PATTERN: Final = (
    r"^https?://(?:[^/?#\s@]*@)?"
    r"(?:\[[0-9A-Fa-f:.]+\]|[^/?#\s@:\[\]]+)"
    r"(?::\d*)?(?:[/?#]\S*)?$"
)
"""An ``http`` or ``https`` URL with a host, after any user and before
any port."""

_CONFIG: Final = ConfigDict(
    extra="forbid",
    frozen=True,
    ser_json_bytes="base64",
    val_json_bytes="base64",
)
"""Immutable, unknown fields rejected, bytes as base64 in JSON."""


class TextPart(BaseAPIType):
    """Text in a turn of several parts.

    Attributes:
        type: Always ``"text"``.
        text: The text.

    Example:
        >>> TextPart(text="What is in this picture?")
    """

    model_config = _CONFIG

    type: Literal["text"] = "text"
    text: str


class _SourcePart(BaseAPIType):
    """A part whose content is its bytes or a URL, exactly one of them.

    Attributes:
        data: The content, sent inline (base64).
        url: Where the provider downloads it from.
    """

    model_config = _CONFIG

    data: bytes | MISSING = Field(default=MISSING, repr=False)
    url: str | MISSING = Field(default=MISSING, pattern=URL_PATTERN)

    @model_validator(mode="after")
    def _one_source(self) -> Self:
        """Check that the part has its bytes or a URL, not both.

        Returns:
            The part.

        Raises:
            ValueError: Neither or both are given, or ``data`` is empty.
        """
        if (self.data is MISSING) == (self.url is MISSING):
            message = "give either data or url"
            raise ValueError(message)
        if self.data is not MISSING and not self.data:
            message = "data is empty"
            raise ValueError(message)
        return self


class ImagePart(_SourcePart):
    """An image in a turn of the end user.

    Give its bytes with their ``media_type``, or a URL the provider
    downloads it from. JPEG, PNG, GIF and WebP are taken by both
    providers; each limits the size and the number of images.

    Attributes:
        type: Always ``"image"``.
        data: The image, sent inline (base64).
        url: Where the provider downloads it from.
        media_type: The format of ``data``; only with ``data``.

    Example:
        >>> ImagePart(data=png, media_type="image/png")
        >>> ImagePart(url="https://example.com/cat.jpg")
    """

    type: Literal["image"] = "image"
    media_type: ImageMediaType | MISSING = MISSING

    @model_validator(mode="after")
    def _media_type_with_data(self) -> Self:
        """Check that ``media_type`` comes with ``data``, and only then.

        Returns:
            The part.

        Raises:
            ValueError: ``data`` without ``media_type``, or a
                ``media_type`` with a URL.
        """
        if (self.data is MISSING) != (self.media_type is MISSING):
            message = "media_type goes with data, and only with it"
            raise ValueError(message)
        return self

    @classmethod
    async def load(
        cls,
        path: str | Path,
        media_type: ImageMediaType | MISSING = MISSING,
    ) -> Self:
        """Read an image file whole, in a worker thread.

        The format is not checked against the file's content, and the
        bytes are kept in the part (base64 in the request body).

        Args:
            path: The file.
            media_type: Its format; guessed from the file name's
                extension when left out.

        Returns:
            The part, with the file's bytes.

        Raises:
            OSError: The file cannot be read.
            ValueError: The format is not given and the extension names
                none of ``ImageMediaType``.
        """
        path = Path(path)
        if media_type is MISSING:
            guessed = guess_content_type(path.name)
            if guessed not in IMAGE_MEDIA_TYPES:
                message = (
                    f"cannot tell the image format of {path.name!r}; "
                    "give media_type"
                )
                raise ValueError(message)
            media_type = guessed  # type: ignore[assignment]
        data = await asyncio.to_thread(path.read_bytes)
        return cls(data=data, media_type=media_type)


class DocumentPart(_SourcePart):
    """A PDF document in a turn of the end user.

    Give its bytes, or a URL the provider downloads it from. Both
    providers read the text and the pages as images; each limits the
    size and the number of pages.

    Attributes:
        type: Always ``"document"``.
        data: The document, sent inline (base64).
        url: Where the provider downloads it from.
        media_type: Always ``"application/pdf"``.
        filename: The document's name, OpenAI's ``filename`` and
            Anthropic's ``title``; only with ``data``.

    Example:
        >>> DocumentPart(data=pdf, filename="report.pdf")
    """

    type: Literal["document"] = "document"
    media_type: DocumentMediaType = "application/pdf"
    filename: Annotated[str, Field(min_length=1)] | MISSING = MISSING

    @model_validator(mode="after")
    def _filename_with_data(self) -> Self:
        """Check that ``filename`` comes only with ``data``.

        Returns:
            The part.

        Raises:
            ValueError: A ``filename`` with a URL.
        """
        if self.filename is not MISSING and self.data is MISSING:
            message = "filename goes with data only"
            raise ValueError(message)
        return self

    @classmethod
    async def load(cls, path: str | Path) -> Self:
        """Read a PDF file whole, in a worker thread.

        The content is not checked: any file is sent as a PDF, and the
        provider refuses one that is not. The bytes are kept in the part
        (base64 in the request body).

        Args:
            path: The file; its name becomes ``filename``.

        Returns:
            The part, with the file's bytes.

        Raises:
            OSError: The file cannot be read.
        """
        path = Path(path)
        data = await asyncio.to_thread(path.read_bytes)
        return cls(data=data, filename=path.name)


type ContentPart = Annotated[
    TextPart | ImagePart | DocumentPart, Discriminator("type")
]
"""A part of a user turn: text, an image or a document."""

"""Request body encoding for ``aiohttp``."""

from __future__ import annotations

import asyncio
import inspect
import io
import math
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, cast

from aiohttp import BytesPayload, FormData

from ai_api.enums import BodyFormat
from ai_api.http.exceptions.transport import APIRequestError
from ai_api.http.headers import JSON_CONTENT_TYPE
from ai_api.serialization.json import JsonDumps, dumps
from ai_api.serialization.normalize import BodyNormalizer
from ai_api.serialization.values import scalar_text
from ai_api.types.files import BytesFile, InputFile, closing_source

if TYPE_CHECKING:
    from types import TracebackType

    from ai_api.serialization.protocols import IValueFormatter


async def _read_source(
    file: InputFile,
    source: AsyncIterator[bytes],
) -> AsyncGenerator[bytes]:
    """Read an upload, marking failures of its source.

    aiohttp reports any error raised while it writes the body as a
    connection error, which would be retried. Wrapping it here lets the
    session tell a broken source from a broken connection. The source is
    closed by ``closing_source`` when the reading ends, also when it is
    cut short; a cancellation goes through as it is.

    Args:
        file: The file being read.
        source: The iterator ``file.open()`` returned.

    Yields:
        Chunks of the file.

    Raises:
        APIRequestError: If the source fails, also while it is closed.
    """
    try:
        async with closing_source(source):
            async for chunk in source:
                yield chunk
    except Exception as exc:
        raise APIRequestError(
            f"Failed to read upload {file.filename!r}: {exc}"
        ) from exc


async def _close_reader(
    reader: AsyncGenerator[bytes],
    file: InputFile,
    source: AsyncIterator[bytes],
    error: BaseException | None,
) -> None:
    """Close an upload reader once nothing is running it.

    aiohttp does not close the iterator of a body it stops writing; it
    cancels the writing task without waiting for it. Until that task has
    taken the cancellation, the reader counts as running and refuses
    ``aclose()``. A reader that never started (the request failed before
    its body was sent) has not run its body, so its source is closed
    here.

    Args:
        reader: A reader made by ``_read_source``.
        file: The file being read.
        source: The iterator the reader reads.
        error: The exception that ends the attempt, if any; a failing
            close is added to it as a note instead of hiding it.

    Raises:
        APIRequestError: If the source fails to close and the attempt has
            no exception of its own.
    """
    started = inspect.getasyncgenstate(reader) != inspect.AGEN_CREATED

    while inspect.getasyncgenstate(reader) == inspect.AGEN_RUNNING:
        await asyncio.sleep(0)

    try:
        await reader.aclose()

        if not started:
            try:
                async with closing_source(source):
                    pass
            except Exception as exc:
                raise APIRequestError(
                    f"Failed to close upload {file.filename!r}: {exc}"
                ) from exc
    except Exception as close_error:
        if error is None:
            raise
        error.add_note(
            f"Closing upload {file.filename!r} also failed: {close_error!r}"
        )


class AiohttpBodyBuilder:
    """Turns a method's body fields into ``aiohttp`` request arguments.

    Values are normalized first, then encoded as JSON bytes or as
    ``FormData``, passed as ``data=``. Files become multipart parts with
    their file name and content type.
    """

    def __init__(
        self,
        json_dumps: JsonDumps = dumps,
        normalizer: BodyNormalizer | None = None,
    ) -> None:
        """Create the builder.

        Args:
            json_dumps: Encoder for JSON bodies and for mappings sent
                inside form fields.
            normalizer: Value normalizer; a default one is used when
                ``None``.
        """
        self._json_dumps = json_dumps
        self._normalizer = normalizer or BodyNormalizer()

    def build(
        self,
        body: Mapping[str, Any],
        body_format: BodyFormat,
        formatter: IValueFormatter,
        *,
        uploads: AsyncExitStack,
    ) -> dict[str, Any]:
        """Encode the body.

        Args:
            body: Body fields of the method.
            body_format: How the body must be encoded.
            formatter: The method's ``value_formatter``, which writes the
                dates, times and durations.
            uploads: The stack of the attempt; every upload reader is
                registered on it and closed when the attempt ends.

        Returns:
            Keyword arguments for ``ClientSession.request``; empty for
            ``BodyFormat.NONE``.

        Raises:
            TypeError: If files or binary data are sent as JSON or as a
                URL-encoded form, a text stream is uploaded, or a value
                cannot be encoded as JSON.
            ValueError: If the body format is unknown, a value is NaN or
                infinite, the formatter rejects a value, or
                ``BodyFormat.NONE`` is given a non-empty body, which would
                be dropped.
            APIRequestError: If a file to upload does not exist or
                cannot be opened.
        """
        if body_format is BodyFormat.NONE:
            if body:
                names = ", ".join(sorted(body))
                message = f"A request without a body cannot send: {names}"
                raise ValueError(message)
            return {}

        normalized = self._normalizer.normalize_mapping(body, formatter)

        match body_format:
            case BodyFormat.JSON:
                self._check_json(normalized)
                return {
                    "data": BytesPayload(
                        self._dumps(normalized),
                        content_type=JSON_CONTENT_TYPE,
                    )
                }

            case BodyFormat.FORM:
                return {
                    "data": self._build_form(
                        normalized,
                        formatter,
                        uploads,
                        multipart=False,
                    )
                }

            case BodyFormat.MULTIPART:
                return {
                    "data": self._build_form(
                        normalized,
                        formatter,
                        uploads,
                        multipart=True,
                    )
                }

        raise ValueError(f"Unsupported body format: {body_format}")

    def _check_json(self, value: Any) -> None:
        """Reject values that cannot be represented in JSON.

        Args:
            value: A normalized body value; containers are checked
                recursively.

        Raises:
            TypeError: If the value is or contains a file or binary data.
            ValueError: If the value is or contains NaN or infinity, which
                JSON cannot represent.
        """
        if isinstance(
            value,
            (InputFile, io.IOBase, bytes, bytearray, memoryview),
        ):
            raise TypeError("Binary data cannot be sent as JSON")

        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("NaN and infinity cannot be sent as JSON")

        if isinstance(value, Mapping):
            for item in value.values():
                self._check_json(item)

        elif isinstance(value, (list, tuple)):
            for item in value:
                self._check_json(item)

    def _dumps(self, value: Any) -> bytes:
        """Encode a checked value as JSON.

        Args:
            value: A normalized value.

        Returns:
            The encoded JSON; a ``str`` from an encoder such as
            ``json.dumps`` is encoded as UTF-8.

        Raises:
            TypeError: If the encoder does not support a value, such as a
                set, a ``Decimal``, a non-string key or an integer out of
                range.
        """
        try:
            encoded = cast(bytes | str, self._json_dumps(value))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Request body cannot be encoded as JSON: {exc}"
            ) from exc

        if isinstance(encoded, str):
            return encoded.encode()

        return encoded

    def _build_form(
        self,
        body: Mapping[str, Any],
        formatter: IValueFormatter,
        uploads: AsyncExitStack,
        *,
        multipart: bool,
    ) -> FormData:
        """Build a URL-encoded or multipart form.

        Args:
            body: Normalized body fields.
            formatter: The method's ``value_formatter``.
            uploads: The stack the upload readers are registered on.
            multipart: Whether to produce ``multipart/form-data``.

        Returns:
            The form to pass as ``data``.
        """
        form = FormData(
            default_to_multipart=multipart,
        )

        for key, value in body.items():
            self._add_field(form, key, value, formatter, uploads, multipart)

        return form

    def _add_field(
        self,
        form: FormData,
        key: str,
        value: Any,
        formatter: IValueFormatter,
        uploads: AsyncExitStack,
        multipart: bool,
    ) -> None:
        """Add one body field to the form.

        Lists add the field once per item, files become file parts,
        mappings are sent as JSON strings and other values by the rule of
        ``scalar_text`` (booleans as ``"true"``/``"false"``). A ``None``
        field is not sent.

        Args:
            form: The form being built.
            key: Field name.
            value: Normalized field value.
            formatter: The method's ``value_formatter``.
            uploads: The stack the upload readers are registered on.
            multipart: Whether the form is multipart.

        Raises:
            TypeError: If a file is added to a non-multipart form, a
                mapping cannot be encoded as JSON, or a list contains a
                list, which a repeated field would flatten.
            ValueError: If a value is NaN or infinite, or a list contains
                ``None``, which a form field cannot express.
            APIRequestError: If a file to upload does not exist.
        """
        if value is None:
            return

        if isinstance(value, (list, tuple)):
            for item in value:
                if item is None:
                    message = f"Form field {key!r} cannot send None in a list"
                    raise ValueError(message)
                if isinstance(item, (list, tuple)):
                    message = f"Form field {key!r} cannot send nested lists"
                    raise TypeError(message)

                self._add_field(form, key, item, formatter, uploads, multipart)
            return

        file = InputFile.from_value(value)

        if file is not None:
            if not multipart:
                raise TypeError("Files require multipart encoding")

            self._add_file(form, key, file, uploads)
            return

        if isinstance(value, Mapping):
            self._check_json(value)
            form.add_field(key, self._dumps(value).decode("utf-8"))
            return

        form.add_field(
            key,
            scalar_text(value, formatter, label=f"Form field {key!r}"),
        )

    @staticmethod
    def _add_file(
        form: FormData,
        key: str,
        file: InputFile,
        uploads: AsyncExitStack,
    ) -> None:
        """Add a file part to a multipart form.

        A ``BytesFile`` is sent with its length; any other file is opened
        now and read by a reader registered on ``uploads``, so the reader
        and its source are closed when the attempt ends, also when the
        body is never sent.

        Args:
            form: The multipart form being built.
            key: Field name.
            file: The file to upload.
            uploads: The stack the upload reader is registered on.

        Raises:
            APIRequestError: If the file cannot be read or opened.
        """
        try:
            file.check()
        except OSError as exc:
            raise APIRequestError(
                f"Cannot read upload {file.filename!r}: {exc}"
            ) from exc

        content: bytes | AsyncGenerator[bytes]

        if isinstance(file, BytesFile):
            content = file.data
        else:
            try:
                source = file.open()
            except Exception as exc:
                raise APIRequestError(
                    f"Cannot open upload {file.filename!r}: {exc}"
                ) from exc

            reader = _read_source(file, source)

            async def close(
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                traceback: TracebackType | None,
            ) -> bool:
                """Close the reader when the attempt ends.

                Args:
                    exc_type: The type of the attempt's exception.
                    exc: The attempt's exception, if any.
                    traceback: Its traceback.

                Returns:
                    ``False``: the exception is never suppressed.
                """
                await _close_reader(reader, file, source, exc)
                return False

            uploads.push_async_exit(close)
            content = reader

        form.add_field(
            key,
            content,
            filename=file.filename or key,
            content_type=file.content_type,
        )

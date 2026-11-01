"""Request body encoding for ``aiohttp``."""

from __future__ import annotations

import io
import math
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast

from aiohttp import BytesPayload, FormData

from ai_api.enums import BodyFormat
from ai_api.http.exceptions.transport import APIRequestError
from ai_api.http.headers import JSON_CONTENT_TYPE
from ai_api.serialization.json import JsonDumps, dumps
from ai_api.serialization.normalize import BodyNormalizer
from ai_api.types.files import BytesFile, InputFile


async def _read_source(file: InputFile) -> AsyncIterator[bytes]:
    """Read an upload, marking failures of its source.

    aiohttp reports any error raised while it writes the body as a
    connection error, which would be retried. Wrapping it here lets the
    session tell a broken source from a broken connection.

    Args:
        file: The file to read.

    Yields:
        Chunks of the file.

    Raises:
        APIRequestError: If the source fails.
    """
    try:
        async for chunk in file.open():
            yield chunk
    except Exception as exc:
        raise APIRequestError(
            f"Failed to read upload {file.filename!r}: {exc}"
        ) from exc


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
    ) -> dict[str, Any]:
        """Encode the body.

        Args:
            body: Body fields of the method.
            body_format: How the body must be encoded.

        Returns:
            Keyword arguments for ``ClientSession.request``; empty for
            ``BodyFormat.NONE``.

        Raises:
            TypeError: If files or binary data are sent as JSON or as a
                URL-encoded form, a text stream is uploaded, or a value
                cannot be encoded as JSON.
            ValueError: If the body format is unknown, a JSON value is NaN
                or infinite, or ``BodyFormat.NONE`` is given a non-empty
                body, which would be dropped.
            APIRequestError: If a file to upload does not exist.
        """
        if body_format is BodyFormat.NONE:
            if body:
                names = ", ".join(sorted(body))
                message = f"A request without a body cannot send: {names}"
                raise ValueError(message)
            return {}

        normalized = self._normalizer.normalize_mapping(body)

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
                        multipart=False,
                    )
                }

            case BodyFormat.MULTIPART:
                return {
                    "data": self._build_form(
                        normalized,
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
        *,
        multipart: bool,
    ) -> FormData:
        """Build a URL-encoded or multipart form.

        Args:
            body: Normalized body fields.
            multipart: Whether to produce ``multipart/form-data``.

        Returns:
            The form to pass as ``data``.
        """
        form = FormData(
            default_to_multipart=multipart,
        )

        for key, value in body.items():
            self._add_field(form, key, value, multipart)

        return form

    def _add_field(
        self,
        form: FormData,
        key: str,
        value: Any,
        multipart: bool,
    ) -> None:
        """Add one body field to the form.

        Lists add the field once per item, files become file parts,
        mappings are sent as JSON strings and booleans as
        ``"true"``/``"false"``.

        Args:
            form: The form being built.
            key: Field name.
            value: Normalized field value.
            multipart: Whether the form is multipart.

        Raises:
            TypeError: If a file is added to a non-multipart form, a text
                stream is uploaded, or a mapping cannot be encoded as
                JSON.
            ValueError: If a mapping contains NaN or infinity.
            APIRequestError: If a file to upload does not exist.
        """
        if value is None:
            return

        if isinstance(value, (list, tuple)):
            for item in value:
                self._add_field(form, key, item, multipart)
            return

        file = InputFile.from_value(value)

        if file is not None:
            if not multipart:
                raise TypeError("Files require multipart encoding")

            try:
                file.check()
            except OSError as exc:
                raise APIRequestError(
                    f"Cannot read upload {file.filename!r}: {exc}"
                ) from exc

            form.add_field(
                key,
                (
                    file.data
                    if isinstance(file, BytesFile)
                    else _read_source(file)
                ),
                filename=file.filename or key,
                content_type=file.content_type,
            )
            return

        if isinstance(value, io.IOBase):
            raise TypeError("Only binary streams can be uploaded")

        if isinstance(value, Mapping):
            self._check_json(value)

            value = self._dumps(value).decode("utf-8")

        elif isinstance(value, bool):
            value = str(value).lower()

        else:
            value = str(value)

        form.add_field(key, value)

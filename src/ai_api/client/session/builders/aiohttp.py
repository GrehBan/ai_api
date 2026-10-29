"""Request body encoding for ``aiohttp``."""

import io
from collections.abc import Mapping
from typing import Any

from aiohttp import FormData

from ai_api.client.serialization.body import BodyNormalizer
from ai_api.client.serialization.json import JsonDumps, dumps
from ai_api.methods.enums import BodyFormat
from ai_api.types.files import BytesFile, InputFile


class AiohttpBodyBuilder:
    """Turns a method's body fields into ``aiohttp`` request arguments.

    Values are normalized first, then encoded as ``json=`` or as
    ``data=FormData``. Files become multipart parts with their file name
    and content type.
    """

    def __init__(
        self,
        json_dumps: JsonDumps = dumps,
        normalizer: BodyNormalizer | None = None,
    ) -> None:
        """Create the builder.

        Args:
            json_dumps: Encoder for mappings sent inside form fields.
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
                URL-encoded form, or a text stream is uploaded.
            ValueError: If the body format is unknown.
        """
        if body_format == BodyFormat.NONE:
            return {}

        normalized = self._normalizer.normalize_mapping(body)

        match body_format:
            case BodyFormat.JSON:
                self._check_json(normalized)
                return {"json": normalized}

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

        raise ValueError(
            f"Unsupported body format: {body_format}"
        )

    def _check_json(self, value: Any) -> None:
        """Reject values that cannot be represented in JSON.

        Args:
            value: A normalized body value; containers are checked
                recursively.

        Raises:
            TypeError: If the value is or contains a file or binary data.
        """
        if isinstance(
            value,
            (InputFile, io.IOBase, bytes, bytearray, memoryview),
        ):
            raise TypeError(
                "Binary data cannot be sent as JSON"
            )

        if isinstance(value, Mapping):
            for item in value.values():
                self._check_json(item)

        elif isinstance(value, (list, tuple)):
            for item in value:
                self._check_json(item)

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
            TypeError: If a file is added to a non-multipart form or a
                text stream is uploaded.
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
                raise TypeError(
                    "Files require multipart encoding"
                )

            form.add_field(
                key,
                (
                    file.data
                    if isinstance(file, BytesFile)
                    else file.open()
                ),
                filename=file.filename or key,
                content_type=file.content_type,
            )
            return

        if isinstance(value, io.IOBase):
            raise TypeError(
                "Only binary streams can be uploaded"
            )

        if isinstance(value, Mapping):
            self._check_json(value)

            value = self._json_dumps(value).decode("utf-8")

        elif isinstance(value, bool):
            value = str(value).lower()

        else:
            value = str(value)

        form.add_field(key, value)
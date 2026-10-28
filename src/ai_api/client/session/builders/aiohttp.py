import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aiohttp import FormData

from ai_api.client.serialization.body import BodyNormalizer
from ai_api.client.serialization.json import JsonDumps, dumps
from ai_api.methods.enums import BodyFormat


class AiohttpBodyBuilder:
    def __init__(
        self,
        json_dumps: JsonDumps = dumps,
        normalizer: BodyNormalizer | None = None,
    ) -> None:
        self._json_dumps = json_dumps
        self._normalizer = normalizer or BodyNormalizer()

    def build(
        self,
        body: Mapping[str, Any],
        body_format: BodyFormat,
    ) -> dict[str, Any]:
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
        if isinstance(
            value,
            (io.IOBase, bytes, bytearray, memoryview),
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
        if value is None:
            return

        if isinstance(value, (list, tuple)):
            for item in value:
                self._add_field(form, key, item, multipart)
            return

        if isinstance(value, io.IOBase):
            if not multipart:
                raise TypeError(
                    "File streams require multipart encoding"
                )

            filename = getattr(value, "name", None)

            if isinstance(filename, str):
                filename = Path(filename).name
            else:
                filename = key

            form.add_field(
                key,
                value,
                filename=filename,
            )
            return

        if isinstance(value, (bytes, bytearray, memoryview)):
            if not multipart:
                raise TypeError(
                    "Binary data requires multipart encoding"
                )

            form.add_field(
                key,
                bytes(value),
                filename=key,
            )
            return

        if isinstance(value, Mapping):
            self._check_json(value)

            serialized = self._json_dumps(value)

            value = (
                serialized.decode("utf-8")
                if isinstance(serialized, bytes)
                else serialized
            )

        elif isinstance(value, bool):
            value = str(value).lower()

        else:
            value = str(value)

        form.add_field(key, value)
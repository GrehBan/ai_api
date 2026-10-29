"""Parser for fully read responses."""

from typing import cast

from pydantic import BaseModel, ValidationError

from ai_api.client.exceptions.transport import APIResponseError
from ai_api.client.response.base import BufferedResponse
from ai_api.client.serialization.json import JsonLoads, loads
from ai_api.methods.enums import ResponseFormat
from ai_api.methods.protocols import IAPIMethod


class BufferedResponseParser:
    """Parses JSON, text, bytes and empty responses."""

    def __init__(self, json_loads: JsonLoads = loads) -> None:
        """Create the parser.

        Args:
            json_loads: JSON decoder used when ``parse`` is not given one.
        """
        self._json_loads = json_loads

    def parse[T](
        self,
        method: IAPIMethod[T],
        buffered: BufferedResponse,
        json_loads: JsonLoads | None = None,
    ) -> T:
        """Parse the body according to ``method.response_format``.

        Args:
            method: The method that was sent.
            buffered: The successful response.
            json_loads: JSON decoder to use instead of the parser's own.
                With the default decoder the body is validated by Pydantic
                directly in JSON mode; a custom decoder's result is
                validated in Python mode, where strict fields reject JSON
                strings for dates, UUIDs and enums.

        Returns:
            A validated model for JSON, ``str`` for TEXT, ``bytes`` for
            BYTES and ``None`` for NONE.

        Raises:
            APIResponseError: If the body is not valid JSON or fails
                validation, fails decoding, or is not empty when NONE is
                expected.
            TypeError: If a JSON method has no model response type.
            ValueError: If a custom decoder rejects the body or the format
                is not supported for buffered responses.
        """
        if json_loads is None:
            json_loads = self._json_loads

        raw = buffered.body

        match method.response_format:
            case ResponseFormat.JSON:
                model_type = method.response_type

                if model_type is None or not issubclass(model_type, BaseModel):
                    raise TypeError("JSON response requires a response model")

                # Pydantic's JSON mode accepts JSON representations that
                # Python mode rejects for strict fields (dates, UUIDs and
                # enums as strings), so it is used whenever the decoder is
                # the default one. A custom decoder needs Python mode.
                try:
                    if json_loads is loads:
                        return model_type.model_validate_json(raw)

                    return model_type.model_validate(json_loads(raw))
                except ValidationError as exc:
                    raise APIResponseError(
                        "Failed to validate JSON response",
                        status=buffered.status,
                    ) from exc

            case ResponseFormat.TEXT:
                try:
                    return cast(
                        T,
                        raw.decode(buffered.charset or "utf-8"),
                    )
                except (UnicodeError, LookupError) as exc:
                    raise APIResponseError(
                        "Failed to decode text response",
                        status=buffered.status,
                    ) from exc

            case ResponseFormat.BYTES:
                return cast(T, raw)

            case ResponseFormat.NONE:
                if raw:
                    raise APIResponseError(
                        "Expected an empty response body",
                        status=buffered.status,
                    )

                return cast(T, None)

            case _:
                raise ValueError(
                    "Unsupported buffered response format: "
                    f"{method.response_format}"
                )

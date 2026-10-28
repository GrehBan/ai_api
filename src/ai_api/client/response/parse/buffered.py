from typing import cast

from pydantic import ValidationError

from ai_api.client.serialization.json import JsonLoads, loads
from ai_api.methods.enums import ResponseFormat
from ai_api.methods.protocols import IAPIMethod
from ai_api.client.response.base import BufferedResponse
from ai_api.client.exceptions.transport import APIResponseError


class BufferedResponseParser:
    def __init__(self, json_loads: JsonLoads = loads) -> None:
        self._json_loads = json_loads

    def parse[T](
        self,
        method: IAPIMethod[T],
        buffered: BufferedResponse,
    ) -> T:
        raw = buffered.body

        match method.__response_format__:
            case ResponseFormat.JSON:
                model_type = method.response_type

                if model_type is None:
                    raise TypeError(
                        "JSON response requires a response model"
                    )

                try:
                    return cast(
                        T,
                        model_type.model_validate_json(raw),
                    )
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
                    f"{method.__response_format__}"
                )
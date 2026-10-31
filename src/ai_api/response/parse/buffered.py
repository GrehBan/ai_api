"""Parser for fully read responses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ValidationError

from ai_api.enums import ResponseFormat
from ai_api.http.exceptions.protocols import IErrorResolver
from ai_api.http.exceptions.response import APIResponseError
from ai_api.response.buffered import BufferedResponse
from ai_api.serialization.json import JsonLoads, loads
from ai_api.types.protocols import IEventMatcher

if TYPE_CHECKING:
    from ai_api.methods.protocols import IAPIMethod


class BufferedResponseParser:
    """Parses JSON, text, bytes and empty responses."""

    def __init__(
        self, *, error_responses: IEventMatcher | None = None
    ) -> None:
        """Create the parser.

        Args:
            error_responses: Recognizes a validated JSON result that
                reports a failure although its status is below 400, such
                as OpenAI's ``Response`` with status ``failed``
                (``FieldValues("status", "failed")``). Such a result is
                raised as the ``APIStatusError`` the session's error
                resolver builds from the body, with the result as its
                ``raw``. ``None`` accepts every result.
        """
        self._error_responses = error_responses

    def parse[T](
        self,
        method: IAPIMethod[T],
        response: BufferedResponse,
        json_loads: JsonLoads,
        error_resolver: IErrorResolver,
    ) -> T:
        """Parse the body according to ``method.response_format``.

        Args:
            method: The method that was sent.
            response: The successful response.
            json_loads: JSON decoder of the session. With the default
                decoder the body is validated by Pydantic
                directly in JSON mode; a custom decoder's result is
                validated in Python mode, where strict fields reject JSON
                strings for dates, UUIDs and enums.
            error_resolver: Error resolver of the session, for results
                that ``error_responses`` matches.

        Returns:
            A validated model for JSON, ``str`` for TEXT, ``bytes`` for
            BYTES and ``None`` for NONE.

        Raises:
            APIStatusError: If ``error_responses`` matches the result.
            APIResponseError: If the body is not valid JSON or fails
                validation, fails decoding, or is not empty when NONE is
                expected.
            TypeError: If a JSON method has no model response type.
            ValueError: If a custom decoder rejects the body or the format
                is not supported for buffered responses.
        """
        raw = response.body

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
                        result = model_type.model_validate_json(raw)
                    else:
                        result = model_type.model_validate(json_loads(raw))
                except ValidationError as exc:
                    raise APIResponseError(
                        "Failed to validate JSON response",
                        status=response.status,
                    ) from exc

                if self._error_responses is not None and (
                    self._error_responses.matches(result)
                ):
                    raise self._failure(
                        result, response, json_loads, error_resolver
                    )

                return cast(T, result)

            case ResponseFormat.TEXT:
                try:
                    return cast(
                        T,
                        raw.decode(response.charset or "utf-8"),
                    )
                except (UnicodeError, LookupError) as exc:
                    raise APIResponseError(
                        "Failed to decode text response",
                        status=response.status,
                    ) from exc

            case ResponseFormat.BYTES:
                return cast(T, raw)

            case ResponseFormat.NONE:
                if raw:
                    raise APIResponseError(
                        "Expected an empty response body",
                        status=response.status,
                    )

                return cast(T, None)

            case _:
                raise ValueError(
                    "Unsupported buffered response format: "
                    f"{method.response_format}"
                )

    @staticmethod
    def _failure(
        result: BaseModel,
        response: BufferedResponse,
        json_loads: JsonLoads,
        error_resolver: IErrorResolver,
    ) -> Exception:
        """Build the error for a result that reports a failure.

        Args:
            result: The validated result.
            response: The response it came from.
            json_loads: JSON decoder of the session.
            error_resolver: Builds the error from the decoded body.

        Returns:
            The resolver's ``APIStatusError``, with the result as its
            ``raw``.
        """
        # Already validated, so the body is valid JSON.
        body: Any = json_loads(response.body)
        error = error_resolver.resolve(
            status=response.status,
            body=body,
            headers=response.headers,
        )
        error.raw = result
        return error

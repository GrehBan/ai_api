from abc import ABC
from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ai_api.client.response.parse.buffered import BufferedResponseParser
from ai_api.client.response.parse.protocols import IResponseParser
from ai_api.methods.enums import BodyFormat, ResponseFormat, ResponseMode
from ai_api.types.base import BaseAPIType


class BaseAPIMethod[ResponseT](BaseModel, ABC):
    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    __request_path__: ClassVar[str]
    __request_method__: ClassVar[str]
    __request_query__: ClassVar[set[str] | None] = None
    __returning__: ClassVar[type[BaseAPIType] | None] = None
    __request_body_format__: ClassVar[BodyFormat] = BodyFormat.NONE
    __retryable__: ClassVar[bool] = False
    __response_format__: ClassVar[ResponseFormat] = ResponseFormat.JSON
    __response_mode__: ClassVar[ResponseMode] = ResponseMode.BUFFERED

    @property
    def response_type(self) -> type[BaseAPIType] | None:
        return self.__returning__

    @property
    def reponse_parser(self) -> IResponseParser:
        return BufferedResponseParser()

    def get_query_map(self) -> Mapping[str, str] | None:
        if self.__request_query__ is None:
            return None

        data = self.model_dump(
            include=self.__request_query__,
            exclude_none=True,
            by_alias=True,
            mode="json",
        )

        return {
            key: str(value).lower() if isinstance(value, bool) else str(value)
            for key, value in data.items()
        }

    def get_body_map(self) -> Mapping[str, Any]:
        return self.model_dump(
            exclude=self.__request_query__ or set(),
            exclude_none=True,
            by_alias=True,
            mode="python",
        )

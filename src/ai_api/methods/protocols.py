from collections.abc import Mapping
from typing import Any, ClassVar, Protocol

from ai_api.client.response.parse.protocols import IResponseParser
from ai_api.methods.enums import BodyFormat, ResponseFormat, ResponseMode
from ai_api.types.base import BaseAPIType
from ai_api.types.typevars import ResponseT_co


class IAPIMethod(Protocol[ResponseT_co]):

    __request_path__: ClassVar[str]
    __request_method__: ClassVar[str]
    __request_query__: ClassVar[set[str] | None]
    __returning__: ClassVar[type[BaseAPIType] | None]
    __request_body_format__: ClassVar[BodyFormat]
    __retryable__: ClassVar[bool] = False
    __response_format__: ClassVar[ResponseFormat]
    __response_mode__: ClassVar[ResponseMode]

    def get_query_map(self) -> Mapping[str, str] | None:
        ...

    def get_body_map(self) -> Mapping[str, Any]:
        ...

    @property
    def response_parser(self) -> IResponseParser:
        ...


    @property
    def response_type(self) -> type[BaseAPIType] | None:
        ...
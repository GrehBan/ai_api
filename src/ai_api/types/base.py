from pydantic import BaseModel, ConfigDict


class BaseAPIType(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )
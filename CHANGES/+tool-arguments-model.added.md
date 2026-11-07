`Tool.from_model(name, Model, ...)` makes a tool whose `parameters` are a
Pydantic model's JSON Schema, and `ToolCall.parse_arguments(Model)`
validates a call's arguments with the model and returns it, raising the
model's `ValidationError` (a `ValueError`) for arguments the tool does
not take. `parse_arguments()` without a model still checks only that the
arguments are a JSON object.

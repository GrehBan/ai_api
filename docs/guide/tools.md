# Tools

`client.generation` lets the model call functions of your code, the same way for every provider: you describe the tools in the request, the model answers with calls instead of (or next to) text, you run them and send their results back in the conversation, until the model answers in text. All of it is imported from `aiollm`.

```python title="tools.py"
import json
from typing import Any

from aiollm import (
    GenerationResponse,
    IGenerationClient,
    InputItem,
    Tool,
    ToolResult,
    Turn,
)

WEATHER = Tool(
    name="get_weather",
    description="Current weather in a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
    strict=True,
)


def get_weather(city: str) -> str:
    return json.dumps({"city": city, "celsius": 15})


async def ask_with_tools(
    client: IGenerationClient, question: str
) -> GenerationResponse[Any]:
    conversation: list[InputItem] = [Turn.user(question)]
    while True:
        response = await client.generation.create(
            model="gpt-6-astra", input=conversation, tools=[WEATHER]
        )
        if not response.tool_calls:
            return response
        conversation.append(response.to_turn())
        for call in response.tool_calls:
            arguments = call.parse_arguments()
            output = get_weather(str(arguments["city"]))
            conversation.append(ToolResult(call_id=call.id, output=output))
```

- **A tool** is a `Tool`: a `name` (letters, digits, `_` and `-`, up to 128 characters, unique in the request), a `description` the model reads to decide when to call it, `parameters`, the JSON Schema of its arguments (an object schema; a tool without arguments takes `{"type": "object", "properties": {}}`), and `strict`. With `strict=True` the arguments follow the schema exactly; the schema then needs `additionalProperties: false` and every property in `required`, and the providers answer 400 otherwise. Left out, neither provider is strict.
- **Which tool.** `tool_choice=` takes a `ToolChoice` (`AUTO`, the default; `NONE`, no call; `REQUIRED`, at least one call) or a `ForcedTool(name=...)` naming one of the tools, which the model must call. `parallel_tool_calls=False` limits a response to one call; by default both providers allow several. Both need `tools`, and a `ForcedTool` naming no tool of the request, or parallel calls with `NONE`, is refused before sending.
- **The calls.** `response.tool_calls` lists the calls as `ToolCall`s: `id`, `name` and `arguments`, the JSON text the model wrote. `parse_arguments()` gives the object and raises `ValueError` for text that is not one, which a model can write without `strict`; it does not check the object against `parameters`. `parse_arguments(Model)` validates it with a Pydantic model as well and returns the model, or raises the model's `ValidationError` (a `ValueError`), so a tool never runs with arguments it does not take, `strict` or not (see below). A response with calls has finished normally (`incomplete_reason` is `None`) and waits for their results. When it stopped early, at the token limit for example, its last call may be cut off: `to_turn()` raises `ValueError` then, so retry with a higher `max_output_tokens` instead of running the calls.
- **The results.** `response.to_turn()` is the model's turn of calls, a `ToolCallTurn`, and a `ToolResult(call_id=call.id, output=...)` answers each call, in the order you like, right after it. `output` is text: describe a failure in it, since OpenAI has no error flag. Before sending, every adapter checks the order both providers require and raises `RequestValidationError` for a call without its result, a result without its call, or a turn between them.
- **Reasoning comes back.** The turn of calls keeps the response's own output (`ToolCallTurn.native`, from `GenerationResponse.native`): OpenAI's reasoning items, Anthropic's thinking with its signature. The same provider gets it back whole, so the model goes on from its own reasoning instead of thinking again. A turn from another provider sends only the shared calls and text. With OpenAI's `store: False`, its reasoning items carry their content only when the call asks for it: `provider_options={"store": False, "include": ["reasoning.encrypted_content"]}`.
- **Streaming.** `stream(...)` yields, for each call, a `ToolCallStartEvent` (`call_id`, `name`), `ToolCallDeltaEvent`s with pieces of the arguments and a `ToolCallDoneEvent` with the whole `ToolCall`, before the response ends, so a tool can start early. That event says the call is whole, not that the generation succeeded: the stream can still fail after it (an error event, a cut connection) and then raises that error instead of ending with a `CompletedEvent`. Start early only a tool that is safe to run for a failed generation, such as one that only reads, and run a tool with side effects after the `CompletedEvent`, from its `tool_calls` (see the example below). `events.final()` gives the `CompletedEvent` with `tool_calls`, and its `to_turn()` continues the conversation as `response.to_turn()` does.
- **Not shared.** OpenAI's built-in tools (web search, file search, code interpreter, MCP, ...), Anthropic's server tools, results with images or files, and the providers' own tool fields (such as Anthropic's `cache_control`) are reached through the provider's own methods, `CreateResponse` and `CreateMessage`. There is no runner that calls your functions for you: the loop above is yours to shape (how many rounds, what to do with a failing tool, whether to run calls in parallel).

A tool can take its arguments from a Pydantic model: `Tool.from_model(name, Model, ...)` sends the model's JSON Schema as `parameters`, and `parse_arguments(Model)` gives the arguments back as the model. For `strict=True`, the model forbids extra fields and has no defaults, since the providers want `additionalProperties: false` and every property required:

```python title="tools_model.py"
from pydantic import BaseModel, ConfigDict, ValidationError


class City(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city: str


CITY_WEATHER = Tool.from_model(
    "get_weather",
    City,
    description="Current weather in a city.",
    strict=True,
)


def answer(call: ToolCall) -> ToolResult:
    try:
        arguments = call.parse_arguments(City)
    except ValidationError as error:
        output = json.dumps({"error": str(error)})
    else:
        output = get_weather(arguments.city)
    return ToolResult(call_id=call.id, output=output)
```

A streamed call can start a tool before the response ends. Start that way only the tools that are safe to run for a generation that then fails, and run the others once the stream has ended with its `CompletedEvent`:

```python title="tools_stream.py"
import asyncio

from aiollm import ToolCall, ToolCallDoneEvent

READ_ONLY = {"get_weather"}


def run(call: ToolCall) -> str:
    if call.name == "get_weather":
        return get_weather(str(call.parse_arguments()["city"]))
    return json.dumps({"error": f"unknown tool {call.name}"})


async def stream_round(
    client: IGenerationClient, conversation: list[InputItem]
) -> list[InputItem]:
    early: dict[str, asyncio.Task[str]] = {}
    try:
        async with client.generation.stream(
            model="gpt-6-astra", input=conversation, tools=[WEATHER]
        ) as events:
            async for event in events:
                if (
                    isinstance(event, ToolCallDoneEvent)
                    and event.call.name in READ_ONLY
                ):
                    early[event.call.id] = asyncio.create_task(
                        asyncio.to_thread(run, event.call)
                    )
            completed = await events.final()
    except BaseException:
        for task in early.values():
            task.cancel()
        raise
    # The generation succeeded: now the tools with side effects run.
    results: list[InputItem] = [completed.to_turn()]
    for call in completed.tool_calls:
        output = await early[call.id] if call.id in early else run(call)
        results.append(ToolResult(call_id=call.id, output=output))
    return [*conversation, *results]
```

What each field becomes for each provider is listed in [Shared field mapping](../unification.md).

The documentation of `ToolCallDoneEvent` said a call "may be run" as
soon as it arrives. It now says what the event guarantees: the
arguments of that call are complete, not that the generation succeeded,
since the stream can still fail before its `CompletedEvent`. The tools
guide shows how to start only the tools that are safe to run early and
run the ones with side effects after the stream has completed.

**Breaking:** `Turn.content` is typed `str | list[ContentPart]` instead of
`str`, since a `user` turn can now hold parts. Turns built from a string
are unchanged at run time, but code that reads `turn.content` as a string
no longer type-checks: narrow it first with
`isinstance(turn.content, str)`. An `assistant` turn is always a string.

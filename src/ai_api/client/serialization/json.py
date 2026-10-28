from collections.abc import Callable
from typing import Any

import orjson

type JsonDumps = Callable[[Any], bytes]
type JsonLoads = Callable[[bytes | str], Any]


def dumps(data: Any, /) -> bytes:
    return orjson.dumps(data)


def loads(data: bytes | str, /) -> Any:
    return orjson.loads(data)
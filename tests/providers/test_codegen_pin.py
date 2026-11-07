"""The generator's input is the file of the contract snapshot."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def test_the_pinned_openapi_file_is_the_snapshot_one() -> None:
    # The types and the wire contracts come from one OpenAI file, so a
    # new file changes the pin and the snapshot together.
    script: dict[str, Any] = runpy.run_path(
        str(ROOT / "scripts" / "generate_provider_types.py")
    )
    manifest = json.loads(
        (ROOT / "tests/providers/contracts/manifest.json").read_text("utf-8")
    )
    snapshot = manifest["endpoints"]["openai/responses"]["openapi"]

    assert script["OPENAI_OPENAPI_SHA256"] == snapshot["sha256"]
    assert "/openai/openai-openapi/" in script["OPENAI_OPENAPI_URL"]

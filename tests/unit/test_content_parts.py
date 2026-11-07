"""Parts of a user turn: text, images and documents."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest
from pydantic import MISSING, ValidationError

from aiollm import DocumentPart, ImagePart, TextPart, Turn

PNG = b"\x89PNG\r\n"


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({}, "either data or url"),
        (
            {"data": PNG, "media_type": "image/png", "url": "https://a.io/x"},
            "either data or url",
        ),
        ({"data": b"", "media_type": "image/png"}, "data is empty"),
        ({"data": PNG}, "media_type goes with data"),
        (
            {"url": "https://a.io/x.png", "media_type": "image/png"},
            "media_type goes with data",
        ),
        ({"url": "file:///etc/passwd"}, "should match pattern"),
        ({"url": "https:///cat.png"}, "should match pattern"),
        ({"url": "https://:443/cat.png"}, "should match pattern"),
        ({"url": "https://@/cat.png"}, "should match pattern"),
        ({"data": PNG, "media_type": "image/bmp"}, "image/jpeg"),
    ],
    ids=[
        "none",
        "both",
        "empty",
        "no-type",
        "url-type",
        "scheme",
        "no-host",
        "port-only",
        "user-only",
        "bmp",
    ],
)
def test_an_image_has_one_source(fields: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        ImagePart(**fields)


@pytest.mark.parametrize(
    "url",
    [
        "https://a.io",
        "https://a.io?page=2",
        "https://user:key@a.io:8443/r.pdf#p=1",
        "http://[::1]:8080/r.pdf",
    ],
)
def test_a_url_with_a_host_is_taken(url: str) -> None:
    assert ImagePart(url=url).url == DocumentPart(url=url).url == url


def test_a_document_url_needs_a_host() -> None:
    with pytest.raises(ValidationError, match="should match pattern"):
        DocumentPart(url="https://:443/r.pdf")


def test_a_document_takes_a_filename_with_data_only() -> None:
    with pytest.raises(ValidationError, match="filename goes with data"):
        DocumentPart(url="https://a.io/r.pdf", filename="r.pdf")
    with pytest.raises(ValidationError, match="at least 1 character"):
        DocumentPart(data=b"%PDF", filename="")

    part = DocumentPart(data=b"%PDF")
    assert (part.media_type, part.filename) == ("application/pdf", MISSING)


def test_a_part_keeps_its_bytes_out_of_its_repr() -> None:
    assert "data" not in repr(ImagePart(data=PNG, media_type="image/png"))


def test_a_turn_of_parts_goes_through_json() -> None:
    turn = Turn.user(
        (
            TextPart(text="What is it?"),
            ImagePart(data=PNG, media_type="image/png"),
            DocumentPart(url="https://a.io/r.pdf"),
        )
    )

    dumped = turn.model_dump_json()

    assert '"data":"iVBORw0K"' in dumped
    assert Turn.model_validate_json(dumped) == turn
    assert isinstance(turn.content, list)


def test_only_a_user_turn_takes_parts() -> None:
    with pytest.raises(ValidationError, match="only a 'user' turn"):
        Turn(role="assistant", content=[TextPart(text="Hi")])  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="at least 1 item"):
        Turn.user([])


async def test_an_image_is_loaded_with_its_format(tmp_path: Path) -> None:
    path = tmp_path / "cat.PNG"
    path.write_bytes(PNG)

    part = await ImagePart.load(path)
    given = await ImagePart.load(str(path), media_type="image/webp")

    assert (part.data, part.media_type) == (PNG, "image/png")
    assert given.media_type == "image/webp"


async def test_an_image_of_unknown_format_needs_its_type(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scan.bmp"
    path.write_bytes(b"BM")

    with pytest.raises(ValueError, match="give media_type"):
        await ImagePart.load(path)
    with pytest.raises(FileNotFoundError):
        await ImagePart.load(tmp_path / "none.png")
    with pytest.raises(IsADirectoryError):
        await DocumentPart.load(tmp_path)


async def test_a_document_is_loaded_with_its_name(tmp_path: Path) -> None:
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-1.7")

    part = await DocumentPart.load(path)

    assert (part.data, part.filename) == (b"%PDF-1.7", "report.pdf")


async def test_a_cancelled_load_reports_a_failed_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started, release = threading.Event(), threading.Event()
    reported: list[dict[str, Any]] = []

    def read_bytes(_: Path) -> bytes:
        started.set()
        release.wait(5)
        raise PermissionError("denied")

    path = tmp_path / "r.pdf"
    path.write_bytes(b"%PDF")
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(lambda _, context: reported.append(context))
    try:
        task = asyncio.create_task(DocumentPart.load(path))
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        for _ in range(100):
            if reported:
                break
            await asyncio.sleep(0.01)
    finally:
        loop.set_exception_handler(None)

    assert [type(c["exception"]) for c in reported] == [PermissionError]

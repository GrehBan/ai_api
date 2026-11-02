# Uploading files

Annotate a field as `InputFile` and use a multipart body:

```python title="uploads.py"
from pathlib import Path

from ai_api import BytesFile, InputFile, StreamFile


class Transcribe(BaseAPIMethod[Transcription]):
    __request_path__ = "/audio/transcriptions"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.MULTIPART
    __retryable__ = True

    file: InputFile
    model: str


Transcribe(file=Path("voice.mp3"), model="whisper")      # PathFile, audio/mpeg
Transcribe(file=BytesFile(data, "image.png"), model="x")  # in memory
Transcribe(file=StreamFile(chunks(), "s.wav"), model="x") # async iterator
```

| Class | Source | Can be resent on retry |
|---|---|---|
| `BytesFile` | `bytes` in memory; sent with a known length | yes |
| `PathFile` | a file on disk, read in chunks in a worker thread | yes |
| `IOFile` | an open binary stream, read from its position at creation | if the stream is seekable |
| `StreamFile` | an async iterator of `bytes`; the upload takes it over and closes it | no |

An `InputFile` field also accepts plain `bytes`, a `Path` or a binary stream and converts it to the matching class. The content type is guessed from the file name when you don't give one. `PathFile` and `IOFile` derive from `ChunkedFile`, the base for a source of your own that is read piece by piece; a `chunk_size` below one raises `ValueError`, since `read(0)` would end the upload before it starts. Your own `InputFile` may return any async iterator from `open()`; it is closed after every attempt, by `aclose()` if it has one, else by `close()`. `StreamFile` takes ownership of its source: the upload closes it when it ends, however it ends, also when the request fails before its body is sent.

How files behave on retry, and why:

- **Every attempt opens the file again.** `InputFile.open()` returns a fresh iterator each time, so a retry sends the whole content, not the rest of a half-read stream.
- **An attempt lets go of its files.** When an attempt ends, also when the connection broke mid-upload or the server answered before it read the body, the session closes what the attempt was reading, so the retry can open the file again.
- **A cancelled read is waited for, not stepped on.** Files and streams are read in a worker thread, which cannot be stopped: a cancelled read runs on to its end. The file is closed, and a retry seeks the stream, only after that read is over, so a cancelled attempt never leaves a file open or reads under the next one. Closing a file runs in a worker thread too, since it may flush and block.
- **A failing close never hides why the upload ended.** If closing a source fails while the upload already fails or is cancelled, the close error is added to that exception as a note; a cancellation stays a cancellation. A close that fails after a clean upload raises `APIRequestError`. The same holds when a stream fails while its upload is still open: the stream's error is raised, with the failing close as a note.
- **A failure nobody awaits is reported, not dropped.** A read or a close that ran on after its caller was cancelled has no caller left to raise into. Its failure goes to the event loop's exception handler (the asyncio channel for such errors, which logs it by default), and is added as a note to the exception that ended the upload.
- **One stream, one upload at a time.** A stream has one position, so two uploads reading it at once (two requests in `asyncio.gather`, say) would take chunks from under each other and both send a mixed file. The second one raises `APIRequestError` instead, also when the stream is wrapped in two `IOFile`s. A stream whose `open()` iterator you took yourself counts as being read until you close that iterator.
- **A one-shot source disables retries.** A request whose files can be read only once (`StreamFile`, a non-seekable `IOFile`) is never retried, even if `__retryable__` is set: a retry would send an empty or truncated file.
- **A failing source is not a network error.** A missing `PathFile` (or one that points to a directory), or a source that fails while it is sent, raises `APIRequestError`, which is not retried. Retrying would fail the same way. So does an `IOFile` over a non-blocking stream that has no data ready: its `read()` returns `None`, which is not the end, and reading on would send a truncated file.

A text stream raises `ValidationError` when the method is created, and so does a `str` in an `InputFile` field: it could be a path or the content itself, so give a path as a `pathlib.Path`. Files in a JSON or URL-encoded body raise `TypeError`.

Files other than `BytesFile` are sent with chunked transfer encoding, so they are never read into memory as a whole. If a server requires `Content-Length`, as some presigned upload URLs do, use `BytesFile`.

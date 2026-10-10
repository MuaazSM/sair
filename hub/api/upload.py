"""Stream a multipart voice-note upload straight to ``<tmp>/<walk_id><ext>`` (FR-J1, FR-J6).

Starlette's own form parser spools big parts into the system temp dir, which would put raw
audio outside the directory the audio lifetime rules cover. This parser writes only the
``audio`` part, in chunks, directly to the final path in the hub's temp dir, and deletes the
partial file on any failure. Nothing from the body is logged or kept in memory beyond a chunk.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Request
from python_multipart.multipart import MultipartParser, parse_options_header

from hub.api.errors import ApiError

__all__ = ["EXT_BY_CONTENT_TYPE", "save_audio_upload"]

EXT_BY_CONTENT_TYPE: dict[str, str] = {
    "audio/webm": ".webm",
    "video/webm": ".webm",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/m4a": ".m4a",
    "audio/aac": ".aac",
    "audio/x-aac": ".aac",
    "video/mp4": ".mp4",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/vnd.wave": ".wav",
}
_ALLOWED_EXT = frozenset({".webm", ".m4a", ".mp4", ".wav", ".aac"})
_GENERIC = frozenset({"", "application/octet-stream", "binary/octet-stream"})
_OVERHEAD = 64 * 1024  # multipart framing and the small non-audio fields


def _ext_for(content_type: str, filename: str) -> str | None:
    base = content_type.split(";", 1)[0].strip().lower()
    if base in EXT_BY_CONTENT_TYPE:
        return EXT_BY_CONTENT_TYPE[base]
    if base in _GENERIC:
        suffix = Path(filename).suffix.lower()
        return suffix if suffix in _ALLOWED_EXT else None
    return None


class _Collector:
    """Callbacks for one request; owns the (single) output file."""

    def __init__(self, dest_dir: Path, walk_id: str, max_bytes: int) -> None:
        self.dest_dir = dest_dir
        self.walk_id = walk_id
        self.max_bytes = max_bytes
        self.path: Path | None = None
        self.size = 0
        self._fh: object | None = None
        self._writing = False
        self._field = bytearray()
        self._value = bytearray()
        self._headers: dict[bytes, bytes] = {}

    def callbacks(self) -> dict[str, object]:
        return {
            "on_part_begin": self._part_begin,
            "on_header_field": self._header_field,
            "on_header_value": self._header_value,
            "on_header_end": self._header_end,
            "on_headers_finished": self._headers_finished,
            "on_part_data": self._part_data,
            "on_part_end": self._part_end,
        }

    def _part_begin(self) -> None:
        self._headers = {}
        self._writing = False

    def _header_field(self, data: bytes, start: int, end: int) -> None:
        self._field += data[start:end]

    def _header_value(self, data: bytes, start: int, end: int) -> None:
        self._value += data[start:end]

    def _header_end(self) -> None:
        self._headers[bytes(self._field).lower()] = bytes(self._value)
        self._field.clear()
        self._value.clear()

    def _headers_finished(self) -> None:
        _, disp = parse_options_header(self._headers.get(b"content-disposition", b""))
        if disp.get(b"name") != b"audio" or self.path is not None:
            return
        filename = disp.get(b"filename", b"").decode("utf-8", "ignore")
        ctype = self._headers.get(b"content-type", b"").decode("latin-1")
        ext = _ext_for(ctype, filename)
        if ext is None:
            raise ApiError(415, "unsupported_media_type")
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dest_dir / f"{self.walk_id}{ext}"
        self._fh = self.path.open("xb")
        self._writing = True

    def _part_data(self, data: bytes, start: int, end: int) -> None:
        if not self._writing or self._fh is None:
            return
        self.size += end - start
        if self.size > self.max_bytes:
            raise ApiError(413, "payload_too_large")
        self._fh.write(data[start:end])  # type: ignore[attr-defined]

    def _part_end(self) -> None:
        self.close()
        self._writing = False

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()  # type: ignore[attr-defined]
            self._fh = None

    def discard(self) -> None:
        self.close()
        if self.path is not None:
            self.path.unlink(missing_ok=True)
            self.path = None


async def save_audio_upload(request: Request, dest_dir: Path, walk_id: str, max_bytes: int) -> Path:
    """Save the ``audio`` part as ``<dest_dir>/<walk_id><ext>`` and return the path.

    Raises :class:`ApiError`: 400 not multipart, 413 over the cap, 415 unsupported type,
    422 no ``audio`` part. The partial file is removed on every failure path.
    """
    mtype, params = parse_options_header(request.headers.get("content-type", ""))
    boundary = params.get(b"boundary")
    if mtype != b"multipart/form-data" or not boundary:
        raise ApiError(400, "bad_request")
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > max_bytes + _OVERHEAD:
        raise ApiError(413, "payload_too_large")
    col = _Collector(dest_dir, walk_id, max_bytes)
    ok = False
    try:
        parser = MultipartParser(boundary, col.callbacks())  # type: ignore[arg-type]
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > max_bytes + _OVERHEAD:
                raise ApiError(413, "payload_too_large")
            parser.write(chunk)
        parser.finalize()
        col.close()
        if col.path is None:
            raise ApiError(422, "missing_audio")
        ok = True
        return col.path
    except ApiError:
        raise
    except Exception:
        # malformed multipart or a dropped connection: fixed code, no detail (NFR-P4)
        raise ApiError(400, "bad_request") from None
    finally:
        if not ok:
            col.discard()

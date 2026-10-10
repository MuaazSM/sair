"""Model-client error taxonomy (IMPLEMENTATION P2, NFR-P4).

* ``Transient``: worth retrying later (connection refused, timeout, 5xx/429, model loading,
  TTS child died).
* ``Permanent``: retrying cannot help (undecodable/too short/too long/silent audio, 4xx,
  malformed server reply, TTS not configured).

Invalid or truncated model *output* is not an error here: clients return the raw text and
grounding decides.

Messages are fixed strings built from enum codes only. Nothing from a request or response
(transcript, prompt, audio bytes, mission text) is ever put in an exception (NFR-P4).
"""

from __future__ import annotations

from hub.schemas import ReasonCode

__all__ = ["ModelError", "Permanent", "Transient"]


class ModelError(Exception):
    """Base class. ``code`` is the pipeline reason code (``None`` for TTS: no enum value yet)."""

    kind = "error"

    def __init__(self, code: ReasonCode | None, component: str = "model") -> None:
        self.code = code
        self.component = component
        super().__init__(f"{component} {self.kind}: {code.value if code else 'unspecified'}")


class Transient(ModelError):
    kind = "transient"


class Permanent(ModelError):
    kind = "permanent"

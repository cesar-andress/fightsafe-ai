"""HTTP/WebSocket exposure for the live event bus (optional ``fastapi`` / ``uvicorn``)."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "app":
        from fightsafe_ai.api.app import app as fastapi_app

        return fastapi_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["app"]

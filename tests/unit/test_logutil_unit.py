"""Tests for :mod:`fightsafe_ai.logutil`."""

from __future__ import annotations

import logging
from io import StringIO

import pytest

from fightsafe_ai.logutil import configure_cli_pipeline_logging


pytestmark = pytest.mark.unit


def test_configure_cli_pipeline_logging_idempotent() -> None:
    """Second call is a no-op when the root logger already has a handler."""
    root = logging.getLogger()
    n0 = len(root.handlers)
    configure_cli_pipeline_logging(level=logging.DEBUG)
    n1 = len(root.handlers)
    configure_cli_pipeline_logging(level=logging.WARNING)
    assert len(root.handlers) == max(n0, n1) and n1 == max(n0, n1)


def test_configure_cli_pipeline_logging_custom_stream() -> None:
    buf = StringIO()
    root = logging.getLogger()
    old = list(root.handlers)
    try:
        for h in list(root.handlers):
            root.removeHandler(h)
        configure_cli_pipeline_logging(level=logging.DEBUG, stream=buf)
        root.info("ping")
        assert "ping" in buf.getvalue()
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in old:
            root.addHandler(h)

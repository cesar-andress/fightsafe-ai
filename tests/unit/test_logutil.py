"""Tests for :mod:`fightsafe_ai.logutil` (idempotent logging setup)."""

from __future__ import annotations

import io
import logging

import pytest

from fightsafe_ai.logutil import configure_cli_pipeline_logging


pytestmark = pytest.mark.unit


def test_configure_cli_pipeline_logging_idempotent() -> None:
    root = logging.getLogger()
    prior = list(root.handlers)
    for h in root.handlers[:]:
        root.removeHandler(h)
    n_handlers = 0
    try:
        buf = io.StringIO()
        configure_cli_pipeline_logging(stream=buf)
        logging.getLogger("fightsafe_ai.testlog").info("expected_message")
        assert "expected_message" in buf.getvalue()
        configure_cli_pipeline_logging(
            stream=buf,
        )  # no duplicate handlers; message count unchanged
        n_handlers = len(root.handlers)
    finally:
        for h in root.handlers[:]:
            root.removeHandler(h)
        for h in prior:
            root.addHandler(h)
    assert n_handlers == 1

"""
Configure :mod:`logging` for pipeline and visualization code when used from the CLI.

Library modules should use ``logger = logging.getLogger(__name__)`` and should not
call :func:`print` for status or diagnostics. The CLI may call
:func:`configure_cli_pipeline_logging` once so INFO logs appear on stdout
(user-facing progress), matching previous ``print`` behavior.
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO


def configure_cli_pipeline_logging(
    *,
    level: int = logging.INFO,
    stream: TextIO | None = None,
) -> None:
    """
    Attach a single :class:`logging.StreamHandler` to the root logger if it has
    no handlers (idempotent).

    Parameters
    ----------
    level
        Root logger level (default ``INFO``).
    stream
        Text stream; default is ``sys.stdout`` for user-visible progress.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    out: TextIO = sys.stdout if stream is None else stream
    h = logging.StreamHandler(out)
    h.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(h)
    root.setLevel(level)


__all__ = ["configure_cli_pipeline_logging"]

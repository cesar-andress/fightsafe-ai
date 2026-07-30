"""Engine and session factory when ``DATABASE_URL`` is set."""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)


def get_database_url() -> str | None:
    raw = os.environ.get("DATABASE_URL", "").strip()
    return raw or None


def create_engine_and_sessionmaker() -> tuple[Engine | None, sessionmaker[Session] | None]:
    url = get_database_url()
    if not url:
        return None, None
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "SQLAlchemy is required when DATABASE_URL is set. "
            "Install with: pip install 'fightsafe-ai[db]'"
        ) from exc

    engine = create_engine(url, pool_pre_ping=True)
    session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return engine, session_local


@contextmanager
def session_scope(session_factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    from sqlalchemy.orm import Session as SaSession

    sess: SaSession = session_factory()
    try:
        yield sess
        sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


__all__ = [
    "create_engine_and_sessionmaker",
    "get_database_url",
    "session_scope",
]

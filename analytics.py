"""
analytics.py
------------
Lightweight SQLite logging for InferLens usage analytics and per-email
rate limiting.

The database lives at ``inferlens.db`` next to this file. On Render Starter
the filesystem is ephemeral (resets on container restart), which is fine
for a demo — we're logging usage patterns, not billing-critical data. For
persistence across restarts, attach a Render disk or point the DB at an
external store.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "inferlens.db"
RATE_LIMIT_PER_DAY = 25  # queries per email per rolling 24h
RATE_LIMIT_WINDOW_HOURS = 24


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables + indexes if they don't exist. Safe to call every run."""
    conn = _connect()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            session_id TEXT NOT NULL,
            email TEXT,
            doc_slug TEXT,
            question TEXT NOT NULL,
            answer_length INTEGER,
            source_count INTEGER,
            latency_ms INTEGER
        );
        CREATE TABLE IF NOT EXISTS sessions (
            email TEXT PRIMARY KEY,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_queries_ts ON queries(ts);
        CREATE INDEX IF NOT EXISTS idx_queries_email ON queries(email);
        """
    )
    conn.commit()
    conn.close()


def log_query(
    session_id: str,
    email: str | None,
    doc_slug: str | None,
    question: str,
    answer_length: int,
    source_count: int,
    latency_ms: int,
) -> None:
    conn = _connect()
    conn.execute(
        "INSERT INTO queries (ts, session_id, email, doc_slug, question, "
        "answer_length, source_count, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            session_id,
            email,
            doc_slug,
            question,
            answer_length,
            source_count,
            latency_ms,
        ),
    )
    conn.commit()
    conn.close()


def upsert_session(email: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect()
    conn.execute(
        "INSERT INTO sessions (email, first_seen, last_seen) "
        "VALUES (?, ?, ?) "
        "ON CONFLICT(email) DO UPDATE SET last_seen = excluded.last_seen",
        (email, now, now),
    )
    conn.commit()
    conn.close()


def count_queries_last_24h(email: str) -> int:
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=RATE_LIMIT_WINDOW_HOURS)
    ).isoformat()
    conn = _connect()
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM queries WHERE email = ? AND ts > ?",
        (email, cutoff),
    ).fetchone()
    conn.close()
    return row["c"] if row else 0


def is_rate_limited(email: str) -> tuple[bool, int]:
    """Return (is_limited, count_in_window)."""
    count = count_queries_last_24h(email)
    return count >= RATE_LIMIT_PER_DAY, count


def get_stats() -> dict:
    """Aggregate usage stats for the admin view."""
    conn = _connect()
    total = conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
    unique_emails = conn.execute(
        "SELECT COUNT(DISTINCT email) FROM queries WHERE email IS NOT NULL"
    ).fetchone()[0]
    avg_latency_row = conn.execute(
        "SELECT AVG(latency_ms) FROM queries"
    ).fetchone()
    avg_latency = int(avg_latency_row[0]) if avg_latency_row and avg_latency_row[0] else 0

    top_docs_rows = conn.execute(
        "SELECT doc_slug, COUNT(*) AS c FROM queries "
        "WHERE doc_slug IS NOT NULL GROUP BY doc_slug ORDER BY c DESC LIMIT 10"
    ).fetchall()
    top_docs = [(r["doc_slug"], r["c"]) for r in top_docs_rows]

    recent_rows = conn.execute(
        "SELECT ts, email, doc_slug, question, latency_ms "
        "FROM queries ORDER BY id DESC LIMIT 30"
    ).fetchall()
    recent = [dict(r) for r in recent_rows]

    queries_today_row = conn.execute(
        "SELECT COUNT(*) FROM queries WHERE ts > ?",
        ((datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(),),
    ).fetchone()
    queries_today = queries_today_row[0] if queries_today_row else 0

    conn.close()
    return {
        "total_queries": total,
        "unique_emails": unique_emails,
        "queries_last_24h": queries_today,
        "avg_latency_ms": avg_latency,
        "top_docs": top_docs,
        "recent": recent,
    }

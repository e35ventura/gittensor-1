"""SQLite durability for control-plane state, nonces, and settlements."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Mapping


class SQLiteStateStore:
    """Small transactional store suitable for one regional control process."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('PRAGMA synchronous=FULL')
        return connection

    def _initialize(self) -> None:
        parent = Path(self.path).expanduser().resolve().parent
        parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS used_nonces (
                    hotkey TEXT NOT NULL,
                    nonce TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    PRIMARY KEY (hotkey, nonce)
                );
                CREATE TABLE IF NOT EXISTS settlements (
                    window_id TEXT PRIMARY KEY,
                    started_at REAL NOT NULL,
                    ended_at REAL NOT NULL,
                    hotkey_rewards TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS settlements_ended_at ON settlements(ended_at DESC);
                """
            )

    def load_state(self) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT value FROM state WHERE key = 'control_plane'").fetchone()
        return json.loads(row['value']) if row else None

    def save_state(self, state: Mapping[str, Any]) -> None:
        value = json.dumps(state, sort_keys=True, separators=(',', ':'))
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            connection.execute(
                """
                INSERT INTO state(key, value, updated_at) VALUES('control_plane', ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (value, now),
            )
            connection.execute('COMMIT')

    def consume_nonce(self, hotkey: str, nonce: str, expires_at: float) -> bool:
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            connection.execute('DELETE FROM used_nonces WHERE expires_at <= ?', (now,))
            try:
                connection.execute(
                    'INSERT INTO used_nonces(hotkey, nonce, expires_at) VALUES(?, ?, ?)',
                    (hotkey, nonce, expires_at),
                )
            except sqlite3.IntegrityError:
                connection.execute('ROLLBACK')
                return False
            connection.execute('COMMIT')
        return True

    def record_settlement(
        self,
        window_id: str,
        started_at: float,
        ended_at: float,
        hotkey_rewards: Mapping[str, str],
        metadata: Mapping[str, Any],
    ) -> bool:
        with self._lock, self._connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO settlements(
                        window_id, started_at, ended_at, hotkey_rewards, metadata, created_at
                    ) VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (
                        window_id,
                        started_at,
                        ended_at,
                        json.dumps(hotkey_rewards, sort_keys=True),
                        json.dumps(metadata, sort_keys=True),
                        time.time(),
                    ),
                )
            except sqlite3.IntegrityError:
                return False
        return True

    def latest_settlement(self, max_age_seconds: float, *, now: float | None = None) -> dict[str, Any] | None:
        timestamp = time.time() if now is None else now
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT window_id, started_at, ended_at, hotkey_rewards, metadata
                FROM settlements
                WHERE ended_at >= ?
                ORDER BY ended_at DESC
                LIMIT 1
                """,
                (timestamp - max_age_seconds,),
            ).fetchone()
        if row is None:
            return None
        return {
            'window_id': row['window_id'],
            'started_at': row['started_at'],
            'ended_at': row['ended_at'],
            'hotkey_rewards': json.loads(row['hotkey_rewards']),
            'metadata': json.loads(row['metadata']),
        }

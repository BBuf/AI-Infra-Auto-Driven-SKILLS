from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from .models import CampaignStatus


class StateError(RuntimeError):
    """Base class for campaign-state errors."""


class CampaignNotFound(StateError):
    pass


class InvalidTransition(StateError):
    pass


class IdempotencyConflict(StateError):
    pass


class LeaseUnavailable(StateError):
    pass


TERMINAL_STATUSES = frozenset(
    {
        CampaignStatus.TARGET_REACHED,
        CampaignStatus.UNREACHABLE_CERTIFIED,
        CampaignStatus.SEARCH_SPACE_EXHAUSTED,
        CampaignStatus.CANCELLED,
    }
)
RECOVERABLE_STATUSES = frozenset(
    {
        CampaignStatus.WAITING_RESOURCE,
        CampaignStatus.INFRA_BLOCKED,
        CampaignStatus.PAUSED_BUDGET,
    }
)
ACTIVE_STATUSES = frozenset(
    set(CampaignStatus) - TERMINAL_STATUSES - RECOVERABLE_STATUSES
)

_FORWARD_TRANSITIONS: dict[CampaignStatus, frozenset[CampaignStatus]] = {
    CampaignStatus.NEW: frozenset({CampaignStatus.BASELINE_LOCKED}),
    CampaignStatus.BASELINE_LOCKED: frozenset({CampaignStatus.PROFILED}),
    CampaignStatus.PROFILED: frozenset({CampaignStatus.AWAITING_AGENT}),
    CampaignStatus.AWAITING_AGENT: frozenset(
        {
            CampaignStatus.SEARCHING,
            CampaignStatus.SEARCH_SPACE_EXHAUSTED,
            CampaignStatus.UNREACHABLE_CERTIFIED,
        }
    ),
    CampaignStatus.SEARCHING: frozenset(
        {
            CampaignStatus.AWAITING_AGENT,
            CampaignStatus.INTEGRATING,
            CampaignStatus.SEARCH_SPACE_EXHAUSTED,
            CampaignStatus.UNREACHABLE_CERTIFIED,
        }
    ),
    CampaignStatus.INTEGRATING: frozenset(
        {
            CampaignStatus.AWAITING_AGENT,
            CampaignStatus.FINAL_VERIFYING,
        }
    ),
    CampaignStatus.FINAL_VERIFYING: frozenset(
        {
            CampaignStatus.AWAITING_AGENT,
            CampaignStatus.TARGET_REACHED,
            CampaignStatus.SEARCH_SPACE_EXHAUSTED,
            CampaignStatus.UNREACHABLE_CERTIFIED,
        }
    ),
}


def _now() -> datetime:
    return datetime.now(UTC)


def _timestamp(value: datetime | None = None) -> str:
    return (value or _now()).isoformat(timespec="microseconds")


class StateStore:
    def __init__(self, connection: sqlite3.Connection, event_log_path: Path) -> None:
        self._connection = connection
        self.event_log_path = event_log_path
        self._lock = threading.RLock()

    @classmethod
    def open(cls, database_path: Path, event_log_path: Path) -> StateStore:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        event_log_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            database_path,
            timeout=30,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        store = cls(connection, event_log_path)
        store._create_schema()
        return store

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> StateStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS campaigns (
              campaign_id TEXT PRIMARY KEY,
              status TEXT NOT NULL,
              epoch INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              campaign_id TEXT NOT NULL,
              event_type TEXT NOT NULL,
              idempotency_key TEXT NOT NULL UNIQUE,
              payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(campaign_id) REFERENCES campaigns(campaign_id)
            );
            CREATE TABLE IF NOT EXISTS leases (
              resource TEXT PRIMARY KEY,
              owner TEXT NOT NULL,
              expires_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS failures (
              signature TEXT PRIMARY KEY,
              campaign_id TEXT NOT NULL,
              technique TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(campaign_id) REFERENCES campaigns(campaign_id)
            );
            """
        )

    @contextmanager
    def _immediate(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def create_campaign(self, campaign_id: str) -> None:
        if not campaign_id:
            raise ValueError("campaign_id must not be empty")
        now = _timestamp()
        with self._immediate() as connection:
            row = connection.execute(
                "SELECT status FROM campaigns WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if row is not None:
                return
            connection.execute(
                """
                INSERT INTO campaigns(
                  campaign_id, status, epoch, created_at, updated_at
                ) VALUES (?, ?, 0, ?, ?)
                """,
                (campaign_id, CampaignStatus.NEW.value, now, now),
            )

    def status(self, campaign_id: str) -> CampaignStatus:
        with self._lock:
            row = self._connection.execute(
                "SELECT status FROM campaigns WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
        if row is None:
            raise CampaignNotFound(campaign_id)
        return CampaignStatus(row["status"])

    def epoch(self, campaign_id: str) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT epoch FROM campaigns WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
        if row is None:
            raise CampaignNotFound(campaign_id)
        return int(row["epoch"])

    def increment_epoch(self, campaign_id: str, *, idempotency_key: str) -> int:
        event: dict[str, Any] | None = None
        with self._immediate() as connection:
            existing = self._idempotent_event(
                connection,
                campaign_id,
                "epoch_incremented",
                idempotency_key,
                expected_payload=None,
            )
            if existing is not None:
                return int(existing["payload"]["epoch"])

            row = connection.execute(
                "SELECT epoch FROM campaigns WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if row is None:
                raise CampaignNotFound(campaign_id)
            epoch = int(row["epoch"]) + 1
            now = _timestamp()
            connection.execute(
                "UPDATE campaigns SET epoch = ?, updated_at = ? WHERE campaign_id = ?",
                (epoch, now, campaign_id),
            )
            event = self._insert_event(
                connection,
                campaign_id,
                "epoch_incremented",
                idempotency_key,
                {"epoch": epoch},
                now,
            )
        assert event is not None
        self._mirror_event(event)
        return int(event["payload"]["epoch"])

    def transition(
        self,
        campaign_id: str,
        status: CampaignStatus,
        *,
        idempotency_key: str,
        payload: Mapping[str, Any] | None = None,
    ) -> CampaignStatus:
        if not idempotency_key:
            raise ValueError("idempotency_key must not be empty")
        target = CampaignStatus(status)
        committed_event: dict[str, Any] | None = None

        with self._immediate() as connection:
            requested_payload = dict(payload or {})
            requested_payload["status"] = target.value
            existing = self._idempotent_event(
                connection,
                campaign_id,
                "transition",
                idempotency_key,
                expected_payload=requested_payload,
                permit_generated_prior=True,
            )
            if existing is not None:
                return CampaignStatus(existing["payload"]["status"])

            row = connection.execute(
                "SELECT status FROM campaigns WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if row is None:
                raise CampaignNotFound(campaign_id)
            current = CampaignStatus(row["status"])
            self._validate_transition(connection, campaign_id, current, target)

            if target in RECOVERABLE_STATUSES:
                if "prior_status" in requested_payload:
                    raise ValueError("prior_status is managed by StateStore")
                requested_payload["prior_status"] = current.value

            now = _timestamp()
            connection.execute(
                "UPDATE campaigns SET status = ?, updated_at = ? WHERE campaign_id = ?",
                (target.value, now, campaign_id),
            )
            committed_event = self._insert_event(
                connection,
                campaign_id,
                "transition",
                idempotency_key,
                requested_payload,
                now,
            )

        assert committed_event is not None
        self._mirror_event(committed_event)
        return target

    def _validate_transition(
        self,
        connection: sqlite3.Connection,
        campaign_id: str,
        current: CampaignStatus,
        target: CampaignStatus,
    ) -> None:
        if current in TERMINAL_STATUSES:
            raise InvalidTransition(
                f"terminal campaign cannot transition: {current.value} -> {target.value}"
            )
        if current in RECOVERABLE_STATUSES:
            if target is CampaignStatus.CANCELLED:
                return
            prior = self._prior_active_status(connection, campaign_id, current)
            if target is not prior:
                raise InvalidTransition(
                    f"{current.value} can only resume {prior.value}, not {target.value}"
                )
            return
        if target in RECOVERABLE_STATUSES or target is CampaignStatus.CANCELLED:
            return
        allowed = _FORWARD_TRANSITIONS.get(current, frozenset())
        if target not in allowed:
            raise InvalidTransition(f"{current.value} -> {target.value} is not allowed")

    @staticmethod
    def _prior_active_status(
        connection: sqlite3.Connection,
        campaign_id: str,
        current: CampaignStatus,
    ) -> CampaignStatus:
        rows = connection.execute(
            """
            SELECT payload_json
            FROM events
            WHERE campaign_id = ? AND event_type = 'transition'
            ORDER BY id DESC
            """,
            (campaign_id,),
        ).fetchall()
        for row in rows:
            payload = json.loads(row["payload_json"])
            if payload.get("status") == current.value and "prior_status" in payload:
                prior = CampaignStatus(payload["prior_status"])
                if prior not in ACTIVE_STATUSES:
                    break
                return prior
        raise InvalidTransition(
            f"{current.value} has no recorded active status to resume"
        )

    def events(
        self, campaign_id: str, *, event_type: str | None = None
    ) -> list[dict[str, Any]]:
        query = (
            "SELECT campaign_id, event_type, idempotency_key, payload_json "
            "FROM events WHERE campaign_id = ?"
        )
        parameters: list[str] = [campaign_id]
        if event_type is not None:
            query += " AND event_type = ?"
            parameters.append(event_type)
        query += " ORDER BY id"
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return [
            {
                "campaign_id": row["campaign_id"],
                "event_type": row["event_type"],
                "idempotency_key": row["idempotency_key"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def record_event(
        self,
        campaign_id: str,
        event_type: str,
        idempotency_key: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Persist an arbitrary campaign event exactly once."""
        if not event_type or not idempotency_key:
            raise ValueError("event_type and idempotency_key must not be empty")
        payload_dict = dict(payload)
        # Validate JSON serializability before opening the write transaction.
        self._canonical_json(payload_dict)
        committed_event: dict[str, Any] | None = None
        with self._immediate() as connection:
            self._require_campaign(connection, campaign_id)
            existing = self._idempotent_event(
                connection,
                campaign_id,
                event_type,
                idempotency_key,
                expected_payload=payload_dict,
            )
            if existing is not None:
                return existing
            committed_event = self._insert_event(
                connection,
                campaign_id,
                event_type,
                idempotency_key,
                payload_dict,
                _timestamp(),
            )

        assert committed_event is not None
        self._mirror_event(committed_event)
        return committed_event

    def acquire_lease(self, resource: str, owner: str, *, ttl_seconds: float) -> None:
        if not resource or not owner:
            raise ValueError("resource and owner must not be empty")
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be non-negative")
        now = _now()
        expires_at = _timestamp(now + timedelta(seconds=ttl_seconds))
        now_text = _timestamp(now)
        with self._immediate() as connection:
            row = connection.execute(
                "SELECT owner, expires_at FROM leases WHERE resource = ?",
                (resource,),
            ).fetchone()
            if (
                row is not None
                and row["owner"] != owner
                and row["expires_at"] > now_text
            ):
                raise LeaseUnavailable(
                    f"{resource!r} is leased by {row['owner']!r} until "
                    f"{row['expires_at']}"
                )
            connection.execute(
                """
                INSERT INTO leases(resource, owner, expires_at)
                VALUES (?, ?, ?)
                ON CONFLICT(resource) DO UPDATE
                SET owner = excluded.owner, expires_at = excluded.expires_at
                """,
                (resource, owner, expires_at),
            )

    def lease_owner(self, resource: str) -> str | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT owner, expires_at FROM leases WHERE resource = ?",
                (resource,),
            ).fetchone()
        if row is None or row["expires_at"] <= _timestamp():
            return None
        return str(row["owner"])

    def expired_leases(self) -> list[str]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT resource FROM leases WHERE expires_at <= ? ORDER BY resource",
                (_timestamp(),),
            ).fetchall()
        return [str(row["resource"]) for row in rows]

    def release_lease(self, resource: str, owner: str) -> bool:
        with self._immediate() as connection:
            cursor = connection.execute(
                "DELETE FROM leases WHERE resource = ? AND owner = ?",
                (resource, owner),
            )
            return cursor.rowcount == 1

    def record_failure(
        self,
        campaign_id: str,
        technique: str,
        signature: str,
        payload: Mapping[str, Any],
    ) -> bool:
        if not technique or not signature:
            raise ValueError("technique and signature must not be empty")
        payload_json = self._canonical_json(dict(payload))
        with self._immediate() as connection:
            self._require_campaign(connection, campaign_id)
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO failures(
                  signature, campaign_id, technique, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (signature, campaign_id, technique, payload_json, _timestamp()),
            )
            return cursor.rowcount == 1

    def has_failure(self, signature: str) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM failures WHERE signature = ?", (signature,)
            ).fetchone()
        return row is not None

    def failures(self, campaign_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT signature, campaign_id, technique, payload_json
                FROM failures
                WHERE campaign_id = ?
                ORDER BY created_at, signature
                """,
                (campaign_id,),
            ).fetchall()
        return [
            {
                "signature": row["signature"],
                "campaign_id": row["campaign_id"],
                "technique": row["technique"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    @staticmethod
    def _require_campaign(connection: sqlite3.Connection, campaign_id: str) -> None:
        row = connection.execute(
            "SELECT 1 FROM campaigns WHERE campaign_id = ?", (campaign_id,)
        ).fetchone()
        if row is None:
            raise CampaignNotFound(campaign_id)

    def _idempotent_event(
        self,
        connection: sqlite3.Connection,
        campaign_id: str,
        event_type: str,
        idempotency_key: str,
        *,
        expected_payload: Mapping[str, Any] | None,
        permit_generated_prior: bool = False,
    ) -> dict[str, Any] | None:
        row = connection.execute(
            """
            SELECT campaign_id, event_type, idempotency_key, payload_json
            FROM events
            WHERE idempotency_key = ?
            """,
            (idempotency_key,),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload_json"])
        comparable_payload = dict(payload)
        if permit_generated_prior:
            comparable_payload.pop("prior_status", None)
        if (
            row["campaign_id"] != campaign_id
            or row["event_type"] != event_type
            or (
                expected_payload is not None
                and comparable_payload != dict(expected_payload)
            )
        ):
            raise IdempotencyConflict(
                f"idempotency key {idempotency_key!r} was already used for a "
                "different operation"
            )
        return {
            "campaign_id": row["campaign_id"],
            "event_type": row["event_type"],
            "idempotency_key": row["idempotency_key"],
            "payload": payload,
        }

    def _insert_event(
        self,
        connection: sqlite3.Connection,
        campaign_id: str,
        event_type: str,
        idempotency_key: str,
        payload: Mapping[str, Any],
        created_at: str,
    ) -> dict[str, Any]:
        payload_dict = dict(payload)
        connection.execute(
            """
            INSERT INTO events(
              campaign_id, event_type, idempotency_key, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                campaign_id,
                event_type,
                idempotency_key,
                self._canonical_json(payload_dict),
                created_at,
            ),
        )
        return {
            "campaign_id": campaign_id,
            "event_type": event_type,
            "idempotency_key": idempotency_key,
            "payload": payload_dict,
        }

    def _mirror_event(self, event: Mapping[str, Any]) -> None:
        encoded = (self._canonical_json(dict(event)) + "\n").encode()
        descriptor = os.open(
            self.event_log_path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _canonical_json(value: Mapping[str, Any]) -> str:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

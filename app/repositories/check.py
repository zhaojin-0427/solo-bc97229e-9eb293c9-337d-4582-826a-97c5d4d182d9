"""Verifier-side repository (check.db).

Runs are the unit of work. The API enqueues a run bound to a specific data
version; exactly one worker across both processes claims each queued/stale
run via an atomic ``UPDATE ... WHERE status`` claim.

Ownership fencing
-----------------
Every (re)claim rotates an opaque ``claim_token`` and bumps ``attempts`` (the
attempt number). All progress and conclusion writes must present the token
the worker received at claim time. When a heartbeat times out the run is
re-queued and another process claims it with a *new* token: the zombie, once
it wakes up, fails every fenced write and can no longer update the
successor's progress, finish the run as failed, trigger a release, or touch
its attempt-scoped working tables. Disk search state is keyed by
``(run_id, attempt_no)`` so two attempts of the same run never share rows.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from ..config import settings
from ..database import connect, transaction

# Working tables holding per-attempt search state.
WORKING_TABLES = (
    "verifier_edge",
    "verifier_vertex",
    "verifier_frontier",
    "verifier_parent",
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_claim_token() -> str:
    return uuid.uuid4().hex


class CheckRepository:
    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or settings.check_db

    # --- queueing ---------------------------------------------------------

    def enqueue(self, data_version: int) -> int:
        """Queue a verification for *data_version*.

        Identical queued/running runs are coalesced so repeated uploads do
        not pile up duplicate work; a finished run for the same version is
        never reused because the data content at a version is immutable.
        """
        with transaction(self.db_path) as conn:
            existing = conn.execute(
                "SELECT id FROM verifier_run WHERE data_version=? "
                "AND status IN ('queued','running')",
                (data_version,),
            ).fetchone()
            if existing:
                return int(existing["id"])
            # Newer versions supersede unfinished older runs.
            conn.execute(
                "UPDATE verifier_run SET status='superseded', finished_at=? "
                "WHERE status IN ('queued','running') AND data_version < ?",
                (_utcnow().isoformat(), data_version),
            )
            cur = conn.execute(
                "INSERT INTO verifier_run(data_version, status) VALUES (?, 'queued')",
                (data_version,),
            )
            return int(cur.lastrowid)

    # --- disk working-state cleanup --------------------------------------

    def _purge_working_state(self, conn: sqlite3.Connection, run_id: int) -> None:
        """Remove every attempt partition of *run_id* from the work tables.

        Called inside the reclaim transaction before a new attempt starts, so
        a re-claimed run always begins with clean tables and the disk stays
        bounded even if a previous process died before its own cleanup ran.
        """
        for table in WORKING_TABLES:
            conn.execute(f"DELETE FROM {table} WHERE run_id=?", (run_id,))

    def purge_orphan_working_state(self) -> int:
        """Delete work rows whose run no longer has an active claimant.

        Rows belonging to queued/running runs (their current attempt may be
        using them) are always retained. This reclaims space left behind by
        processes that crashed hard before their ``finally`` cleanup and is
        safe to call repeatedly from any process.
        """
        removed = 0
        with transaction(self.db_path) as conn:
            for table in WORKING_TABLES:
                cur = conn.execute(
                    f"DELETE FROM {table} WHERE run_id NOT IN ("
                    "SELECT id FROM verifier_run WHERE status IN "
                    "('queued','running'))"
                )
                removed += cur.rowcount
        return removed

    # --- claiming ---------------------------------------------------------

    def claim_next_run(self, worker_id: str) -> Optional[sqlite3.Row]:
        """Atomically claim the highest-priority (newest) queued run.

        Also recovers runs whose worker died (heartbeat stale) after the
        attempt limit is not yet reached. The reclaim runs in the same write
        transaction that destroys the zombie's working partition, so the
        successor builds a fresh, independent graph.
        """
        with transaction(self.db_path) as conn:
            # First reap dead workers.
            stale_cutoff = (
                _utcnow() - timedelta(seconds=settings.heartbeat_timeout)
            ).isoformat()
            stale = conn.execute(
                "SELECT id, attempts FROM verifier_run "
                "WHERE status='running' AND heartbeat_at IS NOT NULL "
                "AND heartbeat_at < ?",
                (stale_cutoff,),
            ).fetchall()
            for row in stale:
                # The zombie can never be saved: drop all of its on-disk
                # search state before deciding what happens to the run.
                self._purge_working_state(conn, row["id"])
                if row["attempts"] >= settings.max_attempts:
                    conn.execute(
                        "UPDATE verifier_run SET status='failed', finished_at=?, "
                        "worker_id=NULL, claim_token=NULL, "
                        "error_message=? WHERE id=?",
                        (
                            _utcnow().isoformat(),
                            f"连续 {row['attempts']} 次核查心跳超时，判定失败",
                            row["id"],
                        ),
                    )
                else:
                    # Re-queue but keep the accumulated attempt count, so the
                    # max_attempts ceiling still applies across worker deaths.
                    # The token is cleared: only the next claimant gets one.
                    conn.execute(
                        "UPDATE verifier_run SET status='queued', "
                        "progress_pct=0, vertices_processed=0, worker_id=NULL, "
                        "heartbeat_at=NULL, claim_token=NULL WHERE id=?",
                        (row["id"],),
                    )
            run = conn.execute(
                "SELECT * FROM verifier_run WHERE status='queued' "
                "ORDER BY data_version DESC LIMIT 1"
            ).fetchone()
            if not run:
                return None
            token = new_claim_token()
            conn.execute(
                "UPDATE verifier_run SET status='running', started_at=?, "
                "worker_id=?, heartbeat_at=?, claim_token=?, "
                "attempts=attempts+1 WHERE id=?",
                (
                    _utcnow().isoformat(),
                    worker_id,
                    _utcnow().isoformat(),
                    token,
                    run["id"],
                ),
            )
            return conn.execute(
                "SELECT * FROM verifier_run WHERE id=?", (run["id"],)
            ).fetchone()

    # --- fenced writes ----------------------------------------------------

    def is_owner(self, run_id: int, claim_token: Optional[str]) -> bool:
        """True only while *claim_token* is still the run's active token."""
        if not claim_token:
            return False
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT claim_token FROM verifier_run WHERE id=?", (run_id,)
            ).fetchone()
        return row is not None and row["claim_token"] == claim_token

    def heartbeat(
        self, run_id: int, attempt_no: int, claim_token: str,
        processed: int, total: int,
    ) -> bool:
        """Renew the heartbeat. Returns False when ownership has been lost.

        A False result means another worker now owns the run (the local
        attempt timed out and was re-claimed); the caller must abort without
        writing anything else. The token predicate guarantees a zombie can
        never move the successor's progress backwards.
        """
        pct = round(100.0 * processed / total, 2) if total else 0.0
        with transaction(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE verifier_run SET heartbeat_at=?, vertices_processed=?, "
                "vertices_total=?, progress_pct=? "
                "WHERE id=? AND status='running' AND attempts=? "
                "AND claim_token=?",
                (
                    _utcnow().isoformat(),
                    processed,
                    total,
                    pct,
                    run_id,
                    attempt_no,
                    claim_token,
                ),
            )
            return cur.rowcount == 1

    def finish_run(
        self,
        run_id: int,
        attempt_no: int,
        claim_token: str,
        status: str,
        components: int,
        conflicts: List[dict],
        error: Optional[str] = None,
    ) -> bool:
        """Persist the conclusion behind the fencing token.

        Returns False (and writes nothing) when the caller no longer owns the
        run — it was re-claimed after a heartbeat timeout, or the run was
        superseded by a newer data version. A late-finishing zombie therefore
        cannot mark its successor's run failed, cannot overwrite a newer
        state, and cannot trigger a release.
        """
        with transaction(self.db_path) as conn:
            run = conn.execute(
                "SELECT status, claim_token FROM verifier_run WHERE id=?",
                (run_id,),
            ).fetchone()
            if (
                not run
                or run["status"] != "running"
                or run["claim_token"] != claim_token
            ):
                return False
            conn.execute(
                "UPDATE verifier_run SET status=?, finished_at=?, "
                "progress_pct=100, vertices_processed=COALESCE(vertices_total, "
                "vertices_processed), components_total=?, conflicts_total=?, "
                "error_message=? WHERE id=? AND claim_token=?",
                (
                    status,
                    _utcnow().isoformat(),
                    components,
                    len(conflicts),
                    error,
                    run_id,
                    claim_token,
                ),
            )
            conn.execute("DELETE FROM verifier_conflict WHERE run_id=?", (run_id,))
            conn.executemany(
                "INSERT INTO verifier_conflict(run_id, sample_a, group_a, "
                "sample_b, group_b, component_size, path_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        run_id,
                        c["sample_a"],
                        c["group_a"],
                        c["sample_b"],
                        c["group_b"],
                        c["component_size"],
                        json.dumps(c["path"], ensure_ascii=False),
                    )
                    for c in conflicts[: settings.max_conflicts]
                ],
            )
        return True

    # --- queries ----------------------------------------------------------

    def get_run(self, run_id: int) -> Optional[sqlite3.Row]:
        with connect(self.db_path) as conn:
            return conn.execute(
                "SELECT * FROM verifier_run WHERE id=?", (run_id,)
            ).fetchone()

    def latest_run(self) -> Optional[sqlite3.Row]:
        with connect(self.db_path) as conn:
            return conn.execute(
                "SELECT * FROM verifier_run ORDER BY id DESC LIMIT 1"
            ).fetchone()

    def latest_run_for_version(self, version: int) -> Optional[sqlite3.Row]:
        with connect(self.db_path) as conn:
            return conn.execute(
                "SELECT * FROM verifier_run WHERE data_version=? "
                "ORDER BY id DESC LIMIT 1",
                (version,),
            ).fetchone()

    def conflicts_for(self, run_id: int) -> List[sqlite3.Row]:
        with connect(self.db_path) as conn:
            return list(
                conn.execute(
                    "SELECT * FROM verifier_conflict WHERE run_id=? ORDER BY id",
                    (run_id,),
                )
            )

    def running_versions(self) -> List[int]:
        with connect(self.db_path) as conn:
            return [
                int(r["data_version"])
                for r in conn.execute(
                    "SELECT DISTINCT data_version FROM verifier_run "
                    "WHERE status IN ('queued','running')"
                )
            ]

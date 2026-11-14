"""Regression tests for verification ownership across reclaim cycles.

Scenario under repair: in the two-process deployment, a worker thread whose
heartbeat timed out was still running. Another process re-claimed the same
run; the zombie later errored and wrote its failure over the successor's
conclusion, so clean groups could never be released and its disk cleanup
erased the successor's in-flight search data.

These tests pin down the required behaviour:

* every progress / conclusion write is fenced by a claim token rotated on
  each (re)claim, so a recovered zombie cannot update the successor, finish
  it as failed, write conflicts or trigger a release;
* disk graph/search state is partitioned per attempt and a zombie only ever
  deletes its own partition;
* a successor independently verifies the complete current version and still
  returns real connecting conflict paths;
* repeated reclaims isolate every stale attempt, nothing is reported safe
  before completion, and uploads/status queries stay available;
* disk working state stays bounded and is migrated from the old schema.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest

from app.config import settings
from app.database import connect, init_databases
from app.repositories.business import BusinessRepository
from app.repositories.check import CheckRepository, WORKING_TABLES
from app.services.verifier import GraphVerifier


# --- helpers ---------------------------------------------------------------

def _force_stale(run_id: int) -> None:
    stale = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    with connect(settings.check_db) as conn:
        conn.execute(
            "UPDATE verifier_run SET heartbeat_at=? WHERE id=?",
            (stale, run_id),
        )
        conn.commit()


def _clean_dataset():
    business = BusinessRepository()
    business.add_relations([("A", "B"), ("B", "C"), ("C", "D")])
    business.replace_groups(
        [("A", "train"), ("B", "train"), ("C", "train"), ("D", "train")]
    )
    return business.current_version()


def _working_row_count(run_id: int, attempt_no: int) -> int:
    with connect(settings.check_db) as conn:
        return sum(
            conn.execute(
                f"SELECT COUNT(*) FROM {t} WHERE run_id=? AND attempt_no=?",
                (run_id, attempt_no),
            ).fetchone()[0]
            for t in WORKING_TABLES
        )


# --- fenced writes ---------------------------------------------------------

def test_heartbeat_with_rotated_token_is_rejected(data_dir):
    _clean_dataset()
    check = CheckRepository()
    run_id = check.enqueue(BusinessRepository().current_version())
    first = check.claim_next_run("worker-a")
    token_a = first["claim_token"]
    assert token_a and check.is_owner(run_id, token_a)

    # The live heartbeat renews progress normally.
    assert check.heartbeat(run_id, 1, token_a, 10, 100) is True

    # Reclaim rotates the token.
    _force_stale(run_id)
    second = check.claim_next_run("worker-b")
    assert second["attempts"] == 2
    token_b = second["claim_token"]
    assert token_b != token_a
    assert check.is_owner(run_id, token_a) is False
    assert check.is_owner(run_id, token_b) is True

    # The zombie's heartbeat is rejected and must not move progress backwards.
    assert check.heartbeat(run_id, 1, token_a, 1, 100) is False
    row = check.get_run(run_id)
    assert row["worker_id"] == "worker-b"
    assert row["progress_pct"] == 0.0  # zombie value (1%) was not written
    # The successor still owns the heartbeat.
    assert check.heartbeat(run_id, 2, token_b, 50, 100) is True
    assert check.get_run(run_id)["progress_pct"] == 50.0


def test_zombie_conclusion_after_reclaim_cannot_fail_successor(data_dir):
    """The original bug: late zombie error overwrites the successor's pass."""
    _clean_dataset()
    check = CheckRepository()
    run_id = check.enqueue(BusinessRepository().current_version())
    zombie_row = check.claim_next_run("worker-a")
    token_a = zombie_row["claim_token"]

    # Heartbeat timeout -> worker-b re-claims and independently verifies.
    _force_stale(run_id)
    successor_row = check.claim_next_run("worker-b")
    assert successor_row["claim_token"] != token_a
    GraphVerifier(check, BusinessRepository()).run(successor_row)
    assert check.get_run(run_id)["status"] == "passed"
    assert BusinessRepository().acceptance() is not None

    # The zombie wakes up and tries to record its (simulated) failure.
    written = check.finish_run(
        run_id, 1, token_a, "failed", 0, [],
        error="RuntimeError: 旧线程迟到的报错",
    )
    assert written is False
    final = check.get_run(run_id)
    assert final["status"] == "passed"          # successor conclusion intact
    assert final["worker_id"] == "worker-b"
    assert final["error_message"] is None
    assert check.conflicts_for(run_id) == []
    # And it cannot trigger a release path either: the existing acceptance is
    # the successor's, bound to the same run.
    assert BusinessRepository().acceptance()["run_id"] == run_id


def test_recovered_zombie_verifier_aborts_without_any_write(data_dir):
    """Full verifier entry point with a stale token must stop immediately."""
    _clean_dataset()
    check = CheckRepository()
    run_id = check.enqueue(BusinessRepository().current_version())
    zombie_row = check.claim_next_run("worker-a")

    _force_stale(run_id)
    successor_row = check.claim_next_run("worker-b")

    # Zombie resumes with its old, now-invalid row.
    GraphVerifier(check, BusinessRepository()).run(zombie_row)

    row = check.get_run(run_id)
    assert row["status"] == "running"             # still the successor's turn
    assert row["claim_token"] == successor_row["claim_token"]
    assert row["worker_id"] == "worker-b"
    assert row["finished_at"] is None
    # The successor can still complete normally afterwards.
    GraphVerifier(check, BusinessRepository()).run(successor_row)
    assert check.get_run(run_id)["status"] == "passed"


# --- disk working-state isolation ------------------------------------------

def test_working_partitions_are_isolated_between_attempts(data_dir):
    _clean_dataset()
    check = CheckRepository()
    run_id = check.enqueue(BusinessRepository().current_version())
    first = check.claim_next_run("worker-a")
    # Materialise attempt 1's on-disk graph, then the worker stalls.
    verifier = GraphVerifier(check, BusinessRepository())
    work = connect(settings.check_db)
    total, _ = verifier._load_snapshot(work, run_id, 1, first["data_version"])
    work.close()
    assert total == 4
    attempt1_rows = _working_row_count(run_id, 1)
    assert attempt1_rows > 0
    assert _working_row_count(run_id, 2) == 0

    # Reclaim purges attempt 1 and starts attempt 2 with a clean partition.
    _force_stale(run_id)
    successor = check.claim_next_run("worker-b")
    assert successor["attempts"] == 2
    assert _working_row_count(run_id, 1) == 0

    work = connect(settings.check_db)
    verifier._load_snapshot(work, run_id, 2, successor["data_version"])
    work.close()
    attempt2_rows = _working_row_count(run_id, 2)
    assert attempt2_rows > 0

    # A zombie cleanup (e.g. its finally block) may only touch attempt 1:
    # simulate exactly the verifier's cleanup statement.
    work = connect(settings.check_db)
    for table in WORKING_TABLES:
        work.execute(
            f"DELETE FROM {table} WHERE run_id=? AND attempt_no=?",
            (run_id, 1),
        )
    work.commit()
    work.close()
    assert _working_row_count(run_id, 1) == 0
    assert _working_row_count(run_id, 2) == attempt2_rows  # successor safe


def test_orphan_working_state_cleanup_keeps_disk_bounded(data_dir):
    _clean_dataset()
    check = CheckRepository()
    business = BusinessRepository()
    # A finished run with leftover working rows (hard-crash simulation).
    rid1 = check.enqueue(business.current_version())
    row1 = check.claim_next_run("w")
    work = connect(settings.check_db)
    GraphVerifier(check, business)._load_snapshot(
        work, rid1, 1, row1["data_version"]
    )
    work.close()
    GraphVerifier(check, business).run(row1)
    # Inject orphan rows as if cleanup never ran.
    with connect(settings.check_db) as conn:
        conn.execute(
            "INSERT INTO verifier_edge(run_id, attempt_no, u, v) "
            "VALUES (?, 1, 'X', 'Y')",
            (rid1,),
        )
        conn.commit()

    # A new in-flight run whose rows must be preserved.
    business.replace_groups([("A", "test")])
    rid2 = check.enqueue(business.current_version())
    row2 = check.claim_next_run("w2")
    work = connect(settings.check_db)
    GraphVerifier(check, business)._load_snapshot(
        work, rid2, row2["attempts"], row2["data_version"]
    )
    work.close()
    active_rows = _working_row_count(rid2, row2["attempts"])
    assert active_rows > 0

    removed = check.purge_orphan_working_state()
    assert removed >= 1
    assert _working_row_count(rid1, 1) == 0          # finished run reclaimed
    assert _working_row_count(rid2, row2["attempts"]) == active_rows
    # Idempotent: a second pass removes nothing.
    assert check.purge_orphan_working_state() == 0


# --- successor verifies the complete current version -----------------------

def test_successor_finds_real_conflict_path_on_full_version(data_dir):
    business = BusinessRepository()
    edges = [("S1", "M1"), ("M1", "M2"), ("M2", "S2")]
    business.add_relations(edges)
    business.replace_groups([("S1", "train"), ("S2", "val")])
    check = CheckRepository()
    run_id = check.enqueue(business.current_version())
    check.claim_next_run("worker-a")
    _force_stale(run_id)
    successor = check.claim_next_run("worker-b")

    GraphVerifier(check, business).run(successor)
    row = check.get_run(run_id)
    assert row["status"] == "failed"
    assert row["attempts"] == 2
    conflict = check.conflicts_for(run_id)[0]
    import json

    path = json.loads(conflict["path_json"])
    assert path[0] == "S1" and path[-1] == "S2"
    edgeset = {tuple(sorted(e)) for e in edges}
    for a, b in zip(path, path[1:]):
        assert tuple(sorted((a, b))) in edgeset      # every hop is a real edge
    assert business.acceptance() is None             # conflict never releases


# --- repeated reclaims ------------------------------------------------------

def test_repeated_reclaims_isolate_every_stale_attempt(data_dir, monkeypatch):
    monkeypatch.setattr(settings, "max_attempts", 3)
    _clean_dataset()
    check = CheckRepository()
    business = BusinessRepository()
    run_id = check.enqueue(business.current_version())

    tokens = []
    for expected_attempt, worker in ((1, "w1"), (2, "w2"), (3, "w3")):
        row = check.claim_next_run(worker)
        assert row["attempts"] == expected_attempt
        tokens.append(row["claim_token"])
        if expected_attempt < 3:
            _force_stale(run_id)

    # All three tokens are distinct and the first two are already dead.
    assert len(set(tokens)) == 3
    for stale_token in tokens[:-1]:
        assert check.is_owner(run_id, stale_token) is False
        assert check.heartbeat(run_id, 1, stale_token, 1, 10) is False
        assert check.finish_run(
            run_id, 1, stale_token, "passed", 1, []
        ) is False

    # Third stale attempt hits the ceiling: the run fails, never "safe".
    _force_stale(run_id)
    assert check.claim_next_run("w4") is None
    final = check.get_run(run_id)
    assert final["status"] == "failed"
    assert final["attempts"] == 3
    assert "心跳超时" in final["error_message"]
    assert business.acceptance() is None
    assert check.get_run(run_id)["progress_pct"] in (0.0, 100.0)


# --- real interleaving: stall, reclaim, resume in concurrent threads -------

def test_concurrent_stall_reclaim_resume_keeps_single_conclusion(
    data_dir, monkeypatch
):
    _clean_dataset()
    check = CheckRepository()
    run_id = check.enqueue(BusinessRepository().current_version())
    row_a = check.claim_next_run("worker-a")
    token_a = row_a["claim_token"]

    stalled = threading.Event()
    release = threading.Event()
    original_heartbeat = CheckRepository.heartbeat

    def stalling_heartbeat(self, run_id_, attempt_no, token, processed, total):
        if token == token_a and not stalled.is_set():
            stalled.set()
            assert release.wait(timeout=10), "test harness stall never released"
        return original_heartbeat(
            self, run_id_, attempt_no, token, processed, total
        )

    monkeypatch.setattr(CheckRepository, "heartbeat", stalling_heartbeat)

    verifier_a = GraphVerifier(check, BusinessRepository())
    thread_a = threading.Thread(target=verifier_a.run, args=(row_a,))
    thread_a.start()
    assert stalled.wait(timeout=10)

    # While A is frozen: its heartbeat is stale, B reclaims the run.
    _force_stale(run_id)
    row_b = check.claim_next_run("worker-b")
    assert row_b["attempts"] == 2 and row_b["claim_token"] != token_a

    # Uploads and status-style reads remain available while verification is
    # in flight: business writes use a separate database/short transaction.
    BusinessRepository().add_relations([("D", "E")])
    assert BusinessRepository().current_version() == 3
    assert check.get_run(run_id)["status"] == "running"

    # Release the zombie: its next fenced heartbeat fails and it must abort.
    release.set()
    thread_a.join(timeout=10)
    assert not thread_a.is_alive()
    assert check.get_run(run_id)["worker_id"] == "worker-b"
    # A never concluded; its working partition was removed, B's untouched.
    assert _working_row_count(run_id, 1) == 0

    # B independently verifies the version it claimed (v2 snapshot), then a
    # fresh run verifies the current v3; clean data passes and releases.
    GraphVerifier(check, BusinessRepository()).run(row_b)
    assert check.get_run(run_id)["status"] == "passed"
    rid3 = check.enqueue(BusinessRepository().current_version())
    row3 = check.claim_next_run("worker-b")
    GraphVerifier(check, BusinessRepository()).run(row3)
    assert check.get_run(rid3)["status"] == "passed"
    acceptance = BusinessRepository().acceptance()
    assert acceptance is not None and acceptance["accepted_version"] == 3


def test_status_query_available_during_inflight_verification(
    data_dir, client, monkeypatch
):
    # Upload through the API, then stall the verifier mid-flight.
    r = client.post(
        "/api/upload/relations",
        files={
            "file": (
                "r.csv",
                b"\xe5\x8e\x9f\xe6\xa0\xb7\xe6\x9c\xac\xe5\x8f\xb7,"
                b"\xe8\xa1\x8d\xe7\x94\x9f\xe6\xa0\xb7\xe6\x9c\xac\xe5\x8f\xb7\n"
                b"A,B\nB,C\n",
                "text/csv",
            )
        },
    )
    assert r.status_code == 200
    r = client.post(
        "/api/upload/groups",
        files={
            "file": (
                "g.csv",
                b"\xe6\xa0\xb7\xe6\x9c\xac\xe5\x8f\xb7,\xe7\xbb\x84\xe5\x88\xab\n"
                b"A,train\nB,train\nC,train\n",
                "text/csv",
            )
        },
    )
    run_id = r.json()["verification"]["run_id"]
    row = CheckRepository().claim_next_run("stalled-worker")
    token = row["claim_token"]

    stalled = threading.Event()
    release = threading.Event()
    original_heartbeat = CheckRepository.heartbeat

    def stalling_heartbeat(self, run_id_, attempt_no, tok, processed, total):
        if tok == token and not stalled.is_set():
            stalled.set()
            release.wait(timeout=10)
        return original_heartbeat(
            self, run_id_, attempt_no, tok, processed, total
        )

    monkeypatch.setattr(CheckRepository, "heartbeat", stalling_heartbeat)
    t = threading.Thread(target=GraphVerifier().run, args=(row,))
    t.start()
    assert stalled.wait(timeout=10)
    try:
        # Status reads must keep working while verification is stalled.
        st = client.get("/api/status")
        assert st.status_code == 200
        body = st.json()
        assert body["latest_run"]["run_id"] == run_id
        assert body["download_available"] is False       # incomplete != safe
        assert client.get(f"/api/runs/{run_id}").status_code == 200
        # An upload still commits (short business-DB transaction).
        up = client.post(
            "/api/upload/relations",
            files={
                "file": (
                    "r2.csv",
                    b"\xe5\x8e\x9f\xe6\xa0\xb7\xe6\x9c\xac\xe5\x8f\xb7,"
                    b"\xe8\xa1\x8d\xe7\x94\x9f\xe6\xa0\xb7\xe6\x9c\xac\xe5\x8f\xb7\n"
                    b"C,D\n",
                    "text/csv",
                )
            },
        )
        assert up.status_code == 200
    finally:
        release.set()
        t.join(timeout=10)
    assert not t.is_alive()


# --- bounded memory & schema migration --------------------------------------

def test_snapshot_loading_is_batch_bounded_and_disk_backed(data_dir, monkeypatch):
    # A graph larger than several batches; Python must never receive a batch
    # bigger than BATCH, and all vertices end up on disk.
    n_nodes = 5_000
    edges = [(f"n{i}", f"n{i+1}") for i in range(n_nodes - 1)]
    BusinessRepository().add_relations(edges)
    BusinessRepository().replace_groups(
        [(f"n{i}", "train") for i in range(n_nodes)]
    )
    check = CheckRepository()
    run_id = check.enqueue(BusinessRepository().current_version())
    row = check.claim_next_run("w")

    from app.services import verifier as verifier_module

    monkeypatch.setattr(verifier_module, "BATCH", 500)
    seen_sizes = []
    original_executemany = verifier_module._executemany

    def tracking_executemany(work, sql, parameters):
        params = list(parameters)
        seen_sizes.append(len(params))
        return original_executemany(work, sql, params)

    monkeypatch.setattr(verifier_module, "_executemany", tracking_executemany)
    GraphVerifier(check, BusinessRepository()).run(row)

    assert seen_sizes and max(seen_sizes) <= 500
    finished = check.get_run(run_id)
    assert finished["status"] == "passed"
    # The full vertex set was materialised on disk during the run; only the
    # run-row counters remain after the working partition is reclaimed.
    assert finished["vertices_total"] == n_nodes
    assert finished["edges_total"] == n_nodes - 1
    with connect(settings.check_db) as conn:
        leftover = conn.execute(
            "SELECT COUNT(*) FROM verifier_vertex WHERE run_id=? AND attempt_no=?",
            (row["id"], row["attempts"]),
        ).fetchone()[0]
    assert leftover == 0  # partition cleaned, disk stays bounded


def test_legacy_schema_migrates_to_fenced_partitioned_tables(data_dir):
    # Build a check.db with the pre-fix schema in this fresh data dir.
    import os

    for suffix in ("", "-wal", "-shm"):
        path = settings.check_db + suffix
        if os.path.exists(path):
            os.remove(path)
    legacy_ddl = """
    CREATE TABLE verifier_run (
        id INTEGER PRIMARY KEY AUTOINCREMENT, data_version INTEGER NOT NULL,
        status TEXT NOT NULL, queued_at TEXT NOT NULL DEFAULT (datetime('now')),
        started_at TEXT, finished_at TEXT, progress_pct REAL NOT NULL DEFAULT 0,
        vertices_total INTEGER, vertices_processed INTEGER NOT NULL DEFAULT 0,
        edges_total INTEGER, components_total INTEGER,
        conflicts_total INTEGER NOT NULL DEFAULT 0, error_message TEXT,
        attempts INTEGER NOT NULL DEFAULT 0, worker_id TEXT, heartbeat_at TEXT
    );
    CREATE TABLE verifier_edge (run_id INTEGER NOT NULL, u TEXT, v TEXT);
    CREATE TABLE verifier_vertex (
        run_id INTEGER NOT NULL, sample_no TEXT NOT NULL, grp TEXT,
        component_id INTEGER, visited INTEGER NOT NULL DEFAULT 0
    );
    CREATE TABLE verifier_frontier (run_id INTEGER NOT NULL, sample_no TEXT);
    CREATE TABLE verifier_parent (
        run_id INTEGER NOT NULL, child TEXT NOT NULL, parent TEXT NOT NULL
    );
    """
    with sqlite3.connect(settings.check_db) as conn:
        conn.executescript(legacy_ddl)
        conn.execute(
            "INSERT INTO verifier_run(data_version, status, attempts) "
            "VALUES (1, 'queued', 0)"
        )
        conn.execute(
            "INSERT INTO verifier_edge(run_id, u, v) VALUES (1, 'A', 'B')"
        )
        conn.commit()

    init_databases()

    with connect(settings.check_db) as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(verifier_run)")}
        assert "claim_token" in cols
        edge_cols = {
            r["name"] for r in conn.execute("PRAGMA table_info(verifier_edge)")
        }
        assert edge_cols == {"run_id", "attempt_no", "u", "v"}
        # Unkeyed legacy search state was discarded; run history survived.
        assert conn.execute("SELECT COUNT(*) FROM verifier_edge").fetchone()[0] == 0
        assert tuple(
            conn.execute("SELECT id, status FROM verifier_run").fetchone()
        ) == (1, "queued")

    # The migrated database is fully usable for a fenced run.
    BusinessRepository().replace_groups([("A", "train")])
    check = CheckRepository()
    rid = check.enqueue(BusinessRepository().current_version())
    row = check.claim_next_run("post-migration")
    assert row["claim_token"]
    GraphVerifier(check, BusinessRepository()).run(row)
    assert check.get_run(rid)["status"] == "passed"


def test_no_safety_reported_while_reclaim_is_pending(data_dir, client):
    # Clean data, but the run is only re-queued after a stale attempt:
    # status must keep reporting unavailable until a fresh run completes.
    r = client.post(
        "/api/upload/groups",
        files={
            "file": (
                "g.csv",
                b"\xe6\xa0\xb7\xe6\x9c\xac\xe5\x8f\xb7,\xe7\xbb\x84\xe5\x88\xab\n"
                b"A,train\n",
                "text/csv",
            )
        },
    )
    run_id = r.json()["verification"]["run_id"]
    check = CheckRepository()
    check.claim_next_run("w1")
    _force_stale(run_id)
    check.claim_next_run("w2")  # re-queued + re-claimed, not yet verified

    st = client.get("/api/status").json()
    assert st["download_available"] is False
    assert st["accepted_version"] is None
    assert client.get("/api/release/groups.csv").status_code == 409

    row = check.get_run(run_id)
    GraphVerifier(check, BusinessRepository()).run(row)
    st = client.get("/api/status").json()
    assert st["download_available"] is True
    assert client.get("/api/release/groups.csv").status_code == 200

#!/usr/bin/env python3
"""Standalone two-process reclaim/fencing integration check.

Two real OS processes (each the equivalent of api1 / api2) share one data
directory. Process A stalls mid-verification until its heartbeat is reaped;
process B re-claims the same run and finishes it; A then wakes up and must
not touch B's conclusion or working data. Run with -I so untrusted-path
import tricks cannot affect the interpreter.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def worker_a(data_dir: str, ready, release) -> dict:
    os.environ["LEAKCHECK_DATA_DIR"] = data_dir
    os.environ["LEAKCHECK_DISABLE_WORKER"] = "1"
    sys.path.insert(0, str(ROOT))
    from app.config import settings
    from app.database import init_databases
    from app.repositories.check import CheckRepository
    from app.services.verifier import GraphVerifier

    init_databases()
    check = CheckRepository()
    row = check.claim_next_run(f"procA-{os.getpid()}")
    token = row["claim_token"]
    original = CheckRepository.heartbeat

    def stalling(self, run_id, attempt_no, tok, processed, total):
        if tok == token:
            ready.set()
            release.wait(timeout=30)
        return original(self, run_id, attempt_no, tok, processed, total)

    CheckRepository.heartbeat = stalling
    GraphVerifier(check).run(row)
    return {"final_status": check.get_run(row["id"])["status"],
            "a_token": token}


def main() -> int:
    data_dir = tempfile.mkdtemp(prefix="leakcheck-2proc-")
    os.environ["LEAKCHECK_DATA_DIR"] = data_dir
    os.environ["LEAKCHECK_DISABLE_WORKER"] = "1"
    os.environ["LEAKCHECK_HEARTBEAT_TIMEOUT"] = "2"
    sys.path.insert(0, str(ROOT))
    from app.config import settings
    settings.data_dir = data_dir
    settings.business_db = os.path.join(data_dir, "business.db")
    settings.check_db = os.path.join(data_dir, "check.db")
    settings.snapshot_dir = os.path.join(data_dir, "snapshots")
    settings.heartbeat_timeout = 2.0
    from app.database import init_databases
    from app.repositories.business import BusinessRepository
    from app.repositories.check import CheckRepository
    from app.services.verifier import GraphVerifier
    from datetime import datetime, timedelta, timezone

    init_databases()
    BusinessRepository().add_relations([("A", "B"), ("B", "C")])
    BusinessRepository().replace_groups(
        [("A", "train"), ("B", "train"), ("C", "train")]
    )
    check = CheckRepository()
    run_id = check.enqueue(BusinessRepository().current_version())

    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    proc_a = ctx.Process(
        target=worker_a, args=(data_dir, ready, release), name="api1"
    )
    proc_a.start()
    if not ready.wait(timeout=30):
        print("FAIL: process A never reached its first heartbeat")
        return 1
    print(f"[api1] stalled mid-run on run {run_id}; ageing heartbeat")

    stale = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    import sqlite3

    with sqlite3.connect(settings.check_db) as conn:
        conn.execute(
            "UPDATE verifier_run SET heartbeat_at=? WHERE id=?", (stale, run_id)
        )
        conn.commit()

    row_b = check.claim_next_run(f"procB-{os.getpid()}")
    assert row_b is not None and row_b["id"] == run_id
    print(f"[api2] re-claimed run {run_id} as attempt {row_b['attempts']} "
          f"with new token {row_b['claim_token'][:8]}")

    # Let the zombie wake up and run to completion first: it must abort.
    release.set()
    proc_a.join(timeout=30)
    assert not proc_a.is_alive(), "zombie process did not exit"
    mid = check.get_run(run_id)
    print(f"[api1] woke up; run still owned by {mid['worker_id']}, "
          f"status={mid['status']}")
    assert mid["worker_id"].startswith("procB"), "zombie overwrote ownership"
    assert mid["status"] == "running", "zombie wrote a conclusion"

    # Now the successor completes the full verification independently.
    GraphVerifier(check).run(row_b)
    final = check.get_run(run_id)
    acceptance = BusinessRepository().acceptance()
    print(f"[api2] finished run {run_id}: status={final['status']}, "
          f"attempts={final['attempts']}, acceptance="
          f"{acceptance['accepted_version'] if acceptance else None}")
    assert final["status"] == "passed"
    assert acceptance is not None and acceptance["accepted_version"] == 2

    # Upload availability during the drama: add one more edge and verify it.
    BusinessRepository().add_relations([("C", "D")])
    rid = check.enqueue(BusinessRepository().current_version())
    GraphVerifier(check).run(check.claim_next_run("procB-followup"))
    assert check.get_run(rid)["status"] == "passed"
    assert BusinessRepository().acceptance()["accepted_version"] == 3
    print("PASS: two-process reclaim fencing, clean release, follow-up upload OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

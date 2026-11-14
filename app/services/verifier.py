"""Streaming cross-group leakage verifier.

The derivation relation is treated as an *undirected* graph. Two samples are
in conflict when they are connected by any path and carry different groups
(samples without a group assignment are neutral and never trigger conflicts).

Memory discipline (requirement: never hold the whole graph or all search
state in memory):

* the immutable snapshot at the bound data version is copied edge by edge
  into disk-backed working tables in ``check.db``;
* BFS frontier and parent pointers live in SQLite tables, not Python sets;
* Python only ever holds one edge/vertex batch and one reconstructed path.

The business database is opened read-only through a snapshot transaction, so
uploads and queries remain available during verification and no business-DB
write transaction is held.

Ownership discipline (reclaim safety): every run attempt is identified by
``(run_id, attempt_no, claim_token)``. All working rows carry the attempt
number and every heartbeat / conclusion write is fenced by the token. When
the attempt is reaped mid-flight and another process re-claims the run, the
zombie's fenced writes fail and its search state stays in its own attempt
partition; it can never update the successor's progress, erase its data,
write a conclusion or trigger a release.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Dict, List, Optional, Tuple

from ..config import settings
from ..database import connect
from ..repositories.business import BusinessRepository
from ..repositories.check import CheckRepository, WORKING_TABLES

logger = logging.getLogger("leakcheck.verifier")

BATCH = 2_000


def _executemany(work: sqlite3.Connection, sql: str, rows: List[tuple]) -> None:
    """Single batched-write entry point for on-disk working tables.

    Keeping it module-level lets tests observe batch sizes without patching
    SQLite's immutable connection type, while production code funnels every
    bounded insert through one place.
    """
    work.executemany(sql, rows)


class OwnershipLost(RuntimeError):
    """Raised when the attempt's claim token is no longer the active one."""


def build_worker_id() -> str:
    import socket
    import uuid

    return f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"


class GraphVerifier:
    def __init__(
        self,
        check_repo: Optional[CheckRepository] = None,
        business_repo: Optional[BusinessRepository] = None,
    ) -> None:
        self.check = check_repo or CheckRepository()
        self.business = business_repo or BusinessRepository()

    # --- ownership helpers ------------------------------------------------

    @staticmethod
    def _require_owner(
        check: CheckRepository, run_id: int, attempt_no: int, token: Optional[str]
    ) -> None:
        if not check.is_owner(run_id, token):
            raise OwnershipLost(
                f"run {run_id} attempt {attempt_no} 已失去核查所有权"
                "（任务被其他进程重领），本执行者放弃所有写入"
            )

    # --- snapshot loading -------------------------------------------------

    def _load_snapshot(
        self,
        work: sqlite3.Connection,
        run_id: int,
        attempt_no: int,
        version: int,
    ) -> Tuple[int, int]:
        """Materialise the graph exactly as it existed at *version*.

        Edges are append-only, so filtering by ``version <= ?`` reconstructs
        the historical edge set. Groups use the newest batch whose own
        version is <= the bound version. Everything lands on disk in the
        attempt's own partition in bounded-size batches; Python never holds
        the complete vertex set in memory.
        """
        read = self._open_ro()
        try:
            current = read.execute(
                "SELECT COALESCE(MAX(version),0) v FROM data_version"
            ).fetchone()["v"]
            if current < version:
                raise RuntimeError(
                    f"核查目标版本 {version} 高于当前版本 {current}，数据已回退？"
                )
            # Undirected: stream both directions of every edge straight into
            # the attempt partition in batches.
            edge_count = 0
            batch: List[tuple] = []
            edge_rows = read.execute(
                "SELECT parent_sample, child_sample FROM derivation_edge "
                "WHERE version <= ?",
                (version,),
            )
            for parent, child in edge_rows:
                batch.extend(
                    [
                        (run_id, attempt_no, parent, child),
                        (run_id, attempt_no, child, parent),
                    ]
                )
                edge_count += 1
                if len(batch) >= BATCH:
                    _executemany(
                        work,
                        "INSERT INTO verifier_edge"
                        "(run_id, attempt_no, u, v) VALUES (?,?,?,?)",
                        batch,
                    )
                    batch.clear()
            if batch:
                _executemany(
                    work,
                    "INSERT INTO verifier_edge"
                    "(run_id, attempt_no, u, v) VALUES (?,?,?,?)",
                    batch,
                )
            work.commit()

            # Vertices = union of edge endpoints and grouped samples. Both
            # inputs are inserted first; a unique index + INSERT OR IGNORE
            # performs the union on disk, so no full vertex set is held in
            # Python memory.
            batch_row = read.execute(
                "SELECT batch_id FROM group_batch WHERE version <= ? "
                "ORDER BY version DESC, batch_id DESC LIMIT 1",
                (version,),
            ).fetchone()

            vertex_count = 0
            work.execute(
                "INSERT OR IGNORE INTO verifier_vertex"
                "(run_id, attempt_no, sample_no, grp) "
                "SELECT DISTINCT run_id, ?, u, NULL FROM verifier_edge "
                "WHERE run_id=? AND attempt_no=?",
                (attempt_no, run_id, attempt_no),
            )
            if batch_row:
                group_batch: List[tuple] = []
                for r in read.execute(
                    "SELECT sample_no, grp FROM sample_group WHERE batch_id=?",
                    (batch_row["batch_id"],),
                ):
                    group_batch.append(
                        (run_id, attempt_no, r["sample_no"], r["grp"])
                    )
                    if len(group_batch) >= BATCH:
                        self._apply_group_batch(work, group_batch)
                        group_batch.clear()
                if group_batch:
                    self._apply_group_batch(work, group_batch)
            work.commit()
            vertex_count = work.execute(
                "SELECT COUNT(*) FROM verifier_vertex "
                "WHERE run_id=? AND attempt_no=?",
                (run_id, attempt_no),
            ).fetchone()[0]
        finally:
            read.close()
        return vertex_count, edge_count

    @staticmethod
    def _apply_group_batch(
        work: sqlite3.Connection, group_batch: List[tuple]
    ) -> None:
        """Merge one bounded group batch into the on-disk vertex table.

        INSERT OR IGNORE covers grouped samples with no edge at all; the
        batched UPDATE backfills groups for vertices discovered from edges.
        Groups come from the read-only business connection, so no business
        table is ever referenced from the check.db connection.
        """
        _executemany(
            work,
            "INSERT OR IGNORE INTO verifier_vertex"
            "(run_id, attempt_no, sample_no, grp) VALUES (?,?,?,?)",
            group_batch,
        )
        _executemany(
            work,
            "UPDATE verifier_vertex SET grp=? "
            "WHERE run_id=? AND attempt_no=? AND sample_no=? AND grp IS NULL",
            [(grp, rid, an, sample) for rid, an, sample, grp in group_batch],
        )

    @staticmethod
    def _open_ro() -> sqlite3.Connection:
        """Open the business DB read-only; writes are impossible by design."""
        from ..database import _apply_pragmas  # reuse row factory + busy timeout

        uri = f"file:{settings.business_db}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=15)
        _apply_pragmas(conn)
        return conn

    # --- BFS --------------------------------------------------------------

    def _component_groups(
        self,
        work: sqlite3.Connection,
        run_id: int,
        attempt_no: int,
        component_id: int,
    ) -> Dict[str, str]:
        """Return one representative sample per group inside a component."""
        reps: Dict[str, str] = {}
        for r in work.execute(
            "SELECT sample_no, grp FROM verifier_vertex "
            "WHERE run_id=? AND attempt_no=? AND component_id=? "
            "AND grp IS NOT NULL",
            (run_id, attempt_no, component_id),
        ):
            reps.setdefault(r["grp"], r["sample_no"])
        return reps

    def _ancestor_chain(
        self, work: sqlite3.Connection, run_id: int, attempt_no: int, node: str
    ) -> List[str]:
        """Return [node, parent, grandparent, ...] until the chain ends."""
        chain: List[str] = []
        current = node
        for _ in range(1_000_000):
            chain.append(current)
            row = work.execute(
                "SELECT parent FROM verifier_parent "
                "WHERE run_id=? AND attempt_no=? AND child=?",
                (run_id, attempt_no, current),
            ).fetchone()
            if row is None:  # component root has no parent row
                break
            current = row["parent"]
        return chain

    def _path_between(
        self,
        work: sqlite3.Connection,
        run_id: int,
        attempt_no: int,
        anchor: str,
        target: str,
    ) -> List[str]:
        """Return one real anchor->target path via the BFS parent tree.

        We take the full ancestor set of ``anchor`` (all nodes reachable by
        following parent pointers, including ``anchor`` itself). Then we walk
        ``target``'s parent chain until the first node in that set: that node
        is the junction of the two tree branches. The final path is

            anchor -> ... -> junction -> ... -> target

        which is a genuine graph path regardless of how the parent pointers
        are oriented relative to the BFS root.
        """
        anchor_chain = self._ancestor_chain(work, run_id, attempt_no, anchor)
        anchor_set = set(anchor_chain)
        target_chain = self._ancestor_chain(work, run_id, attempt_no, target)

        junction_pos = next(
            pos for pos, node in enumerate(target_chain) if node in anchor_set
        )
        junction = target_chain[junction_pos]
        anchor_pos = anchor_chain.index(junction)

        anchor_to_junction = anchor_chain[: anchor_pos + 1]
        junction_to_target = list(reversed(target_chain[:junction_pos]))
        return anchor_to_junction + junction_to_target

    def _bfs_component(
        self,
        work: sqlite3.Connection,
        run_id: int,
        attempt_no: int,
        token: Optional[str],
        root: str,
        component_id: int,
        heartbeat_cb,
        processed: int,
        total: int,
    ) -> Tuple[int, List[dict], int]:
        work.execute(
            "DELETE FROM verifier_frontier WHERE run_id=? AND attempt_no=?",
            (run_id, attempt_no),
        )
        work.execute(
            "DELETE FROM verifier_parent WHERE run_id=? AND attempt_no=?",
            (run_id, attempt_no),
        )
        work.execute(
            "INSERT INTO verifier_frontier(run_id, attempt_no, sample_no) "
            "VALUES (?,?,?)",
            (run_id, attempt_no, root),
        )
        work.execute(
            "UPDATE verifier_vertex SET visited=1, component_id=? "
            "WHERE run_id=? AND attempt_no=? AND sample_no=?",
            (component_id, run_id, attempt_no, root),
        )
        size = 0
        while True:
            nodes = [
                r["sample_no"]
                for r in work.execute(
                    "SELECT sample_no FROM verifier_frontier "
                    "WHERE run_id=? AND attempt_no=? LIMIT ?",
                    (run_id, attempt_no, BATCH),
                )
            ]
            if not nodes:
                break
            placeholders = ",".join("?" for _ in nodes)
            work.execute(
                f"DELETE FROM verifier_frontier WHERE run_id=? AND attempt_no=? "
                f"AND sample_no IN ({placeholders})",
                (run_id, attempt_no, *nodes),
            )
            next_frontier: List[tuple] = []
            for node in nodes:
                size += 1
                processed += 1
                for r in work.execute(
                    "SELECT v FROM verifier_edge "
                    "WHERE run_id=? AND attempt_no=? AND u=?",
                    (run_id, attempt_no, node),
                ):
                    neighbour = r["v"]
                    owner = work.execute(
                        "SELECT visited FROM verifier_vertex "
                        "WHERE run_id=? AND attempt_no=? AND sample_no=?",
                        (run_id, attempt_no, neighbour),
                    ).fetchone()
                    if owner["visited"]:
                        continue
                    work.execute(
                        "UPDATE verifier_vertex SET visited=1, component_id=? "
                        "WHERE run_id=? AND attempt_no=? AND sample_no=?",
                        (component_id, run_id, attempt_no, neighbour),
                    )
                    work.execute(
                        "INSERT INTO verifier_parent"
                        "(run_id, attempt_no, child, parent) "
                        "VALUES (?,?,?,?)",
                        (run_id, attempt_no, neighbour, node),
                    )
                    next_frontier.append((run_id, attempt_no, neighbour))
                if processed % BATCH == 0:
                    work.commit()
                    if not heartbeat_cb(processed, total):
                        raise OwnershipLost(
                            f"run {run_id} attempt {attempt_no} 心跳续约被拒，"
                            "已失去核查所有权"
                        )
            if next_frontier:
                _executemany(
                    work,
                    "INSERT INTO verifier_frontier"
                    "(run_id, attempt_no, sample_no) VALUES (?,?,?)",
                    next_frontier,
                )
            work.commit()
            if not heartbeat_cb(processed, total):
                raise OwnershipLost(
                    f"run {run_id} attempt {attempt_no} 心跳续约被拒，"
                    "已失去核查所有权"
                )

        conflicts: List[dict] = []
        groups = self._component_groups(work, run_id, attempt_no, component_id)
        if len(groups) > 1:
            # Use the first group's representative as a stable anchor and
            # report one conflict per other group. Representatives come from
            # the component group map, so this does not depend on where the
            # BFS root happens to sit in the parent tree.
            group_names = sorted(groups)
            anchor_group = group_names[0]
            anchor_sample = groups[anchor_group]
            for grp in group_names[1:]:
                target = groups[grp]
                full_path = self._path_between(
                    work, run_id, attempt_no, anchor_sample, target
                )
                conflicts.append(
                    {
                        "sample_a": anchor_sample,
                        "group_a": anchor_group,
                        "sample_b": target,
                        "group_b": grp,
                        "component_size": size,
                        "path": full_path,
                    }
                )
        return size, conflicts, processed

    # --- entry point ------------------------------------------------------

    def run(self, run_row: sqlite3.Row) -> None:
        run_id = int(run_row["id"])
        attempt_no = int(run_row["attempts"])
        token = run_row["claim_token"]
        version = int(run_row["data_version"])
        logger.info(
            "run %s attempt %s starts for data version %s",
            run_id,
            attempt_no,
            version,
        )
        work = connect(settings.check_db)
        finished_cleanly = False
        try:
            # Defensive ownership check before touching any working table:
            # the row could already have been re-claimed while queued.
            self._require_owner(self.check, run_id, attempt_no, token)
            total, edge_count = self._load_snapshot(
                work, run_id, attempt_no, version
            )
            self._require_owner(self.check, run_id, attempt_no, token)
            work.execute(
                "UPDATE verifier_run SET vertices_total=?, edges_total=? "
                "WHERE id=? AND status='running' AND claim_token=?",
                (total, edge_count, run_id, token),
            )
            work.commit()

            def heartbeat(processed: int, tot: int) -> bool:
                return self.check.heartbeat(
                    run_id, attempt_no, token, processed, tot
                )

            component_id = 0
            all_conflicts: List[dict] = []
            processed = 0
            while True:
                # Prefer a grouped sample as the BFS root: this keeps every
                # reported group representative inside one rooted parent tree
                # whose root is itself one of the conflict endpoints.
                row = work.execute(
                    "SELECT sample_no FROM verifier_vertex "
                    "WHERE run_id=? AND attempt_no=? AND visited=0 "
                    "AND grp IS NOT NULL LIMIT 1",
                    (run_id, attempt_no),
                ).fetchone()
                if row is None:
                    row = work.execute(
                        "SELECT sample_no FROM verifier_vertex "
                        "WHERE run_id=? AND attempt_no=? AND visited=0 LIMIT 1",
                        (run_id, attempt_no),
                    ).fetchone()
                if row is None:
                    break
                component_id += 1
                _, conflicts, processed = self._bfs_component(
                    work,
                    run_id,
                    attempt_no,
                    token,
                    row["sample_no"],
                    component_id,
                    heartbeat,
                    processed,
                    total,
                )
                all_conflicts.extend(conflicts)
                if not heartbeat(processed, total):
                    raise OwnershipLost(
                        f"run {run_id} attempt {attempt_no} 心跳续约被拒，"
                        "已失去核查所有权"
                    )

            status = "passed" if not all_conflicts else "failed"
            # The fenced finish is the single chokepoint: a zombie whose
            # token was rotated gets False and must not conclude or release.
            accepted = self.check.finish_run(
                run_id, attempt_no, token, status, component_id, all_conflicts
            )
            if not accepted:
                logger.warning(
                    "run %s attempt %s lost ownership before finish; "
                    "result discarded",
                    run_id,
                    attempt_no,
                )
                return
            finished_cleanly = True
            if status == "passed":
                self._promote_if_current(version, run_id, attempt_no, token)
            logger.info(
                "run %s finished: %s (%s conflicts, %s components)",
                run_id,
                status,
                len(all_conflicts),
                component_id,
            )
        except OwnershipLost:
            # The run was re-claimed while we were working. Every fenced
            # write already failed; just stop and leave the successor's data
            # and conclusion untouched.
            logger.warning(
                "run %s attempt %s abandoned: ownership lost to a successor",
                run_id,
                attempt_no,
            )
        except Exception as exc:  # noqa: BLE001 - record failure if still owner
            logger.exception("run %s attempt %s errored", run_id, attempt_no)
            self.check.finish_run(
                run_id,
                attempt_no,
                token,
                "failed",
                0,
                [],
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            # Reclaim only THIS attempt's disk partition. A successor runs on
            # a different attempt_no, so its search data is never deleted.
            # When we still own the run and failed out, drop our rows; when
            # ownership was lost, likewise only our own rows are affected.
            for table in WORKING_TABLES:
                work.execute(
                    f"DELETE FROM {table} WHERE run_id=? AND attempt_no=?",
                    (run_id, attempt_no),
                )
            work.commit()
            work.close()
            logger.info(
                "run %s attempt %s cleaned up its own working partition%s",
                run_id,
                attempt_no,
                "" if finished_cleanly else " (attempt did not conclude)",
            )

    def _promote_if_current(
        self, version: int, run_id: int, attempt_no: int, token: Optional[str]
    ) -> None:
        """Only accept when the conclusion still matches the latest version.

        Re-checks ownership immediately before the release so a reclaim that
        races the finish cannot cause a stale zombie to trigger a release.
        """
        self._require_owner(self.check, run_id, attempt_no, token)
        current = self.business.current_version()
        if current != version:
            logger.info(
                "run %s passed v%s but current is v%s; not releasing",
                run_id,
                version,
                current,
            )
            return
        # One last fence between the version check and the actual release.
        self._require_owner(self.check, run_id, attempt_no, token)
        groups = self.business.active_groups()
        self.business.accept_version(version, run_id, groups)

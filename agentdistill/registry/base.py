"""Registry: the durable record of traces, datasets, runs, adapters, and calibrations.

SQLAlchemy Core over a URL, so the same code serves `sqlite:///.agentdistill/registry.db` locally and a Postgres
URL for a team. Dialect differences (JSONB vs TEXT, arrays vs JSON text, vectors) are confined to the JSON codec
here and to the two migration files; nothing above this layer knows which backend it is talking to.
"""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.exc import OperationalError

MIGRATIONS = Path(__file__).resolve().parent / "migrations"
SCHEMA_VERSION = 6

#: Migrations are applied in order; each is idempotent.
MIGRATION_FILES = (
    "001_init.sql",
    "002_eval_results.sql",
    "003_phase3.sql",
    "004_phase3c.sql",
    "005_prompt_tokens.sql",
    "006_rft_kind.sql",
)


def invocation() -> str:
    """The command that produced a row, so a report can print how to reproduce it.

    `argv[0]` is normalized to `agentdistill`. Run through `python -m agentdistill.cli` it is an absolute path
    to cli.py, which makes the report's "how to reproduce" block something you have to edit before you can run
    it -- and a reproduction command nobody can paste is not one anybody checks.
    """
    import sys
    from pathlib import Path

    argv = list(sys.argv) or ["agentdistill"]
    head = Path(argv[0]).name
    if head in ("cli.py", "__main__.py", "agentdistill", "pytest", "__main__"):
        argv[0] = "agentdistill"
    return " ".join(argv)


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


def dumps(value: Any) -> str | None:
    """JSON columns are stored as text on SQLite and cast to JSONB on Postgres; both accept a JSON string."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=False)


def loads(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list)):
        return value
    return json.loads(value)


def split_statements(sql: str) -> list[str]:
    """Split a migration file into statements.

    Naive `sql.split(";")` is wrong: a `;` inside a `--` comment truncates the statement before it. Comments are
    stripped first, and string literals are respected so a semicolon inside quotes stays put.
    """
    out: list[str] = []
    buf: list[str] = []
    in_str = False
    i = 0
    while i < len(sql):
        ch = sql[i]
        if in_str:
            buf.append(ch)
            if ch == "'":
                if i + 1 < len(sql) and sql[i + 1] == "'":  # escaped quote
                    buf.append(sql[i + 1])
                    i += 2
                    continue
                in_str = False
            i += 1
            continue
        if ch == "'":
            in_str = True
            buf.append(ch)
            i += 1
            continue
        if sql.startswith("--", i):
            i = sql.find("\n", i)
            if i == -1:
                break
            continue
        if ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                out.append(stmt)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def as_bool(value: Any) -> bool | None:
    """SQLite stores booleans as 0/1; Postgres returns real booleans."""
    if value is None:
        return None
    return bool(value)


def _enable_sqlite_foreign_keys(engine: Engine) -> None:
    """SQLite applies `PRAGMA foreign_keys` per connection, not per database.

    Setting it once inside the migration only affects whichever pooled connection happened to run it, so
    referential integrity would be enforced or not depending on pool reuse. Registering it on every checkout makes
    the behaviour deterministic: a sample can never reference a trace that was not ingested.
    """

    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_connection, _record):  # pragma: no cover - trivial callback
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


class Registry:
    """Thin, explicit data access. No ORM models: the schema is the SQL file, and queries stay readable."""

    def __init__(self, url: str, root: Path | None = None) -> None:
        self.url = self._resolve_url(url, root)
        self.dialect = "postgres" if self.url.startswith(("postgresql", "postgres")) else "sqlite"
        self.engine: Engine = create_engine(self.url, future=True)
        if self.dialect == "sqlite":
            _enable_sqlite_foreign_keys(self.engine)

    @staticmethod
    def _resolve_url(url: str, root: Path | None) -> str:
        """Make a relative SQLite path relative to the config, not to the process's cwd."""
        prefix = "sqlite:///"
        if root is None or not url.startswith(prefix):
            return url
        raw = url[len(prefix) :]
        if raw.startswith("/") or raw == ":memory:":
            return url
        resolved = (root / raw).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return f"{prefix}{resolved}"

    @classmethod
    def from_config(cls, cfg: Any) -> Registry:
        return cls(cfg.registry, root=cfg.root)

    # ----------------------------------------------------------------------------------------------------------
    # schema
    # ----------------------------------------------------------------------------------------------------------

    def migrate(self) -> None:
        """Apply migrations that have not been applied yet, once each.

        Every file is recorded in `schema_migrations` after it runs, and a recorded file is skipped. The earlier
        version re-ran every file on every open and relied on each one being idempotent, which held while
        migrations only added columns and indexes. It stops holding the moment a migration has to rebuild a
        table -- SQLite cannot alter a CHECK constraint in place -- and rebuilding a table on every process
        start is both wasteful and a window in which an interrupted run loses rows.
        """
        with self._migration_connection() as conn:
            self._ensure_migration_ledger(conn)
            applied = self._applied_migrations(conn)

            for filename in MIGRATION_FILES:
                if filename in applied:
                    continue
                sql_path = MIGRATIONS / self.dialect / filename
                for stmt in split_statements(sql_path.read_text()):
                    if stmt.upper().startswith("PRAGMA") and self.dialect != "sqlite":
                        continue
                    try:
                        conn.execute(text(stmt))
                    except OperationalError as e:
                        # A database created before the ledger existed has already had these applied. SQLite has
                        # no `ADD COLUMN IF NOT EXISTS`, so replaying one raises rather than being a no-op.
                        if "duplicate column name" not in str(e).lower():
                            raise
                conn.execute(
                    text("INSERT INTO schema_migrations (filename, applied_at) VALUES (:f, :t)"),
                    {"f": filename, "t": utcnow()},
                )

            if not self._has_version(conn):
                conn.execute(
                    text("INSERT INTO schema_version (version, applied_at) VALUES (:v, :t)"),
                    {"v": SCHEMA_VERSION, "t": utcnow()},
                )

    @contextmanager
    def _migration_connection(self) -> Any:
        """A transactional connection with SQLite's foreign keys disabled for the duration.

        Rebuilding a table -- which is the only way to change a CHECK constraint in SQLite -- means dropping
        one that other tables reference, and `PRAGMA foreign_keys` is a no-op inside a transaction. So the
        pragma is set on an autocommitting connection first, and the migrations then run inside an explicit
        transaction on that same connection. This is the procedure SQLite's own documentation prescribes, with
        the `foreign_key_check` at the end that makes it safe: if a rebuild orphaned a row, the whole migration
        rolls back rather than leaving a database that passes startup and fails later.
        """
        connection = self.engine.connect().execution_options(isolation_level="AUTOCOMMIT")
        try:
            if self.dialect == "sqlite":
                connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
            connection.exec_driver_sql("BEGIN")
            try:
                yield connection
                if self.dialect == "sqlite":
                    orphans = connection.exec_driver_sql("PRAGMA foreign_key_check").fetchall()
                    if orphans:
                        raise RuntimeError(
                            f"a migration left {len(orphans)} orphaned row(s): {orphans[:5]}. "
                            f"Nothing was committed."
                        )
                connection.exec_driver_sql("COMMIT")
            except BaseException:
                connection.exec_driver_sql("ROLLBACK")
                raise
            finally:
                if self.dialect == "sqlite":
                    connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        finally:
            connection.close()

    @staticmethod
    def _ensure_migration_ledger(conn: Any) -> None:
        conn.execute(
            text(
                """CREATE TABLE IF NOT EXISTS schema_migrations (
                       filename   TEXT PRIMARY KEY,
                       applied_at TEXT NOT NULL
                   )"""
            )
        )

    @staticmethod
    def _applied_migrations(conn: Any) -> set[str]:
        return {r[0] for r in conn.execute(text("SELECT filename FROM schema_migrations")).fetchall()}

    def _has_version(self, conn: Any) -> bool:
        row = conn.execute(text("SELECT version FROM schema_version WHERE version = :v"), {"v": SCHEMA_VERSION}).first()
        return row is not None

    def schema_version(self) -> int | None:
        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT MAX(version) FROM schema_version")).first()
            return row[0] if row else None

    # ----------------------------------------------------------------------------------------------------------
    # traces
    # ----------------------------------------------------------------------------------------------------------

    _TRACE_INSERT = """
        INSERT INTO traces (id, source, source_ref, task_id, task_input, messages, tools, teacher_model, success,
                            grader, score, n_turns, n_tool_calls, prompt_tokens, completion_tokens, cost_usd,
                            content_hash, cluster, metadata, created_at)
        VALUES (:id, :source, :source_ref, :task_id, :task_input, :messages, :tools, :teacher_model, :success,
                :grader, :score, :n_turns, :n_tool_calls, :prompt_tokens, :completion_tokens, :cost_usd,
                :content_hash, :cluster, :metadata, :created_at)
    """

    def insert_traces(self, traces: list[dict]) -> dict[str, int]:
        """Insert normalized traces, skipping exact duplicates by `content_hash`.

        Returns counts so `ingest` can report what it actually added rather than what it read.
        """
        added = skipped = 0
        existing = self.existing_hashes({t["content_hash"] for t in traces})
        seen: set[str] = set()
        with self.engine.begin() as conn:
            for t in traces:
                h = t["content_hash"]
                if h in existing or h in seen:
                    skipped += 1
                    continue
                seen.add(h)
                conn.execute(text(self._TRACE_INSERT), self._trace_params(t))
                added += 1
        return {"added": added, "skipped_duplicate": skipped, "read": len(traces)}

    @staticmethod
    def _trace_params(t: dict) -> dict[str, Any]:
        return {
            "id": t["id"],
            "source": t["source"],
            "source_ref": t.get("source_ref"),
            "task_id": t.get("task_id"),
            "task_input": dumps(t.get("task_input")),
            "messages": dumps(t["messages"]),
            "tools": dumps(t.get("tools") or []),
            "teacher_model": t.get("teacher_model"),
            "success": t.get("success"),
            "grader": t.get("grader"),
            "score": t.get("score"),
            "n_turns": t.get("n_turns"),
            "n_tool_calls": t.get("n_tool_calls"),
            "prompt_tokens": t.get("prompt_tokens"),
            "completion_tokens": t.get("completion_tokens"),
            "cost_usd": t.get("cost_usd"),
            "content_hash": t["content_hash"],
            "cluster": t.get("cluster"),
            "metadata": dumps(t.get("metadata")),
            "created_at": t.get("created_at") or utcnow(),
        }

    def existing_hashes(self, hashes: set[str]) -> set[str]:
        if not hashes:
            return set()
        out: set[str] = set()
        ordered = list(hashes)
        with self.engine.connect() as conn:
            for i in range(0, len(ordered), 500):
                chunk = ordered[i : i + 500]
                params = {f"h{j}": h for j, h in enumerate(chunk)}
                placeholders = ", ".join(f":{k}" for k in params)
                rows = conn.execute(
                    text(f"SELECT content_hash FROM traces WHERE content_hash IN ({placeholders})"), params
                ).fetchall()
                out.update(r[0] for r in rows)
        return out

    def list_traces(self, source: str | None = None, limit: int | None = None) -> list[dict]:
        """Traces in a stable order (created_at, then id) so that curation is reproducible."""
        q = "SELECT * FROM traces"
        params: dict[str, Any] = {}
        if source:
            q += " WHERE source = :source"
            params["source"] = source
        q += " ORDER BY created_at, id"
        if limit:
            q += " LIMIT :limit"
            params["limit"] = limit
        with self.engine.connect() as conn:
            rows = conn.execute(text(q), params).mappings().fetchall()
        return [self._row_to_trace(r) for r in rows]

    def get_trace(self, trace_id: str) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM traces WHERE id = :id"), {"id": trace_id}).mappings().first()
        return self._row_to_trace(row) if row else None

    @staticmethod
    def _row_to_trace(row: Any) -> dict:
        t = dict(row)
        t["task_input"] = loads(t.get("task_input"))
        t["messages"] = loads(t["messages"])
        t["tools"] = loads(t["tools"])
        t["metadata"] = loads(t.get("metadata"))
        t["success"] = as_bool(t.get("success"))
        return t

    def set_clusters(self, assignments: dict[str, int]) -> None:
        with self.engine.begin() as conn:
            for trace_id, cluster in assignments.items():
                conn.execute(
                    text("UPDATE traces SET cluster = :c WHERE id = :id"), {"c": int(cluster), "id": trace_id}
                )

    def count_traces(self) -> int:
        with self.engine.connect() as conn:
            return int(conn.execute(text("SELECT COUNT(*) FROM traces")).scalar_one())

    # ----------------------------------------------------------------------------------------------------------
    # datasets and samples
    # ----------------------------------------------------------------------------------------------------------

    def next_dataset_version(self, name: str) -> int:
        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT MAX(version) FROM datasets WHERE name = :n"), {"n": name}).first()
        return (row[0] or 0) + 1 if row else 1

    def insert_dataset(self, ds: dict) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """INSERT INTO datasets (id, name, version, kind, filter_config, n_samples, n_tokens,
                                             content_hash, path, report_path, created_at)
                       VALUES (:id, :name, :version, :kind, :filter_config, :n_samples, :n_tokens,
                               :content_hash, :path, :report_path, :created_at)"""
                ),
                {
                    **ds,
                    "filter_config": dumps(ds["filter_config"]),
                    "created_at": ds.get("created_at") or utcnow(),
                    "report_path": ds.get("report_path"),
                },
            )

    def insert_samples(self, samples: list[dict]) -> None:
        if not samples:
            return
        with self.engine.begin() as conn:
            # SQLAlchemy 2.0 executes a list of parameter dicts as an executemany.
            conn.execute(
                text(
                    """INSERT INTO samples (id, dataset_id, trace_id, kind, n_tokens, n_target_tokens, row_idx,
                                            sample_hash)
                       VALUES (:id, :dataset_id, :trace_id, :kind, :n_tokens, :n_target_tokens, :row_idx,
                               :sample_hash)"""
                ),
                samples,
            )

    def get_dataset(self, name: str, version: int | None = None) -> dict | None:
        """Look a dataset up by name (latest version, or a given one) or by id.

        Both, because the selectors print ids -- `dataset latest` is what the GPU script feeds straight into
        `train sft` -- while a person types a name. Accepting only one of the two means the script and the
        human need different commands, and the script's version is the one nobody runs until the GPU day.
        """
        q = "SELECT * FROM datasets WHERE name = :n OR id = :n"
        params: dict[str, Any] = {"n": name}
        if version is not None:
            q += " AND version = :v"
            params["v"] = version
        q += " ORDER BY version DESC LIMIT 1"
        with self.engine.connect() as conn:
            row = conn.execute(text(q), params).mappings().first()
        if not row:
            return None
        ds = dict(row)
        ds["filter_config"] = loads(ds["filter_config"])
        return ds

    def set_dataset_report_path(self, dataset_id: str, path: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text("UPDATE datasets SET report_path = :p WHERE id = :id"), {"p": path, "id": dataset_id}
            )

    def get_dataset_by_hash(self, content_hash: str) -> dict | None:
        """Find an existing dataset with this exact content.

        Curation is deterministic, so re-running it over an unchanged corpus reproduces the same hash. That is the
        guarantee working, not an error: the caller reuses the existing version instead of writing a duplicate.
        """
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    text("SELECT * FROM datasets WHERE content_hash = :h ORDER BY version LIMIT 1"),
                    {"h": content_hash},
                )
                .mappings()
                .first()
            )
        if not row:
            return None
        ds = dict(row)
        ds["filter_config"] = loads(ds["filter_config"])
        return ds

    def list_datasets(self) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(text("SELECT * FROM datasets ORDER BY name, version")).mappings().fetchall()
        out = []
        for r in rows:
            ds = dict(r)
            ds["filter_config"] = loads(ds["filter_config"])
            out.append(ds)
        return out

    # ----------------------------------------------------------------------------------------------------------
    # training runs and adapters
    # ----------------------------------------------------------------------------------------------------------

    def insert_training_run(self, run: dict) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """INSERT INTO training_runs (id, dataset_id, base_model, method, parent_adapter_id, config,
                                                  metrics, adapter_path, status, started_at, ended_at, command)
                       VALUES (:id, :dataset_id, :base_model, :method, :parent_adapter_id, :config,
                               :metrics, :adapter_path, :status, :started_at, :ended_at, :command)"""
                ),
                {
                    **run,
                    "config": dumps(run["config"]),
                    "metrics": dumps(run.get("metrics")),
                    "parent_adapter_id": run.get("parent_adapter_id"),
                    "adapter_path": run.get("adapter_path"),
                    "ended_at": run.get("ended_at"),
                    "command": run.get("command") or invocation(),
                },
            )

    def finish_training_run(self, run_id: str, status: str, metrics: dict | None, adapter_path: str | None) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """UPDATE training_runs SET status = :s, metrics = :m, adapter_path = :p, ended_at = :e
                       WHERE id = :id"""
                ),
                {"s": status, "m": dumps(metrics), "p": adapter_path, "e": utcnow(), "id": run_id},
            )

    def get_training_run(self, run_id: str) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM training_runs WHERE id = :id"), {"id": run_id}).mappings().first()
        if not row:
            return None
        run = dict(row)
        run["config"] = loads(run["config"])
        run["metrics"] = loads(run["metrics"])
        return run

    def next_adapter_version(self, name: str) -> int:
        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT MAX(version) FROM adapters WHERE name = :n"), {"n": name}).first()
        return (row[0] or 0) + 1 if row else 1

    def insert_adapter(self, adapter: dict) -> None:
        """New adapters enter as `candidate`. Nothing is promoted on loss curves; promotion requires a paired
        eval against the current prod adapter."""
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """INSERT INTO adapters (id, training_run_id, name, version, base_model, merged, quantization,
                                             path, status, created_at, tag, parent_adapter_id)
                       VALUES (:id, :training_run_id, :name, :version, :base_model, :merged, :quantization,
                               :path, :status, :created_at, :tag, :parent_adapter_id)"""
                ),
                {
                    "merged": False,
                    "quantization": None,
                    "status": "candidate",
                    "created_at": utcnow(),
                    "tag": None,
                    "parent_adapter_id": None,
                    **adapter,
                },
            )

    def list_adapters(self) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(text("SELECT * FROM adapters ORDER BY name, version")).mappings().fetchall()
        return [dict(r) for r in rows]

    # ----------------------------------------------------------------------------------------------------------
    # eval sets
    # ----------------------------------------------------------------------------------------------------------

    def insert_eval_set(self, eval_set: dict) -> None:
        trace_ids = eval_set["trace_ids"]
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """INSERT INTO eval_sets (id, name, trace_ids, grader, frozen_at, created_at)
                       VALUES (:id, :name, :trace_ids, :grader, :frozen_at, :created_at)"""
                ),
                {
                    "id": eval_set["id"],
                    "name": eval_set["name"],
                    "trace_ids": trace_ids if self.dialect == "postgres" else dumps(trace_ids),
                    "grader": dumps(eval_set.get("grader") or {}),
                    "frozen_at": eval_set.get("frozen_at"),
                    "created_at": eval_set.get("created_at") or utcnow(),
                },
            )

    def get_eval_set(self, name: str) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM eval_sets WHERE name = :n"), {"n": name}).mappings().first()
        if not row:
            return None
        es = dict(row)
        es["trace_ids"] = list(es["trace_ids"]) if self.dialect == "postgres" else loads(es["trace_ids"])
        es["grader"] = loads(es["grader"])
        return es

    def eval_set_trace_ids(self, names: list[str] | None = None) -> set[str]:
        """Trace ids belonging to any registered eval set.

        Eval traces live in the same table as training traces, so curation must exclude them explicitly.
        Decontamination would catch them anyway -- they match themselves exactly -- but only after they have been
        counted as training candidates, which makes the report read as though the corpus were contaminated when
        it is simply the eval set being seen twice.
        """
        ids: set[str] = set()
        with self.engine.connect() as conn:
            if names:
                params: dict[str, Any] = {f"n{i}": n for i, n in enumerate(names)}
                ph = ", ".join(f":{k}" for k in params)
                rows = conn.execute(text(f"SELECT trace_ids FROM eval_sets WHERE name IN ({ph})"), params).fetchall()
            else:
                rows = conn.execute(text("SELECT trace_ids FROM eval_sets")).fetchall()
            for r in rows:
                ids.update(list(r[0]) if self.dialect == "postgres" else loads(r[0]))
        return ids

    def eval_set_task_inputs(self, names: list[str] | None = None) -> list[dict]:
        """Task inputs of every eval set, for the decontamination filter."""
        with self.engine.connect() as conn:
            if names:
                name_params: dict[str, Any] = {f"n{i}": n for i, n in enumerate(names)}
                ph = ", ".join(f":{k}" for k in name_params)
                set_rows = conn.execute(
                    text(f"SELECT trace_ids FROM eval_sets WHERE name IN ({ph})"), name_params
                ).fetchall()
            else:
                set_rows = conn.execute(text("SELECT trace_ids FROM eval_sets")).fetchall()
            ids: list[str] = []
            for r in set_rows:
                ids.extend(list(r[0]) if self.dialect == "postgres" else loads(r[0]))
            if not ids:
                return []
            out: list[dict] = []
            for i in range(0, len(ids), 500):
                chunk = ids[i : i + 500]
                id_params: dict[str, Any] = {f"i{j}": v for j, v in enumerate(chunk)}
                ph = ", ".join(f":{k}" for k in id_params)
                trace_rows = (
                    conn.execute(text(f"SELECT id, task_input FROM traces WHERE id IN ({ph})"), id_params)
                    .mappings()
                    .fetchall()
                )
                out.extend({"id": r["id"], "task_input": loads(r["task_input"])} for r in trace_rows)
        return out

    def record_round(self, round_row: dict) -> None:
        """Persist one on-policy round. Upserts, because a round is recorded on the way out whatever happened."""
        ids = round_row.get("ids") or {}
        params = {
            "id": ids.get("round_id") or f"rd_{uuid.uuid4().hex[:16]}",
            "tag": round_row.get("tag"),
            "round_idx": round_row["round_idx"],
            "start_adapter_id": round_row["start_adapter"],
            "rollout_eval_run": ids.get("rollout_eval_run"),
            "n_rollouts": round_row.get("n_rollouts"),
            "fuzzy_share": round_row.get("fuzzy_share"),
            "rft_dataset_id": ids.get("rft_dataset"),
            "dpo_dataset_id": ids.get("dpo_dataset"),
            "sft_run_id": ids.get("sft_run"),
            "dpo_run_id": ids.get("dpo_run"),
            "candidate_adapter": round_row.get("candidate_adapter"),
            "eval_run_id": ids.get("eval_run"),
            "compare": dumps(round_row.get("compare")),
            "decision": round_row.get("decision"),
            "reason": (round_row.get("reason") or "")[:1000],
            "started_at": utcnow(),
            "ended_at": utcnow(),
        }
        with self.engine.begin() as conn:
            conn.execute(text("DELETE FROM onpolicy_rounds WHERE id = :id"), {"id": params["id"]})
            conn.execute(
                text(
                    """INSERT INTO onpolicy_rounds (id, tag, round_idx, start_adapter_id, rollout_eval_run,
                                                    n_rollouts, fuzzy_share, rft_dataset_id, dpo_dataset_id,
                                                    sft_run_id, dpo_run_id, candidate_adapter, eval_run_id,
                                                    compare, decision, reason, started_at, ended_at)
                       VALUES (:id, :tag, :round_idx, :start_adapter_id, :rollout_eval_run, :n_rollouts,
                               :fuzzy_share, :rft_dataset_id, :dpo_dataset_id, :sft_run_id, :dpo_run_id,
                               :candidate_adapter, :eval_run_id, :compare, :decision, :reason, :started_at,
                               :ended_at)"""
                ),
                params,
            )

    # ----------------------------------------------------------------------------------------------------------
    # eval runs and results
    # ----------------------------------------------------------------------------------------------------------

    def start_eval_run(
        self, run_id: str, eval_set_id: str, subject: str, n_per_task: int, tag: str | None = None
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """INSERT INTO eval_runs (id, eval_set_id, subject, n_per_task, metrics, started_at, tag,
                                             command)
                       VALUES (:id, :es, :subject, :n, :metrics, :started, :tag, :command)"""
                ),
                {"id": run_id, "es": eval_set_id, "subject": subject, "n": n_per_task,
                 "metrics": dumps({}), "started": utcnow(), "tag": tag, "command": invocation()},
            )

    def write_eval_result(self, run_id: str, outcome: Any, cluster: int | None = None,
                          store_messages: bool = True) -> None:
        row = outcome.to_row()
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """INSERT INTO eval_results (id, eval_run_id, task_id, repeat_idx, success, schema_valid,
                                                 diverged, divergence, n_turns, n_tool_calls,
                                                 completion_tokens_est, latency_ms, stop_reason, grader_detail,
                                                 replay_stats, final_text, messages, cluster, escalations,
                                                 wasted_student_tokens)
                       VALUES (:id, :run, :task, :repeat, :success, :schema_valid, :diverged, :divergence,
                               :n_turns, :n_tool_calls, :tokens, :latency, :stop, :detail, :replay, :final,
                               :messages, :cluster, :escalations, :wasted)"""
                ),
                {
                    "id": f"er_{uuid.uuid4().hex[:16]}",
                    "run": run_id,
                    "task": row["task_id"],
                    "repeat": row["repeat_idx"],
                    "success": row["success"],
                    "schema_valid": row["schema_valid"],
                    "diverged": row["diverged"],
                    "divergence": dumps(row["divergence"]),
                    "n_turns": row["n_turns"],
                    "n_tool_calls": row["n_tool_calls"],
                    "tokens": row["completion_tokens_est"],
                    "latency": row["latency_ms"],
                    "stop": row["stop_reason"],
                    "detail": row["grader_detail"],
                    "replay": dumps(row["replay_stats"]),
                    "final": row["final_text"],
                    # Trajectories are large. Kept by default because hand-reading failures is the only way to
                    # find out why a number moved, but `eval run --no-store-messages` turns it off.
                    "messages": dumps(outcome.messages) if store_messages else None,
                    "cluster": cluster,
                    "escalations": row.get("escalations", 0),
                    "wasted": row.get("wasted_student_tokens", 0),
                },
            )

    def eval_results(self, run_id: str) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("SELECT * FROM eval_results WHERE eval_run_id = :r ORDER BY task_id, repeat_idx"),
                {"r": run_id},
            ).mappings().fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["success"] = as_bool(d["success"])
            d["schema_valid"] = as_bool(d["schema_valid"])
            d["diverged"] = bool(d["diverged"])
            d["divergence"] = loads(d["divergence"])
            d["replay_stats"] = loads(d["replay_stats"])
            d["messages"] = loads(d["messages"])
            out.append(d)
        return out

    def finish_eval_run(self, run_id: str, metrics: dict, per_cluster: dict | None = None) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text("UPDATE eval_runs SET metrics = :m, per_cluster = :pc, ended_at = :e WHERE id = :id"),
                {"m": dumps(metrics), "pc": dumps(per_cluster), "e": utcnow(), "id": run_id},
            )

    def get_eval_run(self, run_id: str) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM eval_runs WHERE id = :id"), {"id": run_id}).mappings().first()
        if not row:
            return None
        run = dict(row)
        run["metrics"] = loads(run["metrics"])
        run["paired"] = loads(run["paired"])
        run["per_cluster"] = loads(run["per_cluster"])
        return run

    def list_eval_runs(self, eval_set_id: str | None = None) -> list[dict]:
        q = "SELECT * FROM eval_runs"
        params: dict[str, Any] = {}
        if eval_set_id:
            q += " WHERE eval_set_id = :es"
            params["es"] = eval_set_id
        q += " ORDER BY started_at DESC"
        with self.engine.connect() as conn:
            rows = conn.execute(text(q), params).mappings().fetchall()
        out = []
        for r in rows:
            run = dict(r)
            run["metrics"] = loads(run["metrics"])
            run["per_cluster"] = loads(run["per_cluster"])
            out.append(run)
        return out

    def find_eval_run(self, ref: str) -> dict | None:
        """Resolve a run by id, by id prefix, or as `latest:<subject>`."""
        if ref.startswith("latest:"):
            subject = ref.split(":", 1)[1]
            with self.engine.connect() as conn:
                row = conn.execute(
                    text("SELECT id FROM eval_runs WHERE subject = :s ORDER BY started_at DESC LIMIT 1"),
                    {"s": subject},
                ).first()
            return self.get_eval_run(row[0]) if row else None
        exact = self.get_eval_run(ref)
        if exact:
            return exact
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("SELECT id FROM eval_runs WHERE id LIKE :p"), {"p": f"{ref}%"}
            ).fetchall()
        if len(rows) == 1:
            return self.get_eval_run(rows[0][0])
        return None

    # ----------------------------------------------------------------------------------------------------------
    # embeddings
    # ----------------------------------------------------------------------------------------------------------

    def upsert_embeddings(self, model: str, vectors: dict[str, list[float]]) -> None:
        if not vectors:
            return
        dim = len(next(iter(vectors.values())))
        with self.engine.begin() as conn:
            for trace_id, vec in vectors.items():
                conn.execute(text("DELETE FROM trace_embeddings WHERE trace_id = :t"), {"t": trace_id})
                conn.execute(
                    text(
                        "INSERT INTO trace_embeddings (trace_id, model, dim, embedding) VALUES (:t, :m, :d, :e)"
                        if self.dialect == "sqlite"
                        else "INSERT INTO trace_embeddings (trace_id, model, embedding) VALUES (:t, :m, :e)"
                    ),
                    {"t": trace_id, "m": model, "d": dim, "e": dumps(vec) if self.dialect == "sqlite" else vec},
                )

    def get_embeddings(self) -> dict[str, list[float]]:
        with self.engine.connect() as conn:
            rows = conn.execute(text("SELECT trace_id, embedding FROM trace_embeddings")).fetchall()
        return {r[0]: loads(r[1]) for r in rows}

    def close(self) -> None:
        self.engine.dispose()

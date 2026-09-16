"""Registry: the durable record of traces, datasets, runs, adapters, and calibrations.

SQLAlchemy Core over a URL, so the same code serves `sqlite:///.agentdistill/registry.db` locally and a Postgres
URL for a team. Dialect differences (JSONB vs TEXT, arrays vs JSON text, vectors) are confined to the JSON codec
here and to the two migration files; nothing above this layer knows which backend it is talking to.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event, text

MIGRATIONS = Path(__file__).resolve().parent / "migrations"
SCHEMA_VERSION = 1


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
        """Apply migrations that have not been applied. Idempotent: every statement is IF NOT EXISTS."""
        sql_path = MIGRATIONS / self.dialect / "001_init.sql"
        with self.engine.begin() as conn:
            for stmt in split_statements(sql_path.read_text()):
                if stmt.upper().startswith("PRAGMA") and self.dialect != "sqlite":
                    continue
                conn.execute(text(stmt))
            if not self._has_version(conn):
                conn.execute(
                    text("INSERT INTO schema_version (version, applied_at) VALUES (:v, :t)"),
                    {"v": SCHEMA_VERSION, "t": utcnow()},
                )

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
        q = "SELECT * FROM datasets WHERE name = :n"
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
                                                  metrics, adapter_path, status, started_at, ended_at)
                       VALUES (:id, :dataset_id, :base_model, :method, :parent_adapter_id, :config,
                               :metrics, :adapter_path, :status, :started_at, :ended_at)"""
                ),
                {
                    **run,
                    "config": dumps(run["config"]),
                    "metrics": dumps(run.get("metrics")),
                    "parent_adapter_id": run.get("parent_adapter_id"),
                    "adapter_path": run.get("adapter_path"),
                    "ended_at": run.get("ended_at"),
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
                                             path, status, created_at)
                       VALUES (:id, :training_run_id, :name, :version, :base_model, :merged, :quantization,
                               :path, :status, :created_at)"""
                ),
                {
                    "merged": False,
                    "quantization": None,
                    "status": "candidate",
                    "created_at": utcnow(),
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

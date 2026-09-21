"""Retiring rows that were built before a fix.

Some artifacts are not bad, they are *invalid*, and no amount of re-evaluating them changes that. The case this
was written for: until the render-boundary fix, tool-call arguments reached the chat template as the OpenAI wire
format's JSON string rather than as an object, so every dataset built before it rendered

    "arguments": "{\\"customer_id\\": \\"c_9\\"}"

and a student trained on that text emits tool calls the serving stack's parser silently drops. The dataset's
hash, its sample count and its curation report are all still true; the text inside it is unusable.

Two things must not happen to such a row. It must not be deleted -- the registry is the record of what was run,
and a report that cannot explain where a number came from is worse than a report with a gap. And it must not
stay selectable -- `dataset latest` feeding the GPU day an invalid dataset is exactly the failure this repo
exists to prevent. So the row is *marked*: `retired_at` and `retired_reason` are written, the selectors skip it,
and an adapter additionally leaves the promotion path and gets an `adapter_events` row carrying the reason.

Retirement is a judgement about provenance, so the cutoff is expressed the way provenance is: a commit, or the
time one landed. `--built-before 87e8653` resolves to that commit's committer date and retires everything older.
"""

from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

from agentdistill.registry.base import dumps, utcnow

#: Who the adapter event says did it. Distinct from lifecycle's "cli" so `adapter lineage` can tell a promotion
#: decision from a bulk invalidation.
ACTOR = "registry retire"

#: Tables that carry the retirement columns, and the human name for each.
TABLES = ("datasets", "adapters")


class RetireError(ValueError):
    """The retirement could not be attempted at all: a cutoff nothing could resolve, an empty reason, or a
    registry whose schema predates migration 008."""


@dataclass
class Row:
    """One considered row, whatever was decided about it."""

    table: str
    id: str
    name: str
    version: int | None
    created_at: str | None
    status: str | None = None
    note: str = ""

    def __str__(self) -> str:
        version = f" v{self.version}" if self.version is not None else ""
        note = f" ({self.note})" if self.note else ""
        return f"{self.table[:-1]} {self.id} [{self.name}{version}] built {self.created_at}{note}"


@dataclass
class RetireResult:
    cutoff: str
    cutoff_source: str
    reason: str
    dry_run: bool
    retired: list[Row] = field(default_factory=list)
    already_retired: list[Row] = field(default_factory=list)
    kept: list[Row] = field(default_factory=list)

    @property
    def n_rows_seen(self) -> int:
        return len(self.retired) + len(self.already_retired) + len(self.kept)

    @property
    def outcome(self) -> Any:
        """What `stage_guard` needs: a row written, a cited skip, or a hole.

        "Nothing matched" is a skip only when there was something to match against, and it cites the newest row
        it declined to retire by id -- so the log says which row it looked at and why it survived, rather than
        the bare absence that this repo treats as a bug. An empty registry cites nothing, so it is a hole.
        """
        from agentdistill.cli_stage import StageOutcome

        if self.retired:
            verb = "would retire" if self.dry_run else "retired"
            n_ds = sum(1 for r in self.retired if r.table == "datasets")
            n_ad = len(self.retired) - n_ds
            return StageOutcome(
                wrote=not self.dry_run,
                detail=f"{verb} {n_ds} dataset(s) and {n_ad} adapter(s) built before {self.cutoff}",
                skipped_reason=(
                    f"--dry-run: {len(self.retired)} row(s) would be retired, nothing written" if self.dry_run
                    else None
                ),
            )
        if self.kept or self.already_retired:
            newest = max(
                self.kept + self.already_retired, key=lambda r: (r.created_at or "", r.id)
            )
            return StageOutcome(
                wrote=False,
                detail=f"nothing to retire before {self.cutoff}",
                skipped_reason=(
                    f"nothing built before {self.cutoff}; the newest row {newest.id} was built "
                    f"{newest.created_at}"
                    if newest in self.kept
                    else f"every matching row was already retired; {newest.id} carries a retired_at"
                ),
            )
        return StageOutcome(
            wrote=False,
            detail=f"the registry holds no {' or '.join(TABLES)} at all, so there is nothing a cutoff could select",
        )

    def render(self) -> str:
        head = f"cutoff {self.cutoff} (from {self.cutoff_source})"
        lines = [head + (" -- DRY RUN, nothing written" if self.dry_run else "")]
        for label, rows in (
            ("would retire" if self.dry_run else "retired", self.retired),
            ("already retired", self.already_retired),
            ("kept (built at or after the cutoff)", self.kept),
        ):
            lines.append(f"{label}: {len(rows)}")
            lines += [f"  - {r}" for r in rows]
        return "\n".join(lines)


# --------------------------------------------------------------------------------------------------------------
# the cutoff
# --------------------------------------------------------------------------------------------------------------


def resolve_cutoff(built_before: str, repo_root: Path | str | None = None) -> tuple[datetime, str]:
    """`--built-before` as an aware UTC datetime, plus where it came from.

    An ISO timestamp is taken literally; anything else is handed to git as a revision and resolved to that
    commit's *committer* date, which is when the fix actually entered the branch being retired against. A naive
    timestamp is read as UTC, because every `created_at` in the registry is written in UTC.

    ISO is tried first, so an all-numeric 8-character git revision would have to be spelled in full or given as
    a date. That ambiguity is worth an unusual spelling; reading `20260920` as a revision would not be.
    """
    raw = built_before.strip()
    if not raw:
        raise RetireError("--built-before is empty; give an ISO time or a commit")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        pass
    else:
        return (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).astimezone(UTC), "iso time"

    cmd = ["git"] + (["-C", str(repo_root)] if repo_root else []) + ["show", "-s", "--format=%cI", raw]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        raise RetireError(f"{raw!r} is not an ISO time and git could not be run to resolve it: {e}") from e
    if proc.returncode != 0:
        raise RetireError(
            f"{raw!r} is neither an ISO time nor a revision this repository knows: "
            f"{(proc.stderr or '').strip() or 'git show failed'}"
        )
    stamp = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        resolved = datetime.fromisoformat(stamp)
    except ValueError as e:
        raise RetireError(f"git resolved {raw!r} to {stamp!r}, which is not a timestamp") from e
    return resolved.astimezone(UTC), f"commit {raw} committed {stamp}"


def _built_at(created_at: Any) -> datetime | None:
    if not isinstance(created_at, str) or not created_at.strip():
        return None
    try:
        parsed = datetime.fromisoformat(created_at.strip())
    except ValueError:
        return None
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).astimezone(UTC)


# --------------------------------------------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------------------------------------------


def has_retirement_columns(registry: Any) -> bool:
    """Whether migration 008 has been applied. Probed rather than assumed, so the failure names the migration
    instead of surfacing as a column error from three frames down."""
    with registry.engine.connect() as conn:
        for table in TABLES:
            try:
                conn.execute(text(f"SELECT retired_at, retired_reason FROM {table} LIMIT 0"))
            except Exception:
                return False
    return True


def _all_rows(registry: Any, table: str) -> list[dict]:
    with registry.engine.connect() as conn:
        return [dict(r) for r in conn.execute(text(f"SELECT * FROM {table}")).mappings()]


def retire(
    registry: Any,
    built_before: str,
    reason: str,
    dry_run: bool = False,
    repo_root: Path | str | None = None,
) -> RetireResult:
    """Mark every dataset and adapter built before `built_before` as retired.

    A row whose `created_at` cannot be parsed is retired too, and says so in its note. A row nobody can date is a
    row nobody can vouch for, and retirement is reversible where training on an invalid dataset is not.
    """
    if not (reason or "").strip():
        raise RetireError(
            "--reason is required and may not be empty: a retirement nobody can explain is one nobody can undo"
        )
    if not has_retirement_columns(registry):
        raise RetireError(
            "this registry has no retired_at/retired_reason columns; apply migration 008_retire.sql "
            "(add it to MIGRATION_FILES in agentdistill/registry/base.py and reopen the registry)"
        )

    cutoff, source = resolve_cutoff(built_before, repo_root)
    result = RetireResult(cutoff=cutoff.isoformat(), cutoff_source=source, reason=reason.strip(), dry_run=dry_run)
    now = utcnow()

    for table in TABLES:
        for raw in _all_rows(registry, table):
            built = _built_at(raw.get("created_at"))
            row = Row(
                table=table,
                id=raw["id"],
                name=raw.get("name") or "",
                version=raw.get("version"),
                created_at=raw.get("created_at"),
                status=raw.get("status"),
            )
            if raw.get("retired_at"):
                row.note = f"retired {raw['retired_at']}: {raw.get('retired_reason') or 'no reason recorded'}"
                result.already_retired.append(row)
                continue
            if built is not None and built >= cutoff:
                result.kept.append(row)
                continue
            if built is None:
                row.note = "created_at is not a timestamp, so the row cannot be dated"
            result.retired.append(row)

    if not dry_run and result.retired:
        _write(registry, result, now)
    return result


def _write(registry: Any, result: RetireResult, now: str) -> None:
    """One transaction: either every row is marked and every event written, or none is."""
    with registry.engine.begin() as conn:
        for row in result.retired:
            if row.table == "datasets":
                conn.execute(
                    text("UPDATE datasets SET retired_at = :t, retired_reason = :r WHERE id = :i"),
                    {"t": now, "r": result.reason, "i": row.id},
                )
                continue
            # An invalid adapter must also leave the promotion path, or `adapter promote` would still consider
            # it. `status` and `retired_at` mean different things and both are set here on purpose.
            conn.execute(
                text(
                    "UPDATE adapters SET retired_at = :t, retired_reason = :r, status = 'retired' WHERE id = :i"
                ),
                {"t": now, "r": result.reason, "i": row.id},
            )
            conn.execute(
                text(
                    """INSERT INTO adapter_events (id, adapter_id, from_status, to_status, checks, actor,
                                                   created_at)
                       VALUES (:id, :a, :f, 'retired', :c, :actor, :created)"""
                ),
                {
                    "id": f"ae_{uuid.uuid4().hex[:16]}",
                    "a": row.id,
                    "f": row.status,
                    "c": dumps(
                        {
                            "reason": result.reason,
                            "built_before": result.cutoff,
                            "cutoff_source": result.cutoff_source,
                            "built_at": row.created_at,
                            "note": row.note or None,
                        }
                    ),
                    "actor": ACTOR,
                    "created": now,
                },
            )


def is_retired(row: Any) -> bool:
    """Whether a registry row has been retired.

    One predicate, used by every selector, and tolerant of a row from a registry that predates migration 008:
    the column is simply absent there, which reads as "not retired" -- correct, because nothing could have
    retired it.
    """
    if row is None:
        return False
    get = row.get if hasattr(row, "get") else (lambda k, d=None: getattr(row, k, d))
    return bool(get("retired_at"))


def drop_retired(rows: list[dict]) -> list[dict]:
    return [r for r in rows if not is_retired(r)]

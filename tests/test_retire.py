"""Retiring rows built before a fix, and the selectors that must then refuse them.

A dataset built before the render-boundary fix is invalid, not merely old: it renders tool-call arguments as a
quoted string, and anything trained on it emits calls the serving stack drops. It stays in the registry, because
the registry is the record of what was run. It must not stay selectable, because `dataset latest` is what the
GPU-day script substitutes into `train sft`.

So there are two halves here, and the second is the one that matters on the GPU box: `retire` marks the rows,
and *every* selector skips a marked row. A selector that forgets is not a cosmetic bug -- it is a GPU day spent
training on text nobody can serve.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import text

from agentdistill.cli_stage import StageEmpty, stage_guard
from agentdistill.registry.base import MIGRATIONS
from agentdistill.registry.lifecycle import events
from agentdistill.registry.retire import (
    RetireError,
    has_retirement_columns,
    is_retired,
    resolve_cutoff,
    retire,
)
from agentdistill.registry.select import (
    NoMatch,
    best_adapter,
    canary_adapter,
    latest_adapter,
    latest_dataset,
    prod_adapter,
)
from agentdistill.report.registry_views import latest_quantized

ROOT = Path(__file__).resolve().parents[1]

CUTOFF = "2026-09-20T07:12:39+00:00"  # commit 87e8653, where the render-boundary fix landed
BEFORE = "2026-09-19T00:00:00+00:00"
AFTER = "2026-09-21T00:00:00+00:00"
#: Strictly later than every fixture row: `--built-before` is exclusive, as its name says, so retiring
#: *everything* needs a cutoff past the newest row rather than equal to it.
LATER = "2026-09-22T00:00:00+00:00"
REASON = "built before the render-boundary fix (87e8653): tool-call arguments rendered as a JSON string"


def _apply_008(registry) -> None:
    """Apply `008_retire.sql` if the registry does not already have its columns.

    WP3 owns the migration file; `MIGRATION_FILES` and `SCHEMA_VERSION` in `registry/base.py` are the lead's to
    change at merge time. Until that lands, `open_registry` stops at 007 and these tests would have nothing to
    write to. Once it lands this is a no-op, because the columns will already be there -- so the fixture does
    not have to be removed in the same commit as the base.py change.
    """
    if has_retirement_columns(registry):
        return
    from agentdistill.registry.base import split_statements

    sql = (MIGRATIONS / "sqlite" / "008_retire.sql").read_text()
    with registry.engine.begin() as conn:
        for stmt in split_statements(sql):
            conn.execute(text(stmt))


@pytest.fixture
def reg(registry):
    _apply_008(registry)
    return registry


def add_dataset(reg, ds_id: str, created: str, name: str = "support", version: int = 1) -> None:
    reg.insert_dataset({
        "id": ds_id, "name": name, "version": version, "kind": "sft", "filter_config": {},
        "n_samples": 10, "n_tokens": 100, "content_hash": ds_id, "path": f"/tmp/{ds_id}",
        "created_at": created,
    })


def add_adapter(
    reg, ad_id: str, created: str, *, version: int, status: str = "candidate", tag: str | None = "gpu-day",
    quantization: str | None = None, parent: str | None = None, success: float | None = None,
) -> None:
    reg.insert_adapter({
        "id": ad_id, "training_run_id": "tr1", "name": "support", "version": version, "base_model": "m",
        "path": f"/tmp/{ad_id}", "tag": tag, "quantization": quantization, "status": status,
        "parent_adapter_id": parent, "created_at": created,
    })
    if success is not None:
        run_id = f"ev_{ad_id}"
        reg.start_eval_run(run_id, "es_hold", ad_id, 5, tag=tag)
        reg.finish_eval_run(run_id, {"success": success, "schema_valid": 1.0})


@pytest.fixture
def seeded(reg):
    """Two datasets and four adapters, straddling the cutoff.

    The old adapter scores *better* than the new one on purpose: `best` ranks by measured success, and an eval
    of an invalid adapter is a real number about text the serving stack would have dropped. It must lose anyway.
    """
    # The dataset first: training_runs.dataset_id is a foreign key, and SQLite enforces it here.
    add_dataset(reg, "ds_old", BEFORE, version=1)
    add_dataset(reg, "ds_new", AFTER, version=2)
    reg.insert_training_run({
        "id": "tr1", "dataset_id": "ds_old", "base_model": "m", "method": "sft", "config": {},
        "status": "succeeded", "started_at": BEFORE,
    })
    reg.insert_eval_set({"id": "es_hold", "name": "hold", "trace_ids": [], "grader": {}})

    add_adapter(reg, "ad_old", BEFORE, version=1, success=0.90)
    add_adapter(reg, "ad_new", AFTER, version=2, success=0.60)
    add_adapter(reg, "ad_old_q", BEFORE, version=3, quantization="awq", parent="ad_new", tag=None)
    add_adapter(reg, "ad_new_q", AFTER, version=4, quantization="awq", parent="ad_new", tag=None)
    return reg


def run(reg, **kwargs):
    return retire(reg, built_before=kwargs.pop("built_before", CUTOFF), reason=kwargs.pop("reason", REASON),
                  **kwargs)


def row(reg, table: str, row_id: str) -> dict:
    with reg.engine.connect() as conn:
        return dict(conn.execute(text(f"SELECT * FROM {table} WHERE id = :i"), {"i": row_id}).mappings().one())


# --------------------------------------------------------------------------------------------------------------
# the cutoff
# --------------------------------------------------------------------------------------------------------------


def test_an_iso_time_is_taken_literally():
    when, source = resolve_cutoff(CUTOFF)
    assert when.isoformat() == CUTOFF
    assert source == "iso time"


def test_a_naive_time_is_read_as_utc():
    """Every `created_at` in the registry is UTC, so a cutoff without an offset has to be too -- reading it as
    local time would move the cutoff by hours depending on who ran the command."""
    when, _ = resolve_cutoff("2026-09-20T07:12:39")
    assert when.isoformat() == CUTOFF


def test_a_bare_date_is_a_cutoff_at_midnight():
    assert resolve_cutoff("2026-09-20")[0].isoformat() == "2026-09-20T00:00:00+00:00"


def test_a_commit_resolves_to_its_committer_date():
    """`--built-before <commit>` is the spelling that matters: retirement is a judgement about provenance, and
    provenance is recorded as a commit."""
    head = subprocess.run(
        ["git", "-C", str(ROOT), "show", "-s", "--format=%cI", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    when, source = resolve_cutoff("HEAD", repo_root=ROOT)
    # Compared as instants, not as strings: git prints the committer's own offset and the cutoff is normalized
    # to UTC, because every `created_at` in the registry is UTC.
    assert when == datetime.fromisoformat(head).astimezone(UTC)
    assert when.utcoffset() == timedelta(0)
    assert source.startswith("commit HEAD")


def test_a_cutoff_that_is_neither_a_time_nor_a_revision_is_refused():
    with pytest.raises(RetireError, match="neither an ISO time nor a revision"):
        resolve_cutoff("last tuesday", repo_root=ROOT)


# --------------------------------------------------------------------------------------------------------------
# what gets marked
# --------------------------------------------------------------------------------------------------------------


def test_an_empty_reason_is_refused(seeded):
    """A retirement nobody can explain is one nobody can undo."""
    with pytest.raises(RetireError, match="--reason is required"):
        retire(seeded, built_before=CUTOFF, reason="   ")


def test_a_registry_without_the_migration_names_the_migration(registry):
    """Not "no such column: retired_at" from three frames down."""
    if has_retirement_columns(registry):
        pytest.skip("migration 008 is already in MIGRATION_FILES, so this registry cannot be old")
    with pytest.raises(RetireError, match=r"008_retire\.sql"):
        retire(registry, built_before=CUTOFF, reason=REASON)


def test_rows_built_before_the_cutoff_are_marked(seeded):
    result = run(seeded)
    assert {r.id for r in result.retired} == {"ds_old", "ad_old", "ad_old_q"}
    assert row(seeded, "datasets", "ds_old")["retired_reason"] == REASON
    assert row(seeded, "adapters", "ad_old")["retired_at"]


def test_rows_built_at_or_after_the_cutoff_are_untouched(seeded):
    result = run(seeded)
    assert {r.id for r in result.kept} == {"ds_new", "ad_new", "ad_new_q"}
    assert row(seeded, "datasets", "ds_new")["retired_at"] is None
    assert row(seeded, "adapters", "ad_new")["status"] == "candidate"


def test_a_row_built_exactly_at_the_cutoff_survives(reg):
    """`--built-before` is exclusive: the commit that fixed it produced valid rows."""
    add_dataset(reg, "ds_at", CUTOFF)
    assert run(reg).retired == []


def test_a_retired_adapter_also_leaves_the_promotion_path(seeded):
    """`retired_at` and `status` are different facts, but an invalid adapter must not still be promotable."""
    run(seeded)
    assert row(seeded, "adapters", "ad_old")["status"] == "retired"


def test_the_reason_is_recorded_as_an_adapter_event(seeded):
    run(seeded)
    recorded = events(seeded, "ad_old")
    assert [e["to_status"] for e in recorded] == ["retired"]
    event = recorded[0]
    assert event["from_status"] == "candidate"
    assert event["actor"] == "registry retire"
    assert event["checks"]["reason"] == REASON
    assert event["checks"]["built_before"] == CUTOFF
    assert event["checks"]["built_at"] == BEFORE


def test_no_adapter_event_is_written_for_a_dataset(seeded):
    """Datasets have no lifecycle; the mark is the whole record for one."""
    run(seeded)
    with seeded.engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM adapter_events")).scalar() == 2


def test_a_dry_run_writes_nothing(seeded):
    result = run(seeded, dry_run=True)
    assert {r.id for r in result.retired} == {"ds_old", "ad_old", "ad_old_q"}
    assert row(seeded, "datasets", "ds_old")["retired_at"] is None
    assert row(seeded, "adapters", "ad_old")["status"] == "candidate"


def test_an_undated_row_is_retired_and_says_why(reg):
    """A row nobody can date is a row nobody can vouch for, and retirement is reversible where training on an
    invalid dataset is not."""
    add_dataset(reg, "ds_undated", BEFORE)
    with reg.engine.begin() as conn:
        conn.execute(text("UPDATE datasets SET created_at = 'sometime' WHERE id = 'ds_undated'"))
    retired = run(reg).retired
    assert [r.id for r in retired] == ["ds_undated"]
    assert "cannot be dated" in retired[0].note


def test_running_it_twice_marks_nothing_further(seeded):
    first = run(seeded)
    stamp = row(seeded, "datasets", "ds_old")["retired_at"]
    second = run(seeded)
    assert second.retired == []
    assert {r.id for r in second.already_retired} == {r.id for r in first.retired}
    assert row(seeded, "datasets", "ds_old")["retired_at"] == stamp, "a rerun rewrote the retirement date"


# --------------------------------------------------------------------------------------------------------------
# the stage outcome
# --------------------------------------------------------------------------------------------------------------


def test_retiring_rows_is_a_stage_that_wrote(seeded, capsys):
    assert stage_guard("registry_retire", lambda: run(seeded).outcome) == 0
    assert "ok:" in capsys.readouterr().out


def test_nothing_to_retire_is_a_skip_that_cites_the_row_it_declined(seeded, capsys):
    """A skip must name a row and a reason. "Nothing matched" on its own is the silence this repo exits 3 on."""
    outcome = run(seeded, built_before=BEFORE).outcome
    assert stage_guard("registry_retire", lambda: outcome, allow_skip=True) == 0
    printed = capsys.readouterr().out
    assert "SKIPPED" in printed
    assert "ds_new" in printed or "ad_new" in printed


def test_an_empty_registry_is_a_hole_not_a_skip(reg):
    """There is no row to cite, so there is nothing to justify the silence with."""
    with pytest.raises(StageEmpty, match="nothing a cutoff could select"):
        stage_guard("registry_retire", lambda: run(reg).outcome, allow_skip=True)


def test_a_dry_run_is_a_skip_rather_than_a_write(seeded):
    outcome = run(seeded, dry_run=True).outcome
    assert not outcome.wrote
    assert "dry-run" in outcome.skipped_reason


# --------------------------------------------------------------------------------------------------------------
# no selector may return a retired row
# --------------------------------------------------------------------------------------------------------------


def test_dataset_latest_skips_a_retired_dataset(seeded):
    run(seeded)
    assert latest_dataset(seeded, name="support")["id"] == "ds_new"


def test_dataset_latest_refuses_when_every_candidate_is_retired(seeded):
    """A blank substitution costs the session; `NoMatch` costs a rerun."""
    run(seeded, built_before=LATER)
    with pytest.raises(NoMatch):
        latest_dataset(seeded, name="support")


def test_adapter_best_skips_a_retired_adapter_even_though_it_scored_higher(seeded):
    assert best_adapter(seeded, tag="gpu-day")["id"] == "ad_old", "the fixture's premise is gone"
    run(seeded)
    assert best_adapter(seeded, tag="gpu-day")["id"] == "ad_new"


def test_adapter_best_refuses_when_every_candidate_is_retired(seeded):
    run(seeded, built_before=LATER)
    with pytest.raises(NoMatch):
        best_adapter(seeded, tag="gpu-day")


def test_adapter_latest_skips_a_retired_adapter(seeded):
    run(seeded)
    assert latest_adapter(seeded, tag="gpu-day")["id"] == "ad_new"


def test_the_prod_selector_skips_a_retired_adapter(reg):
    """The gateway asks this what to serve. A retired row here is an invalid model on live traffic."""
    add_dataset(reg, "ds_old", BEFORE)
    reg.insert_training_run({"id": "tr1", "dataset_id": "ds_old", "base_model": "m", "method": "sft",
                             "config": {}, "status": "succeeded", "started_at": BEFORE})
    add_adapter(reg, "ad_prod", BEFORE, version=1, status="prod", tag=None)
    assert prod_adapter(reg)["id"] == "ad_prod"
    run(reg)
    assert prod_adapter(reg) is None


def test_the_canary_selector_skips_a_retired_adapter(reg):
    add_dataset(reg, "ds_old", BEFORE)
    reg.insert_training_run({"id": "tr1", "dataset_id": "ds_old", "base_model": "m", "method": "sft",
                             "config": {}, "status": "succeeded", "started_at": BEFORE})
    add_adapter(reg, "ad_can", BEFORE, version=1, status="canary", tag=None)
    assert canary_adapter(reg)["id"] == "ad_can"
    run(reg)
    assert canary_adapter(reg) is None


def test_the_report_quantized_selector_skips_a_retired_artifact(seeded):
    """Both quantized rows hang off `ad_new`; the newer one is valid, the older is not."""
    assert latest_quantized(seeded, "ad_new")["id"] == "ad_new_q"
    run(seeded, built_before=LATER)
    assert latest_quantized(seeded, "ad_new") is None


def test_the_report_quantized_selector_falls_back_to_an_older_valid_artifact(seeded):
    """Newest-first with a skip, not `LIMIT 1` with a filter: a retired newest must not hide a valid older."""
    with seeded.engine.begin() as conn:
        conn.execute(text("UPDATE adapters SET retired_at = :t, retired_reason = :r WHERE id = 'ad_new_q'"),
                     {"t": AFTER, "r": REASON})
    assert latest_quantized(seeded, "ad_new")["id"] == "ad_old_q"


# --------------------------------------------------------------------------------------------------------------
# the predicate the selectors share
# --------------------------------------------------------------------------------------------------------------


def test_a_row_from_a_registry_without_the_columns_reads_as_not_retired():
    """Nothing could have retired it, so the absent column is not missing data."""
    assert not is_retired({"id": "ds_1", "created_at": BEFORE})
    assert not is_retired(None)


def test_an_empty_retired_at_is_not_a_retirement():
    assert not is_retired({"retired_at": None})
    assert not is_retired({"retired_at": ""})
    assert is_retired({"retired_at": AFTER})


def test_the_result_renders_every_id_it_touched(seeded):
    """The operator has to be able to read back what a bulk invalidation did."""
    printed = run(seeded).render()
    for row_id in ("ds_old", "ad_old", "ad_old_q", "ds_new", "ad_new"):
        assert row_id in printed
    assert json.dumps(REASON)[1:-1].split(":")[0] in printed or CUTOFF in printed

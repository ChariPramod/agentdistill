"""Two clean runs select the same eval sets and build the same dataset.

The clean rehearsal deletes everything and rebuilds it. These tests are the invariants that make two such runs
comparable: the generated eval sets are a pure function of their source and salt, and a dataset's content hash
depends on what it contains, not on the registry it was built into.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agentdistill.evalsets.generate import eval_set_hash, grader_of, row_task_id, select_rows, stable_task_ids

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "support_agent"


def _scenarios(n: int = 30) -> list[dict]:
    return [{"scenario": f"s{i % 5}", "instance": str(1000 + i), "task_id": f"s{i % 5}-{1000 + i}"} for i in range(n)]


def test_two_runs_select_the_same_tasks_and_hash():
    grader = {"type": "predicate"}
    a = stable_task_ids(_scenarios(), 10, "support-holdout-tiny")
    b = stable_task_ids(_scenarios(), 10, "support-holdout-tiny")
    assert a == b
    assert len(a) == 10
    assert eval_set_hash(a, grader) == eval_set_hash(b, grader)


def test_ranking_does_not_depend_on_input_order():
    rows = _scenarios()
    shuffled = rows[::-1][5:] + rows[::-1][:5]
    assert stable_task_ids(rows, 10, "salt") == stable_task_ids(shuffled, 10, "salt")


def test_salt_changes_the_selection():
    # Different sets drawn from one pool must not pick the same top-ranked tasks.
    assert stable_task_ids(_scenarios(), 10, "a") != stable_task_ids(_scenarios(), 10, "b")


def test_hash_does_not_depend_on_task_order():
    ids = stable_task_ids(_scenarios(), 10, "salt")
    assert eval_set_hash(ids, {"type": "predicate"}) == eval_set_hash(list(reversed(ids)), {"type": "predicate"})
    assert eval_set_hash(ids, {"type": "predicate"}) != eval_set_hash(ids, {"type": "llm"})


def test_select_rows_falls_back_to_ids_and_ignores_duplicates():
    rows = [{"id": f"r{i}"} for i in range(6)] + [{"id": "r0"}]
    picked = select_rows(rows, 4, "salt")
    assert len({row_task_id(r) for r in picked}) == 4
    assert [row_task_id(r) for r in picked] == [row_task_id(r) for r in select_rows(list(reversed(rows)), 4, "salt")]


def test_real_calibration_pool_holds_twenty_unshared_tasks():
    """Plan 3.2: `support-calib-tiny` is 20 tasks, drawn from a pool that is neither scored on nor trained on."""
    def ids(name: str) -> set[str]:
        return {row_task_id(json.loads(line)) for line in (EX / name).read_text().splitlines() if line.strip()}

    calib = ids("eval-calib.jsonl")
    assert len(calib) >= 20
    assert not calib & ids("eval-holdout.jsonl")
    assert not calib & ids("eval-unseen.jsonl")
    assert not calib & ids("traces-train.jsonl")


def _generate(src: Path, dst: Path, n: int, salt: str) -> str:
    out = subprocess.run(
        [sys.executable, "-m", "agentdistill.evalsets.generate", "--src", str(src), "--dst", str(dst),
         "--n", str(n), "--salt", salt],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    return out.stdout


@pytest.mark.parametrize("name,n", [("holdout", 10), ("unseen", 10), ("calib", 20)])
def test_cli_is_byte_identical_across_runs(tmp_path, name, n):
    src = EX / f"eval-{name}.jsonl"
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    out_a = _generate(src, first, n, f"support-{name}-tiny")
    out_b = _generate(src, second, n, f"support-{name}-tiny")
    assert first.read_bytes() == second.read_bytes()
    assert out_a.split("hash ")[1] == out_b.split("hash ")[1]
    rows = [json.loads(line) for line in first.read_text().splitlines()]
    assert len({row_task_id(r) for r in rows}) == n
    assert f"{n} tasks, hash {eval_set_hash([row_task_id(r) for r in rows], grader_of(rows))}" in out_a


def test_cli_does_not_depend_on_source_order(tmp_path):
    src = EX / "eval-holdout.jsonl"
    lines = [line for line in src.read_text().splitlines() if line.strip()]
    reordered = tmp_path / "reordered.jsonl"
    reordered.write_text("\n".join(reversed(lines)) + "\n")
    _generate(src, tmp_path / "a.jsonl", 10, "s")
    _generate(reordered, tmp_path / "b.jsonl", 10, "s")
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()


def test_checked_in_tiny_sets_match_what_setup_generates(tmp_path):
    """`tiny_setup.sh` rewrites these files every run; if they drift, every rehearsal dirties the tree."""
    for name, n in (("holdout", 10), ("unseen", 10), ("calib", 20)):
        _generate(EX / f"eval-{name}.jsonl", tmp_path / f"{name}.jsonl", n, f"support-{name}-tiny")
        assert (tmp_path / f"{name}.jsonl").read_bytes() == (EX / f"eval-{name}-tiny.jsonl").read_bytes(), name


def test_dataset_content_hash_is_stable_across_clean_builds(tmp_path, tokenizer):
    """Two fresh registries, the same traces: dataset rows get new ids, the content hash does not move."""
    from agentdistill.config import ProjectConfig
    from agentdistill.data.dataset import build_dataset
    from agentdistill.registry import open_registry
    from tests.conftest import make_trace

    traces = [
        make_trace(
            f"t{i}",
            task=f"Where is order {i} for my account, I have been waiting a while now",
            closing=" ".join(f"Point {j} of case {i} is confirmed as {i * 13 + j}" for j in range(6)),
        )
        for i in range(5)
    ]

    built = []
    for run in ("first", "second"):
        root = tmp_path / run
        # The project directory exists in any real run (the config lives there). The registry URL below is
        # absolute, and `Registry` only creates the parent of a *relative* SQLite path.
        root.mkdir()
        cfg = ProjectConfig(
            name="clean-run",
            registry=f"sqlite:///{root}/registry.db",
            artifacts=str(root / "artifacts"),
            reports=str(root / "reports"),
        )
        cfg.source_path = root / "project.yaml"
        reg = open_registry(cfg.registry, root=cfg.root)
        try:
            reg.insert_traces([dict(t) for t in traces])
            built.append(build_dataset(traces, cfg, name="demo", version=1, registry=reg, tokenizer=tokenizer))
        finally:
            reg.close()

    assert built[0].artifact.content_hash == built[1].artifact.content_hash

"""project.yaml validation. A typo must fail loudly, not silently disable a filter."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from agentdistill.config import FILTER_ORDER, ProjectConfig, parse_duration


def test_minimal_config_has_working_defaults():
    cfg = ProjectConfig(name="p")
    assert cfg.curate.ordered_filters()[0] == "outcome"
    assert cfg.dataset.max_seq_len == 8192
    assert cfg.train is None


def test_unknown_key_is_rejected():
    with pytest.raises(ValidationError, match=r"curatte|extra"):
        ProjectConfig.model_validate({"name": "p", "curatte": {}})


def test_unknown_filter_is_rejected():
    with pytest.raises(ValidationError, match="unknown filters"):
        ProjectConfig.model_validate({"name": "p", "curate": {"filters": ["outcome", "nonsense"]}})


def test_duplicate_filter_is_rejected():
    with pytest.raises(ValidationError, match="duplicate filters"):
        ProjectConfig.model_validate({"name": "p", "curate": {"filters": ["outcome", "outcome"]}})


def test_filters_run_in_canonical_order_regardless_of_config_order():
    cfg = ProjectConfig.model_validate({"name": "p", "curate": {"filters": ["stratify", "outcome", "length"]}})
    assert cfg.curate.ordered_filters() == ["outcome", "length", "stratify"]
    assert cfg.curate.ordered_filters() == [f for f in FILTER_ORDER if f in {"stratify", "outcome", "length"}]


def test_teacher_filter_without_models_is_rejected():
    with pytest.raises(ValidationError, match="teacher_models is empty"):
        ProjectConfig.model_validate({"name": "p", "curate": {"filters": ["teacher"]}})


def test_teacher_filter_with_models_is_accepted():
    cfg = ProjectConfig.model_validate(
        {"name": "p", "curate": {"filters": ["teacher"], "teacher_models": ["m"]}}
    )
    assert cfg.curate.teacher_models == ["m"]


def test_mismatched_seq_len_is_rejected():
    """Samples built at one length cannot be trained at another."""
    with pytest.raises(ValidationError, match="max_seq_len"):
        ProjectConfig.model_validate(
            {"name": "p", "dataset": {"max_seq_len": 4096}, "train": {"base_model": "m", "max_seq_len": 8192}}
        )


def test_min_turns_above_max_turns_is_rejected():
    with pytest.raises(ValidationError, match="exceeds max_turns"):
        ProjectConfig.model_validate({"name": "p", "curate": {"min_turns": 10, "max_turns": 2}})


def test_llm_judge_without_model_is_rejected():
    with pytest.raises(ValidationError, match="judge_model"):
        ProjectConfig.model_validate({"name": "p", "eval": {"grader": {"type": "llm_judge"}}})


@pytest.mark.parametrize(("text", "seconds"), [("30s", 30), ("5m", 300), ("2h", 7200), ("7d", 604800), ("1w", 604800)])
def test_parse_duration(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("bad", ["30", "d", "30days", "", "-1d"])
def test_bad_duration_is_rejected(bad):
    with pytest.raises(ValueError, match="bad duration"):
        parse_duration(bad)


def test_source_discriminator_picks_the_right_model():
    cfg = ProjectConfig.model_validate(
        {"name": "p", "sources": [{"type": "jsonl", "path": "a.jsonl"}, {"type": "gateway", "since": "7d"}]}
    )
    assert cfg.sources[0].path == "a.jsonl"
    assert cfg.sources[1].require_outcome is True


def test_bad_since_in_source_is_rejected():
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate({"name": "p", "sources": [{"type": "gateway", "since": "soon"}]})


def test_fingerprint_covers_selection_and_excludes_the_rest():
    cfg = ProjectConfig(name="p")
    fp = cfg.curation_fingerprint()
    assert "filters" in fp and "near_dedupe_threshold" in fp and "cap_per_cluster" in fp
    assert "gpu_usd_per_hour" not in fp and "lambda_per_usd" not in fp


def test_fingerprint_is_stable_across_irrelevant_changes():
    a = ProjectConfig(name="p")
    b = ProjectConfig(name="p")
    b.serve.gpu_usd_per_hour = 99.0
    b.router.floor = 0.9
    assert a.curation_fingerprint_json() == b.curation_fingerprint_json()


def test_fingerprint_changes_when_a_filter_changes():
    a = ProjectConfig(name="p")
    b = ProjectConfig.model_validate({"name": "p", "curate": {"near_dedupe_threshold": 0.7}})
    assert a.curation_fingerprint_json() != b.curation_fingerprint_json()


def test_load_resolves_relative_paths(tmp_path):
    (tmp_path / "project.yaml").write_text(yaml.safe_dump({"name": "p", "artifacts": "./art"}))
    cfg = ProjectConfig.load(tmp_path / "project.yaml")
    assert cfg.artifacts_dir == (tmp_path / "art").resolve()
    assert cfg.root == tmp_path.resolve()


def test_load_missing_file_names_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="agentdistill init"):
        ProjectConfig.load(tmp_path / "absent.yaml")


def test_load_rejects_non_mapping(tmp_path):
    p = tmp_path / "project.yaml"
    p.write_text("- just\n- a list\n")
    with pytest.raises(ValueError, match="mapping"):
        ProjectConfig.load(p)


def test_example_config_parses():
    from pathlib import Path

    cfg = ProjectConfig.load(Path(__file__).resolve().parents[1] / "project.example.yaml")
    assert cfg.name == "support-agent"
    assert len(cfg.curate.ordered_filters()) == 9

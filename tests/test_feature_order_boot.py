"""The gateway builds exactly the vector its calibration was fitted on, or refuses the calibration and says so.

A gate scored with columns in the wrong places does not crash: it returns confident probabilities for the wrong
inputs and thresholds real traffic on them. These tests pin the boot-time rule and that the offline cascade uses
the same one, so what is measured is what serves.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from agentdistill.cascade.client import FeatureOrderMismatch, from_calibration, scoring_order
from agentdistill.gateway import app as app_module
from agentdistill.gateway.backends import StubBackend
from agentdistill.gateway.state import load_state

CONFIGURED = ["mean_logprob", "min_logprob", "n_tokens", "turn_idx"]


class WidthCheckingModel:
    """Picklable stand-in for a fitted gate: fails loudly if handed a vector of the wrong width."""

    def __init__(self, n_features: int) -> None:
        self.n_features = n_features
        self.seen: list[int] = []

    def predict_proba(self, X):
        X = np.asarray(X)
        assert X.shape[1] == self.n_features, f"fitted on {self.n_features} columns, scored with {X.shape[1]}"
        self.seen.append(X.shape[1])
        return np.array([[0.1, 0.9]] * X.shape[0])


def _prod_with_calibration(registry, tmp_path: Path, fitted_on: list[str]) -> Path:
    registry.insert_dataset({"id": "ds1", "name": "d", "version": 1, "kind": "sft", "filter_config": {},
                             "n_samples": 1, "n_tokens": 1, "content_hash": "h", "path": "/tmp/d"})
    registry.insert_training_run({"id": "tr1", "dataset_id": "ds1", "base_model": "m", "method": "sft",
                                  "config": {}, "status": "succeeded", "started_at": "2026-09-18T00:00:00+00:00"})
    registry.insert_adapter({"id": "ad1", "training_run_id": "tr1", "name": "prod-v1", "version": 1,
                             "base_model": "m", "path": "/tmp/a", "status": "prod"})
    registry.insert_eval_set({"id": "es_x", "name": "x", "trace_ids": [], "grader": {}})
    registry.start_eval_run("ev_x", "es_x", "prod-v1", 1)
    cal_dir = tmp_path / "cal"
    cal_dir.mkdir()
    (cal_dir / "calibration.json").write_text(json.dumps({"usable": True, "feature_order": fitted_on}))
    (cal_dir / "model.pkl").write_bytes(pickle.dumps(WidthCheckingModel(len(fitted_on))))
    registry.insert_calibration({
        "adapter_id": "ad1", "eval_run_id": "ev_x", "feature_order": fitted_on, "model_path": str(cal_dir),
        "threshold": 0.5, "holdout_metrics": {"auroc": 0.8, "ece": 0.02}, "verdict": "usable",
        "report": {"predicted_escalation_rate": 0.3},
    })
    return cal_dir


def _boot(project_config, registry):
    student_choice = {
        "message": {"role": "assistant", "content": "student says hi"}, "finish_reason": "stop",
        "logprobs": {"content": [{"token": "t", "logprob": -0.2, "top_logprobs": []} for _ in range(4)]},
        "text": "tttt",
    }
    teacher = StubBackend([{"message": {"role": "assistant", "content": "teacher says hi"}, "finish_reason": "stop"}])
    return load_state(project_config, registry, student=StubBackend([student_choice]), teacher=teacher)


def _cascade(state):
    app_module.set_state(state)
    with TestClient(app_module.app) as c:
        r = c.post("/v1/chat/completions", json={
            "model": "cascade:prod-v1:auto", "messages": [{"role": "user", "content": "hello"}],
        })
        return r, c.get("/healthz").json()


def test_matching_order_loads_and_scores_with_the_stored_order(project_config, registry, tmp_path):
    project_config.cascade.features = list(CONFIGURED)
    _prod_with_calibration(registry, tmp_path, CONFIGURED)

    state = _boot(project_config, registry)
    assert state.cascade_available
    assert state.feature_names == CONFIGURED

    r, health = _cascade(state)
    assert r.status_code == 200, r.text
    assert r.json()["agentdistill"]["arm"] == "student"
    assert state.calibrator.seen == [len(CONFIGURED)]
    assert health["calibration"]["state"] == "loaded"
    assert health["calibration"]["feature_order"] == CONFIGURED


def test_a_subset_in_configured_order_scores_with_the_subset(project_config, registry, tmp_path):
    """Calibration drops empty columns; the gateway must build the narrower vector, not the configured one."""
    project_config.cascade.features = list(CONFIGURED)
    fitted = ["mean_logprob", "n_tokens"]
    _prod_with_calibration(registry, tmp_path, fitted)

    state = _boot(project_config, registry)
    assert state.cascade_available
    assert state.feature_names == fitted

    r, _ = _cascade(state)
    assert r.status_code == 200, r.text
    assert state.calibrator.seen == [2]


def test_a_reordered_config_refuses_the_calibration_and_says_so(project_config, registry, tmp_path):
    _prod_with_calibration(registry, tmp_path, CONFIGURED)
    project_config.cascade.features = ["min_logprob", "mean_logprob", "n_tokens", "turn_idx"]

    state = _boot(project_config, registry)
    assert not state.cascade_available

    r, health = _cascade(state)
    assert r.json()["agentdistill"]["arm"] == "teacher", "no gate means every turn escalates"
    assert health["cascade_available"] is False
    calibration = health["calibration"]
    assert calibration["state"] == "missing"
    assert "feature order mismatch" in calibration["reason"]
    assert str(CONFIGURED) in calibration["reason"], "the reason names the fitted order"
    assert "['min_logprob', 'mean_logprob', 'n_tokens', 'turn_idx']" in calibration["reason"]
    assert any("feature order mismatch" in n for n in health["notes"])


def test_a_fitted_feature_missing_from_config_refuses(project_config, registry, tmp_path):
    _prod_with_calibration(registry, tmp_path, CONFIGURED)
    project_config.cascade.features = ["mean_logprob", "min_logprob", "n_tokens"]

    state = _boot(project_config, registry)
    assert not state.cascade_available
    _, health = _cascade(state)
    assert health["calibration"]["state"] == "missing"
    assert "['turn_idx'] is not in the configured" in health["calibration"]["reason"]


def test_row_and_artifact_disagreeing_refuses():
    with pytest.raises(FeatureOrderMismatch, match="not the one the row describes"):
        scoring_order(CONFIGURED, ["mean_logprob"], ["min_logprob"])


def test_the_offline_cascade_uses_the_same_rule(tmp_path):
    """`eval run cascade:...` must measure the gate the gateway would serve, or refuse the same way."""
    cal = tmp_path / "cal"
    cal.mkdir()
    (cal / "calibration.json").write_text(json.dumps({"usable": True, "feature_order": ["mean_logprob", "n_tokens"]}))
    (cal / "model.pkl").write_bytes(pickle.dumps(WidthCheckingModel(2)))

    client = from_calibration(None, None, str(cal), CONFIGURED, threshold=0.5)
    assert client.feature_names == ["mean_logprob", "n_tokens"]

    with pytest.raises(FeatureOrderMismatch, match="feature order mismatch"):
        from_calibration(None, None, str(cal), ["n_tokens", "mean_logprob"], threshold=0.5)
    with pytest.raises(FeatureOrderMismatch, match="not the one the row describes"):
        from_calibration(None, None, str(cal), CONFIGURED, threshold=0.5, stored_order=["mean_logprob"])

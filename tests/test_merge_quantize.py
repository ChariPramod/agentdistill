"""Merging and quantizing, and the checks that stop a broken artifact reaching production.

Neither the merge nor the AWQ pass runs here -- both need a GPU and a real model. What runs is everything that
decides whether the result is acceptable, because those are the parts that fail silently. A bad merge loads
cleanly and generates fluent text; a model quantized on generic calibration data chats fine and emits malformed
tool calls. The gates are the only things that notice.
"""

from __future__ import annotations

import json

import pytest

from agentdistill.train.merge import (
    MAX_FULL_MATCH_DRIFT_PP,
    MergeVerificationFailed,
    paired_full_match,
    read_marker,
    verify_merge,
    write_marker,
)
from agentdistill.train.quantize import (
    MIN_CALIB_SAMPLES,
    CalibrationTooSmall,
    calibration_prompts,
    quantization_verdict,
    quantize,
    quantize_fp8,
    read_manifest,
)

# --------------------------------------------------------------------------------------------------------------
# a stub that lets us decide, per turn, whether each path agrees with the recording
# --------------------------------------------------------------------------------------------------------------


def call(name: str, args: dict) -> dict:
    return {"id": f"c{abs(hash(name)) % 1000}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def trace(task_id: str, n_turns: int = 4) -> dict:
    """A trace whose assistant turns each call one tool."""
    messages: list[dict] = [{"role": "user", "content": "help"}]
    for i in range(n_turns):
        messages.append({"role": "assistant", "content": None, "tool_calls": [call("lookup", {"i": i})]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "{}"})
    return {"id": task_id, "task_id": task_id, "messages": messages, "tools": []}


class StubTurnClient:
    """Reproduces the recorded turn, except on the turns named in `wrong`."""

    def __init__(self, wrong: set[int] | None = None) -> None:
        self.wrong = wrong or set()
        self.seen = 0

    def next_turn(self, messages: list[dict], tools: list[dict]) -> dict:
        i = self.seen
        self.seen += 1
        if i in self.wrong:
            return {"role": "assistant", "content": "I am not sure.", "tool_calls": None}
        return _recorded_next(messages)


def _recorded_next(prefix: list[dict]) -> dict:
    """What the recording did next, reconstructed from the prefix length."""
    n_assistant = sum(1 for m in prefix if m["role"] == "assistant")
    return {"role": "assistant", "content": None, "tool_calls": [call("lookup", {"i": n_assistant})]}


def factory(wrong_by_kind: dict[str, set[int]]):
    def make(spec: dict):
        return StubTurnClient(wrong_by_kind.get(spec["kind"], set()))
    return make


TRACES = [trace(f"t{i}") for i in range(10)]


# --------------------------------------------------------------------------------------------------------------
# merge verification
# --------------------------------------------------------------------------------------------------------------


def test_an_identical_merge_passes():
    result = verify_merge("/tmp/merged", "/tmp/lora", "base", TRACES, factory({}))
    assert result["ok"]
    assert result["drift_pp"] == pytest.approx(0.0)
    assert result["merged_full_match"] == pytest.approx(1.0)


def test_a_merge_that_lost_the_adapter_is_refused():
    """A `target_modules` list that missed a projection looks exactly like this: the merged weights behave like
    the base model and the adapter's behaviour is gone."""
    result = verify_merge(
        "/tmp/merged", "/tmp/lora", "base", TRACES,
        factory({"merged": set(range(40))}),  # every merged turn wrong
    )
    assert not result["ok"]
    assert result["drift_pp"] == pytest.approx(-100.0)
    assert "target_modules" in result["reason"]


def test_a_drift_inside_the_tolerance_passes():
    # 60 traces of 4 turns: one disagreeing turn is 1/4 of one task's mean over 60 tasks, or -0.42 pp.
    many = [trace(f"m{i}") for i in range(60)]
    result = verify_merge("/tmp/merged", "/tmp/lora", "base", many, factory({"merged": {0}}))
    assert abs(result["drift_pp"]) <= MAX_FULL_MATCH_DRIFT_PP
    assert result["ok"]
    assert "note" not in result


def test_too_few_turns_to_resolve_the_tolerance_says_so():
    """At the plan's 50 turns, one disagreement is 2 pp: the gate passes only an exact match. That is a
    defensible bar, but not what "within 2 points" sounds like, so the result says which one it is."""
    result = verify_merge("/tmp/merged", "/tmp/lora", "base", TRACES, factory({}))
    assert result["ok"]
    assert "only an exact match" in result["note"]
    assert "at least 50 turns" in result["note"]


def test_a_drift_outside_the_tolerance_is_refused():
    result = verify_merge(
        "/tmp/merged", "/tmp/lora", "base", TRACES, factory({"merged": {0, 1, 2, 3, 4, 5}})
    )
    assert result["drift_pp"] < -MAX_FULL_MATCH_DRIFT_PP
    assert not result["ok"]


def test_the_verdict_carries_an_interval_so_the_tolerance_is_interpretable():
    """On few turns a two-point tolerance sits inside the noise, and a reviewer should be able to see that."""
    result = verify_merge("/tmp/merged", "/tmp/lora", "base", TRACES, factory({"merged": {0}}))
    lo, hi = result["ci95_pp"]
    assert lo <= result["drift_pp"] <= hi
    assert result["n_tasks"] == 10


def test_no_traces_is_a_failure_not_a_pass():
    """An empty verification set must never read as evidence that the merge is fine."""
    result = verify_merge("/tmp/merged", "/tmp/lora", "base", [], factory({}))
    assert not result["ok"]
    assert "no held-out traces" in result["reason"]


def test_paired_comparison_requires_matching_turn_order():
    a = verify_merge("/tmp/m", "/tmp/l", "base", TRACES, factory({}))
    assert a["ok"]
    from agentdistill.eval.teacher_forced import teacher_forced_batched

    merged = teacher_forced_batched(TRACES, StubTurnClient())
    with pytest.raises(ValueError, match="same turns on both sides"):
        paired_full_match(merged, merged[:-1])


def test_the_marker_records_provenance(tmp_path):
    """A directory of safetensors with no provenance is unusable six weeks later."""
    write_marker(str(tmp_path), {"base_model": "b", "adapter_path": "a", "dtype": "bfloat16"})
    assert read_marker(str(tmp_path)) == {"base_model": "b", "adapter_path": "a", "dtype": "bfloat16"}


def test_a_missing_or_corrupt_marker_reads_as_none(tmp_path):
    assert read_marker(str(tmp_path)) is None
    (tmp_path / "agentdistill_merge.json").write_text("{not json")
    assert read_marker(str(tmp_path)) is None


def test_merge_and_verify_raises_rather_than_returning_a_bad_merge(tmp_path, monkeypatch):
    from agentdistill.train import merge as merge_mod

    monkeypatch.setattr(
        merge_mod, "merge_adapter",
        lambda base, adapter, out, dtype="bfloat16": {"out_dir": out, "dtype": dtype},
    )
    with pytest.raises(MergeVerificationFailed):
        merge_mod.merge_and_verify(
            "base", "/tmp/lora", str(tmp_path), TRACES, factory({"merged": set(range(40))})
        )
    # And the marker records the failure, so the directory says why it must not be served.
    assert read_marker(str(tmp_path))["verification"]["ok"] is False


# --------------------------------------------------------------------------------------------------------------
# quantization
# --------------------------------------------------------------------------------------------------------------


def samples(n: int) -> list[dict]:
    return [{"text": f"system prompt with tool schemas, task {i}"} for i in range(n)]


def test_calibration_prompts_come_from_the_training_set():
    prompts = calibration_prompts(samples(500), n=128, seed=0)
    assert len(prompts) == 128
    assert all(p.startswith("system prompt") for p in prompts)


def test_calibration_is_deterministic_for_a_seed():
    assert calibration_prompts(samples(500), seed=3) == calibration_prompts(samples(500), seed=3)


def test_too_few_prompts_is_refused_rather_than_silently_calibrated():
    """AWQ decides what to preserve from what it is shown. On a handful of prompts it optimizes for whatever
    those few happened to contain."""
    with pytest.raises(CalibrationTooSmall, match=str(MIN_CALIB_SAMPLES)):
        calibration_prompts(samples(MIN_CALIB_SAMPLES - 1))


def test_an_empty_dataset_is_refused():
    with pytest.raises(CalibrationTooSmall, match="no usable prompts"):
        calibration_prompts([{"nothing": "here"}])


def test_prompts_are_read_from_whichever_field_a_sample_uses():
    mixed = (
        [{"text": f"a{i}"} for i in range(20)]
        + [{"prompt": f"b{i}"} for i in range(20)]
        + [{"messages": [{"role": "user", "content": "c"}]} for _ in range(20)]
    )
    assert len(calibration_prompts(mixed, n=60)) == 60


def test_fp8_writes_a_marker_and_serves_the_bf16_weights(tmp_path):
    """vLLM applies fp8 online, so there are no new weights -- only a marker, so the registry and the serve
    command cannot disagree about what is being served."""
    out = tmp_path / "q"
    result = quantize_fp8("/weights/merged", str(out))
    assert result["out_dir"] == "/weights/merged"
    assert result["online"] is True
    assert (out / "QUANTIZATION").read_text().strip() == "fp8-online"
    assert read_manifest(str(out))["method"] == "fp8"


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError, match="unknown quantization method"):
        quantize("int3", "/weights", "/out")


def test_gptq_is_accepted_in_config_but_says_it_is_not_implemented(tmp_path):
    with pytest.raises(NotImplementedError, match="gptq"):
        quantize("gptq", "/weights", str(tmp_path))


def test_awq_refuses_to_run_on_too_few_prompts(tmp_path):
    """Checked before the model is loaded, so the failure costs seconds rather than a GPU-hour."""
    with pytest.raises(CalibrationTooSmall):
        quantize("awq", "/weights", str(tmp_path), calib_prompts=["one", "two"])


def test_quantization_that_costs_too_much_success_is_refused():
    verdict = quantization_verdict(bf16_success=0.80, quantized_success=0.74)
    assert not verdict["ok"]
    assert verdict["drop_pp"] == pytest.approx(6.0)


def test_a_small_drop_is_accepted():
    assert quantization_verdict(0.80, 0.79)["ok"]


def test_quantization_scoring_higher_is_not_refused():
    """A quantized model scoring above bf16 is evidence that the eval set is too small to resolve the
    difference, not that quantization improved the model -- and either way it is not a reason to refuse."""
    verdict = quantization_verdict(0.80, 0.84)
    assert verdict["ok"]
    assert verdict["drop_pp"] < 0

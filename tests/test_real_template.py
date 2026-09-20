"""The same mask invariants, against the template that will actually be trained on.

`test_mask_invariants.py` derives its properties from three fixture templates. The real one is a fourth, and it is
the only one that matters on the GPU day: a mask that is right on every fixture and wrong on the configured base
model trains the student to predict tool results, or never to stop. The fixtures also hid a real defect once --
they interpolated tool-call arguments raw, where every real template serializes them -- so a check against the
configured model is not redundant with them.

Skipped when the pinned tokenizer is not in the local HF cache, so the suite stays offline by default. To run it:

    HF_HUB_OFFLINE=0 python -m pytest tests/test_real_template.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentdistill.config import ProjectConfig
from agentdistill.data.build import IGNORE_INDEX, build_trajectory_sample
from agentdistill.data.template_check import (
    SAMPLE_MESSAGES,
    SAMPLE_TOOLS,
    check_template,
    render,
    roundtrip_tool_call,
)
from tests.test_mask_invariants import _header_and_eot, _target_text

CONFIG = Path(__file__).resolve().parents[1] / "examples" / "support_agent" / "project.yaml"


@pytest.fixture(scope="module")
def real_tokenizer():
    """The configured base model's tokenizer, from the local cache; skip rather than download."""
    cfg = ProjectConfig.load(str(CONFIG))
    model = cfg.train.base_model
    if Path(model).exists():
        pytest.skip(f"train.base_model is a local path ({model}); this test is for a Hub model")
    try:
        from transformers import AutoTokenizer

        kwargs = {"revision": cfg.train.base_model_revision} if cfg.train.base_model_revision else {}
        return AutoTokenizer.from_pretrained(model, local_files_only=True, **kwargs)
    except Exception as e:  # any failure here means "not available offline"
        pytest.skip(f"{model} is not in the local HF cache: {type(e).__name__}")


@pytest.fixture(scope="module")
def real_sample(real_tokenizer):
    sample = build_trajectory_sample(real_tokenizer, SAMPLE_MESSAGES, SAMPLE_TOOLS, max_seq_len=8192)
    assert sample is not None, "no sample was built from the real template"
    return sample


def test_the_real_base_model_passes_base_check(real_tokenizer):
    cfg = ProjectConfig.load(str(CONFIG))
    parser = cfg.train.tool_parser
    report = check_template(real_tokenizer, model=cfg.train.base_model)
    assert report.ok, f"base-check fails on the configured base model: {[c.name for c in report.checks if not c.ok]}"

    trip = roundtrip_tool_call(real_tokenizer, parser_name=parser.name if parser else None,
                               family=parser.family if parser else None)
    assert trip["ok"], f"the tool call does not survive the round trip: {trip['detail']} ({trip['parsed']})"


def test_one_end_of_turn_marker_per_assistant_turn(real_tokenizer, real_sample):
    _, eot = _header_and_eot(real_tokenizer, SAMPLE_TOOLS)
    text = _target_text(real_tokenizer, real_sample)
    assert text.count(eot) == sum(1 for m in SAMPLE_MESSAGES if m["role"] == "assistant")


def test_the_assistant_header_does_not_leak_into_targets(real_tokenizer, real_sample):
    header, _ = _header_and_eot(real_tokenizer, SAMPLE_TOOLS)
    assert header.strip() not in _target_text(real_tokenizer, real_sample)


def test_the_last_target_ends_the_turn(real_tokenizer, real_sample):
    _, eot = _header_and_eot(real_tokenizer, SAMPLE_TOOLS)
    assert _target_text(real_tokenizer, real_sample).rstrip().endswith(eot)


def test_every_tool_call_is_fully_inside_the_targets(real_tokenizer, real_sample):
    text = _target_text(real_tokenizer, real_sample)
    for m in SAMPLE_MESSAGES:
        for c in m.get("tool_calls") or []:
            assert c["function"]["name"] in text
            for value in json.loads(c["function"]["arguments"]).values():
                assert str(value) in text


def test_no_environment_text_is_in_the_targets(real_tokenizer, real_sample):
    text = _target_text(real_tokenizer, real_sample)
    assert "You are a support agent" not in text
    assert "Where is the order" not in text


def test_targets_are_a_strict_subset_of_the_sequence(real_sample):
    assert 0 < real_sample.n_target_tokens < real_sample.n_tokens
    for tid, lab in zip(real_sample.input_ids, real_sample.labels, strict=True):
        assert lab in (IGNORE_INDEX, tid)


def test_tool_arguments_render_as_objects_not_quoted_strings(real_tokenizer):
    """The defect base-check caught: a template that serializes what it is given emits `"arguments": "{...}"`
    when handed the OpenAI wire format's string, and the serving stack's parser then recovers a string."""
    text = render(real_tokenizer, SAMPLE_MESSAGES, SAMPLE_TOOLS, add_generation_prompt=False)
    assert '"arguments": "{' not in text and '"arguments":"{' not in text

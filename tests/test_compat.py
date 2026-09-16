"""TRL / transformers compatibility.

These tests simulate versions that lack fields, because the point of the shim is to fail with a clear message on
a rented GPU box rather than after the model is loaded -- or, worse, to succeed while silently dropping a setting
that matters.
"""

from __future__ import annotations

import pytest

pytest.importorskip("trl")

from agentdistill.train.compat import (
    OPTIONAL,
    RENAMED,
    CompatError,
    attn_implementation,
    estimate_total_steps,
    flash_attn_available,
    resolve_report_to,
    sft_config_fields,
    sft_config_kwargs,
)

BASE_CFG = {
    "epochs": 2, "lr": 1e-4, "max_seq_len": 512, "packing": False, "seed": 17,
    "report_to": [], "per_device_batch": 2, "grad_accum": 8, "warmup_ratio": 0.03,
}


# --------------------------------------------------------------------------------------------------------------
# against the installed TRL
# --------------------------------------------------------------------------------------------------------------


def test_resolved_kwargs_construct_a_real_sft_config():
    """The end-to-end guarantee: whatever we build, the installed SFTConfig accepts."""
    from trl import SFTConfig

    kwargs = sft_config_kwargs(BASE_CFG, "/tmp/out", total_steps=100)
    SFTConfig(**kwargs)


def test_sequence_length_is_set_under_whichever_name_exists():
    kwargs = sft_config_kwargs(BASE_CFG, "/tmp/out", total_steps=100)
    used = [k for k in RENAMED["max_seq_len"] if k in kwargs]
    assert len(used) == 1, f"expected exactly one of {RENAMED['max_seq_len']}, got {used}"
    assert kwargs[used[0]] == 512


def test_dataset_preparation_is_always_skipped():
    """Our samples are pre-tokenized and pre-masked; TRL re-rendering them would discard the loss mask."""
    kwargs = sft_config_kwargs(BASE_CFG, "/tmp/out", total_steps=100)
    assert kwargs["dataset_kwargs"] == {"skip_prepare_dataset": True}
    assert kwargs["remove_unused_columns"] is False


def test_eval_strategy_follows_whether_there_is_an_eval_set():
    with_eval = sft_config_kwargs(BASE_CFG, "/tmp/out", has_eval=True, total_steps=100)
    without = sft_config_kwargs(BASE_CFG, "/tmp/out", has_eval=False, total_steps=100)
    key = next(k for k in RENAMED["eval_strategy"] if k in with_eval)
    assert with_eval[key] == "steps" and without[key] == "no"
    assert with_eval["load_best_model_at_end"] is True
    assert without["load_best_model_at_end"] is False


# --------------------------------------------------------------------------------------------------------------
# warmup: a rename that is not a rename
# --------------------------------------------------------------------------------------------------------------


def test_warmup_is_expressed_in_whichever_field_exists():
    """transformers 5.x removed `warmup_ratio`. Dropping it would silently remove warmup from every run."""
    kwargs = sft_config_kwargs(BASE_CFG, "/tmp/out", total_steps=100)
    assert ("warmup_ratio" in kwargs) or ("warmup_steps" in kwargs)
    if "warmup_steps" in kwargs:
        assert kwargs["warmup_steps"] == 3, "0.03 of 100 steps"
    else:
        assert kwargs["warmup_ratio"] == 0.03


def test_warmup_conversion_without_a_known_total_falls_back_to_none(monkeypatch, caplog):
    import agentdistill.train.compat as compat

    monkeypatch.setattr(compat, "sft_config_fields", lambda: _fields_without("warmup_ratio"))
    kwargs = compat.sft_config_kwargs(BASE_CFG, "/tmp/out", total_steps=None)
    assert kwargs.get("warmup_steps") == 0
    assert "could not be converted" in caplog.text


def test_zero_warmup_stays_zero(monkeypatch):
    import agentdistill.train.compat as compat

    monkeypatch.setattr(compat, "sft_config_fields", lambda: _fields_without("warmup_ratio"))
    kwargs = compat.sft_config_kwargs({**BASE_CFG, "warmup_ratio": 0.0}, "/tmp/out", total_steps=100)
    assert kwargs["warmup_steps"] == 0


@pytest.mark.parametrize(
    ("n_samples", "cfg", "expected"),
    [
        (800, {"per_device_batch": 2, "grad_accum": 8, "epochs": 2}, 100),
        (16, {"per_device_batch": 1, "grad_accum": 1, "epochs": 1}, 16),
        (5, {"per_device_batch": 8, "grad_accum": 8, "epochs": 1}, 1),        # never zero
        (800, {"per_device_batch": 2, "grad_accum": 8, "max_steps": 7}, 7),   # max_steps wins
    ],
)
def test_estimate_total_steps(n_samples, cfg, expected):
    assert estimate_total_steps(n_samples, cfg) == expected


# --------------------------------------------------------------------------------------------------------------
# simulated older versions
# --------------------------------------------------------------------------------------------------------------


def _fields_without(*missing: str) -> set[str]:
    return sft_config_fields() - set(missing)


def test_packing_without_padding_free_is_refused(monkeypatch):
    """Packed sequences without padding_free attend across sample boundaries: a silent quality bug."""
    import agentdistill.train.compat as compat

    monkeypatch.setattr(compat, "sft_config_fields", lambda: _fields_without("padding_free"))
    monkeypatch.setattr(compat, "flash_attn_available", lambda: True)
    with pytest.raises(CompatError, match="attend across sample boundaries"):
        compat.sft_config_kwargs({**BASE_CFG, "packing": True}, "/tmp/out", total_steps=100)


def test_missing_padding_free_is_fine_when_not_packing(monkeypatch):
    import agentdistill.train.compat as compat

    monkeypatch.setattr(compat, "sft_config_fields", lambda: _fields_without("padding_free"))
    kwargs = compat.sft_config_kwargs({**BASE_CFG, "packing": False}, "/tmp/out", total_steps=100)
    assert "padding_free" not in kwargs


def test_missing_dataset_kwargs_is_fatal(monkeypatch):
    """Without it, TRL re-renders our samples and throws away the loss mask."""
    import agentdistill.train.compat as compat

    monkeypatch.setattr(compat, "sft_config_fields", lambda: _fields_without("dataset_kwargs"))
    with pytest.raises(CompatError, match="discard the loss mask"):
        compat.sft_config_kwargs(BASE_CFG, "/tmp/out", total_steps=100)


def test_missing_sequence_length_field_is_fatal(monkeypatch):
    import agentdistill.train.compat as compat

    monkeypatch.setattr(compat, "sft_config_fields", lambda: _fields_without(*RENAMED["max_seq_len"]))
    with pytest.raises(CompatError, match="max_length"):
        compat.sft_config_kwargs(BASE_CFG, "/tmp/out", total_steps=100)


def test_missing_required_field_names_itself(monkeypatch):
    import agentdistill.train.compat as compat

    monkeypatch.setattr(compat, "sft_config_fields", lambda: _fields_without("learning_rate"))
    with pytest.raises(CompatError, match="learning_rate"):
        compat.sft_config_kwargs(BASE_CFG, "/tmp/out", total_steps=100)


def test_absent_optional_field_is_simply_not_set(monkeypatch):
    """Nothing to warn about: packing is off, so a TRL without the field changes nothing."""
    import agentdistill.train.compat as compat

    monkeypatch.setattr(compat, "sft_config_fields", lambda: _fields_without("packing"))
    kwargs = compat.sft_config_kwargs({**BASE_CFG, "packing": False}, "/tmp/out", total_steps=100)
    assert "packing" not in kwargs


def test_every_tolerated_field_has_a_documented_reason():
    """A field dropped without a reason is a setting that vanished silently."""
    assert set(OPTIONAL) == {"padding_free", "dataset_kwargs", "packing"}
    assert all(reason.strip() for reason in OPTIONAL.values())


def test_packing_requested_but_no_packing_field_is_fatal(monkeypatch):
    import agentdistill.train.compat as compat

    monkeypatch.setattr(compat, "sft_config_fields", lambda: _fields_without("packing"))
    monkeypatch.setattr(compat, "flash_attn_available", lambda: True)
    with pytest.raises(CompatError, match="no `packing` field"):
        compat.sft_config_kwargs({**BASE_CFG, "packing": True}, "/tmp/out", total_steps=100)


# --------------------------------------------------------------------------------------------------------------
# attention and reporting
# --------------------------------------------------------------------------------------------------------------


def test_packing_falls_back_when_flash_attn_is_absent(monkeypatch, caplog):
    import agentdistill.train.compat as compat

    monkeypatch.setattr(compat, "flash_attn_available", lambda: False)
    kwargs = compat.sft_config_kwargs({**BASE_CFG, "packing": True}, "/tmp/out", total_steps=100)
    assert kwargs.get("packing") is False
    assert kwargs.get("padding_free") is False
    assert "flash-attn is not installed" in caplog.text


def test_attn_implementation_matches_packing(monkeypatch):
    import agentdistill.train.compat as compat

    monkeypatch.setattr(compat, "flash_attn_available", lambda: False)
    assert compat.attn_implementation({"packing": True}) == "sdpa"
    monkeypatch.setattr(compat, "flash_attn_available", lambda: True)
    assert compat.attn_implementation({"packing": True}) == "flash_attention_2"
    assert compat.attn_implementation({"packing": False}) == "sdpa"


def test_report_to_drops_missing_backends(caplog):
    """A missing logging backend must not cost a training run."""
    assert resolve_report_to([]) == []
    assert resolve_report_to(None) == []
    resolved = resolve_report_to(["tensorboard"])
    try:
        import tensorboard  # noqa: F401

        assert resolved == ["tensorboard"]
    except ImportError:
        assert resolved == []
        assert "without it" in caplog.text


def test_flash_attn_probe_does_not_raise():
    assert isinstance(flash_attn_available(), bool)
    assert attn_implementation({}) == "sdpa"

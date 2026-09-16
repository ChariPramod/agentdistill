"""SFT smoke test: two optimizer steps on a tiny randomly-initialized model, on CPU.

This does not check that training *works* -- two steps on a random 4-layer model teaches nothing. It checks the
wiring: that the pre-tokenized dataset reaches the trainer without TRL re-rendering it, that the LoRA adapter
saves and loads, and that the config field names the installed TRL actually accepts are the ones we pass.
Library API churn is the failure this catches, and it is the one the plan calls out as most likely.
"""

from __future__ import annotations

import pytest

from tests.conftest import TOKENIZER_DIR

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")

from agentdistill.data.artifact import write_dataset  # noqa: E402
from agentdistill.data.build import Sample  # noqa: E402
from agentdistill.train.sft import (  # noqa: E402
    build_lora_config,
    build_quantization_config,
    load_dataset_splits,
    train_sft,
)


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """A 4-layer randomly-initialized Llama-shaped model over the fixture tokenizer's vocabulary."""
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    out = tmp_path_factory.mktemp("tiny_model")
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    config = LlamaConfig(
        vocab_size=max(tok.vocab_size, len(tok)),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        bos_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out


@pytest.fixture
def tiny_dataset(tmp_path, tokenizer):
    """A handful of real samples built through the actual masking path."""
    from agentdistill.data.build import build_trajectory_sample
    from agentdistill.data.template_check import SAMPLE_MESSAGES, SAMPLE_TOOLS

    samples: list[Sample] = []
    for i in range(8):
        messages = [
            *SAMPLE_MESSAGES[:-1],
            {"role": "assistant", "content": f"Order o_{i} shipped yesterday and arrives soon."},
        ]
        s = build_trajectory_sample(tokenizer, messages, SAMPLE_TOOLS, max_seq_len=256)
        assert s is not None
        samples.append(s)
    artifact = write_dataset(
        samples, tmp_path / "ds", name="smoke", version=1, tokenizer=str(TOKENIZER_DIR),
        max_seq_len=256, filter_config={},
    )
    return artifact.path


def test_load_dataset_splits_keeps_only_model_columns(tiny_dataset):
    train, evalset, n_target = load_dataset_splits(tiny_dataset, test_size=0.25)
    assert set(train.column_names) == {"input_ids", "labels"}
    assert evalset is not None and len(evalset) >= 1
    assert len(train) + len(evalset) == 8
    assert n_target > 0, "target-token total must be counted before the bookkeeping columns are dropped"


def test_load_dataset_splits_without_holdout(tiny_dataset):
    train, evalset, _ = load_dataset_splits(tiny_dataset, test_size=0)
    assert evalset is None and len(train) == 8


def test_lora_config_uses_project_values():
    lora = build_lora_config({"lora": {"r": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["q_proj"]}})
    assert lora.r == 8 and lora.lora_alpha == 16
    assert set(lora.target_modules) == {"q_proj"}, "peft normalizes target_modules to a set"


def test_quantization_config_is_optional():
    assert build_quantization_config({"quantization": None}) is None
    with pytest.raises(ValueError, match="unknown quantization"):
        build_quantization_config({"quantization": "3bit"})


@pytest.mark.slow
def test_two_steps_on_cpu_saves_a_loadable_adapter(tiny_model_dir, tiny_dataset, tmp_path):
    cfg = {
        "base_model": str(tiny_model_dir),
        "quantization": None,
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
        "max_seq_len": 256,
        "epochs": 1,
        "max_steps": 2,
        "lr": 1e-4,
        "per_device_batch": 1,
        "grad_accum": 1,
        "packing": False,
        "bf16": False,
        "gradient_checkpointing": False,
        "eval_every_steps": 2,
        "logging_steps": 1,
        "seed": 17,
        "report_to": [],
    }
    result = train_sft(cfg, tiny_dataset, tmp_path / "adapter")

    assert result.steps == 2, "max_steps must be honoured"
    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists(), "LoRA adapter was not saved"
    assert result.metrics["n_train_samples"] + result.metrics["n_eval_samples"] == 8
    assert "throughput_target_tok_per_s" in result.metrics, "the GPU-run checklist requires throughput"
    assert result.metrics["attn_implementation"] in {"sdpa", "flash_attention_2"}
    assert result.metrics["packing"] is False

    # The adapter must load back onto the base model.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(str(tiny_model_dir))
    merged = PeftModel.from_pretrained(base, str(tmp_path / "adapter"))
    assert any("lora" in n for n, _ in merged.named_parameters()), "no LoRA parameters in the reloaded adapter"


# --------------------------------------------------------------------------------------------------------------
# CLI wiring
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_cli_train_records_run_and_adapter(tiny_model_dir, tmp_path, monkeypatch):
    """`agentdistill train sft` must record a training run and a candidate adapter, not just write files."""
    import json

    import yaml
    from typer.testing import CliRunner

    from agentdistill.cli import app
    from agentdistill.registry import open_registry
    from tests.conftest import make_trace

    monkeypatch.chdir(tmp_path)
    traces = [
        make_trace(
            f"t{i}",
            task=f"Where is order {i} for my account, it has been a while now",
            closing=" ".join(f"Point {j} of case {i} is confirmed as {i * 11 + j}" for j in range(5)),
        )
        for i in range(8)
    ]
    (tmp_path / "traces.jsonl").write_text("\n".join(json.dumps(t) for t in traces) + "\n")
    (tmp_path / "project.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "demo",
                "registry": "sqlite:///.agentdistill/registry.db",
                "artifacts": "./artifacts",
                "reports": "./reports",
                "dataset": {"max_seq_len": 512},
                "curate": {"clusters": 2, "cap_per_cluster": 50},
                "train": {
                    "base_model": str(tiny_model_dir),
                    "max_seq_len": 512,
                    "quantization": None,
                    "packing": False,
                    "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
                    "epochs": 1,
                    "per_device_batch": 1,
                    "grad_accum": 1,
                },
            }
        )
    )

    runner = CliRunner()
    assert runner.invoke(app, ["ingest", "jsonl", "traces.jsonl"]).exit_code == 0
    assert runner.invoke(app, ["curate"]).exit_code == 0

    result = runner.invoke(app, ["train", "sft", "demo", "--max-steps", "2"])
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.exit_code == 0, combined
    assert "done" in combined and "candidate" in combined

    reg = open_registry("sqlite:///.agentdistill/registry.db", root=tmp_path)
    adapters = reg.list_adapters()
    assert len(adapters) == 1
    assert adapters[0]["status"] == "candidate", "adapters must not be promoted on a loss curve"
    assert adapters[0]["name"] == "demo" and adapters[0]["version"] == 1

    from sqlalchemy import text

    with reg.engine.connect() as conn:
        row = conn.execute(text("SELECT status, metrics FROM training_runs")).mappings().one()
    assert row["status"] == "succeeded"
    assert json.loads(row["metrics"])["steps"] == 2
    reg.close()


@pytest.mark.slow
def test_cli_train_on_unknown_dataset_is_a_clear_error(tmp_path, monkeypatch):
    import yaml
    from typer.testing import CliRunner

    from agentdistill.cli import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "project.yaml").write_text(
        yaml.safe_dump({"name": "demo", "registry": "sqlite:///.agentdistill/registry.db",
                        "train": {"base_model": "x", "max_seq_len": 8192}})
    )
    result = CliRunner().invoke(app, ["train", "sft", "nope"])
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.exit_code == 1
    assert "no dataset named" in combined


@pytest.mark.slow
def test_tokenizer_only_base_model_gives_an_actionable_error(tiny_dataset, tmp_path):
    """The first thing an example-follower hits: `base_model` points at a tokenizer with no weights."""
    from agentdistill.train.sft import NoSuchBaseModel, train_sft

    cfg = {"base_model": str(TOKENIZER_DIR), "quantization": None, "max_seq_len": 256, "packing": False,
           "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj"]}, "report_to": []}
    with pytest.raises(NoSuchBaseModel, match="base-check"):
        train_sft(cfg, tiny_dataset, tmp_path / "adapter")

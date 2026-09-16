"""DPO smoke test: two steps on a tiny CPU model.

Two steps on a random model teach nothing. What this checks is the wiring: pairs render, the installed TRL
accepts the config, the trainer runs, a LoRA adapter saves and reloads, and the reward metrics the round loop
reads back actually exist.
"""

from __future__ import annotations

import json

import pytest

from tests.conftest import TOKENIZER_DIR, make_call

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")

from agentdistill.train.dpo import FLAT_REWARD_ACCURACY, prepare_pairs, train_dpo  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    out = tmp_path_factory.mktemp("tiny_dpo_model")
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=max(tok.vocab_size, len(tok)), hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512,
            bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
        )
    )
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out


def make_pairs(n: int = 4) -> list[dict]:
    tools = [
        {"type": "function", "function": {"name": "refund_order", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "cancel_order", "parameters": {"type": "object"}}},
    ]
    return [
        {
            "prompt": [
                {"role": "system", "content": "You are a support agent."},
                {"role": "user", "content": f"case {i}: my order arrived damaged, please refund it"},
            ],
            "chosen": [{"role": "assistant", "content": "Refunding now.",
                        "tool_calls": [make_call("c1", "refund_order", {"order_id": f"o_{i}"})]}],
            "rejected": [{"role": "assistant", "content": "Cancelling instead.",
                          "tool_calls": [make_call("c1", "cancel_order", {"order_id": f"o_{i}"})]}],
            "tools": tools,
            "task_id": f"t{i}",
            "pair_kind": "rollout",
        }
        for i in range(n)
    ]


CFG = {
    "max_seq_len": 512, "seed": 17, "report_to": [], "bf16": False, "gradient_checkpointing": False,
    "dpo_lr": 5e-6, "dpo_beta": 0.1, "max_steps": 2, "dpo_per_device_batch": 1, "dpo_grad_accum": 1,
    "dpo_lora_r": 4, "dpo_target_modules": ["q_proj", "v_proj"], "warmup_ratio": 0.0, "packing": False,
}


def test_prepare_pairs_reports_what_it_dropped(tokenizer):
    bad = make_pairs(1)
    bad[0]["rejected"] = bad[0]["chosen"]
    rendered, stats = prepare_pairs(tokenizer, make_pairs(3) + bad)
    assert stats["n_in"] == 4 and stats["n_usable"] == 3 and len(rendered) == 3
    assert sum(stats["dropped"].values()) == 1
    assert stats["kinds"]["rollout"] == 3


@pytest.mark.slow
def test_two_steps_saves_a_loadable_adapter(tiny_model_dir, tmp_path):
    result = train_dpo(CFG, make_pairs(4), str(tiny_model_dir), tmp_path / "dpo")

    assert result.steps == 2
    assert result.n_pairs == 4
    assert (tmp_path / "dpo" / "adapter_model.safetensors").exists()

    # The round loop reads these back to decide whether the pairs taught anything.
    assert result.final_reward_accuracy is not None, "rewards/accuracies must be logged"
    assert result.final_reward_margin is not None, "rewards/margins must be logged"
    assert 0.0 <= result.final_reward_accuracy <= 1.0

    metrics = json.loads((tmp_path / "dpo" / "dpo_metrics.json").read_text())
    assert metrics["n_pairs"] == 4 and "looks_flat" in metrics

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(str(tiny_model_dir))
    merged = PeftModel.from_pretrained(base, str(tmp_path / "dpo"))
    assert any("lora" in n for n, _ in merged.named_parameters())


@pytest.mark.slow
def test_flat_reward_accuracy_is_flagged(tiny_model_dir, tmp_path):
    """Two steps on a random model cannot learn a preference, so this run should look flat -- and say so."""
    result = train_dpo(CFG, make_pairs(4), str(tiny_model_dir), tmp_path / "dpo_flat")
    assert result.looks_flat == (result.final_reward_accuracy < FLAT_REWARD_ACCURACY)


def test_no_usable_pairs_is_a_clear_error(tiny_model_dir, tmp_path):
    identical = make_pairs(2)
    for p in identical:
        p["rejected"] = p["chosen"]
    with pytest.raises(ValueError, match="no usable preference pairs"):
        train_dpo(CFG, identical, str(tiny_model_dir), tmp_path / "dpo_none")


def test_missing_base_model_is_actionable(tmp_path):
    from agentdistill.train.dpo import NoSuchBaseModel

    with pytest.raises(NoSuchBaseModel, match="adapter merge"):
        train_dpo(CFG, make_pairs(2), str(TOKENIZER_DIR), tmp_path / "dpo_nomodel")

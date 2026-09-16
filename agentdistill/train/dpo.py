"""Direct preference optimization on rollout pairs.

The metric to watch is `rewards/accuracies`: how often the model already prefers the chosen side. It should climb
well above 0.5 within the first epoch. Sitting at 0.5 almost always means the pairs are not distinguishable --
chosen and rejected differ only in formatting, or they continue different prompts -- and the run is wasted GPU
time. `dpo_data.pair_is_valid` rejects the common causes before training starts, and `train_dpo` reports the
final accuracy so a flat run is visible in the registry rather than only in a dashboard.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentdistill.train.compat import attn_implementation, dpo_config_kwargs, estimate_total_steps
from agentdistill.train.dpo_data import balance_kinds, bos_token_text, filter_pairs, render_pair
from agentdistill.train.sft import NoSuchBaseModel, _require_training_deps

logger = logging.getLogger(__name__)

#: Below this, the pairs are not teaching a preference and the adapter should not be trusted.
FLAT_REWARD_ACCURACY = 0.55


@dataclass
class DpoResult:
    adapter_path: str
    steps: int
    n_pairs: int
    final_reward_margin: float | None = None
    final_reward_accuracy: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def looks_flat(self) -> bool:
        """True when the model never learned to prefer the chosen side."""
        return (
            self.final_reward_accuracy is not None
            and self.final_reward_accuracy < FLAT_REWARD_ACCURACY
        )

    def to_dict(self) -> dict:
        return {
            "adapter_path": self.adapter_path,
            "steps": self.steps,
            "n_pairs": self.n_pairs,
            "final_reward_margin": self.final_reward_margin,
            "final_reward_accuracy": self.final_reward_accuracy,
            "looks_flat": self.looks_flat,
            **self.metrics,
        }


def prepare_pairs(tok: Any, pairs: list[dict], max_teacher_ratio: float = 1.0) -> tuple[list[dict], dict]:
    """Validate, balance, and render. Returns (rendered, stats)."""
    usable, dropped = filter_pairs(pairs)
    balanced, kinds = balance_kinds(usable, max_teacher_ratio=max_teacher_ratio)
    strip = bos_token_text(tok)
    rendered = [render_pair(tok, p, strip_bos=strip) for p in balanced]
    return rendered, {
        "n_in": len(pairs),
        "n_usable": len(usable),
        "n_rendered": len(rendered),
        "dropped": dropped,
        "kinds": kinds,
        "stripped_bos": bool(strip),
    }


def train_dpo(
    cfg: dict,
    pairs: list[dict],
    base_or_merged: str,
    out_dir: str | Path,
    max_teacher_ratio: float = 1.0,
) -> DpoResult:
    """Train a LoRA adapter with DPO on preference pairs."""
    _require_training_deps()

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    cfg = dict(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(base_or_merged)
    rendered, stats = prepare_pairs(tok, pairs, max_teacher_ratio=max_teacher_ratio)
    if not rendered:
        raise ValueError(
            f"no usable preference pairs from {len(pairs)} candidates. Reasons: {stats['dropped']}. "
            f"A pair needs a shared prompt and two continuations that genuinely differ."
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_or_merged,
            dtype=torch.bfloat16 if cfg.get("bf16", True) else torch.float32,
            attn_implementation=attn_implementation(cfg),
        )
    except (OSError, ValueError) as e:
        raise NoSuchBaseModel(
            f"could not load model weights for {base_or_merged!r}: {e}\n"
            "DPO starts from the merged SFT model, so run `agentdistill adapter merge` first."
        ) from e

    from datasets import Dataset

    ds = Dataset.from_list(rendered)
    total_steps = estimate_total_steps(len(rendered), {**cfg, "per_device_batch": cfg.get("dpo_per_device_batch", 1),
                                                      "grad_accum": cfg.get("dpo_grad_accum", 16),
                                                      "epochs": cfg.get("dpo_epochs", 1)})
    args = DPOConfig(**dpo_config_kwargs(cfg, str(out_dir), total_steps=total_steps))

    lora_r = cfg.get("dpo_lora_r", 16)
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=cfg.get("dpo_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    )

    # ref_model=None with a PEFT adapter makes TRL use the frozen base as the reference, which avoids holding a
    # second full copy of the model.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(out_dir))

    margins = [x for x in trainer.state.log_history if "rewards/margins" in x]
    accuracies = [x for x in trainer.state.log_history if "rewards/accuracies" in x]
    result = DpoResult(
        adapter_path=str(out_dir),
        steps=int(trainer.state.global_step),
        n_pairs=len(rendered),
        final_reward_margin=float(margins[-1]["rewards/margins"]) if margins else None,
        final_reward_accuracy=float(accuracies[-1]["rewards/accuracies"]) if accuracies else None,
        metrics={"base_or_merged": base_or_merged, "lora_r": lora_r, **stats},
    )
    if result.looks_flat:
        logger.warning(
            "DPO reward accuracy finished at %.3f. The model is not learning to prefer the chosen side, which "
            "usually means the pairs are not distinguishable. Inspect a few before trusting this adapter.",
            result.final_reward_accuracy,
        )
    (out_dir / "dpo_metrics.json").write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str))
    return result

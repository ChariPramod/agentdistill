"""Supervised fine-tuning with LoRA / QLoRA.

The dataset is already tokenized and masked by `agentdistill.data.build`, so the trainer receives `input_ids` and
`labels` directly and TRL's own dataset preparation is skipped. That is deliberate: the loss mask is the part of
this pipeline most likely to be silently wrong, so it is computed once, asserted in tests, and inspectable with
`agentdistill dataset inspect` -- not recomputed here from a chat template a second time.

TRL's config field names move between releases. `resolve_sft_config` checks what the installed version accepts
and reports an actionable error rather than passing an unknown kwarg into a stack trace.
"""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Fields we ask for that TRL has renamed or moved at some point. If one is missing from the installed version,
#: the trainer says which release changed it instead of failing on an unexpected keyword.
_MOVED_FIELDS = {
    "max_length": "was `max_seq_length` before TRL 0.12",
    "padding_free": "added in TRL 0.11; without it, packing would let sequences attend across sample boundaries",
    "eval_strategy": "was `evaluation_strategy` before transformers 4.41",
}


class TrainingUnavailable(ImportError):
    """The training extra is not installed."""


class NoSuchBaseModel(ValueError):
    """`train.base_model` has no loadable weights. Most often a tokenizer-only path."""


@dataclass
class TrainResult:
    adapter_path: str
    eval_loss: float | None
    steps: int
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "adapter_path": self.adapter_path,
            "eval_loss": self.eval_loss,
            "steps": self.steps,
            **self.metrics,
        }


def _require_training_deps() -> None:
    missing = []
    for mod in ("torch", "transformers", "trl", "peft"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise TrainingUnavailable(
            f"training needs {', '.join(missing)}; install with `pip install 'agentdistill[train]'`"
        )


def accepted_sft_fields() -> set[str]:
    """The config keys the installed TRL's SFTConfig actually accepts.

    Split out so tests can simulate an older TRL without monkeypatching the library in two places.
    """
    from trl import SFTConfig

    accepted = set(inspect.signature(SFTConfig.__init__).parameters)
    # Dataclasses expose fields rather than __init__ params on some versions.
    if hasattr(SFTConfig, "__dataclass_fields__"):
        accepted |= set(SFTConfig.__dataclass_fields__)
    return accepted


def resolve_sft_config(requested: dict[str, Any]) -> dict[str, Any]:
    """Drop config keys the installed TRL does not accept, loudly.

    Silently ignoring an unknown key is how `padding_free` quietly turns off and packed sequences start attending
    across sample boundaries -- a quality bug with no symptom except a slightly worse student.
    """
    accepted = accepted_sft_fields()

    out, dropped = {}, {}
    for k, v in requested.items():
        if k in accepted:
            out[k] = v
        else:
            dropped[k] = v

    for k in dropped:
        note = _MOVED_FIELDS.get(k, "not present in this TRL version")
        logger.warning("SFTConfig does not accept %r (%s); it was dropped from this run", k, note)
    if "padding_free" in dropped and requested.get("packing"):
        raise ValueError(
            "packing was requested but this TRL version has no `padding_free`. Packed sequences would attend "
            "across sample boundaries, which is a silent quality bug. Upgrade TRL or set train.packing: false."
        )
    return out


def build_lora_config(cfg: dict) -> Any:
    from peft import LoraConfig

    lora = cfg.get("lora") or {}
    return LoraConfig(
        r=lora.get("r", 32),
        lora_alpha=lora.get("alpha", 64),
        lora_dropout=lora.get("dropout", 0.05),
        target_modules=lora.get("target_modules"),
        task_type="CAUSAL_LM",
    )


def build_quantization_config(cfg: dict) -> Any | None:
    quant = cfg.get("quantization")
    if not quant:
        return None
    import torch
    from transformers import BitsAndBytesConfig

    if quant == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    if quant == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"unknown quantization {quant!r}; use '4bit', '8bit', or null")


def resolve_report_to(requested: Any) -> list[str]:
    """Drop reporting backends that are not installed.

    Losing a training run at step zero because a logging backend is missing is a bad trade. The run is what
    costs money; the dashboard is not.
    """
    backends = list(requested or [])
    available = []
    for backend in backends:
        if backend == "tensorboard":
            try:
                import tensorboard  # noqa: F401
            except ImportError:
                try:
                    import tensorboardX  # noqa: F401
                except ImportError:
                    logger.warning(
                        "report_to includes 'tensorboard' but it is not installed; training will run without it. "
                        "Install `agentdistill[train]` to get loss curves."
                    )
                    continue
        available.append(backend)
    return available


def _attn_implementation(cfg: dict) -> str:
    """Packing requires flash attention to avoid cross-sample attention; fall back rather than pack unsafely."""
    if not cfg.get("packing"):
        return "sdpa"
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        logger.warning("packing requested but flash-attn is not installed; falling back to sdpa without packing")
        cfg["packing"] = False
        return "sdpa"
    return "flash_attention_2"


def load_dataset_splits(dataset_path: str | Path, seed: int = 17, test_size: float = 0.05):
    """Load the parquet artifact and hold out a slice for eval.

    Only `input_ids` and `labels` are kept: the bookkeeping columns would be passed to the model as kwargs.
    """
    from datasets import load_dataset

    path = Path(dataset_path)
    data_file = path / "data.parquet" if path.is_dir() else path
    ds = load_dataset("parquet", data_files=str(data_file))["train"]
    keep = {"input_ids", "labels"}
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    if test_size <= 0 or len(ds) < 2:
        return ds, None
    split = ds.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"]


def train_sft(cfg: dict, dataset_path: str | Path, out_dir: str | Path) -> TrainResult:
    """LoRA / QLoRA supervised fine-tuning on a pre-tokenized, pre-masked dataset."""
    _require_training_deps()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    cfg = dict(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_model = cfg["base_model"]
    tok = AutoTokenizer.from_pretrained(base_model)
    attn = _attn_implementation(cfg)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=build_quantization_config(cfg),
            dtype=torch.bfloat16 if cfg.get("bf16", True) else torch.float32,
            attn_implementation=attn,
        )
    except (OSError, ValueError) as e:
        raise NoSuchBaseModel(
            f"could not load model weights for train.base_model={base_model!r}: {e}\n"
            "A tokenizer alone is enough to *build* a dataset but not to train one. Point train.base_model at a "
            "real instruct model (check it first with `agentdistill base-check <model>`), keeping in mind that "
            "the dataset was tokenized with the tokenizer named in its manifest -- a different tokenizer means a "
            "new dataset version."
        ) from e

    train_ds, eval_ds = load_dataset_splits(dataset_path, seed=cfg.get("seed", 17))

    requested = {
        "output_dir": str(out_dir),
        "num_train_epochs": cfg.get("epochs", 2),
        "learning_rate": cfg.get("lr", 1e-4),
        "lr_scheduler_type": cfg.get("scheduler", "cosine"),
        "warmup_ratio": cfg.get("warmup_ratio", 0.03),
        "per_device_train_batch_size": cfg.get("per_device_batch", 2),
        "gradient_accumulation_steps": cfg.get("grad_accum", 8),
        "max_length": cfg.get("max_seq_len", 8192),
        "packing": cfg.get("packing", False),
        "padding_free": cfg.get("packing", False),
        "bf16": cfg.get("bf16", True),
        "gradient_checkpointing": cfg.get("gradient_checkpointing", True),
        "eval_strategy": "steps" if eval_ds is not None else "no",
        "eval_steps": cfg.get("eval_every_steps", 100),
        "save_strategy": "steps",
        "save_steps": cfg.get("eval_every_steps", 100),
        "load_best_model_at_end": eval_ds is not None,
        "metric_for_best_model": "eval_loss",
        "logging_steps": cfg.get("logging_steps", 10),
        "seed": cfg.get("seed", 17),
        "report_to": resolve_report_to(cfg.get("report_to", ["tensorboard"])),
        # The dataset is already tokenized and masked; TRL must not re-render it from a chat template.
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "remove_unused_columns": False,
        "max_steps": cfg.get("max_steps", -1),
    }
    args = SFTConfig(**resolve_sft_config(requested))

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        peft_config=build_lora_config(cfg),
    )
    trainer.train()
    trainer.save_model(str(out_dir))

    eval_loss = None
    if eval_ds is not None:
        eval_loss = float(trainer.evaluate().get("eval_loss", float("nan")))

    metrics = {
        "base_model": base_model,
        "n_train_samples": len(train_ds),
        "n_eval_samples": len(eval_ds) if eval_ds is not None else 0,
        "packing": bool(args.packing) if hasattr(args, "packing") else False,
        "attn_implementation": attn,
        "quantization": cfg.get("quantization"),
    }
    (out_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))

    return TrainResult(
        adapter_path=str(out_dir),
        eval_loss=eval_loss,
        steps=int(trainer.state.global_step),
        metrics=metrics,
    )

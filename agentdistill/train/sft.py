"""Supervised fine-tuning with LoRA / QLoRA.

The dataset is already tokenized and masked by `agentdistill.data.build`, so the trainer receives `input_ids` and
`labels` directly and TRL's own dataset preparation is skipped. That is deliberate: the loss mask is the part of
this pipeline most likely to be silently wrong, so it is computed once, asserted in tests, and inspectable with
`agentdistill dataset inspect` -- not recomputed here from a chat template a second time.

The dataset is only valid for the tokenizer it was built with, so `check_dataset_tokenizer` compares the dataset
manifest's tokenizer identity with the configured base model before any weights load. Without it, a base-model
change with a forgotten rebuild trains on token ids from a different vocabulary and looks exactly like a normal run.

TRL's config field names move between releases. `resolve_sft_config` checks what the installed version accepts
and reports an actionable error rather than passing an unknown kwarg into a stack trace.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentdistill.train.compat import (
    attn_implementation,
    estimate_total_steps,
    flash_attn_available,
    sft_config_kwargs,
)

logger = logging.getLogger(__name__)


class TrainingUnavailable(ImportError):
    """The training extra is not installed."""


class NoSuchBaseModel(ValueError):
    """`train.base_model` has no loadable weights. Most often a tokenizer-only path."""


class TokenizerMismatch(ValueError):
    """The dataset was tokenized for a different base model or revision than the one configured to train."""


def _same_model(a: str, b: str) -> bool:
    """Both sides come from `cfg.base_model`-resolved strings; a local path is also compared resolved, so a
    trailing slash or a symlinked checkout does not count as a different model."""
    if a == b:
        return True
    pa, pb = Path(a), Path(b)
    return pa.exists() and pb.exists() and pa.resolve() == pb.resolve()


def check_dataset_tokenizer(cfg: dict, dataset_path: str | Path) -> list[str]:
    """Refuse a dataset tokenized for a different model than `cfg` trains. Returns warnings; raises on mismatch.

    `cfg` is `ProjectConfig.train_config()`, whose `base_model` is resolved the same way the dataset build
    resolved it (`cfg.base_model`), so the two strings are comparable. The model id has been in every manifest
    since the first schema; the pinned revision was added later, so a manifest without it is only a failure when
    the config pins a revision -- then it cannot show it matches. Otherwise it is a warning.
    """
    from agentdistill.data.artifact import MANIFEST_NAME

    path = Path(dataset_path)
    manifest_path = path / MANIFEST_NAME if path.is_dir() else path.parent / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    want_model = str(cfg.get("base_model") or "")
    want_rev = cfg.get("base_model_revision") or None
    have_model = manifest.get("tokenizer")
    rebuild = "Rebuild the dataset with `agentdistill curate` against the configured base model, then train on it."

    if have_model is not None and not _same_model(str(have_model), want_model):
        raise TokenizerMismatch(
            f"dataset at {path} was tokenized for base model {have_model!r} (revision "
            f"{manifest.get('base_model_revision')!r}), but train.base_model is {want_model!r} (revision "
            f"{want_rev!r}). {rebuild}"
        )

    if "base_model_revision" not in manifest:
        why = ("has no manifest" if not manifest else "manifest predates tokenizer recording")
        if want_rev:
            raise TokenizerMismatch(
                f"dataset at {path} {why}, so it cannot show it was tokenized at train.base_model_revision "
                f"{want_rev!r} of {want_model!r} (dataset records base model {have_model!r}, revision unknown). "
                f"{rebuild}"
            )
        return [f"dataset at {path} {why}; its tokenizer ({have_model!r}) is assumed to match "
                f"{want_model!r}. Rebuild with `agentdistill curate` to record it."]

    have_rev = manifest.get("base_model_revision")
    if have_rev != want_rev:
        raise TokenizerMismatch(
            f"dataset at {path} was tokenized for {have_model!r} at revision {have_rev!r}, but "
            f"train.base_model_revision is {want_rev!r} (base model {want_model!r}). {rebuild}"
        )
    return []


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


def load_dataset_splits(dataset_path: str | Path, seed: int = 17, test_size: float = 0.05):
    """Load the parquet artifact and hold out a slice for eval.

    Returns `(train, eval_or_none, n_target_tokens)`. Only `input_ids` and `labels` reach the trainer -- the
    bookkeeping columns would be forwarded to the model as keyword arguments -- so the target-token total is
    counted here, before they are dropped, for the throughput metric.
    """
    from datasets import load_dataset

    path = Path(dataset_path)
    data_file = path / "data.parquet" if path.is_dir() else path
    ds = load_dataset("parquet", data_files=str(data_file))["train"]
    n_target_tokens = int(sum(ds["n_target_tokens"])) if "n_target_tokens" in ds.column_names else 0
    keep = {"input_ids", "labels"}
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    if test_size <= 0 or len(ds) < 2:
        return ds, None, n_target_tokens
    split = ds.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"], n_target_tokens


def _train_runtime(trainer: Any) -> float:
    """Wall-clock training seconds, from whichever log entry carries it."""
    for entry in reversed(trainer.state.log_history or []):
        if "train_runtime" in entry:
            return float(entry["train_runtime"])
    return 0.0


def train_sft(
    cfg: dict,
    dataset_path: str | Path,
    out_dir: str | Path,
    next_action_traces: list[dict] | None = None,
    resume_adapter: str | Path | None = None,
) -> TrainResult:
    """LoRA / QLoRA supervised fine-tuning on a pre-tokenized, pre-masked dataset.

    `next_action_traces` enables the teacher-forced next-action callback during eval. Loss is a proxy; next-action
    accuracy is the first signal that reflects what the agent actually has to do.

    `resume_adapter` continues training an existing adapter instead of initializing a new one. This is what a
    retrain does: the new corpus is mostly the old corpus plus a few weeks of traffic, and restarting from the
    base model would throw away everything the serving adapter knows in order to relearn it. Continuation wants
    a much lower learning rate than a fresh run -- the retrain loop uses a third -- because the weights start
    near a good solution and a fresh-run rate walks straight out of it.
    """
    # Before the dependency check and before any weights load: loading an 8B model only to discover the dataset
    # is for a different one wastes the most expensive minutes of the day.
    for warning in check_dataset_tokenizer(cfg, dataset_path):
        logger.warning("train sft: %s", warning)

    _require_training_deps()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    cfg = dict(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_model = cfg["base_model"]
    # `train.base_model_revision` pins the Hub commit, so a retag upstream cannot change what trains. Passed only
    # when set, so an unpinned config calls transformers exactly as before.
    revision = {"revision": cfg["base_model_revision"]} if cfg.get("base_model_revision") else {}
    tok = AutoTokenizer.from_pretrained(base_model, **revision)
    attn = attn_implementation(cfg)

    model: Any
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            **revision,
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

    peft_config: Any = build_lora_config(cfg)
    if resume_adapter:
        from peft import PeftModel

        # `is_trainable=True` is load-bearing: without it PEFT loads the adapter for inference with requires_grad
        # off, training runs, the loss moves nowhere, and the saved adapter is byte-identical to the one it
        # started from. That failure is silent and looks exactly like a corpus with nothing new in it.
        model = PeftModel.from_pretrained(model, str(resume_adapter), is_trainable=True)
        # The adapter is already attached; handing SFTTrainer a peft_config as well would wrap it a second time.
        peft_config = None

    train_ds, eval_ds, n_target_tokens = load_dataset_splits(dataset_path, seed=cfg.get("seed", 17))
    total_steps = estimate_total_steps(len(train_ds), cfg)

    args = SFTConfig(**sft_config_kwargs(cfg, str(out_dir), has_eval=eval_ds is not None, total_steps=total_steps))

    callbacks: list[Any] = []
    if next_action_traces:
        from agentdistill.train.callbacks import NextActionCallback

        callbacks.append(
            NextActionCallback(
                next_action_traces,
                tok,
                parser_name=(cfg.get("tool_parser") or {}).get("name"),
                family=(cfg.get("tool_parser") or {}).get("family"),
            )
        )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        peft_config=peft_config,
        callbacks=callbacks or None,
    )
    trainer.train()
    trainer.save_model(str(out_dir))

    eval_loss = None
    if eval_ds is not None:
        eval_loss = float(trainer.evaluate().get("eval_loss", float("nan")))

    runtime = _train_runtime(trainer)
    metrics = {
        "base_model": base_model,
        "n_train_samples": len(train_ds),
        "n_eval_samples": len(eval_ds) if eval_ds is not None else 0,
        "packing": bool(getattr(args, "packing", False)),
        "padding_free": bool(getattr(args, "padding_free", False)),
        "attn_implementation": attn,
        "flash_attn": flash_attn_available(),
        "quantization": cfg.get("quantization"),
        "total_steps_estimated": total_steps,
        "resumed_from": str(resume_adapter) if resume_adapter else None,
    }
    if n_target_tokens and runtime:
        epochs = float(cfg.get("epochs", 2))
        metrics["train_runtime_s"] = round(runtime, 2)
        metrics["throughput_target_tok_per_s"] = round(n_target_tokens * epochs / runtime, 2)
    for cb in callbacks:
        last = getattr(cb, "last_summary", None)
        if last:
            metrics["next_action_full_match_final"] = last["full_match"][0]
            metrics["next_action_name_match_final"] = last["name_match_on_tool_turns"]
            metrics["next_action_args_match_final"] = last["args_match_on_tool_turns"]
    (out_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))

    return TrainResult(
        adapter_path=str(out_dir),
        eval_loss=eval_loss,
        steps=int(trainer.state.global_step),
        metrics=metrics,
    )

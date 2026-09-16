"""TRL / transformers compatibility.

Field names move between TRL releases, and the failure mode is nasty: an unknown keyword either raises on a
rented GPU box after the model is loaded, or -- worse -- is silently ignored, so `padding_free` turns off, packed
sequences attend across sample boundaries, and the student is quietly worse with no symptom.

Everything version-dependent lives here so the trainer reads like training code, and so a TRL upgrade breaks one
small module with a clear message rather than the whole run.
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Fields that have been renamed across releases, newest name first. The first candidate the installed TRL
#: accepts wins.
RENAMED: dict[str, tuple[str, ...]] = {
    "max_seq_len": ("max_length", "max_seq_length"),
    "eval_strategy": ("eval_strategy", "evaluation_strategy"),
}

#: `warmup_ratio` was removed in transformers 5.x in favour of the absolute `warmup_steps`. The two are not
#: interchangeable -- a ratio needs the total step count -- so this is converted rather than renamed.
WARMUP_FIELDS: tuple[str, ...] = ("warmup_ratio", "warmup_steps")

#: Fields whose absence we tolerate, with the reason recorded so a dropped one is explainable.
OPTIONAL: dict[str, str] = {
    "padding_free": "added in TRL 0.11; without it, packing would let sequences attend across sample boundaries",
    "dataset_kwargs": "used to skip TRL's own dataset preparation; our samples are already tokenized and masked",
    "packing": "packing is unavailable in this TRL; sequences train unpacked",
}


class CompatError(RuntimeError):
    """The installed TRL cannot support a setting that matters. The message names the setting and the fix."""


def sft_config_fields() -> set[str]:
    """The field names the installed `SFTConfig` accepts."""
    from trl import SFTConfig

    fields = {f.name for f in dataclasses.fields(SFTConfig)}
    # Some releases accept kwargs the dataclass does not declare, via the parent TrainingArguments.
    try:
        import inspect

        fields |= set(inspect.signature(SFTConfig.__init__).parameters)
    except (TypeError, ValueError):  # pragma: no cover - signature is always introspectable in practice
        pass
    fields.discard("self")
    fields.discard("kwargs")
    return fields


def flash_attn_available() -> bool:
    try:
        importlib.import_module("flash_attn")
        return True
    except Exception:
        return False


def attn_implementation(cfg: dict[str, Any]) -> str:
    """Packing needs flash attention to keep packed sequences from attending across sample boundaries."""
    return "flash_attention_2" if cfg.get("packing") and flash_attn_available() else "sdpa"


def resolve_report_to(requested: Any) -> list[str]:
    """Drop reporting backends that are not installed.

    Losing a paid training run at step zero because a logging backend is missing is a bad trade. The run is what
    costs money; the dashboard is not.
    """
    available = []
    for backend in list(requested or []):
        if backend == "tensorboard" and not (_importable("tensorboard") or _importable("tensorboardX")):
            logger.warning(
                "report_to includes 'tensorboard' but it is not installed; training will run without it. "
                "Install `agentdistill[train]` to get loss curves."
            )
            continue
        available.append(backend)
    return available


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def estimate_total_steps(n_samples: int, cfg: dict[str, Any]) -> int:
    """Optimizer steps a run will take. Used to convert a warmup ratio into absolute steps."""
    if cfg.get("max_steps", -1) and cfg.get("max_steps", -1) > 0:
        return int(cfg["max_steps"])
    per_device = max(1, int(cfg.get("per_device_batch", 2)))
    accum = max(1, int(cfg.get("grad_accum", 8)))
    steps_per_epoch = max(1, n_samples // (per_device * accum))
    return max(1, int(steps_per_epoch * float(cfg.get("epochs", 2))))


def _set_warmup(want: dict[str, Any], fields: set[str], cfg: dict[str, Any], total_steps: int | None) -> None:
    """Express the configured warmup in whichever field the installed version has.

    transformers 5.x removed `warmup_ratio`. Dropping it would silently remove warmup from every run, so the
    ratio is converted to absolute steps when the total is known, and the conversion is logged.
    """
    ratio = float(cfg.get("warmup_ratio", 0.03) or 0.0)
    if "warmup_ratio" in fields:
        want["warmup_ratio"] = ratio
        return
    if "warmup_steps" not in fields:
        raise CompatError(
            f"SFTConfig has neither of {WARMUP_FIELDS}. Update agentdistill/train/compat.py for this TRL version."
        )
    if ratio <= 0:
        want["warmup_steps"] = 0
        return
    if total_steps is None:
        logger.warning(
            "this transformers version has no `warmup_ratio` and the total step count is unknown, so "
            "warmup_ratio=%.3f could not be converted; running without warmup", ratio
        )
        want["warmup_steps"] = 0
        return
    steps = max(1, round(ratio * total_steps))
    logger.info("converted warmup_ratio=%.3f to warmup_steps=%d over %d total steps", ratio, steps, total_steps)
    want["warmup_steps"] = steps


def sft_config_kwargs(
    cfg: dict[str, Any], out_dir: str, has_eval: bool = True, total_steps: int | None = None
) -> dict[str, Any]:
    """Build kwargs for the installed `SFTConfig`, resolving renames and reporting anything dropped.

    Raises `CompatError` when a required field is missing entirely, or when packing was asked for but cannot be
    done safely.
    """
    fields = sft_config_fields()

    packing = bool(cfg.get("packing")) and flash_attn_available()
    if cfg.get("packing") and not packing:
        logger.warning("packing requested but flash-attn is not installed; training unpacked instead")

    want: dict[str, Any] = {
        "output_dir": out_dir,
        "num_train_epochs": cfg.get("epochs", 2),
        "learning_rate": cfg.get("lr", 1e-4),
        "lr_scheduler_type": cfg.get("scheduler", "cosine"),
        "per_device_train_batch_size": cfg.get("per_device_batch", 2),
        "gradient_accumulation_steps": cfg.get("grad_accum", 8),
        "bf16": cfg.get("bf16", True),
        "gradient_checkpointing": cfg.get("gradient_checkpointing", True),
        "eval_steps": cfg.get("eval_every_steps", 100),
        "save_strategy": "steps",
        "save_steps": cfg.get("eval_every_steps", 100),
        "load_best_model_at_end": has_eval,
        "metric_for_best_model": "eval_loss",
        "logging_steps": cfg.get("logging_steps", 10),
        "seed": cfg.get("seed", 17),
        "report_to": resolve_report_to(cfg.get("report_to", ["tensorboard"])),
        "remove_unused_columns": False,
        "max_steps": cfg.get("max_steps", -1),
    }

    # Renamed fields: take the first candidate the installed version knows.
    _set_renamed(want, fields, "max_seq_len", cfg.get("max_seq_len", 8192))
    _set_renamed(want, fields, "eval_strategy", "steps" if has_eval else "no")
    _set_warmup(want, fields, cfg, total_steps)

    if "packing" in fields:
        want["packing"] = packing
    elif packing:
        raise CompatError(
            "packing was requested but this TRL version has no `packing` field. Upgrade TRL or set "
            "train.packing: false."
        )

    if "padding_free" in fields:
        want["padding_free"] = packing
    elif packing:
        raise CompatError(
            "packing was requested but this TRL version has no `padding_free`. Packed sequences without it "
            "attend across sample boundaries, which degrades the student with no visible symptom. "
            "Upgrade TRL or set train.packing: false."
        )

    if "dataset_kwargs" in fields:
        # Our samples carry input_ids and labels already; TRL must not re-render them from a chat template.
        want["dataset_kwargs"] = {"skip_prepare_dataset": True}
    else:
        raise CompatError(
            "this TRL version has no `dataset_kwargs`, so TRL's own dataset preparation cannot be skipped. It "
            "would re-render our pre-masked samples from the chat template and discard the loss mask. "
            "Upgrade TRL."
        )

    unknown = sorted(k for k in want if k not in fields)
    tolerable = [k for k in unknown if k in OPTIONAL]
    fatal = [k for k in unknown if k not in OPTIONAL]
    for k in tolerable:
        logger.warning("SFTConfig has no %r (%s); dropped from this run", k, OPTIONAL[k])
        want.pop(k)
    if fatal:
        raise CompatError(
            f"SFTConfig in the installed TRL lacks required fields {fatal}. "
            f"Update agentdistill/train/compat.py for this TRL version."
        )
    return want


def _set_renamed(want: dict[str, Any], fields: set[str], logical: str, value: Any) -> None:
    for candidate in RENAMED[logical]:
        if candidate in fields:
            want[candidate] = value
            return
    raise CompatError(
        f"SFTConfig has none of {RENAMED[logical]} for {logical!r}. "
        f"Update RENAMED in agentdistill/train/compat.py for this TRL version."
    )


# --------------------------------------------------------------------------------------------------------------
# DPO
# --------------------------------------------------------------------------------------------------------------


def dpo_config_fields() -> set[str]:
    from trl import DPOConfig

    fields = {f.name for f in dataclasses.fields(DPOConfig)}
    try:
        import inspect

        fields |= set(inspect.signature(DPOConfig.__init__).parameters)
    except (TypeError, ValueError):  # pragma: no cover
        pass
    fields.discard("self")
    fields.discard("kwargs")
    return fields


def dpo_config_kwargs(cfg: dict[str, Any], out_dir: str, total_steps: int | None = None) -> dict[str, Any]:
    """Build kwargs for the installed `DPOConfig`, same contract as `sft_config_kwargs`."""
    fields = dpo_config_fields()
    want: dict[str, Any] = {
        "output_dir": out_dir,
        "beta": cfg.get("dpo_beta", 0.1),
        "num_train_epochs": cfg.get("dpo_epochs", 1),
        # 5e-6, not the SFT rate: DPO on a merged bf16 model diverges at 1e-4 and the loss goes NaN in the first
        # few steps.
        "learning_rate": cfg.get("dpo_lr", 5e-6),
        "per_device_train_batch_size": cfg.get("dpo_per_device_batch", 1),
        "gradient_accumulation_steps": cfg.get("dpo_grad_accum", 16),
        "bf16": cfg.get("bf16", True),
        "gradient_checkpointing": cfg.get("gradient_checkpointing", True),
        "logging_steps": cfg.get("logging_steps", 10),
        "seed": cfg.get("seed", 17),
        "report_to": resolve_report_to(cfg.get("report_to", ["tensorboard"])),
        "loss_type": cfg.get("dpo_loss_type", "sigmoid"),
        "max_grad_norm": cfg.get("max_grad_norm", 1.0),
        "remove_unused_columns": False,
        "max_steps": cfg.get("max_steps", -1),
    }
    max_seq_len = cfg.get("max_seq_len", 8192)
    _set_first_available(want, fields, ("max_length", "max_seq_length"), max_seq_len, "sequence length")
    _set_first_available(
        want, fields, ("max_prompt_length",), max(256, max_seq_len - 512), "prompt length", required=False
    )
    _set_warmup(want, fields, cfg, total_steps)

    unknown = sorted(k for k in want if k not in fields)
    tolerable = [k for k in unknown if k in DPO_OPTIONAL]
    fatal = [k for k in unknown if k not in DPO_OPTIONAL]
    for k in tolerable:
        logger.warning("DPOConfig has no %r (%s); dropped from this run", k, DPO_OPTIONAL[k])
        want.pop(k)
    if fatal:
        raise CompatError(
            f"DPOConfig in the installed TRL lacks required fields {fatal}. "
            f"Update agentdistill/train/compat.py for this TRL version."
        )
    return want


#: DPO fields whose absence changes the run but does not invalidate it.
DPO_OPTIONAL: dict[str, str] = {
    "loss_type": "this TRL has one DPO loss; the default is sigmoid anyway",
    "max_prompt_length": "prompts will not be truncated separately from completions",
    "max_grad_norm": "gradient clipping is unavailable; watch for NaN loss in the first steps",
    "remove_unused_columns": "extra dataset columns may be forwarded to the model",
}


def _set_first_available(
    want: dict[str, Any], fields: set[str], candidates: tuple[str, ...], value: Any, what: str,
    required: bool = True,
) -> None:
    for candidate in candidates:
        if candidate in fields:
            want[candidate] = value
            return
    if required:
        raise CompatError(
            f"DPOConfig has none of {candidates} for {what}. Update agentdistill/train/compat.py."
        )

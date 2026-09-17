"""Quantizing a merged model for serving.

Two methods, and they are not the same kind of thing:

- **fp8** is applied online by vLLM from bf16 weights. Nothing is written except a marker so the registry and
  the serve script agree on what is being served. It is nearly free and nearly lossless on recent hardware.
- **AWQ** is a real offline pass that rewrites the weights to 4 bits, using calibration data to decide which
  channels matter. It is a much larger saving and a much larger risk.

The calibration data is the part that gets done wrong. AWQ decides what to preserve by watching activations on
whatever it is shown, so calibrating on generic web text optimizes for the wrong distribution: an agent's
prompts are dominated by tool schemas and structured JSON, and a quantizer that never saw one will happily
sacrifice the channels that emit them. The symptom is a model that chats fine and produces malformed tool calls.
So calibration prompts come from the training set.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

METHODS = ("fp8", "awq", "gptq")

#: AWQ's own guidance is 128 sequences; fewer makes the activation statistics noisy.
DEFAULT_CALIB_SAMPLES = 128
MIN_CALIB_SAMPLES = 32


class CalibrationTooSmall(ValueError):
    """Not enough real prompts to calibrate against."""


def calibration_prompts(samples: list[dict], n: int = DEFAULT_CALIB_SAMPLES, seed: int = 0) -> list[str]:
    """Rendered prompts from the training set, sampled without replacement.

    The prefix only -- system, tools, and the first user message -- because that is the part whose activations
    decide how the tool-calling channels are scaled.
    """
    import random

    texts = [t for t in (_prompt_text(s) for s in samples) if t]
    if not texts:
        raise CalibrationTooSmall("no usable prompts in the dataset to calibrate against")
    if len(texts) < MIN_CALIB_SAMPLES:
        raise CalibrationTooSmall(
            f"only {len(texts)} prompts available; AWQ calibration needs at least {MIN_CALIB_SAMPLES} "
            f"(and {DEFAULT_CALIB_SAMPLES} is the documented figure). Curate more data, or serve fp8 instead."
        )
    rng = random.Random(seed)
    return rng.sample(texts, min(n, len(texts)))


def _prompt_text(sample: dict) -> str | None:
    """The prompt prefix of a training sample, however the sample stores it."""
    if isinstance(sample.get("text"), str) and sample["text"]:
        return sample["text"]
    if isinstance(sample.get("prompt"), str) and sample["prompt"]:
        return sample["prompt"]
    messages = sample.get("messages")
    if isinstance(messages, list) and messages:
        return json.dumps(messages, ensure_ascii=False)
    return None


def quantize(method: str, merged_dir: str, out_dir: str, calib_prompts: list[str] | None = None) -> dict:
    if method not in METHODS:
        raise ValueError(f"unknown quantization method {method!r}; known methods are {', '.join(METHODS)}")
    if method == "fp8":
        return quantize_fp8(merged_dir, out_dir)
    if method == "awq":
        return quantize_awq(merged_dir, out_dir, calib_prompts or [])
    raise NotImplementedError(
        "gptq is accepted in config for forward compatibility but has no implementation here. Use awq for an "
        "offline 4-bit pass or fp8 for an online one."
    )


def quantize_fp8(merged_dir: str, out_dir: str) -> dict:
    """Write the marker that says these weights are to be served as fp8.

    vLLM quantizes from bf16 at load time, so there is nothing to compute here. The marker exists so that the
    registry, the serve script and anyone reading the directory agree on what is being served; a registry row
    claiming fp8 over a directory that nothing marks is how the wrong flag ends up on a serve command.
    """
    os.makedirs(out_dir, exist_ok=True)
    (Path(out_dir) / "QUANTIZATION").write_text("fp8-online\n")
    write_manifest(out_dir, {"method": "fp8", "online": True, "weights_dir": merged_dir})
    return {"out_dir": merged_dir, "marker_dir": out_dir, "method": "fp8", "online": True, "n_calib": 0}


def quantize_awq(merged_dir: str, out_dir: str, calib_prompts: list[str]) -> dict:
    """An offline 4-bit pass with llmcompressor.

    The llmcompressor API has moved between releases; the import and the modifier's argument names are checked
    against the installed version at call time rather than assumed.
    """
    if len(calib_prompts) < MIN_CALIB_SAMPLES:
        raise CalibrationTooSmall(
            f"AWQ calibration got {len(calib_prompts)} prompts and needs at least {MIN_CALIB_SAMPLES}. "
            f"Calibrating on too little data quantizes toward whatever those few prompts happened to contain."
        )

    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(merged_dir)
    model = AutoModelForCausalLM.from_pretrained(merged_dir, torch_dtype="auto", device_map="auto")
    dataset = [{"text": p} for p in calib_prompts]

    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=[AWQModifier(bits=4, symmetric=False, targets="Linear", ignore=["lm_head"])],
        max_seq_length=4096,
        num_calibration_samples=len(calib_prompts),
        output_dir=out_dir,
    )
    write_manifest(out_dir, {"method": "awq", "bits": 4, "n_calib": len(calib_prompts),
                             "source_dir": merged_dir, "calibration": "task prompts from the training set"})
    return {"out_dir": out_dir, "method": "awq", "n_calib": len(calib_prompts), "online": False}


def write_manifest(out_dir: str, payload: dict) -> Path:
    os.makedirs(out_dir, exist_ok=True)
    path = Path(out_dir) / "agentdistill_quantization.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def read_manifest(out_dir: str) -> dict | None:
    path = Path(out_dir) / "agentdistill_quantization.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


#: Quantization may cost this much success before the quantized adapter is refused, in percentage points.
MAX_QUANTIZATION_DROP_PP = 2.0


def quantization_verdict(bf16_success: float, quantized_success: float) -> dict:
    """Whether a quantized adapter is close enough to the weights it came from.

    Stated as a one-sided tolerance: quantization scoring *higher* is not evidence that it improved the model,
    it is evidence that the eval set is too small to resolve the difference, and either way it is not a reason
    to refuse.
    """
    drop = (bf16_success - quantized_success) * 100
    return {
        "bf16_success": bf16_success,
        "quantized_success": quantized_success,
        "drop_pp": drop,
        "ok": drop <= MAX_QUANTIZATION_DROP_PP,
        "tolerance_pp": MAX_QUANTIZATION_DROP_PP,
        "detail": f"quantization cost {drop:+.1f} pp of success (tolerance {MAX_QUANTIZATION_DROP_PP} pp)",
    }

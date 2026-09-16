"""Pydantic models for project.yaml.

One file drives every stage. Anything a run depends on lives here so that a run is reproducible from the config
plus a dataset hash; anything that varies per invocation is a CLI flag instead.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_DURATION = re.compile(r"^(\d+)([smhdw])$")
_DURATION_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}

#: Filters the curation pipeline knows how to run, in the canonical order of section 3.1.
FILTER_ORDER: tuple[str, ...] = (
    "outcome",
    "schema_valid",
    "no_error_loops",
    "length",
    "teacher",
    "exact_dedupe",
    "near_dedupe",
    "decontaminate",
    "pii",
    "quality_judge",
    "stratify",
)


def parse_duration(value: str) -> int:
    """`30d` -> seconds. Raises ValueError on anything else."""
    m = _DURATION.match(value.strip())
    if not m:
        raise ValueError(f"bad duration {value!r}; use forms like 30d, 12h, 90m")
    return int(m.group(1)) * _DURATION_SECONDS[m.group(2)]


class StrictModel(BaseModel):
    """Reject unknown keys: a typo in project.yaml should fail loudly, not silently disable a filter."""

    model_config = ConfigDict(extra="forbid")


class AgentReplaySource(StrictModel):
    type: Literal["agentreplay"] = "agentreplay"
    db: str
    since: str = "30d"

    @field_validator("since")
    @classmethod
    def _check_since(cls, v: str) -> str:
        parse_duration(v)
        return v


class JsonlSource(StrictModel):
    type: Literal["jsonl"] = "jsonl"
    path: str


class OtelSource(StrictModel):
    type: Literal["otel"] = "otel"
    path: str


class GatewaySource(StrictModel):
    type: Literal["gateway"] = "gateway"
    since: str = "7d"
    #: Gateway requests without a graded outcome cannot supervise anything.
    require_outcome: bool = True

    @field_validator("since")
    @classmethod
    def _check_since(cls, v: str) -> str:
        parse_duration(v)
        return v


Source = Annotated[
    AgentReplaySource | JsonlSource | OtelSource | GatewaySource,
    Field(discriminator="type"),
]


class TeacherConfig(StrictModel):
    model: str
    provider: str = "anthropic"
    pricing_from_registry: bool = True
    input_per_mtok: float | None = None
    output_per_mtok: float | None = None
    cache_read_per_mtok: float | None = None


class EmbeddingsConfig(StrictModel):
    #: `hash` needs no network and no API key; it is deterministic and good enough to smoke-test the pipeline,
    #: but it clusters by token overlap, not meaning. Use a real provider before you trust the coverage report.
    provider: Literal["hash", "openai", "sentence-transformers"] = "hash"
    model: str | None = None
    dim: int = 256
    batch_size: int = 64


class CurateConfig(StrictModel):
    filters: list[str] = Field(
        default_factory=lambda: [
            "outcome",
            "schema_valid",
            "no_error_loops",
            "length",
            "exact_dedupe",
            "near_dedupe",
            "decontaminate",
            "pii",
            "stratify",
        ]
    )
    near_dedupe_threshold: float = Field(0.85, ge=0.0, le=1.0)
    near_dedupe_num_perm: int = Field(128, ge=16)
    near_dedupe_shingle_n: int = Field(5, ge=1)
    #: Aggressive structural dedupe: mask numbers and ids before shingling, collapsing a corpus to roughly one
    #: sample per trajectory shape. Off by default; see `curate.dedupe.normalize_literals` for when to turn it on.
    near_dedupe_normalize_literals: bool = False
    min_turns: int = Field(2, ge=1)
    max_turns: int = Field(40, ge=1)
    max_consecutive_tool_errors: int = Field(3, ge=1)
    max_repeated_identical_calls: int = Field(2, ge=1)
    decontaminate_ngram: int = Field(8, ge=1)
    decontaminate_overlap: float = Field(0.5, ge=0.0, le=1.0)
    clusters: int = Field(32, ge=1)
    cap_per_cluster: int = Field(400, ge=1)
    teacher_models: list[str] = Field(default_factory=list)
    quality_judge_min_score: float = Field(3.0, ge=1.0, le=5.0)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)

    @field_validator("filters")
    @classmethod
    def _known_filters(cls, v: list[str]) -> list[str]:
        unknown = [f for f in v if f not in FILTER_ORDER]
        if unknown:
            raise ValueError(f"unknown filters {unknown}; known filters are {list(FILTER_ORDER)}")
        dupes = {f for f in v if v.count(f) > 1}
        if dupes:
            raise ValueError(f"duplicate filters {sorted(dupes)}")
        return v

    @model_validator(mode="after")
    def _turn_bounds(self) -> CurateConfig:
        if self.min_turns > self.max_turns:
            raise ValueError(f"min_turns ({self.min_turns}) exceeds max_turns ({self.max_turns})")
        return self

    def ordered_filters(self) -> list[str]:
        """The configured filters in canonical order. Order matters: cheap rejections run before expensive ones,
        and dedupe runs before stratification so caps are computed on the surviving set."""
        return [f for f in FILTER_ORDER if f in set(self.filters)]


class DatasetConfig(StrictModel):
    max_seq_len: int = Field(8192, ge=128)
    window_turns: int = Field(12, ge=1)
    with_rationale: bool = False
    #: Trajectories longer than max_seq_len become turn_window samples rather than being dropped.
    windows_for_long_trajectories: bool = True
    eval_holdout_frac: float = Field(0.0, ge=0.0, lt=1.0)


class LoraConfig(StrictModel):
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


class ToolParserConfig(StrictModel):
    """How tool calls are recovered from generated text.

    `name` is vLLM's parser name and goes straight into `vllm serve --tool-call-parser`. `family` selects the
    fallback regex used when vLLM is not installed. Setting both means `base-check` and the serving command agree
    on one answer instead of each guessing.
    """

    name: str | None = None
    family: Literal["hermes", "llama3_json", "agentdistill_fixture"] | None = None


class TrainConfig(StrictModel):
    base_model: str
    method: Literal["sft", "dpo", "rft", "grpo"] = "sft"
    backend: Literal["trl", "unsloth"] = "trl"
    quantization: Literal["4bit", "8bit"] | None = "4bit"
    lora: LoraConfig = Field(default_factory=LoraConfig)
    tool_parser: ToolParserConfig = Field(default_factory=ToolParserConfig)
    max_seq_len: int = Field(8192, ge=128)
    epochs: float = 2
    lr: float = 1.0e-4
    scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    per_device_batch: int = 2
    grad_accum: int = 8
    packing: bool = True
    eval_every_steps: int = 100
    early_stop_patience: int = 3
    seed: int = 17


class OnPolicyConfig(StrictModel):
    rounds: int = Field(2, ge=0)
    k_rollouts: int = Field(8, ge=1)
    rft_cap_per_task: int = Field(2, ge=1)
    dpo_beta: float = 0.1
    #: Below this many usable pairs, DPO is noise.
    min_pairs: int = Field(40, ge=1)
    #: Above this share of fuzzily-replayed tool results, the rollouts' successes do not mean much.
    max_fuzzy_share: float = Field(0.5, ge=0.0, le=1.0)


class GraderConfig(StrictModel):
    """How a task outcome is decided.

    `predicate` is the strongest: a pure function of the final state, with no model in the loop. `llm_judge` is
    the fallback for projects without state predicates, and it never gets reported as a bare number -- a judge
    has its own error rate and the report must say so.
    """

    type: Literal["label", "exact", "predicate", "replay_predicate", "llm_judge"] = "label"
    rubric: str | None = None
    judge_model: str | None = None
    #: Dotted path to a callable resolving a trace to its predicate, for `predicate` and `replay_predicate`.
    #: The example project uses `examples.support_agent.replay_grader:predicate_for_trace`.
    predicate_source: str | None = None

    @model_validator(mode="after")
    def _judge_needs_model(self) -> GraderConfig:
        if self.type == "llm_judge" and not self.judge_model:
            raise ValueError("grader.type is llm_judge but no judge_model is set")
        return self


class EvalConfig(StrictModel):
    eval_set: str | None = None
    #: Where the judge model is served, for grader.type == "llm_judge".
    judge_base_url: str = "https://api.openai.com/v1"
    n_per_task: int = Field(5, ge=1)
    policy: Literal["strict", "fuzzy"] = "strict"
    grader: GraderConfig = Field(default_factory=GraderConfig)
    max_turns: int = Field(40, ge=1)


class CascadeConfig(StrictModel):
    k_samples: int = Field(2, ge=0)
    max_success_drop_pp: float = Field(1.0, ge=0.0)
    features: list[str] = Field(
        default_factory=lambda: [
            "mean_logprob",
            "min_logprob",
            "p10_logprob",
            "arg_mean_logprob",
            "arg_min_logprob",
            "first_tool_token_entropy",
            "n_tokens",
            "n_tool_calls",
            "has_tool_call",
            "agreement",
            "cluster_prior",
            "turn_idx",
        ]
    )
    #: Escalation-rate drift above the calibrated value by this many points raises a gateway alert.
    drift_alert_pp: float = 10.0


class RouterConfig(StrictModel):
    lambda_per_usd: float = 20.0
    floor: float = Field(0.55, ge=0.0, le=1.0)
    explore_cap: float = Field(0.10, ge=0.0, le=1.0)
    decay: float = Field(0.995, gt=0.0, le=1.0)
    seed: int = 0


class ServeConfig(StrictModel):
    vllm_url: str = "http://vllm:8000"
    quantization: Literal["fp8", "awq", "gptq"] | None = "fp8"
    gpu_usd_per_hour: float = 1.20
    guided_json: bool = True
    tool_call_parser: str | None = None
    max_model_len: int = 16384
    #: Share of student-routed requests served by the canary adapter, for a live A/B.
    canary_share: float = Field(0.0, ge=0.0, le=1.0)
    host: str = "0.0.0.0"
    port: int = 8710


class ProjectConfig(StrictModel):
    """The whole project. Loaded once per CLI invocation and threaded through every stage."""

    name: str
    registry: str = "sqlite:///.agentdistill/registry.db"
    artifacts: str = "./artifacts"
    reports: str = "./reports"
    sources: list[Source] = Field(default_factory=list)
    teacher: TeacherConfig | None = None
    curate: CurateConfig = Field(default_factory=CurateConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    train: TrainConfig | None = None
    onpolicy: OnPolicyConfig = Field(default_factory=OnPolicyConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    cascade: CascadeConfig = Field(default_factory=CascadeConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)

    #: Set by `load`, not by the file: where the config was read from, so relative paths resolve predictably.
    source_path: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _teacher_filter_needs_models(self) -> ProjectConfig:
        if "teacher" in self.curate.filters and not self.curate.teacher_models:
            raise ValueError("the `teacher` filter is enabled but curate.teacher_models is empty")
        return self

    @model_validator(mode="after")
    def _seq_len_agrees(self) -> ProjectConfig:
        if self.train is not None and self.train.max_seq_len != self.dataset.max_seq_len:
            raise ValueError(
                f"train.max_seq_len ({self.train.max_seq_len}) differs from dataset.max_seq_len "
                f"({self.dataset.max_seq_len}); samples built at one length cannot be trained at another"
            )
        return self

    @classmethod
    def load(cls, path: str | Path = "project.yaml") -> ProjectConfig:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"no config at {p}; run `agentdistill init` to write a starter project.yaml")
        with p.open() as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{p} must contain a YAML mapping at the top level")
        cfg = cls.model_validate(raw)
        cfg.source_path = p.resolve()
        return cfg

    @property
    def root(self) -> Path:
        """Directory that relative paths in the config resolve against."""
        return self.source_path.parent if self.source_path else Path.cwd()

    def resolve(self, path: str | Path) -> Path:
        """Resolve a config-relative path. Absolute paths and URLs pass through unchanged."""
        p = Path(path)
        if p.is_absolute() or "://" in str(path):
            return p
        return (self.root / p).resolve()

    @property
    def artifacts_dir(self) -> Path:
        return self.resolve(self.artifacts)

    @property
    def reports_dir(self) -> Path:
        return self.resolve(self.reports)

    def curation_fingerprint(self) -> dict[str, Any]:
        """The subset of config that determines which traces survive curation.

        This is what goes into a dataset's `filter_config`, and re-running curate with the same fingerprint over the
        same traces must produce the same `content_hash`. Anything that does not affect selection stays out, so that
        editing, say, the GPU price does not invalidate a dataset.
        """
        c = self.curate
        fp: dict[str, Any] = {
            "filters": c.ordered_filters(),
            "near_dedupe_threshold": c.near_dedupe_threshold,
            "near_dedupe_num_perm": c.near_dedupe_num_perm,
            "near_dedupe_shingle_n": c.near_dedupe_shingle_n,
            "near_dedupe_normalize_literals": c.near_dedupe_normalize_literals,
            "min_turns": c.min_turns,
            "max_turns": c.max_turns,
            "max_consecutive_tool_errors": c.max_consecutive_tool_errors,
            "max_repeated_identical_calls": c.max_repeated_identical_calls,
            "decontaminate_ngram": c.decontaminate_ngram,
            "decontaminate_overlap": c.decontaminate_overlap,
            "clusters": c.clusters,
            "cap_per_cluster": c.cap_per_cluster,
            "teacher_models": sorted(c.teacher_models),
            "quality_judge_min_score": c.quality_judge_min_score,
            "embeddings": c.embeddings.model_dump(),
        }
        return fp

    def curation_fingerprint_json(self) -> str:
        return json.dumps(self.curation_fingerprint(), sort_keys=True, separators=(",", ":"))

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.eval.metrics import DEFAULT_KS, metric_names
from src.models import PipelineConfig

FROZEN_KEYS = ("alpha", "rrf_k", "adaptive_beta", "rerank_n")
DEFAULT_FROZEN = {"alpha": 0.5, "rrf_k": 60, "adaptive_beta": 0.3, "rerank_n": 30}
DEFAULT_TUNE_GRID = {
    "alpha": [round(0.1 * step, 1) for step in range(11)],
    "rrf_k": [10, 20, 40, 60, 100],
    "adaptive_beta": [0.1, 0.2, 0.3, 0.5],
    "rerank_n": [10, 20, 30, 50],
}
PIPELINE_KEYS = {field.name for field in dataclasses.fields(PipelineConfig)}
EXTRA_KEYS = {"index_dir", "qrels"}


@dataclass(frozen=True)
class ConfigEntry:
    name: str
    pipeline: PipelineConfig
    index_dir: Path | None
    remap_qrels: bool


@dataclass(frozen=True)
class ExperimentSpec:
    path: Path
    raw: dict
    bench_dir: Path
    index_dir: Path | None
    runs_dir: Path
    seed: int
    ks: tuple[int, ...]
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    warmup: int
    defaults: dict
    configs: dict[str, dict]
    comparisons: dict[str, list[tuple[str, str]]]
    error_analysis: dict
    tune_grid: dict

    @property
    def frozen_path(self) -> Path:
        return self.bench_dir / "frozen_params.yaml"


def _expand(value, settings) -> Path | None:
    if value in (None, ""):
        return None
    text = str(value).replace("${DATA_DIR}", str(settings.data_dir)).replace("${RUNS_DIR}", str(settings.runs_dir))
    return Path(text)


def load_spec(path, settings) -> ExperimentSpec:
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    configs = {name: dict(values or {}) for name, values in (raw.get("configs") or {}).items()}
    if not configs:
        raise ValueError("configs must be a non-empty mapping")
    for name, values in configs.items():
        unknown = set(values) - PIPELINE_KEYS - EXTRA_KEYS
        if unknown:
            raise ValueError(f"Config {name} has unknown keys: {sorted(unknown)}")
        if values.get("qrels", "default") not in ("default", "remap"):
            raise ValueError(f"Config {name}: qrels must be 'default' or 'remap'")
        if values.get("index_dir"):
            values["index_dir"] = str(_expand(values["index_dir"], settings))
    ks = tuple(int(k) for k in raw.get("ks", DEFAULT_KS))
    available = metric_names(ks)
    primary = raw.get("primary_metric", "mrr@10")
    secondary = tuple(raw.get("secondary_metrics", ["ndcg@10", "recall@5"]))
    for metric in (primary, *secondary):
        if metric not in available:
            raise ValueError(f"Unknown metric {metric}; available: {available}")
    comparisons = {family: [tuple(pair) for pair in pairs] for family, pairs in (raw.get("comparisons") or {}).items()}
    for family, pairs in comparisons.items():
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError(f"Comparison {family} must list [system, baseline] pairs")
            for name in pair:
                if name != "best_single" and name not in configs:
                    raise ValueError(f"Comparison {family} refers to unknown config {name}")
    error_analysis = {"target": "C4-WS", "k": 10, "sample": 50, **(raw.get("error_analysis") or {})}
    if error_analysis["target"] not in configs:
        raise ValueError(f"error_analysis target {error_analysis['target']} is not a config")
    return ExperimentSpec(
        path=source,
        raw=raw,
        bench_dir=_expand(raw.get("bench_dir", "${DATA_DIR}/benchmark"), settings),
        index_dir=_expand(raw.get("index_dir"), settings),
        runs_dir=_expand(raw.get("runs_dir", "${RUNS_DIR}"), settings),
        seed=int(raw.get("seed", 42)),
        ks=ks,
        primary_metric=primary,
        secondary_metrics=secondary,
        warmup=int((raw.get("latency") or {}).get("warmup", 5)),
        defaults=dict(raw.get("defaults") or {}),
        configs=configs,
        comparisons=comparisons,
        error_analysis=error_analysis,
        tune_grid={**DEFAULT_TUNE_GRID, **(raw.get("tune") or {})},
    )


def resolve_configs(spec: ExperimentSpec, frozen: dict | None) -> dict[str, ConfigEntry]:
    source = {**DEFAULT_FROZEN, **(frozen or {})}
    entries = {}
    for name, values in spec.configs.items():
        merged = {**spec.defaults, **values}
        uses_frozen = [key for key, value in merged.items() if value == "frozen"]
        invalid = [key for key in uses_frozen if key not in FROZEN_KEYS]
        if invalid:
            raise ValueError(f"Config {name}: only {FROZEN_KEYS} may be 'frozen', not {invalid}")
        if uses_frozen and frozen is None:
            raise FileNotFoundError(
                f"Config {name} uses frozen values {uses_frozen}; run `python -m src.cli eval tune` on the dev split first"
            )
        pipeline_values = {
            key: source[key] if value == "frozen" else value for key, value in merged.items() if key in PIPELINE_KEYS
        }
        entries[name] = ConfigEntry(
            name=name,
            pipeline=PipelineConfig.from_dict(pipeline_values),
            index_dir=Path(merged["index_dir"]) if merged.get("index_dir") else None,
            remap_qrels=merged.get("qrels", "default") == "remap",
        )
    return entries


def read_frozen(spec: ExperimentSpec) -> dict | None:
    if not spec.frozen_path.exists():
        return None
    return yaml.safe_load(spec.frozen_path.read_text(encoding="utf-8")) or {}

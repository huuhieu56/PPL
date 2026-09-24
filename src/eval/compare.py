from pathlib import Path

import numpy as np

from src.eval.stats import holm_adjust, paired_bootstrap_ci, paired_randomization_test
from src.io_utils import read_csv, write_csv

COLUMNS = [
    "family", "system", "baseline", "metric", "category", "n", "mean_system", "mean_baseline",
    "diff", "ci_low", "ci_high", "p_value", "p_holm",
]


def _per_query(run_dir: Path) -> dict[str, dict[str, dict]]:
    data: dict[str, dict[str, dict]] = {}
    for row in read_csv(run_dir / "metrics_per_query.csv"):
        values = {key: (value if key in ("config", "query_id", "category", "origin") else float(value)) for key, value in row.items()}
        data.setdefault(row["config"], {})[row["query_id"]] = values
    return data


def compare_run(run_dir, spec, frozen: dict | None, samples: int = 10000) -> list[dict]:
    directory = Path(run_dir)
    data = _per_query(directory)
    best_single = (frozen or {}).get("best_single")
    metrics = (spec.primary_metric, *spec.secondary_metrics)
    rows: list[dict] = []
    for family, pairs in spec.comparisons.items():
        family_rows: list[dict] = []
        for system, baseline in pairs:
            names = [best_single if name == "best_single" else name for name in (system, baseline)]
            if None in names:
                raise ValueError("Comparisons with best_single need frozen_params.yaml (run `eval tune`)")
            missing = [name for name in names if name not in data]
            if missing:
                raise ValueError(f"Run {directory.name} lacks configs {missing}; rerun without --only")
            system_name, baseline_name = names
            shared = sorted(set(data[system_name]) & set(data[baseline_name]))
            categories = ["all", *sorted({data[system_name][query]["category"] for query in shared})]
            for metric in metrics:
                for category in categories:
                    ids = [query for query in shared if category == "all" or data[system_name][query]["category"] == category]
                    left = [data[system_name][query][metric] for query in ids]
                    right = [data[baseline_name][query][metric] for query in ids]
                    low, high = paired_bootstrap_ci(left, right, seed=spec.seed, samples=samples)
                    family_rows.append(
                        {
                            "family": family, "system": system_name, "baseline": baseline_name, "metric": metric,
                            "category": category, "n": len(ids), "mean_system": float(np.mean(left)),
                            "mean_baseline": float(np.mean(right)), "diff": float(np.mean(left) - np.mean(right)),
                            "ci_low": low, "ci_high": high,
                            "p_value": paired_randomization_test(left, right, seed=spec.seed, permutations=samples),
                            "p_holm": "",
                        }
                    )
        adjusted = holm_adjust({position: row["p_value"] for position, row in enumerate(family_rows) if row["category"] == "all"})
        for position, value in adjusted.items():
            family_rows[position]["p_holm"] = value
        rows.extend(family_rows)
    write_csv(directory / "comparisons.csv", rows, COLUMNS)
    return rows

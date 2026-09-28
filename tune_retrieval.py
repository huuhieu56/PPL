"""Select fusion hyperparameters using dev judgments only."""

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.config import load_yaml
from src.evaluation import evaluate_rankings
from src.experiments import select_queries, validate_config
from src.retrieval import RetrievalIndex, adaptive_alpha, fuse_weighted, self_idf


def tune(config_path: Path, experiment: str, alphas: list[float], betas: list[float]) -> Path:
    config = load_yaml(config_path)
    validate_config(config, require_index=True)
    queries = select_queries(Path(config["queries"]), "dev")
    selected = config["experiments"][experiment]
    if selected["method"] not in {"weighted", "adaptive"} or selected.get("use_reranker"):
        raise ValueError("Tune weighted/adaptive fusion without reranker; evaluate reranker separately")
    if not alphas or not betas or any(not 0 <= alpha <= 1 for alpha in alphas) or any(not -1 <= beta <= 1 for beta in betas):
        raise ValueError("Invalid alpha/beta grid")
    if selected["method"] == "weighted":
        betas = [0.0]
    index = RetrievalIndex.load(config["index_dir"])
    idf = self_idf(index)
    grid = list(itertools.product(sorted(set(alphas)), sorted(set(betas))))
    rankings = {parameters: {} for parameters in grid}
    limit = int(config.get("retrieval", {}).get("top_l", 100))
    for query in queries:
        sparse = index.sparse_scores(query["text"], limit)
        dense = index.dense_scores(query["text"], limit)
        for alpha, beta in grid:
            weight = adaptive_alpha(query["text"], alpha, beta, idf)[0] if selected["method"] == "adaptive" else alpha
            rankings[alpha, beta][query["query_id"]] = [result.chunk.chunk_id for result in fuse_weighted(sparse, dense, index.chunks, weight)]
    dev_ids = {query["query_id"] for query in queries}
    qrels = {}
    for line in Path(config["qrels"]).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["query_id"] in dev_ids:
                qrels.setdefault(row["query_id"], {})[row["chunk_id"]] = int(row["relevance"])
    metrics = {parameters: evaluate_rankings(ranking, qrels) for parameters, ranking in rankings.items()}
    best = max(grid, key=lambda parameters: (metrics[parameters]["ndcg@10"], -abs(parameters[1]), -parameters[0]))
    directory = Path(config.get("runs_dir", "runs")) / ("tuning-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    directory.mkdir(parents=True)
    config["experiments"][experiment]["alpha"] = best[0]
    if selected["method"] == "adaptive":
        config["experiments"][experiment]["adaptive_beta"] = best[1]
    (directory / "selected_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    audit = {"experiment": experiment, "selection_split": "dev", "selection_metric": "ndcg@10", "tie_break": "smallest abs(beta), then alpha", "query_ids": sorted(dev_ids), "best": {"alpha": best[0], "beta": best[1]}, "trials": [{"alpha": parameters[0], "beta": parameters[1], "metrics": scores} for parameters, scores in metrics.items()], "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [config_path, Path(config["queries"]), Path(config["qrels"]), *sorted(Path(config["index_dir"]).glob("*"))] if path.is_file()}}
    (directory / "dev_selection.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--experiment", choices=("E3", "E6"), default="E3")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1])
    parser.add_argument("--betas", type=float, nargs="+", default=[-0.5, -0.25, 0, 0.25, 0.5])
    args = parser.parse_args()
    print(tune(args.config, args.experiment, args.alphas, args.betas))


if __name__ == "__main__":
    main()

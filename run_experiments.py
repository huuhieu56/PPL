import argparse
import json
from pathlib import Path

from src.config import load_yaml
from src.experiments import run_experiment, validate_config, select_queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval experiments E0-E7")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if arguments.dry_run:
        config = load_yaml(arguments.config)
        validate_config(config, require_index=True)
        queries = select_queries(Path(config["queries"]), arguments.split)
        print(json.dumps({"status": "valid", "split": arguments.split, "queries": len(queries), "experiments": list(config["experiments"])}))
        return
    run_dir = run_experiment(arguments.config, arguments.resume, arguments.split)
    print(run_dir)


if __name__ == "__main__":
    main()

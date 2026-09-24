import json
from datetime import datetime, timezone
from pathlib import Path

from src.io_utils import sha256_file

MANIFEST_NAME = "benchmark_manifest.json"
TRACKED_FILES = ("queries.jsonl", "qrels.jsonl", "evidence.jsonl")


def _file_hashes(bench_dir: Path) -> dict[str, str]:
    return {name: sha256_file(bench_dir / name) for name in TRACKED_FILES if (bench_dir / name).exists()}


def write_manifest(bench_dir, index_version: str, seed: int, dev_ratio: float) -> dict:
    directory = Path(bench_dir)
    manifest = {
        "index_version": index_version,
        "seed": seed,
        "dev_ratio": dev_ratio,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": _file_hashes(directory),
    }
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def check_test_lock(bench_dir, frozen_params_path) -> dict:
    directory = Path(bench_dir)
    frozen = Path(frozen_params_path)
    if not frozen.exists():
        raise FileNotFoundError(f"{frozen} not found; run `python -m src.cli eval tune` on the dev split first")
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found; run `python -m src.cli bench split` first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = sha256_file(frozen)
    if "test_lock" not in manifest:
        manifest["test_lock"] = {"frozen_params_sha256": digest, "locked_at": datetime.now(timezone.utc).isoformat()}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "lock_violation": manifest["test_lock"]["frozen_params_sha256"] != digest,
        "benchmark_changed": _file_hashes(directory) != manifest["files"],
        "frozen_params_sha256": digest,
    }

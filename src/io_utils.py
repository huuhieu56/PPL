import csv
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path | str) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path | str, rows: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def read_csv(path: Path | str) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return [
            {key: (value or "") for key, value in row.items()}
            for row in csv.DictReader(stream)
            if any((value or "").strip() for value in row.values())
        ]


def write_csv(path: Path | str, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or (list(rows[0]) if rows else [])
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

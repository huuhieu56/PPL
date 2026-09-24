import argparse
import json
import sys

from src.config import load_settings, load_yaml
from src.index import RetrievalIndex
from src.indexing import build_index_from_folder
from src.storage import Database


def _context():
    settings = load_settings()
    database = Database(settings.db_path)
    database.initialize()
    return settings, database


def _print(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _index_build(args) -> int:
    settings, database = _context()
    defaults = load_yaml("configs/default.yaml")
    chunking = {**defaults["chunking"], "strategy": args.strategy, "prefix": not args.no_prefix}
    index_options = {**defaults["index"], "dense_backend": args.dense_backend}
    if args.tokenizers:
        index_options["tokenizers"] = [mode.strip() for mode in args.tokenizers.split(",") if mode.strip()]
    if args.embedding_model:
        index_options["embedding_model"] = args.embedding_model
    version, index_dir = build_index_from_folder(
        args.input,
        course=args.course,
        settings=settings,
        database=database,
        chunking=chunking,
        index_options=index_options,
        metadata_csv=args.metadata,
        activate=not args.no_activate,
    )
    _print({"version_id": version, "index_dir": str(index_dir)})
    return 0


def _index_add_tokenizer(args) -> int:
    RetrievalIndex.load(args.index).add_tokenizer(args.tokenizer)
    _print({"index_dir": args.index, "tokenizer": args.tokenizer})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    groups = parser.add_subparsers(dest="group", required=True)

    index = groups.add_parser("index").add_subparsers(dest="command", required=True)
    build = index.add_parser("build")
    build.add_argument("--input", required=True)
    build.add_argument("--course", required=True)
    build.add_argument("--metadata")
    build.add_argument("--strategy", choices=["structure", "fixed"], default="structure")
    build.add_argument("--no-prefix", action="store_true")
    build.add_argument("--tokenizers")
    build.add_argument("--embedding-model")
    build.add_argument("--dense-backend", choices=["numpy", "faiss"], default="numpy")
    build.add_argument("--no-activate", action="store_true")
    build.set_defaults(handler=_index_build)
    add_tokenizer = index.add_parser("add-tokenizer")
    add_tokenizer.add_argument("--index", required=True)
    add_tokenizer.add_argument("--tokenizer", required=True, choices=["whitespace", "pyvi", "vncorenlp"])
    add_tokenizer.set_defaults(handler=_index_add_tokenizer)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

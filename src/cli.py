import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI

from src.bench.agreement import agreement_report, merge_labels, read_annotations, write_disagreements
from src.bench.describe import describe_benchmark
from src.bench.generate import CATEGORIES, generate_drafts
from src.bench.manifest import write_manifest
from src.bench.pool import DEFAULT_POOL_SYSTEMS, build_pool, write_pool
from src.bench.remap import remap_qrels
from src.bench.review import export_review, import_review
from src.bench.split import split_queries
from src.cache import RetrievalCache
from src.config import load_settings, load_yaml
from src.index import RetrievalIndex, load_encoder
from src.indexing import build_index_from_folder, resolve_index_dir
from src.io_utils import read_csv, read_jsonl, write_jsonl
from src.pipeline import RetrievalPipeline
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


def _bench_context(args):
    settings, database = _context()
    bench = Path(args.bench) if args.bench else settings.data_dir / "benchmark"
    bench.mkdir(parents=True, exist_ok=True)
    index = RetrievalIndex.load(resolve_index_dir(args.index, settings, database))
    return settings, bench, index


def _bench_generate(args) -> int:
    settings, bench, index = _bench_context(args)
    model = args.model or settings.openai_model
    if not settings.openai_api_key or not model:
        raise SystemExit("OPENAI_API_KEY and OPENAI_MODEL (or --model) are required")
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    categories = tuple(args.categories.split(",")) if args.categories else CATEGORIES
    accepted, rejected = generate_drafts(index, client, model, args.per_category, args.seed, categories)
    write_jsonl(bench / "drafts.jsonl", accepted)
    write_jsonl(bench / "rejected.jsonl", rejected)
    _print({"accepted": len(accepted), "rejected": len(rejected), "bench": str(bench)})
    return 0


def _bench_review_export(args) -> int:
    _, bench, index = _bench_context(args)
    export_review(read_jsonl(bench / "drafts.jsonl"), index, bench / "review.csv")
    _print({"review": str(bench / "review.csv")})
    return 0


def _bench_review_import(args) -> int:
    _, bench, index = _bench_context(args)
    queries, duplicates = import_review(
        bench / "review.csv", read_jsonl(bench / "drafts.jsonl"), load_encoder(index.embedding_model), args.human
    )
    write_jsonl(bench / "queries.jsonl", queries)
    write_jsonl(bench / "duplicates.jsonl", duplicates)
    _print({"queries": len(queries), "duplicates": len(duplicates)})
    return 0


def _bench_pool(args) -> int:
    settings, bench, index = _bench_context(args)
    pipeline = RetrievalPipeline(index, cache=RetrievalCache(settings.cache_path))
    queries = read_jsonl(bench / "queries.jsonl")
    manual = read_csv(args.manual) if args.manual else None
    pool = build_pool(queries, pipeline, DEFAULT_POOL_SYSTEMS, args.depth, manual)
    paths = write_pool(pool, queries, index, bench, tuple(args.annotators.split(",")), args.seed)
    _print({"pool_entries": len(pool), "annotation_files": [str(path) for path in paths]})
    return 0


def _bench_agreement(args) -> int:
    _, bench, index = _bench_context(args)
    pool = json.loads((bench / "pool_map.json").read_text(encoding="utf-8"))
    annotations = [read_annotations(path) for path in args.annotations]
    report = agreement_report(annotations[0], annotations[1]) if len(annotations) >= 2 else {"overlap": 0, "kappa": None, "raw_agreement": None, "disagreements": []}
    (bench / "agreement.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if len(annotations) >= 2:
        write_disagreements(bench / "disagreements.csv", report, pool, annotations[0], annotations[1], read_jsonl(bench / "queries.jsonl"), index)
    try:
        qrels, evidence, warnings = merge_labels(pool, annotations, index, args.resolved)
    except ValueError as error:
        _print({"agreement": report, "error": str(error), "next": f"Điền cột 'final' trong {bench / 'disagreements.csv'} rồi chạy lại với --resolved"})
        return 1
    write_jsonl(bench / "qrels.jsonl", qrels)
    write_jsonl(bench / "evidence.jsonl", evidence)
    _print({"agreement": report, "qrels": len(qrels), "evidence": len(evidence), "warnings": warnings})
    return 0


def _bench_split(args) -> int:
    _, bench, index = _bench_context(args)
    queries, dropped = split_queries(read_jsonl(bench / "queries.jsonl"), read_jsonl(bench / "qrels.jsonl"), args.dev, args.seed)
    write_jsonl(bench / "queries.jsonl", queries)
    manifest = write_manifest(bench, index.version, args.seed, args.dev)
    _print({"queries": len(queries), "dropped_without_relevant_chunks": dropped, "manifest": manifest})
    return 0


def _bench_describe(args) -> int:
    _, bench, _ = _bench_context(args)
    pool_path, agreement_path = bench / "pool_map.json", bench / "agreement.json"
    report = describe_benchmark(
        read_jsonl(bench / "queries.jsonl"),
        read_jsonl(bench / "qrels.jsonl"),
        json.loads(pool_path.read_text(encoding="utf-8")) if pool_path.exists() else None,
        json.loads(agreement_path.read_text(encoding="utf-8")) if agreement_path.exists() else None,
    )
    (bench / "benchmark_description.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print(report["totals"])
    return 0


def _bench_remap(args) -> int:
    _, bench, _ = _bench_context(args)
    target = RetrievalIndex.load(args.target_index)
    qrels = remap_qrels(read_jsonl(bench / "evidence.jsonl"), target.chunk_list)
    write_jsonl(args.out, qrels)
    _print({"target_index": target.version, "qrels": len(qrels), "out": args.out})
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
    bench = groups.add_parser("bench").add_subparsers(dest="command", required=True)

    def bench_command(name, handler):
        command = bench.add_parser(name)
        command.add_argument("--bench")
        command.add_argument("--index")
        command.set_defaults(handler=handler)
        return command

    generate = bench_command("generate", _bench_generate)
    generate.add_argument("--per-category", type=int, default=60)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--model")
    generate.add_argument("--categories")
    bench_command("review-export", _bench_review_export)
    bench_command("review-import", _bench_review_import).add_argument("--human")
    pool = bench_command("pool", _bench_pool)
    pool.add_argument("--depth", type=int, default=15)
    pool.add_argument("--annotators", default="A,B")
    pool.add_argument("--manual")
    pool.add_argument("--seed", type=int, default=42)
    agreement = bench_command("agreement", _bench_agreement)
    agreement.add_argument("--annotations", nargs="+", required=True)
    agreement.add_argument("--resolved")
    split = bench_command("split", _bench_split)
    split.add_argument("--dev", type=float, default=0.3)
    split.add_argument("--seed", type=int, default=42)
    bench_command("describe", _bench_describe)
    remap = bench_command("remap", _bench_remap)
    remap.add_argument("--target-index", required=True)
    remap.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

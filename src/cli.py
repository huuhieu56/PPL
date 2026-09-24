import argparse
import json
import sys
from pathlib import Path

import yaml
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
from src.eval.compare import compare_run
from src.eval.errors import classify_failures, export_error_sample, summarize_causes
from src.eval.report import build_report
from src.eval.runner import default_pipeline_factory, load_benchmark, run_evaluation
from src.eval.spec import DEFAULT_FROZEN, load_spec, read_frozen, resolve_configs
from src.eval.tune import tune
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


def _eval_context(args):
    settings, database = _context()
    spec = load_spec(args.config, settings)
    index_dir = spec.index_dir or resolve_index_dir(None, settings, database)
    return settings, spec, index_dir


def _eval_tune(args) -> int:
    settings, spec, index_dir = _eval_context(args)
    path = tune(spec, index_dir=index_dir, pipeline_factory=default_pipeline_factory(settings), force=args.force)
    _print({"frozen_params": str(path), **yaml.safe_load(path.read_text(encoding="utf-8"))})
    return 0


def _eval_run(args) -> int:
    settings, spec, index_dir = _eval_context(args)
    frozen = read_frozen(spec)
    if args.dry_run:
        entries = resolve_configs(spec, frozen or DEFAULT_FROZEN)
        warnings = []
        for name, entry in entries.items():
            directory = entry.index_dir or index_dir
            meta_path = directory / "index_meta.json"
            if not meta_path.exists():
                warnings.append(f"{name}: index not built at {directory}")
            elif entry.pipeline.sparse and entry.pipeline.tokenizer not in json.loads(meta_path.read_text(encoding="utf-8"))["tokenizers"]:
                warnings.append(f"{name}: tokenizer {entry.pipeline.tokenizer} missing in {directory}")
        queries, _, _ = load_benchmark(spec.bench_dir, args.split)
        _print({"status": "valid", "split": args.split, "queries": len(queries), "frozen": frozen is not None,
                "configs": list(entries), "warnings": warnings})
        return 0
    only = [name.strip() for name in args.only.split(",")] if args.only else None
    run_dir = run_evaluation(spec, args.split, index_dir=index_dir, pipeline_factory=default_pipeline_factory(settings),
                             frozen=frozen, resume_run_id=args.resume, only=only, latency=not args.no_latency)
    _print({"run_dir": str(run_dir)})
    return 0


def _eval_compare(args) -> int:
    _, spec, _ = _eval_context(args)
    rows = compare_run(args.run, spec, read_frozen(spec))
    _print({"comparisons": len(rows), "file": str(Path(args.run) / "comparisons.csv")})
    return 0


def _eval_errors(args) -> int:
    _, spec, _ = _eval_context(args)
    run_dir = Path(args.run)
    if args.summarize:
        summary = summarize_causes(args.summarize)
        (run_dir / "errors_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        _print(summary)
        return 0
    target = spec.error_analysis["target"]
    record = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    index = RetrievalIndex.load(record["configs"][target]["index_dir"])
    failures = classify_failures(run_dir, target, int(spec.error_analysis["k"]))
    queries, _, _ = load_benchmark(spec.bench_dir, record["split"])
    rows = export_error_sample(failures, queries, index.chunks, run_dir / "error_sample.csv",
                               int(spec.error_analysis["sample"]), spec.seed)
    stages = {}
    for failure in failures:
        stages[failure["stage"]] = stages.get(failure["stage"], 0) + 1
    (run_dir / "errors_summary.json").write_text(
        json.dumps({"by_stage": stages, "rows": [{"stage": s, "cause": "unlabeled", "count": c} for s, c in sorted(stages.items())]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    _print({"failures": len(failures), "by_stage": stages, "sample": str(run_dir / "error_sample.csv"), "sample_rows": len(rows)})
    return 0


def _eval_report(args) -> int:
    _, spec, _ = _eval_context(args)
    written = build_report(args.run, spec.bench_dir, spec.primary_metric)
    _print({"written": [str(path) for path in written]})
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
    evaluation = groups.add_parser("eval").add_subparsers(dest="command", required=True)

    def eval_command(name, handler):
        command = evaluation.add_parser(name)
        command.add_argument("--config", default="configs/experiment.yaml")
        command.set_defaults(handler=handler)
        return command

    eval_command("tune", _eval_tune).add_argument("--force", action="store_true")
    run = eval_command("run", _eval_run)
    run.add_argument("--split", choices=["dev", "test", "all"], required=True)
    run.add_argument("--resume")
    run.add_argument("--only")
    run.add_argument("--no-latency", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    eval_command("compare", _eval_compare).add_argument("--run", required=True)
    errors = eval_command("errors", _eval_errors)
    errors.add_argument("--run", required=True)
    errors.add_argument("--summarize")
    eval_command("report", _eval_report).add_argument("--run", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

import json
from pathlib import Path

import yaml

from src.bench.manifest import write_manifest
from src.cli import main
from src.index import RetrievalIndex
from src.io_utils import write_jsonl
from tests.fakes import FakeCrossEncoder, FakeEncoder, make_chunk

TEXTS = {
    "c1": "Mã môn AI101 là học phần nhập môn trí tuệ nhân tạo",
    "c2": "Học máy là lĩnh vực nghiên cứu thuật toán học từ dữ liệu",
    "c3": "Cơ sở dữ liệu quan hệ lưu trữ bảng và khóa chính",
    "c4": "Mạng nơ ron sâu gồm nhiều tầng biến đổi phi tuyến",
}


def test_full_eval_flow_on_fake_models(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PPL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PPL_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr("src.index.load_encoder", lambda name: FakeEncoder())
    monkeypatch.setattr("src.reranking.load_cross_encoder", lambda name: FakeCrossEncoder())
    index = RetrievalIndex.build(
        [make_chunk(c, t) for c, t in TEXTS.items()], tmp_path / "idx",
        embedding_model="fake", tokenizers=("whitespace", "pyvi"), encoder=FakeEncoder(),
    )
    bench = tmp_path / "bench"
    queries = [
        {"query_id": f"q{i}", "text": text, "category": ("exact", "concept")[i % 2], "origin": "llm",
         "split": "dev" if i < 2 else "test", "source_chunk_ids": [cid], "evidence": []}
        for i, (cid, text) in enumerate(TEXTS.items())
    ]
    write_jsonl(bench / "queries.jsonl", queries)
    write_jsonl(bench / "qrels.jsonl", [
        {"query_id": q["query_id"], "chunk_id": q["source_chunk_ids"][0], "relevance": 2} for q in queries
    ])
    write_manifest(bench, index.version, 42, 0.5)
    raw = yaml.safe_load(Path("configs/experiment.yaml").read_text(encoding="utf-8"))
    raw.update(bench_dir=str(bench), index_dir=str(index.directory), latency={"warmup": 1},
               tune={"alpha": [0.3, 0.7], "rrf_k": [60], "adaptive_beta": [0.3], "rerank_n": [10]})
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    common = ["--config", str(config)]

    assert main(["eval", "run", *common, "--split", "test", "--dry-run"]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["frozen"] is False and "C4-WS" in dry["configs"] and dry["warnings"] == []

    assert main(["eval", "tune", *common]) == 0
    capsys.readouterr()
    assert main(["eval", "run", *common, "--split", "test"]) == 0
    run_dir = Path(json.loads(capsys.readouterr().out)["run_dir"])
    assert main(["eval", "compare", *common, "--run", str(run_dir)]) == 0
    assert main(["eval", "errors", *common, "--run", str(run_dir)]) == 0
    assert main(["eval", "report", *common, "--run", str(run_dir)]) == 0
    assert (run_dir / "report" / "table_3_5.md").exists()
    assert (run_dir / "report" / "table_3_7.md").exists()

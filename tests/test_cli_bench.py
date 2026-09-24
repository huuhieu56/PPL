import json

from src.cli import main
from src.index import RetrievalIndex
from src.io_utils import read_jsonl, write_csv, write_jsonl
from tests.fakes import FakeCrossEncoder, FakeEncoder, make_chunk

TEXTS = {
    "c1": "Mã môn AI101 là học phần nhập môn trí tuệ nhân tạo",
    "c2": "Học máy là lĩnh vực nghiên cứu thuật toán học từ dữ liệu",
    "c3": "Cơ sở dữ liệu quan hệ lưu trữ bảng và khóa chính",
    "c4": "Mạng nơ ron sâu gồm nhiều tầng biến đổi phi tuyến",
}


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("PPL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PPL_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr("src.index.load_encoder", lambda name: FakeEncoder())
    monkeypatch.setattr("src.reranking.load_cross_encoder", lambda name: FakeCrossEncoder())
    index = RetrievalIndex.build(
        [make_chunk(cid, text) for cid, text in TEXTS.items()], tmp_path / "idx", embedding_model="fake", encoder=FakeEncoder()
    )
    bench = tmp_path / "bench"
    queries = [
        {"query_id": f"q{i}", "text": text, "category": "exact" if i < 2 else "concept", "origin": "llm",
         "split": None, "source_chunk_ids": [cid], "evidence": [], "generator": "m"}
        for i, (cid, text) in enumerate(TEXTS.items())
    ]
    write_jsonl(bench / "queries.jsonl", queries)
    return index, bench


def test_pool_agreement_split_describe_remap_flow(tmp_path, monkeypatch):
    index, bench = _setup(tmp_path, monkeypatch)
    common = ["--bench", str(bench), "--index", str(index.directory)]

    assert main(["bench", "pool", *common, "--depth", "2", "--annotators", "A,B"]) == 0
    pool = json.loads((bench / "pool_map.json").read_text(encoding="utf-8"))
    labels = [{"pool_id": entry["pool_id"], "relevance": "2" if "source" in entry["systems"] else "0",
               "evidence_quote": index.chunks[entry["chunk_id"]].body.split(" là ")[0] if "source" in entry["systems"] else ""}
              for entry in pool]
    write_csv(bench / "annotation_A.csv", labels)
    write_csv(bench / "annotation_B.csv", labels)

    assert main(["bench", "agreement", *common, "--annotations", str(bench / "annotation_A.csv"), str(bench / "annotation_B.csv")]) == 0
    assert json.loads((bench / "agreement.json").read_text(encoding="utf-8"))["kappa"] == 1.0
    assert len(read_jsonl(bench / "evidence.jsonl")) >= 3

    assert main(["bench", "split", *common, "--dev", "0.5", "--seed", "1"]) == 0
    splits = [query["split"] for query in read_jsonl(bench / "queries.jsonl")]
    assert splits.count("dev") == 2 and splits.count("test") == 2
    assert (bench / "benchmark_manifest.json").exists()

    assert main(["bench", "describe", *common]) == 0
    description = json.loads((bench / "benchmark_description.json").read_text(encoding="utf-8"))
    assert description["totals"]["queries"] == 4

    assert main(["bench", "remap", *common, "--target-index", str(index.directory), "--out", str(tmp_path / "remapped.jsonl")]) == 0
    assert {row["chunk_id"] for row in read_jsonl(tmp_path / "remapped.jsonl")} >= {"c1", "c3"}


def test_agreement_with_open_disagreements_writes_no_qrels(tmp_path, monkeypatch, capsys):
    index, bench = _setup(tmp_path, monkeypatch)
    common = ["--bench", str(bench), "--index", str(index.directory)]
    main(["bench", "pool", *common, "--depth", "1"])
    pool = json.loads((bench / "pool_map.json").read_text(encoding="utf-8"))
    write_csv(bench / "annotation_A.csv", [{"pool_id": pool[0]["pool_id"], "relevance": "2", "evidence_quote": ""}])
    write_csv(bench / "annotation_B.csv", [{"pool_id": pool[0]["pool_id"], "relevance": "0", "evidence_quote": ""}])
    assert main(["bench", "agreement", *common, "--annotations", str(bench / "annotation_A.csv"), str(bench / "annotation_B.csv")]) == 1
    assert not (bench / "qrels.jsonl").exists()
    assert (bench / "disagreements.csv").exists()
    assert "disagreements.csv" in capsys.readouterr().out

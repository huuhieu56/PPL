# Kế hoạch 3/3: Bộ đánh giá, báo cáo Chương 3 và notebook Colab — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bộ đánh giá thực nghiệm theo mục 2.6 của báo cáo: metrics (đối chiếu `ranx`), kiểm định thống kê (bootstrap, randomization, Holm), cấu hình thí nghiệm YAML với tham số "frozen", tune trên dev, chạy test có khóa và đo độ trễ, so sánh RQ2/RQ3/X2, phân tích lỗi theo tầng, xuất bảng/hình đánh số khớp Chương 3; thêm khung Chương 3 vào báo cáo, notebook Colab và trang Experiments.

**Architecture:** Package `src/eval/` gồm module nhỏ: `metrics` → `stats` → `spec` (đọc YAML, resolve cấu hình) → `runner` (checkpoint, metrics, latency) → `tune` → `compare` → `errors` → `report`. CLI `python -m src.cli eval …` điều phối. Code cũ `src/experiments.py`, `src/evaluation.py`, `run_experiments.py` bị thay thế.

**Tech Stack:** Python 3.11, numpy, PyYAML, matplotlib (Agg), ranx (chỉ trong test), Streamlit, pytest.

**Spec:** `docs/superpowers/specs/2026-09-24-hybrid-retrieval-evaluation-design.md` (mục 7, 8, 9, 10). Yêu cầu Kế hoạch 1 và 2 đã hoàn tất.

## Global Constraints

- Mọi Global Constraints của Kế hoạch 1 và 2 vẫn áp dụng.
- Chỉ số chính **MRR@10** (khai báo trước); phụ **NDCG@10, Recall@5**; ks mặc định `(1, 3, 5, 10)`. NDCG dùng gain `2^rel − 1`; MRR@10 tính trên relevance ≥ 1.
- Lưới tune mặc định: α ∈ {0.0, 0.1, …, 1.0}; k ∈ {10, 20, 40, 60, 100}; β ∈ {0.1, 0.2, 0.3, 0.5}; N ∈ {10, 20, 30, 50}. Hòa điểm → chọn giá trị gần mặc định (α 0.5, k 60, β 0.3, N 30).
- Bootstrap **10.000** mẫu, randomization **10.000** hoán vị, seed từ spec (mặc định 42); CI 95%; Holm trong từng họ so sánh, chỉ trên hàng `category == "all"`.
- Latency: **5** truy vấn warm-up, đo không cache, báo mean/P50/P95 cho `sparse, dense, fusion, rerank, total`, kèm thông tin phần cứng.
- Error analysis: mục tiêu mặc định **C4-WS**, top-**10**, mẫu tối đa **50** phân tầng theo category. Nhãn nguyên nhân: `extraction, chunk_boundary, tokenization, vocabulary_mismatch, multi_hop, label_error, other`.
- Đánh số bảng/hình Chương 3 cố định (dùng chung cho `src/eval/report.py` và báo cáo): Bảng 3.1–3.11, Hình 3.1–3.3 (xem Task 8).
- Chương 3 chỉ chứa placeholder dạng `[[ĐIỀN: <mô tả> — nguồn: <lệnh / file>]]`; không bịa số liệu.
- Trên Colab: cài bằng `pip install -r requirements.in` (không dùng `requirements.txt` vì nó ghim torch CPU).

## Review Focus

1. **Chạy `eval run --split test` khi chưa tune** — phải báo lỗi rõ ràng hướng dẫn chạy `eval tune`, không tạo run rỗng. Pin: `test_test_split_requires_frozen_params` (Task 4).
2. **Colab bị ngắt giữa chừng rồi `--resume`** — không chạy lại truy vấn đã xong, metrics cuối cùng đủ mọi truy vấn. Pin: `test_run_resumes_without_repeating_completed_queries` (Task 4).
3. **`frozen_params.yaml` bị sửa sau lần chạy test đầu** — run vẫn chạy nhưng `config.json` có `lock.lock_violation = true` và trang Experiments hiện cảnh báo. Pin: `test_test_run_records_lock_violation` (Task 4), `test_experiments_page_warns_on_lock_violation` (Task 9).
4. **So sánh tham chiếu cấu hình không có trong run** (chạy với `--only`) — lỗi nêu tên cấu hình thiếu. Pin: `test_compare_reports_missing_configs` (Task 6).
5. **Chạy `eval report` trên run chưa có latency/comparisons/errors** — vẫn sinh các bảng có dữ liệu, bỏ qua bảng thiếu, không crash. Pin: `test_report_skips_missing_inputs` (Task 8).

---

## File Structure

| File | Trạng thái | Trách nhiệm |
|---|---|---|
| `src/eval/__init__.py` | tạo | trống |
| `src/eval/metrics.py` | tạo | `DEFAULT_KS`, `metric_names`, `query_metrics`, `evaluate_rankings` |
| `src/eval/stats.py` | tạo | `paired_bootstrap_ci`, `paired_randomization_test`, `holm_adjust` |
| `src/eval/spec.py` | tạo | `ExperimentSpec`, `ConfigEntry`, `load_spec`, `resolve_configs`, `read_frozen`, `DEFAULT_FROZEN` |
| `src/eval/runner.py` | tạo | `load_benchmark`, `default_pipeline_factory`, `run_evaluation`, `write_metrics`, `measure_latency`, `hardware_info` |
| `src/eval/tune.py` | tạo | `tune` |
| `src/eval/compare.py` | tạo | `compare_run` |
| `src/eval/errors.py` | tạo | `classify_failures`, `export_error_sample`, `summarize_causes` |
| `src/eval/report.py` | tạo | `TABLES`, `FIGURES`, `markdown_table`, `build_report` |
| `src/cli.py` | sửa | nhóm lệnh `eval` |
| `configs/experiment.yaml`, `configs/experiment.example.yaml` | tạo | ma trận C1–C4, X1–X6 |
| `examples/benchmark/queries.jsonl`, `examples/benchmark/qrels.jsonl` | tạo | benchmark ví dụ (định dạng mới) |
| `src/experiments.py`, `src/evaluation.py`, `run_experiments.py`, `configs/experiments.yaml`, `configs/example_experiments.yaml`, `data/benchmark/*.example.jsonl`, `tests/test_core.py` | xóa | thay bằng `src/eval/` |
| `run_rag_evaluation.py` | sửa | đọc spec mới |
| `pages/4_Experiments.py` | viết lại | liệt kê run, hiển thị report, cảnh báo lock |
| `b_o_c_o_nh_m_3.md` | sửa | thêm Chương 3 khung, cập nhật mục lục/danh mục, ghi chú Bảng 2.2 |
| `notebooks/colab_pipeline.ipynb` | tạo | quy trình Colab |
| `tests/test_eval_*.py`, `tests/test_notebook.py`, `tests/test_pages.py` | tạo/sửa | test |

---

### Task 1: Metrics theo truy vấn, đối chiếu `ranx`

**Files:**
- Create: `src/eval/__init__.py` (trống), `src/eval/metrics.py`, `tests/test_eval_metrics.py`
- Modify: `src/experiments.py` (đổi import `evaluate_rankings`)
- Delete: `src/evaluation.py`; hàm `test_ranking_metrics` trong `tests/test_core.py`

**Interfaces:**
- Produces:
  - `DEFAULT_KS = (1, 3, 5, 10)`
  - `metric_names(ks) -> list[str]` — `["mrr@10", "hit_rate@k", "precision@k", "recall@k", "ndcg@k", …]` theo thứ tự ks
  - `query_metrics(ranking: list[str], grades: dict[str, int], ks=DEFAULT_KS) -> dict[str, float]`
  - `evaluate_rankings(rankings: dict[str, list[str]], qrels: dict[str, dict[str, int]], ks=DEFAULT_KS) -> dict[str, float]` (trung bình; ValueError nếu rỗng)

- [ ] **Step 1: Viết test thất bại `tests/test_eval_metrics.py`**

```python
import random

import pytest
from ranx import Qrels, Run, evaluate

from src.eval.metrics import evaluate_rankings, metric_names, query_metrics


def test_query_metrics_on_hand_example():
    values = query_metrics(["x", "b", "a", "z"], {"a": 2, "b": 1, "z": 0}, ks=(1, 3))
    assert values["mrr@10"] == 0.5
    assert values["hit_rate@1"] == 0.0 and values["hit_rate@3"] == 1.0
    assert values["precision@3"] == pytest.approx(2 / 3)
    assert values["recall@3"] == 1.0
    assert list(values) == metric_names((1, 3))


def test_query_without_relevant_chunks_scores_zero():
    values = query_metrics(["a"], {}, ks=(1,))
    assert values == {"mrr@10": 0.0, "hit_rate@1": 0.0, "precision@1": 0.0, "recall@1": 0.0, "ndcg@1": 0.0}


def test_evaluate_rankings_rejects_empty_input():
    with pytest.raises(ValueError):
        evaluate_rankings({}, {})


def test_metrics_match_ranx_on_random_runs():
    rng = random.Random(7)
    documents = [f"d{i}" for i in range(30)]
    qrels, rankings = {}, {}
    for number in range(25):
        query = f"q{number}"
        qrels[query] = {doc: rng.choice([0, 1, 2]) for doc in rng.sample(documents, 6)}
        if not any(qrels[query].values()):
            qrels[query][documents[0]] = 1
        rankings[query] = rng.sample(documents, 15)
    ours = evaluate_rankings(rankings, qrels, ks=(1, 3, 5, 10))
    reference = evaluate(
        Qrels({q: {d: g for d, g in grades.items() if g > 0} for q, grades in qrels.items()}),
        Run({q: {d: float(len(ids) - i) for i, d in enumerate(ids)} for q, ids in rankings.items()}),
        ["mrr@10", "hit_rate@5", "precision@5", "recall@5", "ndcg_burges@10", "ndcg_burges@3"],
    )
    assert ours["mrr@10"] == pytest.approx(reference["mrr@10"])
    assert ours["hit_rate@5"] == pytest.approx(reference["hit_rate@5"])
    assert ours["precision@5"] == pytest.approx(reference["precision@5"])
    assert ours["recall@5"] == pytest.approx(reference["recall@5"])
    assert ours["ndcg@10"] == pytest.approx(reference["ndcg_burges@10"])
    assert ours["ndcg@3"] == pytest.approx(reference["ndcg_burges@3"])
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_eval_metrics.py -q`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.eval'`

- [ ] **Step 3: Viết `src/eval/__init__.py` (rỗng) và `src/eval/metrics.py`**

```python
import math

import numpy as np

DEFAULT_KS = (1, 3, 5, 10)


def metric_names(ks=DEFAULT_KS) -> list[str]:
    names = ["mrr@10"]
    for k in ks:
        names += [f"hit_rate@{k}", f"precision@{k}", f"recall@{k}", f"ndcg@{k}"]
    return names


def query_metrics(ranking: list[str], grades: dict[str, int], ks=DEFAULT_KS) -> dict[str, float]:
    relevant = {chunk_id for chunk_id, grade in grades.items() if grade > 0}
    first = next((rank for rank, chunk_id in enumerate(ranking, start=1) if chunk_id in relevant), None)
    values = {"mrr@10": 0.0 if first is None or first > 10 else 1.0 / first}
    ideal_grades = sorted(grades.values(), reverse=True)
    for k in ks:
        retrieved = ranking[:k]
        matched = sum(chunk_id in relevant for chunk_id in retrieved)
        dcg = sum(
            (2 ** grades.get(chunk_id, 0) - 1) / math.log2(rank + 1)
            for rank, chunk_id in enumerate(retrieved, start=1)
        )
        idcg = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal_grades[:k], start=1))
        values[f"hit_rate@{k}"] = float(matched > 0)
        values[f"precision@{k}"] = matched / k
        values[f"recall@{k}"] = matched / len(relevant) if relevant else 0.0
        values[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    return values


def evaluate_rankings(rankings: dict[str, list[str]], qrels: dict[str, dict[str, int]], ks=DEFAULT_KS) -> dict[str, float]:
    if not rankings:
        raise ValueError("rankings must not be empty")
    rows = [query_metrics(ranking, qrels.get(query_id, {}), ks) for query_id, ranking in rankings.items()]
    return {name: float(np.mean([row[name] for row in rows])) for name in rows[0]}
```

- [ ] **Step 4: Chuyển code cũ sang module mới**

- Trong `src/experiments.py` đổi `from src.evaluation import evaluate_rankings` thành `from src.eval.metrics import evaluate_rankings`.
- Xóa `src/evaluation.py`.
- Trong `tests/test_core.py` xóa hàm `test_ranking_metrics` và import `evaluate_rankings`.

- [ ] **Step 5: Chạy test, xác nhận pass**

Run: `.venv/Scripts/python -m pytest -q`
Expected: tất cả pass (`tests/test_eval_metrics.py`: 4 passed).

- [ ] **Step 6: Commit**

```bash
git add -A src/eval src/experiments.py src/evaluation.py tests/test_eval_metrics.py tests/test_core.py
git commit -m "feat: add per-query ranking metrics verified against ranx

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Kiểm định thống kê (`eval/stats.py`)

**Files:**
- Create: `src/eval/stats.py`, `tests/test_eval_stats.py`

**Interfaces:**
- Produces:
  - `paired_bootstrap_ci(left: list[float], right: list[float], seed: int = 42, samples: int = 10000, confidence: float = 0.95) -> tuple[float, float]` (CI của mean(left − right))
  - `paired_randomization_test(left, right, seed: int = 42, permutations: int = 10000) -> float` (p hai phía, `(count + 1) / (permutations + 1)`)
  - `holm_adjust(pvalues: dict) -> dict` (cùng khóa, giá trị đã điều chỉnh, đơn điệu, ≤ 1)

- [ ] **Step 1: Viết test thất bại `tests/test_eval_stats.py`**

```python
import pytest

from src.eval.stats import holm_adjust, paired_bootstrap_ci, paired_randomization_test


def test_identical_systems_have_zero_interval_and_p_one():
    values = [0.2, 0.5, 1.0, 0.0, 0.3]
    assert paired_bootstrap_ci(values, values, samples=500) == (0.0, 0.0)
    assert paired_randomization_test(values, values, permutations=500) == 1.0


def test_clear_improvement_is_significant_and_deterministic():
    better = [1.0] * 30
    worse = [0.0] * 25 + [1.0] * 5
    low, high = paired_bootstrap_ci(better, worse, seed=1, samples=2000)
    assert 0 < low <= high <= 1
    p = paired_randomization_test(better, worse, seed=1, permutations=2000)
    assert p < 0.01
    assert p == paired_randomization_test(better, worse, seed=1, permutations=2000)


def test_stats_reject_mismatched_inputs():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        paired_randomization_test([], [])


def test_holm_adjustment_matches_hand_computation():
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    assert holm_adjust({"x": 0.6, "y": 0.9}) == pytest.approx({"x": 1.0, "y": 1.0})
    assert holm_adjust({}) == {}
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_eval_stats.py -q`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.eval.stats'`

- [ ] **Step 3: Viết `src/eval/stats.py`**

```python
import numpy as np


def _differences(left, right) -> np.ndarray:
    if len(left) != len(right) or not len(left):
        raise ValueError("paired samples must have the same non-zero length")
    return np.asarray(left, dtype=float) - np.asarray(right, dtype=float)


def paired_bootstrap_ci(left, right, seed: int = 42, samples: int = 10000, confidence: float = 0.95) -> tuple[float, float]:
    differences = _differences(left, right)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    means = differences[indices].mean(axis=1)
    tail = (1 - confidence) / 2 * 100
    return float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))


def paired_randomization_test(left, right, seed: int = 42, permutations: int = 10000) -> float:
    differences = _differences(left, right)
    observed = abs(differences.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(permutations, len(differences)))
    permuted = np.abs((signs * differences).mean(axis=1))
    extreme = int(np.sum(permuted >= observed - 1e-12))
    return (extreme + 1) / (permutations + 1)


def holm_adjust(pvalues: dict) -> dict:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted = {}
    running = 0.0
    for position, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - position) * value))
        adjusted[key] = running
    return adjusted
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `.venv/Scripts/python -m pytest tests/test_eval_stats.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/eval/stats.py tests/test_eval_stats.py
git commit -m "feat: add paired bootstrap, randomization test and Holm correction

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Cấu hình thí nghiệm YAML (`eval/spec.py`)

**Files:**
- Create: `src/eval/spec.py`, `configs/experiment.yaml`, `configs/experiment.example.yaml`, `examples/benchmark/queries.jsonl`, `examples/benchmark/qrels.jsonl`, `tests/test_eval_spec.py`

**Interfaces:**
- Consumes: `PipelineConfig`, `metric_names`, `DEFAULT_KS`, `Settings`.
- Produces:
  - `FROZEN_KEYS = ("alpha", "rrf_k", "adaptive_beta", "rerank_n")`, `DEFAULT_FROZEN = {"alpha": 0.5, "rrf_k": 60, "adaptive_beta": 0.3, "rerank_n": 30}`, `DEFAULT_TUNE_GRID`
  - `ConfigEntry(name: str, pipeline: PipelineConfig, index_dir: Path | None, remap_qrels: bool)`
  - `ExperimentSpec(path, raw, bench_dir, index_dir, runs_dir, seed, ks, primary_metric, secondary_metrics, warmup, defaults, configs, comparisons, error_analysis, tune_grid)` với property `frozen_path -> Path` (`bench_dir / "frozen_params.yaml"`)
  - `load_spec(path, settings) -> ExperimentSpec` (thay `${DATA_DIR}`, `${RUNS_DIR}`; ValueError với khóa lạ, metric lạ, so sánh tới cấu hình không tồn tại, target error analysis không tồn tại)
  - `resolve_configs(spec, frozen: dict | None) -> dict[str, ConfigEntry]` (FileNotFoundError nếu cấu hình dùng `frozen` mà `frozen is None`)
  - `read_frozen(spec) -> dict | None`

- [ ] **Step 1: Tạo benchmark ví dụ**

`examples/benchmark/queries.jsonl`:
```
{"query_id": "q1", "text": "Mã môn AI101 là học phần gì?", "category": "exact", "origin": "human", "split": "test", "source_chunk_ids": [], "evidence": [], "generator": "human"}
{"query_id": "q2", "text": "Học máy là gì?", "category": "concept", "origin": "human", "split": "dev", "source_chunk_ids": [], "evidence": [], "generator": "human"}
```
`examples/benchmark/qrels.jsonl`:
```
{"query_id": "q1", "chunk_id": "replace-with-real-chunk-id", "relevance": 2}
{"query_id": "q2", "chunk_id": "replace-with-real-chunk-id", "relevance": 1}
```

- [ ] **Step 2: Tạo `configs/experiment.yaml`**

```yaml
# Ma trận thực nghiệm theo Bảng 2.2 (C1–C4) và các hướng khai thác X1–X6 (Chương 3, mục 3.6).
# "frozen" = lấy từ <bench_dir>/frozen_params.yaml do `python -m src.cli eval tune` ghi ra.
bench_dir: ${DATA_DIR}/benchmark
index_dir: null            # null = index của corpus đang active
runs_dir: ${RUNS_DIR}
seed: 42
ks: [1, 3, 5, 10]
primary_metric: mrr@10
secondary_metrics: [ndcg@10, recall@5]
latency:
  warmup: 5

defaults:
  top_l: 100
  context_k: 5
  tokenizer: whitespace
  reranker_model: BAAI/bge-reranker-v2-m3

configs:
  C1: {sparse: true, dense: false, fusion: none, rerank: false}
  C2: {sparse: false, dense: true, fusion: none, rerank: false}
  C3-RRF: {fusion: rrf, rrf_k: frozen, rerank: false}
  C3-WS: {fusion: weighted, alpha: frozen, rerank: false}
  C4-RRF: {fusion: rrf, rrf_k: frozen, rerank: true, rerank_n: frozen}
  C4-WS: {fusion: weighted, alpha: frozen, rerank: true, rerank_n: frozen}
  X1: {sparse: false, dense: true, fusion: none, rerank: true, rerank_n: frozen}
  X2: {fusion: adaptive, alpha: frozen, adaptive_beta: frozen, rerank: false}
  X2-R: {fusion: adaptive, alpha: frozen, adaptive_beta: frozen, rerank: true, rerank_n: frozen}
  X3-pyvi-C1: {sparse: true, dense: false, fusion: none, rerank: false, tokenizer: pyvi}
  X3-pyvi-C3-WS: {fusion: weighted, alpha: frozen, rerank: false, tokenizer: pyvi}
  X5-N10: {fusion: weighted, alpha: frozen, rerank: true, rerank_n: 10}
  X5-N20: {fusion: weighted, alpha: frozen, rerank: true, rerank_n: 20}
  X5-N50: {fusion: weighted, alpha: frozen, rerank: true, rerank_n: 50}
  # X3 với VnCoreNLP (cần Java; build trước: python -m src.cli index add-tokenizer --tokenizer vncorenlp --index <dir>)
  # X3-vncorenlp-C1: {sparse: true, dense: false, fusion: none, rerank: false, tokenizer: vncorenlp}
  # X4 (chunking): build index thứ hai với --strategy fixed hoặc --no-prefix --no-activate, rồi bỏ comment:
  # X4-fixed-C4-WS: {fusion: weighted, alpha: frozen, rerank: true, rerank_n: frozen, index_dir: ${DATA_DIR}/indexes/<version>, qrels: remap}
  # X6 (embedding): build index thứ hai với --embedding-model <model tiếng Việt> --no-activate, rồi bỏ comment:
  # X6-C2: {sparse: false, dense: true, fusion: none, rerank: false, index_dir: ${DATA_DIR}/indexes/<version>}

comparisons:
  RQ2: [[C3-RRF, best_single], [C3-WS, best_single]]
  RQ3: [[C4-RRF, C3-RRF], [C4-WS, C3-WS], [X1, C2]]
  X2: [[X2, C3-WS], [X2-R, C4-WS]]

error_analysis:
  target: C4-WS
  k: 10
  sample: 50
```

`configs/experiment.example.yaml`: sao chép nguyên `configs/experiment.yaml` rồi thay 3 dòng đầu thành:
```yaml
bench_dir: examples/benchmark
index_dir: examples/index
runs_dir: ${RUNS_DIR}
```

- [ ] **Step 3: Viết test thất bại `tests/test_eval_spec.py`**

```python
from pathlib import Path

import pytest
import yaml

from src.config import load_settings
from src.eval.spec import DEFAULT_FROZEN, load_spec, read_frozen, resolve_configs


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("PPL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PPL_RUNS_DIR", str(tmp_path / "runs"))
    return load_settings(tmp_path)


def _write(tmp_path, **overrides):
    raw = yaml.safe_load(Path("configs/experiment.yaml").read_text(encoding="utf-8"))
    raw.update(overrides)
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def test_repository_config_loads_and_expands_placeholders(tmp_path, settings):
    spec = load_spec(_write(tmp_path), settings)
    assert spec.bench_dir == settings.data_dir / "benchmark"
    assert spec.runs_dir == settings.runs_dir
    assert spec.primary_metric == "mrr@10"
    assert spec.frozen_path == settings.data_dir / "benchmark" / "frozen_params.yaml"
    assert ("C4-WS", "C3-WS") in spec.comparisons["RQ3"]
    assert spec.tune_grid["alpha"][:3] == [0.0, 0.1, 0.2]


def test_resolve_requires_frozen_values(tmp_path, settings):
    spec = load_spec(_write(tmp_path), settings)
    with pytest.raises(FileNotFoundError, match="eval tune"):
        resolve_configs(spec, None)
    entries = resolve_configs(spec, {**DEFAULT_FROZEN, "alpha": 0.7, "rerank_n": 20})
    assert entries["C3-WS"].pipeline.alpha == 0.7
    assert entries["C4-WS"].pipeline.rerank_n == 20
    assert entries["X5-N50"].pipeline.rerank_n == 50
    assert entries["X3-pyvi-C1"].pipeline.tokenizer == "pyvi"
    assert entries["C1"].pipeline.top_l == 100 and entries["C1"].index_dir is None


def test_config_without_frozen_values_resolves_without_tuning(tmp_path, settings):
    path = _write(tmp_path, configs={"C1": {"sparse": True, "dense": False, "fusion": "none", "rerank": False}},
                  comparisons={}, error_analysis={"target": "C1"})
    assert resolve_configs(load_spec(path, settings), None)["C1"].pipeline.fusion == "none"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"configs": {"C1": {"method": "bm25"}}}, "unknown keys"),
        ({"primary_metric": "map@10"}, "Unknown metric"),
        ({"comparisons": {"RQ9": [["C1", "C7"]]}}, "C7"),
        ({"error_analysis": {"target": "C9"}}, "C9"),
        ({"configs": {"C1": {"fusion": "none", "sparse": True, "dense": False, "top_l": "frozen"}}}, "top_l"),
    ],
)
def test_invalid_specs_are_rejected(tmp_path, settings, overrides, message):
    base = {"comparisons": {}, "error_analysis": {"target": "C1"}}
    if "configs" not in overrides:
        base["configs"] = {"C1": {"sparse": True, "dense": False, "fusion": "none", "rerank": False}}
    with pytest.raises(ValueError, match=message):
        spec = load_spec(_write(tmp_path, **{**base, **overrides}), settings)
        resolve_configs(spec, DEFAULT_FROZEN)


def test_per_config_index_dir_and_remap(tmp_path, settings):
    path = _write(
        tmp_path,
        configs={"X4": {"fusion": "weighted", "rerank": False, "index_dir": "${DATA_DIR}/indexes/v2", "qrels": "remap"}},
        comparisons={},
        error_analysis={"target": "X4"},
    )
    entry = resolve_configs(load_spec(path, settings), None)["X4"]
    assert entry.index_dir == settings.data_dir / "indexes" / "v2"
    assert entry.remap_qrels is True


def test_read_frozen(tmp_path, settings):
    spec = load_spec(_write(tmp_path), settings)
    assert read_frozen(spec) is None
    spec.frozen_path.parent.mkdir(parents=True)
    spec.frozen_path.write_text("alpha: 0.6\nbest_single: C2\n", encoding="utf-8")
    assert read_frozen(spec) == {"alpha": 0.6, "best_single": "C2"}
```

- [ ] **Step 4: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_eval_spec.py -q`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.eval.spec'`

- [ ] **Step 5: Viết `src/eval/spec.py`**

```python
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.eval.metrics import DEFAULT_KS, metric_names
from src.models import PipelineConfig

FROZEN_KEYS = ("alpha", "rrf_k", "adaptive_beta", "rerank_n")
DEFAULT_FROZEN = {"alpha": 0.5, "rrf_k": 60, "adaptive_beta": 0.3, "rerank_n": 30}
DEFAULT_TUNE_GRID = {
    "alpha": [round(0.1 * step, 1) for step in range(11)],
    "rrf_k": [10, 20, 40, 60, 100],
    "adaptive_beta": [0.1, 0.2, 0.3, 0.5],
    "rerank_n": [10, 20, 30, 50],
}
PIPELINE_KEYS = {field.name for field in dataclasses.fields(PipelineConfig)}
EXTRA_KEYS = {"index_dir", "qrels"}


@dataclass(frozen=True)
class ConfigEntry:
    name: str
    pipeline: PipelineConfig
    index_dir: Path | None
    remap_qrels: bool


@dataclass(frozen=True)
class ExperimentSpec:
    path: Path
    raw: dict
    bench_dir: Path
    index_dir: Path | None
    runs_dir: Path
    seed: int
    ks: tuple[int, ...]
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    warmup: int
    defaults: dict
    configs: dict[str, dict]
    comparisons: dict[str, list[tuple[str, str]]]
    error_analysis: dict
    tune_grid: dict

    @property
    def frozen_path(self) -> Path:
        return self.bench_dir / "frozen_params.yaml"


def _expand(value, settings) -> Path | None:
    if value in (None, ""):
        return None
    text = str(value).replace("${DATA_DIR}", str(settings.data_dir)).replace("${RUNS_DIR}", str(settings.runs_dir))
    return Path(text)


def load_spec(path, settings) -> ExperimentSpec:
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    configs = {name: dict(values or {}) for name, values in (raw.get("configs") or {}).items()}
    if not configs:
        raise ValueError("configs must be a non-empty mapping")
    for name, values in configs.items():
        unknown = set(values) - PIPELINE_KEYS - EXTRA_KEYS
        if unknown:
            raise ValueError(f"Config {name} has unknown keys: {sorted(unknown)}")
        if values.get("qrels", "default") not in ("default", "remap"):
            raise ValueError(f"Config {name}: qrels must be 'default' or 'remap'")
        if values.get("index_dir"):
            values["index_dir"] = str(_expand(values["index_dir"], settings))
    ks = tuple(int(k) for k in raw.get("ks", DEFAULT_KS))
    available = metric_names(ks)
    primary = raw.get("primary_metric", "mrr@10")
    secondary = tuple(raw.get("secondary_metrics", ["ndcg@10", "recall@5"]))
    for metric in (primary, *secondary):
        if metric not in available:
            raise ValueError(f"Unknown metric {metric}; available: {available}")
    comparisons = {family: [tuple(pair) for pair in pairs] for family, pairs in (raw.get("comparisons") or {}).items()}
    for family, pairs in comparisons.items():
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError(f"Comparison {family} must list [system, baseline] pairs")
            for name in pair:
                if name != "best_single" and name not in configs:
                    raise ValueError(f"Comparison {family} refers to unknown config {name}")
    error_analysis = {"target": "C4-WS", "k": 10, "sample": 50, **(raw.get("error_analysis") or {})}
    if error_analysis["target"] not in configs:
        raise ValueError(f"error_analysis target {error_analysis['target']} is not a config")
    return ExperimentSpec(
        path=source,
        raw=raw,
        bench_dir=_expand(raw.get("bench_dir", "${DATA_DIR}/benchmark"), settings),
        index_dir=_expand(raw.get("index_dir"), settings),
        runs_dir=_expand(raw.get("runs_dir", "${RUNS_DIR}"), settings),
        seed=int(raw.get("seed", 42)),
        ks=ks,
        primary_metric=primary,
        secondary_metrics=secondary,
        warmup=int((raw.get("latency") or {}).get("warmup", 5)),
        defaults=dict(raw.get("defaults") or {}),
        configs=configs,
        comparisons=comparisons,
        error_analysis=error_analysis,
        tune_grid={**DEFAULT_TUNE_GRID, **(raw.get("tune") or {})},
    )


def resolve_configs(spec: ExperimentSpec, frozen: dict | None) -> dict[str, ConfigEntry]:
    source = {**DEFAULT_FROZEN, **(frozen or {})}
    entries = {}
    for name, values in spec.configs.items():
        merged = {**spec.defaults, **values}
        uses_frozen = [key for key, value in merged.items() if value == "frozen"]
        invalid = [key for key in uses_frozen if key not in FROZEN_KEYS]
        if invalid:
            raise ValueError(f"Config {name}: only {FROZEN_KEYS} may be 'frozen', not {invalid}")
        if uses_frozen and frozen is None:
            raise FileNotFoundError(
                f"Config {name} uses frozen values {uses_frozen}; run `python -m src.cli eval tune` on the dev split first"
            )
        pipeline_values = {
            key: source[key] if value == "frozen" else value for key, value in merged.items() if key in PIPELINE_KEYS
        }
        entries[name] = ConfigEntry(
            name=name,
            pipeline=PipelineConfig.from_dict(pipeline_values),
            index_dir=Path(merged["index_dir"]) if merged.get("index_dir") else None,
            remap_qrels=merged.get("qrels", "default") == "remap",
        )
    return entries


def read_frozen(spec: ExperimentSpec) -> dict | None:
    if not spec.frozen_path.exists():
        return None
    return yaml.safe_load(spec.frozen_path.read_text(encoding="utf-8")) or {}
```

- [ ] **Step 6: Chạy test, xác nhận pass**

Run: `.venv/Scripts/python -m pytest tests/test_eval_spec.py -q`
Expected: `10 passed`

- [ ] **Step 7: Commit**

```bash
git add src/eval/spec.py configs/experiment.yaml configs/experiment.example.yaml examples/benchmark tests/test_eval_spec.py
git commit -m "feat: add experiment spec with frozen parameter resolution

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Runner có checkpoint, metrics theo nhóm, khóa test và đo độ trễ

**Files:**
- Create: `src/eval/runner.py`, `tests/test_eval_runner.py`
- Delete: `src/experiments.py`, `run_experiments.py`, `configs/experiments.yaml`, `configs/example_experiments.yaml`, `data/benchmark/queries.example.jsonl`, `data/benchmark/qrels.example.jsonl`, `tests/test_core.py`

**Interfaces:**
- Consumes: `ExperimentSpec`, `resolve_configs`, `ConfigEntry`, `query_metrics`, `check_test_lock`, `remap_qrels`, `RetrievalPipeline`, `RetrievalIndex`, `RetrievalCache`, `read_jsonl`, `write_csv`.
- Produces:
  - `RESULT_FIELDS = ("chunk_id", "sparse_score", "sparse_rank", "dense_score", "dense_rank", "fusion_score", "rerank_score")` — mỗi phần tử trong `row["results"]` là list theo thứ tự này
  - `load_benchmark(bench_dir, split: str) -> tuple[list[dict], dict[str, dict[str, int]], list[dict]]` (split ∈ dev/test/all)
  - `default_pipeline_factory(settings) -> Callable[[Path], RetrievalPipeline]`
  - `run_evaluation(spec, split, *, index_dir: Path, pipeline_factory, frozen: dict | None, resume_run_id: str | None = None, only: list[str] | None = None, latency: bool = True) -> Path`
  - Output trong run dir: `config.json` (khóa `split`, `spec`, `frozen`, `lock`, `configs` {tên: PipelineConfig dict + `index_dir`, `index_version`, `remap_qrels`}, `indexes` {đường dẫn: meta}), `status.json`, `per_query.jsonl` (mỗi dòng `{"config", "query_id", "alpha_used", "timings_ms", "results"}`), `qrels_used.json` (`{"default": …, "<config remap>": …}`), `metrics.json` (`{"overall": {cfg: {metric: v}}, "by_category": {cfg: {cat: {metric: v, "n": n}}}, "by_origin": {…}, "n_queries": n}`), `metrics_per_query.csv`, `errors.csv` (nếu có), `latency.json` (nếu bật)
  - `write_metrics(run_dir, rows, queries, qrels_by_config, ks) -> dict`
  - `measure_latency(entries, pipeline_for, queries, warmup) -> dict` — `{"hardware": {...}, "warmup": n, "configs": {cfg: {stage: {"mean", "p50", "p95"}}}}`
  - `hardware_info() -> dict`

- [ ] **Step 1: Viết test thất bại `tests/test_eval_runner.py`**

```python
import json

import pytest
import yaml

from src.config import load_settings
from src.eval.runner import load_benchmark, run_evaluation
from src.eval.spec import DEFAULT_FROZEN, load_spec
from src.io_utils import read_csv, write_jsonl
from src.models import RetrievedChunk, StageScores
from src.pipeline import PipelineResult
from tests.fakes import make_chunk

CHUNKS = {chunk_id: make_chunk(chunk_id, chunk_id) for chunk_id in ("c1", "c2", "c3")}


class FakeIndex:
    version = "fake-v1"
    meta = {"embedding_model": "fake", "chunk_count": 3}
    chunk_list = list(CHUNKS.values())


class FakePipeline:
    def __init__(self, fail_once_on=None):
        self.index = FakeIndex()
        self.calls = []
        self.fail_once_on = fail_once_on

    def run(self, query, config, use_cache=True):
        self.calls.append((query, config.fusion, use_cache))
        if query == self.fail_once_on:
            self.fail_once_on = None
            raise KeyboardInterrupt
        order = ["c1", "c2", "c3"] if config.sparse and not config.dense else ["c2", "c1", "c3"]
        results = [
            RetrievedChunk(CHUNKS[chunk_id], StageScores(rank=rank, sparse_score=1.0 / rank if config.sparse else None))
            for rank, chunk_id in enumerate(order, start=1)
        ]
        return PipelineResult(results, {"sparse": 1.0, "dense": 2.0, "fusion": 0.1, "rerank": 0.0, "total": 3.1})


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("PPL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PPL_RUNS_DIR", str(tmp_path / "runs"))
    settings = load_settings(tmp_path)
    bench = tmp_path / "bench"
    write_jsonl(bench / "queries.jsonl", [
        {"query_id": "q1", "text": "q1", "category": "exact", "origin": "llm", "split": "test"},
        {"query_id": "q2", "text": "q2", "category": "concept", "origin": "human", "split": "test"},
        {"query_id": "q3", "text": "q3", "category": "concept", "origin": "llm", "split": "dev"},
    ])
    write_jsonl(bench / "qrels.jsonl", [
        {"query_id": "q1", "chunk_id": "c1", "relevance": 2},
        {"query_id": "q2", "chunk_id": "c2", "relevance": 1},
        {"query_id": "q3", "chunk_id": "c3", "relevance": 1},
    ])
    config = {
        "bench_dir": str(bench), "runs_dir": "${RUNS_DIR}", "latency": {"warmup": 1},
        "configs": {
            "C1": {"sparse": True, "dense": False, "fusion": "none", "rerank": False},
            "C3-WS": {"fusion": "weighted", "alpha": "frozen", "rerank": False},
        },
        "comparisons": {}, "error_analysis": {"target": "C1"},
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return load_spec(path, settings), bench


def _run(spec, split, pipeline, **kwargs):
    return run_evaluation(spec, split, index_dir=spec.bench_dir, pipeline_factory=lambda _: pipeline, **kwargs)


def test_load_benchmark_filters_split(setup):
    spec, bench = setup
    queries, qrels, evidence = load_benchmark(bench, "test")
    assert [query["query_id"] for query in queries] == ["q1", "q2"]
    assert qrels["q2"] == {"c2": 1} and evidence == []
    with pytest.raises(ValueError, match="bench split"):
        load_benchmark(bench, "holdout")


def test_dev_run_writes_metrics_by_group_and_latency(setup):
    spec, _ = setup
    pipeline = FakePipeline()
    run_dir = _run(spec, "dev", pipeline, frozen=DEFAULT_FROZEN)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_queries"] == 1
    assert metrics["overall"]["C1"]["mrr@10"] == pytest.approx(1 / 3)
    assert metrics["by_category"]["C1"]["concept"]["n"] == 1
    rows = read_csv(run_dir / "metrics_per_query.csv")
    assert {row["config"] for row in rows} == {"C1", "C3-WS"}
    latency = json.loads((run_dir / "latency.json").read_text(encoding="utf-8"))
    assert latency["configs"]["C1"]["total"]["p95"] == pytest.approx(3.1)
    assert "python" in latency["hardware"]
    assert any(use_cache is False for _, _, use_cache in pipeline.calls)
    first = json.loads((run_dir / "per_query.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["results"][0][:3] == ["c1", 1.0, None]
    assert json.loads((run_dir / "status.json").read_text())["status"] == "completed"


def test_test_split_requires_frozen_params(setup):
    spec, _ = setup
    with pytest.raises(FileNotFoundError, match="eval tune"):
        _run(spec, "test", FakePipeline(), frozen=None)
    assert not spec.runs_dir.exists() or not any(spec.runs_dir.iterdir())


def test_test_run_records_lock_violation(setup):
    spec, bench = setup
    from src.bench.manifest import write_manifest

    write_manifest(bench, "fake-v1", 42, 0.3)
    spec.frozen_path.write_text("alpha: 0.5\n", encoding="utf-8")
    first = _run(spec, "test", FakePipeline(), frozen={"alpha": 0.5}, latency=False)
    assert json.loads((first / "config.json").read_text(encoding="utf-8"))["lock"]["lock_violation"] is False
    spec.frozen_path.write_text("alpha: 0.9\n", encoding="utf-8")
    second = _run(spec, "test", FakePipeline(), frozen={"alpha": 0.9}, latency=False)
    assert json.loads((second / "config.json").read_text(encoding="utf-8"))["lock"]["lock_violation"] is True


def test_run_resumes_without_repeating_completed_queries(setup):
    spec, _ = setup
    pipeline = FakePipeline(fail_once_on="q2")
    with pytest.raises(KeyboardInterrupt):
        _run(spec, "all", pipeline, frozen=DEFAULT_FROZEN, only=["C1"], latency=False)
    run_dir = next(spec.runs_dir.iterdir())
    assert json.loads((run_dir / "status.json").read_text())["status"] == "interrupted"
    _run(spec, "all", pipeline, frozen=DEFAULT_FROZEN, only=["C1"], latency=False, resume_run_id=run_dir.name)
    queries_run = [query for query, _, _ in pipeline.calls]
    assert queries_run.count("q1") == 1
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_queries"] == 3


def test_only_rejects_unknown_config(setup):
    spec, _ = setup
    with pytest.raises(ValueError, match="C9"):
        _run(spec, "dev", FakePipeline(), frozen=DEFAULT_FROZEN, only=["C9"])
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_eval_runner.py -q`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.eval.runner'`

- [ ] **Step 3: Viết `src/eval/runner.py`**

```python
import hashlib
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.bench.manifest import check_test_lock
from src.bench.remap import remap_qrels
from src.cache import RetrievalCache
from src.eval.metrics import query_metrics
from src.eval.spec import resolve_configs
from src.index import RetrievalIndex
from src.io_utils import read_jsonl, write_csv
from src.pipeline import RetrievalPipeline

RESULT_FIELDS = ("chunk_id", "sparse_score", "sparse_rank", "dense_score", "dense_rank", "fusion_score", "rerank_score")
STAGES = ("sparse", "dense", "fusion", "rerank", "total")


def load_benchmark(bench_dir, split: str):
    directory = Path(bench_dir)
    queries = read_jsonl(directory / "queries.jsonl")
    if split != "all":
        queries = [query for query in queries if query.get("split") == split]
    if not queries:
        raise ValueError(f"No queries with split={split} in {directory}; run `python -m src.cli bench split` first")
    qrels: dict[str, dict[str, int]] = {}
    for row in read_jsonl(directory / "qrels.jsonl"):
        qrels.setdefault(row["query_id"], {})[row["chunk_id"]] = int(row["relevance"])
    evidence_path = directory / "evidence.jsonl"
    return queries, qrels, read_jsonl(evidence_path) if evidence_path.exists() else []


def default_pipeline_factory(settings):
    cache = RetrievalCache(settings.cache_path)
    return lambda index_dir: RetrievalPipeline(RetrievalIndex.load(index_dir), cache=cache)


def hardware_info() -> dict:
    info = {"platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version()}
    try:
        import torch
    except ImportError:
        return info
    info["torch"] = torch.__version__
    info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return info


def _as_qrels(rows: list[dict]) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for row in rows:
        qrels.setdefault(row["query_id"], {})[row["chunk_id"]] = int(row["relevance"])
    return qrels


def _group_means(per_query: list[dict], names: list[str], key: str | None) -> dict:
    groups = defaultdict(list)
    for row in per_query:
        groups[(row["config"], row[key] if key else None)].append(row)
    summary: dict = {}
    for (config, group), rows in sorted(groups.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        values = {name: float(np.mean([row[name] for row in rows])) for name in names}
        if key:
            summary.setdefault(config, {})[group] = {**values, "n": len(rows)}
        else:
            summary[config] = values
    return summary


def write_metrics(run_dir, rows, queries, qrels_by_config, ks) -> dict:
    info = {query["query_id"]: query for query in queries}
    per_query = []
    for row in rows:
        if row["config"] not in qrels_by_config or row["query_id"] not in info:
            continue
        query = info[row["query_id"]]
        ranking = [item[0] for item in row["results"]]
        per_query.append(
            {
                "config": row["config"],
                "query_id": row["query_id"],
                "category": query["category"],
                "origin": query.get("origin", "llm"),
                **query_metrics(ranking, qrels_by_config[row["config"]].get(row["query_id"], {}), ks),
            }
        )
    names = [name for name in (per_query[0] if per_query else {}) if name not in ("config", "query_id", "category", "origin")]
    summary = {
        "overall": _group_means(per_query, names, None),
        "by_category": _group_means(per_query, names, "category"),
        "by_origin": _group_means(per_query, names, "origin"),
        "n_queries": len(queries),
    }
    target = Path(run_dir)
    (target / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(target / "metrics_per_query.csv", per_query)
    return summary


def measure_latency(entries, pipeline_for, queries, warmup: int) -> dict:
    report = {"hardware": hardware_info(), "warmup": warmup, "configs": {}}
    for name, entry in entries.items():
        pipeline = pipeline_for(entry)
        for query in queries[:warmup]:
            pipeline.run(query["text"], entry.pipeline, use_cache=False)
        samples = {stage: [] for stage in STAGES}
        for query in queries:
            timings = pipeline.run(query["text"], entry.pipeline, use_cache=False).timings_ms
            for stage in STAGES:
                samples[stage].append(float(timings.get(stage, 0.0)))
        report["configs"][name] = {
            stage: {
                "mean": float(np.mean(values)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
            }
            for stage, values in samples.items()
        }
    return report


def _write_status(run_dir: Path, status: str) -> None:
    (run_dir / "status.json").write_text(json.dumps({"status": status}), encoding="utf-8")


def run_evaluation(
    spec,
    split: str,
    *,
    index_dir: Path,
    pipeline_factory,
    frozen: dict | None,
    resume_run_id: str | None = None,
    only: list[str] | None = None,
    latency: bool = True,
) -> Path:
    queries, base_qrels, evidence = load_benchmark(spec.bench_dir, split)
    lock = check_test_lock(spec.bench_dir, spec.frozen_path) if split == "test" else None
    if lock and lock["lock_violation"]:
        print("CẢNH BÁO: frozen_params.yaml đã thay đổi sau lần chạy test đầu tiên (lock_violation).", file=sys.stderr)
    entries = resolve_configs(spec, frozen)
    if only:
        unknown = [name for name in only if name not in entries]
        if unknown:
            raise ValueError(f"Unknown configs in --only: {unknown}")
        entries = {name: entry for name, entry in entries.items() if name in only}

    pipelines: dict[str, object] = {}

    def pipeline_for(entry):
        key = str(entry.index_dir or index_dir)
        if key not in pipelines:
            pipelines[key] = pipeline_factory(Path(key))
        return pipelines[key]

    qrels_by_config = {
        name: _as_qrels(remap_qrels(evidence, pipeline_for(entry).index.chunk_list)) if entry.remap_qrels else base_qrels
        for name, entry in entries.items()
    }
    payload = json.dumps({"spec": spec.raw, "split": split, "frozen": frozen}, sort_keys=True, default=str)
    run_id = resume_run_id or (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{split}-{hashlib.sha256(payload.encode()).hexdigest()[:8]}"
    )
    run_dir = spec.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=bool(resume_run_id))
    config_record = {
        "split": split,
        "spec_path": str(spec.path),
        "spec": spec.raw,
        "frozen": frozen,
        "lock": lock,
        "configs": {
            name: {
                **entry.pipeline.to_dict(),
                "index_dir": str(entry.index_dir or index_dir),
                "index_version": pipeline_for(entry).index.version,
                "remap_qrels": entry.remap_qrels,
            }
            for name, entry in entries.items()
        },
        "indexes": {key: pipeline.index.meta for key, pipeline in pipelines.items()},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "config.json").write_text(json.dumps(config_record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    qrels_used = {"default": base_qrels, **{name: qrels_by_config[name] for name, entry in entries.items() if entry.remap_qrels}}
    (run_dir / "qrels_used.json").write_text(json.dumps(qrels_used, ensure_ascii=False), encoding="utf-8")
    _write_status(run_dir, "running")

    checkpoint = run_dir / "per_query.jsonl"
    rows = read_jsonl(checkpoint) if checkpoint.exists() else []
    done = {(row["config"], row["query_id"]) for row in rows}
    errors = []
    try:
        with checkpoint.open("a", encoding="utf-8") as stream:
            for name, entry in entries.items():
                pipeline = pipeline_for(entry)
                for query in queries:
                    if (name, query["query_id"]) in done:
                        continue
                    try:
                        result = pipeline.run(query["text"], entry.pipeline)
                    except Exception as error:
                        errors.append({"config": name, "query_id": query["query_id"], "error": str(error)})
                        continue
                    row = {
                        "config": name,
                        "query_id": query["query_id"],
                        "alpha_used": result.alpha_used,
                        "timings_ms": result.timings_ms,
                        "results": [
                            [item.chunk.chunk_id, *[getattr(item.scores, field) for field in RESULT_FIELDS[1:]]]
                            for item in result.results
                        ],
                    }
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    rows.append(row)
        write_metrics(run_dir, rows, queries, qrels_by_config, spec.ks)
        if errors:
            write_csv(run_dir / "errors.csv", errors, ["config", "query_id", "error"])
        if latency:
            report = measure_latency(entries, pipeline_for, queries, spec.warmup)
            (run_dir / "latency.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_status(run_dir, "completed")
    except BaseException:
        _write_status(run_dir, "interrupted")
        raise
    return run_dir
```

- [ ] **Step 4: Xóa code thí nghiệm cũ**

Xóa: `src/experiments.py`, `run_experiments.py`, `configs/experiments.yaml`, `configs/example_experiments.yaml`, `data/benchmark/queries.example.jsonl`, `data/benchmark/qrels.example.jsonl`, `tests/test_core.py` (test resume đã chuyển sang `tests/test_eval_runner.py`).

Run: `grep -rn "src.experiments\|run_experiments\|example_experiments" --include=*.py --include=*.md src pages tests app.py run_rag_evaluation.py README.md CLAUDE.md`
Expected: chỉ còn các dòng trong `README.md`/`CLAUDE.md` (sửa ở Task 10). `pages/4_Experiments.py` vẫn đọc `configs/experiments.yaml` khi bấm nút — được viết lại ở Task 9.

- [ ] **Step 5: Chạy test, xác nhận pass**

Run: `.venv/Scripts/python -m pytest tests/test_eval_runner.py -q`
Expected: `6 passed`

Run: `.venv/Scripts/python -m pytest -q`
Expected: tất cả pass.

- [ ] **Step 6: Commit**

```bash
git add -A src/eval/runner.py tests/test_eval_runner.py src/experiments.py run_experiments.py configs data/benchmark tests/test_core.py
git commit -m "feat: add resumable evaluation runner with grouped metrics, test lock and latency

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Tinh chỉnh tham số trên dev (`eval/tune.py`)

**Files:**
- Create: `src/eval/tune.py`, `tests/test_eval_tune.py`

**Interfaces:**
- Consumes: `load_benchmark`, `resolve_configs`, `DEFAULT_FROZEN`, `query_metrics`, `write_csv`.
- Produces: `tune(spec, *, index_dir: Path, pipeline_factory, force: bool = False) -> Path` — ghi `<bench_dir>/tune_results.csv` (cột `param, value, category, metric, score`; `category = "all"` hoặc tên nhóm) và `frozen_params.yaml` (khóa `alpha, rrf_k, adaptive_beta, rerank_n, best_single, primary_metric, tuned_on, index_version, dev_queries, tuned_at`). Yêu cầu spec có `C1, C2, C3-WS, C3-RRF, X2, C4-WS`. FileExistsError nếu frozen đã có và `force=False`.

- [ ] **Step 1: Viết test thất bại `tests/test_eval_tune.py`**

```python
import pytest
import yaml

from src.config import load_settings
from src.eval.spec import load_spec
from src.eval.tune import tune
from src.io_utils import read_csv, write_jsonl
from src.models import RetrievedChunk, StageScores
from src.pipeline import PipelineResult
from tests.fakes import make_chunk

CHUNKS = {chunk_id: make_chunk(chunk_id, chunk_id) for chunk_id in ("good", "other")}


class AlphaSensitivePipeline:
    class index:
        version = "fake-v1"

    def run(self, query, config, use_cache=True):
        good_first = (config.fusion == "weighted" and config.alpha >= 0.7) or (config.sparse and not config.dense)
        order = ["good", "other"] if good_first else ["other", "good"]
        return PipelineResult(
            [RetrievedChunk(CHUNKS[c], StageScores(rank=r)) for r, c in enumerate(order, start=1)],
            {"total": 1.0},
        )


@pytest.fixture
def spec(tmp_path, monkeypatch):
    monkeypatch.setenv("PPL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PPL_RUNS_DIR", str(tmp_path / "runs"))
    bench = tmp_path / "bench"
    write_jsonl(bench / "queries.jsonl", [
        {"query_id": "q1", "text": "a", "category": "exact", "split": "dev"},
        {"query_id": "q2", "text": "b", "category": "concept", "split": "dev"},
        {"query_id": "q3", "text": "c", "category": "concept", "split": "test"},
    ])
    write_jsonl(bench / "qrels.jsonl", [
        {"query_id": q, "chunk_id": "good", "relevance": 2} for q in ("q1", "q2", "q3")
    ])
    raw = yaml.safe_load(open("configs/experiment.yaml", encoding="utf-8"))
    raw.update(bench_dir=str(bench), tune={"alpha": [0.5, 0.7, 0.9], "rrf_k": [60], "adaptive_beta": [0.3], "rerank_n": [10, 30]})
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return load_spec(path, load_settings(tmp_path))


def test_tune_selects_best_values_and_writes_curves(spec):
    frozen_path = tune(spec, index_dir=spec.bench_dir, pipeline_factory=lambda _: AlphaSensitivePipeline())
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    assert frozen["alpha"] == 0.7
    assert frozen["rrf_k"] == 60 and frozen["adaptive_beta"] == 0.3
    assert frozen["rerank_n"] == 30
    assert frozen["best_single"] == "C1"
    assert frozen["tuned_on"] == "dev" and frozen["dev_queries"] == 2
    rows = read_csv(spec.bench_dir / "tune_results.csv")
    alpha_all = [row for row in rows if row["param"] == "alpha" and row["category"] == "all"]
    assert [float(row["value"]) for row in alpha_all] == [0.5, 0.7, 0.9]
    assert any(row["category"] == "concept" for row in rows)


def test_tune_refuses_to_overwrite_without_force(spec):
    tune(spec, index_dir=spec.bench_dir, pipeline_factory=lambda _: AlphaSensitivePipeline())
    with pytest.raises(FileExistsError, match="--force"):
        tune(spec, index_dir=spec.bench_dir, pipeline_factory=lambda _: AlphaSensitivePipeline())
    tune(spec, index_dir=spec.bench_dir, pipeline_factory=lambda _: AlphaSensitivePipeline(), force=True)
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_eval_tune.py -q`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.eval.tune'`

- [ ] **Step 3: Viết `src/eval/tune.py`**

```python
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from src.eval.metrics import query_metrics
from src.eval.runner import load_benchmark
from src.eval.spec import DEFAULT_FROZEN, resolve_configs
from src.io_utils import write_csv

REQUIRED_CONFIGS = ("C1", "C2", "C3-WS", "C3-RRF", "X2", "C4-WS")


def _choose(scores: dict, default):
    best = max(scores.values())
    tied = [value for value, score in scores.items() if math.isclose(score, best)]
    return min(tied, key=lambda value: abs(value - default))


def tune(spec, *, index_dir: Path, pipeline_factory, force: bool = False) -> Path:
    if spec.frozen_path.exists() and not force:
        raise FileExistsError(
            f"{spec.frozen_path} already exists; pass --force to re-tune (recorded as lock violation after a test run)"
        )
    queries, qrels, _ = load_benchmark(spec.bench_dir, "dev")
    base = resolve_configs(spec, DEFAULT_FROZEN)
    missing = [name for name in REQUIRED_CONFIGS if name not in base]
    if missing:
        raise ValueError(f"eval tune requires configs {missing} in {spec.path}")
    pipeline = pipeline_factory(index_dir)
    metric = spec.primary_metric
    rows: list[dict] = []

    def record(param: str, value, config) -> float:
        by_category = defaultdict(list)
        overall = []
        for query in queries:
            ranking = [item.chunk.chunk_id for item in pipeline.run(query["text"], config).results]
            score = query_metrics(ranking, qrels.get(query["query_id"], {}), spec.ks)[metric]
            overall.append(score)
            by_category[query["category"]].append(score)
        rows.append({"param": param, "value": value, "category": "all", "metric": metric, "score": float(np.mean(overall))})
        for category, scores in sorted(by_category.items()):
            rows.append({"param": param, "value": value, "category": category, "metric": metric, "score": float(np.mean(scores))})
        return float(np.mean(overall))

    grid = spec.tune_grid
    single = {name: record("single", name, base[name].pipeline) for name in ("C1", "C2")}
    best_single = max(single, key=lambda name: (single[name], name))
    alpha = _choose({value: record("alpha", value, base["C3-WS"].pipeline.replace(alpha=value)) for value in grid["alpha"]}, 0.5)
    rrf_k = _choose({value: record("rrf_k", value, base["C3-RRF"].pipeline.replace(rrf_k=value)) for value in grid["rrf_k"]}, 60)
    beta = _choose(
        {value: record("adaptive_beta", value, base["X2"].pipeline.replace(alpha=alpha, adaptive_beta=value)) for value in grid["adaptive_beta"]},
        0.3,
    )
    rerank_base = base["C4-WS"].pipeline
    rerank_n = _choose(
        {
            value: record("rerank_n", value, rerank_base.replace(alpha=alpha, rerank_n=value, context_k=min(rerank_base.context_k, value)))
            for value in grid["rerank_n"]
        },
        30,
    )
    frozen = {
        "alpha": float(alpha),
        "rrf_k": int(rrf_k),
        "adaptive_beta": float(beta),
        "rerank_n": int(rerank_n),
        "best_single": best_single,
        "primary_metric": metric,
        "tuned_on": "dev",
        "index_version": pipeline.index.version,
        "dev_queries": len(queries),
        "tuned_at": datetime.now(timezone.utc).isoformat(),
    }
    write_csv(spec.bench_dir / "tune_results.csv", rows, ["param", "value", "category", "metric", "score"])
    spec.frozen_path.write_text(yaml.safe_dump(frozen, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return spec.frozen_path
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `.venv/Scripts/python -m pytest tests/test_eval_tune.py -q`
Expected: `2 passed`

(Kiểm tra logic: C1 luôn xếp "good" đầu → MRR 1.0; C2 → 0.5 ⇒ best_single = C1. α ∈ {0.7, 0.9} cùng đạt 1.0 → chọn 0.7 vì gần 0.5. N: mọi giá trị hòa (C4-WS dùng α = 0.7) → chọn 30.)

- [ ] **Step 5: Commit**

```bash
git add src/eval/tune.py tests/test_eval_tune.py
git commit -m "feat: tune fusion and rerank parameters on the dev split

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: So sánh thống kê (`eval/compare.py`)

**Files:**
- Create: `src/eval/compare.py`, `tests/test_eval_compare.py`

**Interfaces:**
- Consumes: `metrics_per_query.csv` của run, `paired_bootstrap_ci`, `paired_randomization_test`, `holm_adjust`, `read_csv`, `write_csv`.
- Produces: `compare_run(run_dir, spec, frozen: dict | None, samples: int = 10000) -> list[dict]` — ghi `comparisons.csv` với cột `family, system, baseline, metric, category, n, mean_system, mean_baseline, diff, ci_low, ci_high, p_value, p_holm` (`p_holm` chỉ có ở hàng `category == "all"`).

- [ ] **Step 1: Viết test thất bại `tests/test_eval_compare.py`**

```python
import pytest
import yaml

from src.config import load_settings
from src.eval.compare import compare_run
from src.eval.spec import load_spec
from src.io_utils import read_csv, write_csv


@pytest.fixture
def spec(tmp_path, monkeypatch):
    monkeypatch.setenv("PPL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PPL_RUNS_DIR", str(tmp_path / "runs"))
    raw = {
        "bench_dir": str(tmp_path / "bench"),
        "configs": {
            "C1": {"sparse": True, "dense": False, "fusion": "none", "rerank": False},
            "C2": {"sparse": False, "dense": True, "fusion": "none", "rerank": False},
            "C3-WS": {"fusion": "weighted", "rerank": False},
        },
        "comparisons": {"RQ2": [["C3-WS", "best_single"], ["C3-WS", "C1"]]},
        "error_analysis": {"target": "C3-WS"},
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_spec(path, load_settings(tmp_path))


def _per_query(run_dir, configs):
    rows = []
    for config, scores in configs.items():
        for number, score in enumerate(scores):
            rows.append({"config": config, "query_id": f"q{number}", "category": "exact" if number % 2 else "concept",
                         "origin": "llm", "mrr@10": score, "ndcg@10": score, "recall@5": score})
    write_csv(run_dir / "metrics_per_query.csv", rows)


def test_compare_run_uses_best_single_and_holm(tmp_path, spec):
    run_dir = tmp_path / "run"
    _per_query(run_dir, {"C1": [0.0] * 20, "C2": [0.5] * 20, "C3-WS": [1.0] * 20})
    rows = compare_run(run_dir, spec, {"best_single": "C2"}, samples=500)
    overall = [row for row in rows if row["category"] == "all" and row["metric"] == "mrr@10"]
    assert [(row["system"], row["baseline"]) for row in overall] == [("C3-WS", "C2"), ("C3-WS", "C1")]
    assert overall[0]["diff"] == pytest.approx(0.5)
    assert overall[0]["ci_low"] == pytest.approx(0.5)
    assert all(row["p_holm"] != "" and row["p_holm"] >= row["p_value"] for row in overall)
    by_category = [row for row in rows if row["category"] == "exact" and row["metric"] == "mrr@10"]
    assert by_category[0]["n"] == 10 and by_category[0]["p_holm"] == ""
    saved = read_csv(run_dir / "comparisons.csv")
    assert len(saved) == len(rows) == 2 * 3 * 3


def test_compare_reports_missing_configs(tmp_path, spec):
    run_dir = tmp_path / "run"
    _per_query(run_dir, {"C1": [0.0] * 4, "C3-WS": [1.0] * 4})
    with pytest.raises(ValueError, match="C2"):
        compare_run(run_dir, spec, {"best_single": "C2"}, samples=100)
    with pytest.raises(ValueError, match="best_single"):
        compare_run(run_dir, spec, None, samples=100)
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_eval_compare.py -q`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.eval.compare'`

- [ ] **Step 3: Viết `src/eval/compare.py`**

```python
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
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `.venv/Scripts/python -m pytest tests/test_eval_compare.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/eval/compare.py tests/test_eval_compare.py
git commit -m "feat: add paired statistical comparisons per research question

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Phân tích lỗi theo tầng (`eval/errors.py`)

**Files:**
- Create: `src/eval/errors.py`, `tests/test_eval_errors.py`

**Interfaces:**
- Consumes: `per_query.jsonl`, `config.json`, `qrels_used.json` của run; `RESULT_FIELDS`; `read_jsonl`, `read_csv`, `write_csv`.
- Produces:
  - `STAGES = ("first_stage_miss", "fusion_demoted", "rerank_demoted", "ranked_below_k")`, `CAUSES = ("extraction", "chunk_boundary", "tokenization", "vocabulary_mismatch", "multi_hop", "label_error", "other")`
  - `classify_failures(run_dir, target: str, k: int = 10) -> list[dict]` — mỗi dòng `{"query_id", "stage", "chunk_id", "grade", "final_rank", "sparse_rank", "dense_rank", "rerank_score", "top1_chunk_id"}`
  - `export_error_sample(failures, queries, chunks: dict, path, sample: int = 50, seed: int = 42) -> list[dict]` — CSV cột `query_id, category, query, stage, final_rank, sparse_rank, dense_rank, relevant_chunk, top1_chunk, cause`
  - `summarize_causes(path) -> dict` — `{"by_stage": {...}, "by_cause": {...}, "rows": [{"stage", "cause", "count"}]}`; cause trống → `"unlabeled"`; cause lạ → ValueError nêu query_id

- [ ] **Step 1: Viết test thất bại `tests/test_eval_errors.py`**

```python
import json

import pytest

from src.eval.errors import classify_failures, export_error_sample, summarize_causes
from src.io_utils import read_csv, write_csv, write_jsonl
from tests.fakes import make_chunk


def _row(query_id, results):
    return {"config": "C4-WS", "query_id": query_id, "alpha_used": 0.5, "timings_ms": {}, "results": results}


@pytest.fixture
def run_dir(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"configs": {"C4-WS": {"fusion": "weighted", "rerank": True}}}))
    (directory / "qrels_used.json").write_text(json.dumps({"default": {
        "ok": {"a": 2}, "miss": {"z": 2}, "fusion": {"d": 1}, "rerank": {"b": 2},
    }}))
    write_jsonl(directory / "per_query.jsonl", [
        _row("ok", [["a", 1.0, 1, 0.9, 1, 0.9, 3.0]]),
        _row("miss", [["a", 1.0, 1, 0.9, 1, 0.9, 3.0]]),
        _row("fusion", [["a", None, None, 0.9, 1, 0.9, 3.0]] + [["x", None, None, 0.1, 2, 0.1, None]] * 10 + [["d", 0.5, 40, None, None, 0.01, None]]),
        _row("rerank", [["a", 1.0, 1, 0.9, 1, 0.9, 3.0]] * 10 + [["b", 2.0, 1, 0.8, 2, 0.95, -1.0]]),
    ])
    return directory


def test_classify_failures_by_stage(run_dir):
    failures = {row["query_id"]: row for row in classify_failures(run_dir, "C4-WS", k=10)}
    assert set(failures) == {"miss", "fusion", "rerank"}
    assert failures["miss"]["stage"] == "first_stage_miss" and failures["miss"]["final_rank"] is None
    assert failures["fusion"]["stage"] == "fusion_demoted" and failures["fusion"]["sparse_rank"] == 40
    assert failures["rerank"]["stage"] == "rerank_demoted" and failures["rerank"]["final_rank"] == 11
    assert failures["rerank"]["top1_chunk_id"] == "a"


def test_export_sample_and_summarize_causes(tmp_path, run_dir):
    failures = classify_failures(run_dir, "C4-WS")
    queries = [{"query_id": q, "text": f"câu {q}", "category": "exact" if q == "miss" else "concept"} for q in ("miss", "fusion", "rerank")]
    chunks = {cid: make_chunk(cid, f"nội dung {cid}") for cid in ("a", "b", "d", "x", "z")}
    rows = export_error_sample(failures, queries, chunks, tmp_path / "sample.csv", sample=2, seed=1)
    assert len(rows) == 2 and {row["category"] for row in rows} == {"exact", "concept"}
    saved = read_csv(tmp_path / "sample.csv")
    saved[0]["cause"] = "tokenization"
    write_csv(tmp_path / "sample.csv", saved)
    summary = summarize_causes(tmp_path / "sample.csv")
    assert summary["by_cause"] == {"tokenization": 1, "unlabeled": 1}
    saved[1]["cause"] = "bad"
    write_csv(tmp_path / "sample.csv", saved)
    with pytest.raises(ValueError, match=saved[1]["query_id"]):
        summarize_causes(tmp_path / "sample.csv")
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_eval_errors.py -q`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.eval.errors'`

- [ ] **Step 3: Viết `src/eval/errors.py`**

```python
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.io_utils import read_csv, read_jsonl, write_csv

STAGES = ("first_stage_miss", "fusion_demoted", "rerank_demoted", "ranked_below_k")
CAUSES = ("extraction", "chunk_boundary", "tokenization", "vocabulary_mismatch", "multi_hop", "label_error", "other")
SAMPLE_COLUMNS = [
    "query_id", "category", "query", "stage", "final_rank", "sparse_rank", "dense_rank",
    "relevant_chunk", "top1_chunk", "cause",
]


def classify_failures(run_dir, target: str, k: int = 10) -> list[dict]:
    directory = Path(run_dir)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))["configs"][target]
    qrels_used = json.loads((directory / "qrels_used.json").read_text(encoding="utf-8"))
    qrels = qrels_used.get(target, qrels_used["default"])
    failures = []
    for row in read_jsonl(directory / "per_query.jsonl"):
        if row["config"] != target:
            continue
        relevant = {chunk_id: grade for chunk_id, grade in qrels.get(row["query_id"], {}).items() if grade > 0}
        ids = [item[0] for item in row["results"]]
        if not relevant or any(chunk_id in relevant for chunk_id in ids[:k]):
            continue
        positions = {item[0]: (rank, item) for rank, item in enumerate(row["results"], start=1)}
        found = [chunk_id for chunk_id in relevant if chunk_id in positions]
        top1 = ids[0] if ids else None
        if not found:
            best = max(relevant, key=lambda chunk_id: (relevant[chunk_id], chunk_id))
            failures.append({"query_id": row["query_id"], "stage": "first_stage_miss", "chunk_id": best,
                             "grade": relevant[best], "final_rank": None, "sparse_rank": None, "dense_rank": None,
                             "rerank_score": None, "top1_chunk_id": top1})
            continue
        best = min(found, key=lambda chunk_id: (-relevant[chunk_id], positions[chunk_id][0]))
        rank, item = positions[best]
        if config["rerank"]:
            stage = "rerank_demoted" if item[6] is not None else "fusion_demoted"
        else:
            stage = "fusion_demoted" if config["fusion"] != "none" else "ranked_below_k"
        failures.append({"query_id": row["query_id"], "stage": stage, "chunk_id": best, "grade": relevant[best],
                         "final_rank": rank, "sparse_rank": item[2], "dense_rank": item[4],
                         "rerank_score": item[6], "top1_chunk_id": top1})
    return failures


def export_error_sample(failures, queries, chunks: dict, path, sample: int = 50, seed: int = 42) -> list[dict]:
    info = {query["query_id"]: query for query in queries}
    groups = defaultdict(list)
    for failure in sorted(failures, key=lambda item: item["query_id"]):
        groups[info[failure["query_id"]]["category"]].append(failure)
    rng = random.Random(seed)
    for members in groups.values():
        rng.shuffle(members)
    picked = []
    while len(picked) < sample and any(groups.values()):
        for category in sorted(groups):
            if groups[category] and len(picked) < sample:
                picked.append(groups[category].pop(0))
    rows = [
        {
            "query_id": failure["query_id"],
            "category": info[failure["query_id"]]["category"],
            "query": info[failure["query_id"]]["text"],
            "stage": failure["stage"],
            "final_rank": failure["final_rank"],
            "sparse_rank": failure["sparse_rank"],
            "dense_rank": failure["dense_rank"],
            "relevant_chunk": chunks[failure["chunk_id"]].body if failure["chunk_id"] in chunks else "",
            "top1_chunk": chunks[failure["top1_chunk_id"]].body if failure["top1_chunk_id"] in chunks else "",
            "cause": "",
        }
        for failure in picked
    ]
    write_csv(path, rows, SAMPLE_COLUMNS)
    return rows


def summarize_causes(path) -> dict:
    by_stage, by_cause, pairs = Counter(), Counter(), Counter()
    for row in read_csv(path):
        cause = row.get("cause", "").strip() or "unlabeled"
        if cause != "unlabeled" and cause not in CAUSES:
            raise ValueError(f"Unknown cause '{cause}' for {row['query_id']}; use one of {CAUSES}")
        by_stage[row["stage"]] += 1
        by_cause[cause] += 1
        pairs[(row["stage"], cause)] += 1
    return {
        "by_stage": dict(by_stage),
        "by_cause": dict(by_cause),
        "rows": [{"stage": stage, "cause": cause, "count": count} for (stage, cause), count in sorted(pairs.items())],
    }
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `.venv/Scripts/python -m pytest tests/test_eval_errors.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/eval/errors.py tests/test_eval_errors.py
git commit -m "feat: classify retrieval failures by pipeline stage and export samples

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Xuất bảng và hình cho Chương 3 (`eval/report.py`)

**Files:**
- Create: `src/eval/report.py`, `tests/test_eval_report.py`

**Interfaces:**
- Consumes: file trong run dir (`config.json`, `metrics.json`, `latency.json`, `comparisons.csv`, `errors_summary.json`) và bench dir (`benchmark_description.json`, `tune_results.csv`).
- Produces:
  - `TABLES: dict[str, str]` và `FIGURES: dict[str, str]` (khóa `"3.1"`… — nguồn duy nhất cho đánh số trong báo cáo)
  - `markdown_table(rows: list[dict], columns: list[str], headers: list[str] | None = None, digits: int = 3) -> str`
  - `build_report(run_dir, bench_dir, primary_metric: str = "mrr@10") -> list[Path]` — ghi `report/table_3_<n>.md` + `.csv` và `report/figure_3_<n>.png`; bỏ qua bảng/hình thiếu dữ liệu.

Đánh số cố định:

| Khóa | Bảng | Nguồn |
|---|---|---|
| 3.1 | Môi trường và thiết lập thực nghiệm | `config.json`, `latency.json` |
| 3.2 | Thống kê bộ benchmark theo nhóm, nguồn và tập | `benchmark_description.json` |
| 3.3 | Độ đồng thuận dán nhãn và đóng góp của từng hệ thống vào pool | `benchmark_description.json` |
| 3.4 | Tham số tối ưu chọn trên tập dev | `config.json["frozen"]` |
| 3.5 | Kết quả tổng thể trên tập test | `metrics.json` (cấu hình không bắt đầu bằng X3–X6) |
| 3.6 | Kết quả theo nhóm truy vấn (RQ1) | `metrics.json["by_category"]` |
| 3.7 | So sánh ghép cặp có kiểm định (RQ2, RQ3, X2) | `comparisons.csv` (hàng `all`) |
| 3.8 | Độ trễ theo tầng xử lý (ms) | `latency.json` |
| 3.9 | Kết quả theo nguồn câu hỏi | `metrics.json["by_origin"]` |
| 3.10 | Phân bố truy vấn thất bại theo tầng và nguyên nhân | `errors_summary.json` |
| 3.11 | Kết quả các hướng khai thác X3–X6 | `metrics.json` (cấu hình X3–X6) |
| Hình 3.1 | Ảnh hưởng của α đến chỉ số chính trên tập dev | `tune_results.csv` |
| Hình 3.2 | Đánh đổi chất lượng – độ trễ theo số ứng viên rerank N | cấu hình weighted + rerank trong `config.json`, `metrics.json`, `latency.json` |
| Hình 3.3 | Chỉ số chính theo nhóm truy vấn trên tập test | `metrics.json["by_category"]` |

- [ ] **Step 1: Viết test thất bại `tests/test_eval_report.py`**

```python
import json

from src.eval.report import FIGURES, TABLES, build_report, markdown_table
from src.io_utils import write_csv


def test_markdown_table_formats_numbers_and_missing_values():
    table = markdown_table([{"a": "C1", "b": 0.12345, "c": None}], ["a", "b", "c"], ["Cấu hình", "MRR", "X"])
    assert table.splitlines() == ["| Cấu hình | MRR | X |", "| --- | --- | --- |", "| C1 | 0.123 | – |"]


def _minimal_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    configs = {
        "C1": {"fusion": "none", "rerank": False, "rerank_n": 30, "tokenizer": "whitespace", "reranker_model": "r", "index_dir": "i", "index_version": "v1"},
        "C4-WS": {"fusion": "weighted", "rerank": True, "rerank_n": 30, "tokenizer": "whitespace", "reranker_model": "r", "index_dir": "i", "index_version": "v1"},
        "X5-N10": {"fusion": "weighted", "rerank": True, "rerank_n": 10, "tokenizer": "whitespace", "reranker_model": "r", "index_dir": "i", "index_version": "v1"},
    }
    (run_dir / "config.json").write_text(json.dumps({
        "split": "test", "frozen": {"alpha": 0.6, "rrf_k": 60, "adaptive_beta": 0.3, "rerank_n": 30, "best_single": "C2"},
        "lock": {"lock_violation": False}, "configs": configs,
        "indexes": {"i": {"embedding_model": "BAAI/bge-m3", "chunk_count": 120, "tokenizers": ["whitespace"]}},
    }), encoding="utf-8")
    metric = {"mrr@10": 0.5, "ndcg@10": 0.6, "recall@5": 0.7, "hit_rate@1": 0.4, "hit_rate@5": 0.8, "precision@5": 0.2}
    (run_dir / "metrics.json").write_text(json.dumps({
        "overall": {name: metric for name in configs},
        "by_category": {name: {"exact": {**metric, "n": 3}, "concept": {**metric, "n": 2}} for name in configs},
        "by_origin": {name: {"llm": {**metric, "n": 4}, "human": {**metric, "n": 1}} for name in configs},
        "n_queries": 5,
    }), encoding="utf-8")
    return run_dir


def test_report_skips_missing_inputs(tmp_path):
    run_dir = _minimal_run(tmp_path)
    written = build_report(run_dir, tmp_path / "bench")
    names = sorted(path.name for path in written)
    assert "table_3_5.md" in names and "table_3_6.md" in names and "table_3_11.md" in names
    assert "table_3_8.md" not in names and "table_3_2.md" not in names
    main = (run_dir / "report" / "table_3_5.md").read_text(encoding="utf-8")
    assert main.startswith(f"**Bảng 3.5. {TABLES['3.5']}**")
    assert "X5-N10" not in main
    assert "X5-N10" in (run_dir / "report" / "table_3_11.md").read_text(encoding="utf-8")
    assert (run_dir / "report" / "figure_3_3.png").exists()


def test_report_with_all_inputs_writes_every_table_and_figure(tmp_path):
    run_dir = _minimal_run(tmp_path)
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "benchmark_description.json").write_text(json.dumps({
        "counts": [{"category": "exact", "origin": "llm", "split": "test", "queries": 3}],
        "relevant_per_query": [{"category": "exact", "mean": 1.5, "median": 1.0}],
        "pool_contribution": [{"system": "C1", "relevant_found": 4, "unique_relevant": 1}],
        "agreement": {"overlap": 20, "kappa": 0.71, "raw_agreement": 0.8},
        "totals": {"queries": 5, "qrels": 30, "relevant": 9},
    }), encoding="utf-8")
    write_csv(bench / "tune_results.csv", [
        {"param": "alpha", "value": value, "category": category, "metric": "mrr@10", "score": score}
        for value, score in ((0.0, 0.3), (0.5, 0.5), (1.0, 0.4))
        for category in ("all", "exact")
    ])
    write_csv(run_dir / "comparisons.csv", [
        {"family": "RQ3", "system": "C4-WS", "baseline": "C3-WS", "metric": "mrr@10", "category": "all", "n": 5,
         "mean_system": 0.6, "mean_baseline": 0.5, "diff": 0.1, "ci_low": 0.02, "ci_high": 0.2, "p_value": 0.01, "p_holm": 0.03},
    ])
    stage = {"mean": 10.0, "p50": 9.0, "p95": 20.0}
    (run_dir / "latency.json").write_text(json.dumps({
        "hardware": {"platform": "Linux", "cuda_device": "Tesla T4", "torch": "2.x", "python": "3.11"},
        "configs": {name: {s: stage for s in ("sparse", "dense", "fusion", "rerank", "total")} for name in ("C1", "C4-WS", "X5-N10")},
    }), encoding="utf-8")
    (run_dir / "errors_summary.json").write_text(json.dumps({
        "rows": [{"stage": "rerank_demoted", "cause": "tokenization", "count": 2}],
    }), encoding="utf-8")

    written = {path.name for path in build_report(run_dir, bench)}
    assert {f"table_3_{key.split('.')[1]}.md" for key in TABLES} <= written
    assert {f"figure_3_{key.split('.')[1]}.png" for key in FIGURES} <= written
    environment = (run_dir / "report" / "table_3_1.md").read_text(encoding="utf-8")
    assert "Tesla T4" in environment and "BAAI/bge-m3" in environment
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_eval_report.py -q`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.eval.report'`

- [ ] **Step 3: Viết `src/eval/report.py`**

```python
import json
from collections import defaultdict
from pathlib import Path

from src.io_utils import read_csv, write_csv

TABLES = {
    "3.1": "Môi trường và thiết lập thực nghiệm",
    "3.2": "Thống kê bộ benchmark theo nhóm, nguồn và tập",
    "3.3": "Độ đồng thuận dán nhãn và đóng góp của từng hệ thống vào pool",
    "3.4": "Tham số tối ưu chọn trên tập dev",
    "3.5": "Kết quả tổng thể trên tập test",
    "3.6": "Kết quả theo nhóm truy vấn trên tập test (RQ1)",
    "3.7": "So sánh ghép cặp có kiểm định thống kê (RQ2, RQ3, X2)",
    "3.8": "Độ trễ theo tầng xử lý (ms)",
    "3.9": "Kết quả theo nguồn câu hỏi (phân tích độ nhạy)",
    "3.10": "Phân bố truy vấn thất bại theo tầng và nguyên nhân",
    "3.11": "Kết quả các hướng khai thác X3–X6",
}
FIGURES = {
    "3.1": "Ảnh hưởng của α đến chỉ số chính trên tập dev",
    "3.2": "Đánh đổi chất lượng – độ trễ theo số ứng viên rerank N",
    "3.3": "Chỉ số chính theo nhóm truy vấn trên tập test",
}
MAIN_METRICS = ["mrr@10", "ndcg@10", "recall@5", "hit_rate@1", "hit_rate@5", "precision@5"]
EXPLORATION_PREFIXES = ("X3", "X4", "X5", "X6")


def markdown_table(rows: list[dict], columns: list[str], headers: list[str] | None = None, digits: int = 3) -> str:
    def cell(value) -> str:
        if value is None or value == "":
            return "–"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    lines = [
        "| " + " | ".join(headers or columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines += ["| " + " | ".join(cell(row.get(column)) for column in columns) + " |" for row in rows]
    return "\n".join(lines)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _table(report_dir: Path, key: str, rows: list[dict], columns: list[str], headers: list[str] | None = None) -> list[Path]:
    if not rows:
        return []
    suffix = key.split(".")[1]
    markdown = report_dir / f"table_3_{suffix}.md"
    markdown.write_text(f"**Bảng {key}. {TABLES[key]}**\n\n{markdown_table(rows, columns, headers)}\n", encoding="utf-8")
    write_csv(report_dir / f"table_3_{suffix}.csv", rows, columns)
    return [markdown]


def _figure(report_dir: Path, key: str, draw) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    drawn = draw(axis)
    if not drawn:
        plt.close(figure)
        return []
    axis.set_title(f"Hình {key}. {FIGURES[key]}", fontsize=10)
    figure.tight_layout()
    path = report_dir / f"figure_3_{key.split('.')[1]}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return [path]


def build_report(run_dir, bench_dir, primary_metric: str = "mrr@10") -> list[Path]:
    run = Path(run_dir)
    bench = Path(bench_dir)
    report_dir = run / "report"
    report_dir.mkdir(exist_ok=True)
    config = _load_json(run / "config.json") or {}
    metrics = _load_json(run / "metrics.json") or {"overall": {}, "by_category": {}, "by_origin": {}}
    latency = _load_json(run / "latency.json")
    description = _load_json(bench / "benchmark_description.json")
    errors = _load_json(run / "errors_summary.json")
    configs = config.get("configs", {})
    main_names = [name for name in metrics["overall"] if not name.startswith(EXPLORATION_PREFIXES)]
    exploration = [name for name in metrics["overall"] if name.startswith(EXPLORATION_PREFIXES)]
    metric_columns = [metric for metric in MAIN_METRICS if any(metric in values for values in metrics["overall"].values())]
    written: list[Path] = []

    environment = [
        {"item": "Tập đánh giá", "value": config.get("split")},
        {"item": "Số truy vấn", "value": metrics.get("n_queries")},
        {"item": "Vi phạm khóa test", "value": (config.get("lock") or {}).get("lock_violation")},
    ]
    for directory, meta in (config.get("indexes") or {}).items():
        environment += [
            {"item": f"Index {Path(directory).name} — mô hình embedding", "value": meta.get("embedding_model")},
            {"item": f"Index {Path(directory).name} — số chunk", "value": meta.get("chunk_count")},
            {"item": f"Index {Path(directory).name} — tokenizer", "value": ", ".join(meta.get("tokenizers", []))},
        ]
    rerankers = sorted({values.get("reranker_model") for values in configs.values() if values.get("rerank")})
    environment.append({"item": "Mô hình reranker", "value": ", ".join(item for item in rerankers if item)})
    if latency:
        environment += [{"item": f"Phần cứng — {key}", "value": value} for key, value in latency["hardware"].items()]
    written += _table(report_dir, "3.1", environment, ["item", "value"], ["Thành phần", "Giá trị"])

    if description:
        written += _table(report_dir, "3.2", description["counts"], ["category", "origin", "split", "queries"],
                          ["Nhóm", "Nguồn", "Tập", "Số câu hỏi"])
        agreement = description.get("agreement") or {}
        rows = [{"item": "Cohen's κ (trọng số bậc hai)", "value": agreement.get("kappa")},
                {"item": "Tỉ lệ đồng ý thô", "value": agreement.get("raw_agreement")},
                {"item": "Số cặp dán nhãn đôi", "value": agreement.get("overlap")}]
        rows += [{"item": f"Pool — {row['system']}: tìm thấy / duy nhất", "value": f"{row['relevant_found']} / {row['unique_relevant']}"}
                 for row in description.get("pool_contribution", [])]
        written += _table(report_dir, "3.3", rows, ["item", "value"], ["Chỉ tiêu", "Giá trị"])

    frozen = config.get("frozen") or {}
    written += _table(report_dir, "3.4", [{"param": key, "value": value} for key, value in frozen.items()
                                          if key in ("alpha", "rrf_k", "adaptive_beta", "rerank_n", "best_single")],
                      ["param", "value"], ["Tham số", "Giá trị"])
    written += _table(report_dir, "3.5", [{"config": name, **metrics["overall"][name]} for name in main_names],
                      ["config", *metric_columns], ["Cấu hình", *metric_columns])

    categories = sorted({category for groups in metrics["by_category"].values() for category in groups})
    written += _table(report_dir, "3.6", [
        {"config": name, **{category: metrics["by_category"][name].get(category, {}).get(primary_metric) for category in categories}}
        for name in main_names
    ], ["config", *categories], ["Cấu hình", *categories])

    comparisons = read_csv(run / "comparisons.csv") if (run / "comparisons.csv").exists() else []
    comparison_rows = [
        {**row, **{key: float(row[key]) for key in ("diff", "ci_low", "ci_high", "p_value") if row[key] != ""},
         "p_holm": float(row["p_holm"]) if row["p_holm"] != "" else None}
        for row in comparisons if row["category"] == "all"
    ]
    written += _table(report_dir, "3.7", comparison_rows,
                      ["family", "system", "baseline", "metric", "diff", "ci_low", "ci_high", "p_value", "p_holm"],
                      ["Họ", "Hệ thống", "Đối chứng", "Chỉ số", "Chênh lệch", "CI thấp", "CI cao", "p", "p (Holm)"])

    if latency:
        written += _table(report_dir, "3.8", [
            {"config": name, "sparse": stages["sparse"]["mean"], "dense": stages["dense"]["mean"],
             "rerank": stages["rerank"]["mean"], "total": stages["total"]["mean"], "total_p95": stages["total"]["p95"]}
            for name, stages in latency["configs"].items()
        ], ["config", "sparse", "dense", "rerank", "total", "total_p95"],
            ["Cấu hình", "BM25", "Dense", "Rerank", "Tổng (TB)", "Tổng (P95)"])

    origins = sorted({origin for groups in metrics["by_origin"].values() for origin in groups})
    written += _table(report_dir, "3.9", [
        {"config": name, **{origin: metrics["by_origin"][name].get(origin, {}).get(primary_metric) for origin in origins}}
        for name in main_names
    ], ["config", *origins], ["Cấu hình", *origins])

    if errors:
        written += _table(report_dir, "3.10", errors["rows"], ["stage", "cause", "count"], ["Tầng thất bại", "Nguyên nhân", "Số truy vấn"])
    written += _table(report_dir, "3.11", [{"config": name, **metrics["overall"][name]} for name in exploration],
                      ["config", *metric_columns], ["Cấu hình", *metric_columns])

    tune_rows = read_csv(bench / "tune_results.csv") if (bench / "tune_results.csv").exists() else []

    def draw_alpha(axis) -> bool:
        series = defaultdict(list)
        for row in tune_rows:
            if row["param"] == "alpha":
                series[row["category"]].append((float(row["value"]), float(row["score"])))
        for category, points in sorted(series.items()):
            points.sort()
            axis.plot([x for x, _ in points], [y for _, y in points], marker="o", label=category,
                      linewidth=2.5 if category == "all" else 1.2)
        axis.set_xlabel("α (trọng số BM25)")
        axis.set_ylabel(primary_metric)
        if series:
            axis.legend(fontsize=8)
        return bool(series)

    def draw_tradeoff(axis) -> bool:
        points = sorted(
            (values["rerank_n"], metrics["overall"][name][primary_metric], latency["configs"][name]["total"]["mean"])
            for name, values in configs.items()
            if latency and values.get("rerank") and values.get("fusion") == "weighted"
            and name in metrics["overall"] and name in latency["configs"]
        )
        if len(points) < 2:
            return False
        axis.plot([n for n, _, _ in points], [q for _, q, _ in points], marker="o", color="tab:blue")
        axis.set_xlabel("N (số ứng viên rerank)")
        axis.set_ylabel(primary_metric, color="tab:blue")
        twin = axis.twinx()
        twin.plot([n for n, _, _ in points], [t for _, _, t in points], marker="s", color="tab:red")
        twin.set_ylabel("Độ trễ trung bình (ms)", color="tab:red")
        return True

    def draw_categories(axis) -> bool:
        if not categories or not main_names:
            return False
        width = 0.8 / len(main_names)
        for offset, name in enumerate(main_names):
            axis.bar([position + offset * width for position in range(len(categories))],
                     [metrics["by_category"][name].get(category, {}).get(primary_metric, 0.0) for category in categories],
                     width, label=name)
        axis.set_xticks([position + 0.4 - width / 2 for position in range(len(categories))], categories)
        axis.set_ylabel(primary_metric)
        axis.legend(fontsize=7, ncol=2)
        return True

    written += _figure(report_dir, "3.1", draw_alpha)
    written += _figure(report_dir, "3.2", draw_tradeoff)
    written += _figure(report_dir, "3.3", draw_categories)
    return written
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `.venv/Scripts/python -m pytest tests/test_eval_report.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/eval/report.py tests/test_eval_report.py
git commit -m "feat: export numbered Chapter 3 tables and figures from runs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Lệnh CLI `eval …`, trang Experiments, `run_rag_evaluation.py`

**Files:**
- Modify: `src/cli.py`, `pages/4_Experiments.py` (viết lại), `run_rag_evaluation.py`, `tests/test_pages.py`
- Create: `tests/test_cli_eval.py`

**Interfaces:**
- Consumes: mọi module `src/eval/*`, `resolve_index_dir`, `RetrievalIndex`.
- Produces lệnh:
  - `eval tune --config FILE [--force]`
  - `eval run --config FILE --split dev|test|all [--resume RUN_ID] [--only C1,C2] [--no-latency] [--dry-run]`
  - `eval compare --config FILE --run RUN_DIR`
  - `eval errors --config FILE --run RUN_DIR [--summarize FILE]` → `error_sample.csv`; `--summarize` → `errors_summary.json`
  - `eval report --config FILE --run RUN_DIR`

- [ ] **Step 1: Viết test thất bại `tests/test_cli_eval.py`**

Mỗi handler `eval` in đúng **một** JSON object ra stdout.
```python
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
```

Thêm vào `tests/test_pages.py`:
```python
def test_experiments_page_warns_on_lock_violation(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    run_dir = tmp_path / "runs" / "20260101T000000Z-test-abcd1234"
    (run_dir / "report").mkdir(parents=True)
    (run_dir / "status.json").write_text('{"status": "completed"}', encoding="utf-8")
    (run_dir / "config.json").write_text('{"split": "test", "lock": {"lock_violation": true}}', encoding="utf-8")
    (run_dir / "report" / "table_3_5.md").write_text("**Bảng 3.5. Kết quả tổng thể trên tập test**", encoding="utf-8")
    app = AppTest.from_file(str(PAGES / "4_Experiments.py"), default_timeout=30)
    app.session_state["user"] = {"username": "admin", "role": "admin"}
    app.run()
    assert not app.exception
    assert any("khóa" in warning.value for warning in app.warning)
    assert any("Bảng 3.5" in block.value for block in app.markdown)
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_cli_eval.py tests/test_pages.py -q`
Expected: FAIL (`invalid choice: 'eval'`; trang Experiments còn import code cũ).

- [ ] **Step 3: Thêm lệnh `eval` vào `src/cli.py`**

Thêm import:
```python
import yaml

from src.eval.compare import compare_run
from src.eval.errors import classify_failures, export_error_sample, summarize_causes
from src.eval.report import build_report
from src.eval.runner import default_pipeline_factory, load_benchmark, run_evaluation
from src.eval.spec import DEFAULT_FROZEN, load_spec, read_frozen, resolve_configs
from src.eval.tune import tune
```
Thêm các hàm xử lý:
```python
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
```
Trong `build_parser`, trước `return parser`:
```python
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
```

- [ ] **Step 4: Viết lại `pages/4_Experiments.py`**

```python
import json

import streamlit as st

from src.config import load_settings
from src.ui import database, require_role

st.set_page_config(page_title="Thực nghiệm", page_icon="📊", layout="wide")
require_role("admin")
settings = load_settings()
st.title("Thực nghiệm truy xuất")

active = database().get_active_corpus()
st.caption(f"Corpus đang active: {active['version_id'] if active else 'chưa có'}")
st.markdown(
    """
Quy trình (chạy trong terminal hoặc notebook Colab `notebooks/colab_pipeline.ipynb`):

```bash
python -m src.cli eval tune --config configs/experiment.yaml
python -m src.cli eval run --config configs/experiment.yaml --split test
python -m src.cli eval compare --config configs/experiment.yaml --run <RUN_DIR>
python -m src.cli eval errors --config configs/experiment.yaml --run <RUN_DIR>
python -m src.cli eval report --config configs/experiment.yaml --run <RUN_DIR>
```
"""
)

st.subheader("Các lần chạy")
runs = sorted((path for path in settings.runs_dir.glob("*") if path.is_dir()), reverse=True)
if not runs:
    st.info("Chưa có lần chạy nào.")
for run_dir in runs:
    with st.expander(run_dir.name, expanded=run_dir == runs[0]):
        status_path, config_path = run_dir / "status.json", run_dir / "config.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))["status"] if status_path.exists() else "unknown"
        record = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        st.write(f"Trạng thái: **{status}** — tập: **{record.get('split', '?')}**")
        if (record.get("lock") or {}).get("lock_violation"):
            st.warning("Vi phạm khóa tập test: frozen_params.yaml đã thay đổi sau lần chạy test đầu tiên.")
        for table in sorted((run_dir / "report").glob("table_*.md"), key=lambda path: int(path.stem.split("_")[-1])):
            st.markdown(table.read_text(encoding="utf-8"))
        for figure in sorted((run_dir / "report").glob("figure_*.png")):
            st.image(str(figure))
        for name in ("metrics.json", "comparisons.csv", "error_sample.csv"):
            path = run_dir / name
            if path.exists():
                st.download_button(f"Tải {name}", path.read_bytes(), file_name=name, key=f"{run_dir.name}-{name}")
```

- [ ] **Step 5: Sửa `run_rag_evaluation.py` cho spec mới**

Thay import `from src.config import load_settings, load_yaml` bằng `from src.config import load_settings` và thêm:
```python
from src.eval.runner import load_benchmark
from src.eval.spec import load_spec
from src.indexing import resolve_index_dir
from src.storage import Database
```
Trong `generate`, thay các dòng `config = load_yaml(config_path)` … `queries = _jsonl(Path(config["queries"]))` và `seed = int(config.get("seed", 42))` bằng:
```python
    settings = load_settings()
    if not settings.openai_api_key or not settings.openai_model:
        raise ValueError("OPENAI_API_KEY and OPENAI_MODEL are required")
    spec = load_spec(config_path, settings)
    database = Database(settings.db_path)
    database.initialize()
    pipeline = RetrievalPipeline(RetrievalIndex.load(spec.index_dir or resolve_index_dir(None, settings, database)))
    queries, _, _ = load_benchmark(spec.bench_dir, "test")
    seed = spec.seed
```
và `output_dir = Path(config.get("runs_dir", settings.runs_dir)) / (...)` thành `output_dir = spec.runs_dir / (...)`. Xóa hàm `_jsonl` nếu không còn dùng.

- [ ] **Step 6: Chạy test**

Run: `.venv/Scripts/python -m pytest tests/test_cli_eval.py tests/test_pages.py -q`
Expected: `5 passed`

Run: `.venv/Scripts/python -m pytest -q`
Expected: tất cả pass.

- [ ] **Step 7: Commit**

```bash
git add src/cli.py pages/4_Experiments.py run_rag_evaluation.py tests/test_cli_eval.py tests/test_pages.py
git commit -m "feat: add eval CLI and show run reports on the Experiments page

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Khung Chương 3 trong báo cáo, notebook Colab, tài liệu

**Files:**
- Modify: `b_o_c_o_nh_m_3.md`, `README.md`, `CLAUDE.md`
- Create: `notebooks/colab_pipeline.ipynb`, `tests/test_notebook.py`

**Interfaces:**
- Consumes: `TABLES`, `FIGURES` (Task 8), `build_parser` (Task 9).

- [ ] **Step 1: Viết test thất bại `tests/test_notebook.py`**

```python
import json
import re
import shlex
from pathlib import Path

import pytest

from src.cli import build_parser
from src.eval.report import FIGURES, TABLES

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "b_o_c_o_nh_m_3.md"


def test_notebook_cli_commands_parse():
    notebook = json.loads((ROOT / "notebooks" / "colab_pipeline.ipynb").read_text(encoding="utf-8"))
    commands = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for line in "".join(cell["source"]).splitlines():
            if "python -m src.cli" in line:
                commands.append(shlex.split(line.split("python -m src.cli", 1)[1].split("#")[0]))
    assert len(commands) >= 12
    parser = build_parser()
    for arguments in commands:
        parser.parse_args(arguments)


@pytest.mark.skipif(not REPORT.exists(), reason="report file is not in this checkout")
def test_report_chapter_three_references_every_table_and_figure():
    text = REPORT.read_text(encoding="utf-8")
    assert "# CHƯƠNG 3" in text
    for key, title in TABLES.items():
        assert f"Bảng {key}. {title}" in text
    for key, title in FIGURES.items():
        assert f"Hình {key}. {title}" in text
    assert text.index("# CHƯƠNG 3") < text.index("# DANH MỤC TÀI LIỆU THAM KHẢO")
    numbers = re.findall(r"\[\[ĐIỀN:", text)
    assert len(numbers) >= 20
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_notebook.py -q`
Expected: FAIL (`FileNotFoundError` notebook; chưa có Chương 3).

- [ ] **Step 3: Cập nhật mục lục và danh mục trong `b_o_c_o_nh_m_3.md`**

Trong MỤC LỤC, chèn trước dòng `* DANH MỤC TÀI LIỆU THAM KHẢO`:
```markdown
* CHƯƠNG 3: THỰC NGHIỆM VÀ ĐÁNH GIÁ KẾT QUẢ
  * 3.1. Môi trường và thiết lập thực nghiệm
  * 3.2. Bộ dữ liệu và benchmark
  * 3.3. Tinh chỉnh siêu tham số trên tập dev
  * 3.4. Kết quả trên tập test
    * 3.4.1. RQ1 – So sánh BM25 và Dense Retrieval theo nhóm truy vấn
    * 3.4.2. RQ2 – Hiệu quả của Hybrid Retrieval
    * 3.4.3. RQ3 – Đóng góp và chi phí của Reranker
    * 3.4.4. Phân tích độ nhạy theo nguồn câu hỏi
  * 3.5. Phân tích lỗi
  * 3.6. Các hướng khai thác mở rộng
  * 3.7. Thảo luận và các yếu tố ảnh hưởng tính hợp lệ
  * 3.8. Kết luận chương
```
Trong DANH MỤC CÁC BẢNG, thêm sau dòng Bảng 2.2 (tiêu đề phải khớp nguyên văn `TABLES`):
```markdown
* **Bảng 3.1.** Môi trường và thiết lập thực nghiệm
* **Bảng 3.2.** Thống kê bộ benchmark theo nhóm, nguồn và tập
* **Bảng 3.3.** Độ đồng thuận dán nhãn và đóng góp của từng hệ thống vào pool
* **Bảng 3.4.** Tham số tối ưu chọn trên tập dev
* **Bảng 3.5.** Kết quả tổng thể trên tập test
* **Bảng 3.6.** Kết quả theo nhóm truy vấn trên tập test (RQ1)
* **Bảng 3.7.** So sánh ghép cặp có kiểm định thống kê (RQ2, RQ3, X2)
* **Bảng 3.8.** Độ trễ theo tầng xử lý (ms)
* **Bảng 3.9.** Kết quả theo nguồn câu hỏi (phân tích độ nhạy)
* **Bảng 3.10.** Phân bố truy vấn thất bại theo tầng và nguyên nhân
* **Bảng 3.11.** Kết quả các hướng khai thác X3–X6
```
Trong DANH MỤC CÁC HÌNH, thêm sau Hình 1.6:
```markdown
* **Hình 3.1.** Ảnh hưởng của α đến chỉ số chính trên tập dev
* **Hình 3.2.** Đánh đổi chất lượng – độ trễ theo số ứng viên rerank N
* **Hình 3.3.** Chỉ số chính theo nhóm truy vấn trên tập test
```
Ngay sau Bảng 2.2 (sau dòng `| **C4** | … | Đóng góp của Reranker |`), thêm đoạn:
```markdown

*Ghi chú:* C3 và C4 được triển khai thành hai biến thể theo cơ chế dung hợp: C3-RRF/C3-WS và C4-RRF/C4-WS. Ngoài ra, cấu hình đối chứng X1 (Dense + Reranker) được bổ sung để tách riêng đóng góp của Reranker khỏi đóng góp của bước dung hợp (xem mục 3.6).
```

- [ ] **Step 4: Chèn Chương 3 vào `b_o_c_o_nh_m_3.md`**

Chèn khối sau ngay **trước** dòng `# DANH MỤC TÀI LIỆU THAM KHẢO` (giữ dòng `---` phân cách):

```markdown
# CHƯƠNG 3: THỰC NGHIỆM VÀ ĐÁNH GIÁ KẾT QUẢ

Chương này trình bày quá trình triển khai thực nghiệm theo quy trình chuẩn tắc đã thiết kế ở mục 2.6: mô tả môi trường và bộ dữ liệu, tinh chỉnh siêu tham số trên tập dev, đánh giá chính thức các cấu hình C1–C4 trên tập test để trả lời ba câu hỏi nghiên cứu RQ1–RQ3, phân tích lỗi, và trình bày các hướng khai thác mở rộng X1–X6. Toàn bộ số liệu trong chương được sinh tự động bởi công cụ `python -m src.cli eval …`; mỗi bảng/hình ghi rõ tệp nguồn để bảo đảm khả năng tái lập.

## 3.1. Môi trường và thiết lập thực nghiệm

Thực nghiệm được thực hiện trên [[ĐIỀN: nền tảng và GPU, ví dụ Google Colab Pro – GPU … — nguồn: runs/<RUN_ID>/latency.json, mục hardware]]. Các thành phần được cố định như sau: mô hình embedding BAAI/bge-m3 (1024 chiều, chuẩn hóa L2, tìm kiếm chính xác bằng tích vô hướng), mô hình reranker BAAI/bge-reranker-v2-m3, BM25 với k1 = 1,5 và b = 0,75, mỗi nhánh truy xuất trả về top-L = 100 ứng viên, số chunk đưa vào ngữ cảnh K = 5, seed = 42. Kho tài liệu được phân đoạn theo cấu trúc (structure-aware) với tối đa 350 từ mỗi chunk, chồng lấn 50 từ, và gắn tiền tố `[Tài liệu] > [Chương] > [Mục]`.

**Bảng 3.1. Môi trường và thiết lập thực nghiệm**

[[ĐIỀN: dán bảng — nguồn: runs/<RUN_ID>/report/table_3_1.md (lệnh `python -m src.cli eval report`)]]

## 3.2. Bộ dữ liệu và benchmark

### 3.2.1. Kho tài liệu học tập

Kho tài liệu gồm [[ĐIỀN: số tài liệu, loại (giáo trình/slide/đề thi), môn học, tổng số trang — nguồn: trang Documents / data/processed/<version>/manifest.json]], sau phân đoạn thu được [[ĐIỀN: số chunk — nguồn: data/indexes/<version>/index_meta.json]] chunk.

### 3.2.2. Xây dựng tập truy vấn

Tập truy vấn được xây dựng theo hai nguồn. (1) Câu hỏi nháp do mô hình ngôn ngữ [[ĐIỀN: tên mô hình LLM — nguồn: data/benchmark/drafts.jsonl, trường generator]] sinh từ các chunk được lấy mẫu phân tầng theo môn học, loại tài liệu và tài liệu, với câu lệnh riêng cho bốn nhóm truy vấn (mục 2.2.2). Mỗi câu hỏi nháp phải vượt qua kiểm tra tự động: nhóm định danh phải chứa mã/ký hiệu có trong chunk; nhóm diễn đạt lại phải có độ trùng từ vựng (Jaccard trên từ nội dung) với chunk nguồn không quá 0,2; đoạn trích căn cứ phải xuất hiện nguyên văn trong chunk. Sau đó thành viên nhóm rà soát từng câu (giữ/sửa/loại). (2) Câu hỏi do người viết trực tiếp mà không nhìn tài liệu, nhằm mô phỏng câu hỏi thực của người học và giảm thiên lệch từ vựng của câu hỏi sinh tự động. Các câu gần trùng lặp (cosine ≥ 0,92) được loại bỏ. Tập cuối cùng gồm [[ĐIỀN: tổng số câu hỏi, số câu mỗi nhóm, tỉ lệ câu do người viết — nguồn: data/benchmark/benchmark_description.json]].

**Bảng 3.2. Thống kê bộ benchmark theo nhóm, nguồn và tập**

[[ĐIỀN: dán bảng — nguồn: runs/<RUN_ID>/report/table_3_2.md]]

### 3.2.3. Gán nhãn mức độ liên quan và độ tin cậy

Để chống thiên vị pooling (mục 2.4.3), với mỗi câu hỏi, tập ứng viên được hợp nhất từ top-15 của sáu hệ thống (C1, C2, C3-RRF, C3-WS, C4-WS, X1), chunk nguồn và các chunk do người dán nhãn tự tìm thêm. Tệp dán nhãn được xáo trộn và ẩn tên hệ thống. Hai người dán nhãn độc lập theo thang 0/1/2 trên [[ĐIỀN: số cặp/tỉ lệ câu hỏi được dán nhãn đôi — nguồn: data/benchmark/agreement.json]]; độ đồng thuận Cohen's κ có trọng số bậc hai đạt [[ĐIỀN: κ — nguồn: data/benchmark/agreement.json]], [[ĐIỀN: diễn giải theo thang Landis–Koch]]. Các trường hợp bất đồng được cả nhóm thống nhất lần cuối. Mỗi nhãn liên quan kèm đoạn trích căn cứ, cho phép ánh xạ nhãn sang các cách phân đoạn khác (mục 3.6.4).

**Bảng 3.3. Độ đồng thuận dán nhãn và đóng góp của từng hệ thống vào pool**

[[ĐIỀN: dán bảng — nguồn: runs/<RUN_ID>/report/table_3_3.md]]

[[ĐIỀN: nhận xét về số chunk liên quan chỉ một hệ thống tìm thấy — bằng chứng pooling đa nguồn là cần thiết]]

### 3.2.4. Phân tách dev/test

Tập câu hỏi được chia phân tầng theo nhóm truy vấn thành dev (30%) và test (70%) với seed 42; câu hỏi không có chunk liên quan bị loại ([[ĐIỀN: số câu bị loại — nguồn: kết quả lệnh `bench split`]]). Mã băm của tệp truy vấn, qrels và tham số tối ưu được ghi vào `benchmark_manifest.json`; công cụ đánh giá ghi nhận mọi thay đổi tham số sau lần chạy test đầu tiên (trạng thái khóa: [[ĐIỀN: có/không vi phạm — nguồn: runs/<RUN_ID>/config.json, trường lock]]).

## 3.3. Tinh chỉnh siêu tham số trên tập dev

Chỉ số chính MRR@10 được khai báo trước khi thực nghiệm. Trên tập dev, hệ thống quét α ∈ {0; 0,1; …; 1} cho Weighted Sum, k ∈ {10, 20, 40, 60, 100} cho RRF, β ∈ {0,1; 0,2; 0,3; 0,5} cho dung hợp thích nghi và N ∈ {10, 20, 30, 50} cho reranker; khi hòa điểm, giá trị gần mặc định được chọn.

**Bảng 3.4. Tham số tối ưu chọn trên tập dev**

[[ĐIỀN: dán bảng — nguồn: runs/<RUN_ID>/report/table_3_4.md hoặc data/benchmark/frozen_params.yaml]]

**Hình 3.1. Ảnh hưởng của α đến chỉ số chính trên tập dev**

[[ĐIỀN: chèn hình — nguồn: runs/<RUN_ID>/report/figure_3_1.png]]

[[ĐIỀN: nhận xét — α tối ưu có khác nhau giữa các nhóm truy vấn không (ví dụ nhóm định danh ưa α cao, nhóm diễn đạt lại ưa α thấp)? Đây là căn cứ cho hướng X2]]

## 3.4. Kết quả trên tập test

**Bảng 3.5. Kết quả tổng thể trên tập test**

[[ĐIỀN: dán bảng — nguồn: runs/<RUN_ID>/report/table_3_5.md]]

### 3.4.1. RQ1 – So sánh BM25 và Dense Retrieval theo nhóm truy vấn

**Bảng 3.6. Kết quả theo nhóm truy vấn trên tập test (RQ1)**

[[ĐIỀN: dán bảng — nguồn: runs/<RUN_ID>/report/table_3_6.md]]

**Hình 3.3. Chỉ số chính theo nhóm truy vấn trên tập test**

[[ĐIỀN: chèn hình — nguồn: runs/<RUN_ID>/report/figure_3_3.png]]

[[ĐIỀN: kết luận giả thuyết RQ1 — BM25 có vượt trội ở nhóm định danh và Dense ở nhóm khái niệm/diễn đạt lại không; nêu chênh lệch và khoảng tin cậy theo nhóm từ comparisons.csv]]

### 3.4.2. RQ2 – Hiệu quả của Hybrid Retrieval

**Bảng 3.7. So sánh ghép cặp có kiểm định thống kê (RQ2, RQ3, X2)**

[[ĐIỀN: dán bảng — nguồn: runs/<RUN_ID>/report/table_3_7.md (lệnh `python -m src.cli eval compare` trước `eval report`)]]

[[ĐIỀN: kết luận giả thuyết RQ2 — C3-RRF/C3-WS so với cấu hình đơn lẻ tốt nhất (chọn trên dev): chênh lệch MRR@10, CI 95% bootstrap, p đã hiệu chỉnh Holm]]

### 3.4.3. RQ3 – Đóng góp và chi phí của Reranker

**Bảng 3.8. Độ trễ theo tầng xử lý (ms)**

[[ĐIỀN: dán bảng — nguồn: runs/<RUN_ID>/report/table_3_8.md]]

**Hình 3.2. Đánh đổi chất lượng – độ trễ theo số ứng viên rerank N**

[[ĐIỀN: chèn hình — nguồn: runs/<RUN_ID>/report/figure_3_2.png]]

[[ĐIỀN: kết luận giả thuyết RQ3 — mức tăng MRR@10/NDCG@10 của C4 so với C3 và của X1 so với C2 (Bảng 3.7), đổi lại độ trễ tăng bao nhiêu ms (trung bình và P95)]]

### 3.4.4. Phân tích độ nhạy theo nguồn câu hỏi

**Bảng 3.9. Kết quả theo nguồn câu hỏi (phân tích độ nhạy)**

[[ĐIỀN: dán bảng — nguồn: runs/<RUN_ID>/report/table_3_9.md]]

[[ĐIỀN: thứ hạng các cấu hình có giữ nguyên giữa câu hỏi do LLM sinh và câu hỏi do người viết không? Nếu có, kết luận của RQ1–RQ3 vững hơn trước thiên lệch của câu hỏi sinh tự động]]

## 3.5. Phân tích lỗi

Với cấu hình mục tiêu C4-WS, các truy vấn không có chunk liên quan nào trong top-10 được phân loại tự động theo tầng gây lỗi dựa trên điểm phân rã (Bảng 2.1): (i) *first_stage_miss* – cả BM25 và Dense đều không đưa chunk đúng vào top-L; (ii) *fusion_demoted* – chunk đúng có trong top-L nhưng bị đẩy ra ngoài top-N sau dung hợp; (iii) *rerank_demoted* – chunk đúng có trong top-N nhưng bị Reranker hạ xuống ngoài top-10. Một mẫu tối đa 50 truy vấn thất bại được nhóm gắn nhãn nguyên nhân thủ công: lỗi trích xuất/OCR, chunk cắt ngang ý, tách từ, lệch từ vựng, cần suy luận nhiều bước, nhãn qrels sai, khác.

**Bảng 3.10. Phân bố truy vấn thất bại theo tầng và nguyên nhân**

[[ĐIỀN: dán bảng — nguồn: runs/<RUN_ID>/report/table_3_10.md (lệnh `eval errors`, gắn nhãn cột cause trong error_sample.csv, rồi `eval errors --summarize`)]]

[[ĐIỀN: 2–3 ví dụ lỗi tiêu biểu kèm câu hỏi, chunk đúng, chunk xếp đầu và nguyên nhân]]

## 3.6. Các hướng khai thác mở rộng

Phần này trình bày các thí nghiệm bổ sung nhằm khai thác các khoảng trống nghiên cứu đã nêu ở mục 1.2.3, vượt ra ngoài ma trận C1–C4.

### 3.6.1. X1 – Dense + Reranker: tách đóng góp của Reranker

So sánh C4 với C2 gộp chung hai tác động (dung hợp và tái xếp hạng). Cấu hình X1 áp dụng Reranker trực tiếp lên Dense Retrieval, cho phép tách riêng: đóng góp của Reranker (X1 so với C2) và giá trị gia tăng của dung hợp khi đã có Reranker (C4 so với X1). [[ĐIỀN: kết quả — nguồn: Bảng 3.5 và Bảng 3.7]]

### 3.6.2. X2 – Dung hợp thích nghi theo truy vấn

Thay vì một α cố định, α được điều chỉnh theo đặc trưng từ vựng của truy vấn: α = clamp(α₀ + β·(2·s − 1), 0,1, 0,9), trong đó s ∈ [0, 1] tổng hợp tỉ lệ chữ số, tỉ lệ ký hiệu, sự hiện diện của mã định danh và IDF lớn nhất (chuẩn hóa) của các từ trong truy vấn. Truy vấn giàu định danh nhận α lớn (ưu tiên BM25), truy vấn ngữ nghĩa nhận α nhỏ (ưu tiên Dense). Hướng này trực tiếp giải quyết khoảng trống (2) về tối ưu hóa trọng số dung hợp. [[ĐIỀN: kết quả X2/X2-R so với C3-WS/C4-WS — nguồn: Bảng 3.7]]

### 3.6.3. X3 – Ảnh hưởng của tách từ tiếng Việt đến BM25

So sánh tách từ theo khoảng trắng (âm tiết) với tách từ ghép bằng pyvi và VnCoreNLP [26] cho C1 và C3-WS. [[ĐIỀN: kết quả — nguồn: Bảng 3.11]]

### 3.6.4. X4 – Structure-aware chunking và tiền tố tiêu đề

So sánh phân đoạn theo cấu trúc với phân đoạn cửa sổ cố định (450 từ, chồng lấn 75), và bật/tắt tiền tố `[Tài liệu] > [Chương] > [Mục]`. Nhãn qrels được ánh xạ tự động sang từng cách phân đoạn thông qua đoạn trích căn cứ, nên không cần dán nhãn lại. [[ĐIỀN: kết quả — nguồn: Bảng 3.11]]

### 3.6.5. X5 – Số ứng viên đưa vào Reranker

Quét N ∈ {10, 20, 30, 50} để xác định điểm cân bằng giữa chất lượng và độ trễ. [[ĐIỀN: kết quả — nguồn: Hình 3.2 và Bảng 3.11]]

### 3.6.6. X6 – Độ nhạy theo mô hình embedding

Thay BGE-M3 bằng [[ĐIỀN: tên mô hình embedding tiếng Việt được chọn]] cho C2 và C4-WS để kiểm tra kết luận có phụ thuộc vào mô hình embedding hay không. [[ĐIỀN: kết quả — nguồn: Bảng 3.11]]

**Bảng 3.11. Kết quả các hướng khai thác X3–X6**

[[ĐIỀN: dán bảng — nguồn: runs/<RUN_ID>/report/table_3_11.md]]

## 3.7. Thảo luận và các yếu tố ảnh hưởng tính hợp lệ

### 3.7.1. Thảo luận

[[ĐIỀN: tổng hợp câu trả lời RQ1–RQ3, đối chiếu với các công trình ở Bảng 1.1 (Strich và cộng sự, 2026; Lyu và cộng sự, 2024; Lian, 2026), và đề xuất cấu hình tối ưu cho học liệu tiếng Việt (nhiệm vụ 5, mục 1.3.2)]]

### 3.7.2. Các yếu tố ảnh hưởng tính hợp lệ

* **Nguồn câu hỏi:** phần lớn câu hỏi được sinh bởi LLM từ chính các chunk nên có thể thiên về khớp từ vựng; ảnh hưởng này được kiểm tra bằng phân tích độ nhạy theo nguồn câu hỏi (Bảng 3.9) và kiểm tra độ trùng từ vựng với nhóm diễn đạt lại.
* **Cỡ mẫu theo nhóm:** mỗi nhóm truy vấn trên tập test có khoảng [[ĐIỀN: số câu mỗi nhóm]] câu, nên khoảng tin cậy theo nhóm rộng; kết luận theo nhóm chỉ mang tính định hướng.
* **Độ sâu pooling:** chunk liên quan nằm ngoài top-15 của mọi hệ thống và không được tìm thủ công sẽ không có nhãn, có thể làm giảm Recall tuyệt đối (nhưng tác động như nhau lên các cấu hình).
* **Phạm vi dữ liệu:** thực nghiệm trên [[ĐIỀN: số môn học]] môn học của một cơ sở đào tạo; khả năng khái quát sang môn học/cơ sở khác cần được kiểm chứng.
* **Phần cứng:** độ trễ đo trên GPU của Colab có biến động giữa các phiên; số liệu độ trễ dùng để so sánh tương đối giữa các cấu hình trong cùng một lần chạy.

## 3.8. Kết luận chương

[[ĐIỀN: tóm tắt kết quả chính của RQ1–RQ3, cấu hình khuyến nghị, đóng góp của các hướng khai thác X1–X6 và hạn chế còn lại]]

---
```

- [ ] **Step 5: Tạo `notebooks/colab_pipeline.ipynb`**

Lưu script sau vào file tạm (ví dụ trong scratchpad, **không** commit), chạy bằng `PYTHONIOENCODING=utf-8 .venv/Scripts/python <đường-dẫn-script>` từ thư mục gốc repo; chỉ commit file `.ipynb` sinh ra.
```python
import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Quy trình thực nghiệm Hybrid Retrieval trên Colab
Chạy các cell theo thứ tự. Các bước có nhãn **[NGƯỜI]** cần tải file về, chỉnh sửa, rồi tải lên lại đúng thư mục trên Google Drive.
"""),
    code("""
from google.colab import drive
drive.mount('/content/drive')
"""),
    code("""
%cd /content
!git clone https://github.com/huuhieu56/PPL.git || (cd PPL && git pull)
%cd /content/PPL
!pip install -q -r requirements.in
!apt-get -qq install -y openjdk-17-jre-headless  # chỉ cần cho tokenizer VnCoreNLP (X3)
"""),
    code("""
import os
from google.colab import userdata
os.environ['PPL_DATA_DIR'] = '/content/drive/MyDrive/ppl-data/data'
os.environ['PPL_RUNS_DIR'] = '/content/drive/MyDrive/ppl-data/runs'
os.environ['PPL_VNCORENLP_DIR'] = '/content/drive/MyDrive/ppl-data/vncorenlp'
os.environ['PYTHONIOENCODING'] = 'utf-8'
for key in ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL'):
    os.environ[key] = userdata.get(key)
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'KHÔNG CÓ GPU')
"""),
    md("""
## 1. Lập chỉ mục
Đặt tài liệu PDF/DOCX/PPTX vào `MyDrive/ppl-data/docs/`. Tùy chọn: `docs_metadata.csv` (cột `filename, course, source_type, doc_title`), thêm `--metadata` vào lệnh.
Thay `VERSION` ở lệnh thứ hai bằng `version_id` vừa in ra (chỉ cần khi chạy X3 với VnCoreNLP).
"""),
    code("""
!python -m src.cli index build --input /content/drive/MyDrive/ppl-data/docs --course CS101 --tokenizers whitespace,pyvi
!python -m src.cli index add-tokenizer --index $PPL_DATA_DIR/indexes/VERSION --tokenizer vncorenlp  # thay VERSION
"""),
    md("""
## 2. Sinh câu hỏi nháp và rà soát
**[NGƯỜI]** Sau `review-export`: tải `data/benchmark/review.csv` về, điền cột `action` (keep/edit/drop), `new_text`, `new_category`.
Viết thêm `human_queries.csv` (cột `text, category`) khoảng 25–30% tổng số câu, không nhìn tài liệu khi viết. Tải cả hai lên `MyDrive/ppl-data/data/benchmark/`.
"""),
    code("""
!python -m src.cli bench generate --per-category 60
!python -m src.cli bench review-export
"""),
    code("""
!python -m src.cli bench review-import --human $PPL_DATA_DIR/benchmark/human_queries.csv
"""),
    md("""
## 3. Pooling và dán nhãn
**[NGƯỜI]** Hai người dán nhãn độc lập `annotation_A.csv` và `annotation_B.csv`: cột `relevance` (0/1/2), cột `evidence_quote` (trích nguyên văn đoạn căn cứ khi relevance ≥ 1).
Nếu lệnh agreement đầu tiên báo còn bất đồng: điền cột `final` trong `disagreements.csv`, rồi chạy lệnh thứ hai.
"""),
    code("""
!python -m src.cli bench pool --depth 15 --annotators A,B
"""),
    code("""
!python -m src.cli bench agreement --annotations $PPL_DATA_DIR/benchmark/annotation_A.csv $PPL_DATA_DIR/benchmark/annotation_B.csv
!python -m src.cli bench agreement --annotations $PPL_DATA_DIR/benchmark/annotation_A.csv $PPL_DATA_DIR/benchmark/annotation_B.csv --resolved $PPL_DATA_DIR/benchmark/disagreements.csv
"""),
    code("""
!python -m src.cli bench split --dev 0.3 --seed 42
!python -m src.cli bench describe
"""),
    md("""
## 4. Tinh chỉnh trên dev và đánh giá trên test
Chỉ chạy `eval run --split test` sau khi đã chốt `frozen_params.yaml`. Mọi thay đổi tham số sau lần chạy test đầu tiên đều bị ghi nhận là vi phạm khóa.
"""),
    code("""
!python -m src.cli eval run --config configs/experiment.yaml --split test --dry-run
!python -m src.cli eval tune --config configs/experiment.yaml
!python -m src.cli eval run --config configs/experiment.yaml --split test
"""),
    code("""
import glob
RUN = sorted(glob.glob(os.environ['PPL_RUNS_DIR'] + '/*-test-*'))[-1]
print(RUN)
!python -m src.cli eval compare --config configs/experiment.yaml --run {RUN}
!python -m src.cli eval errors --config configs/experiment.yaml --run {RUN}
"""),
    md("""
## 5. Phân tích lỗi và báo cáo
**[NGƯỜI]** Tải `error_sample.csv` trong thư mục run về, điền cột `cause` (extraction, chunk_boundary, tokenization, vocabulary_mismatch, multi_hop, label_error, other), tải lên lại rồi chạy cell dưới.
"""),
    code("""
!python -m src.cli eval errors --config configs/experiment.yaml --run {RUN} --summarize {RUN}/error_sample.csv
!python -m src.cli eval report --config configs/experiment.yaml --run {RUN}
from IPython.display import Markdown, Image, display
for path in sorted(glob.glob(RUN + '/report/table_*.md'), key=lambda p: int(p.rsplit('_', 1)[1][:-3])):
    display(Markdown(open(path, encoding='utf-8').read()))
for path in sorted(glob.glob(RUN + '/report/figure_*.png')):
    display(Image(path))
"""),
]
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                 "language_info": {"name": "python"}, "accelerator": "GPU"},
    "cells": cells,
}
Path("notebooks").mkdir(exist_ok=True)
Path("notebooks/colab_pipeline.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(len(cells), "cells")
```
Expected: in `18 cells`. (`test_notebook_cli_commands_parse` cắt phần sau `python -m src.cli` và bỏ comment `#`; `{RUN}` và `$PPL_DATA_DIR/...` là chuỗi hợp lệ với argparse.)

- [ ] **Step 6: Chạy test notebook và báo cáo**

Run: `.venv/Scripts/python -m pytest tests/test_notebook.py -q`
Expected: `2 passed`

- [ ] **Step 7: Cập nhật `README.md` và `CLAUDE.md`**

`README.md`:
- Thay mục "Run retrieval experiments" và "Evaluate RAG answers" bằng mục "Run experiments" liệt kê: `eval run --dry-run`, `eval tune`, `eval run --split test`, `eval compare`, `eval errors` (+ `--summarize`), `eval report`; mô tả output `runs/<RUN_ID>/` (`config.json`, `per_query.jsonl`, `metrics.json`, `metrics_per_query.csv`, `latency.json`, `comparisons.csv`, `error_sample.csv`, `report/table_3_*.md`, `report/figure_3_*.png`); ghi rõ quy tắc khóa test; nhắc `run_rag_evaluation.py --config configs/experiment.yaml` vẫn dùng được cho chấm câu trả lời LLM.
- Thêm mục "Google Colab": mở `notebooks/colab_pipeline.ipynb`, đặt Colab Secrets `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`; dữ liệu trên Drive qua `PPL_DATA_DIR`/`PPL_RUNS_DIR`; cài bằng `requirements.in`.
- Mục "Verification":
```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m src.cli eval run --config configs/experiment.example.yaml --split test --dry-run
.venv/bin/python -m compileall -q app.py pages src run_rag_evaluation.py
```

`CLAUDE.md`:
- Commands: thay mọi lệnh `run_experiments.py` bằng các lệnh `python -m src.cli eval …`; lệnh verification như README.
- Architecture: thêm đoạn "Evaluation (`src/eval/`)": `spec.load_spec` (YAML `configs/experiment.yaml`, giá trị `frozen` lấy từ `<bench_dir>/frozen_params.yaml`, placeholder `${DATA_DIR}`/`${RUNS_DIR}`) → `runner.run_evaluation` (checkpoint `per_query.jsonl` lưu điểm từng tầng theo `RESULT_FIELDS`, resume, `check_test_lock` khi split test, qrels remap cho cấu hình có index khác, latency không cache) → `tune`/`compare`/`errors`/`report`; đánh số bảng/hình Chương 3 nằm ở `src/eval/report.TABLES/FIGURES` và phải khớp `b_o_c_o_nh_m_3.md` (có test).
- Gotchas: xóa ghi chú "`src/experiments.py` là cầu nối tạm"; thêm: Colab cài từ `requirements.in` vì `requirements.txt` ghim torch CPU.

- [ ] **Step 8: Kiểm tra cuối**

Run:
```bash
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m src.cli eval run --config configs/experiment.example.yaml --split test --dry-run
.venv/Scripts/python -m compileall -q app.py pages src run_rag_evaluation.py
```
Expected: pytest toàn bộ pass; dry-run in JSON `"status": "valid"` với `"warnings"` chứa `index not built at examples/index` (index ví dụ không phải index thật); compileall không lỗi.

- [ ] **Step 9: Commit**

`b_o_c_o_nh_m_3.md` hiện chưa được theo dõi bởi git — **hỏi người dùng trước** có muốn commit báo cáo vào repo không. Nếu có:
```bash
git add b_o_c_o_nh_m_3.md notebooks/colab_pipeline.ipynb tests/test_notebook.py README.md CLAUDE.md
```
Nếu không, bỏ `b_o_c_o_nh_m_3.md` khỏi lệnh `git add` (test báo cáo sẽ tự skip ở máy khác).
```bash
git commit -m "docs: add Chapter 3 skeleton, Colab notebook and evaluation workflow docs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

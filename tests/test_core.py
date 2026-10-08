import json
from dataclasses import replace
from types import SimpleNamespace

from pptx import Presentation
from pptx.util import Inches, Pt
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import pymupdf

from src.config import load_settings
from src.chunking import build_hierarchy, validate_hierarchy
from src.corpus import build_corpus
from src.documents import PageText, extract_document, usable_ocr
from src.models import Chunk, RagConfig, SearchResult
from src.rag import answer_question
from src.reranking import _load_model
from src.retrieval import adaptive_alpha, fuse_weighted, minmax_scores, retrieve
from src.storage import Database


def test_foundation_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-persisted")
    settings = load_settings(tmp_path)
    assert "must-not-be-persisted" not in repr(settings)
    db = Database(settings.db_path)
    db.initialize()
    db.save_rag_config("demo", RagConfig(method="rrf", use_reranker=False))

    saved = db.list_rag_configs()
    assert saved[0]["name"] == "demo"
    assert saved[0]["config"]["method"] == "rrf"
    assert "must-not-be-persisted" not in json.dumps(saved)


def test_interactive_default_uses_fast_measured_baseline():
    assert RagConfig().method == "dense"
    assert RagConfig().use_reranker is True


def test_greeting_does_not_start_retrieval_or_llm(monkeypatch):
    monkeypatch.setattr("src.rag.retrieve", lambda *_args: (_ for _ in ()).throw(AssertionError("retrieval called")))
    answer = answer_question("hi", object(), RagConfig(), object(), "unused")
    assert "hỏi" in answer.text.lower()
    assert answer.citations == []


def test_rag_uses_expanded_parent_context_and_cites_its_full_span(monkeypatch):
    leaf = Chunk("leaf", "doc", "Môn học", "slide", 1, "Bài 1", "Câu mở đầu", 1, "parent", 0, 3)
    monkeypatch.setattr("src.rag.retrieve", lambda *_args: [SearchResult(leaf, 0.8, 1, "dense")])

    class Index:
        def expand_chunk(self, chunk, max_words):
            assert chunk.chunk_id == "leaf" and max_words == 20
            return replace(chunk, text="Câu mở đầu. Nội dung bổ sung ở trang sau.", page_end=2)

    class Completions:
        def invoke(self, messages, **kwargs):
            assert "Nội dung bổ sung ở trang sau" in messages[1]["content"]
            return SimpleNamespace(text="Trả lời [1].", usage_metadata=None)

    client = Completions()
    answer = answer_question("Hỏi về bài 1", Index(), RagConfig(context_parent_words=20), client, "test")
    assert answer.citations[0]["page_end"] == 2
    assert "Nội dung bổ sung" in answer.citations[0]["text"]


def test_rag_does_not_repeat_overlapping_parent_context(monkeypatch):
    first = Chunk("a", "d1", "M", "slide", 1, "Mục", "A", 1, "parent", 0, 5)
    second = Chunk("b", "d1", "M", "slide", 2, "Mục", "B", 2, "parent", 4, 9)
    third = Chunk("c", "d2", "M", "slide", 3, "Mục khác", "C", 3, "other", 0, 5)
    monkeypatch.setattr("src.rag.retrieve", lambda *_args: [SearchResult(chunk, 1.0 - rank / 10, rank, "dense") for rank, chunk in enumerate((first, second, third), 1)])

    class Index:
        def expand_chunk(self, chunk, _budget):
            return replace(chunk, text="A B" if chunk.parent_id == "parent" else "C", word_start=0, word_end=10)

    class Completions:
        def invoke(self, messages, **kwargs):
            prompt = messages[1]["content"]
            assert prompt.count("Tài liệu: d1") == 1
            assert prompt.count("Tài liệu: d2") == 1
            return SimpleNamespace(text="Kết quả [1] [2].", usage_metadata=None)

    answer = answer_question("Hỏi", Index(), RagConfig(context_k=2), Completions(), "test")
    assert [citation["chunk_id"] for citation in answer.citations] == ["a", "c"]


def test_reranker_stays_on_cpu_when_gpu_has_only_four_gigabytes(monkeypatch):
    import torch

    seen = {}

    def fake_cross_encoder(name, **kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr("sentence_transformers.CrossEncoder", fake_cross_encoder)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: SimpleNamespace(total_memory=4 * 1024**3))
    _load_model.cache_clear()
    _load_model("test-reranker")
    assert seen["device"] == "cpu"
    _load_model.cache_clear()
    _load_model("BAAI/bge-reranker-v2-m3")
    assert seen["revision"] == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    _load_model.cache_clear()


def test_ingestion_preserves_source_and_reading_order(tmp_path):
    path = tmp_path / "lesson.pptx"
    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[5])
    slide.shapes.title.text = "Bài 1"
    left = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(3), Inches(1))
    right = slide.shapes.add_textbox(Inches(5), Inches(2), Inches(3), Inches(1))
    left.text = "Nội dung bên trái"
    right.text = "Nội dung bên phải"
    deck.save(path)

    pages = extract_document(path)
    assert pages[0].section == "Bài 1"
    assert pages[0].text.index("Bài 1") < pages[0].text.index("Nội dung bên trái")
    assert pages[0].text.index("Nội dung bên trái") < pages[0].text.index("Nội dung bên phải")

    chunks, _ = build_hierarchy(
        pages,
        doc_id="doc-1",
        course="AI101",
        source_type="slide",
        chunk_tokens=450,
        overlap_tokens=75,
    )
    assert chunks
    assert all(
        chunk.doc_id == "doc-1"
        and chunk.course == "AI101"
        and chunk.page == 1
        and chunk.text
        for chunk in chunks
    )

    settings = load_settings(tmp_path)
    db = Database(settings.db_path)
    db.initialize()
    result = build_corpus(
        [path],
        {str(path): {"course": "AI101", "source_type": "slide"}},
        settings,
        {"chunk_tokens": 450, "overlap_tokens": 75},
    )
    assert result.chunk_count == len(chunks)
    assert result.chunks_path.exists()
    assert db.get_active_corpus() is None
    assert build_corpus(
        [path],
        {str(path): {"course": "AI101", "source_type": "slide"}},
        settings,
        {"chunk_tokens": 450, "overlap_tokens": 75},
    ).version_id == result.version_id


def test_pptx_custom_textbox_title_and_slide_boundary(tmp_path):
    path = tmp_path / "custom.pptx"
    deck = Presentation()
    for number in (1, 2):
        slide = deck.slides.add_slide(deck.slide_layouts[6])
        heading = slide.shapes.add_textbox(Inches(1), Inches(0.5), Inches(8), Inches(1))
        heading.text = f"Bài học {number}"
        heading.text_frame.paragraphs[0].runs[0].font.size = Pt(32)
        body = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(8), Inches(1))
        body.text = f"Nội dung slide {number}"
        body.text_frame.paragraphs[0].runs[0].font.size = Pt(18)
    deck.save(path)

    pages = extract_document(path)
    assert [page.section for page in pages] == ["Bài học 1", "Bài học 2"]
    chunks, nodes = build_hierarchy(pages, doc_id="deck", course="Học tập", source_type="slide", file_type="pptx", chunk_tokens=50, overlap_tokens=0)
    assert {(chunk.page, chunk.page_end) for chunk in chunks} == {(1, 1), (2, 2)}
    validate_hierarchy(chunks, nodes)


def test_hierarchical_chunks_keep_slide_sources_separate_with_shared_section():
    pages = [
        PageText(1, "GIAO DỊCH", "GIAO DỊCH\nTính nguyên tử."),
        PageText(2, "GIAO DỊCH (Cont.)", "GIAO DỊCH (Cont.)\nTính nhất quán."),
        PageText(3, "CHỈ MỤC", "CHỈ MỤC\nCấu trúc B-tree."),
    ]
    chunks, _ = build_hierarchy(pages, doc_id="d", course="CSDL", source_type="slide", chunk_tokens=30, overlap_tokens=5)
    assert len(chunks) == 3
    assert [(chunk.page, chunk.page_end) for chunk in chunks] == [(1, 1), (2, 2), (3, 3)]
    assert [chunk.section for chunk in chunks] == ["GIAO DỊCH", "GIAO DỊCH", "CHỈ MỤC"]
    assert chunks[0].parent_id != chunks[1].parent_id


def test_chunk_citations_cover_only_pages_in_each_window():
    pages = [
        PageText(1, "Bài 1", "một hai ba bốn năm sáu"),
        PageText(2, "Bài 1", "bảy tám chín mười mười_một mười_hai"),
    ]
    chunks, _ = build_hierarchy(pages, doc_id="d", course="M", source_type="slide", chunk_tokens=8, overlap_tokens=2)
    assert (chunks[0].page, chunks[0].page_end) == (1, 1)
    assert (chunks[1].page, chunks[1].page_end) == (2, 2)
    assert (chunks[-1].page, chunks[-1].page_end) == (2, 2)


def test_chunking_preserves_paragraph_boundaries_until_a_paragraph_is_too_long():
    blocks = [
        PageText(1, "Unit 1", "alpha beta gamma delta"),
        PageText(2, "Unit 1", "epsilon zeta eta theta"),
        PageText(3, "Unit 1", "iota kappa lambda mu"),
    ]
    chunks, nodes = build_hierarchy(
        blocks, doc_id="d", course="Biology", source_type="docx",
        file_type="docx", chunk_tokens=12, overlap_tokens=3,
    )
    assert [(chunk.page, chunk.page_end) for chunk in chunks] == [(1, 2), (3, 3)]
    assert [chunk.word_start for chunk in chunks] == [0, 12]
    assert [chunk.word_end for chunk in chunks] == [12, 18]
    validate_hierarchy(chunks, nodes)


def test_docx_headings_and_tables_keep_reading_order(tmp_path):
    path = tmp_path / "book.docx"
    document = Document()
    document.add_heading("Chương 1", level=1)
    document.add_paragraph("Định nghĩa ban đầu")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Khái niệm"
    table.cell(0, 1).text = "Giải thích"
    document.add_heading("Mục 1.1", level=2)
    document.add_paragraph("Định nghĩa tiếp theo")
    document.save(path)

    blocks = extract_document(path)
    assert [block.text for block in blocks] == ["Định nghĩa ban đầu", "Khái niệm | Giải thích", "Định nghĩa tiếp theo"]
    assert [block.page for block in blocks] == [1, 2, 3]
    assert [block.section for block in blocks] == ["Chương 1", "Chương 1", "Chương 1 > Mục 1.1"]
    chunks, _ = build_hierarchy(blocks, doc_id="d", course="Lịch sử", source_type="docx", chunk_tokens=50, overlap_tokens=5)
    assert "[Đoạn 1]" in chunks[0].text
    assert "[Trang" not in chunks[0].text


def test_docx_flat_heading_styles_keep_chapter_numbering_scope(tmp_path):
    path = tmp_path / "lecture.docx"
    document = Document()
    for title, body in (
        ("Chương 2", None),
        ("NỘI DUNG CHƯƠNG HAI", None),
        ("II. Mục cuối", "Cuối chương hai"),
        ("Chương 3", None),
        ("NỘI DUNG CHƯƠNG BA", "Dẫn nhập chương ba"),
        ("I. Mục đầu", "Đầu chương ba"),
    ):
        document.add_heading(title, level=1)
        if body:
            document.add_paragraph(body)
    document.add_heading("Chi tiết", level=2)
    document.add_paragraph("Nội dung chi tiết")
    bold_heading = document.add_paragraph()
    bold_heading.add_run("II. Mục tiếp theo").bold = True
    document.add_paragraph("Nội dung mục II")
    bold_heading = document.add_paragraph()
    bold_heading.add_run("1. Tiểu mục").bold = True
    document.add_paragraph("Nội dung tiểu mục")
    document.add_heading("KẾT LUẬN", level=1)
    document.add_paragraph("Tóm lược")
    document.save(path)

    pages = extract_document(path)
    assert [page.section for page in pages] == [
        "Chương 2: NỘI DUNG CHƯƠNG HAI > II. Mục cuối",
        "Chương 3: NỘI DUNG CHƯƠNG BA",
        "Chương 3: NỘI DUNG CHƯƠNG BA > I. Mục đầu",
        "Chương 3: NỘI DUNG CHƯƠNG BA > I. Mục đầu > Chi tiết",
        "Chương 3: NỘI DUNG CHƯƠNG BA > II. Mục tiếp theo",
        "Chương 3: NỘI DUNG CHƯƠNG BA > II. Mục tiếp theo > 1. Tiểu mục",
        "KẾT LUẬN",
    ]


def test_docx_numbering_infers_depth_without_subject_specific_titles(tmp_path):
    path = tmp_path / "handbook.docx"
    document = Document()
    for heading, body in (
        ("Unit 1", None),
        ("Cell structure", None),
        ("1. Introduction", "Overview"),
        ("1.1 Components", "Details"),
        ("2. Methods", "Procedures"),
        ("Unit 2", None),
        ("Genetics", None),
        ("1. Introduction", "New unit"),
    ):
        document.add_heading(heading, level=1)
        if body:
            document.add_paragraph(body)
    document.save(path)
    assert [page.section for page in extract_document(path)] == [
        "Unit 1: Cell structure > 1. Introduction",
        "Unit 1: Cell structure > 1. Introduction > 1.1 Components",
        "Unit 1: Cell structure > 2. Methods",
        "Unit 2: Genetics > 1. Introduction",
    ]


def test_docx_custom_outline_style_is_a_heading(tmp_path):
    path = tmp_path / "custom.docx"
    document = Document()
    style = document.styles.add_style("CourseSection", WD_STYLE_TYPE.PARAGRAPH)
    outline = OxmlElement("w:outlineLvl")
    outline.set(qn("w:val"), "0")
    style.element.get_or_add_pPr().append(outline)
    document.add_paragraph("Course overview", style="CourseSection")
    document.add_paragraph("Learning outcomes")
    document.save(path)
    assert extract_document(path)[0].section == "Course overview"


def test_corpus_persists_navigable_document_section_leaf_tree(tmp_path):
    path = tmp_path / "course.docx"
    document = Document()
    document.add_heading("Chương A", level=1)
    document.add_paragraph("Nội dung chương A.")
    document.add_heading("Khái niệm", level=2)
    document.add_paragraph("Giải thích khái niệm A.")
    document.add_heading("Chương B", level=1)
    document.add_heading("Khái niệm", level=2)
    document.add_paragraph("Giải thích khái niệm B.")
    document.save(path)
    settings = load_settings(tmp_path)
    db = Database(settings.db_path)
    db.initialize()

    corpus = build_corpus([path], {str(path): {"course": "Môn học", "source_type": "docx"}}, settings, {"chunk_tokens": 20, "overlap_tokens": 3})
    nodes = {row["node_id"]: row for row in map(json.loads, corpus.hierarchy_path.read_text().splitlines())}
    chunks = [Chunk(**json.loads(line)) for line in corpus.chunks_path.read_text().splitlines()]
    root = next(node for node in nodes.values() if node["kind"] == "document")
    chapters = [nodes[child] for child in root["children"]]
    assert [node["title"] for node in chapters] == ["Chương A", "Chương B"]
    first_subsection = next(nodes[child] for child in chapters[0]["children"] if child in nodes)
    second_subsection = next(nodes[child] for child in chapters[1]["children"] if child in nodes)
    assert first_subsection["title"] == second_subsection["title"] == "Khái niệm"
    assert first_subsection["node_id"] != second_subsection["node_id"]
    assert all(chunk.parent_id in nodes and chunk.chunk_id in nodes[chunk.parent_id]["children"] for chunk in chunks)
    assert next(chunk for chunk in chunks if "Giải thích khái niệm B" in chunk.text).parent_id == second_subsection["node_id"]


def test_pdf_without_outline_does_not_invent_section_from_first_line(tmp_path):
    path = tmp_path / "book.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "This is a running header")
    page.insert_text((72, 100), "Body paragraph")
    document.save(path)
    document.close()
    assert extract_document(path)[0].section == ""
    chunks, _ = build_hierarchy(extract_document(path), doc_id="d", course="M", source_type="textbook", chunk_tokens=50, overlap_tokens=5)
    assert chunks[0].section == ""


def test_landscape_pdf_slide_uses_prominent_title_without_bookmarks(tmp_path):
    path = tmp_path / "slides.pdf"
    document = pymupdf.open()
    page = document.new_page(width=720, height=540)
    page.insert_text((70, 60), "Cell structure", fontsize=32)
    page.insert_text((70, 180), "A cell has a membrane", fontsize=18)
    page.insert_text((70, 220), "and genetic material", fontsize=18)
    page.insert_text((680, 520), "1", fontsize=10)
    document.save(path)
    document.close()
    assert extract_document(path)[0].section == "Cell structure"


def test_ocr_quality_gate_rejects_diagram_noise():
    assert usable_ocr("Bài học về xử lý ảnh. Phát hiện biên giúp nhận biết ranh giới của các đối tượng trong ảnh.")
    assert not usable_ocr("⁄⁄ ey ¬ VA / Z ⁄ ⁄/ ‚/ LZ ⁄ \\ 7 + WIM p47 /,j ¢)")


def test_retrieval_fusion_and_grounded_answer(monkeypatch):
    chunks = {
        "c1": Chunk("c1", "d1", "AI101", "slide", 1, "Mã môn", "Mã môn AI101"),
        "c2": Chunk("c2", "d1", "AI101", "slide", 2, "Khái niệm", "Giải thích học máy"),
    }
    assert minmax_scores({"c1": 5.0, "c2": 5.0}) == {"c1": 1.0, "c2": 1.0}

    exact_alpha, exact_signals = adaptive_alpha(
        "Mã môn AI101 là gì?", alpha0=0.5, beta=0.3, idf={"ai101": 1.0, "rare": 5.0}
    )
    semantic_alpha, _ = adaptive_alpha(
        "Giải thích học máy", alpha0=0.5, beta=0.3, idf={"học": 0.1, "máy": 0.1, "rare": 5.0}
    )
    assert exact_signals["code"] == 1.0
    assert exact_alpha > semantic_alpha

    fused = fuse_weighted(
        bm25_scores={"c1": 8.0, "c2": 1.0},
        dense_scores={"c1": 0.6, "c2": 0.5},
        chunks=chunks,
        alpha=0.7,
    )
    assert fused[0].chunk.chunk_id == "c1"

    monkeypatch.setattr("src.rag.retrieve", lambda query, index, config: fused)

    class FakeCompletions:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                text="Theo tài liệu [1] và [99].",
                usage_metadata={"input_tokens": 10, "output_tokens": 5},
            )

    completions = FakeCompletions()
    client = completions
    answer = answer_question(
        "Mã môn AI101 là gì?",
        index=SimpleNamespace(expand_chunk=lambda chunk, _budget: chunk),
        rag_config=RagConfig(method="weighted", use_reranker=False),
        client=client,
        model="test-model",
    )
    assert "[1]" in answer.text and "[99]" not in answer.text
    assert answer.citations[0]["chunk_id"] == "c1"
    assert answer.citations[0]["source_type"] == "slide"

    refused = answer_question(
        "Câu hỏi ngoài tài liệu",
        index=SimpleNamespace(expand_chunk=lambda chunk, _budget: chunk),
        rag_config=RagConfig(method="weighted", refusal_threshold=2.0, use_reranker=False),
        client=client,
        model="test-model",
    )
    assert refused.refused is True
    assert completions.calls == 1


def test_single_retrievers_do_not_run_the_other_model():
    chunk = Chunk("c1", "d1", "CSDL", "slide", 1, "", "Cơ sở dữ liệu")

    class Index:
        chunks = {"c1": chunk}

        def sparse_scores(self, query, limit):
            assert query == "Cơ sở dữ liệu" and limit == 10
            return {"c1": 1.0}

        def dense_scores(self, query, limit):
            assert query == "Cơ sở dữ liệu" and limit == 10
            return {"c1": 0.9}

    index = Index()
    index.dense_scores = lambda *args: (_ for _ in ()).throw(AssertionError("dense called"))
    assert retrieve("Cơ sở dữ liệu", index, RagConfig(method="bm25", top_l=10))[0].chunk.chunk_id == "c1"
    index.dense_scores = lambda *args: {"c1": 0.9}
    index.sparse_scores = lambda *args: (_ for _ in ()).throw(AssertionError("bm25 called"))
    assert retrieve("Cơ sở dữ liệu", index, RagConfig(method="dense", top_l=10))[0].chunk.chunk_id == "c1"

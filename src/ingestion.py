import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from docx import Document
from docx.table import Table
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pyvi import ViTokenizer

from src.models import Chunk


@dataclass(frozen=True)
class PageText:
    page: int
    section: str
    text: str
    warning: str = ""


@dataclass(frozen=True)
class CorpusBuildResult:
    version_id: str
    chunks_path: Path
    hierarchy_path: Path
    chunk_count: int
    manifest_hash: str


def tokenize_vi(text: str, mode: str = "whitespace") -> list[str]:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    if not normalized:
        return []
    if mode == "whitespace":
        return normalized.split()
    if mode == "pyvi":
        return ViTokenizer.tokenize(normalized).split()
    raise ValueError(f"Unsupported tokenizer mode: {mode}")


def usable_ocr(text: str) -> bool:
    clean = text.strip()
    return bool(clean) and sum(character.isalpha() for character in clean) / len(clean) >= 0.55 and len(re.findall(r"\b[^\W\d_]{3,}\b", clean)) >= 5


def extract_document(path: Path | str) -> list[PageText]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(source)
    if suffix == ".docx":
        return _extract_docx(source)
    if suffix == ".pptx":
        return _extract_pptx(source)
    raise ValueError(f"Unsupported document type: {suffix}")


def _native_pdf_text(page) -> str:
    spans = [
        span
        for block in page.get_text("dict")["blocks"] if "lines" in block
        for line in block["lines"] for span in line["spans"]
        if span["text"].strip()
    ]
    middle = page.rect.width / 2
    left = [span for span in spans if span["bbox"][2] < middle - 10]
    right = [span for span in spans if span["bbox"][0] > middle + 10]
    middle_spans = [span for span in spans if span not in left and span not in right]
    if len(left) >= 2 and len(right) >= 2 and not middle_spans and max(left[0]["bbox"][1], right[0]["bbox"][1]) < min(left[-1]["bbox"][3], right[-1]["bbox"][3]):
        # ponytail: two-column heuristic; replace with layout analysis when complex PDFs enter the corpus.
        ordered = sorted(left, key=lambda span: (span["bbox"][1], span["bbox"][0])) + sorted(right, key=lambda span: (span["bbox"][1], span["bbox"][0]))
        return "\n".join(span["text"].strip() for span in ordered)
    return page.get_text("text", sort=True).strip()


def _extract_pdf(path: Path) -> list[PageText]:
    pages: list[PageText] = []
    with pymupdf.open(path) as document:
        outline = document.get_toc()
        headings: list[str] = []
        outline_index = 0
        for number, page in enumerate(document, start=1):
            while outline_index < len(outline) and outline[outline_index][2] <= number:
                level, title, _ = outline[outline_index]
                headings = headings[: level - 1] + [title.strip()]
                outline_index += 1
            text = _native_pdf_text(page)
            warning = ""
            if not text or (len(text.split()) < 20 and page.get_images()):
                try:
                    ocr_text = page.get_text("text", textpage=page.get_textpage_ocr(language="vie+eng", dpi=300)).strip()
                    if len(ocr_text) > len(text) and usable_ocr(ocr_text):
                        text = ocr_text
                    elif len(ocr_text) > len(text):
                        warning = "OCR bỏ qua vì kết quả giống nhiễu từ hình/sơ đồ."
                except RuntimeError:
                    warning = "Trang không có text layer và OCR không khả dụng."
            pages.append(PageText(number, " > ".join(headings), text, warning))
    return pages


def _extract_docx(path: Path) -> list[PageText]:
    document = Document(path)
    pages: list[PageText] = []
    headings: list[str] = []
    for block in document.iter_inner_content():
        if isinstance(block, Table):
            text = "\n".join(
                " | ".join(cell.text.strip() for cell in row.cells)
                for row in block.rows
                if any(cell.text.strip() for cell in row.cells)
            ).strip()
        else:
            text = block.text.strip()
            match = re.fullmatch(r"Heading\s*(\d+)", block.style.name, re.I)
            if match and text:
                level = int(match.group(1))
                headings = headings[: level - 1] + [text]
                continue
        if text:
            pages.append(PageText(len(pages) + 1, " > ".join(headings), text))
    return pages


def _shape_texts(shapes, title_shape=None):
    items: list[tuple[int, int, str]] = []
    title = ""
    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            nested_title, nested_items = _shape_texts(shape.shapes)
            title = title or nested_title
            items.extend(nested_items)
            continue
        if shape is title_shape and getattr(shape, "has_text_frame", False):
            title = shape.text.strip()
            continue
        if getattr(shape, "has_table", False):
            text = "\n".join(
                " | ".join(cell.text.strip() for cell in row.cells)
                for row in shape.table.rows
            ).strip()
        elif getattr(shape, "has_text_frame", False):
            text = shape.text.strip()
        else:
            text = ""
        if text:
            items.append((int(shape.top), int(shape.left), text))
    return title, items


def _extract_pptx(path: Path) -> list[PageText]:
    presentation = Presentation(path)
    pages: list[PageText] = []
    band_height = 228600  # 0.25 inch in EMU
    for number, slide in enumerate(presentation.slides, start=1):
        title_shape = slide.shapes.title
        title, items = _shape_texts(slide.shapes, title_shape)
        ordered = sorted(items, key=lambda item: (round(item[0] / band_height), item[1]))
        blocks = ([title] if title else []) + [item[2] for item in ordered]
        pages.append(PageText(number, title, "\n".join(blocks)))
    return pages


def chunk_pages(
    pages: list[PageText],
    *,
    doc_id: str,
    course: str,
    source_type: str,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    chunks, _ = build_hierarchy(pages, doc_id=doc_id, course=course, source_type=source_type, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
    return chunks


def build_hierarchy(
    pages: list[PageText],
    *,
    doc_id: str,
    course: str,
    source_type: str,
    chunk_tokens: int,
    overlap_tokens: int,
    title: str = "",
    file_type: str = "",
) -> tuple[list[Chunk], list[dict]]:
    if chunk_tokens <= 0 or not 0 <= overlap_tokens < chunk_tokens:
        raise ValueError("chunk_tokens must be positive and overlap smaller than chunk_tokens")
    nodes: dict[str, dict] = {}
    words_by_node: dict[str, list[tuple[str, int]]] = {}
    node_number = 0

    def add_node(kind: str, node_title: str, parent_id: str | None, path: str = "") -> str:
        nonlocal node_number
        node_number += 1
        node_id = hashlib.sha256(f"{doc_id}|{node_number}|{kind}|{node_title}".encode()).hexdigest()[:24]
        nodes[node_id] = {"node_id": node_id, "parent_id": parent_id, "doc_id": doc_id, "kind": kind, "title": node_title, "path": path, "text": "", "children": []}
        words_by_node[node_id] = []
        if parent_id:
            nodes[parent_id]["children"].append(node_id)
        return node_id

    root_id = add_node("document", title, None)
    section_stack: list[tuple[str, str]] = []
    locator = "Đoạn" if file_type == "docx" or source_type == "docx" else "Trang"
    for page in pages:
        if not page.text.strip():
            continue
        parts = [re.sub(r"\s*\((?:cont\.?|tiếp|tt\.?)\)\s*$", "", part, flags=re.I).strip() for part in page.section.split(" > ") if part.strip()]
        shared = 0
        while shared < min(len(parts), len(section_stack)) and parts[shared] == section_stack[shared][0]:
            shared += 1
        section_stack = section_stack[:shared]
        for part in parts[shared:]:
            parent_id = section_stack[-1][1] if section_stack else root_id
            section_stack.append((part, add_node("section", part, parent_id, " > ".join(name for name, _ in section_stack) + (" > " if section_stack else "") + part)))
        parent_id = section_stack[-1][1] if section_stack else root_id
        target_id = (
            parent_id if locator == "Đoạn"
            else add_node("page", f"{locator} {page.page}", parent_id, nodes[parent_id]["path"])
        )
        words_by_node[target_id].extend((word, page.page) for word in f"[{locator} {page.page}] {page.text}".split())

    chunks: list[Chunk] = []
    for node_id, node in nodes.items():
        tagged_words = words_by_node[node_id]
        if not tagged_words:
            continue
        node["text"] = " ".join(word for word, _ in tagged_words)
        step = chunk_tokens - overlap_tokens
        for index, start in enumerate(range(0, len(tagged_words), step)):
            window = tagged_words[start : start + chunk_tokens]
            body = " ".join(word for word, _ in window)
            if not body:
                continue
            section = node["path"]
            text = f"{section}\n{body}" if section else body
            first_page, last_page = window[0][1], window[-1][1]
            digest = hashlib.sha256(
                f"{doc_id}|{node_id}|{start}|{text}".encode()
            ).hexdigest()[:24]
            chunks.append(Chunk(digest, doc_id, course, source_type, first_page, section, text, last_page, node_id, start, start + len(window), file_type))
            node["children"].append(digest)
            if start + chunk_tokens >= len(tagged_words):
                break
    return chunks, list(nodes.values())


def build_corpus(
    files: list[Path],
    metadata: dict[str, dict],
    settings,
    config: dict,
    previous_dir: Path | None = None,
) -> CorpusBuildResult:
    cached_records: dict[str, dict] = {}
    cached_chunks: dict[str, list[Chunk]] = {}
    cached_nodes: dict[str, list[dict]] = {}
    if previous_dir and (previous_dir / "manifest.json").is_file():
        previous = json.loads((previous_dir / "manifest.json").read_text(encoding="utf-8"))
        if previous.get("config") == config and previous.get("chunking") == "hierarchical-v7":
            cached_records = {item["doc_id"]: item for item in previous["documents"]}
            for line in (previous_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
                chunk = Chunk(**json.loads(line))
                cached_chunks.setdefault(chunk.doc_id, []).append(chunk)
            for line in (previous_dir / "hierarchy.jsonl").read_text(encoding="utf-8").splitlines():
                node = json.loads(line)
                cached_nodes.setdefault(node["doc_id"], []).append(node)
    manifest_items = []
    all_chunks: list[Chunk] = []
    all_nodes: list[dict] = []
    for path in sorted((Path(file) for file in files), key=lambda item: str(item)):
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        item_metadata = metadata[str(path)]
        doc_id = file_hash[:24]
        record = {
            "doc_id": doc_id,
            "filename": path.name,
            "sha256": file_hash,
            **item_metadata,
        }
        if cached_records.get(doc_id) == record and cached_chunks.get(doc_id) and cached_nodes.get(doc_id):
            chunks, nodes = cached_chunks[doc_id], cached_nodes[doc_id]
        else:
            pages = extract_document(path)
            chunks, nodes = build_hierarchy(
                pages,
                doc_id=doc_id,
                course=item_metadata["course"],
                source_type=item_metadata["source_type"],
                chunk_tokens=int(config["chunk_tokens"]),
                overlap_tokens=int(config["overlap_tokens"]),
                title=path.name,
                file_type=path.suffix.lower().lstrip("."),
            )
            if not chunks:
                raise ValueError(f"{path.name}: không có nội dung văn bản để lập chỉ mục")
        all_chunks.extend(chunks)
        all_nodes.extend(nodes)
        manifest_items.append(record)

    manifest = {"documents": manifest_items, "config": config, "chunking": "hierarchical-v7"}
    manifest_json = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    manifest_hash = hashlib.sha256(manifest_json.encode()).hexdigest()
    version_id = manifest_hash[:16]
    output_dir = settings.data_dir / "processed" / version_id
    chunks_path = output_dir / "chunks.jsonl"
    hierarchy_path = output_dir / "hierarchy.jsonl"
    chunks_jsonl = "\n".join(json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in all_chunks) + "\n"
    hierarchy_jsonl = "\n".join(json.dumps(node, ensure_ascii=False) for node in all_nodes) + "\n"
    if output_dir.exists():
        if (output_dir / "manifest.json").is_file() and chunks_path.is_file() and hierarchy_path.is_file() and (output_dir / "manifest.json").read_text(encoding="utf-8") == manifest_json and chunks_path.read_text(encoding="utf-8") == chunks_jsonl and hierarchy_path.read_text(encoding="utf-8") == hierarchy_jsonl:
            return CorpusBuildResult(version_id, chunks_path, hierarchy_path, len(all_chunks), manifest_hash)
        raise FileExistsError(f"Corpus version exists with different or incomplete contents: {version_id}")
    output_dir.mkdir(parents=True)
    chunks_path.write_text(chunks_jsonl, encoding="utf-8")
    hierarchy_path.write_text(hierarchy_jsonl, encoding="utf-8")
    (output_dir / "manifest.json").write_text(manifest_json, encoding="utf-8")
    return CorpusBuildResult(version_id, chunks_path, hierarchy_path, len(all_chunks), manifest_hash)

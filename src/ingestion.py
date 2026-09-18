import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from docx import Document
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


def _extract_pdf(path: Path) -> list[PageText]:
    pages: list[PageText] = []
    with pymupdf.open(path) as document:
        for number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            warning = ""
            if not text:
                try:
                    text = page.get_text("text", textpage=page.get_textpage_ocr()).strip()
                except RuntimeError:
                    warning = "Trang không có text layer và OCR không khả dụng."
            pages.append(PageText(number, "", text, warning))
    return pages


def _extract_docx(path: Path) -> list[PageText]:
    document = Document(path)
    blocks = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    for table in document.tables:
        blocks.extend(
            " | ".join(cell.text.strip() for cell in row.cells)
            for row in table.rows
            if any(cell.text.strip() for cell in row.cells)
        )
    title = blocks[0] if blocks else ""
    return [PageText(1, title, "\n".join(blocks))]


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
    if chunk_tokens <= 0 or not 0 <= overlap_tokens < chunk_tokens:
        raise ValueError("chunk_tokens must be positive and overlap smaller than chunk_tokens")
    chunks: list[Chunk] = []
    step = chunk_tokens - overlap_tokens
    for page in pages:
        words = page.text.split()
        for index, start in enumerate(range(0, len(words), step)):
            body = " ".join(words[start : start + chunk_tokens]).strip()
            if not body:
                continue
            text = f"{page.section}\n{body}".strip() if page.section else body
            digest = hashlib.sha256(
                f"{doc_id}|{page.page}|{page.section}|{index}|{text}".encode()
            ).hexdigest()[:24]
            chunks.append(
                Chunk(digest, doc_id, course, source_type, page.page, page.section, text)
            )
            if start + chunk_tokens >= len(words):
                break
    return chunks


def build_corpus(
    files: list[Path],
    metadata: dict[str, dict],
    settings,
    database,
    config: dict,
) -> CorpusBuildResult:
    manifest_items = []
    all_chunks: list[Chunk] = []
    for path in sorted((Path(file) for file in files), key=lambda item: str(item)):
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        item_metadata = metadata[str(path)]
        doc_id = file_hash[:24]
        pages = extract_document(path)
        chunks = chunk_pages(
            pages,
            doc_id=doc_id,
            course=item_metadata["course"],
            source_type=item_metadata["source_type"],
            chunk_tokens=int(config["chunk_tokens"]),
            overlap_tokens=int(config["overlap_tokens"]),
        )
        all_chunks.extend(chunks)
        record = {
            "doc_id": doc_id,
            "filename": path.name,
            "sha256": file_hash,
            **item_metadata,
        }
        manifest_items.append(record)
        database.save_document({**record, "status": "processed"})

    manifest = {"documents": manifest_items, "config": config}
    manifest_json = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    manifest_hash = hashlib.sha256(manifest_json.encode()).hexdigest()
    version_id = manifest_hash[:16]
    output_dir = settings.data_dir / "processed" / version_id
    if output_dir.exists():
        raise FileExistsError(f"Corpus version already exists: {version_id}")
    output_dir.mkdir(parents=True)
    chunks_path = output_dir / "chunks.jsonl"
    chunks_path.write_text(
        "\n".join(json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in all_chunks) + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(manifest_json, encoding="utf-8")
    database.save_corpus_version(
        {
            "version_id": version_id,
            "manifest_hash": manifest_hash,
            "chunks_path": str(chunks_path),
            "chunk_count": len(all_chunks),
        }
    )
    return CorpusBuildResult(version_id, chunks_path, len(all_chunks), manifest_hash)

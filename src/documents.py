import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from docx import Document
from docx.table import Table
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE


@dataclass(frozen=True)
class PageText:
    page: int
    section: str
    text: str
    warning: str = ""


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

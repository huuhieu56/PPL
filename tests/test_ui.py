import hashlib
from types import SimpleNamespace

from src.ui import citation_label, stage_uploads


def _upload(name, content):
    return SimpleNamespace(name=name, getvalue=lambda: content)


def test_stage_uploads_is_content_addressed_and_drops_duplicates(tmp_path):
    paths = stage_uploads(
        [_upload("bài 1.pdf", b"one"), _upload("bài 1.pdf", b"two"), _upload("copy.pdf", b"one")], tmp_path
    )
    assert len(paths) == 2
    assert paths[0].parent.name == hashlib.sha256(b"one").hexdigest()[:24]
    assert paths[0].name == paths[1].name and paths[0] != paths[1]
    assert paths[1].read_bytes() == b"two"


def test_citation_label_omits_page_for_docx():
    citation = {"number": 2, "doc_id": "d1", "doc_title": "Giáo trình", "heading_path": ["Chương 1"], "page": 1}
    assert citation_label(citation, {"d1": "gt.docx"}) == "[2] Giáo trình > Chương 1"
    assert citation_label(citation, {"d1": "gt.pdf"}) == "[2] Giáo trình > Chương 1 — trang/slide 1"

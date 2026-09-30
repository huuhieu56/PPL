import hashlib
import re

from src.documents import PageText
from src.models import Chunk


def tokenize_vi(text: str, mode: str = "whitespace") -> list[str]:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    if not normalized:
        return []
    if mode == "whitespace":
        return normalized.split()
    if mode == "pyvi":
        from pyvi import ViTokenizer

        return ViTokenizer.tokenize(normalized).split()
    raise ValueError(f"Unsupported tokenizer mode: {mode}")


def validate_hierarchy(chunks: list[Chunk], nodes: list[dict]) -> dict[str, dict]:
    hierarchy = {node["node_id"]: node for node in nodes}
    leaves = {chunk.chunk_id: chunk for chunk in chunks}
    if len(hierarchy) != len(nodes) or len(leaves) != len(chunks) or set(hierarchy) & set(leaves):
        raise ValueError("Duplicate hierarchy node or chunk ID")
    roots = [node for node in nodes if node["parent_id"] is None]
    if {node["doc_id"] for node in roots} != {chunk.doc_id for chunk in chunks} or len(roots) != len({node["doc_id"] for node in roots}) or any(node["kind"] != "document" for node in roots):
        raise ValueError("Hierarchy must have one document root per source")
    owners: dict[str, str] = {}
    for node in nodes:
        parent_id = node["parent_id"]
        if parent_id is not None and (parent_id not in hierarchy or hierarchy[parent_id]["doc_id"] != node["doc_id"]):
            raise ValueError("Hierarchy node has an invalid parent")
        for child_id in node["children"]:
            child = hierarchy.get(child_id) or leaves.get(child_id)
            if child is None or child_id in owners:
                raise ValueError("Hierarchy has an unknown or duplicate child")
            child_parent = child["parent_id"] if isinstance(child, dict) else child.parent_id
            child_doc = child["doc_id"] if isinstance(child, dict) else child.doc_id
            if child_parent != node["node_id"] or child_doc != node["doc_id"]:
                raise ValueError("Hierarchy child points to the wrong parent")
            owners[child_id] = node["node_id"]
    if set(owners) != (set(hierarchy) - {node["node_id"] for node in roots}) | set(leaves):
        raise ValueError("Hierarchy has an unlinked child")
    for chunk in chunks:
        parent = hierarchy[chunk.parent_id]
        words = parent["text"].split()
        if not 0 <= chunk.word_start < chunk.word_end <= len(words) or not chunk.text.endswith(" ".join(words[chunk.word_start:chunk.word_end])):
            raise ValueError("Hierarchy leaf span does not match parent text")
    visited: set[str] = set()

    def walk(node_id: str) -> None:
        if node_id in visited:
            raise ValueError("Hierarchy contains a cycle")
        visited.add(node_id)
        for child_id in hierarchy[node_id]["children"]:
            if child_id in hierarchy:
                walk(child_id)

    for root in roots:
        walk(root["node_id"])
    if len(visited) != len(hierarchy):
        raise ValueError("Hierarchy contains an unreachable node")
    return hierarchy


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

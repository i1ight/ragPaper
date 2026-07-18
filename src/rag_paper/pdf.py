from __future__ import annotations

from pathlib import Path

import fitz

from rag_paper.models import PaperDocument
from rag_paper.title_quality import is_trusted_title, pick_title_line


def extract_pdf_text(path: Path, extra_metadata: dict[str, object] | None = None) -> PaperDocument:
    metadata = {
        "source_path": str(path.resolve()),
        "file_name": path.name,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    pages: list[str] = []
    with fitz.open(path) as doc:
        if doc.metadata:
            pdf_title = doc.metadata.get("title")
            if pdf_title and not metadata.get("title") and is_trusted_title(pdf_title):
                metadata["title"] = pdf_title
            if doc.metadata.get("author") and not metadata.get("authors"):
                metadata["authors"] = [doc.metadata["author"]]
        metadata["page_count"] = doc.page_count
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append(f"\n\n[Page {page_number}]\n{text}")

    text = "".join(pages).strip()
    if not is_trusted_title(metadata.get("title")):
        metadata.pop("title", None)
        inferred_title = pick_title_line(text.splitlines(), file_name=path.name)
        if inferred_title:
            metadata["title"] = inferred_title

    return PaperDocument(path=path, text=text, metadata=metadata)

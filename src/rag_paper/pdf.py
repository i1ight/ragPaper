from __future__ import annotations

from pathlib import Path

import fitz

from rag_paper.models import PaperDocument


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
            if doc.metadata.get("title") and not metadata.get("title"):
                metadata["title"] = doc.metadata["title"]
            if doc.metadata.get("author") and not metadata.get("authors"):
                metadata["authors"] = [doc.metadata["author"]]
        metadata["page_count"] = doc.page_count
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append(f"\n\n[Page {page_number}]\n{text}")

    return PaperDocument(path=path, text="".join(pages).strip(), metadata=metadata)


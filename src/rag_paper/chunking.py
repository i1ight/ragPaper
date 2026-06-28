from __future__ import annotations

import hashlib

from llama_index.core.node_parser import SentenceSplitter

from rag_paper.models import PaperChunk, PaperDocument


def stable_chunk_id(source_path: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(f"{source_path}:{chunk_index}:{text}".encode("utf-8")).hexdigest()
    return digest[:24]


def chunk_document(document: PaperDocument, chunk_size: int, chunk_overlap: int) -> list[PaperChunk]:
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = splitter.split_text(document.text)
    source_path = str(document.path.resolve())

    chunks: list[PaperChunk] = []
    for index, text in enumerate(texts):
        metadata = dict(document.metadata)
        metadata.update(
            {
                "source_path": source_path,
                "chunk_index": index,
                "chunk_count": len(texts),
            }
        )
        chunks.append(
            PaperChunk(
                id=stable_chunk_id(source_path, index, text),
                text=text,
                metadata=metadata,
            )
        )
    return chunks


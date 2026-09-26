"""Document ingestion: loaders and overlap-aware chunking.

Chunking is character-based with word-boundary snapping so chunks never
split mid-word, and a configurable overlap so context survives boundaries.
Everything is deterministic: the same document always yields the same
chunks, which keeps the vector store and evals reproducible.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class Document:
    """A single source document before chunking."""

    id: str
    title: str
    source: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of a document."""

    doc_id: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int

    @property
    def est_tokens(self) -> int:
        """Rough token estimate (chars/4) for cost accounting."""
        return max(1, len(self.text) // 4)


def _doc_id(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def load_text_file(path: str | Path) -> Document:
    """Load a plain-text file as a Document."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return Document(id=_doc_id(path), title=path.stem, source=str(path), text=text)


def load_markdown_file(path: str | Path) -> Document:
    """Load a Markdown file, taking the title from its first heading."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    title = path.stem
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            break
    return Document(id=_doc_id(path), title=title, source=str(path), text=text)


def load_documents(docs_dir: str | Path) -> list[Document]:
    """Load every .md and .txt file under a directory as Documents."""
    docs: list[Document] = []
    for path in sorted(Path(docs_dir).rglob("*")):
        if path.suffix.lower() == ".md":
            docs.append(load_markdown_file(path))
        elif path.suffix.lower() == ".txt":
            docs.append(load_text_file(path))
    return docs


def _snap_backward(text: str, pos: int) -> int:
    """Move pos back to the start of the word it is inside (or just after).

    Snapping backward (rather than forward) guarantees no character is left
    uncovered between consecutive chunks.
    """
    start = pos
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    return start


def _snap_forward(text: str, pos: int) -> int:
    """Move pos forward to the next word boundary (never split a word)."""
    match = _WORD_RE.search(text, pos)
    if match is None:
        return len(text)
    # Start at the word's beginning, but don't leave trailing whitespace
    # dangling on the previous chunk: back up over the gap.
    start = match.start()
    while start > pos and text[start - 1].isspace():
        start -= 1
    return start if start > pos else match.end()


def chunk_document(
    doc: Document,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    min_chunk_chars: int = 50,
) -> list[Chunk]:
    """Split a document into overlapping chunks.

    Windows advance by ``chunk_size - chunk_overlap`` characters with the
    end snapped to a word boundary. A trailing chunk shorter than
    ``min_chunk_chars`` is dropped (unless the document yields only one
    chunk).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not (0 <= chunk_overlap < chunk_size):
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

    text = doc.text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            end = _snap_forward(text, end)
            if end <= start:  # pathological: one giant word
                end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    doc_id=doc.id,
                    chunk_index=index,
                    text=chunk_text,
                    char_start=start,
                    char_end=end,
                )
            )
            index += 1
        if end >= len(text):
            break
        new_start = start + step
        # Snap back to the start of the word the window edge landed inside,
        # so no character is skipped and no chunk starts mid-word. Fall back
        # to the raw edge in the pathological case (a word longer than the
        # step) to guarantee forward progress.
        snapped = _snap_backward(text, new_start)
        start = snapped if snapped > start else new_start
        # Skip whitespace the next window would start with.
        while start < len(text) and text[start].isspace():
            start += 1

    if len(chunks) > 1 and len(chunks[-1].text) < min_chunk_chars:
        chunks.pop()
        # Re-index so chunk_index stays dense.
        chunks = [
            Chunk(
                doc_id=c.doc_id,
                chunk_index=i,
                text=c.text,
                char_start=c.char_start,
                char_end=c.char_end,
            )
            for i, c in enumerate(chunks)
        ]
    return chunks


def chunk_corpus(
    docs: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    min_chunk_chars: int = 50,
) -> list[Chunk]:
    """Chunk a list of documents, preserving document order."""
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, chunk_size, chunk_overlap, min_chunk_chars))
    return chunks

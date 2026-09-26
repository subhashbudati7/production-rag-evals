"""Tests for rag_service.ingest."""

from itertools import pairwise

import pytest

from rag_service.ingest import (
    Chunk,
    Document,
    chunk_corpus,
    chunk_document,
    load_documents,
    load_markdown_file,
    load_text_file,
)


def _doc(text: str, doc_id: str = "d1") -> Document:
    return Document(id=doc_id, title="t", source="s", text=text)


def test_empty_document_yields_no_chunks():
    assert chunk_document(_doc("")) == []
    assert chunk_document(_doc("   \n  ")) == []


def test_short_document_is_one_chunk():
    chunks = chunk_document(_doc("hello world"))
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].chunk_index == 0
    assert (chunks[0].char_start, chunks[0].char_end) == (0, 11)


def test_chunks_cover_the_document():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_document(_doc(text), chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 3
    # Every character of the stripped source appears in some chunk.
    covered = set()
    for c in chunks:
        covered.update(range(c.char_start, c.char_end))
    assert covered == set(range(len(text.strip())))


def test_overlap_between_consecutive_chunks():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_document(_doc(text), chunk_size=200, chunk_overlap=60)
    for a, b in pairwise(chunks):
        assert b.char_start < a.char_end  # windows overlap
        assert a.text.split()[-1] in b.text or b.text.split()[0] in a.text


def test_no_mid_word_splits():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_document(_doc(text), chunk_size=200, chunk_overlap=40)
    words = set(text.split())
    for c in chunks:
        for token in c.text.split():
            assert token in words, f"chunk contains fragment {token!r}"


def test_tiny_trailing_chunk_is_dropped():
    text = "a " * 400 + "tail"
    chunks = chunk_document(_doc(text), chunk_size=200, chunk_overlap=40, min_chunk_chars=50)
    assert all(len(c.text) >= 50 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunking_is_deterministic():
    text = " ".join(f"word{i}" for i in range(500))
    doc = _doc(text)
    first = chunk_document(doc, chunk_size=200, chunk_overlap=40)
    second = chunk_document(doc, chunk_size=200, chunk_overlap=40)
    assert first == second


def test_invalid_params_raise():
    doc = _doc("hello")
    with pytest.raises(ValueError):
        chunk_document(doc, chunk_size=0)
    with pytest.raises(ValueError):
        chunk_document(doc, chunk_size=100, chunk_overlap=100)


def test_chunk_token_estimate():
    c = Chunk(doc_id="d", chunk_index=0, text="x" * 400, char_start=0, char_end=400)
    assert c.est_tokens == 100


def test_markdown_title_from_first_heading(tmp_path):
    p = tmp_path / "notes.md"
    p.write_text("# My Title\n\nSome body text.\n")
    doc = load_markdown_file(p)
    assert doc.title == "My Title"
    assert "Some body text." in doc.text


def test_text_file_uses_filename_as_title(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("plain body")
    doc = load_text_file(p)
    assert doc.title == "notes"
    assert doc.text == "plain body"


def test_load_documents_reads_md_and_txt(tmp_path):
    (tmp_path / "a.md").write_text("# A\nbody a")
    (tmp_path / "b.txt").write_text("body b")
    (tmp_path / "c.json").write_text("{}")
    docs = load_documents(tmp_path)
    assert sorted(d.title for d in docs) == ["A", "b"]
    assert len({d.id for d in docs}) == 2  # distinct ids


def test_chunk_corpus_preserves_document_order():
    docs = [_doc("alpha beta gamma", "d1"), _doc("delta epsilon", "d2")]
    chunks = chunk_corpus(docs)
    assert [c.doc_id for c in chunks] == ["d1", "d2"]

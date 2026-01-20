"""Tests for document loaders."""

import tempfile
from pathlib import Path

import pytest

from doc2dataset.loaders import (
    Document,
    TextLoader,
    MarkdownLoader,
    JSONLoader,
    load_document,
    load_folder,
    get_supported_extensions,
)


class TestDocument:
    """Tests for Document class."""

    def test_create_basic(self):
        """Test creating a basic document."""
        doc = Document(content="Hello world", source="test.txt")
        assert doc.content == "Hello world"
        assert doc.source == "test.txt"
        assert doc.filename == "test.txt"

    def test_word_count(self):
        """Test word count property."""
        doc = Document(content="One two three four five")
        assert doc.word_count == 5

    def test_length(self):
        """Test length of document."""
        doc = Document(content="Hello")
        assert len(doc) == 5


class TestTextLoader:
    """Tests for TextLoader."""

    def test_supports(self):
        """Test file support detection."""
        loader = TextLoader()
        assert loader.supports("test.txt")
        assert loader.supports("test.text")
        assert not loader.supports("test.pdf")

    def test_load(self):
        """Test loading a text file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("Hello world!\nThis is a test.")
            path = f.name

        try:
            loader = TextLoader()
            doc = loader.load(path)

            assert "Hello world!" in doc.content
            assert doc.metadata["format"] == "text"
        finally:
            Path(path).unlink()

    def test_chunk(self):
        """Test text chunking."""
        loader = TextLoader()
        content = "Paragraph one.\n\nParagraph two.\n\nParagraph three."

        chunks = loader.chunk(content, chunk_size=30, overlap=5)
        assert len(chunks) >= 1


class TestMarkdownLoader:
    """Tests for MarkdownLoader."""

    def test_supports(self):
        """Test file support detection."""
        loader = MarkdownLoader()
        assert loader.supports("test.md")
        assert loader.supports("test.markdown")
        assert not loader.supports("test.txt")

    def test_load(self):
        """Test loading a markdown file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write("# Title\n\nSome content here.\n\n## Section\n\nMore content.")
            path = f.name

        try:
            loader = MarkdownLoader()
            doc = loader.load(path)

            assert "Title" in doc.content
            assert "Section" in doc.content
            assert doc.metadata["format"] == "markdown"
        finally:
            Path(path).unlink()

    def test_strip_html(self):
        """Test HTML stripping."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write("# Title\n\n<div>HTML content</div>\n\nNormal text.")
            path = f.name

        try:
            loader = MarkdownLoader(strip_html=True)
            doc = loader.load(path)

            assert "<div>" not in doc.content
            assert "HTML content" in doc.content
        finally:
            Path(path).unlink()


class TestJSONLoader:
    """Tests for JSONLoader."""

    def test_supports(self):
        """Test file support detection."""
        loader = JSONLoader()
        assert loader.supports("test.json")
        assert not loader.supports("test.txt")

    def test_load(self):
        """Test loading a JSON file."""
        import json

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            data = {
                "text": "Some content",
                "items": [
                    {"content": "Item 1"},
                    {"content": "Item 2"},
                ],
            }
            json.dump(data, f)
            path = f.name

        try:
            loader = JSONLoader()
            doc = loader.load(path)

            assert "Some content" in doc.content
            assert "Item 1" in doc.content
        finally:
            Path(path).unlink()


class TestLoadDocument:
    """Tests for load_document function."""

    def test_auto_detect_txt(self):
        """Test auto-detection of text files."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("Test content")
            path = f.name

        try:
            doc = load_document(path)
            assert doc.content == "Test content"
        finally:
            Path(path).unlink()

    def test_unsupported_format(self):
        """Test error on unsupported format."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xyz", delete=False
        ) as f:
            f.write("Test")
            path = f.name

        try:
            with pytest.raises(ValueError, match="No loader found"):
                load_document(path)
        finally:
            Path(path).unlink()


class TestLoadFolder:
    """Tests for load_folder function."""

    def test_load_folder(self):
        """Test loading all documents from a folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "file1.txt").write_text("Content 1")
            (Path(tmpdir) / "file2.txt").write_text("Content 2")
            (Path(tmpdir) / "file3.md").write_text("# Markdown")

            docs = list(load_folder(tmpdir))
            assert len(docs) == 3

    def test_filter_extensions(self):
        """Test filtering by extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file1.txt").write_text("Content 1")
            (Path(tmpdir) / "file2.md").write_text("# Markdown")

            docs = list(load_folder(tmpdir, extensions=[".txt"]))
            assert len(docs) == 1
            assert docs[0].metadata["format"] == "text"

    def test_recursive(self):
        """Test recursive loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file1.txt").write_text("Content 1")

            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (subdir / "file2.txt").write_text("Content 2")

            # Recursive (default)
            docs = list(load_folder(tmpdir, recursive=True))
            assert len(docs) == 2

            # Non-recursive
            docs = list(load_folder(tmpdir, recursive=False))
            assert len(docs) == 1


class TestSupportedExtensions:
    """Tests for extension support."""

    def test_get_supported(self):
        """Test getting supported extensions."""
        extensions = get_supported_extensions()

        assert ".txt" in extensions
        assert ".md" in extensions
        assert ".json" in extensions
        assert ".pdf" in extensions

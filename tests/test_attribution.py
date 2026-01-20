"""Tests for attribution module."""

import json
import tempfile
from pathlib import Path
import pytest

from doc2dataset.attribution import (
    SourceLocation,
    Attribution,
    SourceTracker,
    AttributionManager,
    AttributedDataset,
    extract_section_headers,
    get_section_at_offset,
)


class TestSourceLocation:
    """Tests for SourceLocation dataclass."""

    def test_basic_creation(self):
        loc = SourceLocation(file_path="/path/to/doc.pdf")
        assert loc.file_path == "/path/to/doc.pdf"
        assert loc.page_number is None
        assert loc.line_start is None

    def test_full_creation(self):
        loc = SourceLocation(
            file_path="/path/to/doc.pdf",
            page_number=5,
            line_start=10,
            line_end=15,
            char_start=100,
            char_end=200,
            section="Introduction",
            chunk_index=2,
        )
        assert loc.page_number == 5
        assert loc.line_start == 10
        assert loc.section == "Introduction"

    def test_to_dict(self):
        loc = SourceLocation(
            file_path="/doc.pdf",
            page_number=3,
            section="Methods",
        )
        d = loc.to_dict()
        assert d["file_path"] == "/doc.pdf"
        assert d["page_number"] == 3
        assert d["section"] == "Methods"
        # None values should be excluded
        assert "line_start" not in d

    def test_to_citation_basic(self):
        loc = SourceLocation(file_path="/path/to/document.pdf")
        citation = loc.to_citation()
        assert citation == "document.pdf"

    def test_to_citation_with_page(self):
        loc = SourceLocation(file_path="/doc.pdf", page_number=5)
        citation = loc.to_citation()
        assert "p. 5" in citation

    def test_to_citation_with_section(self):
        loc = SourceLocation(file_path="/doc.pdf", section="Introduction")
        citation = loc.to_citation()
        assert '"Introduction"' in citation

    def test_to_citation_with_lines(self):
        loc = SourceLocation(file_path="/doc.pdf", line_start=10, line_end=15)
        citation = loc.to_citation()
        assert "lines 10-15" in citation

    def test_to_citation_single_line(self):
        loc = SourceLocation(file_path="/doc.pdf", line_start=10, line_end=10)
        citation = loc.to_citation()
        assert "line 10" in citation


class TestAttribution:
    """Tests for Attribution dataclass."""

    def test_basic_creation(self):
        attr = Attribution(item_id="abc123")
        assert attr.item_id == "abc123"
        assert attr.sources == []
        assert attr.confidence == 1.0

    def test_full_creation(self):
        source = SourceLocation(file_path="/doc.pdf", page_number=1)
        attr = Attribution(
            item_id="abc123",
            sources=[source],
            extraction_method="qa_extraction",
            confidence=0.95,
            timestamp=1234567890.0,
            metadata={"model": "gpt-4"},
        )
        assert len(attr.sources) == 1
        assert attr.extraction_method == "qa_extraction"
        assert attr.confidence == 0.95

    def test_to_dict(self):
        source = SourceLocation(file_path="/doc.pdf")
        attr = Attribution(
            item_id="abc123",
            sources=[source],
            extraction_method="rules",
        )
        d = attr.to_dict()
        assert d["item_id"] == "abc123"
        assert len(d["sources"]) == 1
        assert d["extraction_method"] == "rules"

    def test_from_dict(self):
        data = {
            "item_id": "abc123",
            "sources": [{"file_path": "/doc.pdf", "page_number": 5}],
            "extraction_method": "facts",
            "confidence": 0.9,
            "timestamp": 1234567890.0,
            "metadata": {"key": "value"},
        }
        attr = Attribution.from_dict(data)
        assert attr.item_id == "abc123"
        assert len(attr.sources) == 1
        assert attr.sources[0].page_number == 5
        assert attr.confidence == 0.9


class TestSourceTracker:
    """Tests for SourceTracker."""

    def test_basic_creation(self):
        tracker = SourceTracker("/path/to/doc.pdf")
        assert tracker.file_path == "/path/to/doc.pdf"

    def test_with_content(self):
        content = "Line 1\nLine 2\nLine 3"
        tracker = SourceTracker("/doc.txt", content=content)
        assert tracker.content == content

    def test_set_page(self):
        tracker = SourceTracker("/doc.pdf")
        tracker.set_page(5)
        assert tracker.current_page == 5

    def test_set_section(self):
        tracker = SourceTracker("/doc.pdf")
        tracker.set_section("Introduction")
        assert tracker.current_section == "Introduction"

    def test_mark_span(self):
        content = "Hello world\nThis is a test\nMore content"
        tracker = SourceTracker("/doc.txt", content=content)
        tracker.set_page(1)
        tracker.set_section("Intro")

        loc = tracker.mark_span(char_start=0, char_end=11)
        assert loc.file_path == "/doc.txt"
        assert loc.page_number == 1
        assert loc.section == "Intro"
        assert loc.char_start == 0
        assert loc.char_end == 11

    def test_mark_span_line_numbers(self):
        content = "Line 1\nLine 2\nLine 3"
        tracker = SourceTracker("/doc.txt", content=content)

        # Mark span in line 2
        loc = tracker.mark_span(char_start=7, char_end=13)
        assert loc.line_start == 2

    def test_find_text(self):
        content = "This is some content with keywords inside."
        tracker = SourceTracker("/doc.txt", content=content)

        loc = tracker.find_text("keywords")
        assert loc is not None
        assert loc.char_start >= 0

    def test_find_text_not_found(self):
        content = "This is content"
        tracker = SourceTracker("/doc.txt", content=content)

        loc = tracker.find_text("nonexistent")
        assert loc is None

    def test_find_text_no_content(self):
        tracker = SourceTracker("/doc.pdf")
        loc = tracker.find_text("test")
        assert loc is None


class TestAttributionManager:
    """Tests for AttributionManager."""

    def test_basic_creation(self):
        manager = AttributionManager()
        assert len(manager.get_all()) == 0

    def test_create_attribution(self):
        manager = AttributionManager()
        attr = manager.create_attribution(
            content="Q: What is Python?\nA: A programming language",
            source_file="/doc.pdf",
            page=5,
            section="FAQ",
            extraction_method="qa_extraction",
        )
        assert attr.item_id is not None
        assert len(attr.sources) == 1
        assert attr.sources[0].page_number == 5

    def test_get_attribution(self):
        manager = AttributionManager()
        attr = manager.create_attribution(
            content="Test content",
            source_file="/doc.pdf",
        )
        retrieved = manager.get_attribution(attr.item_id)
        assert retrieved is not None
        assert retrieved.item_id == attr.item_id

    def test_get_nonexistent_attribution(self):
        manager = AttributionManager()
        result = manager.get_attribution("nonexistent")
        assert result is None

    def test_add_source(self):
        manager = AttributionManager()
        attr = manager.create_attribution(
            content="Test",
            source_file="/doc1.pdf",
        )
        new_source = SourceLocation(file_path="/doc2.pdf", page_number=10)
        manager.add_source(attr.item_id, new_source)

        updated = manager.get_attribution(attr.item_id)
        assert len(updated.sources) == 2

    def test_get_all(self):
        manager = AttributionManager()
        manager.create_attribution(content="Test 1", source_file="/doc1.pdf")
        manager.create_attribution(content="Test 2", source_file="/doc2.pdf")

        all_attrs = manager.get_all()
        assert len(all_attrs) == 2

    def test_get_tracker(self):
        manager = AttributionManager()
        tracker1 = manager.get_tracker("/doc.pdf", content="content")
        tracker2 = manager.get_tracker("/doc.pdf")

        # Should return same tracker for same file
        assert tracker1 is tracker2

    def test_to_dict(self):
        manager = AttributionManager()
        manager.create_attribution(content="Test", source_file="/doc.pdf")

        d = manager.to_dict()
        assert "attributions" in d
        assert "total_count" in d
        assert "source_files" in d
        assert d["total_count"] == 1

    def test_save_and_load(self):
        manager = AttributionManager()
        manager.create_attribution(
            content="Test content",
            source_file="/doc.pdf",
            page=5,
            extraction_method="rules",
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        manager.save(path)

        # Load into new manager
        new_manager = AttributionManager()
        new_manager.load(path)

        assert len(new_manager.get_all()) == 1
        attr = new_manager.get_all()[0]
        assert attr.sources[0].page_number == 5

        path.unlink()


class TestAttributedDataset:
    """Tests for AttributedDataset."""

    def test_basic_creation(self):
        dataset = AttributedDataset()
        assert len(dataset) == 0

    def test_add_item(self):
        dataset = AttributedDataset()
        source = SourceLocation(file_path="/doc.pdf")
        attr = Attribution(item_id="abc123", sources=[source])

        dataset.add(
            item={"input": "Question?", "output": "Answer"},
            attribution=attr,
        )

        assert len(dataset) == 1

    def test_iteration(self):
        dataset = AttributedDataset()
        attr = Attribution(item_id="abc")

        dataset.add({"input": "Q1"}, attr)
        dataset.add({"input": "Q2"}, attr)

        items = list(dataset)
        assert len(items) == 2
        # Each item is (item_dict, attribution)
        assert items[0][0]["input"] == "Q1"

    def test_get_items(self):
        dataset = AttributedDataset()
        attr = Attribution(item_id="abc")

        dataset.add({"input": "Q1", "output": "A1"}, attr)

        items = dataset.get_items()
        assert len(items) == 1
        # Attribution ID should be stripped
        assert "_attribution_id" not in items[0]

    def test_save(self):
        dataset = AttributedDataset()
        source = SourceLocation(file_path="/doc.pdf")
        attr = Attribution(item_id="abc123", sources=[source])

        dataset.add({"input": "Q", "output": "A"}, attr)

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        dataset.save(path, include_attribution=True)

        # Read back
        with open(path) as f:
            data = json.loads(f.readline())

        assert data["input"] == "Q"
        assert "_attribution" in data

        path.unlink()
        # Also clean up the attribution file
        attr_path = path.with_suffix(".attributions.json")
        if attr_path.exists():
            attr_path.unlink()

    def test_get_sources_for_item(self):
        dataset = AttributedDataset()
        source = SourceLocation(file_path="/doc.pdf", page_number=5)
        attr = Attribution(item_id="abc", sources=[source])

        dataset.add({"input": "Q"}, attr)

        sources = dataset.get_sources_for_item(0)
        assert len(sources) == 1
        assert "p. 5" in sources[0]

    def test_filter_by_source(self):
        dataset = AttributedDataset()
        attr1 = Attribution(
            item_id="a",
            sources=[SourceLocation(file_path="/doc1.pdf")],
        )
        attr2 = Attribution(
            item_id="b",
            sources=[SourceLocation(file_path="/doc2.pdf")],
        )

        dataset.add({"input": "Q1"}, attr1)
        dataset.add({"input": "Q2"}, attr2)

        filtered = dataset.filter_by_source("/doc1.pdf")
        assert len(filtered) == 1

    def test_stats(self):
        dataset = AttributedDataset()
        attr = Attribution(
            item_id="a",
            sources=[SourceLocation(file_path="/doc.pdf")],
            extraction_method="qa",
        )

        dataset.add({"input": "Q1"}, attr)
        dataset.add({"input": "Q2"}, attr)

        stats = dataset.stats()
        assert stats["total_items"] == 2
        assert stats["unique_sources"] == 1
        assert "qa" in stats["extraction_methods"]


class TestExtractSectionHeaders:
    """Tests for extract_section_headers function."""

    def test_markdown_headers(self):
        content = "# Header 1\nSome content\n## Header 2\nMore content"
        headers = extract_section_headers(content)
        assert len(headers) >= 2
        assert any("Header 1" in h[0] for h in headers)

    def test_underlined_headers(self):
        content = "Header\n======\nContent"
        headers = extract_section_headers(content)
        assert len(headers) >= 1
        assert any("Header" in h[0] for h in headers)

    def test_numbered_sections(self):
        content = "1.2.3 Section Name\nContent here"
        headers = extract_section_headers(content)
        # Should find numbered section
        assert len(headers) >= 1


class TestGetSectionAtOffset:
    """Tests for get_section_at_offset function."""

    def test_basic_lookup(self):
        headers = [("Intro", 0), ("Methods", 100), ("Results", 200)]
        section = get_section_at_offset(headers, 150)
        assert section == "Methods"

    def test_before_first_header(self):
        headers = [("Intro", 50), ("Methods", 100)]
        section = get_section_at_offset(headers, 25)
        assert section is None

    def test_at_header_boundary(self):
        headers = [("Intro", 0), ("Methods", 100)]
        section = get_section_at_offset(headers, 100)
        assert section == "Methods"

    def test_empty_headers(self):
        section = get_section_at_offset([], 50)
        assert section is None

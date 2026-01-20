"""Tests for extractors module."""

import json
import pytest

from doc2dataset.loaders import Document
from doc2dataset.extractors import (
    QAExtractor,
    RulesExtractor,
    FactsExtractor,
    InstructionExtractor,
    get_extractor,
)


def create_mock_llm(response: str):
    """Create a mock LLM function that returns a fixed response."""
    def mock_llm(prompt: str) -> str:
        return response
    return mock_llm


class TestQAExtractor:
    """Tests for QAExtractor."""

    def test_extraction(self):
        """Test Q&A extraction."""
        response = json.dumps([
            {"question": "What is Python?", "answer": "A programming language."},
            {"question": "Who created it?", "answer": "Guido van Rossum."},
        ])

        llm_fn = create_mock_llm(response)
        extractor = QAExtractor(llm_fn)

        doc = Document(content="Python is a programming language created by Guido.")
        result = extractor.extract(doc)

        assert len(result.items) == 2
        assert result.items[0]["question"] == "What is Python?"
        assert result.items[0]["answer"] == "A programming language."

    def test_parse_response_with_extras(self):
        """Test parsing response with extra content."""
        response = """
Here are some Q&A pairs:
[
    {"question": "What is X?", "answer": "X is Y."},
    {"question": "How does Z work?", "answer": "Z works by..."}
]
        """

        llm_fn = create_mock_llm(response)
        extractor = QAExtractor(llm_fn)
        items = extractor.parse_response(response)

        assert len(items) == 2

    def test_invalid_json_handling(self):
        """Test handling of invalid JSON."""
        response = "This is not JSON at all."

        llm_fn = create_mock_llm(response)
        extractor = QAExtractor(llm_fn)
        items = extractor.parse_response(response)

        assert len(items) == 0


class TestRulesExtractor:
    """Tests for RulesExtractor."""

    def test_extraction(self):
        """Test rules extraction."""
        response = json.dumps([
            {
                "rule": "Always wash hands before cooking",
                "context": "Food preparation",
                "category": "safety",
                "rationale": "Prevents contamination"
            },
        ])

        llm_fn = create_mock_llm(response)
        extractor = RulesExtractor(llm_fn)

        doc = Document(content="Food safety guidelines: Always wash hands.")
        result = extractor.extract(doc)

        assert len(result.items) == 1
        assert result.items[0]["rule"] == "Always wash hands before cooking"
        assert result.items[0]["category"] == "safety"

    def test_custom_categories(self):
        """Test with custom categories."""
        response = json.dumps([
            {"rule": "Test rule", "category": "custom_cat"}
        ])

        llm_fn = create_mock_llm(response)
        extractor = RulesExtractor(
            llm_fn,
            categories=["custom_cat", "another_cat"]
        )

        assert "custom_cat" in extractor.categories


class TestFactsExtractor:
    """Tests for FactsExtractor."""

    def test_extraction(self):
        """Test facts extraction."""
        response = json.dumps([
            {"fact": "Python was released in 1991", "topic": "history", "confidence": "high"},
            {"fact": "It uses dynamic typing", "topic": "features", "confidence": "high"},
        ])

        llm_fn = create_mock_llm(response)
        extractor = FactsExtractor(llm_fn)

        doc = Document(content="Python history and features.")
        result = extractor.extract(doc)

        assert len(result.items) == 2
        assert result.items[0]["fact"] == "Python was released in 1991"


class TestInstructionExtractor:
    """Tests for InstructionExtractor."""

    def test_extraction(self):
        """Test instruction extraction."""
        response = json.dumps([
            {
                "instruction": "Explain how to install Python",
                "input": "",
                "output": "To install Python, download from python.org..."
            },
        ])

        llm_fn = create_mock_llm(response)
        extractor = InstructionExtractor(llm_fn)

        doc = Document(content="Python installation guide.")
        result = extractor.extract(doc)

        assert len(result.items) == 1
        assert "Explain how to install" in result.items[0]["instruction"]


class TestGetExtractor:
    """Tests for get_extractor factory."""

    def test_get_qa(self):
        """Test getting QA extractor."""
        llm_fn = create_mock_llm("[]")
        extractor = get_extractor("qa", llm_fn)
        assert isinstance(extractor, QAExtractor)

    def test_get_rules(self):
        """Test getting rules extractor."""
        llm_fn = create_mock_llm("[]")
        extractor = get_extractor("rules", llm_fn)
        assert isinstance(extractor, RulesExtractor)

    def test_unknown_type(self):
        """Test error on unknown type."""
        llm_fn = create_mock_llm("[]")
        with pytest.raises(ValueError, match="Unknown extractor type"):
            get_extractor("unknown_type", llm_fn)


class TestChunking:
    """Tests for content chunking."""

    def test_short_content(self):
        """Test that short content isn't chunked."""
        llm_fn = create_mock_llm("[]")
        extractor = QAExtractor(llm_fn, chunk_size=1000)

        chunks = extractor._chunk_content("Short content")
        assert len(chunks) == 1

    def test_long_content(self):
        """Test that long content is chunked."""
        llm_fn = create_mock_llm("[]")
        extractor = QAExtractor(llm_fn, chunk_size=100)

        long_content = "This is a long paragraph. " * 50
        chunks = extractor._chunk_content(long_content)

        assert len(chunks) > 1
        # Each chunk should be roughly within size limit
        for chunk in chunks:
            assert len(chunk) <= 150  # Some tolerance for break points

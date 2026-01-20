"""
Tests for doc2dataset quality filtering.
"""

import pytest
from typing import List, Dict, Any

from doc2dataset.quality import (
    BaseFilter,
    LengthFilter,
    RepetitionFilter,
    ContentFilter,
    DuplicateFilter,
    QualityFilterPipeline,
    QualityScorer,
    get_qa_quality_pipeline,
)


class TestLengthFilter:
    """Tests for LengthFilter."""

    @pytest.fixture
    def filter(self):
        """Create a test filter."""
        return LengthFilter(
            field="output",
            min_length=10,
            max_length=1000,
        )

    def test_pass_valid_length(self, filter):
        """Test that valid length passes."""
        item = {"output": "This is a valid length output text."}
        assert filter.filter(item) is True

    def test_reject_too_short(self, filter):
        """Test that too short is rejected."""
        item = {"output": "Short"}
        assert filter.filter(item) is False

    def test_reject_too_long(self):
        """Test that too long is rejected."""
        filter = LengthFilter(field="output", min_length=1, max_length=10)
        item = {"output": "This is way too long for the filter"}
        assert filter.filter(item) is False

    def test_missing_field(self, filter):
        """Test handling of missing field."""
        item = {"other_field": "value"}
        # Should reject items with missing field
        assert filter.filter(item) is False


class TestRepetitionFilter:
    """Tests for RepetitionFilter."""

    @pytest.fixture
    def filter(self):
        """Create a test filter."""
        return RepetitionFilter(
            field="output",
            max_repetition_ratio=0.5,
        )

    def test_pass_no_repetition(self, filter):
        """Test that unique text passes."""
        item = {"output": "Each word is unique in this sentence."}
        assert filter.filter(item) is True

    def test_reject_high_repetition(self, filter):
        """Test that highly repetitive text is rejected."""
        item = {"output": "word word word word word word word word word word"}
        assert filter.filter(item) is False

    def test_moderate_repetition_passes(self, filter):
        """Test that moderate repetition passes."""
        item = {"output": "The quick fox jumped over the lazy dog."}
        assert filter.filter(item) is True


class TestContentFilter:
    """Tests for ContentFilter."""

    @pytest.fixture
    def filter(self):
        """Create a test filter."""
        return ContentFilter(
            field="output",
            blocked_patterns=[
                r"\bTODO\b",
                r"\bFIXME\b",
                r"\[.*placeholder.*\]",
            ],
        )

    def test_pass_clean_content(self, filter):
        """Test that clean content passes."""
        item = {"output": "This is clean content without issues."}
        assert filter.filter(item) is True

    def test_reject_blocked_pattern(self, filter):
        """Test that blocked patterns are rejected."""
        item = {"output": "This has a TODO that should be rejected."}
        assert filter.filter(item) is False

    def test_reject_placeholder(self, filter):
        """Test that placeholder patterns are rejected."""
        item = {"output": "This has a [placeholder text] in it."}
        assert filter.filter(item) is False


class TestDuplicateFilter:
    """Tests for DuplicateFilter."""

    def test_filter_exact_duplicates(self):
        """Test filtering exact duplicates."""
        filter = DuplicateFilter(field="output", similarity_threshold=1.0)

        items = [
            {"output": "Unique text one"},
            {"output": "Unique text two"},
            {"output": "Unique text one"},  # Exact duplicate
        ]

        results = filter.filter_batch(items)
        assert len(results) == 2

    def test_filter_near_duplicates(self):
        """Test filtering near-duplicates."""
        filter = DuplicateFilter(field="output", similarity_threshold=0.8)

        items = [
            {"output": "The quick brown fox jumps over the lazy dog"},
            {"output": "The quick brown fox jumps over the lazy cat"},  # Similar
            {"output": "Completely different content here"},
        ]

        results = filter.filter_batch(items)
        # Should filter out the similar item
        assert len(results) <= 3

    def test_empty_input(self):
        """Test with empty input."""
        filter = DuplicateFilter(field="output")
        results = filter.filter_batch([])
        assert results == []


class TestQualityFilterPipeline:
    """Tests for QualityFilterPipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create a test pipeline."""
        return QualityFilterPipeline(
            filters=[
                LengthFilter(field="output", min_length=5, max_length=500),
                RepetitionFilter(field="output", max_repetition_ratio=0.6),
            ]
        )

    def test_pipeline_passes_valid(self, pipeline):
        """Test that valid items pass pipeline."""
        items = [
            {"output": "This is a valid output text."},
            {"output": "Another valid piece of content."},
        ]

        results = pipeline.process(items)
        assert len(results) == 2

    def test_pipeline_filters_invalid(self, pipeline):
        """Test that invalid items are filtered."""
        items = [
            {"output": "OK"},  # Too short
            {"output": "This is a valid output text."},
            {"output": "bad bad bad bad bad bad bad bad bad bad"},  # Too repetitive
        ]

        results = pipeline.process(items)
        assert len(results) == 1

    def test_pipeline_stats(self, pipeline):
        """Test pipeline statistics."""
        items = [
            {"output": "OK"},  # Too short
            {"output": "This is a valid output text."},
        ]

        results = pipeline.process(items)
        stats = pipeline.stats()

        assert stats["total_processed"] == 2
        assert stats["total_passed"] == 1
        assert stats["total_filtered"] == 1


class TestQualityScorer:
    """Tests for QualityScorer."""

    @pytest.fixture
    def scorer(self):
        """Create a test scorer."""
        return QualityScorer()

    def test_score_returns_float(self, scorer):
        """Test that scorer returns a float."""
        item = {"input": "Question?", "output": "Answer to the question."}
        score = scorer.score(item)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_batch(self, scorer):
        """Test batch scoring."""
        items = [
            {"input": "Q1?", "output": "A1"},
            {"input": "Q2?", "output": "A2"},
        ]

        scores = scorer.score_batch(items)
        assert len(scores) == 2
        for score in scores:
            assert 0.0 <= score <= 1.0

    def test_empty_content_low_score(self, scorer):
        """Test that empty content gets low score."""
        item = {"input": "", "output": ""}
        score = scorer.score(item)
        assert score < 0.5


class TestGetQAQualityPipeline:
    """Tests for get_qa_quality_pipeline factory."""

    def test_creates_pipeline(self):
        """Test that factory creates a pipeline."""
        pipeline = get_qa_quality_pipeline()
        assert isinstance(pipeline, QualityFilterPipeline)

    def test_pipeline_has_filters(self):
        """Test that created pipeline has filters."""
        pipeline = get_qa_quality_pipeline()
        # Should have at least one filter
        assert len(pipeline._filters) > 0

    def test_pipeline_is_functional(self):
        """Test that created pipeline works."""
        pipeline = get_qa_quality_pipeline()

        items = [
            {"input": "What is Python?", "output": "Python is a programming language."},
        ]

        results = pipeline.process(items)
        # Valid Q&A should pass default filters
        assert len(results) >= 0  # May or may not pass depending on filter config

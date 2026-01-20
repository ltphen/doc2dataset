"""
Tests for doc2dataset analytics.
"""

import pytest
import json
from pathlib import Path
from typing import List, Dict, Any

from doc2dataset.analytics import (
    DatasetStats,
    DatasetAnalyzer,
    ProcessingAnalytics,
    analyze_jsonl_file,
    compare_datasets,
)


class TestDatasetStats:
    """Tests for DatasetStats."""

    def test_create_stats(self):
        """Test creating dataset stats."""
        stats = DatasetStats(
            total_items=100,
            total_tokens=5000,
            avg_tokens_per_item=50.0,
        )

        assert stats.total_items == 100
        assert stats.total_tokens == 5000
        assert stats.avg_tokens_per_item == 50.0

    def test_stats_to_dict(self):
        """Test converting stats to dict."""
        stats = DatasetStats(
            total_items=100,
            total_tokens=5000,
            avg_tokens_per_item=50.0,
        )

        d = stats.to_dict()
        assert d["total_items"] == 100
        assert d["total_tokens"] == 5000


class TestDatasetAnalyzer:
    """Tests for DatasetAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a test analyzer."""
        return DatasetAnalyzer()

    @pytest.fixture
    def sample_items(self):
        """Create sample dataset items."""
        return [
            {
                "input": "What is Python?",
                "output": "Python is a programming language.",
                "type": "qa",
                "source": "doc1.pdf",
            },
            {
                "input": "How do decorators work?",
                "output": "Decorators are functions that modify other functions.",
                "type": "qa",
                "source": "doc1.pdf",
            },
            {
                "input": "What is a list comprehension?",
                "output": "A concise way to create lists in Python.",
                "type": "qa",
                "source": "doc2.pdf",
            },
        ]

    def test_analyze_basic(self, analyzer, sample_items):
        """Test basic analysis."""
        stats = analyzer.analyze(sample_items)

        assert stats.total_items == 3
        assert stats.total_tokens > 0
        assert stats.avg_tokens_per_item > 0

    def test_analyze_empty(self, analyzer):
        """Test analyzing empty dataset."""
        stats = analyzer.analyze([])

        assert stats.total_items == 0

    def test_analyze_field_stats(self, analyzer, sample_items):
        """Test that field stats are computed."""
        stats = analyzer.analyze(sample_items)

        assert "input" in stats.field_stats
        assert "output" in stats.field_stats

        # Check input field stats
        input_stats = stats.field_stats["input"]
        assert "char_length" in input_stats
        assert "token_length" in input_stats

    def test_analyze_extraction_types(self, analyzer, sample_items):
        """Test extraction type distribution."""
        stats = analyzer.analyze(sample_items)

        assert "qa" in stats.extraction_types
        assert stats.extraction_types["qa"] == 3

    def test_analyze_source_files(self, analyzer, sample_items):
        """Test source file distribution."""
        stats = analyzer.analyze(sample_items)

        assert len(stats.source_files) == 2
        assert "doc1.pdf" in stats.source_files

    def test_analyze_text_quality(self, analyzer, sample_items):
        """Test text quality analysis."""
        quality = analyzer.analyze_text_quality(sample_items, field="output")

        assert "word_count" in quality
        assert "sentence_count" in quality
        assert "vocabulary_diversity" in quality

    def test_analyze_duplicates(self, analyzer):
        """Test duplicate analysis."""
        items = [
            {"output": "Duplicate text here."},
            {"output": "Duplicate text here."},  # Exact duplicate
            {"output": "Unique content."},
        ]

        dup_stats = analyzer.analyze_duplicates(items)

        assert dup_stats["exact_duplicates"] == 1
        assert dup_stats["unique_items"] == 2

    def test_analyze_vocabulary(self, analyzer, sample_items):
        """Test vocabulary analysis."""
        vocab = analyzer.analyze_vocabulary(sample_items, field="output")

        assert vocab["total_words"] > 0
        assert vocab["unique_words"] > 0
        assert "top_words" in vocab

    def test_generate_text_report(self, analyzer, sample_items):
        """Test text report generation."""
        stats = analyzer.analyze(sample_items)
        report = analyzer.generate_report(stats, format="text")

        assert "DATASET ANALYSIS REPORT" in report
        assert "Total Items" in report

    def test_generate_markdown_report(self, analyzer, sample_items):
        """Test markdown report generation."""
        stats = analyzer.analyze(sample_items)
        report = analyzer.generate_report(stats, format="markdown")

        assert "# Dataset Analysis Report" in report
        assert "|" in report  # Markdown tables


class TestProcessingAnalytics:
    """Tests for ProcessingAnalytics."""

    @pytest.fixture
    def analytics(self):
        """Create a test analytics tracker."""
        return ProcessingAnalytics()

    def test_start_session(self, analytics):
        """Test starting a session."""
        analytics.start_session()
        assert analytics._start_time is not None

    def test_record_document_processed(self, analytics):
        """Test recording processed document."""
        analytics.start_session()
        analytics.record_document_processed(
            source="doc1.pdf",
            items_extracted=10,
            processing_time=1.5,
            tokens_used=100,
            cost=0.01,
        )

        summary = analytics.summary()
        assert summary["documents_processed"] == 1
        assert summary["items_extracted"] == 10

    def test_record_error(self, analytics):
        """Test recording errors."""
        analytics.record_error("doc1.pdf", "Error message")

        errors = analytics.get_errors()
        assert len(errors) == 1
        assert errors[0]["source"] == "doc1.pdf"

    def test_summary(self, analytics):
        """Test getting summary."""
        analytics.start_session()
        analytics.record_document_processed("doc1.pdf", 5, 1.0)
        analytics.record_document_processed("doc2.pdf", 10, 2.0)

        summary = analytics.summary()

        assert summary["documents_processed"] == 2
        assert summary["items_extracted"] == 15
        assert "processing_time" in summary
        assert "throughput" in summary

    def test_export(self, analytics, tmp_path):
        """Test exporting analytics."""
        analytics.start_session()
        analytics.record_document_processed("doc1.pdf", 5, 1.0)

        output_path = tmp_path / "analytics.json"
        analytics.export(output_path)

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert "summary" in data
        assert "per_document" in data


class TestAnalyzeJsonlFile:
    """Tests for analyze_jsonl_file function."""

    def test_analyze_file(self, tmp_path):
        """Test analyzing a JSONL file."""
        # Create test file
        file_path = tmp_path / "test.jsonl"
        items = [
            {"input": "Q1", "output": "A1"},
            {"input": "Q2", "output": "A2"},
        ]

        with open(file_path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

        stats = analyze_jsonl_file(file_path)

        assert stats.total_items == 2


class TestCompareDatasets:
    """Tests for compare_datasets function."""

    def test_compare_two_datasets(self):
        """Test comparing two datasets."""
        dataset1 = [
            {"input": "Q", "output": "A"},
            {"input": "Q", "output": "A"},
        ]
        dataset2 = [
            {"input": "Q", "output": "A"},
        ]

        results = compare_datasets({
            "dataset1": dataset1,
            "dataset2": dataset2,
        })

        assert "dataset1" in results
        assert "dataset2" in results
        assert "_comparison" in results

        assert results["_comparison"]["size_ratio"] == 2.0

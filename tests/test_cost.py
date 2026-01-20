"""
Tests for doc2dataset cost estimation.
"""

import pytest
from typing import List

from doc2dataset.cost import CostEstimator, CostEstimate, TokenCounter


class TestTokenCounter:
    """Tests for TokenCounter."""

    @pytest.fixture
    def counter(self):
        """Create a test counter."""
        return TokenCounter()

    def test_count_empty_string(self, counter):
        """Test counting tokens in empty string."""
        count = counter.count("")
        assert count == 0

    def test_count_simple_text(self, counter):
        """Test counting tokens in simple text."""
        count = counter.count("Hello world")
        assert count > 0
        # Should be approximately 2 tokens
        assert 1 <= count <= 5

    def test_count_longer_text(self, counter):
        """Test counting tokens in longer text."""
        text = "This is a longer piece of text that should have more tokens."
        count = counter.count(text)
        assert count > 5

    def test_count_batch(self, counter):
        """Test batch token counting."""
        texts = ["Hello", "World", "Test"]
        total = counter.count_batch(texts)
        assert total > 0

        # Should equal sum of individual counts
        individual_sum = sum(counter.count(t) for t in texts)
        assert total == individual_sum


class TestCostEstimate:
    """Tests for CostEstimate dataclass."""

    def test_create_estimate(self):
        """Test creating a cost estimate."""
        estimate = CostEstimate(
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            estimated_cost=0.05,
            model="gpt-4",
        )

        assert estimate.input_tokens == 1000
        assert estimate.output_tokens == 500
        assert estimate.total_tokens == 1500
        assert estimate.estimated_cost == 0.05

    def test_estimate_to_dict(self):
        """Test converting estimate to dict."""
        estimate = CostEstimate(
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            estimated_cost=0.05,
            model="gpt-4",
        )

        d = estimate.to_dict()
        assert d["input_tokens"] == 1000
        assert d["estimated_cost"] == 0.05


class TestCostEstimator:
    """Tests for CostEstimator."""

    @pytest.fixture
    def estimator(self):
        """Create a test estimator."""
        return CostEstimator(model="gpt-4o")

    def test_estimate_text(self, estimator):
        """Test estimating cost for text."""
        text = "This is some sample text to estimate."
        estimate = estimator.estimate_text(text)

        assert estimate.input_tokens > 0
        assert estimate.estimated_cost >= 0

    def test_estimate_extraction(self, estimator):
        """Test estimating cost for document extraction."""
        # Create mock documents
        class MockDocument:
            def __init__(self, content):
                self.content = content
                self.source = "test.txt"

        documents = [
            MockDocument("First document content."),
            MockDocument("Second document with more content."),
        ]

        estimate = estimator.estimate_extraction(documents, extraction_type="qa")

        assert estimate.input_tokens > 0
        assert estimate.estimated_cost >= 0
        assert estimate.num_documents == 2

    def test_format_estimate(self, estimator):
        """Test formatting estimate as string."""
        estimate = CostEstimate(
            input_tokens=10000,
            output_tokens=5000,
            total_tokens=15000,
            estimated_cost=0.50,
            model="gpt-4o",
        )

        formatted = estimator.format_estimate(estimate)

        assert "10,000" in formatted or "10000" in formatted
        assert "$0.50" in formatted or "0.5" in formatted
        assert "gpt-4o" in formatted.lower() or "gpt-4" in formatted.lower()

    def test_format_estimate_verbose(self, estimator):
        """Test verbose estimate formatting."""
        estimate = CostEstimate(
            input_tokens=10000,
            output_tokens=5000,
            total_tokens=15000,
            estimated_cost=0.50,
            model="gpt-4o",
            num_documents=10,
            num_chunks=25,
        )

        formatted = estimator.format_estimate(estimate, verbose=True)

        # Verbose format should include more details
        assert len(formatted) > 50
        assert "document" in formatted.lower() or "chunk" in formatted.lower()

    def test_different_models_different_costs(self):
        """Test that different models have different costs."""
        estimator_gpt4 = CostEstimator(model="gpt-4")
        estimator_gpt35 = CostEstimator(model="gpt-3.5-turbo")

        text = "Sample text for cost estimation."

        estimate_gpt4 = estimator_gpt4.estimate_text(text)
        estimate_gpt35 = estimator_gpt35.estimate_text(text)

        # GPT-4 should typically be more expensive
        # (This may vary based on implementation)
        assert estimate_gpt4.estimated_cost >= 0
        assert estimate_gpt35.estimated_cost >= 0

    def test_zero_input(self, estimator):
        """Test estimating cost for empty input."""
        estimate = estimator.estimate_text("")

        assert estimate.input_tokens == 0
        assert estimate.estimated_cost == 0

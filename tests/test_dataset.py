"""Tests for dataset module."""

import json
import tempfile
from pathlib import Path

import pytest

from doc2dataset.dataset import Dataset, DatasetItem


class TestDatasetItem:
    """Tests for DatasetItem."""

    def test_create(self):
        """Test creating a dataset item."""
        item = DatasetItem(
            data={"question": "What?", "answer": "That."},
            source="test.txt",
            extractor_type="qa",
        )

        assert item.data["question"] == "What?"
        assert item.source == "test.txt"

    def test_to_dict(self):
        """Test converting to dictionary."""
        item = DatasetItem(
            data={"key": "value"},
            source="test.txt",
        )

        d = item.to_dict()
        assert d["data"] == {"key": "value"}
        assert d["source"] == "test.txt"

    def test_from_dict(self):
        """Test creating from dictionary."""
        d = {
            "data": {"key": "value"},
            "source": "test.txt",
            "extractor_type": "qa",
            "metadata": {},
        }

        item = DatasetItem.from_dict(d)
        assert item.data == {"key": "value"}


class TestDataset:
    """Tests for Dataset."""

    def test_create_empty(self):
        """Test creating empty dataset."""
        dataset = Dataset()
        assert len(dataset) == 0

    def test_add_item(self):
        """Test adding items."""
        dataset = Dataset()
        item = dataset.add({"question": "What?", "answer": "That."})

        assert len(dataset) == 1
        assert dataset[0].data["question"] == "What?"

    def test_add_many(self):
        """Test adding multiple items."""
        dataset = Dataset()
        items = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]

        added = dataset.add_many(items, source="test.txt")

        assert len(dataset) == 2
        assert all(item.source == "test.txt" for item in added)

    def test_filter(self):
        """Test filtering items."""
        dataset = Dataset()
        dataset.add({"type": "qa"}, extractor_type="qa")
        dataset.add({"type": "rules"}, extractor_type="rules")
        dataset.add({"type": "qa"}, extractor_type="qa")

        filtered = dataset.filter(lambda item: item.extractor_type == "qa")

        assert len(filtered) == 2

    def test_filter_by_source(self):
        """Test filtering by source."""
        dataset = Dataset()
        dataset.add({"a": 1}, source="doc1.txt")
        dataset.add({"b": 2}, source="doc2.txt")
        dataset.add({"c": 3}, source="doc1.txt")

        filtered = dataset.filter_by_source("doc1")

        assert len(filtered) == 2

    def test_filter_by_type(self):
        """Test filtering by extractor type."""
        dataset = Dataset()
        dataset.add({"a": 1}, extractor_type="qa")
        dataset.add({"b": 2}, extractor_type="rules")

        filtered = dataset.filter_by_type("qa")

        assert len(filtered) == 1

    def test_shuffle(self):
        """Test shuffling."""
        dataset = Dataset()
        for i in range(10):
            dataset.add({"i": i})

        shuffled = dataset.shuffle(seed=42)

        assert len(shuffled) == 10
        # Check that order changed (very unlikely to be same with seed)
        original_order = [item.data["i"] for item in dataset]
        shuffled_order = [item.data["i"] for item in shuffled]
        assert original_order != shuffled_order

    def test_split(self):
        """Test train/val split."""
        dataset = Dataset()
        for i in range(100):
            dataset.add({"i": i})

        train, val = dataset.split(train_ratio=0.8, seed=42)

        assert len(train) == 80
        assert len(val) == 20

    def test_deduplicate(self):
        """Test deduplication."""
        dataset = Dataset()
        dataset.add({"q": "What?", "a": "That."})
        dataset.add({"q": "What?", "a": "That."})  # Duplicate
        dataset.add({"q": "Different", "a": "Answer"})

        deduped = dataset.deduplicate()

        assert len(deduped) == 2

    def test_merge(self):
        """Test merging datasets."""
        ds1 = Dataset()
        ds1.add({"a": 1})

        ds2 = Dataset()
        ds2.add({"b": 2})

        merged = ds1.merge(ds2)

        assert len(merged) == 2

    def test_sample(self):
        """Test random sampling."""
        dataset = Dataset()
        for i in range(100):
            dataset.add({"i": i})

        sampled = dataset.sample(10, seed=42)

        assert len(sampled) == 10

    def test_to_jsonl(self):
        """Test JSONL export."""
        dataset = Dataset()
        dataset.add({"question": "Q1", "answer": "A1"})
        dataset.add({"question": "Q2", "answer": "A2"})

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            path = f.name

        try:
            count = dataset.to_jsonl(path, format="openai")

            assert count == 2

            # Verify content
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2

            data = json.loads(lines[0])
            assert "messages" in data
        finally:
            Path(path).unlink()

    def test_to_json(self):
        """Test JSON export."""
        dataset = Dataset()
        dataset.add({"key": "value"})

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            count = dataset.to_json(path)

            assert count == 1

            with open(path) as f:
                data = json.load(f)
            assert "items" in data
        finally:
            Path(path).unlink()

    def test_from_json(self):
        """Test loading from JSON."""
        data = {
            "metadata": {"source": "test"},
            "items": [
                {"data": {"key": "value"}, "source": "", "extractor_type": "", "metadata": {}}
            ]
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            path = f.name

        try:
            dataset = Dataset.from_json(path)

            assert len(dataset) == 1
            assert dataset[0].data["key"] == "value"
        finally:
            Path(path).unlink()

    def test_from_jsonl(self):
        """Test loading from JSONL."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            f.write('{"question": "Q1", "answer": "A1"}\n')
            f.write('{"question": "Q2", "answer": "A2"}\n')
            path = f.name

        try:
            dataset = Dataset.from_jsonl(path)

            assert len(dataset) == 2
        finally:
            Path(path).unlink()

    def test_statistics(self):
        """Test computing statistics."""
        dataset = Dataset()
        dataset.add({"a": 1}, source="doc1.txt", extractor_type="qa")
        dataset.add({"b": 2}, source="doc1.txt", extractor_type="rules")
        dataset.add({"c": 3}, source="doc2.txt", extractor_type="qa")

        stats = dataset.statistics()

        assert stats["total_items"] == 3
        assert stats["sources"]["doc1.txt"] == 2
        assert stats["extractor_types"]["qa"] == 2

    def test_iteration(self):
        """Test iterating over dataset."""
        dataset = Dataset()
        dataset.add({"i": 0})
        dataset.add({"i": 1})

        items = list(dataset)
        assert len(items) == 2

    def test_repr(self):
        """Test string representation."""
        dataset = Dataset()
        dataset.add({"a": 1})

        assert "Dataset(items=1)" in repr(dataset)

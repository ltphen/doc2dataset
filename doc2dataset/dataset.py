"""
Dataset management for doc2dataset.

This module provides classes for managing and manipulating
extracted training data.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from pydantic import BaseModel


class DatasetItem(BaseModel):
    """
    A single item in the dataset.

    Uses Pydantic for validation and serialization.
    """

    data: Dict[str, Any]
    source: str = ""
    extractor_type: str = ""
    metadata: Dict[str, Any] = {}

    class Config:
        extra = "allow"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data": self.data,
            "source": self.source,
            "extractor_type": self.extractor_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetItem":
        """Create from dictionary."""
        return cls(**d)


class Dataset:
    """
    Collection of extracted training data items.

    Provides methods for manipulation, filtering, and export
    to various fine-tuning formats.

    Attributes:
        items: List of dataset items.
        metadata: Dataset-level metadata.

    Example:
        >>> dataset = Dataset()
        >>> dataset.add({"question": "What is X?", "answer": "X is..."})
        >>> dataset.to_jsonl("output.jsonl", format="openai")
    """

    def __init__(
        self,
        items: Optional[List[DatasetItem]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            items: Initial list of items.
            metadata: Dataset metadata.
        """
        self.items: List[DatasetItem] = items or []
        self.metadata: Dict[str, Any] = metadata or {}

    def __len__(self) -> int:
        """Return number of items."""
        return len(self.items)

    def __iter__(self) -> Iterator[DatasetItem]:
        """Iterate over items."""
        return iter(self.items)

    def __getitem__(self, index: int) -> DatasetItem:
        """Get item by index."""
        return self.items[index]

    def add(
        self,
        data: Dict[str, Any],
        source: str = "",
        extractor_type: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DatasetItem:
        """
        Add an item to the dataset.

        Args:
            data: The extracted data dictionary.
            source: Source document reference.
            extractor_type: Type of extractor used.
            metadata: Additional metadata.

        Returns:
            The added DatasetItem.
        """
        item = DatasetItem(
            data=data,
            source=source,
            extractor_type=extractor_type,
            metadata=metadata or {},
        )
        self.items.append(item)
        return item

    def add_many(
        self,
        items: List[Dict[str, Any]],
        source: str = "",
        extractor_type: str = "",
    ) -> List[DatasetItem]:
        """
        Add multiple items to the dataset.

        Args:
            items: List of data dictionaries.
            source: Source document reference.
            extractor_type: Type of extractor used.

        Returns:
            List of added DatasetItems.
        """
        added = []
        for data in items:
            item = self.add(data, source, extractor_type)
            added.append(item)
        return added

    def filter(
        self,
        predicate: callable,
    ) -> "Dataset":
        """
        Filter items by a predicate function.

        Args:
            predicate: Function that takes DatasetItem and returns bool.

        Returns:
            New Dataset with filtered items.
        """
        filtered_items = [item for item in self.items if predicate(item)]
        return Dataset(items=filtered_items, metadata=self.metadata.copy())

    def filter_by_source(self, source_pattern: str) -> "Dataset":
        """
        Filter items by source pattern.

        Args:
            source_pattern: Pattern to match in source string.

        Returns:
            New Dataset with filtered items.
        """
        return self.filter(lambda item: source_pattern in item.source)

    def filter_by_type(self, extractor_type: str) -> "Dataset":
        """
        Filter items by extractor type.

        Args:
            extractor_type: Type to filter by.

        Returns:
            New Dataset with filtered items.
        """
        return self.filter(lambda item: item.extractor_type == extractor_type)

    def shuffle(self, seed: Optional[int] = None) -> "Dataset":
        """
        Return a new dataset with shuffled items.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            New Dataset with shuffled items.
        """
        items_copy = self.items.copy()
        if seed is not None:
            random.seed(seed)
        random.shuffle(items_copy)
        return Dataset(items=items_copy, metadata=self.metadata.copy())

    def split(
        self,
        train_ratio: float = 0.8,
        seed: Optional[int] = None,
    ) -> tuple["Dataset", "Dataset"]:
        """
        Split dataset into train and validation sets.

        Args:
            train_ratio: Proportion of data for training.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        shuffled = self.shuffle(seed)
        split_idx = int(len(shuffled) * train_ratio)

        train_items = shuffled.items[:split_idx]
        val_items = shuffled.items[split_idx:]

        return (
            Dataset(items=train_items, metadata=self.metadata.copy()),
            Dataset(items=val_items, metadata=self.metadata.copy()),
        )

    def deduplicate(self, key_fn: Optional[callable] = None) -> "Dataset":
        """
        Remove duplicate items.

        Args:
            key_fn: Function to generate dedup key from item.
                   Defaults to JSON string of data.

        Returns:
            New Dataset with duplicates removed.
        """
        if key_fn is None:
            key_fn = lambda item: json.dumps(item.data, sort_keys=True)

        seen = set()
        unique_items = []

        for item in self.items:
            key = key_fn(item)
            if key not in seen:
                seen.add(key)
                unique_items.append(item)

        return Dataset(items=unique_items, metadata=self.metadata.copy())

    def merge(self, other: "Dataset") -> "Dataset":
        """
        Merge with another dataset.

        Args:
            other: Dataset to merge with.

        Returns:
            New Dataset with combined items.
        """
        combined_items = self.items + other.items
        combined_metadata = {**self.metadata, **other.metadata}
        return Dataset(items=combined_items, metadata=combined_metadata)

    def sample(self, n: int, seed: Optional[int] = None) -> "Dataset":
        """
        Random sample of items.

        Args:
            n: Number of items to sample.
            seed: Random seed.

        Returns:
            New Dataset with sampled items.
        """
        if seed is not None:
            random.seed(seed)
        sampled = random.sample(self.items, min(n, len(self.items)))
        return Dataset(items=sampled, metadata=self.metadata.copy())

    def to_jsonl(
        self,
        path: Union[str, Path],
        format: str = "generic",
        **kwargs: Any,
    ) -> int:
        """
        Export dataset to JSONL file.

        Args:
            path: Output file path.
            format: Output format ("generic", "openai", "alpaca", "sharegpt").
            **kwargs: Additional format-specific options.

        Returns:
            Number of items written.
        """
        from doc2dataset.formatters import get_formatter

        formatter = get_formatter(format)
        path = Path(path)

        with open(path, "w", encoding="utf-8") as f:
            count = 0
            for item in self.items:
                formatted = formatter.format(item.data, **kwargs)
                if formatted:
                    f.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                    count += 1

        return count

    def to_json(self, path: Union[str, Path]) -> int:
        """
        Export dataset to JSON file.

        Args:
            path: Output file path.

        Returns:
            Number of items written.
        """
        path = Path(path)
        data = {
            "metadata": self.metadata,
            "items": [item.to_dict() for item in self.items],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return len(self.items)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "Dataset":
        """
        Load dataset from JSON file.

        Args:
            path: Input file path.

        Returns:
            Dataset instance.
        """
        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = [DatasetItem.from_dict(item) for item in data.get("items", [])]
        return cls(items=items, metadata=data.get("metadata", {}))

    @classmethod
    def from_jsonl(cls, path: Union[str, Path], extractor_type: str = "") -> "Dataset":
        """
        Load dataset from JSONL file.

        Args:
            path: Input file path.
            extractor_type: Type to assign to items.

        Returns:
            Dataset instance.
        """
        path = Path(path)
        items = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    item = DatasetItem(
                        data=data,
                        source=str(path),
                        extractor_type=extractor_type,
                    )
                    items.append(item)

        return cls(items=items)

    def statistics(self) -> Dict[str, Any]:
        """
        Compute dataset statistics.

        Returns:
            Dictionary with statistics.
        """
        stats = {
            "total_items": len(self.items),
            "sources": {},
            "extractor_types": {},
        }

        for item in self.items:
            # Count by source
            source = item.source or "unknown"
            stats["sources"][source] = stats["sources"].get(source, 0) + 1

            # Count by extractor type
            ext_type = item.extractor_type or "unknown"
            stats["extractor_types"][ext_type] = stats["extractor_types"].get(ext_type, 0) + 1

        return stats

    def __repr__(self) -> str:
        """String representation."""
        return f"Dataset(items={len(self.items)})"

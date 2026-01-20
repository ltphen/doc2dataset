"""
Analytics and reporting for doc2dataset.

Provides insights into dataset quality, distribution,
and processing statistics.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class DatasetStats:
    """
    Statistics for a dataset.

    Attributes:
        total_items: Total number of items.
        total_tokens: Total token count.
        avg_tokens_per_item: Average tokens per item.
        field_stats: Per-field statistics.
        extraction_types: Distribution of extraction types.
        source_files: Distribution of source files.
    """

    total_items: int = 0
    total_tokens: int = 0
    avg_tokens_per_item: float = 0.0
    field_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    extraction_types: Dict[str, int] = field(default_factory=dict)
    source_files: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_items": self.total_items,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_item": self.avg_tokens_per_item,
            "field_stats": self.field_stats,
            "extraction_types": self.extraction_types,
            "source_files": self.source_files,
        }


class TokenCounter:
    """Simple token counter using tiktoken or fallback."""

    def __init__(self, model: str = "gpt-4") -> None:
        self.model = model
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            try:
                import tiktoken
                self._encoder = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                self._encoder = None
        return self._encoder

    def count(self, text: str) -> int:
        """Count tokens in text."""
        encoder = self._get_encoder()
        if encoder:
            return len(encoder.encode(text))
        return len(text.split())


class DatasetAnalyzer:
    """
    Analyzes training datasets for quality and distribution.

    Example:
        >>> analyzer = DatasetAnalyzer()
        >>> stats = analyzer.analyze(items)
        >>> print(analyzer.generate_report(stats))
    """

    def __init__(
        self,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        """
        Initialize analyzer.

        Args:
            token_counter: Token counter instance.
        """
        self.token_counter = token_counter or TokenCounter()

    def analyze(
        self,
        items: List[Dict[str, Any]],
        input_field: str = "input",
        output_field: str = "output",
    ) -> DatasetStats:
        """
        Analyze a dataset.

        Args:
            items: Dataset items.
            input_field: Field containing input/question.
            output_field: Field containing output/answer.

        Returns:
            DatasetStats with analysis results.
        """
        stats = DatasetStats(total_items=len(items))

        if not items:
            return stats

        # Collect field data
        input_lengths = []
        output_lengths = []
        input_tokens = []
        output_tokens = []
        extraction_types = Counter()
        source_files = Counter()

        for item in items:
            # Input field
            input_text = str(item.get(input_field, ""))
            input_lengths.append(len(input_text))
            input_tokens.append(self.token_counter.count(input_text))

            # Output field
            output_text = str(item.get(output_field, ""))
            output_lengths.append(len(output_text))
            output_tokens.append(self.token_counter.count(output_text))

            # Extraction type
            ext_type = item.get("extraction_type", item.get("type", "unknown"))
            extraction_types[ext_type] += 1

            # Source file
            source = item.get("source", item.get("_source", "unknown"))
            if isinstance(source, str):
                source_files[Path(source).name] += 1

        # Calculate statistics
        stats.total_tokens = sum(input_tokens) + sum(output_tokens)
        stats.avg_tokens_per_item = stats.total_tokens / len(items)

        stats.field_stats[input_field] = self._compute_field_stats(
            input_lengths, input_tokens, "input"
        )
        stats.field_stats[output_field] = self._compute_field_stats(
            output_lengths, output_tokens, "output"
        )

        stats.extraction_types = dict(extraction_types)
        stats.source_files = dict(source_files)

        return stats

    def _compute_field_stats(
        self,
        lengths: List[int],
        tokens: List[int],
        field_name: str,
    ) -> Dict[str, Any]:
        """Compute statistics for a field."""
        if not lengths:
            return {}

        arr_lengths = np.array(lengths)
        arr_tokens = np.array(tokens)

        return {
            "count": len(lengths),
            "char_length": {
                "mean": float(np.mean(arr_lengths)),
                "std": float(np.std(arr_lengths)),
                "min": int(np.min(arr_lengths)),
                "max": int(np.max(arr_lengths)),
                "median": float(np.median(arr_lengths)),
                "p25": float(np.percentile(arr_lengths, 25)),
                "p75": float(np.percentile(arr_lengths, 75)),
                "p95": float(np.percentile(arr_lengths, 95)),
            },
            "token_length": {
                "mean": float(np.mean(arr_tokens)),
                "std": float(np.std(arr_tokens)),
                "min": int(np.min(arr_tokens)),
                "max": int(np.max(arr_tokens)),
                "median": float(np.median(arr_tokens)),
                "total": int(np.sum(arr_tokens)),
            },
        }

    def analyze_text_quality(
        self,
        items: List[Dict[str, Any]],
        field: str = "output",
    ) -> Dict[str, Any]:
        """
        Analyze text quality metrics.

        Args:
            items: Dataset items.
            field: Field to analyze.

        Returns:
            Quality metrics.
        """
        if not items:
            return {}

        # Collect metrics
        word_counts = []
        sentence_counts = []
        unique_word_ratios = []
        avg_word_lengths = []

        for item in items:
            text = str(item.get(field, ""))

            # Word count
            words = text.split()
            word_counts.append(len(words))

            # Sentence count
            sentences = re.split(r'[.!?]+', text)
            sentence_counts.append(len([s for s in sentences if s.strip()]))

            # Unique word ratio (vocabulary diversity)
            if words:
                unique_ratio = len(set(w.lower() for w in words)) / len(words)
                unique_word_ratios.append(unique_ratio)

                # Average word length
                avg_word_lengths.append(
                    sum(len(w) for w in words) / len(words)
                )

        return {
            "word_count": {
                "mean": float(np.mean(word_counts)),
                "std": float(np.std(word_counts)),
                "min": int(np.min(word_counts)),
                "max": int(np.max(word_counts)),
            },
            "sentence_count": {
                "mean": float(np.mean(sentence_counts)),
                "std": float(np.std(sentence_counts)),
            },
            "vocabulary_diversity": {
                "mean": float(np.mean(unique_word_ratios)) if unique_word_ratios else 0,
                "std": float(np.std(unique_word_ratios)) if unique_word_ratios else 0,
            },
            "avg_word_length": {
                "mean": float(np.mean(avg_word_lengths)) if avg_word_lengths else 0,
            },
        }

    def analyze_duplicates(
        self,
        items: List[Dict[str, Any]],
        field: str = "output",
        similarity_threshold: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Analyze duplicates and near-duplicates.

        Args:
            items: Dataset items.
            field: Field to check.
            similarity_threshold: Threshold for near-duplicates.

        Returns:
            Duplicate analysis.
        """
        import hashlib

        if not items:
            return {"exact_duplicates": 0, "near_duplicates": 0}

        # Exact duplicates
        hashes = []
        for item in items:
            text = str(item.get(field, ""))
            normalized = " ".join(text.lower().split())
            h = hashlib.sha256(normalized.encode()).hexdigest()
            hashes.append(h)

        hash_counts = Counter(hashes)
        exact_duplicates = sum(c - 1 for c in hash_counts.values() if c > 1)

        # Near-duplicates (sample for efficiency)
        near_duplicates = 0
        sample_size = min(500, len(items))
        sample_indices = np.random.choice(len(items), sample_size, replace=False)

        texts = [str(items[i].get(field, "")) for i in sample_indices]
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = self._jaccard_similarity(texts[i], texts[j])
                if sim > similarity_threshold:
                    near_duplicates += 1

        # Extrapolate near-duplicates
        if sample_size < len(items):
            scale = (len(items) / sample_size) ** 2
            near_duplicates = int(near_duplicates * scale)

        return {
            "exact_duplicates": exact_duplicates,
            "exact_duplicate_rate": exact_duplicates / max(len(items), 1),
            "estimated_near_duplicates": near_duplicates,
            "unique_items": len(set(hashes)),
        }

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)

    def analyze_vocabulary(
        self,
        items: List[Dict[str, Any]],
        field: str = "output",
        top_n: int = 50,
    ) -> Dict[str, Any]:
        """
        Analyze vocabulary distribution.

        Args:
            items: Dataset items.
            field: Field to analyze.
            top_n: Number of top words to return.

        Returns:
            Vocabulary analysis.
        """
        word_counter = Counter()
        bigram_counter = Counter()

        for item in items:
            text = str(item.get(field, ""))
            words = re.findall(r'\b\w+\b', text.lower())

            word_counter.update(words)

            # Bigrams
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                bigram_counter[bigram] += 1

        return {
            "total_words": sum(word_counter.values()),
            "unique_words": len(word_counter),
            "top_words": dict(word_counter.most_common(top_n)),
            "top_bigrams": dict(bigram_counter.most_common(top_n)),
        }

    def generate_report(
        self,
        stats: DatasetStats,
        format: str = "text",
    ) -> str:
        """
        Generate a human-readable report.

        Args:
            stats: Dataset statistics.
            format: "text" or "markdown".

        Returns:
            Formatted report.
        """
        if format == "markdown":
            return self._generate_markdown_report(stats)
        return self._generate_text_report(stats)

    def _generate_text_report(self, stats: DatasetStats) -> str:
        """Generate plain text report."""
        lines = [
            "=" * 60,
            "DATASET ANALYSIS REPORT",
            "=" * 60,
            f"Generated: {datetime.now().isoformat()}",
            "",
            "OVERVIEW",
            "-" * 40,
            f"Total Items: {stats.total_items:,}",
            f"Total Tokens: {stats.total_tokens:,}",
            f"Avg Tokens/Item: {stats.avg_tokens_per_item:.1f}",
            "",
        ]

        # Field statistics
        for field_name, field_stats in stats.field_stats.items():
            if field_stats:
                lines.append(f"FIELD: {field_name.upper()}")
                lines.append("-" * 40)

                if "char_length" in field_stats:
                    cl = field_stats["char_length"]
                    lines.append(f"  Character Length:")
                    lines.append(f"    Mean: {cl['mean']:.1f}")
                    lines.append(f"    Median: {cl['median']:.1f}")
                    lines.append(f"    Range: {cl['min']} - {cl['max']}")

                if "token_length" in field_stats:
                    tl = field_stats["token_length"]
                    lines.append(f"  Token Length:")
                    lines.append(f"    Mean: {tl['mean']:.1f}")
                    lines.append(f"    Total: {tl['total']:,}")
                lines.append("")

        # Extraction types
        if stats.extraction_types:
            lines.append("EXTRACTION TYPES")
            lines.append("-" * 40)
            for ext_type, count in sorted(
                stats.extraction_types.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                pct = count / stats.total_items * 100
                lines.append(f"  {ext_type}: {count:,} ({pct:.1f}%)")
            lines.append("")

        # Source files
        if stats.source_files:
            lines.append("TOP SOURCE FILES")
            lines.append("-" * 40)
            for source, count in sorted(
                stats.source_files.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]:
                pct = count / stats.total_items * 100
                lines.append(f"  {source}: {count:,} ({pct:.1f}%)")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _generate_markdown_report(self, stats: DatasetStats) -> str:
        """Generate markdown report."""
        lines = [
            "# Dataset Analysis Report",
            "",
            f"*Generated: {datetime.now().isoformat()}*",
            "",
            "## Overview",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Items | {stats.total_items:,} |",
            f"| Total Tokens | {stats.total_tokens:,} |",
            f"| Avg Tokens/Item | {stats.avg_tokens_per_item:.1f} |",
            "",
        ]

        # Field statistics
        for field_name, field_stats in stats.field_stats.items():
            if field_stats:
                lines.append(f"## Field: {field_name}")
                lines.append("")

                if "char_length" in field_stats:
                    cl = field_stats["char_length"]
                    lines.append("### Character Length")
                    lines.append("")
                    lines.append("| Statistic | Value |")
                    lines.append("|-----------|-------|")
                    lines.append(f"| Mean | {cl['mean']:.1f} |")
                    lines.append(f"| Median | {cl['median']:.1f} |")
                    lines.append(f"| Min | {cl['min']} |")
                    lines.append(f"| Max | {cl['max']} |")
                    lines.append(f"| P95 | {cl['p95']:.1f} |")
                    lines.append("")

        # Extraction types
        if stats.extraction_types:
            lines.append("## Extraction Types")
            lines.append("")
            lines.append("| Type | Count | Percentage |")
            lines.append("|------|-------|------------|")
            for ext_type, count in sorted(
                stats.extraction_types.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                pct = count / stats.total_items * 100
                lines.append(f"| {ext_type} | {count:,} | {pct:.1f}% |")
            lines.append("")

        return "\n".join(lines)


class ProcessingAnalytics:
    """
    Tracks and reports on processing metrics.

    Example:
        >>> analytics = ProcessingAnalytics()
        >>> analytics.record_document_processed("doc.pdf", 100, 0.5)
        >>> print(analytics.summary())
    """

    def __init__(self) -> None:
        """Initialize analytics tracker."""
        self._documents_processed = 0
        self._items_extracted = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._processing_times: List[float] = []
        self._errors: List[Dict[str, Any]] = []
        self._per_document: List[Dict[str, Any]] = []
        self._start_time: Optional[float] = None

    def start_session(self) -> None:
        """Start a processing session."""
        import time
        self._start_time = time.time()

    def record_document_processed(
        self,
        source: str,
        items_extracted: int,
        processing_time: float,
        tokens_used: int = 0,
        cost: float = 0.0,
    ) -> None:
        """
        Record a processed document.

        Args:
            source: Document source path.
            items_extracted: Number of items extracted.
            processing_time: Processing time in seconds.
            tokens_used: Tokens consumed.
            cost: Cost incurred.
        """
        self._documents_processed += 1
        self._items_extracted += items_extracted
        self._total_tokens += tokens_used
        self._total_cost += cost
        self._processing_times.append(processing_time)

        self._per_document.append({
            "source": source,
            "items": items_extracted,
            "time_seconds": processing_time,
            "tokens": tokens_used,
            "cost": cost,
        })

    def record_error(
        self,
        source: str,
        error: str,
    ) -> None:
        """Record a processing error."""
        self._errors.append({
            "source": source,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        })

    def summary(self) -> Dict[str, Any]:
        """Get processing summary."""
        import time

        elapsed = 0.0
        if self._start_time:
            elapsed = time.time() - self._start_time

        processing_times = np.array(self._processing_times) if self._processing_times else np.array([0])

        return {
            "documents_processed": self._documents_processed,
            "items_extracted": self._items_extracted,
            "total_tokens": self._total_tokens,
            "total_cost": self._total_cost,
            "total_time_seconds": elapsed,
            "errors": len(self._errors),
            "success_rate": (
                self._documents_processed /
                max(self._documents_processed + len(self._errors), 1)
            ),
            "processing_time": {
                "mean": float(np.mean(processing_times)),
                "median": float(np.median(processing_times)),
                "min": float(np.min(processing_times)),
                "max": float(np.max(processing_times)),
                "total": float(np.sum(processing_times)),
            },
            "throughput": {
                "docs_per_minute": (
                    self._documents_processed / max(elapsed, 1) * 60
                ),
                "items_per_minute": (
                    self._items_extracted / max(elapsed, 1) * 60
                ),
            },
        }

    def get_errors(self) -> List[Dict[str, Any]]:
        """Get all recorded errors."""
        return self._errors

    def export(self, path: Union[str, Path]) -> None:
        """Export analytics to JSON file."""
        path = Path(path)

        data = {
            "summary": self.summary(),
            "per_document": self._per_document,
            "errors": self._errors,
            "exported_at": datetime.now().isoformat(),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def analyze_jsonl_file(
    path: Union[str, Path],
    input_field: str = "input",
    output_field: str = "output",
) -> DatasetStats:
    """
    Analyze a JSONL dataset file.

    Args:
        path: Path to JSONL file.
        input_field: Input field name.
        output_field: Output field name.

    Returns:
        DatasetStats with analysis.
    """
    path = Path(path)
    items = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    analyzer = DatasetAnalyzer()
    return analyzer.analyze(items, input_field, output_field)


def compare_datasets(
    datasets: Dict[str, List[Dict[str, Any]]],
    input_field: str = "input",
    output_field: str = "output",
) -> Dict[str, Any]:
    """
    Compare multiple datasets.

    Args:
        datasets: Dict mapping name to dataset items.
        input_field: Input field name.
        output_field: Output field name.

    Returns:
        Comparison results.
    """
    analyzer = DatasetAnalyzer()
    results = {}

    for name, items in datasets.items():
        stats = analyzer.analyze(items, input_field, output_field)
        results[name] = stats.to_dict()

    # Find differences
    if len(datasets) >= 2:
        names = list(datasets.keys())
        comparison = {
            "size_ratio": (
                results[names[0]]["total_items"] /
                max(results[names[1]]["total_items"], 1)
            ),
            "token_ratio": (
                results[names[0]]["total_tokens"] /
                max(results[names[1]]["total_tokens"], 1)
            ),
        }
        results["_comparison"] = comparison

    return results

"""
Quality filtering for doc2dataset.

Provides filters and scorers to ensure high-quality
training data output.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


@dataclass
class QualityScore:
    """
    Quality score for a training example.

    Attributes:
        overall: Overall quality score (0-1).
        dimensions: Individual dimension scores.
        passed: Whether it passes quality threshold.
        reasons: Reasons for low scores or failure.
    """

    overall: float
    dimensions: Dict[str, float] = field(default_factory=dict)
    passed: bool = True
    reasons: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"QualityScore({status}, overall={self.overall:.2f})"


class BaseFilter(ABC):
    """Abstract base class for quality filters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Filter name."""
        pass

    @abstractmethod
    def filter(self, item: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if item passes filter.

        Args:
            item: Training data item to check.

        Returns:
            Tuple of (passed, reason if failed).
        """
        pass


class LengthFilter(BaseFilter):
    """Filter based on text length."""

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 10000,
        field: str = "output",
    ) -> None:
        """
        Initialize length filter.

        Args:
            min_length: Minimum characters.
            max_length: Maximum characters.
            field: Field to check length of.
        """
        self.min_length = min_length
        self.max_length = max_length
        self.field = field

    @property
    def name(self) -> str:
        return "length"

    def filter(self, item: Dict[str, Any]) -> Tuple[bool, str]:
        """Check length constraints."""
        text = item.get(self.field, "")
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)

        length = len(str(text))

        if length < self.min_length:
            return False, f"Too short ({length} < {self.min_length})"
        if length > self.max_length:
            return False, f"Too long ({length} > {self.max_length})"
        return True, ""


class TokenLengthFilter(BaseFilter):
    """Filter based on token count."""

    def __init__(
        self,
        min_tokens: int = 5,
        max_tokens: int = 2000,
        field: str = "output",
    ) -> None:
        """
        Initialize token filter.

        Args:
            min_tokens: Minimum tokens.
            max_tokens: Maximum tokens.
            field: Field to check.
        """
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.field = field
        self._encoder = None

    def _get_encoder(self):
        """Get tiktoken encoder."""
        if self._encoder is None:
            try:
                import tiktoken
                self._encoder = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                self._encoder = None
        return self._encoder

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        encoder = self._get_encoder()
        if encoder:
            return len(encoder.encode(text))
        return len(text.split())

    @property
    def name(self) -> str:
        return "token_length"

    def filter(self, item: Dict[str, Any]) -> Tuple[bool, str]:
        """Check token length constraints."""
        text = item.get(self.field, "")
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)

        tokens = self._count_tokens(str(text))

        if tokens < self.min_tokens:
            return False, f"Too few tokens ({tokens} < {self.min_tokens})"
        if tokens > self.max_tokens:
            return False, f"Too many tokens ({tokens} > {self.max_tokens})"
        return True, ""


class LanguageFilter(BaseFilter):
    """Filter based on language detection."""

    def __init__(
        self,
        allowed_languages: Optional[Set[str]] = None,
        field: str = "output",
        min_confidence: float = 0.8,
    ) -> None:
        """
        Initialize language filter.

        Args:
            allowed_languages: Set of allowed language codes.
            field: Field to check.
            min_confidence: Minimum detection confidence.
        """
        self.allowed_languages = allowed_languages or {"en"}
        self.field = field
        self.min_confidence = min_confidence

    @property
    def name(self) -> str:
        return "language"

    def _detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language of text."""
        try:
            from langdetect import detect_langs

            results = detect_langs(text)
            if results:
                return results[0].lang, results[0].prob
        except Exception:
            pass

        # Fallback: assume English
        return "en", 0.5

    def filter(self, item: Dict[str, Any]) -> Tuple[bool, str]:
        """Check language."""
        text = item.get(self.field, "")
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)

        if len(text) < 20:
            return True, ""  # Too short to detect

        lang, conf = self._detect_language(str(text))

        if lang not in self.allowed_languages:
            return False, f"Language '{lang}' not allowed"
        if conf < self.min_confidence:
            return False, f"Low language confidence ({conf:.2f})"
        return True, ""


class RepetitionFilter(BaseFilter):
    """Filter to detect repetitive content."""

    def __init__(
        self,
        max_word_repetition: float = 0.3,
        max_ngram_repetition: float = 0.2,
        field: str = "output",
    ) -> None:
        """
        Initialize repetition filter.

        Args:
            max_word_repetition: Max ratio of repeated words.
            max_ngram_repetition: Max ratio of repeated n-grams.
            field: Field to check.
        """
        self.max_word_repetition = max_word_repetition
        self.max_ngram_repetition = max_ngram_repetition
        self.field = field

    @property
    def name(self) -> str:
        return "repetition"

    def _word_repetition_ratio(self, text: str) -> float:
        """Calculate word repetition ratio."""
        words = text.lower().split()
        if len(words) < 5:
            return 0.0

        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        max_count = max(word_counts.values())
        return max_count / len(words)

    def _ngram_repetition_ratio(self, text: str, n: int = 3) -> float:
        """Calculate n-gram repetition ratio."""
        words = text.lower().split()
        if len(words) < n + 2:
            return 0.0

        ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
        if not ngrams:
            return 0.0

        ngram_counts = {}
        for ngram in ngrams:
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        max_count = max(ngram_counts.values())
        return max_count / len(ngrams)

    def filter(self, item: Dict[str, Any]) -> Tuple[bool, str]:
        """Check for repetition."""
        text = item.get(self.field, "")
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)

        text = str(text)

        word_rep = self._word_repetition_ratio(text)
        if word_rep > self.max_word_repetition:
            return False, f"High word repetition ({word_rep:.2f})"

        ngram_rep = self._ngram_repetition_ratio(text)
        if ngram_rep > self.max_ngram_repetition:
            return False, f"High n-gram repetition ({ngram_rep:.2f})"

        return True, ""


class ContentFilter(BaseFilter):
    """Filter based on content patterns."""

    DEFAULT_BLOCKED_PATTERNS = [
        r"(?i)\b(api[_-]?key|password|secret|token)\s*[:=]\s*\S+",
        r"(?i)\b(todo|fixme|xxx)\b",
        r"(?i)(error|exception|traceback)",
        r"(?i)lorem ipsum",
        r"(?i)\[placeholder\]",
    ]

    def __init__(
        self,
        blocked_patterns: Optional[List[str]] = None,
        required_patterns: Optional[List[str]] = None,
        field: str = "output",
    ) -> None:
        """
        Initialize content filter.

        Args:
            blocked_patterns: Regex patterns that should not appear.
            required_patterns: Regex patterns that must appear.
            field: Field to check.
        """
        self.blocked_patterns = blocked_patterns or self.DEFAULT_BLOCKED_PATTERNS
        self.required_patterns = required_patterns or []
        self.field = field

        # Compile patterns
        self._blocked_compiled = [
            re.compile(p) for p in self.blocked_patterns
        ]
        self._required_compiled = [
            re.compile(p) for p in self.required_patterns
        ]

    @property
    def name(self) -> str:
        return "content"

    def filter(self, item: Dict[str, Any]) -> Tuple[bool, str]:
        """Check content patterns."""
        text = item.get(self.field, "")
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)

        text = str(text)

        # Check blocked patterns
        for pattern in self._blocked_compiled:
            if pattern.search(text):
                return False, f"Contains blocked pattern: {pattern.pattern}"

        # Check required patterns
        for pattern in self._required_compiled:
            if not pattern.search(text):
                return False, f"Missing required pattern: {pattern.pattern}"

        return True, ""


class DuplicateFilter(BaseFilter):
    """Filter to detect duplicates."""

    def __init__(
        self,
        similarity_threshold: float = 0.9,
        field: str = "output",
    ) -> None:
        """
        Initialize duplicate filter.

        Args:
            similarity_threshold: Threshold for duplicate detection.
            field: Field to check.
        """
        self.similarity_threshold = similarity_threshold
        self.field = field
        self._seen_hashes: Set[str] = set()
        self._seen_texts: List[str] = []

    @property
    def name(self) -> str:
        return "duplicate"

    def _get_hash(self, text: str) -> str:
        """Get hash of normalized text."""
        import hashlib

        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union

    def filter(self, item: Dict[str, Any]) -> Tuple[bool, str]:
        """Check for duplicates."""
        text = item.get(self.field, "")
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)

        text = str(text)

        # Exact duplicate check
        text_hash = self._get_hash(text)
        if text_hash in self._seen_hashes:
            return False, "Exact duplicate"

        # Near-duplicate check (sample for efficiency)
        for seen in self._seen_texts[-100:]:  # Check last 100
            similarity = self._jaccard_similarity(text, seen)
            if similarity > self.similarity_threshold:
                return False, f"Near-duplicate (similarity: {similarity:.2f})"

        # Add to seen
        self._seen_hashes.add(text_hash)
        self._seen_texts.append(text)

        return True, ""

    def reset(self) -> None:
        """Reset seen items."""
        self._seen_hashes.clear()
        self._seen_texts.clear()


class QASpecificFilter(BaseFilter):
    """Filter specific to Q&A pairs."""

    def __init__(
        self,
        min_question_length: int = 10,
        min_answer_length: int = 20,
        question_field: str = "input",
        answer_field: str = "output",
    ) -> None:
        """
        Initialize Q&A filter.

        Args:
            min_question_length: Minimum question length.
            min_answer_length: Minimum answer length.
            question_field: Field containing question.
            answer_field: Field containing answer.
        """
        self.min_question_length = min_question_length
        self.min_answer_length = min_answer_length
        self.question_field = question_field
        self.answer_field = answer_field

    @property
    def name(self) -> str:
        return "qa_specific"

    def filter(self, item: Dict[str, Any]) -> Tuple[bool, str]:
        """Check Q&A specific rules."""
        question = str(item.get(self.question_field, ""))
        answer = str(item.get(self.answer_field, ""))

        # Length checks
        if len(question) < self.min_question_length:
            return False, "Question too short"
        if len(answer) < self.min_answer_length:
            return False, "Answer too short"

        # Question should end with ? or be interrogative
        question_lower = question.lower().strip()
        interrogatives = ["what", "why", "how", "when", "where", "who", "which", "can", "could", "would", "should", "is", "are", "do", "does", "did"]
        has_question_marker = (
            question.strip().endswith("?") or
            any(question_lower.startswith(w) for w in interrogatives)
        )
        if not has_question_marker:
            return False, "Question doesn't appear to be interrogative"

        # Answer should not just repeat the question
        if question_lower in answer.lower():
            q_words = set(question_lower.split())
            a_words = set(answer.lower().split())
            overlap = len(q_words & a_words) / max(len(a_words), 1)
            if overlap > 0.7:
                return False, "Answer repeats too much of question"

        return True, ""


class QualityScorer:
    """
    Scores quality of training examples.

    Example:
        >>> scorer = QualityScorer()
        >>> score = scorer.score({"input": "Q", "output": "A"})
        >>> if score.passed:
        ...     print("High quality!")
    """

    def __init__(
        self,
        filters: Optional[List[BaseFilter]] = None,
        threshold: float = 0.6,
    ) -> None:
        """
        Initialize quality scorer.

        Args:
            filters: List of filters to apply.
            threshold: Minimum overall score to pass.
        """
        self.filters = filters or self._default_filters()
        self.threshold = threshold

    def _default_filters(self) -> List[BaseFilter]:
        """Get default filters."""
        return [
            LengthFilter(min_length=20, max_length=5000),
            RepetitionFilter(),
            ContentFilter(),
        ]

    def score(self, item: Dict[str, Any]) -> QualityScore:
        """
        Score a training example.

        Args:
            item: Training data item.

        Returns:
            QualityScore with details.
        """
        dimensions = {}
        reasons = []
        passed_count = 0

        for filter_obj in self.filters:
            passed, reason = filter_obj.filter(item)
            dimensions[filter_obj.name] = 1.0 if passed else 0.0

            if passed:
                passed_count += 1
            elif reason:
                reasons.append(f"{filter_obj.name}: {reason}")

        overall = passed_count / max(len(self.filters), 1)
        passed = overall >= self.threshold and not reasons

        return QualityScore(
            overall=overall,
            dimensions=dimensions,
            passed=passed,
            reasons=reasons,
        )

    def filter_batch(
        self,
        items: List[Dict[str, Any]],
        return_failed: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], QualityScore]]]:
        """
        Filter a batch of items.

        Args:
            items: Items to filter.
            return_failed: Include failed items in return.

        Returns:
            Tuple of (passed_items, failed_items_with_scores).
        """
        passed = []
        failed = []

        for item in items:
            score = self.score(item)
            if score.passed:
                passed.append(item)
            else:
                failed.append((item, score))

        return passed, failed if return_failed else []


class QualityFilterPipeline:
    """
    Pipeline for quality filtering with stats.

    Example:
        >>> pipeline = QualityFilterPipeline()
        >>> results = pipeline.process(items)
        >>> print(pipeline.stats())
    """

    def __init__(
        self,
        filters: Optional[List[BaseFilter]] = None,
    ) -> None:
        """
        Initialize pipeline.

        Args:
            filters: Filters to apply in sequence.
        """
        self.filters = filters or [
            LengthFilter(),
            RepetitionFilter(),
            ContentFilter(),
            DuplicateFilter(),
        ]
        self._stats: Dict[str, int] = {}
        self._reset_stats()

    def _reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = {
            "total_input": 0,
            "total_passed": 0,
            "total_failed": 0,
        }
        for f in self.filters:
            self._stats[f"failed_{f.name}"] = 0

    def process(
        self,
        items: List[Dict[str, Any]],
        stop_on_first_fail: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Process items through filter pipeline.

        Args:
            items: Items to filter.
            stop_on_first_fail: Stop checking on first filter fail.

        Returns:
            List of items that passed all filters.
        """
        self._reset_stats()
        passed_items = []

        for item in items:
            self._stats["total_input"] += 1
            item_passed = True

            for filter_obj in self.filters:
                passed, reason = filter_obj.filter(item)
                if not passed:
                    item_passed = False
                    self._stats[f"failed_{filter_obj.name}"] += 1
                    if stop_on_first_fail:
                        break

            if item_passed:
                self._stats["total_passed"] += 1
                passed_items.append(item)
            else:
                self._stats["total_failed"] += 1

        return passed_items

    def stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        total = max(self._stats["total_input"], 1)
        return {
            **self._stats,
            "pass_rate": self._stats["total_passed"] / total,
            "filter_breakdown": {
                f.name: self._stats.get(f"failed_{f.name}", 0) / total
                for f in self.filters
            },
        }


def get_qa_quality_pipeline() -> QualityFilterPipeline:
    """Get pre-configured pipeline for Q&A data."""
    return QualityFilterPipeline(filters=[
        LengthFilter(min_length=20, max_length=5000, field="output"),
        QASpecificFilter(),
        RepetitionFilter(field="output"),
        ContentFilter(field="output"),
        DuplicateFilter(field="output"),
    ])


def get_instruction_quality_pipeline() -> QualityFilterPipeline:
    """Get pre-configured pipeline for instruction data."""
    return QualityFilterPipeline(filters=[
        LengthFilter(min_length=50, max_length=8000, field="output"),
        TokenLengthFilter(min_tokens=20, max_tokens=2000, field="output"),
        RepetitionFilter(field="output"),
        ContentFilter(field="output"),
        DuplicateFilter(field="output"),
    ])

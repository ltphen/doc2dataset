"""
Source attribution for doc2dataset.

Tracks and preserves source information for extracted training data,
enabling traceability back to original documents.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class SourceLocation:
    """
    Location within a source document.

    Attributes:
        file_path: Path to the source file.
        page_number: Page number (for PDFs).
        line_start: Starting line number.
        line_end: Ending line number.
        char_start: Starting character offset.
        char_end: Ending character offset.
        section: Section name or heading.
        chunk_index: Index of chunk within document.
    """

    file_path: str
    page_number: Optional[int] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    section: Optional[str] = None
    chunk_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            k: v for k, v in {
                "file_path": self.file_path,
                "page_number": self.page_number,
                "line_start": self.line_start,
                "line_end": self.line_end,
                "char_start": self.char_start,
                "char_end": self.char_end,
                "section": self.section,
                "chunk_index": self.chunk_index,
            }.items() if v is not None
        }

    def to_citation(self) -> str:
        """Format as human-readable citation."""
        parts = [Path(self.file_path).name]

        if self.page_number is not None:
            parts.append(f"p. {self.page_number}")
        if self.section:
            parts.append(f'"{self.section}"')
        if self.line_start is not None:
            if self.line_end and self.line_end != self.line_start:
                parts.append(f"lines {self.line_start}-{self.line_end}")
            else:
                parts.append(f"line {self.line_start}")

        return ", ".join(parts)


@dataclass
class Attribution:
    """
    Full attribution for an extracted item.

    Attributes:
        item_id: Unique identifier for the extracted item.
        sources: List of source locations.
        extraction_method: How the item was extracted.
        confidence: Confidence score (0-1).
        timestamp: When extraction occurred.
        metadata: Additional metadata.
    """

    item_id: str
    sources: List[SourceLocation] = field(default_factory=list)
    extraction_method: str = ""
    confidence: float = 1.0
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "item_id": self.item_id,
            "sources": [s.to_dict() for s in self.sources],
            "extraction_method": self.extraction_method,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Attribution":
        """Create from dictionary."""
        return cls(
            item_id=data["item_id"],
            sources=[
                SourceLocation(**s) for s in data.get("sources", [])
            ],
            extraction_method=data.get("extraction_method", ""),
            confidence=data.get("confidence", 1.0),
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata", {}),
        )


class SourceTracker:
    """
    Tracks source locations during document processing.

    Example:
        >>> tracker = SourceTracker("document.pdf")
        >>> tracker.set_page(1)
        >>> loc = tracker.mark_span(100, 200, section="Introduction")
    """

    def __init__(
        self,
        file_path: str,
        content: Optional[str] = None,
    ) -> None:
        """
        Initialize source tracker.

        Args:
            file_path: Path to source file.
            content: Full document content for offset tracking.
        """
        self.file_path = file_path
        self.content = content
        self.current_page: Optional[int] = None
        self.current_section: Optional[str] = None
        self._line_offsets: Optional[List[int]] = None

        if content:
            self._build_line_index()

    def _build_line_index(self) -> None:
        """Build index of line start offsets."""
        if not self.content:
            return

        self._line_offsets = [0]
        for i, char in enumerate(self.content):
            if char == "\n":
                self._line_offsets.append(i + 1)

    def _char_to_line(self, char_offset: int) -> int:
        """Convert character offset to line number."""
        if not self._line_offsets:
            return 1

        # Binary search for line
        low, high = 0, len(self._line_offsets) - 1
        while low <= high:
            mid = (low + high) // 2
            if self._line_offsets[mid] <= char_offset:
                if mid + 1 >= len(self._line_offsets) or self._line_offsets[mid + 1] > char_offset:
                    return mid + 1  # 1-indexed
                low = mid + 1
            else:
                high = mid - 1

        return 1

    def set_page(self, page_number: int) -> None:
        """Set current page number."""
        self.current_page = page_number

    def set_section(self, section: str) -> None:
        """Set current section."""
        self.current_section = section

    def mark_span(
        self,
        char_start: int,
        char_end: int,
        section: Optional[str] = None,
        page: Optional[int] = None,
        chunk_index: Optional[int] = None,
    ) -> SourceLocation:
        """
        Mark a span of text.

        Args:
            char_start: Starting character offset.
            char_end: Ending character offset.
            section: Override section.
            page: Override page.
            chunk_index: Chunk index.

        Returns:
            SourceLocation for the span.
        """
        line_start = self._char_to_line(char_start) if self._line_offsets else None
        line_end = self._char_to_line(char_end) if self._line_offsets else None

        return SourceLocation(
            file_path=self.file_path,
            page_number=page or self.current_page,
            line_start=line_start,
            line_end=line_end,
            char_start=char_start,
            char_end=char_end,
            section=section or self.current_section,
            chunk_index=chunk_index,
        )

    def find_text(
        self,
        text: str,
        start_from: int = 0,
    ) -> Optional[SourceLocation]:
        """
        Find text in document and return its location.

        Args:
            text: Text to find.
            start_from: Start searching from this offset.

        Returns:
            SourceLocation if found, None otherwise.
        """
        if not self.content:
            return None

        # Normalize for matching
        normalized_text = " ".join(text.split())
        normalized_content = " ".join(self.content.split())

        # Find in normalized content
        idx = normalized_content.find(normalized_text, start_from)
        if idx == -1:
            # Try fuzzy match
            idx = self._fuzzy_find(normalized_text, normalized_content, start_from)

        if idx == -1:
            return None

        # Map back to original offsets (approximate)
        char_start = idx
        char_end = idx + len(text)

        return self.mark_span(char_start, char_end)

    def _fuzzy_find(
        self,
        needle: str,
        haystack: str,
        start_from: int = 0,
    ) -> int:
        """Fuzzy find with some tolerance."""
        # Try finding first few words
        words = needle.split()[:5]
        if not words:
            return -1

        partial = " ".join(words)
        idx = haystack.find(partial, start_from)
        return idx


class AttributionManager:
    """
    Manages attribution data for a processing job.

    Example:
        >>> manager = AttributionManager()
        >>> attr = manager.create_attribution(
        ...     content="Q&A pair",
        ...     source_file="doc.pdf",
        ...     page=5,
        ...     extraction_method="qa_extraction"
        ... )
        >>> manager.save("attributions.json")
    """

    def __init__(self) -> None:
        """Initialize attribution manager."""
        self._attributions: Dict[str, Attribution] = {}
        self._source_trackers: Dict[str, SourceTracker] = {}

    def get_tracker(
        self,
        file_path: str,
        content: Optional[str] = None,
    ) -> SourceTracker:
        """
        Get or create tracker for a file.

        Args:
            file_path: Path to file.
            content: File content.

        Returns:
            SourceTracker for the file.
        """
        if file_path not in self._source_trackers:
            self._source_trackers[file_path] = SourceTracker(file_path, content)
        return self._source_trackers[file_path]

    def _generate_id(self, content: str) -> str:
        """Generate unique ID for content."""
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def create_attribution(
        self,
        content: str,
        source_file: str,
        page: Optional[int] = None,
        section: Optional[str] = None,
        char_start: Optional[int] = None,
        char_end: Optional[int] = None,
        line_start: Optional[int] = None,
        line_end: Optional[int] = None,
        extraction_method: str = "",
        confidence: float = 1.0,
        chunk_index: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Attribution:
        """
        Create attribution for extracted content.

        Args:
            content: The extracted content.
            source_file: Source file path.
            page: Page number.
            section: Section name.
            char_start: Character offset start.
            char_end: Character offset end.
            line_start: Line number start.
            line_end: Line number end.
            extraction_method: Extraction method used.
            confidence: Confidence score.
            chunk_index: Chunk index.
            metadata: Additional metadata.

        Returns:
            Attribution object.
        """
        item_id = self._generate_id(content)

        location = SourceLocation(
            file_path=source_file,
            page_number=page,
            line_start=line_start,
            line_end=line_end,
            char_start=char_start,
            char_end=char_end,
            section=section,
            chunk_index=chunk_index,
        )

        import time
        attribution = Attribution(
            item_id=item_id,
            sources=[location],
            extraction_method=extraction_method,
            confidence=confidence,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        self._attributions[item_id] = attribution
        return attribution

    def add_source(
        self,
        item_id: str,
        location: SourceLocation,
    ) -> None:
        """
        Add additional source to an attribution.

        Args:
            item_id: Attribution item ID.
            location: Additional source location.
        """
        if item_id in self._attributions:
            self._attributions[item_id].sources.append(location)

    def get_attribution(self, item_id: str) -> Optional[Attribution]:
        """Get attribution by ID."""
        return self._attributions.get(item_id)

    def get_all(self) -> List[Attribution]:
        """Get all attributions."""
        return list(self._attributions.values())

    def to_dict(self) -> Dict[str, Any]:
        """Export all attributions as dict."""
        return {
            "attributions": [a.to_dict() for a in self._attributions.values()],
            "total_count": len(self._attributions),
            "source_files": list(set(
                s.file_path
                for a in self._attributions.values()
                for s in a.sources
            )),
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save attributions to JSON file."""
        import json

        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self, path: Union[str, Path]) -> None:
        """Load attributions from JSON file."""
        import json

        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for attr_data in data.get("attributions", []):
            attr = Attribution.from_dict(attr_data)
            self._attributions[attr.item_id] = attr


def extract_section_headers(content: str) -> List[Tuple[str, int]]:
    """
    Extract section headers from markdown/text content.

    Args:
        content: Document content.

    Returns:
        List of (header_text, char_offset) tuples.
    """
    headers = []

    # Markdown headers
    md_pattern = r"^(#{1,6})\s+(.+)$"
    for match in re.finditer(md_pattern, content, re.MULTILINE):
        headers.append((match.group(2).strip(), match.start()))

    # Underlined headers
    underline_pattern = r"^(.+)\n[=\-]{2,}$"
    for match in re.finditer(underline_pattern, content, re.MULTILINE):
        headers.append((match.group(1).strip(), match.start()))

    # Numbered sections (e.g., "1.2.3 Section Name")
    numbered_pattern = r"^(\d+(?:\.\d+)*\.?\s+[A-Z].+)$"
    for match in re.finditer(numbered_pattern, content, re.MULTILINE):
        headers.append((match.group(1).strip(), match.start()))

    return sorted(headers, key=lambda x: x[1])


def get_section_at_offset(
    headers: List[Tuple[str, int]],
    offset: int,
) -> Optional[str]:
    """
    Get section name for a character offset.

    Args:
        headers: List of (header_text, offset) tuples.
        offset: Character offset to look up.

    Returns:
        Section name or None.
    """
    current_section = None

    for header, header_offset in headers:
        if header_offset <= offset:
            current_section = header
        else:
            break

    return current_section


class AttributedDataset:
    """
    Dataset wrapper with attribution support.

    Combines training data items with their attributions.

    Example:
        >>> dataset = AttributedDataset()
        >>> dataset.add(
        ...     item={"input": "Q", "output": "A"},
        ...     attribution=attr,
        ... )
        >>> dataset.save("dataset_with_attribution.jsonl")
    """

    def __init__(self) -> None:
        """Initialize attributed dataset."""
        self._items: List[Dict[str, Any]] = []
        self._attributions: List[Attribution] = []

    def add(
        self,
        item: Dict[str, Any],
        attribution: Attribution,
    ) -> None:
        """
        Add item with attribution.

        Args:
            item: Training data item.
            attribution: Attribution for the item.
        """
        # Add attribution ID to item
        item_with_attr = {**item, "_attribution_id": attribution.item_id}
        self._items.append(item_with_attr)
        self._attributions.append(attribution)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(zip(self._items, self._attributions))

    def get_items(self) -> List[Dict[str, Any]]:
        """Get items without attributions (for training)."""
        return [
            {k: v for k, v in item.items() if not k.startswith("_")}
            for item in self._items
        ]

    def save(
        self,
        path: Union[str, Path],
        include_attribution: bool = True,
    ) -> None:
        """
        Save dataset.

        Args:
            path: Output path.
            include_attribution: Include inline attribution.
        """
        import json

        path = Path(path)

        with open(path, "w", encoding="utf-8") as f:
            for item, attr in zip(self._items, self._attributions):
                if include_attribution:
                    item["_attribution"] = attr.to_dict()
                f.write(json.dumps(item) + "\n")

        # Save separate attribution file
        if include_attribution:
            attr_path = path.with_suffix(".attributions.json")
            manager = AttributionManager()
            for attr in self._attributions:
                manager._attributions[attr.item_id] = attr
            manager.save(attr_path)

    def get_sources_for_item(self, index: int) -> List[str]:
        """Get source citations for an item."""
        if 0 <= index < len(self._attributions):
            return [
                source.to_citation()
                for source in self._attributions[index].sources
            ]
        return []

    def filter_by_source(self, file_path: str) -> "AttributedDataset":
        """Filter to items from a specific source."""
        filtered = AttributedDataset()

        for item, attr in zip(self._items, self._attributions):
            if any(s.file_path == file_path for s in attr.sources):
                filtered._items.append(item)
                filtered._attributions.append(attr)

        return filtered

    def stats(self) -> Dict[str, Any]:
        """Get dataset statistics with attribution info."""
        source_files = set()
        sources_per_item = []

        for attr in self._attributions:
            sources_per_item.append(len(attr.sources))
            for source in attr.sources:
                source_files.add(source.file_path)

        return {
            "total_items": len(self._items),
            "unique_sources": len(source_files),
            "source_files": list(source_files),
            "avg_sources_per_item": sum(sources_per_item) / max(len(sources_per_item), 1),
            "extraction_methods": list(set(a.extraction_method for a in self._attributions)),
        }

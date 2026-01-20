"""
Knowledge extractors for doc2dataset.

This module provides LLM-powered extractors that transform document
content into structured training data.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

from doc2dataset.loaders import Document


@dataclass
class ExtractionResult:
    """
    Result from an extraction operation.

    Attributes:
        items: List of extracted items.
        source: Source document reference.
        extractor_type: Type of extractor used.
        metadata: Additional extraction metadata.
    """

    items: List[Dict[str, Any]]
    source: str = ""
    extractor_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return number of extracted items."""
        return len(self.items)

    def __iter__(self):
        """Iterate over items."""
        return iter(self.items)


class BaseExtractor(ABC):
    """
    Abstract base class for knowledge extractors.

    Extractors use LLMs to transform document content into
    structured training data formats.
    """

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        chunk_size: int = 3000,
        max_items_per_chunk: int = 10,
    ) -> None:
        """
        Initialize the extractor.

        Args:
            llm_fn: Function that takes a prompt and returns LLM response.
            chunk_size: Maximum characters per chunk for processing.
            max_items_per_chunk: Maximum items to extract per chunk.
        """
        self.llm_fn = llm_fn
        self.chunk_size = chunk_size
        self.max_items_per_chunk = max_items_per_chunk

    @property
    @abstractmethod
    def extractor_type(self) -> str:
        """Return the type name of this extractor."""
        pass

    @abstractmethod
    def get_extraction_prompt(self, content: str) -> str:
        """
        Generate the extraction prompt for the content.

        Args:
            content: Document content to extract from.

        Returns:
            Prompt string for the LLM.
        """
        pass

    @abstractmethod
    def parse_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM response into structured items.

        Args:
            response: Raw LLM response text.

        Returns:
            List of extracted items as dictionaries.
        """
        pass

    def extract(self, document: Document) -> ExtractionResult:
        """
        Extract knowledge from a document.

        Args:
            document: Document to extract from.

        Returns:
            ExtractionResult containing all extracted items.
        """
        content = document.content
        all_items = []

        # Process in chunks if content is too long
        chunks = self._chunk_content(content)

        for i, chunk in enumerate(chunks):
            prompt = self.get_extraction_prompt(chunk)
            response = self.llm_fn(prompt)
            items = self.parse_response(response)

            # Add chunk metadata
            for item in items:
                item["_chunk_index"] = i
                item["_source"] = document.source

            all_items.extend(items)

        return ExtractionResult(
            items=all_items,
            source=document.source,
            extractor_type=self.extractor_type,
            metadata={
                "chunk_count": len(chunks),
                "total_items": len(all_items),
            },
        )

    def _chunk_content(self, content: str) -> List[str]:
        """Split content into processable chunks."""
        if len(content) <= self.chunk_size:
            return [content]

        chunks = []
        start = 0

        while start < len(content):
            end = start + self.chunk_size

            if end < len(content):
                # Find a good break point
                for sep in ["\n\n", "\n", ". ", " "]:
                    break_point = content[start:end].rfind(sep)
                    if break_point > self.chunk_size // 2:
                        end = start + break_point + len(sep)
                        break

            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end

        return chunks

    def _parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """Helper to parse JSON from LLM response."""
        # Try to extract JSON from response
        # Look for JSON array
        json_match = re.search(r"\[[\s\S]*\]", response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try parsing individual JSON objects
        items = []
        for match in re.finditer(r"\{[^{}]+\}", response):
            try:
                item = json.loads(match.group())
                items.append(item)
            except json.JSONDecodeError:
                continue

        return items


class QAExtractor(BaseExtractor):
    """
    Extract question-answer pairs from documents.

    Generates training data in Q&A format suitable for
    instruction fine-tuning.

    Example output:
        {"question": "What is X?", "answer": "X is..."}
    """

    @property
    def extractor_type(self) -> str:
        return "qa"

    def get_extraction_prompt(self, content: str) -> str:
        return f"""Analyze the following document content and extract question-answer pairs that capture the key knowledge.

Generate {self.max_items_per_chunk} high-quality Q&A pairs that:
1. Cover the most important information in the text
2. Have clear, specific questions
3. Have accurate, comprehensive answers based on the content
4. Would be useful for training an AI assistant about this domain

Output format: Return a JSON array of objects with "question" and "answer" fields.

Document content:
---
{content}
---

Extract Q&A pairs as JSON array:"""

    def parse_response(self, response: str) -> List[Dict[str, Any]]:
        items = self._parse_json_response(response)

        # Validate and normalize
        validated = []
        for item in items:
            if "question" in item and "answer" in item:
                validated.append({
                    "question": str(item["question"]).strip(),
                    "answer": str(item["answer"]).strip(),
                })

        return validated


class RulesExtractor(BaseExtractor):
    """
    Extract rules, guidelines, and best practices from documents.

    Particularly useful for compliance documents, manuals, and
    policy documents.

    Example output:
        {"rule": "Always do X", "context": "When Y happens", "category": "safety"}
    """

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        categories: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the rules extractor.

        Args:
            llm_fn: LLM function for extraction.
            categories: Optional list of rule categories to look for.
            **kwargs: Additional arguments for base class.
        """
        super().__init__(llm_fn, **kwargs)
        self.categories = categories or [
            "general",
            "procedure",
            "safety",
            "compliance",
            "best_practice",
        ]

    @property
    def extractor_type(self) -> str:
        return "rules"

    def get_extraction_prompt(self, content: str) -> str:
        categories_str = ", ".join(self.categories)
        return f"""Analyze the following document and extract all rules, guidelines, procedures, and best practices.

For each rule, identify:
1. The rule or guideline itself (clear, actionable statement)
2. The context or condition when it applies
3. The category: {categories_str}
4. The rationale or reason for the rule (if stated)

Output format: Return a JSON array of objects with "rule", "context", "category", and "rationale" fields.

Extract up to {self.max_items_per_chunk} rules.

Document content:
---
{content}
---

Extract rules as JSON array:"""

    def parse_response(self, response: str) -> List[Dict[str, Any]]:
        items = self._parse_json_response(response)

        validated = []
        for item in items:
            if "rule" in item:
                validated.append({
                    "rule": str(item["rule"]).strip(),
                    "context": str(item.get("context", "")).strip(),
                    "category": str(item.get("category", "general")).strip().lower(),
                    "rationale": str(item.get("rationale", "")).strip(),
                })

        return validated


class FactsExtractor(BaseExtractor):
    """
    Extract factual statements and knowledge from documents.

    Generates atomic facts that can be used for knowledge base
    construction or fact-based training.

    Example output:
        {"fact": "X was founded in 2020", "topic": "history", "confidence": "high"}
    """

    @property
    def extractor_type(self) -> str:
        return "facts"

    def get_extraction_prompt(self, content: str) -> str:
        return f"""Analyze the following document and extract factual statements.

For each fact:
1. State the fact clearly and concisely
2. Identify the topic or category
3. Assess confidence level (high/medium/low) based on how explicit the information is

Extract atomic facts - each fact should express a single piece of information.
Extract up to {self.max_items_per_chunk} facts.

Output format: Return a JSON array of objects with "fact", "topic", and "confidence" fields.

Document content:
---
{content}
---

Extract facts as JSON array:"""

    def parse_response(self, response: str) -> List[Dict[str, Any]]:
        items = self._parse_json_response(response)

        validated = []
        for item in items:
            if "fact" in item:
                validated.append({
                    "fact": str(item["fact"]).strip(),
                    "topic": str(item.get("topic", "general")).strip(),
                    "confidence": str(item.get("confidence", "medium")).strip().lower(),
                })

        return validated


class InstructionExtractor(BaseExtractor):
    """
    Extract instruction-response pairs from documents.

    Generates data in instruction-following format commonly used
    for training instruction-tuned models.

    Example output:
        {"instruction": "Explain how to...", "input": "", "output": "To do this..."}
    """

    @property
    def extractor_type(self) -> str:
        return "instruction"

    def get_extraction_prompt(self, content: str) -> str:
        return f"""Analyze the following document and generate instruction-response training pairs.

Create diverse instructions that a user might ask, along with appropriate responses based on the document content.

Types of instructions to generate:
1. Explanation requests ("Explain...", "What is...", "Describe...")
2. How-to instructions ("How do I...", "What are the steps to...")
3. Comparison requests ("Compare...", "What's the difference between...")
4. Analysis requests ("Why does...", "What causes...")
5. Summary requests ("Summarize...", "What are the key points...")

Generate {self.max_items_per_chunk} instruction-response pairs.

Output format: Return a JSON array with "instruction", "input" (optional context), and "output" fields.

Document content:
---
{content}
---

Generate instruction-response pairs as JSON array:"""

    def parse_response(self, response: str) -> List[Dict[str, Any]]:
        items = self._parse_json_response(response)

        validated = []
        for item in items:
            if "instruction" in item and "output" in item:
                validated.append({
                    "instruction": str(item["instruction"]).strip(),
                    "input": str(item.get("input", "")).strip(),
                    "output": str(item["output"]).strip(),
                })

        return validated


class ConversationExtractor(BaseExtractor):
    """
    Extract multi-turn conversation examples from documents.

    Generates conversational training data with context.

    Example output:
        {"conversation": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        turns_per_conversation: int = 3,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the conversation extractor.

        Args:
            llm_fn: LLM function for extraction.
            turns_per_conversation: Number of turns per conversation.
            **kwargs: Additional arguments.
        """
        super().__init__(llm_fn, **kwargs)
        self.turns_per_conversation = turns_per_conversation

    @property
    def extractor_type(self) -> str:
        return "conversation"

    def get_extraction_prompt(self, content: str) -> str:
        return f"""Based on the following document, generate realistic multi-turn conversations.

Each conversation should:
1. Have {self.turns_per_conversation} turns (alternating user/assistant)
2. Be about topics covered in the document
3. Show natural conversation flow with follow-up questions
4. Have accurate, helpful assistant responses

Generate {self.max_items_per_chunk // 2} conversations.

Output format: Return a JSON array where each item has a "conversation" field containing an array of {{role, content}} objects.

Document content:
---
{content}
---

Generate conversations as JSON array:"""

    def parse_response(self, response: str) -> List[Dict[str, Any]]:
        items = self._parse_json_response(response)

        validated = []
        for item in items:
            if "conversation" in item and isinstance(item["conversation"], list):
                # Validate conversation structure
                conv = item["conversation"]
                valid = True
                for turn in conv:
                    if not isinstance(turn, dict):
                        valid = False
                        break
                    if "role" not in turn or "content" not in turn:
                        valid = False
                        break

                if valid and len(conv) >= 2:
                    validated.append({"conversation": conv})

        return validated


class SummaryExtractor(BaseExtractor):
    """
    Extract summaries at various levels from documents.

    Generates summary training data with different granularities.
    """

    @property
    def extractor_type(self) -> str:
        return "summary"

    def get_extraction_prompt(self, content: str) -> str:
        return f"""Analyze the following document and create training examples for summarization.

Generate different types of summaries:
1. One-sentence summary (TL;DR)
2. Bullet point summary (key points)
3. Paragraph summary (comprehensive)
4. Section summaries (if applicable)

For each summary, include the original text span and the summary.

Output format: Return a JSON array with "summary_type", "original_text", and "summary" fields.

Document content:
---
{content}
---

Generate summary examples as JSON array:"""

    def parse_response(self, response: str) -> List[Dict[str, Any]]:
        items = self._parse_json_response(response)

        validated = []
        for item in items:
            if "summary_type" in item and "summary" in item:
                validated.append({
                    "summary_type": str(item["summary_type"]).strip(),
                    "original_text": str(item.get("original_text", "")).strip(),
                    "summary": str(item["summary"]).strip(),
                })

        return validated


class CustomExtractor(BaseExtractor):
    """
    Custom extractor with user-defined prompts and parsing.

    Allows full customization of the extraction process.
    """

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        prompt_template: str,
        parser_fn: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
        extractor_name: str = "custom",
        **kwargs: Any,
    ) -> None:
        """
        Initialize custom extractor.

        Args:
            llm_fn: LLM function for extraction.
            prompt_template: Template with {content} placeholder.
            parser_fn: Custom parser function for responses.
            extractor_name: Name for this extractor type.
            **kwargs: Additional arguments.
        """
        super().__init__(llm_fn, **kwargs)
        self.prompt_template = prompt_template
        self.parser_fn = parser_fn
        self._extractor_name = extractor_name

    @property
    def extractor_type(self) -> str:
        return self._extractor_name

    def get_extraction_prompt(self, content: str) -> str:
        return self.prompt_template.format(
            content=content,
            max_items=self.max_items_per_chunk,
        )

    def parse_response(self, response: str) -> List[Dict[str, Any]]:
        if self.parser_fn:
            return self.parser_fn(response)
        return self._parse_json_response(response)


def get_extractor(
    extractor_type: str,
    llm_fn: Callable[[str], str],
    **kwargs: Any,
) -> BaseExtractor:
    """
    Factory function to create extractors.

    Args:
        extractor_type: Type of extractor ("qa", "rules", "facts", etc.).
        llm_fn: LLM function for extraction.
        **kwargs: Additional arguments for the extractor.

    Returns:
        Extractor instance.
    """
    extractors = {
        "qa": QAExtractor,
        "rules": RulesExtractor,
        "facts": FactsExtractor,
        "instruction": InstructionExtractor,
        "conversation": ConversationExtractor,
        "summary": SummaryExtractor,
    }

    if extractor_type not in extractors:
        raise ValueError(
            f"Unknown extractor type: {extractor_type}. "
            f"Available: {list(extractors.keys())}"
        )

    return extractors[extractor_type](llm_fn, **kwargs)

"""
Cost estimation for doc2dataset.

Provides accurate cost estimation before running expensive
LLM operations for document processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from doc2dataset.loaders import Document


@dataclass
class TokenCounts:
    """
    Token counts for a document or operation.

    Attributes:
        input_tokens: Number of input tokens.
        output_tokens: Estimated output tokens.
        total_tokens: Total tokens.
    """

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class CostEstimate:
    """
    Cost estimate for processing.

    Attributes:
        tokens: Token counts.
        input_cost: Cost for input tokens.
        output_cost: Cost for output tokens.
        total_cost: Total estimated cost.
        currency: Currency (default USD).
        breakdown: Per-document breakdown.
    """

    tokens: TokenCounts
    input_cost: float
    output_cost: float
    currency: str = "USD"
    breakdown: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost

    def __repr__(self) -> str:
        return (
            f"CostEstimate(tokens={self.tokens.total_tokens:,}, "
            f"cost=${self.total_cost:.4f} {self.currency})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_tokens": self.tokens.input_tokens,
            "output_tokens": self.tokens.output_tokens,
            "total_tokens": self.tokens.total_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
            "currency": self.currency,
            "num_documents": len(self.breakdown),
        }


# Pricing per 1M tokens (as of 2024)
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Anthropic
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    # Other common models
    "mistral-large": {"input": 4.00, "output": 12.00},
    "mistral-medium": {"input": 2.70, "output": 8.10},
    "mistral-small": {"input": 1.00, "output": 3.00},
    "llama-3-70b": {"input": 0.90, "output": 0.90},
    "llama-3-8b": {"input": 0.20, "output": 0.20},
    "gemini-pro": {"input": 0.50, "output": 1.50},
    "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
}


class TokenCounter:
    """
    Token counter for various models.

    Uses tiktoken for OpenAI models and approximations for others.
    """

    def __init__(self, model: str = "gpt-4o") -> None:
        """
        Initialize token counter.

        Args:
            model: Model name for tokenizer selection.
        """
        self.model = model
        self._encoder = None

    def _get_encoder(self):
        """Get or create tiktoken encoder."""
        if self._encoder is None:
            try:
                import tiktoken

                # Map model to encoding
                if "gpt-4" in self.model or "gpt-3.5" in self.model:
                    self._encoder = tiktoken.encoding_for_model(self.model)
                else:
                    # Default to cl100k_base for other models
                    self._encoder = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                self._encoder = None
        return self._encoder

    def count(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for.

        Returns:
            Token count.
        """
        encoder = self._get_encoder()
        if encoder:
            return len(encoder.encode(text))
        # Fallback: rough approximation (4 chars per token)
        return len(text) // 4

    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            Token count including message overhead.
        """
        total = 0
        for message in messages:
            # Overhead per message (role, separators)
            total += 4
            total += self.count(message.get("role", ""))
            total += self.count(message.get("content", ""))
        total += 2  # Reply priming
        return total


class CostEstimator:
    """
    Estimates cost of document processing.

    Provides accurate cost estimates before running expensive
    LLM operations.

    Example:
        >>> estimator = CostEstimator(model="gpt-4o")
        >>> docs = load_folder("./documents")
        >>> estimate = estimator.estimate_extraction(docs, extraction_type="qa")
        >>> print(f"Estimated cost: ${estimate.total_cost:.2f}")
    """

    # Output token multipliers by extraction type
    OUTPUT_MULTIPLIERS = {
        "qa": 0.8,  # Q&A pairs tend to be concise
        "rules": 0.5,  # Rules are usually brief
        "facts": 0.6,  # Facts are brief statements
        "instructions": 1.0,  # Instructions can be verbose
        "conversations": 1.5,  # Conversations are longer
        "summary": 0.3,  # Summaries compress content
        "custom": 0.7,  # Default estimate
    }

    def __init__(
        self,
        model: str = "gpt-4o",
        custom_pricing: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize cost estimator.

        Args:
            model: Model name for pricing lookup.
            custom_pricing: Custom pricing dict with 'input' and 'output' keys.
        """
        self.model = model
        self.token_counter = TokenCounter(model)

        # Get pricing
        if custom_pricing:
            self.pricing = custom_pricing
        else:
            self.pricing = self._get_pricing(model)

    def _get_pricing(self, model: str) -> Dict[str, float]:
        """Get pricing for a model."""
        # Exact match
        if model in MODEL_PRICING:
            return MODEL_PRICING[model]

        # Fuzzy match
        model_lower = model.lower()
        for key, pricing in MODEL_PRICING.items():
            if key in model_lower or model_lower in key:
                return pricing

        # Default pricing (conservative estimate)
        return {"input": 5.00, "output": 15.00}

    def estimate_document(
        self,
        document: Document,
        extraction_type: str = "qa",
        system_prompt_tokens: int = 500,
    ) -> Tuple[TokenCounts, Dict[str, Any]]:
        """
        Estimate tokens for a single document.

        Args:
            document: Document to estimate.
            extraction_type: Type of extraction.
            system_prompt_tokens: Tokens used by system prompt.

        Returns:
            Tuple of (TokenCounts, breakdown dict).
        """
        content_tokens = self.token_counter.count(document.content)
        input_tokens = system_prompt_tokens + content_tokens

        # Estimate output based on extraction type
        multiplier = self.OUTPUT_MULTIPLIERS.get(extraction_type, 0.7)
        output_tokens = int(content_tokens * multiplier)

        # Ensure minimum output
        output_tokens = max(output_tokens, 100)

        breakdown = {
            "source": document.source,
            "content_tokens": content_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

        return TokenCounts(input_tokens, output_tokens), breakdown

    def estimate_extraction(
        self,
        documents: List[Document],
        extraction_type: str = "qa",
        system_prompt: Optional[str] = None,
    ) -> CostEstimate:
        """
        Estimate cost for extracting from documents.

        Args:
            documents: List of documents to process.
            extraction_type: Type of extraction to perform.
            system_prompt: Optional system prompt (for accurate counting).

        Returns:
            CostEstimate with breakdown.
        """
        if system_prompt:
            system_prompt_tokens = self.token_counter.count(system_prompt)
        else:
            system_prompt_tokens = 500  # Default estimate

        total_input = 0
        total_output = 0
        breakdown = []

        for doc in documents:
            counts, doc_breakdown = self.estimate_document(
                doc,
                extraction_type=extraction_type,
                system_prompt_tokens=system_prompt_tokens,
            )
            total_input += counts.input_tokens
            total_output += counts.output_tokens
            breakdown.append(doc_breakdown)

        # Calculate costs (pricing is per 1M tokens)
        input_cost = (total_input / 1_000_000) * self.pricing["input"]
        output_cost = (total_output / 1_000_000) * self.pricing["output"]

        return CostEstimate(
            tokens=TokenCounts(total_input, total_output),
            input_cost=input_cost,
            output_cost=output_cost,
            breakdown=breakdown,
        )

    def estimate_batch(
        self,
        documents: List[Document],
        extraction_types: List[str],
        system_prompts: Optional[Dict[str, str]] = None,
    ) -> Dict[str, CostEstimate]:
        """
        Estimate cost for multiple extraction types.

        Args:
            documents: Documents to process.
            extraction_types: List of extraction types.
            system_prompts: Optional dict mapping type to prompt.

        Returns:
            Dict mapping extraction type to estimate.
        """
        estimates = {}
        for ext_type in extraction_types:
            prompt = system_prompts.get(ext_type) if system_prompts else None
            estimates[ext_type] = self.estimate_extraction(
                documents,
                extraction_type=ext_type,
                system_prompt=prompt,
            )
        return estimates

    def format_estimate(
        self,
        estimate: CostEstimate,
        verbose: bool = False,
    ) -> str:
        """
        Format estimate for display.

        Args:
            estimate: Cost estimate.
            verbose: Include per-document breakdown.

        Returns:
            Formatted string.
        """
        lines = [
            "=" * 50,
            "COST ESTIMATE",
            "=" * 50,
            f"Model: {self.model}",
            f"Documents: {len(estimate.breakdown)}",
            "",
            "Token Usage:",
            f"  Input tokens:  {estimate.tokens.input_tokens:,}",
            f"  Output tokens: {estimate.tokens.output_tokens:,}",
            f"  Total tokens:  {estimate.tokens.total_tokens:,}",
            "",
            "Estimated Cost:",
            f"  Input:  ${estimate.input_cost:.4f}",
            f"  Output: ${estimate.output_cost:.4f}",
            f"  Total:  ${estimate.total_cost:.4f} {estimate.currency}",
            "=" * 50,
        ]

        if verbose and estimate.breakdown:
            lines.append("")
            lines.append("Per-Document Breakdown:")
            lines.append("-" * 50)
            for item in estimate.breakdown[:10]:  # Limit to first 10
                lines.append(
                    f"  {item['source']}: "
                    f"{item['content_tokens']:,} tokens"
                )
            if len(estimate.breakdown) > 10:
                lines.append(f"  ... and {len(estimate.breakdown) - 10} more")

        return "\n".join(lines)


def estimate_processing_cost(
    documents: List[Document],
    model: str = "gpt-4o",
    extraction_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Quick cost estimation function.

    Args:
        documents: Documents to process.
        model: Model to use.
        extraction_types: Extraction types (default: all).

    Returns:
        Dict with estimates and totals.
    """
    extraction_types = extraction_types or [
        "qa", "rules", "facts", "instructions"
    ]

    estimator = CostEstimator(model=model)
    estimates = estimator.estimate_batch(documents, extraction_types)

    # Calculate totals
    total_tokens = sum(e.tokens.total_tokens for e in estimates.values())
    total_cost = sum(e.total_cost for e in estimates.values())

    return {
        "model": model,
        "num_documents": len(documents),
        "extraction_types": extraction_types,
        "estimates": {k: v.to_dict() for k, v in estimates.items()},
        "total_tokens": total_tokens,
        "total_cost": total_cost,
    }

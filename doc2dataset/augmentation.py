"""
Data augmentation for doc2dataset.

Provides techniques to increase training data diversity
and improve model generalization.
"""

from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class AugmentedItem:
    """
    An augmented training item.

    Attributes:
        original: Original item.
        augmented: Augmented version.
        augmentation_type: Type of augmentation applied.
        metadata: Additional metadata.
    """

    original: Dict[str, Any]
    augmented: Dict[str, Any]
    augmentation_type: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseAugmenter(ABC):
    """Abstract base class for augmenters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Augmenter name."""
        pass

    @abstractmethod
    def augment(
        self,
        item: Dict[str, Any],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Augment a training item.

        Args:
            item: Item to augment.
            **kwargs: Additional options.

        Returns:
            List of augmented versions.
        """
        pass


class ParaphraseAugmenter(BaseAugmenter):
    """
    Augments by paraphrasing text using LLM.

    Creates variations with different wording
    while preserving meaning.
    """

    PROMPT_TEMPLATE = """Paraphrase the following text while preserving its exact meaning.
Use different words and sentence structure.

Original:
{text}

Paraphrase:"""

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        field: str = "output",
        num_variations: int = 2,
    ) -> None:
        """
        Initialize paraphrase augmenter.

        Args:
            llm_fn: Function that takes prompt and returns response.
            field: Field to paraphrase.
            num_variations: Number of paraphrases to generate.
        """
        self.llm_fn = llm_fn
        self.field = field
        self.num_variations = num_variations

    @property
    def name(self) -> str:
        return "paraphrase"

    def augment(
        self,
        item: Dict[str, Any],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Generate paraphrased versions."""
        text = item.get(self.field, "")
        if not text:
            return []

        augmented = []
        for _ in range(self.num_variations):
            prompt = self.PROMPT_TEMPLATE.format(text=text)
            try:
                paraphrased = self.llm_fn(prompt).strip()
                if paraphrased and paraphrased != text:
                    new_item = {**item, self.field: paraphrased}
                    new_item["_augmentation"] = "paraphrase"
                    augmented.append(new_item)
            except Exception:
                continue

        return augmented


class QuestionVariationAugmenter(BaseAugmenter):
    """
    Augments Q&A pairs with question variations.

    Generates different ways to ask the same question.
    """

    PROMPT_TEMPLATE = """Generate {num_variations} different ways to ask the same question.
Keep the same meaning but use different wording.

Original question:
{question}

Generate each variation on a new line, numbered 1-{num_variations}:"""

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        question_field: str = "input",
        num_variations: int = 3,
    ) -> None:
        """
        Initialize question variation augmenter.

        Args:
            llm_fn: LLM function.
            question_field: Field containing the question.
            num_variations: Number of variations.
        """
        self.llm_fn = llm_fn
        self.question_field = question_field
        self.num_variations = num_variations

    @property
    def name(self) -> str:
        return "question_variation"

    def augment(
        self,
        item: Dict[str, Any],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Generate question variations."""
        question = item.get(self.question_field, "")
        if not question:
            return []

        prompt = self.PROMPT_TEMPLATE.format(
            question=question,
            num_variations=self.num_variations,
        )

        try:
            response = self.llm_fn(prompt)
            variations = self._parse_variations(response)

            augmented = []
            for var in variations:
                if var and var != question:
                    new_item = {**item, self.question_field: var}
                    new_item["_augmentation"] = "question_variation"
                    augmented.append(new_item)

            return augmented
        except Exception:
            return []

    def _parse_variations(self, response: str) -> List[str]:
        """Parse numbered variations from response."""
        # Match lines starting with numbers
        pattern = r"^\d+[.):\s]+(.+)$"
        matches = re.findall(pattern, response, re.MULTILINE)
        return [m.strip() for m in matches if m.strip()]


class BackTranslationAugmenter(BaseAugmenter):
    """
    Augments using back-translation.

    Translates to another language and back to create variations.
    """

    TRANSLATE_PROMPT = """Translate the following text to {target_lang}:

{text}

Translation:"""

    BACK_TRANSLATE_PROMPT = """Translate the following {source_lang} text to English:

{text}

Translation:"""

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        field: str = "output",
        intermediate_languages: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize back-translation augmenter.

        Args:
            llm_fn: LLM function.
            field: Field to augment.
            intermediate_languages: Languages to translate through.
        """
        self.llm_fn = llm_fn
        self.field = field
        self.intermediate_languages = intermediate_languages or [
            "French", "German", "Spanish"
        ]

    @property
    def name(self) -> str:
        return "back_translation"

    def augment(
        self,
        item: Dict[str, Any],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Generate back-translated versions."""
        text = item.get(self.field, "")
        if not text or len(text) < 20:
            return []

        augmented = []
        for lang in self.intermediate_languages:
            try:
                # Translate to intermediate language
                translate_prompt = self.TRANSLATE_PROMPT.format(
                    target_lang=lang,
                    text=text,
                )
                translated = self.llm_fn(translate_prompt).strip()

                # Translate back
                back_prompt = self.BACK_TRANSLATE_PROMPT.format(
                    source_lang=lang,
                    text=translated,
                )
                back_translated = self.llm_fn(back_prompt).strip()

                if back_translated and back_translated != text:
                    new_item = {**item, self.field: back_translated}
                    new_item["_augmentation"] = f"back_translation_{lang.lower()}"
                    augmented.append(new_item)

            except Exception:
                continue

        return augmented


class SynonymReplacementAugmenter(BaseAugmenter):
    """
    Augments by replacing words with synonyms.

    Simple augmentation that doesn't require LLM.
    """

    # Common synonym pairs
    SYNONYMS = {
        "important": ["crucial", "vital", "essential", "significant"],
        "good": ["excellent", "great", "fine", "positive"],
        "bad": ["poor", "negative", "unfavorable", "adverse"],
        "big": ["large", "substantial", "significant", "major"],
        "small": ["little", "minor", "slight", "modest"],
        "fast": ["quick", "rapid", "swift", "speedy"],
        "slow": ["gradual", "unhurried", "leisurely", "sluggish"],
        "help": ["assist", "aid", "support", "facilitate"],
        "use": ["utilize", "employ", "apply", "leverage"],
        "show": ["demonstrate", "display", "indicate", "reveal"],
        "make": ["create", "produce", "generate", "develop"],
        "get": ["obtain", "acquire", "receive", "gain"],
        "think": ["believe", "consider", "assume", "suppose"],
        "know": ["understand", "recognize", "realize", "comprehend"],
        "need": ["require", "necessitate", "demand", "want"],
    }

    def __init__(
        self,
        field: str = "output",
        replacement_prob: float = 0.2,
        num_variations: int = 2,
    ) -> None:
        """
        Initialize synonym replacement augmenter.

        Args:
            field: Field to augment.
            replacement_prob: Probability of replacing each word.
            num_variations: Number of variations to generate.
        """
        self.field = field
        self.replacement_prob = replacement_prob
        self.num_variations = num_variations

    @property
    def name(self) -> str:
        return "synonym_replacement"

    def augment(
        self,
        item: Dict[str, Any],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Generate synonym-replaced versions."""
        text = item.get(self.field, "")
        if not text:
            return []

        augmented = []
        for _ in range(self.num_variations):
            replaced = self._replace_synonyms(text)
            if replaced != text:
                new_item = {**item, self.field: replaced}
                new_item["_augmentation"] = "synonym_replacement"
                augmented.append(new_item)

        return augmented

    def _replace_synonyms(self, text: str) -> str:
        """Replace words with synonyms."""
        words = text.split()
        result = []

        for word in words:
            word_lower = word.lower().strip(".,!?;:")
            punctuation = ""
            if word[-1:] in ".,!?;:":
                punctuation = word[-1]
                word = word[:-1]

            if (
                word_lower in self.SYNONYMS and
                random.random() < self.replacement_prob
            ):
                synonyms = self.SYNONYMS[word_lower]
                replacement = random.choice(synonyms)

                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()

                result.append(replacement + punctuation)
            else:
                result.append(word + punctuation)

        return " ".join(result)


class NegationAugmenter(BaseAugmenter):
    """
    Augments by creating negated versions.

    Useful for teaching models to distinguish
    positive and negative statements.
    """

    PROMPT_TEMPLATE = """Create a negated or opposite version of the following statement.
The negated version should express the opposite meaning.

Original:
{text}

Negated version:"""

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        field: str = "output",
    ) -> None:
        """
        Initialize negation augmenter.

        Args:
            llm_fn: LLM function.
            field: Field to negate.
        """
        self.llm_fn = llm_fn
        self.field = field

    @property
    def name(self) -> str:
        return "negation"

    def augment(
        self,
        item: Dict[str, Any],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Generate negated version."""
        text = item.get(self.field, "")
        if not text:
            return []

        prompt = self.PROMPT_TEMPLATE.format(text=text)

        try:
            negated = self.llm_fn(prompt).strip()
            if negated and negated != text:
                new_item = {**item, self.field: negated}
                new_item["_augmentation"] = "negation"
                new_item["_is_negation"] = True
                return [new_item]
        except Exception:
            pass

        return []


class InstructionStyleAugmenter(BaseAugmenter):
    """
    Augments instructions by varying style.

    Converts between formal/informal, detailed/concise styles.
    """

    STYLE_PROMPTS = {
        "formal": "Rewrite the following in a formal, professional style:\n\n{text}\n\nFormal version:",
        "informal": "Rewrite the following in a casual, conversational style:\n\n{text}\n\nInformal version:",
        "detailed": "Expand the following with more details and explanations:\n\n{text}\n\nDetailed version:",
        "concise": "Make the following more concise while keeping key information:\n\n{text}\n\nConcise version:",
        "technical": "Rewrite the following in a technical, precise style:\n\n{text}\n\nTechnical version:",
    }

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        field: str = "output",
        styles: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize style augmenter.

        Args:
            llm_fn: LLM function.
            field: Field to augment.
            styles: Styles to generate (default: all).
        """
        self.llm_fn = llm_fn
        self.field = field
        self.styles = styles or list(self.STYLE_PROMPTS.keys())

    @property
    def name(self) -> str:
        return "style_variation"

    def augment(
        self,
        item: Dict[str, Any],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Generate style variations."""
        text = item.get(self.field, "")
        if not text or len(text) < 30:
            return []

        augmented = []
        for style in self.styles:
            prompt = self.STYLE_PROMPTS[style].format(text=text)

            try:
                varied = self.llm_fn(prompt).strip()
                if varied and varied != text:
                    new_item = {**item, self.field: varied}
                    new_item["_augmentation"] = f"style_{style}"
                    augmented.append(new_item)
            except Exception:
                continue

        return augmented


class DataAugmenter:
    """
    Combines multiple augmentation strategies.

    Example:
        >>> augmenter = DataAugmenter(llm_fn=my_llm)
        >>> augmenter.add(ParaphraseAugmenter(llm_fn))
        >>> augmenter.add(SynonymReplacementAugmenter())
        >>> augmented_data = augmenter.augment_dataset(items)
    """

    def __init__(
        self,
        augmenters: Optional[List[BaseAugmenter]] = None,
        augmentation_factor: float = 2.0,
    ) -> None:
        """
        Initialize data augmenter.

        Args:
            augmenters: List of augmenters to use.
            augmentation_factor: Target increase in dataset size.
        """
        self.augmenters = augmenters or []
        self.augmentation_factor = augmentation_factor

    def add(self, augmenter: BaseAugmenter) -> "DataAugmenter":
        """Add an augmenter."""
        self.augmenters.append(augmenter)
        return self

    def augment_item(
        self,
        item: Dict[str, Any],
        max_augmentations: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Augment a single item.

        Args:
            item: Item to augment.
            max_augmentations: Maximum augmentations to create.

        Returns:
            List of augmented items.
        """
        all_augmented = []

        for augmenter in self.augmenters:
            try:
                augmented = augmenter.augment(item)
                all_augmented.extend(augmented)
            except Exception:
                continue

        # Limit if needed
        if max_augmentations and len(all_augmented) > max_augmentations:
            all_augmented = random.sample(all_augmented, max_augmentations)

        return all_augmented

    def augment_dataset(
        self,
        items: List[Dict[str, Any]],
        include_original: bool = True,
        shuffle: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Augment a dataset.

        Args:
            items: Original items.
            include_original: Include original items in output.
            shuffle: Shuffle final dataset.

        Returns:
            Augmented dataset.
        """
        # Calculate augmentations per item
        target_size = int(len(items) * self.augmentation_factor)
        augmentations_needed = target_size - len(items)
        per_item = max(1, augmentations_needed // len(items))

        result = list(items) if include_original else []

        for item in items:
            augmented = self.augment_item(item, max_augmentations=per_item)
            result.extend(augmented)

            # Stop if we've reached target
            if len(result) >= target_size:
                break

        if shuffle:
            random.shuffle(result)

        return result

    def stats(self) -> Dict[str, Any]:
        """Get augmenter statistics."""
        return {
            "num_augmenters": len(self.augmenters),
            "augmenter_names": [a.name for a in self.augmenters],
            "augmentation_factor": self.augmentation_factor,
        }


def create_default_augmenter(
    llm_fn: Callable[[str], str],
    field: str = "output",
) -> DataAugmenter:
    """
    Create a default augmenter with common strategies.

    Args:
        llm_fn: LLM function for generation-based augmentation.
        field: Field to augment.

    Returns:
        Configured DataAugmenter.
    """
    augmenter = DataAugmenter(augmentation_factor=2.0)
    augmenter.add(ParaphraseAugmenter(llm_fn, field=field, num_variations=1))
    augmenter.add(SynonymReplacementAugmenter(field=field, num_variations=2))
    return augmenter


def create_qa_augmenter(
    llm_fn: Callable[[str], str],
) -> DataAugmenter:
    """
    Create augmenter optimized for Q&A data.

    Args:
        llm_fn: LLM function.

    Returns:
        Configured DataAugmenter.
    """
    augmenter = DataAugmenter(augmentation_factor=3.0)
    augmenter.add(QuestionVariationAugmenter(llm_fn, num_variations=2))
    augmenter.add(ParaphraseAugmenter(llm_fn, field="output", num_variations=1))
    augmenter.add(SynonymReplacementAugmenter(field="output", num_variations=1))
    return augmenter

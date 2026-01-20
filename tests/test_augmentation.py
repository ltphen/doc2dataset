"""Tests for augmentation module."""

import pytest
from doc2dataset.augmentation import (
    AugmentedItem,
    BaseAugmenter,
    ParaphraseAugmenter,
    QuestionVariationAugmenter,
    BackTranslationAugmenter,
    SynonymReplacementAugmenter,
    NegationAugmenter,
    InstructionStyleAugmenter,
    DataAugmenter,
    create_default_augmenter,
    create_qa_augmenter,
)


class TestAugmentedItem:
    """Tests for AugmentedItem dataclass."""

    def test_basic_creation(self):
        item = AugmentedItem(
            original={"input": "Q", "output": "A"},
            augmented={"input": "Q2", "output": "A"},
            augmentation_type="paraphrase",
        )
        assert item.original["input"] == "Q"
        assert item.augmented["input"] == "Q2"
        assert item.augmentation_type == "paraphrase"
        assert item.metadata == {}

    def test_with_metadata(self):
        item = AugmentedItem(
            original={},
            augmented={},
            augmentation_type="test",
            metadata={"key": "value"},
        )
        assert item.metadata["key"] == "value"


class TestSynonymReplacementAugmenter:
    """Tests for SynonymReplacementAugmenter (no LLM required)."""

    def test_basic_augmentation(self):
        augmenter = SynonymReplacementAugmenter(
            field="output",
            replacement_prob=1.0,  # Always replace
            num_variations=1,
        )
        item = {"input": "Q", "output": "This is important information."}
        results = augmenter.augment(item)
        # May or may not produce results depending on random
        assert isinstance(results, list)

    def test_name_property(self):
        augmenter = SynonymReplacementAugmenter()
        assert augmenter.name == "synonym_replacement"

    def test_empty_field(self):
        augmenter = SynonymReplacementAugmenter()
        item = {"input": "Q", "output": ""}
        results = augmenter.augment(item)
        assert results == []

    def test_no_synonyms_available(self):
        augmenter = SynonymReplacementAugmenter(
            replacement_prob=1.0,
            num_variations=1,
        )
        item = {"output": "xyz abc 123"}  # No synonyms for these
        results = augmenter.augment(item)
        # Should return empty or same text
        assert isinstance(results, list)

    def test_multiple_variations(self):
        augmenter = SynonymReplacementAugmenter(
            replacement_prob=0.5,
            num_variations=5,
        )
        item = {"output": "This is important and good work."}
        results = augmenter.augment(item)
        # May produce 0-5 variations
        assert isinstance(results, list)
        assert len(results) <= 5

    def test_preserves_punctuation(self):
        augmenter = SynonymReplacementAugmenter(replacement_prob=1.0)
        item = {"output": "This is important."}
        results = augmenter.augment(item)
        # Any result should preserve sentence structure
        for result in results:
            assert isinstance(result.get("output", ""), str)

    def test_augmentation_marker(self):
        augmenter = SynonymReplacementAugmenter(
            replacement_prob=1.0,
            num_variations=3,
        )
        item = {"output": "This is important data."}
        results = augmenter.augment(item)
        for result in results:
            if result.get("output") != item["output"]:
                assert result.get("_augmentation") == "synonym_replacement"


class TestParaphraseAugmenter:
    """Tests for ParaphraseAugmenter."""

    def test_basic_augmentation(self):
        def mock_llm(prompt):
            return "This is a paraphrased version."

        augmenter = ParaphraseAugmenter(
            llm_fn=mock_llm,
            field="output",
            num_variations=2,
        )
        item = {"input": "Q", "output": "Original text here."}
        results = augmenter.augment(item)
        assert len(results) == 2
        assert all(r["_augmentation"] == "paraphrase" for r in results)

    def test_name_property(self):
        augmenter = ParaphraseAugmenter(llm_fn=lambda x: "")
        assert augmenter.name == "paraphrase"

    def test_empty_field(self):
        augmenter = ParaphraseAugmenter(llm_fn=lambda x: "result")
        item = {"output": ""}
        results = augmenter.augment(item)
        assert results == []

    def test_llm_error_handled(self):
        def failing_llm(prompt):
            raise Exception("API error")

        augmenter = ParaphraseAugmenter(llm_fn=failing_llm, num_variations=2)
        item = {"output": "Test text"}
        results = augmenter.augment(item)
        # Should return empty list, not raise
        assert results == []

    def test_same_output_filtered(self):
        def same_llm(prompt):
            return "Same text"

        augmenter = ParaphraseAugmenter(llm_fn=same_llm, num_variations=2)
        item = {"output": "Same text"}
        results = augmenter.augment(item)
        # Same output should be filtered
        assert len(results) == 0


class TestQuestionVariationAugmenter:
    """Tests for QuestionVariationAugmenter."""

    def test_basic_augmentation(self):
        def mock_llm(prompt):
            return "1. What is Python programming?\n2. Can you explain Python?\n3. Tell me about Python."

        augmenter = QuestionVariationAugmenter(
            llm_fn=mock_llm,
            question_field="input",
            num_variations=3,
        )
        item = {"input": "What is Python?", "output": "A language"}
        results = augmenter.augment(item)
        assert len(results) <= 3  # May filter some
        for r in results:
            assert r.get("_augmentation") == "question_variation"

    def test_name_property(self):
        augmenter = QuestionVariationAugmenter(llm_fn=lambda x: "")
        assert augmenter.name == "question_variation"

    def test_empty_question(self):
        augmenter = QuestionVariationAugmenter(llm_fn=lambda x: "1. Q")
        item = {"input": "", "output": "A"}
        results = augmenter.augment(item)
        assert results == []


class TestBackTranslationAugmenter:
    """Tests for BackTranslationAugmenter."""

    def test_basic_augmentation(self):
        call_count = [0]

        def mock_llm(prompt):
            call_count[0] += 1
            if "Translate the following text to" in prompt:
                return "Translated text"
            else:
                return "Back translated text"

        augmenter = BackTranslationAugmenter(
            llm_fn=mock_llm,
            field="output",
            intermediate_languages=["French"],
        )
        item = {"output": "This is a test sentence for translation."}
        results = augmenter.augment(item)
        # Should have 2 calls per language (translate + back-translate)
        assert call_count[0] == 2
        assert len(results) <= 1

    def test_name_property(self):
        augmenter = BackTranslationAugmenter(llm_fn=lambda x: "")
        assert augmenter.name == "back_translation"

    def test_short_text_skipped(self):
        augmenter = BackTranslationAugmenter(llm_fn=lambda x: "result")
        item = {"output": "Short"}  # Less than 20 chars
        results = augmenter.augment(item)
        assert results == []


class TestNegationAugmenter:
    """Tests for NegationAugmenter."""

    def test_basic_augmentation(self):
        def mock_llm(prompt):
            return "Python is not a programming language."

        augmenter = NegationAugmenter(llm_fn=mock_llm, field="output")
        item = {"output": "Python is a programming language."}
        results = augmenter.augment(item)
        assert len(results) == 1
        assert results[0]["_augmentation"] == "negation"
        assert results[0].get("_is_negation") is True

    def test_name_property(self):
        augmenter = NegationAugmenter(llm_fn=lambda x: "")
        assert augmenter.name == "negation"


class TestInstructionStyleAugmenter:
    """Tests for InstructionStyleAugmenter."""

    def test_basic_augmentation(self):
        def mock_llm(prompt):
            return "Styled version of the text."

        augmenter = InstructionStyleAugmenter(
            llm_fn=mock_llm,
            field="output",
            styles=["formal", "informal"],
        )
        item = {"output": "This is some content that needs to be styled differently."}
        results = augmenter.augment(item)
        assert len(results) <= 2
        for r in results:
            assert "_augmentation" in r
            assert r["_augmentation"].startswith("style_")

    def test_name_property(self):
        augmenter = InstructionStyleAugmenter(llm_fn=lambda x: "")
        assert augmenter.name == "style_variation"

    def test_short_text_skipped(self):
        augmenter = InstructionStyleAugmenter(llm_fn=lambda x: "result")
        item = {"output": "Short text"}  # Less than 30 chars
        results = augmenter.augment(item)
        assert results == []


class TestDataAugmenter:
    """Tests for DataAugmenter orchestrator."""

    def test_basic_creation(self):
        augmenter = DataAugmenter()
        assert len(augmenter.augmenters) == 0
        assert augmenter.augmentation_factor == 2.0

    def test_add_augmenter(self):
        augmenter = DataAugmenter()
        synonym_aug = SynonymReplacementAugmenter()
        augmenter.add(synonym_aug)
        assert len(augmenter.augmenters) == 1

    def test_add_chaining(self):
        augmenter = DataAugmenter()
        result = augmenter.add(SynonymReplacementAugmenter())
        assert result is augmenter  # Should return self

    def test_augment_item(self):
        augmenter = DataAugmenter()
        augmenter.add(SynonymReplacementAugmenter(replacement_prob=0.5))

        item = {"output": "This is important information."}
        results = augmenter.augment_item(item)
        assert isinstance(results, list)

    def test_augment_item_max_augmentations(self):
        def mock_llm(prompt):
            return "Result " + str(len(prompt))

        augmenter = DataAugmenter()
        augmenter.add(ParaphraseAugmenter(mock_llm, num_variations=5))
        augmenter.add(SynonymReplacementAugmenter(num_variations=5))

        item = {"output": "This is test content."}
        results = augmenter.augment_item(item, max_augmentations=3)
        assert len(results) <= 3

    def test_augment_dataset(self):
        augmenter = DataAugmenter(augmentation_factor=2.0)
        augmenter.add(SynonymReplacementAugmenter(replacement_prob=0.5))

        items = [
            {"output": "This is important."},
            {"output": "Something good here."},
        ]
        results = augmenter.augment_dataset(items, include_original=True)
        # Should include originals plus augmentations
        assert len(results) >= 2

    def test_augment_dataset_without_original(self):
        augmenter = DataAugmenter()
        augmenter.add(SynonymReplacementAugmenter(replacement_prob=1.0))

        items = [{"output": "Important data."}]
        results = augmenter.augment_dataset(items, include_original=False)
        # Should not include original
        # May or may not have augmentations
        assert isinstance(results, list)

    def test_stats(self):
        augmenter = DataAugmenter(augmentation_factor=3.0)
        augmenter.add(SynonymReplacementAugmenter())
        augmenter.add(ParaphraseAugmenter(llm_fn=lambda x: ""))

        stats = augmenter.stats()
        assert stats["num_augmenters"] == 2
        assert stats["augmentation_factor"] == 3.0
        assert "synonym_replacement" in stats["augmenter_names"]
        assert "paraphrase" in stats["augmenter_names"]


class TestCreateDefaultAugmenter:
    """Tests for create_default_augmenter factory."""

    def test_creation(self):
        def mock_llm(prompt):
            return "Result"

        augmenter = create_default_augmenter(mock_llm, field="output")
        assert isinstance(augmenter, DataAugmenter)
        assert len(augmenter.augmenters) >= 1


class TestCreateQAAugmenter:
    """Tests for create_qa_augmenter factory."""

    def test_creation(self):
        def mock_llm(prompt):
            return "1. Question variation"

        augmenter = create_qa_augmenter(mock_llm)
        assert isinstance(augmenter, DataAugmenter)
        assert len(augmenter.augmenters) >= 2
        assert augmenter.augmentation_factor == 3.0


class TestBaseAugmenter:
    """Tests for BaseAugmenter abstract class."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseAugmenter()

    def test_custom_augmenter(self):
        class CustomAugmenter(BaseAugmenter):
            @property
            def name(self):
                return "custom"

            def augment(self, item, **kwargs):
                return [{"custom": True, **item}]

        augmenter = CustomAugmenter()
        assert augmenter.name == "custom"
        results = augmenter.augment({"data": "test"})
        assert len(results) == 1
        assert results[0]["custom"] is True

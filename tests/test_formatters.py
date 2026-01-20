"""Tests for formatters module."""

import pytest

from doc2dataset.formatters import (
    OpenAIFormatter,
    AlpacaFormatter,
    ShareGPTFormatter,
    GenericFormatter,
    get_formatter,
    list_formats,
)


class TestOpenAIFormatter:
    """Tests for OpenAI formatter."""

    def test_qa_format(self):
        """Test Q&A formatting."""
        formatter = OpenAIFormatter()
        item = {"question": "What is X?", "answer": "X is Y."}

        result = formatter.format(item)

        assert "messages" in result
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"

    def test_instruction_format(self):
        """Test instruction formatting."""
        formatter = OpenAIFormatter()
        item = {
            "instruction": "Explain this",
            "input": "Some input",
            "output": "The explanation"
        }

        result = formatter.format(item)

        assert "messages" in result
        assert "Some input" in result["messages"][0]["content"]

    def test_conversation_format(self):
        """Test conversation formatting."""
        formatter = OpenAIFormatter()
        item = {
            "conversation": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well!"},
            ]
        }

        result = formatter.format(item)

        assert len(result["messages"]) == 4

    def test_with_system_prompt(self):
        """Test adding system prompt."""
        formatter = OpenAIFormatter()
        item = {"question": "What?", "answer": "That."}

        result = formatter.format(item, system_prompt="You are helpful.")

        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are helpful."

    def test_rules_format(self):
        """Test rules formatting."""
        formatter = OpenAIFormatter()
        item = {
            "rule": "Always do X",
            "context": "When Y happens",
            "rationale": "Because Z"
        }

        result = formatter.format(item)

        assert "messages" in result
        assert "Always do X" in result["messages"][1]["content"]


class TestAlpacaFormatter:
    """Tests for Alpaca formatter."""

    def test_qa_format(self):
        """Test Q&A formatting."""
        formatter = AlpacaFormatter()
        item = {"question": "What is X?", "answer": "X is Y."}

        result = formatter.format(item)

        assert result["instruction"] == "What is X?"
        assert result["output"] == "X is Y."
        assert result["input"] == ""

    def test_instruction_format(self):
        """Test instruction formatting."""
        formatter = AlpacaFormatter()
        item = {
            "instruction": "Do this",
            "input": "With this",
            "output": "Result"
        }

        result = formatter.format(item)

        assert result["instruction"] == "Do this"
        assert result["input"] == "With this"
        assert result["output"] == "Result"


class TestShareGPTFormatter:
    """Tests for ShareGPT formatter."""

    def test_qa_format(self):
        """Test Q&A formatting."""
        formatter = ShareGPTFormatter()
        item = {"question": "What?", "answer": "That."}

        result = formatter.format(item)

        assert "conversations" in result
        assert result["conversations"][0]["from"] == "human"
        assert result["conversations"][1]["from"] == "gpt"

    def test_conversation_format(self):
        """Test conversation formatting."""
        formatter = ShareGPTFormatter()
        item = {
            "conversation": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }

        result = formatter.format(item)

        assert result["conversations"][0]["from"] == "human"
        assert result["conversations"][1]["from"] == "gpt"


class TestGenericFormatter:
    """Tests for Generic formatter."""

    def test_passthrough(self):
        """Test data passes through."""
        formatter = GenericFormatter()
        item = {"key1": "value1", "key2": "value2"}

        result = formatter.format(item)

        assert result == item

    def test_strips_metadata(self):
        """Test metadata keys are stripped."""
        formatter = GenericFormatter()
        item = {"key": "value", "_internal": "hidden"}

        result = formatter.format(item, include_metadata=False)

        assert "key" in result
        assert "_internal" not in result


class TestFormatterFactory:
    """Tests for formatter factory."""

    def test_get_openai(self):
        """Test getting OpenAI formatter."""
        formatter = get_formatter("openai")
        assert isinstance(formatter, OpenAIFormatter)

    def test_get_alpaca(self):
        """Test getting Alpaca formatter."""
        formatter = get_formatter("alpaca")
        assert isinstance(formatter, AlpacaFormatter)

    def test_get_sharegpt(self):
        """Test getting ShareGPT formatter."""
        formatter = get_formatter("sharegpt")
        assert isinstance(formatter, ShareGPTFormatter)

    def test_unknown_format(self):
        """Test error on unknown format."""
        with pytest.raises(ValueError, match="Unknown format"):
            get_formatter("unknown_format")

    def test_list_formats(self):
        """Test listing available formats."""
        formats = list_formats()

        assert "openai" in formats
        assert "alpaca" in formats
        assert "sharegpt" in formats
        assert "generic" in formats

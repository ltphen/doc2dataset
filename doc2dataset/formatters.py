"""
Output formatters for doc2dataset.

This module provides formatters for converting extracted data
into various fine-tuning formats (OpenAI, Alpaca, ShareGPT, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseFormatter(ABC):
    """
    Abstract base class for output formatters.

    Formatters convert extracted data items into specific
    training data formats.
    """

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return the name of this format."""
        pass

    @abstractmethod
    def format(self, item: Dict[str, Any], **kwargs: Any) -> Optional[Dict[str, Any]]:
        """
        Format a single item.

        Args:
            item: The extracted data item.
            **kwargs: Additional formatting options.

        Returns:
            Formatted dictionary or None if item cannot be formatted.
        """
        pass

    def format_many(
        self,
        items: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Format multiple items.

        Args:
            items: List of extracted data items.
            **kwargs: Additional formatting options.

        Returns:
            List of formatted dictionaries.
        """
        formatted = []
        for item in items:
            result = self.format(item, **kwargs)
            if result:
                formatted.append(result)
        return formatted


class OpenAIFormatter(BaseFormatter):
    """
    Formatter for OpenAI fine-tuning format.

    Outputs data in the format expected by OpenAI's fine-tuning API:
    {"messages": [{"role": "...", "content": "..."}]}

    Supports:
    - Q&A pairs -> user question, assistant answer
    - Instruction-response -> user instruction, assistant response
    - Conversations -> multi-turn messages
    """

    @property
    def format_name(self) -> str:
        return "openai"

    def format(
        self,
        item: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Format item for OpenAI fine-tuning.

        Args:
            item: The extracted data item.
            system_prompt: Optional system message to prepend.
            **kwargs: Additional options.

        Returns:
            Formatted dictionary with "messages" key.
        """
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        # Handle different item types

        # Q&A format
        if "question" in item and "answer" in item:
            messages.append({
                "role": "user",
                "content": item["question"],
            })
            messages.append({
                "role": "assistant",
                "content": item["answer"],
            })

        # Instruction format
        elif "instruction" in item and "output" in item:
            user_content = item["instruction"]
            if item.get("input"):
                user_content += f"\n\nInput: {item['input']}"

            messages.append({
                "role": "user",
                "content": user_content,
            })
            messages.append({
                "role": "assistant",
                "content": item["output"],
            })

        # Conversation format
        elif "conversation" in item:
            for turn in item["conversation"]:
                messages.append({
                    "role": turn["role"],
                    "content": turn["content"],
                })

        # Rules format - convert to instruction
        elif "rule" in item:
            context = item.get("context", "")
            question = f"What is the rule regarding {context}?" if context else "What is the rule?"
            messages.append({
                "role": "user",
                "content": question,
            })
            answer = item["rule"]
            if item.get("rationale"):
                answer += f"\n\nRationale: {item['rationale']}"
            messages.append({
                "role": "assistant",
                "content": answer,
            })

        # Facts format
        elif "fact" in item:
            topic = item.get("topic", "this topic")
            messages.append({
                "role": "user",
                "content": f"Tell me a fact about {topic}.",
            })
            messages.append({
                "role": "assistant",
                "content": item["fact"],
            })

        # Summary format
        elif "summary" in item:
            original = item.get("original_text", "")
            if original:
                messages.append({
                    "role": "user",
                    "content": f"Summarize the following:\n\n{original}",
                })
            else:
                messages.append({
                    "role": "user",
                    "content": "Provide a summary.",
                })
            messages.append({
                "role": "assistant",
                "content": item["summary"],
            })

        else:
            return None

        if len(messages) < 2:
            return None

        return {"messages": messages}


class AlpacaFormatter(BaseFormatter):
    """
    Formatter for Alpaca/Stanford format.

    Outputs data in the Alpaca format:
    {"instruction": "...", "input": "...", "output": "..."}
    """

    @property
    def format_name(self) -> str:
        return "alpaca"

    def format(self, item: Dict[str, Any], **kwargs: Any) -> Optional[Dict[str, Any]]:
        """
        Format item for Alpaca format.

        Args:
            item: The extracted data item.
            **kwargs: Additional options.

        Returns:
            Formatted dictionary with instruction/input/output.
        """
        # Already in Alpaca format
        if "instruction" in item and "output" in item:
            return {
                "instruction": item["instruction"],
                "input": item.get("input", ""),
                "output": item["output"],
            }

        # Q&A format
        if "question" in item and "answer" in item:
            return {
                "instruction": item["question"],
                "input": "",
                "output": item["answer"],
            }

        # Rules format
        if "rule" in item:
            context = item.get("context", "")
            instruction = f"What is the rule for: {context}" if context else "Explain the following rule."
            output = item["rule"]
            if item.get("rationale"):
                output += f"\n\nRationale: {item['rationale']}"
            return {
                "instruction": instruction,
                "input": "",
                "output": output,
            }

        # Facts format
        if "fact" in item:
            topic = item.get("topic", "this topic")
            return {
                "instruction": f"Tell me a fact about {topic}.",
                "input": "",
                "output": item["fact"],
            }

        # Summary format
        if "summary" in item:
            return {
                "instruction": "Summarize the following text.",
                "input": item.get("original_text", ""),
                "output": item["summary"],
            }

        # Conversation - take first exchange
        if "conversation" in item and len(item["conversation"]) >= 2:
            conv = item["conversation"]
            user_msg = next((t for t in conv if t["role"] == "user"), None)
            asst_msg = next((t for t in conv if t["role"] == "assistant"), None)

            if user_msg and asst_msg:
                return {
                    "instruction": user_msg["content"],
                    "input": "",
                    "output": asst_msg["content"],
                }

        return None


class ShareGPTFormatter(BaseFormatter):
    """
    Formatter for ShareGPT format.

    Outputs data in ShareGPT conversation format:
    {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
    """

    @property
    def format_name(self) -> str:
        return "sharegpt"

    def format(self, item: Dict[str, Any], **kwargs: Any) -> Optional[Dict[str, Any]]:
        """
        Format item for ShareGPT format.

        Args:
            item: The extracted data item.
            **kwargs: Additional options.

        Returns:
            Formatted dictionary with conversations.
        """
        conversations = []

        # Conversation format - direct mapping
        if "conversation" in item:
            for turn in item["conversation"]:
                role = turn["role"]
                from_field = "human" if role == "user" else "gpt"
                conversations.append({
                    "from": from_field,
                    "value": turn["content"],
                })

        # Q&A format
        elif "question" in item and "answer" in item:
            conversations.append({
                "from": "human",
                "value": item["question"],
            })
            conversations.append({
                "from": "gpt",
                "value": item["answer"],
            })

        # Instruction format
        elif "instruction" in item and "output" in item:
            user_content = item["instruction"]
            if item.get("input"):
                user_content += f"\n\n{item['input']}"

            conversations.append({
                "from": "human",
                "value": user_content,
            })
            conversations.append({
                "from": "gpt",
                "value": item["output"],
            })

        # Rules format
        elif "rule" in item:
            context = item.get("context", "")
            question = f"What is the rule for {context}?" if context else "What is the rule?"
            conversations.append({
                "from": "human",
                "value": question,
            })
            answer = item["rule"]
            if item.get("rationale"):
                answer += f"\n\nRationale: {item['rationale']}"
            conversations.append({
                "from": "gpt",
                "value": answer,
            })

        # Facts format
        elif "fact" in item:
            topic = item.get("topic", "this topic")
            conversations.append({
                "from": "human",
                "value": f"Tell me a fact about {topic}.",
            })
            conversations.append({
                "from": "gpt",
                "value": item["fact"],
            })

        else:
            return None

        if len(conversations) < 2:
            return None

        return {"conversations": conversations}


class GenericFormatter(BaseFormatter):
    """
    Generic formatter that passes through data as-is.

    Useful for custom formats or when no transformation is needed.
    """

    @property
    def format_name(self) -> str:
        return "generic"

    def format(
        self,
        item: Dict[str, Any],
        include_metadata: bool = False,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Format item by passing through as-is.

        Args:
            item: The extracted data item.
            include_metadata: Whether to include internal metadata keys.
            **kwargs: Ignored.

        Returns:
            The item dictionary.
        """
        if include_metadata:
            return item

        # Remove internal metadata keys
        return {k: v for k, v in item.items() if not k.startswith("_")}


class LlamaFactoryFormatter(BaseFormatter):
    """
    Formatter for LlamaFactory format.

    Outputs data compatible with LlamaFactory training:
    {"instruction": "...", "input": "...", "output": "...", "history": [...]}
    """

    @property
    def format_name(self) -> str:
        return "llamafactory"

    def format(self, item: Dict[str, Any], **kwargs: Any) -> Optional[Dict[str, Any]]:
        """Format for LlamaFactory."""
        # Multi-turn conversation
        if "conversation" in item:
            conv = item["conversation"]
            if len(conv) < 2:
                return None

            # Last exchange is instruction/output, rest is history
            history = []
            for i in range(0, len(conv) - 2, 2):
                if i + 1 < len(conv) - 2:
                    history.append([conv[i]["content"], conv[i + 1]["content"]])

            last_user = conv[-2]["content"] if conv[-2]["role"] == "user" else conv[-1]["content"]
            last_asst = conv[-1]["content"] if conv[-1]["role"] == "assistant" else conv[-2]["content"]

            return {
                "instruction": last_user,
                "input": "",
                "output": last_asst,
                "history": history if history else [],
            }

        # Single turn - delegate to Alpaca
        alpaca = AlpacaFormatter()
        result = alpaca.format(item, **kwargs)
        if result:
            result["history"] = []
        return result


class ChatMLFormatter(BaseFormatter):
    """
    Formatter for ChatML format.

    Outputs data in ChatML template format.
    """

    @property
    def format_name(self) -> str:
        return "chatml"

    def format(
        self,
        item: Dict[str, Any],
        system_prompt: str = "You are a helpful assistant.",
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Format for ChatML."""
        messages = []

        # Add system
        messages.append({
            "role": "system",
            "content": system_prompt,
        })

        # Handle different formats
        if "conversation" in item:
            for turn in item["conversation"]:
                messages.append({
                    "role": turn["role"],
                    "content": turn["content"],
                })
        elif "question" in item and "answer" in item:
            messages.append({"role": "user", "content": item["question"]})
            messages.append({"role": "assistant", "content": item["answer"]})
        elif "instruction" in item and "output" in item:
            user_content = item["instruction"]
            if item.get("input"):
                user_content += f"\n\n{item['input']}"
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": item["output"]})
        else:
            return None

        # Build ChatML string
        chatml = ""
        for msg in messages:
            chatml += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"

        return {"text": chatml}


# Formatter registry
_FORMATTERS = {
    "openai": OpenAIFormatter,
    "alpaca": AlpacaFormatter,
    "sharegpt": ShareGPTFormatter,
    "generic": GenericFormatter,
    "llamafactory": LlamaFactoryFormatter,
    "chatml": ChatMLFormatter,
}


def get_formatter(format_name: str) -> BaseFormatter:
    """
    Get a formatter by name.

    Args:
        format_name: Name of the format.

    Returns:
        Formatter instance.

    Raises:
        ValueError: If format is unknown.
    """
    if format_name not in _FORMATTERS:
        raise ValueError(
            f"Unknown format: {format_name}. "
            f"Available: {list(_FORMATTERS.keys())}"
        )

    return _FORMATTERS[format_name]()


def register_formatter(name: str, formatter_class: type) -> None:
    """
    Register a custom formatter.

    Args:
        name: Name for the formatter.
        formatter_class: Formatter class to register.
    """
    _FORMATTERS[name] = formatter_class


def list_formats() -> List[str]:
    """List available format names."""
    return list(_FORMATTERS.keys())

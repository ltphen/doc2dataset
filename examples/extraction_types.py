#!/usr/bin/env python3
"""
Example showing different extraction types.

doc2dataset supports various extraction strategies depending
on your fine-tuning goals.
"""

from doc2dataset import DocProcessor, Dataset


# Sample document content
SAMPLE_CONTENT = """
# Company Security Policy

## Password Requirements
All employees must use passwords that are at least 12 characters long.
Passwords must include uppercase letters, lowercase letters, numbers, and symbols.
Passwords must be changed every 90 days.

## Data Handling
Confidential data must never be shared via email.
All customer data must be encrypted at rest and in transit.
Access to production databases requires manager approval.

## Remote Work
Employees may work remotely up to 3 days per week.
VPN must be used when accessing company resources remotely.
Personal devices must have endpoint security software installed.

## Incident Response
All security incidents must be reported within 24 hours.
Contact the security team at security@company.com.
Document all steps taken during incident resolution.
"""


def example_qa_extraction():
    """Extract question-answer pairs."""
    print("=" * 60)
    print("Q&A Extraction")
    print("=" * 60)

    processor = DocProcessor(
        provider="openai",
        model="gpt-4",
        extractors=["qa"],
    )

    dataset = processor.process_text(SAMPLE_CONTENT, source="security_policy.md")

    print(f"\nExtracted {len(dataset)} Q&A pairs:\n")

    for i, item in enumerate(dataset, 1):
        print(f"{i}. Q: {item.data.get('question', 'N/A')}")
        print(f"   A: {item.data.get('answer', 'N/A')[:100]}...")
        print()


def example_rules_extraction():
    """Extract rules and guidelines."""
    print("=" * 60)
    print("Rules Extraction")
    print("=" * 60)

    processor = DocProcessor(
        provider="openai",
        model="gpt-4",
        extractors=["rules"],
    )

    dataset = processor.process_text(SAMPLE_CONTENT, source="security_policy.md")

    print(f"\nExtracted {len(dataset)} rules:\n")

    for i, item in enumerate(dataset, 1):
        print(f"{i}. Rule: {item.data.get('rule', 'N/A')}")
        print(f"   Context: {item.data.get('context', 'N/A')}")
        print(f"   Category: {item.data.get('category', 'N/A')}")
        print()


def example_facts_extraction():
    """Extract factual statements."""
    print("=" * 60)
    print("Facts Extraction")
    print("=" * 60)

    processor = DocProcessor(
        provider="openai",
        model="gpt-4",
        extractors=["facts"],
    )

    dataset = processor.process_text(SAMPLE_CONTENT, source="security_policy.md")

    print(f"\nExtracted {len(dataset)} facts:\n")

    for i, item in enumerate(dataset, 1):
        print(f"{i}. Fact: {item.data.get('fact', 'N/A')}")
        print(f"   Topic: {item.data.get('topic', 'N/A')}")
        print(f"   Confidence: {item.data.get('confidence', 'N/A')}")
        print()


def example_instruction_extraction():
    """Extract instruction-response pairs."""
    print("=" * 60)
    print("Instruction Extraction")
    print("=" * 60)

    processor = DocProcessor(
        provider="openai",
        model="gpt-4",
        extractors=["instruction"],
    )

    dataset = processor.process_text(SAMPLE_CONTENT, source="security_policy.md")

    print(f"\nExtracted {len(dataset)} instructions:\n")

    for i, item in enumerate(dataset, 1):
        print(f"{i}. Instruction: {item.data.get('instruction', 'N/A')}")
        print(f"   Output: {item.data.get('output', 'N/A')[:100]}...")
        print()


def example_conversation_extraction():
    """Extract multi-turn conversations."""
    print("=" * 60)
    print("Conversation Extraction")
    print("=" * 60)

    processor = DocProcessor(
        provider="openai",
        model="gpt-4",
        extractors=["conversation"],
    )

    dataset = processor.process_text(SAMPLE_CONTENT, source="security_policy.md")

    print(f"\nExtracted {len(dataset)} conversations:\n")

    for i, item in enumerate(dataset, 1):
        print(f"Conversation {i}:")
        for turn in item.data.get("conversation", []):
            role = turn.get("role", "unknown")
            content = turn.get("content", "")[:80]
            print(f"  [{role}]: {content}...")
        print()


def example_combined_extraction():
    """Extract multiple types at once."""
    print("=" * 60)
    print("Combined Extraction (All Types)")
    print("=" * 60)

    processor = DocProcessor(
        provider="openai",
        model="gpt-4",
        extractors=["qa", "rules", "facts", "instruction"],
    )

    dataset = processor.process_text(SAMPLE_CONTENT, source="security_policy.md")

    print(f"\nExtracted {len(dataset)} total items:")
    stats = dataset.statistics()
    for ext_type, count in stats["extractor_types"].items():
        print(f"  - {ext_type}: {count} items")

    # Filter by type
    qa_only = dataset.filter_by_type("qa")
    rules_only = dataset.filter_by_type("rules")

    print(f"\nQ&A items: {len(qa_only)}")
    print(f"Rules items: {len(rules_only)}")

    # Export each type separately
    qa_only.to_jsonl("qa_data.jsonl", format="openai")
    rules_only.to_jsonl("rules_data.jsonl", format="alpaca")

    # Or export all together
    dataset.to_jsonl("all_data.jsonl", format="openai")


def main():
    """Run all examples."""
    # Uncomment the examples you want to run

    # example_qa_extraction()
    # example_rules_extraction()
    # example_facts_extraction()
    # example_instruction_extraction()
    # example_conversation_extraction()
    example_combined_extraction()


if __name__ == "__main__":
    main()

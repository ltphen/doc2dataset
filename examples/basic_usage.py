#!/usr/bin/env python3
"""
Basic usage example for doc2dataset.

This example demonstrates how to convert documents into
fine-tuning training data.
"""

from doc2dataset import DocProcessor, Dataset


def main():
    # Initialize the processor with OpenAI
    # (requires OPENAI_API_KEY environment variable)
    processor = DocProcessor(
        provider="openai",
        model="gpt-4",
        extractors=["qa", "rules", "facts"],  # Types of data to extract
        chunk_size=3000,  # Characters per chunk
        max_items_per_chunk=10,  # Items to extract per chunk
    )

    # Process a single document
    # The processor automatically:
    # 1. Loads the document
    # 2. Chunks it into processable pieces
    # 3. Sends each chunk to the LLM for extraction
    # 4. Parses and validates the responses
    dataset = processor.process_document("./my_document.pdf")

    print(f"Extracted {len(dataset)} items")

    # Or process an entire folder
    dataset = processor.process_folder(
        "./documents",
        recursive=True,  # Search subdirectories
        extensions=[".pdf", ".txt", ".md"],  # File types to include
        progress=True,  # Show progress bar
    )

    print(f"Extracted {len(dataset)} items from folder")

    # View statistics
    stats = dataset.statistics()
    print(f"\nStatistics:")
    print(f"  Total items: {stats['total_items']}")
    print(f"  By extractor: {stats['extractor_types']}")

    # Export to different formats

    # OpenAI fine-tuning format
    dataset.to_jsonl("training_openai.jsonl", format="openai")

    # Alpaca format (instruction/input/output)
    dataset.to_jsonl("training_alpaca.jsonl", format="alpaca")

    # ShareGPT format (conversations)
    dataset.to_jsonl("training_sharegpt.jsonl", format="sharegpt")

    print("\nExported to multiple formats!")

    # You can also split into train/val
    train, val = dataset.split(train_ratio=0.9, seed=42)
    train.to_jsonl("train.jsonl", format="openai")
    val.to_jsonl("val.jsonl", format="openai")

    print(f"Split: {len(train)} train, {len(val)} val")


if __name__ == "__main__":
    main()

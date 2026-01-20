# doc2dataset

**Transform Documents into LLM Fine-tuning Datasets**

doc2dataset is a Python package that converts documents (PDFs, text files, etc.) into high-quality training data for fine-tuning language models. It uses LLMs to intelligently extract knowledge, rules, Q&A pairs, and more from your domain-specific documents.

## Features

- **Multiple Document Formats**: PDF, TXT, Markdown, DOCX, JSON
- **Smart Extraction**: Uses LLMs to extract structured knowledge
- **Multiple Extraction Types**: Q&A pairs, rules, facts, instructions, conversations
- **Flexible Output Formats**: OpenAI, Alpaca, ShareGPT, LlamaFactory, ChatML
- **CLI and Python API**: Use from command line or programmatically
- **Chunking & Batching**: Handles large documents automatically
- **Cost Estimation**: Estimate API costs before processing
- **Checkpointing**: Resume interrupted processing jobs
- **Quality Filtering**: Filter and score extraction quality
- **Source Attribution**: Track data lineage back to source documents
- **Parallel Processing**: Process documents in parallel with rate limiting
- **Analytics**: Built-in metrics and reporting for processed datasets

## Installation

```bash
# Basic installation
pip install doc2dataset

# With PDF support
pip install doc2dataset[pdf]

# With all document formats and providers
pip install doc2dataset[all]
```

## Quick Start

### Command Line

```bash
# Process a folder of documents
doc2dataset process ./documents ./training.jsonl

# Use specific extractors
doc2dataset process ./docs ./output.jsonl --extractors qa rules facts

# Use different providers
doc2dataset process ./docs ./output.jsonl --provider anthropic --model claude-3-sonnet-20240229

# Output in Alpaca format
doc2dataset process ./docs ./output.jsonl --format alpaca

# Estimate cost before processing
doc2dataset process ./docs ./output.jsonl --estimate-cost

# Resume interrupted processing
doc2dataset process ./docs ./output.jsonl --resume

# Parallel processing with rate limiting
doc2dataset process ./docs ./output.jsonl --workers 4 --rate-limit 10

# Enable quality filtering
doc2dataset process ./docs ./output.jsonl --quality-filter

# Track source attribution
doc2dataset process ./docs ./output.jsonl --with-attribution

# Analyze existing dataset
doc2dataset analyze ./training.jsonl --format markdown --output report.md

# Manage checkpoints
doc2dataset checkpoints list
doc2dataset checkpoints cleanup --max-age 7

# Upload to HuggingFace Hub
doc2dataset upload ./training.jsonl username/my-dataset
doc2dataset upload ./train.jsonl ./val.jsonl username/my-dataset --splits train validation

# Download from HuggingFace Hub
doc2dataset download username/my-dataset ./data.jsonl
```

### Python API

```python
from doc2dataset import DocProcessor

# Initialize with OpenAI
processor = DocProcessor(provider="openai", model="gpt-4")

# Process a folder of documents
dataset = processor.process_folder("./documents")

# Export for OpenAI fine-tuning
dataset.to_jsonl("training.jsonl", format="openai")

print(f"Created {len(dataset)} training examples")
```

## How It Works

doc2dataset uses a three-stage pipeline:

1. **Load**: Documents are loaded and parsed from various formats
2. **Extract**: LLMs analyze content and extract structured knowledge
3. **Format**: Extracted data is formatted for your fine-tuning framework

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Documents  │───▶│   Loaders   │───▶│  Extractors │───▶│  Formatters │
│ (PDF, TXT)  │    │  (Parse)    │    │  (LLM)      │    │  (JSONL)    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## Extraction Types

### Q&A Pairs
Extract question-answer pairs suitable for instruction tuning:

```json
{"question": "What is the password policy?", "answer": "Passwords must be at least 12 characters..."}
```

### Rules & Guidelines
Extract rules, procedures, and best practices:

```json
{"rule": "Always encrypt customer data", "context": "Data handling", "category": "security"}
```

### Facts
Extract atomic factual statements:

```json
{"fact": "The company was founded in 2020", "topic": "history", "confidence": "high"}
```

### Instructions
Extract instruction-response pairs:

```json
{"instruction": "Explain how to reset a password", "input": "", "output": "To reset your password..."}
```

### Conversations
Extract multi-turn dialogue examples:

```json
{"conversation": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Output Formats

### OpenAI Fine-tuning

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Alpaca

```json
{"instruction": "...", "input": "...", "output": "..."}
```

### ShareGPT

```json
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
```

### LlamaFactory

```json
{"instruction": "...", "input": "...", "output": "...", "history": [...]}
```

## Advanced Usage

### Custom Extractors

```python
from doc2dataset import DocProcessor, CustomExtractor

def my_parser(response):
    # Parse LLM response into list of dicts
    return [{"key": "value"}]

processor = DocProcessor(provider="openai")
processor.add_extractor(
    "custom",
    CustomExtractor(
        llm_fn=processor.llm_fn,
        prompt_template="Extract {max_items} items from:\n{content}",
        parser_fn=my_parser,
    )
)
```

### Dataset Manipulation

```python
from doc2dataset import Dataset

# Load existing dataset
dataset = Dataset.from_jsonl("data.jsonl")

# Filter by extractor type
qa_data = dataset.filter_by_type("qa")

# Deduplicate
dataset = dataset.deduplicate()

# Split into train/val
train, val = dataset.split(train_ratio=0.9, seed=42)

# Sample subset
sample = dataset.sample(1000)

# Merge datasets
combined = dataset1.merge(dataset2)
```

### Using Different Providers

```python
# OpenAI
processor = DocProcessor(provider="openai", model="gpt-4")

# Anthropic
processor = DocProcessor(provider="anthropic", model="claude-3-opus-20240229")

# Any LiteLLM-supported model
processor = DocProcessor(provider="litellm", model="together_ai/llama-3-70b")

# Custom LLM function
def my_llm(prompt: str) -> str:
    return my_api.generate(prompt)

processor = DocProcessor(llm_fn=my_llm)
```

### Cost Estimation

Estimate API costs before processing large document sets.

```python
from doc2dataset import DocProcessor

processor = DocProcessor(provider="openai", model="gpt-4")

# Estimate cost for a folder
estimate = processor.estimate_cost("./documents")
print(f"Estimated tokens: {estimate['total_tokens']:,}")
print(f"Estimated cost: ${estimate['estimated_cost']:.2f}")
print(f"Documents to process: {estimate['document_count']}")

# Decide whether to proceed
if estimate['estimated_cost'] < 50:
    dataset = processor.process_folder("./documents")
```

### Checkpointing & Resumable Processing

Resume interrupted jobs without reprocessing completed documents.

```python
from doc2dataset import DocProcessor

processor = DocProcessor(
    provider="openai",
    enable_checkpointing=True,
    checkpoint_dir="./checkpoints",
)

# Process with automatic checkpointing
dataset = processor.process_folder("./documents")

# If interrupted, resume where you left off
dataset = processor.process_folder("./documents", resume=True)

# List all checkpoints
checkpoints = processor.list_checkpoints()
for cp in checkpoints:
    print(f"Job: {cp['job_id']}, Progress: {cp['processed']}/{cp['total']}")

# Clean up old checkpoints
processor.cleanup_checkpoints(max_age_days=7)
```

### Quality Filtering

Filter and score extracted data for quality.

```python
from doc2dataset import DocProcessor
from doc2dataset.quality import (
    QualityFilterPipeline,
    LengthFilter,
    RepetitionFilter,
    ContentFilter,
    DuplicateFilter,
    QualityScorer,
    get_qa_quality_pipeline,
)

# Use built-in Q&A quality pipeline
processor = DocProcessor(
    provider="openai",
    enable_quality_filter=True,
    quality_pipeline=get_qa_quality_pipeline(),
)

# Or create custom quality pipeline
pipeline = QualityFilterPipeline([
    LengthFilter(field="output", min_length=50, max_length=2000),
    RepetitionFilter(field="output", max_repetition_ratio=0.5),
    ContentFilter(field="output", blocked_patterns=[r"\bTODO\b", r"\bFIXME\b"]),
])

processor = DocProcessor(
    provider="openai",
    quality_pipeline=pipeline,
)

# Process and filter
dataset = processor.process_folder("./documents")

# Check filter statistics
stats = processor.get_quality_filter_stats()
print(f"Total processed: {stats['total_processed']}")
print(f"Passed filters: {stats['total_passed']}")
print(f"Filtered out: {stats['total_filtered']}")

# Score items for quality
scorer = QualityScorer()
for item in dataset:
    item['quality_score'] = scorer.score(item)
```

### Source Attribution

Track data lineage back to source documents.

```python
from doc2dataset import DocProcessor

processor = DocProcessor(
    provider="openai",
    enable_attribution=True,
)

dataset = processor.process_folder("./documents")

# Get attributions
attributions = processor.get_attributions()
for attr in attributions:
    print(f"Item: {attr['item_id']}")
    print(f"  Source: {attr['source_file']}")
    print(f"  Page: {attr['page_number']}")
    print(f"  Chunk: {attr['chunk_index']}")

# Save attributions to file
processor.save_attributions("attributions.json")

# Items include attribution metadata
for item in dataset:
    print(f"Source: {item['_attribution']['source']}")
```

### Parallel Processing

Process documents in parallel with rate limiting.

```python
from doc2dataset import DocProcessor

processor = DocProcessor(
    provider="openai",
    workers=4,              # Number of parallel workers
    rate_limit=10,          # Max requests per second
    batch_size=5,           # Documents per batch
)

# Process folder in parallel
dataset = processor.process_folder("./documents")

# For async processing
import asyncio

async def main():
    dataset = await processor.aprocess_folder("./documents")
    return dataset

dataset = asyncio.run(main())
```

### Processing Analytics

Get detailed metrics and reports for processed datasets.

```python
from doc2dataset import DocProcessor
from doc2dataset.analytics import (
    DatasetAnalyzer,
    ProcessingAnalytics,
    analyze_jsonl_file,
    compare_datasets,
)

processor = DocProcessor(
    provider="openai",
    enable_analytics=True,
)

# Process with analytics tracking
dataset = processor.process_folder("./documents")

# Get processing summary
summary = processor.get_analytics_summary()
print(f"Documents processed: {summary['documents_processed']}")
print(f"Items extracted: {summary['items_extracted']}")
print(f"Processing time: {summary['processing_time']:.1f}s")
print(f"Throughput: {summary['throughput']:.1f} docs/sec")
print(f"Total cost: ${summary.get('total_cost', 0):.2f}")

# Export analytics
processor.export_analytics("analytics.json")

# Analyze existing dataset files
analyzer = DatasetAnalyzer()
stats = analyzer.analyze(dataset)
print(f"Total items: {stats.total_items}")
print(f"Total tokens: {stats.total_tokens}")
print(f"Extraction types: {stats.extraction_types}")

# Generate report
report = analyzer.generate_report(stats, format="markdown")
print(report)

# Analyze a JSONL file directly
stats = analyze_jsonl_file("training.jsonl")

# Compare multiple datasets
comparison = compare_datasets({
    "v1": dataset1,
    "v2": dataset2,
})
print(f"Size ratio: {comparison['_comparison']['size_ratio']}")
```

### HuggingFace Hub Integration

Upload and download datasets from HuggingFace Hub.

```python
from doc2dataset import DocProcessor
from doc2dataset.huggingface import (
    HuggingFaceUploader,
    HuggingFaceDownloader,
    DatasetCard,
    upload_to_hub,
    download_from_hub,
)

# Process documents
processor = DocProcessor(provider="openai")
dataset = processor.process_folder("./documents")

# Upload to HuggingFace Hub (simple)
url = upload_to_hub(
    items=list(dataset),
    repo_id="username/my-dataset",
    name="My Training Dataset",
    description="Fine-tuning data extracted from documents",
)
print(f"Uploaded to: {url}")

# Upload with custom dataset card
uploader = HuggingFaceUploader(token="hf_xxx", private=False)
card = DatasetCard(
    name="username/my-dataset",
    description="Training data for my model",
    language=["en"],
    license="mit",
    tags=["doc2dataset", "fine-tuning", "qa"],
)

url = uploader.upload_dataset(
    items=list(dataset),
    repo_id="username/my-dataset",
    dataset_card=card,
    split="train",
    format="parquet",  # or "jsonl"
)

# Upload multiple splits
train_data, val_data = dataset.split(train_ratio=0.9)
url = uploader.upload_splits(
    splits={
        "train": list(train_data),
        "validation": list(val_data),
    },
    repo_id="username/my-dataset",
    dataset_card=card,
)

# Download from HuggingFace Hub
items = download_from_hub("username/my-dataset", split="train")

# Download to file
download_from_hub(
    "username/my-dataset",
    output_path="./downloaded_data.jsonl",
    split="train",
)

# Use HuggingFace downloader directly
downloader = HuggingFaceDownloader(token="hf_xxx")
ds = downloader.download("username/my-dataset")  # Returns HF Dataset
items = downloader.to_items("username/my-dataset")  # Returns list of dicts
```

### Using ProcessorConfig

For cleaner configuration, use the `ProcessorConfig` dataclass.

```python
from doc2dataset import DocProcessor, ProcessorConfig
from doc2dataset.quality import get_qa_quality_pipeline

config = ProcessorConfig(
    provider="openai",
    model="gpt-4",
    extractors=["qa", "rules"],
    chunk_size=3000,
    # New features
    enable_cost_estimation=True,
    enable_checkpointing=True,
    checkpoint_dir="./checkpoints",
    enable_quality_filter=True,
    quality_pipeline=get_qa_quality_pipeline(),
    enable_attribution=True,
    workers=4,
    rate_limit=10,
    enable_analytics=True,
)

processor = DocProcessor.from_config(config)
dataset = processor.process_folder("./documents")
```

## API Reference

### DocProcessor

```python
DocProcessor(
    provider="openai",           # LLM provider
    model=None,                  # Model name
    extractors=["qa", "rules"],  # Extraction types
    chunk_size=3000,             # Characters per chunk
    max_items_per_chunk=10,      # Items to extract per chunk
    # New parameters
    enable_checkpointing=False,  # Enable checkpoint/resume
    checkpoint_dir=None,         # Directory for checkpoints
    enable_quality_filter=False, # Enable quality filtering
    quality_pipeline=None,       # Custom quality filter pipeline
    enable_attribution=False,    # Track source attribution
    workers=1,                   # Parallel workers
    rate_limit=None,             # Max requests per second
    batch_size=1,                # Documents per batch
    enable_analytics=False,      # Enable processing analytics
)
```

#### Methods

- `process_document(path)` - Process single document
- `process_folder(path, resume=False)` - Process all documents in folder
- `aprocess_folder(path, resume=False)` - Async process folder
- `process_text(text)` - Process raw text
- `add_extractor(name, extractor)` - Add custom extractor
- `estimate_cost(path)` - Estimate API cost for documents
- `get_analytics_summary()` - Get processing analytics
- `export_analytics(path)` - Export analytics to file
- `get_quality_filter_stats()` - Get quality filter statistics
- `get_attributions()` - Get source attributions
- `save_attributions(path)` - Save attributions to file
- `list_checkpoints()` - List all checkpoints
- `cleanup_checkpoints(max_age_days)` - Clean up old checkpoints
- `from_config(config)` - Create instance from ProcessorConfig

### Dataset

```python
Dataset(items=[], metadata={})
```

#### Methods

- `add(data, source, extractor_type)` - Add item
- `filter(predicate)` - Filter items
- `filter_by_type(type)` - Filter by extractor type
- `deduplicate()` - Remove duplicates
- `shuffle(seed)` - Shuffle items
- `split(train_ratio)` - Split into train/val
- `sample(n)` - Random sample
- `merge(other)` - Merge datasets
- `to_jsonl(path, format)` - Export to JSONL
- `to_json(path)` - Export to JSON
- `from_jsonl(path)` - Load from JSONL
- `from_json(path)` - Load from JSON
- `statistics()` - Get statistics

### Quality Filters

```python
from doc2dataset.quality import (
    LengthFilter,
    RepetitionFilter,
    ContentFilter,
    DuplicateFilter,
    QualityFilterPipeline,
    QualityScorer,
)

# Length filter
LengthFilter(field="output", min_length=10, max_length=1000)

# Repetition filter (detect repetitive text)
RepetitionFilter(field="output", max_repetition_ratio=0.5)

# Content filter (block patterns)
ContentFilter(field="output", blocked_patterns=[r"\bTODO\b"])

# Duplicate filter (remove near-duplicates)
DuplicateFilter(field="output", similarity_threshold=0.9)

# Pipeline (combine filters)
QualityFilterPipeline(filters=[...])

# Scorer (score quality 0-1)
QualityScorer()
```

### Analytics

```python
from doc2dataset.analytics import (
    DatasetStats,
    DatasetAnalyzer,
    ProcessingAnalytics,
    analyze_jsonl_file,
    compare_datasets,
)

# DatasetStats - statistics container
stats = DatasetStats(total_items=100, total_tokens=5000, ...)

# DatasetAnalyzer - analyze datasets
analyzer = DatasetAnalyzer()
stats = analyzer.analyze(items)
report = analyzer.generate_report(stats, format="markdown")

# ProcessingAnalytics - track processing metrics
analytics = ProcessingAnalytics()
analytics.start_session()
analytics.record_document_processed(source, items, time, tokens, cost)
summary = analytics.summary()
analytics.export(path)

# Helper functions
stats = analyze_jsonl_file("data.jsonl")
comparison = compare_datasets({"v1": data1, "v2": data2})
```

### Checkpointing

```python
from doc2dataset.checkpoint import CheckpointManager, ProcessingState

# CheckpointManager - manage processing checkpoints
manager = CheckpointManager(checkpoint_dir="./checkpoints")

# Create new job
state = manager.create_job(documents, extraction_type, model)

# Resume existing job
state = manager.create_job(documents, ..., resume_if_exists=True)

# Track progress
manager.mark_complete(state, doc_source, results)
manager.mark_failed(state, doc_source, error_message)

# Iterate pending documents
for doc in manager.iterate_pending(documents, state):
    process(doc)

# Finalize and cleanup
manager.finalize_job(state)
manager.cleanup_completed(max_age_days=7)
```

### Cost Estimation

```python
from doc2dataset.cost import CostEstimator, TokenCounter

# TokenCounter - count tokens for text
counter = TokenCounter(model="gpt-4")
tokens = counter.count("Some text")
tokens = counter.count_messages(messages)

# CostEstimator - estimate API costs
estimator = CostEstimator(model="gpt-4")
estimate = estimator.estimate(documents)
# Returns: {"total_tokens": N, "estimated_cost": $X.XX, ...}
```

### HuggingFace Integration

```python
from doc2dataset.huggingface import (
    HuggingFaceUploader,
    HuggingFaceDownloader,
    DatasetCard,
    upload_to_hub,
    download_from_hub,
)

# HuggingFaceUploader - upload datasets to Hub
uploader = HuggingFaceUploader(token="...", private=False)
uploader.upload_jsonl(file_path, repo_id, dataset_card, split)
uploader.upload_dataset(items, repo_id, dataset_card, split, format)
uploader.upload_splits(splits_dict, repo_id, dataset_card, format)

# HuggingFaceDownloader - download datasets from Hub
downloader = HuggingFaceDownloader(token="...")
ds = downloader.download(repo_id, split, streaming)
items = downloader.to_items(repo_id, split)
downloader.to_jsonl(repo_id, output_path, split)

# DatasetCard - dataset metadata for Hub
card = DatasetCard(
    name="repo/name",
    description="...",
    language=["en"],
    license="mit",
    tags=["doc2dataset"],
)

# Convenience functions
url = upload_to_hub(items, repo_id, token, private, name, description)
items = download_from_hub(repo_id, output_path, split, token)
```

## Supported Document Formats

| Format | Extension | Required Package |
|--------|-----------|-----------------|
| Text | `.txt` | (built-in) |
| Markdown | `.md` | (built-in) |
| PDF | `.pdf` | `pymupdf` |
| Word | `.docx` | `python-docx` |
| JSON | `.json` | (built-in) |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

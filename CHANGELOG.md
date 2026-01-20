# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-19

### Added

- Initial release of doc2dataset
- Document loaders:
  - `PDFLoader` for PDF documents (requires pymupdf)
  - `TextLoader` for plain text files
  - `MarkdownLoader` for markdown files
  - `DOCXLoader` for Word documents (requires python-docx)
  - `JSONLoader` for JSON files
- Knowledge extractors:
  - `QAExtractor` for question-answer pairs
  - `RulesExtractor` for rules and guidelines
  - `FactsExtractor` for factual statements
  - `InstructionsExtractor` for how-to content
  - `ConversationsExtractor` for dialogue data
- Output formatters:
  - `OpenAIFormatter` for OpenAI fine-tuning format
  - `AlpacaFormatter` for Alpaca format
  - `ShareGPTFormatter` for ShareGPT format
  - `ChatMLFormatter` for ChatML format
  - `LlamaFactoryFormatter` for LlamaFactory
- Main `DocProcessor` class with integrated features:
  - Cost estimation before processing
  - Checkpointing for resumable processing
  - Quality filtering pipeline
  - Source attribution tracking
  - Parallel processing support
  - Processing analytics
- Quality filtering:
  - `LengthFilter` for min/max length constraints
  - `RepetitionFilter` for detecting repetitive content
  - `ContentFilter` for content validation
  - `DuplicateFilter` for removing duplicates
  - `QualityFilterPipeline` for combining filters
- Cost estimation:
  - `TokenCounter` for accurate token counting
  - `CostEstimator` for pre-processing cost estimates
  - Support for multiple LLM pricing models
- Checkpointing:
  - `CheckpointManager` for state persistence
  - Automatic resume on failure
  - Progress tracking
- Attribution:
  - `SourceTracker` for tracking source locations
  - `AttributionManager` for managing attributions
  - `AttributedDataset` for datasets with provenance
- Parallel processing:
  - `ThreadPoolProcessor` for I/O-bound tasks
  - `ProcessPoolProcessor` for CPU-bound tasks
  - `AsyncProcessor` for async operations
  - `BatchingProcessor` for batch API calls
  - `PipelineProcessor` for multi-stage processing
- Data augmentation:
  - `ParaphraseAugmenter` for paraphrasing
  - `QuestionVariationAugmenter` for Q&A data
  - `BackTranslationAugmenter` for translation-based augmentation
  - `SynonymReplacementAugmenter` (no LLM required)
  - `NegationAugmenter` for contrastive data
  - `InstructionStyleAugmenter` for style variations
- Analytics:
  - `DatasetAnalyzer` for dataset statistics
  - `ProcessingAnalytics` for processing metrics
- HuggingFace Hub integration:
  - `HuggingFaceUploader` for uploading datasets
  - `HuggingFaceDownloader` for downloading datasets
  - Dataset card generation
- Command-line interface:
  - `process` command for document processing
  - `analyze` command for dataset analysis
  - `augment` command for data augmentation
  - `checkpoints` command for checkpoint management
  - `upload` command for HuggingFace Hub upload
  - `download` command for HuggingFace Hub download
- `ProcessorConfig` dataclass for simplified configuration
- Comprehensive test suite
- Full documentation in README

### Security

- All API keys handled via environment variables
- No credentials stored in code or configuration files
- Sensitive data (like documents) processed locally

[Unreleased]: https://github.com/ltphen/doc2dataset/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ltphen/doc2dataset/releases/tag/v0.1.0

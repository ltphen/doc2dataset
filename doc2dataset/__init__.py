"""
doc2dataset - Transform Documents into LLM Fine-tuning Datasets

A Python package for converting documents (PDFs, text files, etc.) into
high-quality training data for fine-tuning language models. Uses LLMs to
extract knowledge, rules, and Q&A pairs from your domain-specific documents.

Example:
    >>> from doc2dataset import DocProcessor
    >>>
    >>> # Process a folder of PDFs
    >>> processor = DocProcessor(provider="openai", model="gpt-4")
    >>> dataset = processor.process_folder("./documents")
    >>>
    >>> # Export as JSONL for fine-tuning
    >>> dataset.to_jsonl("training_data.jsonl", format="openai")
"""

__version__ = "0.1.0"

# Core loaders
from doc2dataset.loaders import (
    Document,
    BaseLoader,
    PDFLoader,
    TextLoader,
    DocxLoader,
    MarkdownLoader,
    load_document,
    load_folder,
)

# Extractors
from doc2dataset.extractors import (
    BaseExtractor,
    RulesExtractor,
    QAExtractor,
    FactsExtractor,
    InstructionExtractor,
    ConversationExtractor,
)

# Formatters
from doc2dataset.formatters import (
    BaseFormatter,
    OpenAIFormatter,
    AlpacaFormatter,
    ShareGPTFormatter,
    GenericFormatter,
)

# Main processor and dataset
from doc2dataset.processor import DocProcessor
from doc2dataset.dataset import Dataset, DatasetItem

# Cost estimation
from doc2dataset.cost import (
    CostEstimator,
    CostEstimate,
    TokenCounts,
    estimate_processing_cost,
)

# Checkpointing and resumable processing
from doc2dataset.checkpoint import (
    CheckpointManager,
    ProcessingState,
    ResumableProcessor,
)

# Quality filtering
from doc2dataset.quality import (
    QualityScorer,
    QualityScore,
    QualityFilterPipeline,
    BaseFilter,
    LengthFilter,
    TokenLengthFilter,
    RepetitionFilter,
    ContentFilter,
    DuplicateFilter,
    QASpecificFilter,
    get_qa_quality_pipeline,
    get_instruction_quality_pipeline,
)

# Source attribution
from doc2dataset.attribution import (
    Attribution,
    SourceLocation,
    SourceTracker,
    AttributionManager,
    AttributedDataset,
)

# Parallel processing
from doc2dataset.parallel import (
    ThreadPoolProcessor,
    ProcessPoolProcessor,
    AsyncProcessor,
    BatchingProcessor,
    PipelineProcessor,
    ProcessingResult,
    BatchResult,
    parallel_map,
)

# Data augmentation
from doc2dataset.augmentation import (
    DataAugmenter,
    BaseAugmenter,
    ParaphraseAugmenter,
    QuestionVariationAugmenter,
    BackTranslationAugmenter,
    SynonymReplacementAugmenter,
    NegationAugmenter,
    InstructionStyleAugmenter,
    AugmentedItem,
    create_default_augmenter,
    create_qa_augmenter,
)

# Analytics
from doc2dataset.analytics import (
    DatasetAnalyzer,
    DatasetStats,
    ProcessingAnalytics,
    analyze_jsonl_file,
    compare_datasets,
)

# HuggingFace integration (optional, check if available)
try:
    from doc2dataset.huggingface import (
        HuggingFaceUploader,
        HuggingFaceDownloader,
        DatasetCard,
        upload_to_hub,
        download_from_hub,
    )
    _HF_EXPORTS = [
        "HuggingFaceUploader",
        "HuggingFaceDownloader",
        "DatasetCard",
        "upload_to_hub",
        "download_from_hub",
    ]
except ImportError:
    _HF_EXPORTS = []

__all__ = [
    # Version
    "__version__",
    # Main interface
    "DocProcessor",
    "Dataset",
    "DatasetItem",
    "Document",
    # Loaders
    "BaseLoader",
    "PDFLoader",
    "TextLoader",
    "DocxLoader",
    "MarkdownLoader",
    "load_document",
    "load_folder",
    # Extractors
    "BaseExtractor",
    "RulesExtractor",
    "QAExtractor",
    "FactsExtractor",
    "InstructionExtractor",
    "ConversationExtractor",
    # Formatters
    "BaseFormatter",
    "OpenAIFormatter",
    "AlpacaFormatter",
    "ShareGPTFormatter",
    "GenericFormatter",
    # Cost estimation
    "CostEstimator",
    "CostEstimate",
    "TokenCounts",
    "estimate_processing_cost",
    # Checkpointing
    "CheckpointManager",
    "ProcessingState",
    "ResumableProcessor",
    # Quality filtering
    "QualityScorer",
    "QualityScore",
    "QualityFilterPipeline",
    "BaseFilter",
    "LengthFilter",
    "TokenLengthFilter",
    "RepetitionFilter",
    "ContentFilter",
    "DuplicateFilter",
    "QASpecificFilter",
    "get_qa_quality_pipeline",
    "get_instruction_quality_pipeline",
    # Attribution
    "Attribution",
    "SourceLocation",
    "SourceTracker",
    "AttributionManager",
    "AttributedDataset",
    # Parallel processing
    "ThreadPoolProcessor",
    "ProcessPoolProcessor",
    "AsyncProcessor",
    "BatchingProcessor",
    "PipelineProcessor",
    "ProcessingResult",
    "BatchResult",
    "parallel_map",
    # Augmentation
    "DataAugmenter",
    "BaseAugmenter",
    "ParaphraseAugmenter",
    "QuestionVariationAugmenter",
    "BackTranslationAugmenter",
    "SynonymReplacementAugmenter",
    "NegationAugmenter",
    "InstructionStyleAugmenter",
    "AugmentedItem",
    "create_default_augmenter",
    "create_qa_augmenter",
    # Analytics
    "DatasetAnalyzer",
    "DatasetStats",
    "ProcessingAnalytics",
    "analyze_jsonl_file",
    "compare_datasets",
] + _HF_EXPORTS

"""
Main document processor for doc2dataset.

This module provides the main DocProcessor class that orchestrates
the document loading, extraction, and dataset creation pipeline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from tqdm import tqdm

from doc2dataset.dataset import Dataset, DatasetItem
from doc2dataset.extractors import (
    BaseExtractor,
    ExtractionResult,
    QAExtractor,
    RulesExtractor,
    FactsExtractor,
    InstructionExtractor,
    ConversationExtractor,
    get_extractor,
)
from doc2dataset.loaders import Document, load_document, load_folder

# New module imports
from doc2dataset.cost import CostEstimator, CostEstimate
from doc2dataset.checkpoint import CheckpointManager, ProcessingState
from doc2dataset.quality import QualityFilterPipeline, QualityScorer, get_qa_quality_pipeline
from doc2dataset.attribution import AttributionManager, Attribution, SourceLocation
from doc2dataset.parallel import ThreadPoolProcessor, BatchResult, ProcessingResult
from doc2dataset.analytics import ProcessingAnalytics, DatasetAnalyzer, DatasetStats

logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """
    Configuration for DocProcessor.

    Simplifies setup by grouping all configuration options.

    Attributes:
        provider: LLM provider name.
        model: Model name.
        api_key: API key for provider.
        extractors: List of extractor types.
        chunk_size: Maximum characters per chunk.
        max_items_per_chunk: Maximum items to extract per chunk.
        enable_cost_estimation: Show cost estimate before processing.
        enable_checkpointing: Enable resumable processing.
        checkpoint_dir: Directory for checkpoint files.
        enable_quality_filter: Apply quality filtering to results.
        quality_pipeline: Custom quality filter pipeline.
        enable_attribution: Track source attribution.
        enable_parallel: Use parallel processing.
        max_workers: Maximum parallel workers.
        rate_limit: Rate limit for API calls (requests/second).
        enable_analytics: Track processing analytics.
    """
    provider: str = "openai"
    model: Optional[str] = None
    api_key: Optional[str] = None
    extractors: Optional[List[str]] = None
    chunk_size: int = 3000
    max_items_per_chunk: int = 10

    # Cost estimation
    enable_cost_estimation: bool = True

    # Checkpointing
    enable_checkpointing: bool = True
    checkpoint_dir: str = "./.doc2dataset_checkpoints"

    # Quality filtering
    enable_quality_filter: bool = True
    quality_pipeline: Optional[QualityFilterPipeline] = None

    # Attribution
    enable_attribution: bool = True

    # Parallel processing
    enable_parallel: bool = True
    max_workers: int = 5
    rate_limit: Optional[float] = None  # Requests per second

    # Analytics
    enable_analytics: bool = True


class DocProcessor:
    """
    Main processor for converting documents to training datasets.

    DocProcessor orchestrates the entire pipeline:
    1. Load documents from various formats
    2. Extract knowledge using LLM-powered extractors
    3. Format and export as training data

    Attributes:
        llm_fn: Function for LLM calls.
        extractors: List of extractors to use.
        default_extractor: Default extraction type.

    Example:
        >>> processor = DocProcessor(provider="openai", model="gpt-4")
        >>> dataset = processor.process_folder("./documents")
        >>> dataset.to_jsonl("training.jsonl", format="openai")
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        extractors: Optional[List[str]] = None,
        chunk_size: int = 3000,
        max_items_per_chunk: int = 10,
        # New parameters for integrated modules
        enable_cost_estimation: bool = True,
        enable_checkpointing: bool = True,
        checkpoint_dir: str = "./.doc2dataset_checkpoints",
        enable_quality_filter: bool = True,
        quality_pipeline: Optional[QualityFilterPipeline] = None,
        enable_attribution: bool = True,
        enable_parallel: bool = True,
        max_workers: int = 5,
        rate_limit: Optional[float] = None,
        enable_analytics: bool = True,
        config: Optional[ProcessorConfig] = None,
        **provider_kwargs: Any,
    ) -> None:
        """
        Initialize the document processor.

        Args:
            provider: LLM provider ("openai", "anthropic", "litellm").
            model: Model name to use.
            api_key: API key for the provider.
            llm_fn: Custom LLM function (overrides provider settings).
            extractors: List of extractor types to use.
            chunk_size: Maximum characters per chunk.
            max_items_per_chunk: Maximum items to extract per chunk.
            enable_cost_estimation: Show cost estimate before processing.
            enable_checkpointing: Enable resumable processing.
            checkpoint_dir: Directory for checkpoint files.
            enable_quality_filter: Apply quality filtering.
            quality_pipeline: Custom quality filter pipeline.
            enable_attribution: Track source attribution.
            enable_parallel: Use parallel processing.
            max_workers: Maximum parallel workers.
            rate_limit: Rate limit for API calls.
            enable_analytics: Track processing analytics.
            config: ProcessorConfig instance (overrides individual params).
            **provider_kwargs: Additional provider arguments.
        """
        # Apply config if provided
        if config:
            provider = config.provider
            model = config.model
            api_key = config.api_key
            extractors = config.extractors
            chunk_size = config.chunk_size
            max_items_per_chunk = config.max_items_per_chunk
            enable_cost_estimation = config.enable_cost_estimation
            enable_checkpointing = config.enable_checkpointing
            checkpoint_dir = config.checkpoint_dir
            enable_quality_filter = config.enable_quality_filter
            quality_pipeline = config.quality_pipeline
            enable_attribution = config.enable_attribution
            enable_parallel = config.enable_parallel
            max_workers = config.max_workers
            rate_limit = config.rate_limit
            enable_analytics = config.enable_analytics

        # Store model for cost estimation
        self._model = model or "gpt-4o"
        self._provider = provider

        # Set up LLM function
        if llm_fn:
            self.llm_fn = llm_fn
        else:
            self.llm_fn = self._create_llm_fn(
                provider=provider,
                model=model,
                api_key=api_key,
                **provider_kwargs,
            )

        self.chunk_size = chunk_size
        self.max_items_per_chunk = max_items_per_chunk

        # Default extractors
        self.extractor_types = extractors or ["qa", "rules", "facts"]

        # Create extractor instances
        self._extractors: Dict[str, BaseExtractor] = {}
        for ext_type in self.extractor_types:
            self._extractors[ext_type] = get_extractor(
                ext_type,
                self.llm_fn,
                chunk_size=chunk_size,
                max_items_per_chunk=max_items_per_chunk,
            )

        # ========== NEW MODULE INTEGRATIONS ==========

        # Cost estimation
        self._enable_cost_estimation = enable_cost_estimation
        self._cost_estimator: Optional[CostEstimator] = None
        if enable_cost_estimation:
            self._cost_estimator = CostEstimator(model=self._model)

        # Checkpointing
        self._enable_checkpointing = enable_checkpointing
        self._checkpoint_manager: Optional[CheckpointManager] = None
        if enable_checkpointing:
            self._checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

        # Quality filtering
        self._enable_quality_filter = enable_quality_filter
        self._quality_pipeline: Optional[QualityFilterPipeline] = quality_pipeline
        if enable_quality_filter and quality_pipeline is None:
            self._quality_pipeline = get_qa_quality_pipeline()

        # Attribution
        self._enable_attribution = enable_attribution
        self._attribution_manager: Optional[AttributionManager] = None
        if enable_attribution:
            self._attribution_manager = AttributionManager()

        # Parallel processing
        self._enable_parallel = enable_parallel
        self._parallel_processor: Optional[ThreadPoolProcessor] = None
        if enable_parallel:
            self._parallel_processor = ThreadPoolProcessor(
                max_workers=max_workers,
                rate_limit=rate_limit,
            )

        # Analytics
        self._enable_analytics = enable_analytics
        self._analytics: Optional[ProcessingAnalytics] = None
        if enable_analytics:
            self._analytics = ProcessingAnalytics()

    def _create_llm_fn(
        self,
        provider: str,
        model: Optional[str],
        api_key: Optional[str],
        **kwargs: Any,
    ) -> Callable[[str], str]:
        """Create LLM function from provider settings."""

        if provider == "openai":
            return self._create_openai_fn(model, api_key, **kwargs)
        elif provider == "anthropic":
            return self._create_anthropic_fn(model, api_key, **kwargs)
        elif provider == "litellm":
            return self._create_litellm_fn(model, api_key, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _create_openai_fn(
        self,
        model: Optional[str],
        api_key: Optional[str],
        **kwargs: Any,
    ) -> Callable[[str], str]:
        """Create OpenAI LLM function."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. "
                "Install with: pip install doc2dataset[openai]"
            )

        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key

        client = OpenAI(**client_kwargs)
        model = model or "gpt-4"

        def llm_fn(prompt: str) -> str:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                **kwargs,
            )
            return response.choices[0].message.content or ""

        return llm_fn

    def _create_anthropic_fn(
        self,
        model: Optional[str],
        api_key: Optional[str],
        **kwargs: Any,
    ) -> Callable[[str], str]:
        """Create Anthropic LLM function."""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package not installed. "
                "Install with: pip install doc2dataset[anthropic]"
            )

        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key

        client = anthropic.Anthropic(**client_kwargs)
        model = model or "claude-3-sonnet-20240229"

        def llm_fn(prompt: str) -> str:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return response.content[0].text if response.content else ""

        return llm_fn

    def _create_litellm_fn(
        self,
        model: Optional[str],
        api_key: Optional[str],
        **kwargs: Any,
    ) -> Callable[[str], str]:
        """Create LiteLLM function."""
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "LiteLLM package not installed. "
                "Install with: pip install doc2dataset[litellm]"
            )

        model = model or "gpt-4"

        def llm_fn(prompt: str) -> str:
            api_kwargs = {"model": model, **kwargs}
            if api_key:
                api_kwargs["api_key"] = api_key

            response = litellm.completion(
                messages=[{"role": "user", "content": prompt}],
                **api_kwargs,
            )
            return response.choices[0].message.content or ""

        return llm_fn

    def process_document(
        self,
        document: Union[str, Path, Document],
        extractor_types: Optional[List[str]] = None,
        progress: bool = False,
    ) -> Dataset:
        """
        Process a single document.

        Args:
            document: Document path or Document object.
            extractor_types: Specific extractors to use (defaults to all).
            progress: Show progress bar.

        Returns:
            Dataset with extracted items.
        """
        # Load document if path
        if isinstance(document, (str, Path)):
            document = load_document(document)

        dataset = Dataset()
        types_to_use = extractor_types or list(self._extractors.keys())

        # Ensure extractors exist
        for ext_type in types_to_use:
            if ext_type not in self._extractors:
                self._extractors[ext_type] = get_extractor(
                    ext_type,
                    self.llm_fn,
                    chunk_size=self.chunk_size,
                    max_items_per_chunk=self.max_items_per_chunk,
                )

        # Run each extractor
        iterator = types_to_use
        if progress:
            iterator = tqdm(iterator, desc="Extractors")

        for ext_type in iterator:
            extractor = self._extractors[ext_type]
            try:
                result = extractor.extract(document)
                for item in result.items:
                    dataset.add(
                        data=item,
                        source=document.source,
                        extractor_type=ext_type,
                    )
            except Exception as e:
                logger.warning(f"Extraction failed for {ext_type}: {e}")

        return dataset

    def process_folder(
        self,
        folder: Union[str, Path],
        extractor_types: Optional[List[str]] = None,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
        progress: bool = True,
        estimate_cost: bool = True,
        confirm_cost: bool = True,
        resume: bool = True,
        apply_quality_filter: bool = True,
    ) -> Dataset:
        """
        Process all documents in a folder.

        Args:
            folder: Path to the folder.
            extractor_types: Specific extractors to use.
            recursive: Search subdirectories.
            extensions: File extensions to include.
            progress: Show progress bar.
            estimate_cost: Show cost estimate before processing.
            confirm_cost: Ask for confirmation before proceeding.
            resume: Resume from checkpoint if available.
            apply_quality_filter: Apply quality filtering to results.

        Returns:
            Dataset with all extracted items.
        """
        folder = Path(folder)
        dataset = Dataset()
        types_to_use = extractor_types or list(self._extractors.keys())

        # Get all documents
        documents = list(load_folder(folder, recursive=recursive, extensions=extensions))

        if not documents:
            logger.warning(f"No documents found in {folder}")
            return dataset

        # ========== COST ESTIMATION ==========
        if estimate_cost and self._enable_cost_estimation and self._cost_estimator:
            estimate = self._cost_estimator.estimate_extraction(
                documents,
                extraction_type=types_to_use[0] if types_to_use else "qa",
            )
            print(self._cost_estimator.format_estimate(estimate, verbose=True))

            if confirm_cost:
                response = input("\nProceed with processing? [y/N]: ")
                if response.lower() != 'y':
                    logger.info("Processing cancelled by user")
                    return dataset

        # ========== START ANALYTICS SESSION ==========
        if self._enable_analytics and self._analytics:
            self._analytics.start_session()

        # ========== CHECKPOINTING SETUP ==========
        state: Optional[ProcessingState] = None
        if resume and self._enable_checkpointing and self._checkpoint_manager:
            state = self._checkpoint_manager.create_job(
                documents,
                extraction_type=",".join(types_to_use),
                model=self._model,
                config={
                    "folder": str(folder),
                    "recursive": recursive,
                    "extensions": extensions,
                },
                resume_if_exists=True,
            )

            if state.processed_documents > 0:
                logger.info(
                    f"Resuming from checkpoint: {state.processed_documents}/{state.total_documents} complete"
                )

        # ========== PROCESS DOCUMENTS ==========
        def process_single_doc(doc: Document) -> List[Dict[str, Any]]:
            """Process a single document and return extracted items."""
            doc_start = time.time()
            items = []

            for ext_type in types_to_use:
                extractor = self._extractors.get(ext_type)
                if not extractor:
                    continue

                try:
                    result = extractor.extract(doc)
                    for item in result.items:
                        item_data = {
                            **item,
                            "_source": doc.source,
                            "_extractor": ext_type,
                        }

                        # Add attribution
                        if self._enable_attribution and self._attribution_manager:
                            attr = self._attribution_manager.create_attribution(
                                content=str(item.get("input", "") + item.get("output", "")),
                                source_file=doc.source,
                                extraction_method=ext_type,
                            )
                            item_data["_attribution_id"] = attr.item_id

                        items.append(item_data)
                except Exception as e:
                    logger.warning(f"Extraction failed for {ext_type} on {doc.source}: {e}")
                    if self._analytics:
                        self._analytics.record_error(doc.source, str(e))

            # Record analytics
            if self._analytics:
                self._analytics.record_document_processed(
                    source=doc.source,
                    items_extracted=len(items),
                    processing_time=time.time() - doc_start,
                )

            return items

        # Filter documents based on checkpoint state
        if state and self._checkpoint_manager:
            pending_docs = list(self._checkpoint_manager.iterate_pending(documents, state))
        else:
            pending_docs = documents

        all_items: List[Dict[str, Any]] = []

        # Use parallel processing if enabled and multiple documents
        if self._enable_parallel and self._parallel_processor and len(pending_docs) > 1:
            def progress_callback(completed: int, total: int):
                if progress:
                    print(f"\rProcessing: {completed}/{total} documents", end="", flush=True)

            batch_result = self._parallel_processor.process(
                pending_docs,
                process_single_doc,
                progress_callback=progress_callback if progress else None,
            )

            if progress:
                print()  # Newline after progress

            # Collect results and update checkpoint
            for result in batch_result.results:
                if result.success and result.result:
                    all_items.extend(result.result)
                    if state and self._checkpoint_manager:
                        doc_id = self._checkpoint_manager._get_doc_id(result.item)
                        self._checkpoint_manager.mark_complete(state, doc_id, result.result)
                elif not result.success and state and self._checkpoint_manager:
                    doc_id = self._checkpoint_manager._get_doc_id(result.item)
                    self._checkpoint_manager.mark_failed(state, doc_id, result.error or "Unknown error")
        else:
            # Sequential processing
            iterator = pending_docs
            if progress:
                iterator = tqdm(pending_docs, desc="Processing documents")

            for doc in iterator:
                try:
                    items = process_single_doc(doc)
                    all_items.extend(items)

                    if state and self._checkpoint_manager:
                        doc_id = self._checkpoint_manager._get_doc_id(doc)
                        self._checkpoint_manager.mark_complete(state, doc_id, items)
                except Exception as e:
                    logger.warning(f"Failed to process {doc.source}: {e}")
                    if state and self._checkpoint_manager:
                        doc_id = self._checkpoint_manager._get_doc_id(doc)
                        self._checkpoint_manager.mark_failed(state, doc_id, str(e))

        # ========== QUALITY FILTERING ==========
        if apply_quality_filter and self._enable_quality_filter and self._quality_pipeline:
            original_count = len(all_items)
            all_items = self._quality_pipeline.process(all_items)
            filtered_count = original_count - len(all_items)

            if filtered_count > 0:
                logger.info(f"Quality filter removed {filtered_count} items ({len(all_items)} remaining)")

            if self._analytics:
                # Store filter stats
                pass  # Filter stats available via self._quality_pipeline.stats()

        # ========== BUILD DATASET ==========
        for item in all_items:
            source = item.pop("_source", "unknown")
            extractor_type = item.pop("_extractor", "unknown")
            attribution_id = item.pop("_attribution_id", None)

            dataset.add(
                data=item,
                source=source,
                extractor_type=extractor_type,
            )

        # ========== FINALIZE ==========
        if state and self._checkpoint_manager:
            summary = self._checkpoint_manager.finalize_job(state)
            logger.info(f"Processing complete: {summary}")

        if self._analytics:
            summary = self._analytics.summary()
            logger.info(
                f"Analytics: {summary['documents_processed']} docs, "
                f"{summary['items_extracted']} items, "
                f"{summary['total_time_seconds']:.1f}s"
            )

        return dataset

    def process_text(
        self,
        text: str,
        source: str = "text_input",
        extractor_types: Optional[List[str]] = None,
    ) -> Dataset:
        """
        Process raw text content.

        Args:
            text: Text content to process.
            source: Source identifier.
            extractor_types: Specific extractors to use.

        Returns:
            Dataset with extracted items.
        """
        document = Document(content=text, source=source)
        return self.process_document(document, extractor_types=extractor_types)

    def add_extractor(
        self,
        extractor_type: str,
        extractor: Optional[BaseExtractor] = None,
    ) -> None:
        """
        Add or replace an extractor.

        Args:
            extractor_type: Type name for the extractor.
            extractor: Extractor instance (or creates default).
        """
        if extractor is None:
            extractor = get_extractor(
                extractor_type,
                self.llm_fn,
                chunk_size=self.chunk_size,
                max_items_per_chunk=self.max_items_per_chunk,
            )

        self._extractors[extractor_type] = extractor

    def list_extractors(self) -> List[str]:
        """List available extractor types."""
        return list(self._extractors.keys())

    def set_llm_fn(self, llm_fn: Callable[[str], str]) -> None:
        """
        Set a new LLM function.

        Args:
            llm_fn: New LLM function to use.
        """
        self.llm_fn = llm_fn

        # Update all extractors
        for ext_type in self._extractors:
            self._extractors[ext_type] = get_extractor(
                ext_type,
                llm_fn,
                chunk_size=self.chunk_size,
                max_items_per_chunk=self.max_items_per_chunk,
            )

    # ========== NEW MODULE ACCESS METHODS ==========

    def estimate_cost(
        self,
        documents: List[Document],
        extraction_type: str = "qa",
    ) -> CostEstimate:
        """
        Estimate cost for processing documents.

        Args:
            documents: Documents to estimate.
            extraction_type: Type of extraction.

        Returns:
            CostEstimate with breakdown.
        """
        if not self._cost_estimator:
            raise ValueError("Cost estimation not enabled")
        return self._cost_estimator.estimate_extraction(documents, extraction_type)

    def get_analytics_summary(self) -> Dict[str, Any]:
        """
        Get processing analytics summary.

        Returns:
            Dict with analytics metrics.
        """
        if self._analytics:
            return self._analytics.summary()
        return {}

    def get_quality_filter_stats(self) -> Dict[str, Any]:
        """
        Get quality filter statistics.

        Returns:
            Dict with filter stats.
        """
        if self._quality_pipeline:
            return self._quality_pipeline.stats()
        return {}

    def get_attributions(self) -> List[Attribution]:
        """
        Get all tracked attributions.

        Returns:
            List of Attribution objects.
        """
        if self._attribution_manager:
            return self._attribution_manager.get_all()
        return []

    def save_attributions(self, path: Union[str, Path]) -> None:
        """
        Save attributions to file.

        Args:
            path: Output file path.
        """
        if self._attribution_manager:
            self._attribution_manager.save(path)

    def export_analytics(self, path: Union[str, Path]) -> None:
        """
        Export analytics to file.

        Args:
            path: Output file path.
        """
        if self._analytics:
            self._analytics.export(path)

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all checkpoint jobs.

        Returns:
            List of job summaries.
        """
        if self._checkpoint_manager:
            return self._checkpoint_manager.list_jobs()
        return []

    def cleanup_checkpoints(self, max_age_days: int = 7) -> int:
        """
        Clean up old completed checkpoints.

        Args:
            max_age_days: Delete jobs older than this.

        Returns:
            Number of jobs deleted.
        """
        if self._checkpoint_manager:
            return self._checkpoint_manager.cleanup_completed(max_age_days)
        return 0

    @classmethod
    def from_config(cls, config: ProcessorConfig, **kwargs: Any) -> "DocProcessor":
        """
        Create DocProcessor from a configuration object.

        Args:
            config: ProcessorConfig instance.
            **kwargs: Additional arguments passed to __init__.

        Returns:
            Configured DocProcessor instance.

        Example:
            >>> config = ProcessorConfig(
            ...     provider="openai",
            ...     model="gpt-4o",
            ...     enable_parallel=True,
            ...     max_workers=10,
            ... )
            >>> processor = DocProcessor.from_config(config)
        """
        return cls(config=config, **kwargs)


def process_documents(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    provider: str = "openai",
    model: Optional[str] = None,
    extractors: Optional[List[str]] = None,
    output_format: str = "openai",
    **kwargs: Any,
) -> int:
    """
    Convenience function to process documents and export.

    Args:
        input_path: Input file or folder path.
        output_path: Output JSONL file path.
        provider: LLM provider.
        model: Model name.
        extractors: List of extractor types.
        output_format: Output format for JSONL.
        **kwargs: Additional processor arguments.

    Returns:
        Number of items exported.

    Example:
        >>> count = process_documents(
        ...     "./documents",
        ...     "./training.jsonl",
        ...     provider="openai",
        ...     model="gpt-4",
        ...     extractors=["qa", "rules"],
        ...     output_format="openai"
        ... )
    """
    processor = DocProcessor(
        provider=provider,
        model=model,
        extractors=extractors,
        **kwargs,
    )

    input_path = Path(input_path)

    if input_path.is_file():
        dataset = processor.process_document(input_path)
    elif input_path.is_dir():
        dataset = processor.process_folder(input_path)
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    # Deduplicate
    dataset = dataset.deduplicate()

    # Export
    return dataset.to_jsonl(output_path, format=output_format)

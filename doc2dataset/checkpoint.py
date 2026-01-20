"""
Checkpointing and resumable processing for doc2dataset.

Enables saving and resuming long-running extraction jobs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, TypeVar

from doc2dataset.loaders import Document

logger = logging.getLogger("doc2dataset")

T = TypeVar("T")


@dataclass
class ProcessingState:
    """
    State of a processing job.

    Attributes:
        job_id: Unique identifier for the job.
        created_at: When the job was created.
        updated_at: Last update time.
        total_documents: Total documents to process.
        processed_documents: Number processed so far.
        failed_documents: Number that failed.
        processed_ids: Set of processed document IDs.
        failed_ids: Dict of failed document IDs to error messages.
        extraction_type: Type of extraction being performed.
        model: Model being used.
        config: Job configuration.
        results: Accumulated results.
    """

    job_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    processed_ids: Set[str] = field(default_factory=set)
    failed_ids: Dict[str, str] = field(default_factory=dict)
    extraction_type: str = ""
    model: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    results: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """Check if job is complete."""
        return self.processed_documents + self.failed_documents >= self.total_documents

    @property
    def progress(self) -> float:
        """Get progress percentage."""
        if self.total_documents == 0:
            return 0.0
        return (self.processed_documents + self.failed_documents) / self.total_documents

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "job_id": self.job_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "total_documents": self.total_documents,
            "processed_documents": self.processed_documents,
            "failed_documents": self.failed_documents,
            "processed_ids": list(self.processed_ids),
            "failed_ids": self.failed_ids,
            "extraction_type": self.extraction_type,
            "model": self.model,
            "config": self.config,
            # Results stored separately for large jobs
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingState":
        """Create from dict."""
        return cls(
            job_id=data["job_id"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            total_documents=data["total_documents"],
            processed_documents=data["processed_documents"],
            failed_documents=data["failed_documents"],
            processed_ids=set(data.get("processed_ids", [])),
            failed_ids=data.get("failed_ids", {}),
            extraction_type=data.get("extraction_type", ""),
            model=data.get("model", ""),
            config=data.get("config", {}),
        )


class CheckpointManager:
    """
    Manages checkpoints for resumable processing.

    Saves processing state to disk to enable resuming
    interrupted jobs.

    Example:
        >>> manager = CheckpointManager("./checkpoints")
        >>> state = manager.create_job(documents, extraction_type="qa")
        >>>
        >>> for doc in manager.iterate_pending(documents, state):
        ...     result = process(doc)
        ...     manager.mark_complete(state, doc.id, result)
        >>>
        >>> # If interrupted, can resume later:
        >>> state = manager.load_job(job_id)
        >>> for doc in manager.iterate_pending(documents, state):
        ...     ...
    """

    def __init__(
        self,
        checkpoint_dir: str = "./.doc2dataset_checkpoints",
        auto_save_interval: int = 10,
    ) -> None:
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint files.
            auto_save_interval: Save after this many documents.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.auto_save_interval = auto_save_interval
        self._save_counter = 0

    def _generate_job_id(
        self,
        documents: List[Document],
        extraction_type: str,
        model: str,
    ) -> str:
        """Generate deterministic job ID."""
        # Create hash from document sources and config
        sources = sorted([doc.source for doc in documents])
        key = f"{extraction_type}:{model}:{':'.join(sources)}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _get_state_path(self, job_id: str) -> Path:
        """Get path for state file."""
        return self.checkpoint_dir / f"{job_id}_state.json"

    def _get_results_path(self, job_id: str) -> Path:
        """Get path for results file."""
        return self.checkpoint_dir / f"{job_id}_results.jsonl"

    def create_job(
        self,
        documents: List[Document],
        extraction_type: str,
        model: str = "",
        config: Optional[Dict[str, Any]] = None,
        resume_if_exists: bool = True,
    ) -> ProcessingState:
        """
        Create a new processing job or resume existing.

        Args:
            documents: Documents to process.
            extraction_type: Type of extraction.
            model: Model being used.
            config: Job configuration.
            resume_if_exists: Resume existing job if found.

        Returns:
            ProcessingState for the job.
        """
        job_id = self._generate_job_id(documents, extraction_type, model)

        # Check for existing job
        if resume_if_exists:
            existing = self.load_job(job_id)
            if existing:
                logger.info(
                    f"Resuming job {job_id}: "
                    f"{existing.processed_documents}/{existing.total_documents} complete"
                )
                return existing

        # Create new job
        state = ProcessingState(
            job_id=job_id,
            total_documents=len(documents),
            extraction_type=extraction_type,
            model=model,
            config=config or {},
        )

        self.save_state(state)
        logger.info(f"Created new job {job_id} with {len(documents)} documents")

        return state

    def load_job(self, job_id: str) -> Optional[ProcessingState]:
        """
        Load existing job state.

        Args:
            job_id: Job identifier.

        Returns:
            ProcessingState or None if not found.
        """
        state_path = self._get_state_path(job_id)
        if not state_path.exists():
            return None

        try:
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            state = ProcessingState.from_dict(data)

            # Load results
            results_path = self._get_results_path(job_id)
            if results_path.exists():
                with open(results_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            state.results.append(json.loads(line))

            return state
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load job {job_id}: {e}")
            return None

    def save_state(self, state: ProcessingState) -> None:
        """
        Save job state.

        Args:
            state: State to save.
        """
        state.updated_at = time.time()
        state_path = self._get_state_path(state.job_id)

        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2)

    def _append_result(self, state: ProcessingState, result: Dict[str, Any]) -> None:
        """Append result to results file."""
        results_path = self._get_results_path(state.job_id)
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

    def mark_complete(
        self,
        state: ProcessingState,
        doc_id: str,
        results: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Mark a document as successfully processed.

        Args:
            state: Job state.
            doc_id: Document ID.
            results: Extracted results from this document.
        """
        if doc_id in state.processed_ids:
            return

        state.processed_ids.add(doc_id)
        state.processed_documents += 1

        # Save results
        if results:
            for result in results:
                result["_source_id"] = doc_id
                state.results.append(result)
                self._append_result(state, result)

        # Auto-save
        self._save_counter += 1
        if self._save_counter >= self.auto_save_interval:
            self.save_state(state)
            self._save_counter = 0

    def mark_failed(
        self,
        state: ProcessingState,
        doc_id: str,
        error: str,
    ) -> None:
        """
        Mark a document as failed.

        Args:
            state: Job state.
            doc_id: Document ID.
            error: Error message.
        """
        if doc_id in state.processed_ids or doc_id in state.failed_ids:
            return

        state.failed_ids[doc_id] = error
        state.failed_documents += 1

        # Auto-save
        self._save_counter += 1
        if self._save_counter >= self.auto_save_interval:
            self.save_state(state)
            self._save_counter = 0

    def iterate_pending(
        self,
        documents: List[Document],
        state: ProcessingState,
    ) -> Iterator[Document]:
        """
        Iterate over pending documents.

        Args:
            documents: All documents.
            state: Job state.

        Yields:
            Documents that haven't been processed.
        """
        for doc in documents:
            doc_id = self._get_doc_id(doc)
            if doc_id not in state.processed_ids and doc_id not in state.failed_ids:
                yield doc

    def _get_doc_id(self, doc: Document) -> str:
        """Get unique ID for a document."""
        # Use source path as ID, or hash content
        if doc.source:
            return hashlib.sha256(doc.source.encode()).hexdigest()[:16]
        return hashlib.sha256(doc.content.encode()).hexdigest()[:16]

    def get_results(self, state: ProcessingState) -> List[Dict[str, Any]]:
        """
        Get all results from a job.

        Args:
            state: Job state.

        Returns:
            List of extracted results.
        """
        return state.results

    def finalize_job(self, state: ProcessingState) -> Dict[str, Any]:
        """
        Finalize a completed job.

        Args:
            state: Job state.

        Returns:
            Job summary.
        """
        self.save_state(state)

        return {
            "job_id": state.job_id,
            "total_documents": state.total_documents,
            "processed_documents": state.processed_documents,
            "failed_documents": state.failed_documents,
            "success_rate": state.processed_documents / max(1, state.total_documents),
            "total_results": len(state.results),
            "duration_seconds": state.updated_at - state.created_at,
        }

    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all jobs.

        Returns:
            List of job summaries.
        """
        jobs = []
        for state_file in self.checkpoint_dir.glob("*_state.json"):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                jobs.append({
                    "job_id": data["job_id"],
                    "extraction_type": data.get("extraction_type", ""),
                    "progress": (
                        data["processed_documents"] + data["failed_documents"]
                    ) / max(1, data["total_documents"]),
                    "created_at": datetime.fromtimestamp(data["created_at"]).isoformat(),
                    "updated_at": datetime.fromtimestamp(data["updated_at"]).isoformat(),
                    "is_complete": (
                        data["processed_documents"] + data["failed_documents"]
                    ) >= data["total_documents"],
                })
            except (json.JSONDecodeError, KeyError):
                continue

        return sorted(jobs, key=lambda x: x["updated_at"], reverse=True)

    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job and its data.

        Args:
            job_id: Job to delete.

        Returns:
            True if deleted.
        """
        state_path = self._get_state_path(job_id)
        results_path = self._get_results_path(job_id)

        deleted = False
        if state_path.exists():
            state_path.unlink()
            deleted = True
        if results_path.exists():
            results_path.unlink()
            deleted = True

        return deleted

    def cleanup_completed(self, max_age_days: int = 7) -> int:
        """
        Clean up old completed jobs.

        Args:
            max_age_days: Delete jobs older than this.

        Returns:
            Number of jobs deleted.
        """
        cutoff = time.time() - (max_age_days * 86400)
        deleted = 0

        for state_file in self.checkpoint_dir.glob("*_state.json"):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Only delete completed jobs that are old
                is_complete = (
                    data["processed_documents"] + data["failed_documents"]
                ) >= data["total_documents"]

                if is_complete and data["updated_at"] < cutoff:
                    self.delete_job(data["job_id"])
                    deleted += 1
            except (json.JSONDecodeError, KeyError):
                continue

        return deleted


class ResumableProcessor:
    """
    Wrapper for resumable document processing.

    Provides a simple interface for processing documents
    with automatic checkpointing.

    Example:
        >>> processor = ResumableProcessor(
        ...     process_fn=extract_qa,
        ...     checkpoint_dir="./checkpoints",
        ... )
        >>> results = processor.process(documents, extraction_type="qa")
    """

    def __init__(
        self,
        process_fn: Callable[[Document], List[Dict[str, Any]]],
        checkpoint_dir: str = "./.doc2dataset_checkpoints",
        model: str = "",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """
        Initialize resumable processor.

        Args:
            process_fn: Function to process each document.
            checkpoint_dir: Directory for checkpoints.
            model: Model being used (for job ID).
            max_retries: Maximum retries per document.
            retry_delay: Delay between retries in seconds.
        """
        self.process_fn = process_fn
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def process(
        self,
        documents: List[Document],
        extraction_type: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process documents with checkpointing.

        Args:
            documents: Documents to process.
            extraction_type: Type of extraction.
            progress_callback: Optional callback(processed, total).

        Returns:
            List of extracted results.
        """
        state = self.checkpoint_manager.create_job(
            documents,
            extraction_type=extraction_type,
            model=self.model,
        )

        # Build ID to document mapping
        doc_map = {
            self.checkpoint_manager._get_doc_id(doc): doc
            for doc in documents
        }

        # Process pending documents
        pending = list(self.checkpoint_manager.iterate_pending(documents, state))
        total = len(documents)

        for doc in pending:
            doc_id = self.checkpoint_manager._get_doc_id(doc)

            # Try with retries
            for attempt in range(self.max_retries):
                try:
                    results = self.process_fn(doc)
                    self.checkpoint_manager.mark_complete(state, doc_id, results)
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        self.checkpoint_manager.mark_failed(state, doc_id, str(e))
                        logger.warning(f"Failed to process {doc.source}: {e}")
                    else:
                        time.sleep(self.retry_delay * (attempt + 1))

            # Progress callback
            if progress_callback:
                processed = state.processed_documents + state.failed_documents
                progress_callback(processed, total)

        # Finalize
        summary = self.checkpoint_manager.finalize_job(state)
        logger.info(
            f"Job complete: {summary['processed_documents']}/{summary['total_documents']} "
            f"documents, {summary['total_results']} results"
        )

        return self.checkpoint_manager.get_results(state)

"""
Parallel processing for doc2dataset.

Provides efficient parallel and async processing for
document extraction at scale.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, TypeVar

from doc2dataset.loaders import Document

logger = logging.getLogger("doc2dataset")

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ProcessingResult(Generic[T]):
    """
    Result of processing a single item.

    Attributes:
        item: Original item.
        result: Processing result.
        error: Error message if failed.
        duration_ms: Processing duration.
        worker_id: ID of worker that processed this.
    """

    item: T
    result: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    worker_id: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class BatchResult(Generic[T]):
    """
    Results from batch processing.

    Attributes:
        results: List of individual results.
        total_duration_ms: Total processing time.
        success_count: Number of successful items.
        failure_count: Number of failed items.
    """

    results: List[ProcessingResult[T]] = field(default_factory=list)
    total_duration_ms: float = 0.0

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self.results if not r.success)

    @property
    def success_rate(self) -> float:
        total = len(self.results)
        return self.success_count / total if total > 0 else 0.0

    def successful_results(self) -> List[Any]:
        """Get only successful results."""
        return [r.result for r in self.results if r.success and r.result is not None]

    def failed_items(self) -> List[T]:
        """Get items that failed."""
        return [r.item for r in self.results if not r.success]


class ThreadPoolProcessor:
    """
    Thread-based parallel processor for I/O-bound tasks.

    Good for API calls where most time is spent waiting.

    Example:
        >>> processor = ThreadPoolProcessor(max_workers=10)
        >>> results = processor.process(documents, extract_fn)
    """

    def __init__(
        self,
        max_workers: int = 5,
        rate_limit: Optional[float] = None,
    ) -> None:
        """
        Initialize thread pool processor.

        Args:
            max_workers: Maximum concurrent threads.
            rate_limit: Maximum requests per second (None for unlimited).
        """
        self.max_workers = max_workers
        self.rate_limit = rate_limit
        self._last_request_time = 0.0
        self._rate_lock = threading.Lock()

    def _apply_rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        if self.rate_limit is None:
            return

        with self._rate_lock:
            now = time.time()
            min_interval = 1.0 / self.rate_limit
            elapsed = now - self._last_request_time

            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            self._last_request_time = time.time()

    def process(
        self,
        items: List[T],
        process_fn: Callable[[T], R],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult[T]:
        """
        Process items in parallel using threads.

        Args:
            items: Items to process.
            process_fn: Function to process each item.
            progress_callback: Optional callback(completed, total).

        Returns:
            BatchResult with all results.
        """
        results = BatchResult[T]()
        start_time = time.time()
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {}
            for item in items:
                future = executor.submit(self._process_item, item, process_fn)
                future_to_item[future] = item

            # Collect results
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                result = future.result()
                results.results.append(result)

                completed += 1
                if progress_callback:
                    progress_callback(completed, len(items))

        results.total_duration_ms = (time.time() - start_time) * 1000
        return results

    def _process_item(self, item: T, process_fn: Callable[[T], R]) -> ProcessingResult[T]:
        """Process a single item with error handling."""
        self._apply_rate_limit()

        start_time = time.time()
        worker_id = threading.current_thread().name

        try:
            result = process_fn(item)
            return ProcessingResult(
                item=item,
                result=result,
                duration_ms=(time.time() - start_time) * 1000,
                worker_id=worker_id,
            )
        except Exception as e:
            logger.warning(f"Processing failed: {e}")
            return ProcessingResult(
                item=item,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
                worker_id=worker_id,
            )


class ProcessPoolProcessor:
    """
    Process-based parallel processor for CPU-bound tasks.

    Good for heavy local processing that benefits from multiple cores.

    Example:
        >>> processor = ProcessPoolProcessor(max_workers=4)
        >>> results = processor.process(documents, heavy_process_fn)
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
    ) -> None:
        """
        Initialize process pool processor.

        Args:
            max_workers: Maximum concurrent processes (default: CPU count).
        """
        import os
        self.max_workers = max_workers or os.cpu_count() or 4

    def process(
        self,
        items: List[T],
        process_fn: Callable[[T], R],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult[T]:
        """
        Process items in parallel using processes.

        Args:
            items: Items to process.
            process_fn: Function to process each item (must be picklable).
            progress_callback: Optional callback(completed, total).

        Returns:
            BatchResult with all results.
        """
        results = BatchResult[T]()
        start_time = time.time()
        completed = 0

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {}
            for item in items:
                future = executor.submit(process_fn, item)
                future_to_item[future] = item

            # Collect results
            for future in as_completed(future_to_item):
                item = future_to_item[future]

                try:
                    result = future.result()
                    results.results.append(ProcessingResult(
                        item=item,
                        result=result,
                        duration_ms=0,  # Can't measure per-item in process pool
                    ))
                except Exception as e:
                    results.results.append(ProcessingResult(
                        item=item,
                        error=str(e),
                    ))

                completed += 1
                if progress_callback:
                    progress_callback(completed, len(items))

        results.total_duration_ms = (time.time() - start_time) * 1000
        return results


class AsyncProcessor:
    """
    Async processor for concurrent I/O operations.

    Most efficient for API calls with proper async support.

    Example:
        >>> processor = AsyncProcessor(max_concurrent=20)
        >>> results = await processor.process(documents, async_extract_fn)
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        rate_limit: Optional[float] = None,
    ) -> None:
        """
        Initialize async processor.

        Args:
            max_concurrent: Maximum concurrent operations.
            rate_limit: Maximum requests per second.
        """
        self.max_concurrent = max_concurrent
        self.rate_limit = rate_limit
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._rate_limiter: Optional[asyncio.Lock] = None
        self._last_request_time = 0.0

    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting."""
        if self.rate_limit is None:
            return

        now = time.time()
        min_interval = 1.0 / self.rate_limit
        elapsed = now - self._last_request_time

        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)

        self._last_request_time = time.time()

    async def process(
        self,
        items: List[T],
        process_fn: Callable[[T], Any],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult[T]:
        """
        Process items concurrently.

        Args:
            items: Items to process.
            process_fn: Async function to process each item.
            progress_callback: Optional callback(completed, total).

        Returns:
            BatchResult with all results.
        """
        results = BatchResult[T]()
        start_time = time.time()
        completed = 0
        completed_lock = asyncio.Lock()

        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._rate_limiter = asyncio.Lock()

        async def process_with_semaphore(item: T) -> ProcessingResult[T]:
            nonlocal completed

            async with self._semaphore:
                async with self._rate_limiter:
                    await self._apply_rate_limit()

                item_start = time.time()
                try:
                    # Handle both sync and async functions
                    if asyncio.iscoroutinefunction(process_fn):
                        result = await process_fn(item)
                    else:
                        result = process_fn(item)

                    processing_result = ProcessingResult(
                        item=item,
                        result=result,
                        duration_ms=(time.time() - item_start) * 1000,
                    )
                except Exception as e:
                    processing_result = ProcessingResult(
                        item=item,
                        error=str(e),
                        duration_ms=(time.time() - item_start) * 1000,
                    )

                async with completed_lock:
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(items))

                return processing_result

        # Process all items
        tasks = [process_with_semaphore(item) for item in items]
        results.results = await asyncio.gather(*tasks)
        results.total_duration_ms = (time.time() - start_time) * 1000

        return results

    def process_sync(
        self,
        items: List[T],
        process_fn: Callable[[T], Any],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult[T]:
        """
        Synchronous wrapper for process().

        Args:
            items: Items to process.
            process_fn: Function to process each item.
            progress_callback: Optional callback.

        Returns:
            BatchResult with all results.
        """
        return asyncio.run(self.process(items, process_fn, progress_callback))


class BatchingProcessor:
    """
    Processor that batches items for efficient processing.

    Groups items into batches for bulk API calls.

    Example:
        >>> processor = BatchingProcessor(batch_size=10)
        >>> results = processor.process(items, batch_process_fn)
    """

    def __init__(
        self,
        batch_size: int = 10,
        max_workers: int = 4,
    ) -> None:
        """
        Initialize batching processor.

        Args:
            batch_size: Items per batch.
            max_workers: Concurrent batches.
        """
        self.batch_size = batch_size
        self.max_workers = max_workers

    def _create_batches(self, items: List[T]) -> List[List[T]]:
        """Split items into batches."""
        batches = []
        for i in range(0, len(items), self.batch_size):
            batches.append(items[i:i + self.batch_size])
        return batches

    def process(
        self,
        items: List[T],
        batch_process_fn: Callable[[List[T]], List[R]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult[T]:
        """
        Process items in batches.

        Args:
            items: Items to process.
            batch_process_fn: Function that processes a batch.
            progress_callback: Optional callback.

        Returns:
            BatchResult with all results.
        """
        batches = self._create_batches(items)
        results = BatchResult[T]()
        start_time = time.time()
        completed_items = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {}
            for batch in batches:
                future = executor.submit(self._process_batch, batch, batch_process_fn)
                future_to_batch[future] = batch

            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                batch_results = future.result()
                results.results.extend(batch_results)

                completed_items += len(batch)
                if progress_callback:
                    progress_callback(completed_items, len(items))

        results.total_duration_ms = (time.time() - start_time) * 1000
        return results

    def _process_batch(
        self,
        batch: List[T],
        batch_process_fn: Callable[[List[T]], List[R]],
    ) -> List[ProcessingResult[T]]:
        """Process a single batch."""
        start_time = time.time()

        try:
            batch_results = batch_process_fn(batch)

            # Match results to items
            results = []
            for item, result in zip(batch, batch_results):
                results.append(ProcessingResult(
                    item=item,
                    result=result,
                    duration_ms=(time.time() - start_time) * 1000 / len(batch),
                ))
            return results

        except Exception as e:
            # All items in batch failed
            return [
                ProcessingResult(item=item, error=str(e))
                for item in batch
            ]


class PipelineProcessor:
    """
    Multi-stage processing pipeline.

    Chains multiple processing stages together.

    Example:
        >>> pipeline = PipelineProcessor()
        >>> pipeline.add_stage("load", load_fn, workers=2)
        >>> pipeline.add_stage("extract", extract_fn, workers=10)
        >>> pipeline.add_stage("filter", filter_fn, workers=4)
        >>> results = pipeline.process(documents)
    """

    @dataclass
    class Stage:
        """Pipeline stage configuration."""
        name: str
        process_fn: Callable
        workers: int = 4
        is_filter: bool = False

    def __init__(self) -> None:
        """Initialize pipeline."""
        self._stages: List[PipelineProcessor.Stage] = []

    def add_stage(
        self,
        name: str,
        process_fn: Callable,
        workers: int = 4,
        is_filter: bool = False,
    ) -> "PipelineProcessor":
        """
        Add a processing stage.

        Args:
            name: Stage name.
            process_fn: Processing function.
            workers: Number of workers for this stage.
            is_filter: If True, stage filters items (returns bool).

        Returns:
            Self for chaining.
        """
        self._stages.append(self.Stage(
            name=name,
            process_fn=process_fn,
            workers=workers,
            is_filter=is_filter,
        ))
        return self

    def process(
        self,
        items: List[Any],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run items through all pipeline stages.

        Args:
            items: Initial items.
            progress_callback: Callback(stage_name, completed, total).

        Returns:
            Dict with results and statistics.
        """
        current_items = items
        stats = {"stages": {}}
        total_start = time.time()

        for stage in self._stages:
            stage_start = time.time()
            processor = ThreadPoolProcessor(max_workers=stage.workers)

            def stage_callback(completed: int, total: int):
                if progress_callback:
                    progress_callback(stage.name, completed, total)

            batch_result = processor.process(
                current_items,
                stage.process_fn,
                progress_callback=stage_callback,
            )

            if stage.is_filter:
                # Filter stage: keep items where result is truthy
                current_items = [
                    r.item for r in batch_result.results
                    if r.success and r.result
                ]
            else:
                # Transform stage: use results as next input
                current_items = batch_result.successful_results()

            stats["stages"][stage.name] = {
                "input_count": len(batch_result.results),
                "output_count": len(current_items),
                "success_rate": batch_result.success_rate,
                "duration_ms": (time.time() - stage_start) * 1000,
            }

            logger.info(
                f"Stage '{stage.name}': {len(batch_result.results)} -> "
                f"{len(current_items)} items"
            )

        stats["total_duration_ms"] = (time.time() - total_start) * 1000
        stats["final_count"] = len(current_items)

        return {
            "results": current_items,
            "stats": stats,
        }


def parallel_map(
    fn: Callable[[T], R],
    items: List[T],
    max_workers: int = 5,
    use_processes: bool = False,
) -> List[R]:
    """
    Simple parallel map function.

    Args:
        fn: Function to apply.
        items: Items to process.
        max_workers: Number of workers.
        use_processes: Use processes instead of threads.

    Returns:
        List of results (in order).
    """
    if use_processes:
        processor = ProcessPoolProcessor(max_workers=max_workers)
    else:
        processor = ThreadPoolProcessor(max_workers=max_workers)

    batch_result = processor.process(items, fn)

    # Maintain order
    result_map = {id(r.item): r.result for r in batch_result.results}
    return [result_map.get(id(item)) for item in items]


async def async_map(
    fn: Callable[[T], Any],
    items: List[T],
    max_concurrent: int = 10,
) -> List[Any]:
    """
    Async parallel map function.

    Args:
        fn: Function to apply (can be sync or async).
        items: Items to process.
        max_concurrent: Maximum concurrency.

    Returns:
        List of results (in order).
    """
    processor = AsyncProcessor(max_concurrent=max_concurrent)
    batch_result = await processor.process(items, fn)

    # Maintain order
    result_map = {id(r.item): r.result for r in batch_result.results}
    return [result_map.get(id(item)) for item in items]

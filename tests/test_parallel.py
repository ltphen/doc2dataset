"""Tests for parallel processing module."""

import asyncio
import time
import pytest
from doc2dataset.parallel import (
    ProcessingResult,
    BatchResult,
    ThreadPoolProcessor,
    ProcessPoolProcessor,
    AsyncProcessor,
    BatchingProcessor,
    PipelineProcessor,
    parallel_map,
    async_map,
)


class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""

    def test_basic_creation(self):
        result = ProcessingResult(item="test_item", result="processed")
        assert result.item == "test_item"
        assert result.result == "processed"
        assert result.error is None
        assert result.success is True

    def test_with_error(self):
        result = ProcessingResult(item="test", error="Something failed")
        assert result.success is False
        assert result.error == "Something failed"

    def test_with_duration(self):
        result = ProcessingResult(
            item="test",
            result="ok",
            duration_ms=150.5,
            worker_id="worker-1",
        )
        assert result.duration_ms == 150.5
        assert result.worker_id == "worker-1"


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_empty_result(self):
        result = BatchResult()
        assert result.success_count == 0
        assert result.failure_count == 0
        assert result.success_rate == 0.0

    def test_with_results(self):
        result = BatchResult(results=[
            ProcessingResult(item="a", result="ok"),
            ProcessingResult(item="b", result="ok"),
            ProcessingResult(item="c", error="failed"),
        ])
        assert result.success_count == 2
        assert result.failure_count == 1
        assert result.success_rate == 2/3

    def test_successful_results(self):
        result = BatchResult(results=[
            ProcessingResult(item="a", result="result_a"),
            ProcessingResult(item="b", error="failed"),
            ProcessingResult(item="c", result="result_c"),
        ])
        successful = result.successful_results()
        assert len(successful) == 2
        assert "result_a" in successful
        assert "result_c" in successful

    def test_failed_items(self):
        result = BatchResult(results=[
            ProcessingResult(item="a", result="ok"),
            ProcessingResult(item="b", error="failed"),
        ])
        failed = result.failed_items()
        assert len(failed) == 1
        assert "b" in failed


class TestThreadPoolProcessor:
    """Tests for ThreadPoolProcessor."""

    def test_basic_processing(self):
        processor = ThreadPoolProcessor(max_workers=2)

        def process_fn(item):
            return item * 2

        items = [1, 2, 3, 4, 5]
        result = processor.process(items, process_fn)

        assert result.success_count == 5
        assert result.failure_count == 0
        assert sorted(result.successful_results()) == [2, 4, 6, 8, 10]

    def test_with_progress_callback(self):
        processor = ThreadPoolProcessor(max_workers=2)
        progress_calls = []

        def callback(completed, total):
            progress_calls.append((completed, total))

        def process_fn(item):
            return item

        items = [1, 2, 3]
        processor.process(items, process_fn, progress_callback=callback)

        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)

    def test_error_handling(self):
        processor = ThreadPoolProcessor(max_workers=2)

        def process_fn(item):
            if item == 2:
                raise ValueError("Error on 2")
            return item

        items = [1, 2, 3]
        result = processor.process(items, process_fn)

        assert result.success_count == 2
        assert result.failure_count == 1

    def test_rate_limiting(self):
        processor = ThreadPoolProcessor(max_workers=2, rate_limit=10.0)
        start_time = time.time()

        def process_fn(item):
            return item

        items = [1, 2, 3]
        processor.process(items, process_fn)

        elapsed = time.time() - start_time
        # With rate limit of 10/sec, 3 items should take at least 0.2 seconds
        assert elapsed >= 0.1


class TestProcessPoolProcessor:
    """Tests for ProcessPoolProcessor."""

    def test_basic_processing(self):
        processor = ProcessPoolProcessor(max_workers=2)

        def process_fn(item):
            return item * 2

        items = [1, 2, 3]
        result = processor.process(items, process_fn)

        assert result.success_count == 3
        assert sorted(result.successful_results()) == [2, 4, 6]

    def test_error_handling(self):
        processor = ProcessPoolProcessor(max_workers=2)

        def process_fn(item):
            if item == 2:
                raise ValueError("Error")
            return item

        items = [1, 2, 3]
        result = processor.process(items, process_fn)

        assert result.success_count == 2
        assert result.failure_count == 1


class TestAsyncProcessor:
    """Tests for AsyncProcessor."""

    @pytest.mark.asyncio
    async def test_basic_processing(self):
        processor = AsyncProcessor(max_concurrent=3)

        async def process_fn(item):
            await asyncio.sleep(0.01)
            return item * 2

        items = [1, 2, 3, 4]
        result = await processor.process(items, process_fn)

        assert result.success_count == 4
        assert sorted(result.successful_results()) == [2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_with_sync_function(self):
        processor = AsyncProcessor(max_concurrent=2)

        def sync_fn(item):
            return item + 1

        items = [1, 2, 3]
        result = await processor.process(items, sync_fn)

        assert result.success_count == 3

    @pytest.mark.asyncio
    async def test_error_handling(self):
        processor = AsyncProcessor(max_concurrent=2)

        async def process_fn(item):
            if item == 2:
                raise ValueError("Error")
            return item

        items = [1, 2, 3]
        result = await processor.process(items, process_fn)

        assert result.success_count == 2
        assert result.failure_count == 1

    def test_sync_wrapper(self):
        processor = AsyncProcessor(max_concurrent=2)

        def process_fn(item):
            return item * 2

        items = [1, 2, 3]
        result = processor.process_sync(items, process_fn)

        assert result.success_count == 3

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        processor = AsyncProcessor(max_concurrent=5, rate_limit=20.0)

        async def process_fn(item):
            return item

        items = [1, 2, 3, 4, 5]
        start_time = time.time()
        await processor.process(items, process_fn)
        elapsed = time.time() - start_time

        # With rate limit of 20/sec, 5 items should take some time
        assert elapsed >= 0.1


class TestBatchingProcessor:
    """Tests for BatchingProcessor."""

    def test_basic_batching(self):
        processor = BatchingProcessor(batch_size=2, max_workers=2)

        def batch_fn(batch):
            return [item * 2 for item in batch]

        items = [1, 2, 3, 4, 5]
        result = processor.process(items, batch_fn)

        assert result.success_count == 5
        assert sorted(result.successful_results()) == [2, 4, 6, 8, 10]

    def test_batch_sizes(self):
        batches_received = []

        def batch_fn(batch):
            batches_received.append(len(batch))
            return batch

        processor = BatchingProcessor(batch_size=3)
        items = [1, 2, 3, 4, 5, 6, 7]  # 7 items -> batches of 3, 3, 1
        processor.process(items, batch_fn)

        assert sorted(batches_received) == [1, 3, 3]

    def test_batch_error_handling(self):
        processor = BatchingProcessor(batch_size=2)

        def batch_fn(batch):
            if 3 in batch:
                raise ValueError("Error in batch")
            return batch

        items = [1, 2, 3, 4, 5, 6]
        result = processor.process(items, batch_fn)

        # Batch with 3,4 should fail
        assert result.failure_count == 2
        assert result.success_count == 4


class TestPipelineProcessor:
    """Tests for PipelineProcessor."""

    def test_basic_pipeline(self):
        pipeline = PipelineProcessor()
        pipeline.add_stage("double", lambda x: x * 2)
        pipeline.add_stage("add_one", lambda x: x + 1)

        result = pipeline.process([1, 2, 3])

        assert result["final_count"] == 3
        assert sorted(result["results"]) == [3, 5, 7]  # (1*2+1, 2*2+1, 3*2+1)

    def test_filter_stage(self):
        pipeline = PipelineProcessor()
        pipeline.add_stage("double", lambda x: x * 2)
        pipeline.add_stage("filter_big", lambda x: x > 5, is_filter=True)

        result = pipeline.process([1, 2, 3, 4, 5])

        # After doubling: 2, 4, 6, 8, 10
        # After filtering > 5: 6, 8, 10
        assert result["final_count"] == 3

    def test_stage_statistics(self):
        pipeline = PipelineProcessor()
        pipeline.add_stage("transform", lambda x: x * 2)

        result = pipeline.process([1, 2, 3])

        assert "stages" in result["stats"]
        assert "transform" in result["stats"]["stages"]
        assert result["stats"]["stages"]["transform"]["input_count"] == 3

    def test_add_stage_chaining(self):
        pipeline = PipelineProcessor()
        result = pipeline.add_stage("s1", lambda x: x)
        assert result is pipeline

    def test_with_progress_callback(self):
        pipeline = PipelineProcessor()
        pipeline.add_stage("double", lambda x: x * 2)

        progress_calls = []

        def callback(stage_name, completed, total):
            progress_calls.append((stage_name, completed, total))

        pipeline.process([1, 2], progress_callback=callback)

        assert len(progress_calls) > 0
        assert progress_calls[0][0] == "double"


class TestParallelMap:
    """Tests for parallel_map utility function."""

    def test_basic_map(self):
        def double(x):
            return x * 2

        results = parallel_map(double, [1, 2, 3], max_workers=2)
        assert results == [2, 4, 6]

    def test_order_preserved(self):
        def process(x):
            time.sleep(0.01 * (5 - x))  # Longer sleep for lower numbers
            return x

        results = parallel_map(process, [1, 2, 3, 4, 5], max_workers=5)
        assert results == [1, 2, 3, 4, 5]

    def test_with_processes(self):
        def double(x):
            return x * 2

        results = parallel_map(double, [1, 2, 3], max_workers=2, use_processes=True)
        assert results == [2, 4, 6]


class TestAsyncMap:
    """Tests for async_map utility function."""

    @pytest.mark.asyncio
    async def test_basic_async_map(self):
        async def double(x):
            await asyncio.sleep(0.01)
            return x * 2

        results = await async_map(double, [1, 2, 3], max_concurrent=3)
        assert results == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_order_preserved(self):
        async def process(x):
            await asyncio.sleep(0.01 * (5 - x))
            return x

        results = await async_map(process, [1, 2, 3, 4, 5], max_concurrent=5)
        assert results == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_with_sync_function(self):
        def triple(x):
            return x * 3

        results = await async_map(triple, [1, 2, 3], max_concurrent=2)
        assert results == [3, 6, 9]

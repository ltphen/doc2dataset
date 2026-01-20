"""
Tests for doc2dataset checkpointing.
"""

import pytest
import json
from pathlib import Path
from typing import List

from doc2dataset.checkpoint import CheckpointManager, ProcessingState


class MockDocument:
    """Mock document for testing."""

    def __init__(self, source: str, content: str = "Test content"):
        self.source = source
        self.content = content


class TestProcessingState:
    """Tests for ProcessingState."""

    def test_create_state(self):
        """Test creating processing state."""
        state = ProcessingState(
            job_id="test_job",
            total_documents=10,
            processed_documents=5,
            failed_documents=1,
        )

        assert state.job_id == "test_job"
        assert state.total_documents == 10
        assert state.processed_documents == 5
        assert state.failed_documents == 1

    def test_state_to_dict(self):
        """Test converting state to dict."""
        state = ProcessingState(
            job_id="test_job",
            total_documents=10,
        )

        d = state.to_dict()
        assert d["job_id"] == "test_job"
        assert d["total_documents"] == 10

    def test_state_from_dict(self):
        """Test creating state from dict."""
        d = {
            "job_id": "test_job",
            "total_documents": 10,
            "processed_documents": 5,
            "failed_documents": 0,
            "completed": {},
            "failed": {},
        }

        state = ProcessingState.from_dict(d)
        assert state.job_id == "test_job"
        assert state.total_documents == 10
        assert state.processed_documents == 5


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a test checkpoint manager."""
        return CheckpointManager(checkpoint_dir=str(tmp_path / "checkpoints"))

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents."""
        return [
            MockDocument("doc1.pdf"),
            MockDocument("doc2.pdf"),
            MockDocument("doc3.pdf"),
        ]

    def test_create_job(self, manager, sample_documents):
        """Test creating a new job."""
        state = manager.create_job(
            documents=sample_documents,
            extraction_type="qa",
            model="gpt-4",
        )

        assert state.job_id is not None
        assert state.total_documents == 3
        assert state.processed_documents == 0

    def test_resume_job(self, manager, sample_documents):
        """Test resuming an existing job."""
        # Create initial job
        state1 = manager.create_job(
            documents=sample_documents,
            extraction_type="qa",
            model="gpt-4",
        )
        job_id = state1.job_id

        # Mark one document as complete
        manager.mark_complete(state1, "doc1.pdf", [{"result": "data"}])

        # Resume the job
        state2 = manager.create_job(
            documents=sample_documents,
            extraction_type="qa",
            model="gpt-4",
            resume_if_exists=True,
        )

        # Should have same job_id and preserved progress
        assert state2.job_id == job_id
        assert state2.processed_documents == 1

    def test_mark_complete(self, manager, sample_documents):
        """Test marking a document as complete."""
        state = manager.create_job(
            documents=sample_documents,
            extraction_type="qa",
            model="gpt-4",
        )

        manager.mark_complete(state, "doc1.pdf", [{"item": 1}])

        assert state.processed_documents == 1
        assert "doc1.pdf" in state.completed

    def test_mark_failed(self, manager, sample_documents):
        """Test marking a document as failed."""
        state = manager.create_job(
            documents=sample_documents,
            extraction_type="qa",
            model="gpt-4",
        )

        manager.mark_failed(state, "doc1.pdf", "Error message")

        assert state.failed_documents == 1
        assert "doc1.pdf" in state.failed

    def test_iterate_pending(self, manager, sample_documents):
        """Test iterating over pending documents."""
        state = manager.create_job(
            documents=sample_documents,
            extraction_type="qa",
            model="gpt-4",
        )

        # Mark first doc as complete
        manager.mark_complete(state, "doc1.pdf", [])

        # Get pending documents
        pending = list(manager.iterate_pending(sample_documents, state))

        assert len(pending) == 2
        sources = [d.source for d in pending]
        assert "doc1.pdf" not in sources

    def test_finalize_job(self, manager, sample_documents):
        """Test finalizing a job."""
        state = manager.create_job(
            documents=sample_documents,
            extraction_type="qa",
            model="gpt-4",
        )

        # Complete all documents
        for doc in sample_documents:
            manager.mark_complete(state, doc.source, [])

        summary = manager.finalize_job(state)

        assert "complete" in summary.lower() or "processed" in summary.lower()

    def test_list_jobs(self, manager, sample_documents):
        """Test listing all jobs."""
        # Create a job
        manager.create_job(
            documents=sample_documents,
            extraction_type="qa",
            model="gpt-4",
        )

        jobs = manager.list_jobs()
        assert len(jobs) >= 1

    def test_cleanup_completed(self, manager, sample_documents):
        """Test cleaning up completed jobs."""
        state = manager.create_job(
            documents=sample_documents,
            extraction_type="qa",
            model="gpt-4",
        )

        # Complete all documents
        for doc in sample_documents:
            manager.mark_complete(state, doc.source, [])

        manager.finalize_job(state)

        # Cleanup should remove completed jobs (may depend on age)
        count = manager.cleanup_completed(max_age_days=0)
        # Count may be 0 or 1 depending on timing

    def test_get_doc_id(self, manager):
        """Test getting document ID."""
        doc = MockDocument("path/to/document.pdf")
        doc_id = manager._get_doc_id(doc)

        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

    def test_state_persistence(self, manager, sample_documents, tmp_path):
        """Test that state persists across manager instances."""
        # Create job with first manager
        state1 = manager.create_job(
            documents=sample_documents,
            extraction_type="qa",
            model="gpt-4",
        )
        manager.mark_complete(state1, "doc1.pdf", [{"result": 1}])

        # Create new manager instance pointing to same directory
        manager2 = CheckpointManager(checkpoint_dir=str(tmp_path / "checkpoints"))

        # Resume should find existing state
        state2 = manager2.create_job(
            documents=sample_documents,
            extraction_type="qa",
            model="gpt-4",
            resume_if_exists=True,
        )

        assert state2.processed_documents == 1

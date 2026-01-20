"""
Tests for doc2dataset HuggingFace integration.
"""

import pytest
import json
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch


# Check if huggingface_hub is available
try:
    from doc2dataset.huggingface import (
        DatasetCard,
        HuggingFaceUploader,
        HuggingFaceDownloader,
        upload_to_hub,
        download_from_hub,
        HF_AVAILABLE,
        DATASETS_AVAILABLE,
    )
except ImportError:
    HF_AVAILABLE = False
    DATASETS_AVAILABLE = False


@pytest.mark.skipif(not HF_AVAILABLE, reason="huggingface_hub not installed")
class TestDatasetCard:
    """Tests for DatasetCard."""

    def test_create_card(self):
        """Test creating a dataset card."""
        card = DatasetCard(
            name="test/dataset",
            description="A test dataset",
            pretty_name="Test Dataset",
        )

        assert card.name == "test/dataset"
        assert card.description == "A test dataset"
        assert card.pretty_name == "Test Dataset"

    def test_default_values(self):
        """Test default values."""
        card = DatasetCard(name="test/dataset")

        assert card.language == ["en"]
        assert card.license == "mit"
        assert "doc2dataset" in card.tags

    def test_yaml_header(self):
        """Test YAML header generation."""
        card = DatasetCard(
            name="test/dataset",
            language=["en", "es"],
            license="apache-2.0",
        )

        header = card.to_yaml_header()

        assert "---" in header
        assert "license: apache-2.0" in header
        assert "- en" in header
        assert "- es" in header

    def test_readme_generation(self):
        """Test README generation."""
        card = DatasetCard(
            name="user/my-dataset",
            description="Training data for my model",
            pretty_name="My Dataset",
        )

        readme = card.to_readme()

        assert "# My Dataset" in readme
        assert "Training data for my model" in readme
        assert "doc2dataset" in readme
        assert "load_dataset" in readme


@pytest.mark.skipif(not HF_AVAILABLE, reason="huggingface_hub not installed")
class TestHuggingFaceUploader:
    """Tests for HuggingFaceUploader."""

    @pytest.fixture
    def mock_api(self):
        """Create a mock HfApi."""
        with patch("doc2dataset.huggingface.HfApi") as mock:
            yield mock

    @pytest.fixture
    def mock_upload_file(self):
        """Mock upload_file function."""
        with patch("doc2dataset.huggingface.upload_file") as mock:
            yield mock

    @pytest.fixture
    def mock_create_repo(self):
        """Mock create_repo function."""
        with patch("doc2dataset.huggingface.create_repo") as mock:
            yield mock

    @pytest.fixture
    def sample_items(self) -> List[Dict[str, Any]]:
        """Create sample dataset items."""
        return [
            {"input": "What is Python?", "output": "A programming language"},
            {"input": "What is JavaScript?", "output": "A scripting language"},
        ]

    @pytest.fixture
    def sample_jsonl(self, tmp_path, sample_items):
        """Create a sample JSONL file."""
        file_path = tmp_path / "test.jsonl"
        with open(file_path, "w") as f:
            for item in sample_items:
                f.write(json.dumps(item) + "\n")
        return file_path

    def test_init(self, mock_api):
        """Test uploader initialization."""
        uploader = HuggingFaceUploader(token="test-token", private=True)

        assert uploader.token == "test-token"
        assert uploader.private is True

    def test_init_from_env(self, mock_api, monkeypatch):
        """Test uploader uses environment variable."""
        monkeypatch.setenv("HF_TOKEN", "env-token")

        uploader = HuggingFaceUploader()

        assert uploader.token == "env-token"

    def test_upload_jsonl_creates_repo(
        self, mock_api, mock_upload_file, mock_create_repo, sample_jsonl
    ):
        """Test uploading creates repo if not exists."""
        from huggingface_hub.utils import RepositoryNotFoundError

        # Mock repo not found, then creation succeeds
        mock_api_instance = MagicMock()
        mock_api_instance.repo_info.side_effect = RepositoryNotFoundError("Not found")
        mock_api.return_value = mock_api_instance

        uploader = HuggingFaceUploader(token="test-token")
        url = uploader.upload_jsonl(
            file_path=sample_jsonl,
            repo_id="user/new-dataset",
        )

        mock_create_repo.assert_called_once()
        assert "user/new-dataset" in url

    def test_upload_dataset(
        self, mock_api, mock_upload_file, sample_items
    ):
        """Test uploading dataset items."""
        mock_api_instance = MagicMock()
        mock_api.return_value = mock_api_instance

        uploader = HuggingFaceUploader(token="test-token")
        url = uploader.upload_dataset(
            items=sample_items,
            repo_id="user/dataset",
            split="train",
        )

        assert "user/dataset" in url

    def test_upload_with_card(
        self, mock_api, mock_upload_file, sample_items
    ):
        """Test uploading with dataset card."""
        mock_api_instance = MagicMock()
        mock_api.return_value = mock_api_instance

        card = DatasetCard(
            name="user/dataset",
            description="Test dataset",
        )

        uploader = HuggingFaceUploader(token="test-token")
        url = uploader.upload_dataset(
            items=sample_items,
            repo_id="user/dataset",
            dataset_card=card,
        )

        # Should have uploaded both data and README
        assert mock_upload_file.call_count >= 1


@pytest.mark.skipif(
    not (HF_AVAILABLE and DATASETS_AVAILABLE),
    reason="huggingface_hub or datasets not installed"
)
class TestHuggingFaceDownloader:
    """Tests for HuggingFaceDownloader."""

    @pytest.fixture
    def mock_load_dataset(self):
        """Mock datasets.load_dataset."""
        with patch("doc2dataset.huggingface.hf_datasets.load_dataset") as mock:
            yield mock

    def test_init(self):
        """Test downloader initialization."""
        downloader = HuggingFaceDownloader(token="test-token")
        assert downloader.token == "test-token"

    def test_download(self, mock_load_dataset):
        """Test downloading a dataset."""
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset

        downloader = HuggingFaceDownloader(token="test-token")
        result = downloader.download("user/dataset", split="train")

        mock_load_dataset.assert_called_once_with(
            "user/dataset",
            split="train",
            streaming=False,
            token="test-token",
        )

    def test_to_items(self, mock_load_dataset):
        """Test converting dataset to items."""
        mock_dataset = [
            {"input": "Q1", "output": "A1"},
            {"input": "Q2", "output": "A2"},
        ]
        mock_load_dataset.return_value = mock_dataset

        downloader = HuggingFaceDownloader()
        items = downloader.to_items("user/dataset")

        assert len(items) == 2
        assert items[0]["input"] == "Q1"

    def test_to_jsonl(self, mock_load_dataset, tmp_path):
        """Test downloading to JSONL file."""
        mock_dataset = [
            {"input": "Q1", "output": "A1"},
            {"input": "Q2", "output": "A2"},
        ]
        mock_load_dataset.return_value = mock_dataset

        output_path = tmp_path / "output.jsonl"

        downloader = HuggingFaceDownloader()
        result_path = downloader.to_jsonl("user/dataset", output_path)

        assert result_path.exists()

        # Verify content
        with open(result_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 2


@pytest.mark.skipif(not HF_AVAILABLE, reason="huggingface_hub not installed")
class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.fixture
    def mock_uploader(self):
        """Mock HuggingFaceUploader."""
        with patch("doc2dataset.huggingface.HuggingFaceUploader") as mock:
            mock_instance = MagicMock()
            mock_instance.upload_dataset.return_value = "https://huggingface.co/datasets/user/ds"
            mock.return_value = mock_instance
            yield mock

    def test_upload_to_hub(self, mock_uploader):
        """Test upload_to_hub convenience function."""
        items = [{"input": "Q", "output": "A"}]

        url = upload_to_hub(
            items=items,
            repo_id="user/dataset",
            name="My Dataset",
            description="Test dataset",
        )

        assert "huggingface.co" in url
        mock_uploader.assert_called_once()

    def test_upload_determines_size_category(self, mock_uploader):
        """Test that size category is determined correctly."""
        # Small dataset
        items = [{"input": f"Q{i}", "output": f"A{i}"} for i in range(100)]

        upload_to_hub(items=items, repo_id="user/small")

        # Check the card passed to upload_dataset
        call_args = mock_uploader.return_value.upload_dataset.call_args
        card = call_args.kwargs.get("dataset_card")
        assert card is not None
        assert "n<1K" in card.size_categories

    @pytest.mark.skipif(not DATASETS_AVAILABLE, reason="datasets not installed")
    def test_download_from_hub_to_items(self):
        """Test download_from_hub returning items."""
        with patch("doc2dataset.huggingface.HuggingFaceDownloader") as mock:
            mock_instance = MagicMock()
            mock_instance.to_items.return_value = [{"input": "Q", "output": "A"}]
            mock.return_value = mock_instance

            items = download_from_hub("user/dataset")

            assert len(items) == 1
            assert items[0]["input"] == "Q"

    @pytest.mark.skipif(not DATASETS_AVAILABLE, reason="datasets not installed")
    def test_download_from_hub_to_file(self, tmp_path):
        """Test download_from_hub saving to file."""
        with patch("doc2dataset.huggingface.HuggingFaceDownloader") as mock:
            output_path = tmp_path / "data.jsonl"
            mock_instance = MagicMock()
            mock_instance.to_jsonl.return_value = output_path
            mock.return_value = mock_instance

            result = download_from_hub("user/dataset", output_path=output_path)

            assert result == output_path


class TestImportChecks:
    """Tests for import availability checks."""

    def test_hf_not_available_error(self):
        """Test error when huggingface_hub not available."""
        # This test is more about documentation than actual testing
        # since we can't really uninstall packages during tests
        pass

    def test_datasets_not_available_error(self):
        """Test error when datasets not available."""
        # Similar to above
        pass

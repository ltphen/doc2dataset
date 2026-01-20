"""
HuggingFace Hub integration for doc2dataset.

This module provides functionality to upload and download datasets
from HuggingFace Hub.
"""

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
    from huggingface_hub.utils import RepositoryNotFoundError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import datasets as hf_datasets
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


def _check_hf_available():
    """Check if huggingface_hub is installed."""
    if not HF_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required for HuggingFace integration. "
            "Install it with: pip install huggingface_hub"
        )


def _check_datasets_available():
    """Check if datasets library is installed."""
    if not DATASETS_AVAILABLE:
        raise ImportError(
            "datasets library is required for this feature. "
            "Install it with: pip install datasets"
        )


@dataclass
class DatasetCard:
    """HuggingFace dataset card metadata."""

    name: str
    description: str = ""
    language: List[str] = field(default_factory=lambda: ["en"])
    license: str = "mit"
    task_categories: List[str] = field(default_factory=lambda: ["text-generation"])
    task_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=lambda: ["doc2dataset", "fine-tuning"])
    size_categories: List[str] = field(default_factory=lambda: ["n<1K"])
    source_datasets: List[str] = field(default_factory=list)
    pretty_name: Optional[str] = None

    def to_yaml_header(self) -> str:
        """Generate YAML header for dataset card."""
        lines = ["---"]
        lines.append(f"language:")
        for lang in self.language:
            lines.append(f"  - {lang}")
        lines.append(f"license: {self.license}")
        lines.append(f"task_categories:")
        for cat in self.task_categories:
            lines.append(f"  - {cat}")
        if self.task_ids:
            lines.append(f"task_ids:")
            for tid in self.task_ids:
                lines.append(f"  - {tid}")
        lines.append(f"tags:")
        for tag in self.tags:
            lines.append(f"  - {tag}")
        lines.append(f"size_categories:")
        for size in self.size_categories:
            lines.append(f"  - {size}")
        if self.source_datasets:
            lines.append(f"source_datasets:")
            for src in self.source_datasets:
                lines.append(f"  - {src}")
        if self.pretty_name:
            lines.append(f"pretty_name: {self.pretty_name}")
        lines.append("---")
        return "\n".join(lines)

    def to_readme(self) -> str:
        """Generate full README.md content."""
        header = self.to_yaml_header()

        content = f"""
# {self.pretty_name or self.name}

{self.description}

## Dataset Description

This dataset was generated using [doc2dataset](https://github.com/doc2dataset/doc2dataset),
a tool for converting documents into LLM fine-tuning datasets.

### Languages

{', '.join(self.language)}

### License

{self.license}

## Dataset Structure

### Data Fields

The dataset contains the following fields:
- `input`: The input/question/instruction
- `output`: The expected response/answer
- `type`: The extraction type (qa, rules, facts, etc.)
- `source`: The source document

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{self.name}")
```

## Citation

If you use this dataset, please cite doc2dataset:

```bibtex
@software{{doc2dataset,
  title = {{doc2dataset: Document to Dataset Converter}},
  url = {{https://github.com/doc2dataset/doc2dataset}}
}}
```
"""
        return header + content


class HuggingFaceUploader:
    """Upload datasets to HuggingFace Hub."""

    def __init__(
        self,
        token: Optional[str] = None,
        private: bool = False,
    ):
        """
        Initialize the uploader.

        Args:
            token: HuggingFace API token. If not provided, uses HF_TOKEN env var.
            private: Whether to create private repositories.
        """
        _check_hf_available()

        self.token = token or os.environ.get("HF_TOKEN")
        self.private = private
        self.api = HfApi(token=self.token)

    def upload_jsonl(
        self,
        file_path: Union[str, Path],
        repo_id: str,
        dataset_card: Optional[DatasetCard] = None,
        split: str = "train",
        commit_message: str = "Upload dataset via doc2dataset",
    ) -> str:
        """
        Upload a JSONL file to HuggingFace Hub.

        Args:
            file_path: Path to the JSONL file.
            repo_id: Repository ID (e.g., "username/dataset-name").
            dataset_card: Optional dataset card metadata.
            split: Dataset split name.
            commit_message: Commit message for the upload.

        Returns:
            URL to the uploaded dataset.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Create repository if it doesn't exist
        try:
            self.api.repo_info(repo_id=repo_id, repo_type="dataset")
        except RepositoryNotFoundError:
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=self.private,
                token=self.token,
            )

        # Upload the JSONL file
        upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=f"data/{split}.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
            token=self.token,
            commit_message=commit_message,
        )

        # Upload dataset card if provided
        if dataset_card:
            readme_content = dataset_card.to_readme()
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
                f.write(readme_content)
                readme_path = f.name

            try:
                upload_file(
                    path_or_fileobj=readme_path,
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=self.token,
                    commit_message="Update dataset card",
                )
            finally:
                os.unlink(readme_path)

        return f"https://huggingface.co/datasets/{repo_id}"

    def upload_dataset(
        self,
        items: List[Dict[str, Any]],
        repo_id: str,
        dataset_card: Optional[DatasetCard] = None,
        split: str = "train",
        format: str = "jsonl",
        commit_message: str = "Upload dataset via doc2dataset",
    ) -> str:
        """
        Upload a list of items to HuggingFace Hub.

        Args:
            items: List of dataset items.
            repo_id: Repository ID.
            dataset_card: Optional dataset card metadata.
            split: Dataset split name.
            format: Output format ("jsonl" or "parquet").
            commit_message: Commit message.

        Returns:
            URL to the uploaded dataset.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            if format == "jsonl":
                file_path = Path(tmpdir) / f"{split}.jsonl"
                with open(file_path, "w") as f:
                    for item in items:
                        f.write(json.dumps(item) + "\n")
                return self.upload_jsonl(
                    file_path=file_path,
                    repo_id=repo_id,
                    dataset_card=dataset_card,
                    split=split,
                    commit_message=commit_message,
                )
            elif format == "parquet":
                _check_datasets_available()
                # Convert to HuggingFace dataset and save as parquet
                hf_ds = hf_datasets.Dataset.from_list(items)
                file_path = Path(tmpdir) / f"{split}.parquet"
                hf_ds.to_parquet(str(file_path))

                # Create repository if needed
                try:
                    self.api.repo_info(repo_id=repo_id, repo_type="dataset")
                except RepositoryNotFoundError:
                    create_repo(
                        repo_id=repo_id,
                        repo_type="dataset",
                        private=self.private,
                        token=self.token,
                    )

                # Upload parquet file
                upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=f"data/{split}.parquet",
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=self.token,
                    commit_message=commit_message,
                )

                # Upload dataset card if provided
                if dataset_card:
                    readme_content = dataset_card.to_readme()
                    readme_path = Path(tmpdir) / "README.md"
                    with open(readme_path, "w") as f:
                        f.write(readme_content)

                    upload_file(
                        path_or_fileobj=str(readme_path),
                        path_in_repo="README.md",
                        repo_id=repo_id,
                        repo_type="dataset",
                        token=self.token,
                        commit_message="Update dataset card",
                    )

                return f"https://huggingface.co/datasets/{repo_id}"
            else:
                raise ValueError(f"Unsupported format: {format}")

    def upload_splits(
        self,
        splits: Dict[str, List[Dict[str, Any]]],
        repo_id: str,
        dataset_card: Optional[DatasetCard] = None,
        format: str = "jsonl",
        commit_message: str = "Upload dataset via doc2dataset",
    ) -> str:
        """
        Upload multiple splits to HuggingFace Hub.

        Args:
            splits: Dictionary mapping split names to items.
            repo_id: Repository ID.
            dataset_card: Optional dataset card metadata.
            format: Output format.
            commit_message: Commit message.

        Returns:
            URL to the uploaded dataset.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()

            # Create repository if needed
            try:
                self.api.repo_info(repo_id=repo_id, repo_type="dataset")
            except RepositoryNotFoundError:
                create_repo(
                    repo_id=repo_id,
                    repo_type="dataset",
                    private=self.private,
                    token=self.token,
                )

            # Write each split
            for split_name, items in splits.items():
                if format == "jsonl":
                    file_path = data_dir / f"{split_name}.jsonl"
                    with open(file_path, "w") as f:
                        for item in items:
                            f.write(json.dumps(item) + "\n")
                elif format == "parquet":
                    _check_datasets_available()
                    hf_ds = hf_datasets.Dataset.from_list(items)
                    file_path = data_dir / f"{split_name}.parquet"
                    hf_ds.to_parquet(str(file_path))

            # Upload the data folder
            upload_folder(
                folder_path=str(data_dir),
                path_in_repo="data",
                repo_id=repo_id,
                repo_type="dataset",
                token=self.token,
                commit_message=commit_message,
            )

            # Upload dataset card if provided
            if dataset_card:
                readme_content = dataset_card.to_readme()
                readme_path = Path(tmpdir) / "README.md"
                with open(readme_path, "w") as f:
                    f.write(readme_content)

                upload_file(
                    path_or_fileobj=str(readme_path),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=self.token,
                    commit_message="Update dataset card",
                )

            return f"https://huggingface.co/datasets/{repo_id}"


class HuggingFaceDownloader:
    """Download datasets from HuggingFace Hub."""

    def __init__(self, token: Optional[str] = None):
        """
        Initialize the downloader.

        Args:
            token: HuggingFace API token for private datasets.
        """
        _check_datasets_available()
        self.token = token or os.environ.get("HF_TOKEN")

    def download(
        self,
        repo_id: str,
        split: Optional[str] = None,
        streaming: bool = False,
    ) -> Union["hf_datasets.Dataset", "hf_datasets.DatasetDict"]:
        """
        Download a dataset from HuggingFace Hub.

        Args:
            repo_id: Repository ID.
            split: Optional split to download.
            streaming: Whether to stream the dataset.

        Returns:
            HuggingFace Dataset or DatasetDict.
        """
        return hf_datasets.load_dataset(
            repo_id,
            split=split,
            streaming=streaming,
            token=self.token,
        )

    def to_items(
        self,
        repo_id: str,
        split: str = "train",
    ) -> List[Dict[str, Any]]:
        """
        Download a dataset and convert to list of items.

        Args:
            repo_id: Repository ID.
            split: Split to download.

        Returns:
            List of dataset items.
        """
        ds = self.download(repo_id, split=split)
        return [dict(item) for item in ds]

    def to_jsonl(
        self,
        repo_id: str,
        output_path: Union[str, Path],
        split: str = "train",
    ) -> Path:
        """
        Download a dataset and save as JSONL.

        Args:
            repo_id: Repository ID.
            output_path: Output file path.
            split: Split to download.

        Returns:
            Path to the saved file.
        """
        output_path = Path(output_path)
        items = self.to_items(repo_id, split=split)

        with open(output_path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

        return output_path


def upload_to_hub(
    items: List[Dict[str, Any]],
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    name: Optional[str] = None,
    description: str = "",
    split: str = "train",
    format: str = "jsonl",
) -> str:
    """
    Convenience function to upload items to HuggingFace Hub.

    Args:
        items: List of dataset items.
        repo_id: Repository ID (e.g., "username/dataset-name").
        token: HuggingFace API token.
        private: Whether to create a private repository.
        name: Dataset name for the card.
        description: Dataset description.
        split: Dataset split name.
        format: Output format ("jsonl" or "parquet").

    Returns:
        URL to the uploaded dataset.
    """
    uploader = HuggingFaceUploader(token=token, private=private)

    # Create dataset card
    card = DatasetCard(
        name=repo_id,
        description=description,
        pretty_name=name or repo_id.split("/")[-1],
    )

    # Determine size category
    n_items = len(items)
    if n_items < 1000:
        card.size_categories = ["n<1K"]
    elif n_items < 10000:
        card.size_categories = ["1K<n<10K"]
    elif n_items < 100000:
        card.size_categories = ["10K<n<100K"]
    else:
        card.size_categories = ["n>100K"]

    return uploader.upload_dataset(
        items=items,
        repo_id=repo_id,
        dataset_card=card,
        split=split,
        format=format,
    )


def download_from_hub(
    repo_id: str,
    output_path: Optional[Union[str, Path]] = None,
    split: str = "train",
    token: Optional[str] = None,
) -> Union[List[Dict[str, Any]], Path]:
    """
    Convenience function to download a dataset from HuggingFace Hub.

    Args:
        repo_id: Repository ID.
        output_path: Optional path to save as JSONL.
        split: Split to download.
        token: HuggingFace API token for private datasets.

    Returns:
        List of items if no output_path, otherwise Path to saved file.
    """
    downloader = HuggingFaceDownloader(token=token)

    if output_path:
        return downloader.to_jsonl(repo_id, output_path, split=split)
    else:
        return downloader.to_items(repo_id, split=split)

#!/usr/bin/env python3
"""
Command-line interface for doc2dataset.

Usage:
    doc2dataset process ./documents ./output.jsonl --provider openai --model gpt-4
    doc2dataset process ./manual.pdf ./qa_data.jsonl --extractors qa
    doc2dataset analyze ./output.jsonl
    doc2dataset augment ./output.jsonl ./augmented.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from doc2dataset.processor import DocProcessor, process_documents
from doc2dataset.formatters import list_formats


def cmd_process(args):
    """Process documents command."""
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"Processing: {args.input}")
        print(f"Provider: {args.provider}")
        print(f"Model: {args.model or 'default'}")
        print(f"Extractors: {', '.join(args.extractors)}")
        print(f"Output format: {args.format}")
        if args.workers > 1:
            print(f"Workers: {args.workers}")
        print()

    try:
        # Create processor with new features
        processor = DocProcessor(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            extractors=args.extractors,
            chunk_size=args.chunk_size,
            max_items_per_chunk=args.max_items,
            enable_cost_estimation=args.estimate_cost,
            enable_checkpointing=not args.no_checkpoint,
            enable_quality_filter=not args.no_quality_filter,
            enable_attribution=args.with_attribution,
            enable_parallel=args.workers > 1,
            max_workers=args.workers,
            rate_limit=args.rate_limit,
            enable_analytics=True,
        )

        # Process based on input type
        if input_path.is_file():
            dataset = processor.process_document(input_path)
        else:
            dataset = processor.process_folder(
                input_path,
                recursive=not args.no_recursive,
                extensions=args.extensions,
                progress=not args.quiet,
                estimate_cost=args.estimate_cost,
                confirm_cost=not args.yes,
                resume=args.resume,
                apply_quality_filter=not args.no_quality_filter,
            )

        # Deduplicate
        dataset = dataset.deduplicate()

        # Export
        count = dataset.to_jsonl(output_path, format=args.format)

        # Save attribution if enabled
        if args.with_attribution and args.attribution_output:
            processor.save_attributions(args.attribution_output)
            if not args.quiet:
                print(f"Saved attributions to {args.attribution_output}")

        # Export analytics if requested
        if args.analytics_output:
            processor.export_analytics(args.analytics_output)
            if not args.quiet:
                print(f"Saved analytics to {args.analytics_output}")

        if not args.quiet:
            print(f"\nDone! Extracted {count} items to {args.output}")

            # Show analytics summary
            summary = processor.get_analytics_summary()
            if summary:
                print(f"  Documents processed: {summary.get('documents_processed', 0)}")
                print(f"  Total time: {summary.get('total_time_seconds', 0):.1f}s")
                if summary.get('errors', 0) > 0:
                    print(f"  Errors: {summary.get('errors', 0)}")

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_analyze(args):
    """Analyze dataset command."""
    from doc2dataset.analytics import analyze_jsonl_file, DatasetAnalyzer

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"Analyzing: {args.input}")
        print()

    try:
        # Load and analyze
        stats = analyze_jsonl_file(
            input_path,
            input_field=args.input_field,
            output_field=args.output_field,
        )

        # Generate report
        analyzer = DatasetAnalyzer()
        report = analyzer.generate_report(
            stats,
            format="markdown" if args.markdown else "text",
        )
        print(report)

        # Save report if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                f.write(report)
            if not args.quiet:
                print(f"\nSaved report to {args.output}")

        # JSON output
        if args.json:
            print("\n--- JSON ---")
            print(json.dumps(stats.to_dict(), indent=2))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_augment(args):
    """Augment dataset command."""
    from doc2dataset.augmentation import (
        ParaphraseAugmenter,
        SynonymAugmenter,
        BackTranslationAugmenter,
        AugmentationPipeline,
    )

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"Augmenting: {args.input}")
        print(f"Methods: {', '.join(args.methods)}")
        print(f"Factor: {args.factor}x")
        print()

    try:
        # Load dataset
        items = []
        with open(input_path, "r") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))

        original_count = len(items)

        # Build augmentation pipeline
        augmenters = []
        for method in args.methods:
            if method == "paraphrase":
                # Would need LLM function - skip if not available
                print(f"Warning: Paraphrase augmentation requires LLM. Skipping.")
            elif method == "synonym":
                augmenters.append(SynonymAugmenter(
                    fields=[args.input_field, args.output_field],
                    replacement_prob=0.1,
                ))
            elif method == "backtranslation":
                print(f"Warning: Back-translation requires LLM. Skipping.")

        if not augmenters:
            print("No augmenters available. Copying original data.")
            augmented = items
        else:
            pipeline = AugmentationPipeline(augmenters)
            augmented = pipeline.augment(items, multiplier=args.factor)

        # Save augmented dataset
        with open(output_path, "w") as f:
            for item in augmented:
                f.write(json.dumps(item) + "\n")

        if not args.quiet:
            print(f"\nDone! {original_count} -> {len(augmented)} items")
            print(f"Saved to {args.output}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_checkpoints(args):
    """Manage checkpoints command."""
    from doc2dataset.checkpoint import CheckpointManager

    manager = CheckpointManager(checkpoint_dir=args.checkpoint_dir)

    if args.action == "list":
        jobs = manager.list_jobs()
        if not jobs:
            print("No checkpoints found.")
            return

        print(f"Found {len(jobs)} checkpoint(s):\n")
        for job in jobs:
            status = "COMPLETE" if job.get("completed", False) else "IN PROGRESS"
            print(f"  {job.get('job_id', 'unknown')[:8]}...")
            print(f"    Status: {status}")
            print(f"    Progress: {job.get('processed', 0)}/{job.get('total', 0)} documents")
            print(f"    Created: {job.get('created_at', 'unknown')}")
            print()

    elif args.action == "cleanup":
        count = manager.cleanup_completed(max_age_days=args.max_age)
        print(f"Cleaned up {count} completed checkpoint(s)")

    elif args.action == "delete":
        if not args.job_id:
            print("Error: --job-id required for delete action", file=sys.stderr)
            sys.exit(1)
        try:
            manager.delete_job(args.job_id)
            print(f"Deleted checkpoint: {args.job_id}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


def cmd_upload(args):
    """Upload dataset to HuggingFace Hub."""
    try:
        from doc2dataset.huggingface import HuggingFaceUploader, DatasetCard
    except ImportError:
        print("Error: huggingface_hub is required for upload.", file=sys.stderr)
        print("Install it with: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    # Validate inputs
    input_files = args.inputs[:-1]  # All but last are inputs
    repo_id = args.inputs[-1]  # Last is repo_id

    if not input_files:
        print("Error: At least one input file required", file=sys.stderr)
        sys.exit(1)

    for f in input_files:
        if not Path(f).exists():
            print(f"Error: Input file does not exist: {f}", file=sys.stderr)
            sys.exit(1)

    # Determine splits
    if args.splits:
        if len(args.splits) != len(input_files):
            print(f"Error: Number of splits ({len(args.splits)}) must match number of inputs ({len(input_files)})", file=sys.stderr)
            sys.exit(1)
        splits = args.splits
    else:
        if len(input_files) == 1:
            splits = ["train"]
        else:
            splits = [f"split_{i}" for i in range(len(input_files))]

    if not args.quiet:
        print(f"Uploading to HuggingFace Hub: {repo_id}")
        print(f"Files: {', '.join(input_files)}")
        print(f"Splits: {', '.join(splits)}")
        if args.private:
            print("Visibility: Private")
        print()

    try:
        uploader = HuggingFaceUploader(token=args.token, private=args.private)

        # Create dataset card
        card = DatasetCard(
            name=repo_id,
            description=args.description,
            pretty_name=args.name or repo_id.split("/")[-1],
        )

        # Load and upload data
        if len(input_files) == 1:
            # Single file upload
            items = []
            with open(input_files[0], "r") as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))

            url = uploader.upload_dataset(
                items=items,
                repo_id=repo_id,
                dataset_card=card,
                split=splits[0],
                format=args.format,
            )
        else:
            # Multiple files - upload as splits
            splits_data = {}
            for input_file, split_name in zip(input_files, splits):
                items = []
                with open(input_file, "r") as f:
                    for line in f:
                        if line.strip():
                            items.append(json.loads(line))
                splits_data[split_name] = items

            url = uploader.upload_splits(
                splits=splits_data,
                repo_id=repo_id,
                dataset_card=card,
                format=args.format,
            )

        if not args.quiet:
            print(f"\nSuccess! Dataset uploaded to: {url}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_download(args):
    """Download dataset from HuggingFace Hub."""
    try:
        from doc2dataset.huggingface import HuggingFaceDownloader
    except ImportError:
        print("Error: datasets library is required for download.", file=sys.stderr)
        print("Install it with: pip install datasets", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"Downloading from HuggingFace Hub: {args.repo_id}")
        print(f"Split: {args.split}")
        print(f"Output: {args.output}")
        print()

    try:
        downloader = HuggingFaceDownloader(token=args.token)
        path = downloader.to_jsonl(
            repo_id=args.repo_id,
            output_path=output_path,
            split=args.split,
        )

        # Count items
        with open(path, "r") as f:
            count = sum(1 for line in f if line.strip())

        if not args.quiet:
            print(f"\nSuccess! Downloaded {count} items to {args.output}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Transform documents into LLM fine-tuning datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ========== PROCESS COMMAND ==========
    process_parser = subparsers.add_parser(
        "process",
        help="Process documents and extract training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a folder of PDFs
  doc2dataset process ./documents ./training.jsonl

  # Use specific extractors
  doc2dataset process ./docs ./output.jsonl --extractors qa rules facts

  # Use Anthropic
  doc2dataset process ./docs ./output.jsonl --provider anthropic --model claude-3-sonnet-20240229

  # Resume from checkpoint
  doc2dataset process ./docs ./output.jsonl --resume

  # Parallel processing with 10 workers
  doc2dataset process ./docs ./output.jsonl --workers 10

  # Skip cost estimation confirmation
  doc2dataset process ./docs ./output.jsonl --yes
        """,
    )

    process_parser.add_argument(
        "input",
        type=str,
        help="Input file or folder path",
    )

    process_parser.add_argument(
        "output",
        type=str,
        help="Output JSONL file path",
    )

    process_parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic", "litellm"],
        help="LLM provider (default: openai)",
    )

    process_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: provider default)",
    )

    process_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (or use environment variable)",
    )

    process_parser.add_argument(
        "--extractors",
        type=str,
        nargs="+",
        default=["qa", "rules", "facts"],
        choices=["qa", "rules", "facts", "instruction", "conversation", "summary"],
        help="Extractors to use (default: qa rules facts)",
    )

    process_parser.add_argument(
        "--format",
        type=str,
        default="openai",
        choices=list_formats(),
        help=f"Output format (default: openai). Available: {', '.join(list_formats())}",
    )

    process_parser.add_argument(
        "--chunk-size",
        type=int,
        default=3000,
        help="Maximum characters per chunk (default: 3000)",
    )

    process_parser.add_argument(
        "--max-items",
        type=int,
        default=10,
        help="Maximum items per chunk (default: 10)",
    )

    process_parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=None,
        help="File extensions to include (default: all supported)",
    )

    process_parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories",
    )

    # New options
    process_parser.add_argument(
        "--estimate-cost",
        action="store_true",
        default=True,
        help="Show cost estimate before processing (default: True)",
    )

    process_parser.add_argument(
        "--no-estimate-cost",
        action="store_false",
        dest="estimate_cost",
        help="Skip cost estimation",
    )

    process_parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompts",
    )

    process_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )

    process_parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpointing",
    )

    process_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )

    process_parser.add_argument(
        "--rate-limit",
        type=float,
        default=None,
        help="Rate limit for API calls (requests/second)",
    )

    process_parser.add_argument(
        "--no-quality-filter",
        action="store_true",
        help="Disable quality filtering",
    )

    process_parser.add_argument(
        "--with-attribution",
        action="store_true",
        help="Track source attribution",
    )

    process_parser.add_argument(
        "--attribution-output",
        type=str,
        default=None,
        help="Output file for attributions (JSON)",
    )

    process_parser.add_argument(
        "--analytics-output",
        type=str,
        default=None,
        help="Output file for processing analytics (JSON)",
    )

    process_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    process_parser.set_defaults(func=cmd_process)

    # ========== ANALYZE COMMAND ==========
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a dataset file",
        epilog="""
Examples:
  # Basic analysis
  doc2dataset analyze ./output.jsonl

  # Save report to file
  doc2dataset analyze ./output.jsonl --output report.md --markdown

  # JSON output
  doc2dataset analyze ./output.jsonl --json
        """,
    )

    analyze_parser.add_argument(
        "input",
        type=str,
        help="Input JSONL file to analyze",
    )

    analyze_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save report to file",
    )

    analyze_parser.add_argument(
        "--input-field",
        type=str,
        default="input",
        help="Field name for input/question (default: input)",
    )

    analyze_parser.add_argument(
        "--output-field",
        type=str,
        default="output",
        help="Field name for output/answer (default: output)",
    )

    analyze_parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output in markdown format",
    )

    analyze_parser.add_argument(
        "--json",
        action="store_true",
        help="Include JSON output",
    )

    analyze_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    analyze_parser.set_defaults(func=cmd_analyze)

    # ========== AUGMENT COMMAND ==========
    augment_parser = subparsers.add_parser(
        "augment",
        help="Augment a dataset with variations",
        epilog="""
Examples:
  # Synonym augmentation
  doc2dataset augment ./data.jsonl ./augmented.jsonl --methods synonym

  # Multiple methods, 3x augmentation
  doc2dataset augment ./data.jsonl ./augmented.jsonl --methods synonym paraphrase --factor 3
        """,
    )

    augment_parser.add_argument(
        "input",
        type=str,
        help="Input JSONL file",
    )

    augment_parser.add_argument(
        "output",
        type=str,
        help="Output JSONL file",
    )

    augment_parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["synonym"],
        choices=["paraphrase", "synonym", "backtranslation"],
        help="Augmentation methods (default: synonym)",
    )

    augment_parser.add_argument(
        "--factor",
        type=int,
        default=2,
        help="Augmentation factor (default: 2x)",
    )

    augment_parser.add_argument(
        "--input-field",
        type=str,
        default="input",
        help="Field name for input (default: input)",
    )

    augment_parser.add_argument(
        "--output-field",
        type=str,
        default="output",
        help="Field name for output (default: output)",
    )

    augment_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    augment_parser.set_defaults(func=cmd_augment)

    # ========== CHECKPOINTS COMMAND ==========
    checkpoint_parser = subparsers.add_parser(
        "checkpoints",
        help="Manage processing checkpoints",
        epilog="""
Examples:
  # List checkpoints
  doc2dataset checkpoints list

  # Clean up old checkpoints
  doc2dataset checkpoints cleanup --max-age 7

  # Delete specific checkpoint
  doc2dataset checkpoints delete --job-id abc123
        """,
    )

    checkpoint_parser.add_argument(
        "action",
        choices=["list", "cleanup", "delete"],
        help="Checkpoint action",
    )

    checkpoint_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./.doc2dataset_checkpoints",
        help="Checkpoint directory",
    )

    checkpoint_parser.add_argument(
        "--max-age",
        type=int,
        default=7,
        help="Max age in days for cleanup (default: 7)",
    )

    checkpoint_parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Job ID for delete action",
    )

    checkpoint_parser.set_defaults(func=cmd_checkpoints)

    # ========== UPLOAD COMMAND (HuggingFace) ==========
    upload_parser = subparsers.add_parser(
        "upload",
        help="Upload dataset to HuggingFace Hub",
        epilog="""
Examples:
  # Upload to HuggingFace
  doc2dataset upload ./training.jsonl username/my-dataset

  # Upload with custom name and description
  doc2dataset upload ./training.jsonl username/my-dataset --name "My Dataset" --description "Training data"

  # Upload as private
  doc2dataset upload ./training.jsonl username/my-dataset --private

  # Upload with train/val splits
  doc2dataset upload ./train.jsonl ./val.jsonl username/my-dataset --splits train validation
        """,
    )

    upload_parser.add_argument(
        "inputs",
        type=str,
        nargs="+",
        help="Input JSONL file(s)",
    )

    upload_parser.add_argument(
        "repo_id",
        type=str,
        help="HuggingFace repo ID (e.g., username/dataset-name)",
    )

    upload_parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        help="Split names for multiple inputs (default: train)",
    )

    upload_parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Dataset display name",
    )

    upload_parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Dataset description",
    )

    upload_parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository",
    )

    upload_parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (or use HF_TOKEN env var)",
    )

    upload_parser.add_argument(
        "--format",
        type=str,
        default="jsonl",
        choices=["jsonl", "parquet"],
        help="Upload format (default: jsonl)",
    )

    upload_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    upload_parser.set_defaults(func=cmd_upload)

    # ========== DOWNLOAD COMMAND (HuggingFace) ==========
    download_parser = subparsers.add_parser(
        "download",
        help="Download dataset from HuggingFace Hub",
        epilog="""
Examples:
  # Download to JSONL
  doc2dataset download username/my-dataset ./data.jsonl

  # Download specific split
  doc2dataset download username/my-dataset ./data.jsonl --split validation
        """,
    )

    download_parser.add_argument(
        "repo_id",
        type=str,
        help="HuggingFace repo ID",
    )

    download_parser.add_argument(
        "output",
        type=str,
        help="Output JSONL file",
    )

    download_parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to download (default: train)",
    )

    download_parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token for private datasets",
    )

    download_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    download_parser.set_defaults(func=cmd_download)

    # ========== VERSION ==========
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.2.0",
    )

    # Parse and execute
    args = parser.parse_args()

    # Handle legacy usage (no subcommand)
    if args.command is None:
        # Check if first positional arg looks like a path
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            # Legacy mode: treat as process command
            sys.argv.insert(1, "process")
            args = parser.parse_args()
        else:
            parser.print_help()
            sys.exit(0)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

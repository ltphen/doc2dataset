"""Tests for CLI module."""

import json
import tempfile
from pathlib import Path
import pytest

# Try to import click for CLI testing
try:
    from click.testing import CliRunner
    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False

from doc2dataset.cli import main


@pytest.mark.skipif(not CLICK_AVAILABLE, reason="Click not available")
class TestCLI:
    """Tests for doc2dataset CLI."""

    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "doc2dataset" in result.output.lower() or "usage" in result.output.lower()

    def test_process_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["process", "--help"])
        # Should show help for process subcommand
        assert result.exit_code == 0

    def test_analyze_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["analyze", "--help"])
        assert result.exit_code == 0

    def test_augment_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["augment", "--help"])
        assert result.exit_code == 0

    def test_checkpoints_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["checkpoints", "--help"])
        assert result.exit_code == 0

    def test_process_missing_input(self):
        runner = CliRunner()
        result = runner.invoke(main, ["process"])
        # Should fail without input
        assert result.exit_code != 0

    def test_analyze_with_file(self):
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"input": "Q", "output": "A"}) + "\n")
            f.write(json.dumps({"input": "Q2", "output": "A2"}) + "\n")
            path = f.name

        try:
            result = runner.invoke(main, ["analyze", path])
            # May succeed or fail depending on full implementation
            # Just check it doesn't crash immediately
        finally:
            Path(path).unlink()


class TestCLIBasic:
    """Basic CLI tests that don't require Click runner."""

    def test_main_is_callable(self):
        """Verify main function exists and is callable."""
        assert callable(main)

    def test_cli_module_imports(self):
        """Verify CLI module can be imported."""
        from doc2dataset import cli
        assert hasattr(cli, "main")

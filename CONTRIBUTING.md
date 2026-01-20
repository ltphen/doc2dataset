# Contributing to doc2dataset

Thank you for your interest in contributing to doc2dataset! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.9 or higher
- pip or another Python package manager

### Setting up the development environment

1. Clone the repository:
   ```bash
   git clone https://github.com/ltphen/doc2dataset.git
   cd doc2dataset
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode with dev dependencies:
   ```bash
   pip install -e ".[dev,all]"
   ```

## Code Style

We use the following tools to maintain code quality:

- **black** for code formatting (line length: 100)
- **ruff** for linting
- **mypy** for type checking

### Running code quality checks

```bash
# Format code
black doc2dataset tests

# Lint code
ruff check doc2dataset tests

# Type checking
mypy doc2dataset
```

## Testing

We use pytest for testing. Run the test suite with:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=doc2dataset

# Run specific test file
pytest tests/test_extractors.py

# Run specific test
pytest tests/test_extractors.py::TestQAExtractor::test_basic_extraction
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test names that explain what is being tested
- Mock external dependencies (LLM calls, file I/O) when possible

Example:

```python
class TestMyExtractor:
    """Tests for MyExtractor."""

    def test_basic_extraction(self):
        """Test that basic extraction works."""
        def mock_llm(prompt):
            return '{"question": "Q", "answer": "A"}'

        extractor = MyExtractor(llm_fn=mock_llm)
        result = extractor.extract("Document content")

        assert len(result) > 0
```

## Making Changes

### Branching Strategy

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes in small, focused commits

3. Write clear commit messages following conventional commits:
   - `feat: add new document loader for HTML`
   - `fix: handle empty documents in processor`
   - `docs: update CLI documentation`
   - `test: add tests for augmentation module`
   - `refactor: improve checkpoint serialization`

### Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Add tests for new functionality
4. Update CHANGELOG.md if applicable
5. Submit a pull request with a clear description of changes

## Architecture Guidelines

### Adding New Loaders

To add a new document loader:

1. Create a class inheriting from `BaseLoader`
2. Implement the `load()` method
3. Add to `loaders.py` exports
4. Add tests in `tests/test_loaders.py`

```python
class HTMLLoader(BaseLoader):
    """Load HTML documents."""

    def load(self, file_path: str) -> Document:
        # Implementation
        pass
```

### Adding New Extractors

To add a new extraction type:

1. Create a class inheriting from `BaseExtractor`
2. Implement the `extract()` method
3. Define appropriate prompts
4. Add tests

### Adding New Formatters

To add a new output format:

1. Create a class inheriting from `BaseFormatter`
2. Implement the `format()` method
3. Add to `formatters.py`

## Directory Structure

```
doc2dataset/
├── __init__.py          # Public API exports
├── processor.py         # Main orchestration
├── loaders.py           # Document loaders
├── extractors.py        # Extraction types
├── formatters.py        # Output formatters
├── cli.py               # CLI interface
├── quality.py           # Quality filtering
├── parallel.py          # Parallel processing
└── ...
```

## CLI Development

When modifying the CLI:

1. Use Click decorators for commands
2. Provide helpful descriptions and examples
3. Support both positional and flag-based arguments
4. Include `--help` documentation

## Reporting Issues

When reporting issues, please include:

1. Python version and OS
2. doc2dataset version
3. Minimal reproducible example
4. Expected vs actual behavior
5. Full error traceback (if applicable)
6. Sample document (if relevant and non-sensitive)

## Questions?

If you have questions, feel free to:
- Open a GitHub issue
- Start a discussion in the repository

Thank you for contributing!

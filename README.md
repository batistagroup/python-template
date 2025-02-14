# Batista Template

A modern Python project template that provides a solid foundation for building Python packages with best practices and development tools.

## Features

- 🚀 Fast dependency management with `uv`
- 📚 Documentation with MkDocs and Material theme
- ✨ Code quality tools (Ruff for linting and formatting)
- 🔄 GitHub Actions for CI/CD
- 📦 Modern Python packaging with `pyproject.toml`
- 🧪 Testing infrastructure
- 📝 Pre-commit hooks for code quality

## Quick Start

This project uses `uv` for fast and reliable Python package management:

```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install the package and all development dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Documentation

After installing dependencies, you can run the documentation locally:

```bash
mkdocs serve
```

Then open your browser at `http://127.0.0.1:8000`

## Project Structure

```md
.
├── data/           # Data files and resources
├── docs/           # Documentation files (MkDocs)
├── scripts/        # Utility and automation scripts
├── src/            # Source code
│   └── batistatemplate/
├── tests/          # Test files
├── .github/        # GitHub Actions workflows
├── mkdocs.yml      # MkDocs configuration
├── pyproject.toml  # Project dependencies and settings
└── .pre-commit-config.yaml  # Pre-commit hooks configuration
```

## Development

For detailed development instructions, please refer to our [documentation](docs/index.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

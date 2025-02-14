# Welcome to Batista Template

This is a Python project template that provides a solid foundation for building Python packages with modern development tools and best practices.

## Installation

This project uses `uv` for fast and reliable Python package management. To set up your development environment:

```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install the package and all development dependencies
uv pip install -e ".[dev]"
# Initiate pre-commit checks
pre-commit install
```

This will install all necessary dependencies, including development tools like pre-commit hooks, testing frameworks, and documentation generators.

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

## Development Setup

### Documentation

This project uses MkDocs with the Material theme for documentation. To work with the documentation locally:

1. Make sure you have all development dependencies installed
2. Run the documentation server:

   ```bash
   mkdocs serve
   ```

3. Open your browser and navigate to `http://127.0.0.1:8000`

The documentation will automatically reload when you make changes to the markdown files.

### Code Quality Tools

We use pre-commit hooks to ensure code quality and consistency. The following tools are configured in `.pre-commit-config.yaml`:

- **Ruff**: A fast Python linter and formatter
  - Runs linting checks with auto-fix capability
  - Handles code formatting

After installing the development dependencies (as described in the Installation section), enable the pre-commit hooks by running:

```bash
pre-commit install
```

Now the hooks will run automatically on every commit, ensuring code quality and consistency.

### Managing Dependencies

This project uses `uv` for fast and reliable dependency management. Here's how to manage your dependencies:

#### Adding New Dependencies

To add a new package dependency:

```bash
uv add package_name
# Add a development dependency
uv add --dev package_name
```

This will:

1. Install the package in your virtual environment
2. Update your `pyproject.toml` with the new dependency
3. Update the `uv.lock` file with exact versions

#### Synchronizing Dependencies

If you pull changes that include new dependencies or switch branches, synchronize your environment:

```bash
uv sync
```

This ensures your virtual environment exactly matches the dependencies specified in the lock file, removing any packages you don't need and installing any that are missing.

### Writing Documentation

This project follows a structured approach to documentation. Each module should have its own markdown file in the `docs/batistatemplate/` directory. Documentation files might include:

1. **Overview**: A brief description of the module's purpose and key features
2. **Concepts**: Explanation of important concepts and design decisions
3. **Examples**: Code examples showing common usage patterns
4. **Source Code**: Auto-generated documentation from source code annotations

#### Source Code Documentation

Each documentation file should end with a Source Code section that imports and displays the module's classes and functions. Use the following structure:

```markdown
## Source Code

::: batistatemplate.module_name
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - ClassName1
        - ClassName2
        - function_name1
        - function_name2
```

## Getting Started

For more detailed information about specific components and usage examples, please navigate through the documentation using the navigation menu.

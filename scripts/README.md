# Scripts Directory

This directory contains user-facing Python scripts that demonstrate how to use the package. While the core functionality lives in `src/`, these scripts show practical examples of how to:

- Call and combine package features
- Configure common use cases
- Customize package behavior

## Available Scripts

- `example-use.py`: Shows basic package usage with common configurations

## Running Scripts

Make sure you have the package installed:

```bash
source .venv/bin/activate
uv pip install -e ".[dev]"
python scripts/example-use.py
```

## Creating New Scripts

When adding new scripts:

1. Follow the existing naming convention
2. Include docstrings and comments explaining the script's purpose
3. Add error handling and logging where appropriate
4. Document any command-line arguments or configuration options
5. Consider adding the script to the documentation with usage examples

## Best Practices

- Keep scripts focused on a single task or example
- Include proper error handling and user feedback
- Document dependencies and requirements
- Use the project's logging configuration
- Follow the project's code style guidelines

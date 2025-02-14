# Data Directory

This directory is intended for storing data files and resources that are used by the project. This can include:

- Sample data files
- Configuration files
- Resource files
- Test data
- Data templates

## Structure

```md
data/
├── README.md          # This file
├── sample/            # Sample data files (if any)
└── config/            # Configuration files (if any)
```

## Usage

When adding data files to this directory:

1. Ensure the files are in an appropriate format (CSV, JSON, YAML, etc.)
2. Document any data schema or format requirements
3. If the data is large, consider adding it to `.gitignore` and providing download instructions
4. Include sample data for testing and documentation purposes

## Data Guidelines

- Keep sensitive data out of version control
- Document data sources and any licensing requirements
- Maintain a clear structure for different types of data
- Include README files in subdirectories as needed

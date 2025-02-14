# GitHub Workflows

This project uses several GitHub Actions workflows to automate various development and release processes. Here's a detailed overview of each workflow:

## Code Quality (`quality.yml`)

This workflow ensures code quality standards are maintained:

- **Trigger**: Runs on every push and pull request
- **Actions**:
  - Runs Ruff for linting and formatting checks
  - Runs MyPy for static type checking
- **Environment**: Uses Python 3.12 and uv for dependency management

## Tests (`tests.yml`)

Handles automated testing of the codebase:

- **Trigger**: Runs on every push and pull request
- **Actions**:
  - Sets up Python 3.12 environment
  - Installs project dependencies using uv
  - Runs pytest with verbose output
- **Features**:
  - Caches dependencies for faster workflow execution
  - Uses virtual environment for isolated testing

## Release Management

### Release Drafter (`release-drafter.yml`)

Automates the creation of release notes and manages PR labeling:

- **Triggers**:
  - On push to main branch
  - On pull request events (opened, reopened, synchronize)
  - Manual trigger (workflow_dispatch)
- **Actions**:
  - Automatically assigns labels to PRs based on commit message prefixes
  - Drafts release notes based on PR labels
- **Commit Prefix to Label Mapping**:
  - `BREAKING CHANGE`, `BREAKING`, `MAJOR` → breaking-change
  - `feat`, `FEAT`, `Feature` → new-feature
  - `fix`, `FIX`, `Fixed` → bugfix
  - `enhance`, `ENHANCE`, `improvement` → enhancement
  - `refactor`, `REFACTOR` → refactor
  - `perf`, `PERFORMANCE` → performance
  - `docs`, `DOC` → documentation
  - `tests`, `TESTS` → tests
- **Categories**:
  - 🚨 Breaking changes
  - ✨ New features
  - 🐛 Bug fixes
  - 🚀 Enhancements
  - 🧰 Maintenance
  - 📚 Documentation
  - ⬆️ Dependency updates
- **Version Resolution**:
  - Major: breaking changes
  - Minor: new features, enhancements
  - Patch: bugfixes, default updates

### Publish Release (`publish-release.yml`)

Handles the publication of releases:

- **Trigger**: Runs when a release is published
- **Actions**:
  - Builds Python package using uv
  - Uploads built artifacts to the GitHub release
- **Environment**: Uses Python 3.12 and uv for building

### Version Management

#### Version Bump (`version-bump.yml`)

Handles automated version bumping of the project.

#### Update Major/Minor Tags (`update-major-minor-tags.yml`)

Updates major and minor version tags when new releases are published.

## Workflow Dependencies

All workflows use:

- `uv` for Python package management
- GitHub Actions cache for optimizing workflow execution
- Ubuntu latest as the running environment

## Best Practices

When contributing to this project:

1. Ensure your commits are properly labeled to be categorized in release notes
2. All tests must pass and code quality checks must succeed
3. New features should include appropriate tests
4. Breaking changes must be clearly marked with the "breaking-change" label

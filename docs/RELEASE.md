# Release Notes For Maintainers

This document describes the release flow. Do not run upload commands until the
release is explicitly approved.

## Pre-Release Checklist

- Run `pytest tests/ -q` and confirm the suite is green.
- Finalize the `CHANGELOG.md` Unreleased section into a concrete version section.
- Bump `pyproject.toml` to the release version.
- Confirm examples and docs do not reference private paths, API keys, local
  caches, or unpublished experiment logs.

## Build

```bash
python -m pip install build
python -m build
```

## Upload

PyPI upload requires a PyPI API token configured for the maintainer account.

```bash
python -m pip install twine
twine upload dist/*
```

## Tag

```bash
git tag vX.Y.Z
git push --tags
```

## Notes

The `0.4.0` package is the first beta release. It adds the CCGE-LA public alpha
API, tests, CI, pinned dependency floors, Hugging Face Hub checkpoint loading,
and API hygiene.

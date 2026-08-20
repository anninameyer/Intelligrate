# Release checklist

This checklist describes how to publish `intelligrate`.

The package on PyPI contains only the installable Python library, metadata,
README, and license. Example data, notebooks, tutorials, CI files, tests, and
results stay in the GitHub repository.

## Prerequisites

1. Confirm CI is green on GitHub for `main`.
2. Confirm `pyproject.toml` has the intended version.
3. Use Python 3.10, 3.11, or 3.12.
4. Install local release tools in the current environment:

```bash
python -m pip install --upgrade build twine
```

5. Create accounts on TestPyPI and PyPI:
   - https://test.pypi.org/
   - https://pypi.org/

Use API tokens for upload. Do not commit tokens, `.pypirc` files, or passwords.

## Build

Remove old local artifacts, then build fresh source and wheel distributions:

```bash
rm -rf dist build src/*.egg-info
python -m build
python -m twine check dist/*
```

Expected artifacts:

```text
dist/intelligrate-<version>.tar.gz
dist/intelligrate-<version>-py3-none-any.whl
```

The version in the filenames changes when `project.version` changes.

## TestPyPI upload

Upload to TestPyPI first:

```bash
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

When prompted:

```text
username: __token__
password: <paste TestPyPI API token>
```

The token input is hidden in the terminal.

## TestPyPI install check

Create a fresh environment and install from TestPyPI:

```bash
python -m venv .tmp-testpypi-install
.tmp-testpypi-install/bin/python -m pip install --upgrade pip
.tmp-testpypi-install/bin/python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  intelligrate==0.1.1
```

Verify imports and installed commands:

```bash
.tmp-testpypi-install/bin/python -c "import intelligrate; print(intelligrate.__file__)"
.tmp-testpypi-install/bin/intelligrate --help
.tmp-testpypi-install/bin/intelligrate subset distance --help
.tmp-testpypi-install/bin/intelligrate subset ga --help
.tmp-testpypi-install/bin/intelligrate extrapolate train --help
.tmp-testpypi-install/bin/intelligrate extrapolate full-predict --help
```

## Tag the release

After TestPyPI succeeds, create and push a matching Git tag:

```bash
git tag v0.1.1
git push origin v0.1.1
```

Use the version from `pyproject.toml`.

## PyPI upload

Upload the same already-checked artifacts to PyPI:

```bash
python -m twine upload dist/*
```

When prompted:

```text
username: __token__
password: <paste PyPI API token>
```

## PyPI install check

Create a fresh environment and install from PyPI:

```bash
python -m venv .tmp-pypi-install
.tmp-pypi-install/bin/python -m pip install --upgrade pip
.tmp-pypi-install/bin/python -m pip install intelligrate
```

Verify:

```bash
.tmp-pypi-install/bin/python -c "import intelligrate; print(intelligrate.__file__)"
.tmp-pypi-install/bin/intelligrate --help
.tmp-pypi-install/bin/intelligrate subset --help
.tmp-pypi-install/bin/intelligrate extrapolate --help
```

## Next release

For a later release:

1. Update `project.version` in `pyproject.toml`.
2. Add or update tests for all changed functionality.
3. Confirm GitHub Actions is green.
4. Repeat this checklist with the new version and tag.

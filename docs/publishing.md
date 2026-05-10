# Publishing Guide

This fork is published on PyPI as `softcpsrecsimulator`. The Python package still imports as `citylearn`.

Portuguese version: [pt/publishing.md](pt/publishing.md).

## Release Owner

Default release owner for this fork: [@calofonseca](https://github.com/calofonseca).

## Version Bump

1. Update `citylearn/__init__.py`.
2. Update [releases.md](releases.md).
3. Update affected reference pages when schema, actions, observations, KPIs or dataset contracts change.
4. Run validation from [developer_guide.md](developer_guide.md).
5. Commit and tag.

Example:

```console
git add citylearn/__init__.py README.md docs
git commit -m "Release v0.4.3"
git tag -a v0.4.3 -m "Release v0.4.3"
git push --follow-tags origin master
```

## PyPI Workflow

1. Create or keep the PyPI project named `softcpsrecsimulator`.
2. In GitHub repository settings, add `PYPI_API_TOKEN` with permission to publish this project.
3. Push the release commit and tag.
4. Create a GitHub Release or run the `Publish Python Package` workflow manually.
5. Workflow `.github/workflows/pypi_deploy.yml` builds `dist/*` and uploads to PyPI.

## Local Build Check

```console
.venv/bin/python -m pip install --upgrade pip build twine
.venv/bin/python -m build
.venv/bin/python -m twine check dist/*
```

## Release Discipline

| Version type | When to use |
|---|---|
| Patch | Additive compatible features, fixes, docs and tests. |
| Minor | New simulator capability or schema/API change. |
| Major | Broad breaking changes. |

Before publishing, make sure `docs/releases.md` states:

| Field | Purpose |
|---|---|
| Summary | What changed and why. |
| Release owner | GitHub tag of the person responsible for the release. |
| Dataset/schema impact | Migration risk for existing datasets and configs. |
| Compatibility | Whether algorithms/wrappers need changes. |
| Validation | Commands or simulations that passed. |
| Migration notes | User-facing actions required after upgrade. |

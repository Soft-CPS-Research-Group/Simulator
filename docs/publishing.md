# Publishing Guide

This fork publishes **`softcpsrecsimulator`** from
[Soft-CPS-Research-Group/Simulator](https://github.com/Soft-CPS-Research-Group/Simulator).
The Python import remains `citylearn`. Portuguese: [pt/publishing.md](pt/publishing.md).
Default release owner: [@calofonseca](https://github.com/calofonseca).

Publication is a separate maintainer action. Local testing and documentation
cleanup do not imply a version bump, tag, push or publication. A version string
alone does not prove that local changes have been published.

## Prepare and validate

1. Review the release diff. Update `citylearn/__init__.py`, the source version in
   `README.md` and `README_PYPI.md`, and [releases.md](releases.md). Update affected
   product/schema/KPI references and explain compatibility or migration limits.
2. Run the full suite and mandatory auditors from [developer_guide.md](developer_guide.md).
   Keep run evidence outside the checkout or in CI/release artifacts, rather than
   adding release-run JSON to `docs/`.
3. Build and check the distributions. Use a fresh output directory for the version.
4. Verify an installed wheel outside the source tree. The
   [CI workflow](../.github/workflows/run_tests.yml) demonstrates the isolated
   `scripts/ci/installed_export_smoke.py` check, with explicit schema/output paths.

```console
.venv/bin/python -m pip install build wheel twine
.venv/bin/python -m pytest -q
.venv/bin/python scripts/audit/run_full_audit.py --output-dir /tmp/softcps-release-audit
.venv/bin/python -m build --outdir /tmp/softcps-dist-X.Y.Z
.venv/bin/python -m twine check /tmp/softcps-dist-X.Y.Z/*
```

## Commit, CI and tag

For an authorised release, commit with the normal maintainer identity and push the
reviewed release commit to `origin/master` once. Wait for its CI before publishing.
The current CI covers Python 3.9/3.10, lint, tests, accounting audits, performance
smoke and installed-distribution exports. Then create an unused release tag for
the approved commit:

```console
git push origin master
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

These are release steps, not local-maintenance commands. Tag pushes do not repeat
the branch-only CI. Do not manually dispatch the same CI again without a specific
need. Preserve inherited upstream tags: fork 1.7.0 uses
`softcpsrecsimulator-v1.7.0`, because upstream already owned `v1.7.0`.

## Publication and verification

The [Publish Python Package workflow](../.github/workflows/pypi_deploy.yml) runs
on a **published GitHub Release** or explicit `workflow_dispatch`; a tag push
alone does not publish. Choose one trigger per release, avoiding an additional
manual dispatch after publishing the GitHub Release.

- Build uses Python 3.11, creates wheel/sdist, runs `twine check`, installs the
  wheel and verifies exports outside the checkout before uploading artifacts.
- Publish depends on that build, uses the `pypi` environment and sends artifacts
  to the fork's project with the configured `PYPI_API_TOKEN`.
- A manual publication must select the approved release tag. Never push release
  tags to `upstream` or publish this fork as upstream's `CityLearn` project.

The maintainer confirms master CI before creating the GitHub Release. The
publication workflow's automatic dependency is its own build/export job. This
guide describes the existing workflows; it does not change their triggers.

Confirm both jobs succeeded for the approved ref and check the exact version on
[PyPI](https://pypi.org/project/softcpsrecsimulator/). Install that pinned version in
a clean environment, outside the checkout; verify `citylearn.__version__`, the
`CityLearnEnv` import and the installed export smoke check. Upload success alone
is not an installed-package verification.

## Version policy and release entry

| Type | Use when |
|---|---|
| Patch | Compatible fixes, documentation and additive changes. |
| Minor | A substantial capability or schema/API change. |
| Major | Broad breaking changes. |

Keep one entry per release in the central changelog, with actual validation scope
and limits. An uncommitted work iteration does not create a new version.

```markdown
## vX.Y.Z — YYYY-MM-DD — summary

Release owner: maintainer.

- Changes and user impact.
- Dataset/schema impact and compatibility.
- Validation, coverage and limitations.
- Migration actions, when required.
```

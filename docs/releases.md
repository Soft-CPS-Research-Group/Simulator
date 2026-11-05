# Releases

This is the changelog for the fork. Version policy, the release template and
publication procedure are maintained in the [publishing guide](publishing.md).
Historical validation links point to immutable evidence in Git history; they
describe the scope of that release rather than a new execution on this checkout.

Portuguese version: [pt/releases.md](pt/releases.md).

Default release owner for this fork: [@calofonseca](https://github.com/calofonseca).

## v2.0.2 — 2026-11-05 — reproducible Portugal datasets and nested downloads

Release owner: **Tiago Fonseca** ([@calofonseca](https://github.com/calofonseca)).

- Deliver `portugal-realism-4` and `portugal-ev-mobility-1` for all four Portugal
  communities across 2023/2024: diverse charging sessions, household activity,
  commercial occupancy and equipment, annual consumption budgets, and irregular
  local faults and demand-response requests. Preserve labelled stress cases,
  equipment inventories, nine main scenarios and six matched references.
- Include the deterministic generator, compact empirical references, assumptions,
  seed/sensitivity checks and reproducibility evidence. These are synthetic
  benchmarks, not measurements representative of the Portuguese population.
- Download nested dataset directories and expose the cache only after a complete,
  successful download. Failed downloads can be retried without accepting a partial
  dataset. Add online-loader regressions and fetch the immutable source commit in CI.
- Keep simulation dynamics, public interfaces and the OMIE pricing convention
  unchanged. Old experiment results must retain their original dataset reference;
  upgrading the default dataset changes the benchmark inputs.

Pre-release data validation passed 161 focused tests, 72 boundary windows,
27 fault probes and 18 accounting windows; two regenerations produced identical
Parquet files. Release-wide tests, audits and installed-package verification are
reported with the GitHub release. No annual controller campaign or MARL training
is included. Reference-data reuse and empirical limitations are documented in
[the dataset validation](portugal_rec_native/Heldout_2024.md).

Package/import names remain `softcpsrecsimulator` / `citylearn`. The release tag
and default online dataset ref are `softcpsrecsimulator-v2.0.2`. Install the
`parquet` extra for the Portugal datasets. Explicit dataset-ref overrides remain
supported; no schema migration is required.

## v2.0.1 — 2026-11-04 — episode selection and subhourly corrections

Release owner: **Tiago Fonseca** ([@calofonseca](https://github.com/calofonseca)).

- Honor explicit `random_episode_split=True` and `False` constructor overrides;
  `None` continues to inherit the schema setting.
- Compound the configured hourly standby-loss fraction over the physical step,
  independently of dataset cadence. A 1% hourly loss leaves 99% after one hour
  at both hourly and 15-minute resolution, within float precision.
- Convert dataset energy to average step power when autosizing heat pumps,
  electric heaters and batteries; calculate daily battery peaks over physical
  24-hour periods. Thermal capacity remains energy for one control step.
- Retain public energy-command interfaces, explicitly configured equipment, BAU
  control logic, KPI definitions, dataset files and CSV/Parquet exports.
  Subhourly SOC, derived KPIs and automatically selected equipment can change;
  compare matching versions and resolved equipment. No file migration is needed.
- Keep release documentation concise and use CPU-only PyTorch wheels in hosted
  CI to avoid runner disk exhaustion; runtime dependency requirements are unchanged.

The corrected runtime passed 1,300 CI tests, the physical-step/topology audit,
Portuguese winter/summer windows and supported-dataset checks. The common
standby/autosizing/BAU/reference subset passed 223 local tests. The reviewed
fixed-command reference reproduces the old baseline exactly when only the old
loss formula is restored; hourly reference results are unchanged.
Release CI and installed-distribution checks accompany the GitHub release.
The two-year campaign and performance measurements below describe the earlier
runtime: they have not been repeated for this patch, and no new speedup, full
annual equivalence or empirical equipment calibration is claimed.

Package/import names remain `softcpsrecsimulator` / `citylearn`. The release tag
and default online dataset ref are `softcpsrecsimulator-v2.0.1`; dataset files
are unchanged from 2.0.0. Explicit dataset-ref overrides remain supported.

## v2.0.0 — 2026-11-02 — audited-runtime consolidation milestone

- Close the audit, robustness, internal-organization and performance phase by
  releasing the v1.9.13 runtime as v2.0.0. This major-number milestone introduces
  no functional change or migration requirement.
- Preserve physical models, BAU/NormalPolicy, KPI formulas/tolerances, rewards,
  complete histories, export schemas, dependencies and the current datasets.
- Keep the distribution name `softcpsrecsimulator` and import namespace `citylearn`.
  The fork tag is `softcpsrecsimulator-v2.0.0`; the inherited upstream `v2.0.0`
  tag remains unchanged.

Validation: all 81 runtime Python files match v1.9.13 byte for byte, allowing
only the version literal in `citylearn/__init__.py` and the release-tag prefix
in `DataSet.REPOSITORY_TAG`. Online data use the namespaced fork tag so they
resolve the same dataset content rather than the inherited upstream tag. Explicit
environment/constructor overrides are preserved and regression-tested. Wheel/sdist
checks and isolated installed KPI/BAU/manifest exports passed. Datasets, workflows
and all physical, BAU, KPI and optimization implementations are unchanged.
The unchanged runtime retains the v1.9.13 validation: 1,186 local integration tests,
four strict auditors and the nine-case campaign (631,575 transitions; 197,347,896
physical/accounting checks; 77,603 exact KPI comparisons across seven complete
historical references). The current release CI and publication checks are recorded
in the release artifacts. The long campaign is reused, not repeated for renumbering.
Performance measurements and their method remain in the v1.9.13 entry below.

Scope limits remain unchanged: isolated exact SOC-target misses need diagnosis;
all observed minimum acceptable service was met. The long campaign used the
pre-v3.3 inputs, while current v3.3 data passed bounded integration checks.
Empirical equipment calibration remains outside this validation.

## v1.9.13 — 2026-11-02 — lower-memory histories and faster evaluation

- expose evaluation progress by stage/member/KPI family, with optional callback
  and auditor progress files; instrumentation does not add values to KPI results.
- retain complete market and robustness histories in compact lossless
  representations with bounded window/stream readers and compatible public views.
- reuse active-population masks within an evaluation and maximum-charge EV
  feasibility replay within each departure/context, without caching physical state.
- index deferrable-appliance cycles and reuse observation/forecast preparation
  within a request, including a narrowly guarded default-reward fast path. Custom
  rewards retain the full observation contract; state/RNG/physics are unchanged.
- integrate installed-export checks, regression tests and final long-run evidence.

### Observed two-year campaign performance

Each case executes 70,175 transitions, seed 0, NormalPolicy/BAU and active physical
checks. RSS is the process high-water mark in MiB; wall time includes load, dispatch,
independent auditing, final KPI evaluation and export. Baseline and optimized runs
are separate observations on the same host, not a repeated controlled benchmark.

| Scenario | Peak RSS before → after (MiB) | Total wall before → after | KPI/export before → after (s) |
|---|---:|---:|---:|
| Micro4 | 920.5 → 691.6 | 7m04s → 4m31s | 51.7 → 21.1 |
| Premium All-in | 13,709.7 → 5,333.4 | 4h23m01s → 2h08m40s | 4,667.2 → 479.9 |
| Core30 Topology | 3,521.1 → 1,844.1 | 57m57s → 26m51s | 646.3 → 110.1 |
| Core30 Combined | 3,536.2 → 1,849.2 | 52m43s → 27m25s | 647.5 → 112.9 |
| Core30 Safety | 3,821.7 → 1,899.7 | 50m04s → 29m26s | 290.8 → 115.9 |
| Core30 Normal | 3,822.7 → 1,902.2 | 50m49s → 29m30s | 293.2 → 116.9 |
| Core15 | 2,036.3 → 1,158.3 | 24m39s → 14m46s | 162.8 → 65.5 |

Premium All-in peak RSS fell 61.1%, total wall time 51.1%, and KPI/export time
89.7%. Audit-inclusive mean dispatch time fell from 157.97 to 102.79 ms/transition.
Core30 Robustness completed in 43m51s at 2,045.8 MiB and Premium Normal in 1h53m07s
at 5,523.6 MiB. Their old runs were incomplete: no final historical speedup or
full historical KPI-equivalence claim is made for these two cases.

Validation: all nine cases passed, totaling 631,575 transitions and 197,347,896
physical/accounting checks. The seven complete historical comparisons matched
77,603 KPI rows exactly, including identities and undefined values. Independent
reconciliation passed in all nine cases. All 3,499 campaign input hashes were
preserved. The campaign used the pre-v3.3 dataset edition on the frozen P8 source;
it does NOT certify a full annual run of the newer heldout-2024 scenarios now on
master. The measured input edition is retained at commit
`8157f605e26fb75651037451fb453b1a48056787`; combine those data with the
v1.9.13 runtime for reproduction, resolving the archived local paths for your machine.
Current-master tests and bounded audits are recorded separately in release
artifacts. Measured runtime modules match the release except for the version string.
The frozen auditor adds measurement-only dispatch timing; parity of its results
was checked before running the campaign.

Current-master integration: 1,186 local tests passed (22 known Gymnasium
float32-bound warnings); four strict auditors, critical lint, performance smoke,
wheel/sdist checks and isolated wheel export passed.

[Campaign evidence](https://github.com/Soft-CPS-Research-Group/Simulator/releases/download/v1.9.13/campaign-P8-evidence.zip) ·
[Release integration evidence](https://github.com/Soft-CPS-Research-Group/Simulator/releases/download/v1.9.13/validation-1.9.13.zip).

Compatibility: no information sampling/discard, relaxed tolerances, changes to
physical equipment, BAU strategy, reward values, tariffs, KPI formulas or export
schemas. Minimum SOC service was met for every observed departure; isolated exact
target misses remain, with diagnosis pending. Accounting PASS is not a guarantee
of target attainment, universal scheduling feasibility or real-equipment calibration.

## v1.9.12 — 2026-11-02 — robustness contracts and maintainable internals

- Support deterministic composition of robustness events, explicit
  observation/action-space and target contracts, and versioned robustness audit
  metadata available in evaluation and export sidecars.
- Share electrical-service allocation between NormalPolicy and runtime limiting;
  preserve sequential outage budgets and immediate maximum charging BAU rules.
- Split KPI, entity-observation, loading and robustness responsibilities into
  focused internal modules while keeping the public facades and export contracts.
- Make audit references verifiable without silently rewriting expected results;
  add portable fixtures, physical regressions and installed-distribution checks.
- Remove temporary release-run notes and frozen working evidence from the source
  tree. Keep maintained product documentation and immutable historical links.
- Preserve the Portugal v3.3 datasets and matched 2024 references already merged
  on master; their edition is separate from the simulator version. Reconcile the
  documentation bundle after cleanup without regenerating physical inputs.

Validation: the corrected runtime previously passed 1,020 tests, 16 physical
scenarios, 63 dataset-interface audits, two Portuguese outage windows, six installed
reference cases and four installed export formats. The release integration is
retested against current master; current-run evidence is attached to the GitHub
Release. The performance improvements are released separately in v1.9.13.

Release integration: 1,063 tests passed on current master; all four strict
auditors, critical lint, performance smoke, wheel/sdist checks and installed-wheel
KPI/BAU/manifest export passed. No reference snapshots were rewritten.

Compatibility: no change to BAU scheduling, physical models, KPI definitions or
numerical tolerances. Internal modules are implementation details. Exact target
SOC is not guaranteed by immediate-charging BAU; minimum service and target service
remain separate metrics. No empirical equipment calibration is claimed.

Earlier releases, with their original validation notes: [release archive](releases_archive.md).

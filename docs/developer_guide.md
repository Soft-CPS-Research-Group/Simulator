# Developer Guide

This page collects local development commands, validation checks, performance smoke tests and the current internal architecture map.

Portuguese version: [pt/developer_guide.md](pt/developer_guide.md).

## Local Environment

Use the repository virtual environment when available:

```console
.venv/bin/pip install -e .
```

Optional dependencies:

| Feature | Command |
|---|---|
| Parquet datasets | `.venv/bin/pip install pyarrow` |
| PV autosizing | `.venv/bin/pip install ".[pysam]"` |
| Build checks | `.venv/bin/pip install build twine` |

## Core Validation

| Check | Command |
|---|---|
| Full test suite | `.venv/bin/pytest -q` |
| Critical lint | `.venv/bin/python -m ruff check citylearn tests scripts/manual scripts/ci --select E9,F821` |
| Entity contract audit | `.venv/bin/python scripts/audit/audit_entity_contract.py --strict` |
| Physics audit | `.venv/bin/python scripts/audit/audit_physics.py` |
| Diff whitespace check | `git diff --check` |

The physics audit intentionally exercises resolution conversion paths, so unit-conversion warnings can be expected when the audit fixture forces dataset/schema mismatches.

## Portable tests and mandatory audit verification

Test fixtures are resolved from their source files, not the current working
directory. Missing required schemas fail clearly; a synthetic or cached dataset
must not silently replace a missing fixture. To check portability, run the same
suite from the checkout and from another directory, using absolute paths to the
checkout's Python and `tests` directory for the latter invocation.

```console
.venv/bin/python scripts/audit/run_full_audit.py --output-dir /tmp/simulator-audit
```

The default is verification: any failed/missing auditor or dependency exit blocks
success. `--strict` remains an alias for that mode. `--report-only` explicitly
collects diagnostics with exit code zero; inspect `verification_passed` in its
summary rather than interpreting that exit as approval. Relative output paths
are resolved from the caller's directory before subprocesses start.

Default coverage is physics, entity contracts, bounded offline dataset rollouts
and the fixed-command refactoring references below. The summary names the selected
families and whether all defaults were selected. `--audits` can select a subset;
that subset does not certify the entire simulator or replace pytest, extended
campaigns or empirical calibration. The legacy `performance_results` audit
includes RBC/SAC comparisons and is opt-in, outside the default simulator checks.
Snapshot-changing flags require `--report-only`; a verification cannot update its
own references. Work reports belong outside the versioned source tree.

## References for internal refactoring

`audit_refactor_parity.py` uses six short, deterministic fixed-command cases:
flat/entity interfaces, 60/300/900/3600-second steps, phase limits, outages and
member lifecycle/fault composition. It reuses the checkout's bounded test fixture
factory and requires `test_requirements.txt` as well as runtime dependencies.
It runs no agent, training or BAU sidecar. The versioned fixture
`scripts/audit/snapshots/refactor_parity.json` freezes the behaviour after the C1/C2
corrections on the 1.9.11 base, before further internal refactoring.

Each case independently checks energy, cash and provenance, then compares input
fingerprints, requested/executed actions, SOC, flows, rewards, machine service,
KPIs, interface layouts/spaces and CSV/JSON export semantics. Runtime versions,
commit IDs and temporary paths are not behaviour references. Layouts are stored
once per topology change. Both delivered and unserved machine requests are
covered; references establish behavioural parity, not optimal scheduling.

```console
.venv/bin/python scripts/audit/audit_refactor_parity.py --output /tmp/parity-check.json
```

Units determine numeric tolerances: kWh `1e-6`, EUR `1e-7`, SOC/ratios `1e-7`,
kW `1e-5`, other numeric values `1e-6` absolute, all with `1e-7` relative tolerance.
Counts, tolerance policy, flags, names, ordering, missing keys and unknown/zero
distinctions remain exact. Non-finite values are explicitly represented in the
fixture. A mismatch or missing reference fails; checks never regenerate it.

For an intentional contract change, capture a separate **new** proposal, inspect
the differences and physical/service consequences, and only then replace the
reviewed fixture as a deliberate change:

```console
.venv/bin/python scripts/audit/audit_refactor_parity.py --capture-reference /tmp/parity-proposal.json --output /tmp/parity-capture.json
```

Capture refuses to overwrite an existing file and reports `reference_proposal`
with `verification_passed=false`. It is not a successful verification. The saved
JSON is a test baseline; development diaries and execution reports are not fixtures.
Report destinations must also differ from references/proposals, including path aliases.

## Manual Simulation Scripts

Manual utility scripts live in `scripts/manual` and are excluded from default pytest collection.

```console
.venv/bin/python scripts/manual/demo_ev_rbc.py
.venv/bin/python scripts/manual/demo_ev_rbc_export_end.py
```

Runtime benchmark:

```console
.venv/bin/python scripts/manual/bench_runtime.py --seconds 5 60 --render-modes none end --episode-steps 1200
```

CI performance smoke check:

```console
.venv/bin/python scripts/ci/perf_smoke.py --episode-steps 600 --seconds 60 --baseline-file scripts/ci/perf_baseline.json
```

Step/component profiling:

```console
.venv/bin/python scripts/audit/profile_step_breakdown.py --episode-steps 300 --seconds 60 --agent rbc --interface flat --render-mode none --no-write --table-limit 14
```

Lifecycle and KPI/BAU profiling:

```console
.venv/bin/python scripts/audit/profile_lifecycle.py --episode-steps 300 --seconds 60 --skip-exports --no-write
```

Drop `--skip-exports` when the CSV export paths themselves need to be timed.

For ad hoc code, pass `debug_timing=True` to `CityLearnEnv` and inspect `info` keys such as `apply_actions_time`, `reward_observations_time`, `next_observations_time`, `terminal_export_time` and `step_total_time`.

## Internal Architecture

Public APIs remain centered on `CityLearnEnv` and `Building`. Internal orchestration is split into service modules under `citylearn/internal`.

| Module | Responsibility |
|---|---|
| `loading.py` | Schema-driven loading and building assembly. |
| `loading_inputs.py` | Building input series, shared-cache access, dynamic windows and time-step conversion. |
| `loading_components.py` | Component constructor lookup, optional models, chargers, device seeds and autosizing. |
| `runtime.py` | Episode runtime orchestration, `step`, action parsing, time progression and EV/charger association. |
| `building_ops.py` | Building observation/action orchestration. |
| `electrical_limits.py` | Pure kW allocation shared by native baseline prediction and runtime service limiting. |
| `kpi.py` | KPI/evaluation pipeline. |
| `kpi_energy.py` | Energy, carbon, cost, comfort and district grid-shape records. |
| `kpi_equipment.py` | Equipment calculations and per-evaluation EV/BESS/PV/phase aggregation. |
| `kpi_community.py` | Community settlement and equity records. |
| `kpi_service_metrics.py` | Executed/requested appliance and escalator service aggregation. |
| `entity_interface.py` | Entity tables, edges, action specs and observation bundles. |
| `entity_observations.py` | Per-table observation population, relation masks, sanitisation and independent payload copies. |
| `entity_layout.py` | Static observation rows, numeric spaces and declarative interface specifications. |

Electrical-service allocation takes signed total/phase requests and a caller's
base-load estimate, returning proportional reductions and requested/residual
pressure without changing devices. It preserves total-before-phase,
import-before-export ordering and the eight-pass maximum, recalculating totals
after every reduction. Uncontrollable overload remains visible as residual
pressure; the calculation cannot create supply or shed fixed loads.

Prediction uses prospective loads. Runtime independently recalculates limits,
converts powers to device actions, records feedback and applies battery/charger
physics. During outages it additionally reads the live remaining kWh before each
controllable load; do not replace that sequential check with a predicted budget.
The legacy charger-only limiter and future appliance-profile feasibility checks
retain their distinct contracts. BAU still requests immediate maximum EV charging
to 100%, with proportional service sharing and no price/departure optimisation.

## Development Rules

The KPI, entity and loading services retain their entry points and orchestrate
these helpers. KPI families preserve legacy/extended row order and use a fresh
equipment accumulator per evaluation. Counts, capacities and sample denominators
are aggregated before district ratios; do not average member ratios instead.
Entity population shares one time/bundle context per payload. Information masking
and fault transformations retain their outer contract stages, while returned
tables and edges own their arrays. Layout discovery/reference types stay in the
interface service; reset/topology changes rebuild spaces and specifications.

Loading separates series preparation from component creation while retaining
conversion, cache, window, metadata, optional-input defaults and seed rules.
Invalid component type lookups raise `ValueError` with their schema path and
the import/attribute failure as cause. Errors inside a constructor propagate;
the loader does not silently substitute another class. Facade methods used by
existing internal consumers continue to delegate with their original signatures.
Absent optional pricing/carbon inputs retain the existing zero-series defaults.
Charger inputs still require their six core columns by name, while session IDs
and measured SOC are optional telemetry. Legacy custom building facades may
still expose aggregate BESS series instead of per-storage consumption histories.

| Rule | Reason |
|---|---|
| Keep schema/API changes documented in the same change. | Algorithms and datasets depend on explicit contracts. |
| Add tests for new physics or observations. | Sub-hourly and entity behavior are easy to regress silently. |
| Prefer additive observations in patch releases. | Avoid breaking trained algorithms and wrappers. |
| Run audits before tagging. | They catch contract drift outside normal unit tests. |

## Repository content

Keep runtime code, required product/dataset documentation, licences, the central
changelog, normative test fixtures and necessary dataset provenance in source
control. Planning notes, development diaries, release-run JSON, rendering QA and
ad hoc observation dumps belong outside the checkout or in CI/release artifacts.
Before removing work material, archive it externally and verify hashes; keep the
source/destination mapping with that external evidence. Do not rewrite Git history.
The [dataset catalogue](portugal_rec_native/README.md#proveniência-conservada-no-produto)
identifies the retained provenance files and their consumers. Distribution
checksums describe the current package, not a previous documentation bundle.

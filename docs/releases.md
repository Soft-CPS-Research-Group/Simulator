# Releases

This is the operational changelog for the fork. From now on, every version bump should explain what changed, user impact, dataset/schema impact, validation and compatibility risk.

Portuguese version: [pt/releases.md](pt/releases.md).

Default release owner for this fork: [@calofonseca](https://github.com/calofonseca).

## Version Policy

| Type | Use when | Example |
|---|---|---|
| Patch | Compatible fixes and additive features. | `0.4.1 -> 0.4.2`. |
| Minor | New major capability or schema/API change. | `0.3.x -> 0.4.0`. |
| Major | Broad breaking changes. | `0.x -> 1.0`. |

## Release Checklist

1. Update `citylearn/__init__.py`.
2. Update this file.
3. Update README when relevant features change.
4. Update affected reference pages when schema/action/observation/KPI contracts change.
5. Run validation:
   - `.venv/bin/pytest -q`
   - `.venv/bin/python scripts/audit/audit_entity_contract.py --strict`
   - `.venv/bin/python scripts/audit/audit_physics.py`
6. Run at least one representative smoke simulation.
7. Commit and tag.
8. Publish package when applicable.

## Template

```markdown
## vX.Y.Z - YYYY-MM-DD

Release owner: [@calofonseca](https://github.com/calofonseca).

### Summary
- ...

### Added
- ...

### Changed
- ...

### Fixed
- ...

### Dataset/Schema Impact
- ...

### Compatibility
- ...

### Validation
- `...`: pass

### Migration Notes
- ...
```

## v0.5.2 - EV Departure Service KPIs

Release owner: [@calofonseca](https://github.com/calofonseca).

### Summary

Patch release that separates EV departure target fulfillment, minimum acceptable user service and symmetric SOC target accuracy.

### Added

- `ev_departure_service_tolerance` environment/schema setting, default `0.05`.
- Building and district KPIs for minimum acceptable EV departures.
- Building and district KPIs for SOC shortfall beyond service tolerance.
- EV departure SOC surplus, absolute error and configured tolerance diagnostics.

### Changed

- Existing EV departure success, within-tolerance and deficit KPI names and semantics are preserved.
- `ev_departure_within_tolerance` remains the symmetric proximity tolerance.

### Dataset/Schema Impact

- Existing schemas remain valid.
- New optional top-level schema key: `ev_departure_service_tolerance`.
- Existing optional top-level key `ev_departure_within_tolerance` can configure symmetric target accuracy tolerance.

### Compatibility

- Backward compatible additive KPI release.
- Consumers that enumerate all v2 KPI names should allow the new EV rows in `evaluate_v2()` and `exported_kpis.csv`.

### Validation

- `.venv/bin/python -m pytest -q tests/test_kpi_v2.py -q`: pass
- `.venv/bin/python -m pytest -q tests/test_kpi_golden.py -q`: pass
- `.venv/bin/python -m pytest -q tests/unit/test_export_logic.py tests/unit/test_ui_export_contract.py -q`: pass
- `.venv/bin/python -m pytest -q --ignore=scripts/manual`: pass (`276 passed`)
- `.venv/bin/python -m compileall -q citylearn tests/test_kpi_v2.py`: pass

### Migration Notes

- Use `*_ev_performance_departure_min_acceptable_ratio` as the primary user-comfort EV departure KPI.
- Use `*_ev_performance_departure_success_ratio` for strict target fulfillment.
- Use `*_ev_performance_departure_within_tolerance_ratio` and absolute error for target accuracy and efficiency analysis.

## v0.5.1 - Deferrable Start Command Hardening

Release owner: [@calofonseca](https://github.com/calofonseca).

### Summary

Patch release focused on making deferrable appliance control safer for RL training by treating start actions as explicit ON/OFF commands and avoiding accidental early starts from small positive continuous actions.

### Changed

| Area | Change |
|---|---|
| Deferrable action semantics | `DeferrableAppliance.trigger_threshold` default changed from `0.0` to `0.5`. |
| Start command robustness | Non-finite values (`nan`, `inf`) are treated as OFF and do not start cycles. |
| Controller contract | Documentation now states binary start intent (`0` off, `1` on) and recommends using `deferrable_appliance_can_start` as readiness signal. |

### Dataset/Schema Impact

- No schema structure change.
- Existing optional `attributes.trigger_threshold` remains supported.
- Schemas that relied on implicit default `0.0` now use safer default `0.5` unless explicitly overridden.

### Compatibility

Behavioral patch change:
- with default settings, low positive deferrable actions no longer trigger starts.
- to preserve legacy behavior, set `deferrable_appliances.<id>.attributes.trigger_threshold: 0.0`.

### Validation

| Command/group | Result |
|---|---|
| `.venv/bin/pytest -q tests/unit/test_deferrable_appliance.py tests/test_deferrable_appliance_integration.py` | Pass (`10 passed`). |
| `.venv/bin/pytest -q tests/test_dynamic_topology_entity_mode.py -k deferrable` | Pass (`1 passed`, `13 deselected`). |

### Migration Notes

- If a controller previously emitted arbitrary continuous values in `[0, 1]`, map intent explicitly to ON/OFF (for example, `0.0` or `1.0`).
- If you intentionally need old permissive behavior, configure `trigger_threshold=0.0` per deferrable appliance.

## v0.5.0 - Action-Asset Consistency Hardening (Breaking)

Release owner: [@calofonseca](https://github.com/calofonseca).

### Summary

Breaking release that removes silent action/device mismatches for storage control, enforces strict topology-mode compatibility for dynamic schemas and consolidates the public deferrable dataset contract.

### Added

| Area | Change |
|---|---|
| 15s Parquet datasets | Added tracked 15-second parquet datasets for `citylearn_three_phase_dynamic_assets_only_demo_15s_parquet` and `citylearn_three_phase_electrical_service_demo_15s_parquet`. |
| Dataset naming normalization | Migrated legacy dataset files/asset IDs from `washing_machine_*` to `deferrable_appliance_*` in `citylearn_three_phase_electrical_service_demo`, `citylearn_three_phase_dynamic_topology_demo` and `citylearn_challenge_2022_phase_all_plus_evs`. |

### Changed

| Area | Change |
|---|---|
| Topology mode validation | Schemas that declare dynamic topology (`topology_mode: dynamic` or `topology_events`) now reject `topology_mode='static'` at initialization. |
| Static consistency | In static mode, if `electrical_storage` action is active for a building without an effective storage asset and without `inactive_actions` opt-out, initialization now fails fast. |
| Dynamic action exposure | Dynamic metadata synchronization now keeps `inactive_actions` precedence for `electrical_storage`, EV charger actions and deferrable appliance actions. |

### Dataset/Schema Impact

- Dynamic datasets must be executed with `topology_mode='dynamic'` and `interface='entity'`.
- Static datasets with `actions.electrical_storage.active=true` must either:
  - declare an effective `electrical_storage` asset for participating buildings, or
  - explicitly opt-out per building via `inactive_actions: ["electrical_storage"]`.
- Official deferrable naming in provided datasets is now `deferrable_appliance_*`; legacy `washing_machine_*` naming was removed from `data/datasets`.
- 15-second parquet schemas are now first-class tracked dataset artifacts for direct use in smoke/regression validation.

### Compatibility

Breaking behavior change by design:
- no more silent fallback for static storage-action inconsistencies;
- no more running dynamic-topology schemas in static mode.

### Validation

| Command/group | Result |
|---|---|
| `tests/test_dynamic_topology_entity_mode.py` targeted consistency tests | Pass. |
| `.venv/bin/pytest -q tests/test_dynamic_topology_entity_mode.py tests/test_dynamic_topology_assets_only_dataset.py` | Pass (`15 passed`). |
| `.venv/bin/pytest -q tests/test_dynamic_topology_entity_mode.py tests/test_15_second_power_fixture.py tests/test_dataset_loader_window_parquet.py` | Pass (`17 passed`). |
| 15s dataset smoke (`CityLearnEnv` reset + step) for CSV and parquet variants of dynamic-assets-only and electrical-service demos | Pass (`4/4`). |
| Whole-catalog smoke with windowed loading (`31` schemas) | `27` pass; `4` known pre-existing HVAC sizing assertion failures outside EV/deferrable migration scope. |

### Migration Notes

- If you previously ran dynamic schemas in static mode, update runtime configs to `topology_mode='dynamic'` and `interface='entity'`.
- If static runs now fail with storage consistency errors, fix schema intent explicitly (declare storage or set per-building `inactive_actions`).

## v0.4.3 - Documentation Contract and Release Readiness

Release owner: [@calofonseca](https://github.com/calofonseca).

### Summary

Release focused on making the simulator easier to adopt, audit and integrate by turning the README into a documentation portal and adding bilingual reference documentation.

### Added

| Area | Changes |
|---|---|
| README | English default documentation map with Portuguese links. |
| Running simulations | Install, quickstarts, CLI, `CityLearnEnv` parameters, render/export and validation commands. |
| Schema | Full schema reference for buildings, devices, PV modes, EVs, chargers, deferrables, topology and community market. |
| Interfaces | Flat/entity contracts, entity specs, tables, edges and dynamic topology guidance. |
| Actions | Action names, ranges, physical conversions and entity action payloads. |
| Observations | Observation dictionary with units, sentinels, bundles, entity tables and edges. |
| Datasets | CSV/Parquet contract, real-data conversion, 15s guidance and deferrable dataset format. |
| KPIs | v1/v2 explanation, units, equations and EV/BESS/deferrable/community/phase families. |
| Features | Simulator capability inventory. |
| Releases | Release checklist, versioning policy and changelog template. |
| Developer guide | Tests, audits, performance checks and internal architecture. |
| Publishing guide | PyPI release workflow and local build checks. |
| Portuguese docs | Full Portuguese mirror under `docs/pt/`. |

### Compatibility

No runtime API break is introduced by the documentation work. The release keeps the additive simulator changes documented under `v0.4.2`.

### Validation

| Check | Result |
|---|---|
| Local Markdown links | Pass. |
| `git diff --check` | Pass. |
| Flat smoke simulation | Pass. |
| Entity smoke simulation | Pass. |
| 15s Parquet smoke simulation | Pass. |

## v0.4.2 - Additive Physical Observation Expansion

Release owner: [@calofonseca](https://github.com/calofonseca).

### Summary

Additive release focused on exposing richer physical observations to algorithms without removing older features.

### Added

| Area | Changes |
|---|---|
| Chargers | `min_charging_power_kw`, `min_discharging_power_kw`, `charger_efficiency_ratio`. |
| Charger core | `charge_efficiency_at_max_ratio`, `discharge_efficiency_at_max_ratio`, incoming EV required/departure/hours/ratio. |
| Storage | `min_charge_power_kw`, `min_discharge_power_kw`, `efficiency_ratio`, `round_trip_efficiency_ratio`. |
| Storage core | `current_efficiency_ratio`, `degraded_capacity_kwh`, `soc_min_ratio`. |
| Storage phases | `phase_connection_L1/L2/L3`. |
| Deferrables | `must_run`, average/peak/load factor, peak offset, remaining duration/power and current-step power. |
| Building core | `charging_total_service_power_kw`, `charging_phase_L1/L2/L3_power_kw`. |
| PV | `generation_capacity_factor_ratio`. |

### Compatibility

Compatible additive change. Existing observations and EV/charger duplication are intentionally preserved.

### Validation

| Command/group | Result |
|---|---|
| Focused entity/bundle/deferrable/parquet tests | Pass. |
| Broader simulator tests | Pass. |
| Full suite | Pass. |
| `audit_entity_contract.py --strict` | Pass. |
| `audit_physics.py` | Pass. |

## v0.4.0 - Normalized Deferrable Appliances

Release owner: [@calofonseca](https://github.com/calofonseca).

### Summary

Introduced the official `deferrable_appliances` model with a cycle catalog and sparse flexibility schedule, replacing the old washing-machine format.

### Added

| Area | Changes |
|---|---|
| Schema | `deferrable_appliances` with `cycle_profiles_file` and `flexibility_schedule_file`. |
| Model | `DeferrableAppliance` with pending/running/missed/completed states. |
| Actions | `deferrable_appliance_{id}` in flat mode and `start` in the entity table. |
| Observations | Pending, running, can_start, deadlines, slack, priority and remaining energy. |
| KPIs | Completed/missed cycles, service level, served/unserved energy and average delay. |
| Dynamic topology | Add/remove `deferrable_appliance`. |

### Compatibility

Breaking schema change: `washing_machines` is no longer the official dataset format.

## v0.3.2 - Performance, 15s and Parquet

Release owner: [@calofonseca](https://github.com/calofonseca).

### Summary

Consolidated sub-hourly support and improved performance for large datasets, especially 15-second datasets.

### Added/Changed

| Area | Changes |
|---|---|
| Loader | Windowed CSV and Parquet reads. |
| Cache | Shared weather/pricing/carbon when the file is identical and `noise_std=0`. |
| Parquet | 15s datasets can use `.parquet`. |
| Sub-hourly | Fixtures/tests for 15s, 1min, 5min, 15min and 1h. |
| PV | `absolute` mode for real datasets in `kWh/step`. |
| Unit contract | Formal documentation for `kWh/step`, `kW`, prices and emissions. |

## Pre-v0.3 - Initial EVs and Washing Machines

Fork owner: [@calofonseca](https://github.com/calofonseca).

### Summary

Before normalized deferrables, the fork introduced initial EV/charger support and early washing-machine flexibility.

### Added

| Area | Changes |
|---|---|
| EVs | EV definitions, charger schedules and connected/incoming observations. |
| Chargers | Charging/discharging actions and charger schedules. |
| Washing machines | First flexible appliance model. |
| Algorithms integration | Initial support for RBC/MADDPG and external algorithm repositories. |

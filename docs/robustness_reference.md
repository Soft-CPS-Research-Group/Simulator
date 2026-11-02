# Robustness Reference

Robustness is optional and dataset-driven. If `schema["robustness"]["enabled"]` is missing or `false`, `CityLearnEnv` behaves as before.

Portuguese version: [pt/robustness_reference.md](pt/robustness_reference.md).

## Schema

```json
"robustness": {
  "enabled": true,
  "events_file": "robustness_events.csv",
  "random_seed": 0,
  "missing_replacement_value": -9999.0,
  "modules": {
    "observations": {"enabled": true},
    "forecasts": {"enabled": true},
    "actions": {"enabled": true},
    "assets": {"enabled": true}
  }
}
```

`events_file` may be CSV or Parquet and is resolved relative to the dataset root.

## Events File

| Column | Meaning |
|---|---|
| `event_id` | Stable event identifier used in metadata and audits. |
| `module` | `observation`, `forecast`, `action` or `asset`. |
| `target_type` | `district`, `building`, `storage`, `charger`, `ev`, `pv` or `deferrable_appliance`. |
| `target_id` | Entity id/name, or `*` for compatible targets. |
| `target_feature` | Feature/action name. For assets use `telemetry`, `control` or `both`. |
| `start_time_step`, `end_time_step` | Inclusive global timestep window. |
| `mode` | Module-specific perturbation mode. |
| `value`, `std`, `min_value`, `max_value` | Optional mode parameters. |
| `replacement_value` | Missing/telemetry sentinel; defaults to `missing_replacement_value`. |
| `delay_steps` | Action delay horizon; defaults to `1`. |

## Modes

| Module | Modes |
|---|---|
| observation/forecast | `missing`, `noise`, `bias`, `stuck`, `clip` |
| action | `dropout`, `noise`, `bias`, `stuck`, `delay`, `clip` |
| asset | `unavailable` |

Observation and forecast corruption is agent-facing only. Rewards, physical KPIs and demand-response settlement still use the real simulator state. Action and asset-control events change the action effectively applied, so they can change physics and normal KPIs.

## Validation and composition

Event identifiers must be unique across the file, including disabled modules.
Global start/end steps must be non-negative integers with `end >= start`;
`delay_steps` must be a positive integer. Integral numeric values such as `1.0`
are accepted; fractional values are rejected rather than truncated.

Supplied numeric parameters must be finite, `std >= 0`, and supplied clip bounds
must satisfy `min_value <= max_value`. Empty CSV fields or Parquet nulls leave
optional parameters unspecified. An explicit invalid value such as CSV `nan`,
`inf` or text is rejected. Omitted bias/noise parameters still default to zero,
omitted delay to one step, and omitted clip bounds remain unbounded. The schema's
robustness seed must be an integer and its missing replacement must be finite.
Errors identify the event/field when definitions load, before episode execution.

Action-channel events compose in file row order, including when their windows
start at different steps. Bias then clip can legitimately differ from clip then
bias. Asset `control`/`both` unavailability is enforced **after all action-channel
transforms**, regardless of its row position: the affected command is zero.
Action-channel applications and the availability restriction are both recorded.
Delay history retains requested commands, and state is cleared on reset.

Control unavailability is a logical control restriction, not a physical
disconnection or a universal equipment failure model. Passive battery losses or
an already-running non-interruptible appliance cycle can continue; blocked new
commands cannot restart charging or start a new cycle through an action bias.

## Observation spaces and dynamic targets

In `known_faults`, only affected telemetry channels have their bounds expanded.
The envelope includes inactive event windows, file-ordered compositions, missing
replacements, clip limits and values retained by `stuck`. Positive Gaussian noise
has unbounded support, so its affected channels have infinite bounds. Bias/clip
values remain unmodified; bounds round outward to accommodate float32 tables.
Disabled modules and action-only events do not widen telemetry spaces. In
`hidden_faults`, numeric table bounds remain uniformly unbounded so they do not
reveal which channel is faulty; the feature allowlist and metadata masking still
apply, and relation/edge spaces are unchanged.

Dynamic targets are checked against the declared member/asset inventory, including
future member clones and asset additions. Validation uses a lightweight structural
view without resetting or controlling the live equipment. Unknown IDs/features
and disabled control ports remain errors. A known target may be absent for all or
part of an event window: no perturbation/application is recorded during absence.
Targets use the stable IDs in entity specs. Stuck samples and delayed requested
commands retain their logical event/target identity across absence; reset clears
that state. Declared future features are also valid in hidden-mode allowlists,
without permitting clean alternate measurements or diagnostic labels.

## Entity Diagnostics

Enable the diagnostic bundle when using entity observations:

```json
"observation_bundles": {
  "entity_robustness": {"active": true}
}
```

This adds district features for active state, previous-step corruption counts and active asset outages. Lightweight metadata is also exposed in `observations["meta"]["robustness"]`.

## KPIs

When robustness is enabled, `evaluate_v2()` adds district/building counters for event count, active timesteps, observation/forecast/action corruptions, asset unavailable applications, missing observations and action dropouts.

These legacy counters retain their meaning: unique applied event/target effects
per global index, including queried **terminal telemetry**. Re-reading the same
observation does not increment a counter. Channel applications are recorded before
the hidden-mode allowlist mask; shared district channels have no building owner.
`robustness_active` and `active_event_ids` describe enabled event windows, even
when their target is absent; `applied_event_ids` identifies actual effects at the
queried index. Scheduled activity is not proof that equipment was affected.

`env.get_robustness_audit()` returns `robustness-audit-v1` with separate `legacy`
and `executed_intervals` summaries, the latter restricted to completed physical
intervals. `district` and `events` expose configured/enabled/scheduled events,
queried observation indices, completed intervals, terminal/unexecuted observations
and window/applied event-step counts. Coverage is the number of event-steps with
at least one matching application divided by the corresponding enabled window
event-steps. It is not individual sensor uptime or physical service compliance;
wildcards still count once per event-step. Empty denominators are `null`. Sensor
opportunities use indices actually queried, including the terminal one; control
opportunities use completed intervals. Fast rollouts without observations do not
invent sensor applications. Observation-index tracking is off when robustness is
disabled. Queries are passive and return an independent snapshot.

The same audit is in `evaluate_v2().attrs['robustness_audit']`. KPI exports with
robustness enabled add `<kpi-file-stem>.robustness.json` beside CSV/Parquet outputs.
Existing KPI rows/names/formulas are preserved; the JSON provides explicit
populations, denominators and coverage for the researcher, not hidden-policy input.

In `MultiCommunityEnv`, robustness stays independent per child community. Portfolio rows aggregate robustness `_count` KPIs in the same way as other count KPIs.


## Agent observation contract

The default mode is `known_faults`. Entity studies without event labels must
explicitly select `hidden_faults` and a feature allowlist. Unlisted numeric
channels are masked; direct clean aliases of noisy measurements are rejected.
See [comparison and observation contracts](experiment_comparisons.md) for configuration, boundaries and migration.

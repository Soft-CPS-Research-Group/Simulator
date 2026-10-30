# Deferrable appliance service audit

Iteration 2 adds `deferrable-service-audit-v1`. It accounts for executed appliance
service independently of the full profiles reserved by `start_cycle`. Existing
`deferrable_appliance_service_*` KPIs retain their historical definitions; use the
new `building_deferrable_appliance_audit_*` and
`district_deferrable_appliance_audit_*` families for complete request accounting.
This definition is released in version 1.9.5.

## Population and states

A logical request is identified by member, appliance name and cycle ID. It is
**offered** when its admissible start window intersects an actually executed
interval while the appliance is active, and at least one full profile can fit
before the episode's terminal observation and the request deadline. Eligibility
is a calendar/window definition, not a claim that electrical supply was sufficient.
A request blocked by an outage can therefore become missed service.

Each offered request has exactly one state:

| State | Meaning | Remaining energy |
| --- | --- | --- |
| `pending` | Offered, not started, start window still open | Outstanding |
| `running` | Started, final profile interval not executed | Outstanding |
| `completed` | All profile intervals executed and requested energy delivered | Zero within tolerance |
| `missed` | Start window elapsed, or finished profile has a measured energy shortfall | Final unserved |
| `cancelled` | Eligible request interrupted by asset/member removal | Final unserved |

Removal reasons are `asset_removed` and `member_removed`. A start deadline already
elapsed takes precedence over a later removal. A physically completed request
remains completed. Pending/running requests at an intermediate query are unresolved;
they are not counted as completed or converted to failure just because a report
was requested. Completion uses an absolute/relative energy tolerance of
`1e-6 * max(1, requested_kWh)`; residuals remain visible rather than being rounded away.

The audit covers loaded catalogues of members active at least once in the episode;
never-activated members are outside this population. Excluded catalogue requests identify `expired_before_episode`,
`expired_before_activation`, `not_yet_available`, `removed_before_available`,
`outside_episode_service_horizon`, or `already_accounted_previous_instance`.
Excluded energy is reported separately and does not enter the service denominator.
The terminal observation is never a served interval. A fresh episode has no offered
requests until its first interval executes.

Reinstallation/re-entry retains the removed instance's history and does not offer
an already offered logical request again. Requests not yet available before removal
may be offered by the later instance. Member IDs are part of the identity: moving
an appliance to a different member creates a different request population.
Aggregated exports contain one row per logical request, preferring its offered
owner to excluded copies. `duplicate_offered_count` and physical residuals expose
unexpected duplicate ownership instead of concealing it.

## Energy and ratios

Only actual per-interval device consumption is attributed to running profiles.
Future reservations do not count as service. Cancellation clears future reserved
energy while preserving the executed prefix. Across offered requests:

`requested_energy_kwh = served_energy_kwh + final_unserved_energy_kwh + outstanding_energy_kwh`

`physical_balance_residual_kwh = physical_energy_kwh - served_energy_kwh`

`service_balance_residual_kwh` is the residual of the first equation. Both should
be zero within floating point tolerance. The independent physical total includes
all historical appliance instances; unattributed consumption remains visible.
The interval ledger runs in both normal steps and internal no-feedback rollouts.
Queries are pure, repeated interval recording is idempotent, and reset clears it.

| Ratio | Numerator | Denominator |
| --- | --- | --- |
| `observed_completion_ratio` | Completed | All offered requests |
| `resolved_completion_ratio` | Completed | Completed + missed + cancelled |
| `resolved_coverage_ratio` | Completed + missed + cancelled | All offered requests |

Every count, exclusion reason and state reason is exported. Zero denominators are
undefined (`None`/empty CSV cells), not perfect service. Interpret completion ratios
together with coverage and energy; a running cycle has delivered some service even
though it has not completed.

## API and export

`env.get_deferrable_service_audit()` returns the definition, per-request records,
building summaries and district summary. Request times use global dataset indexes,
including cancellation in a simulation starting partway through the calendar.

`env.export_final_kpis()` writes an adjacent `.deferrable_service.json` alongside
its CSV or Parquet KPIs and `.manifest.json`. The manifest identifies the audit
definition and denominators. Multi-community exports provide this evidence for each
child community; no cross-community logical-request deduplication is claimed.

Action/observation interfaces, datasets, price conventions and physical dispatch
rules are unchanged. Sparse calendars are no longer filtered by a local loading
window or the first membership period. This restores requests previously omitted
from partial-window and re-entry experiments and can change their consumption.
Lifecycle fixes affect appliance removal/re-entry: interrupted
requests stay cancelled and cannot be replayed as a new request under the same
identity. No empirical calibration of appliance profiles, user satisfaction model,
annual controller ranking or general repair of other assets' member re-entry
accounting is claimed.

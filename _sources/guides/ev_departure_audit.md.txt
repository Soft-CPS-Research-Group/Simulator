# EV departure audit: information quality and observed service

Definition: `ev-departure-audit-v1`. The additive family is exported by
`evaluate_v2()`, final KPI CSVs and the installed distribution. Each name starts
with `building_ev_departure_audit_` or `district_ev_departure_audit_`.
The export manifest records this definition, population, denominators and limits.

## Population and compatibility

Count actual departure events on active charger instances in the evaluated member
window, including events
without a target, vehicle history or observed SOC. A transition away from a
connection, an identity/session change, or a declared terminal departure is an
event. An open session at the end of an episode is not an invented departure.
A missing schedule cannot establish an event and is outside this population.

Service and feasibility are independent questions. A departure may have known
observed service but unknown feasibility, or a known physical opportunity but
missing observed service. Neither unknown case becomes success or failure.

The historical `*_ev_events_*` and `*_ev_performance_*` names keep their existing
definitions. In particular, their feasibility helper remains permissive when
input data is missing; its battery adapters and arrival fallbacks are retained.
Its invalid/non-finite interval duration now follows that same permissive rule
instead of attempting an invalid replay. Do not combine historical and audited
feasibility counts. The default scorecard selects the audited service ratios and
also includes coverage and unknown counts. Consumers with fixed KPI lists must
select the new names explicitly; exporting them does not change external jobs.

## Counts and denominators

For each threshold `T` in `target`, `min_acceptable`, `within_tolerance`:

| Suffix after the family prefix | Definition |
| --- | --- |
| `departures_count` | All detected departures, D. |
| `service_observed_count` | Departures with a valid target and finite observed SOC in [0, 1], O. |
| `service_unknown_count` | D − O. |
| `service_coverage_ratio` | O / D. |
| `T_feasible_count` | Confirmed individually feasible departures, F. |
| `T_infeasible_count` | Confirmed individually infeasible departures, I. |
| `T_unknown_count` | Insufficient information, U. D = F + I + U for each T. |
| `T_coverage_ratio` | (F + I) / D. |
| `T_unknown_REASON_count` | One primary reason per unknown departure; reason counts sum to U. |
| `T_service_met_count` | Observed departures meeting the service criterion, M. |
| `T_service_observed_ratio` | M / O, without filtering on feasibility. |
| `T_feasible_service_observed_count` | Confirmed feasible departures with observed service, OF. |
| `T_feasible_service_met_count` | Departures in OF that meet the service criterion, MF. |
| `T_service_feasible_ratio` | MF / OF. |

Zero denominators produce `None`/NaN (blank in CSV), not zero or full success.
District counts sum building counts; district ratios are recomputed from those
counts, never averaged across buildings.

The target criterion is SOC ≥ requested SOC. Minimum acceptable service uses
SOC ≥ target − `ev_departure_service_tolerance`. Symmetric accuracy uses
abs(SOC − target) ≤ `ev_departure_within_tolerance`. The latter's feasibility
test concerns reaching its lower SOC threshold; it does not prove that every
charging policy can stop inside the symmetric band. Existing numerical SOC
tolerance is retained.

Example: four observed departures, two confirmed feasible, one infeasible and
one unknown. If one feasible and the unknown departure meet the target, coverage
is 3/4, observed service is 2/4, and service among confirmed feasible departures
is 1/2. The unknown departure is not credited as confirmed feasible.

## Missing information: exclusive reasons in precedence order

1. `missing_target`: target absent, non-finite or outside supported SOC units.
2. `missing_ev`: no vehicle in the connection history at departure.
3. `missing_battery`: the recorded vehicle has no battery.
4. `invalid_capacity`: capacity missing, non-finite or non-positive.
5. `invalid_charger_power`: charger power missing, non-finite or negative. Zero
   power is valid: initial energy may suffice, subject to standby losses.
6. `invalid_charger_efficiency`: efficiency missing or outside [0, 1].
7. `invalid_time_step`: resolved charger/battery/environment interval missing,
   non-finite or non-positive. There is no implicit one-hour audit fallback.
8. `missing_arrival_soc`: no valid boundary SOC. Use the schedule's current SOC,
   an applicable arrival estimate, or initial SOC at episode index zero; never
   use the post-action departure SOC as its own starting evidence.
9. `unsupported_battery_model`: a historical adapter does not implement the
   native Battery replay contract. Battery and InventoryBattery are supported.
10. `invalid_battery_parameters`: native power, efficiency, standby-loss or SOC
    bounds are invalid, or the replay yields a non-finite SOC.
11. `outage_supply_not_replayed`: the conservative replay fails during a session
    with an outage, and local PV/storage allocation has not been replayed.

No early success shortcut is used for a zero threshold before validating the
data. Unknown means the audit cannot establish the answer, not that the
simulation necessarily contains invalid data.

## Window, dynamics and outage limits

Replay starts at the observed session boundary. It intersects the schedule with
the connection history, so it cannot borrow charging time before a dynamic
activation or from an earlier back-to-back session. A window that starts inside
a session evaluates feasibility from its declared boundary SOC, not from an
unobserved original arrival. Removed chargers remain available through the
topology history; a replacement does not duplicate departures scheduled while
that instance was inactive. Repeated evaluation does not advance the real vehicle, battery,
episode, or physical trajectory.

The replay remains individual, conditioned on the existing electrical-service
limits and base-load estimation. It is not a joint scheduler for competing EVs
and does not prove joint service feasibility. Default model parameters are part
of the declared model, not field measurements. It does not reconstruct all
robustness events or solve local outage dispatch.

During an outage the audit conservatively assigns zero charging to the replay.
If subsequent charging still reaches the threshold, that is sufficient for this
individual test. If it does not, the result is unknown rather than a false
infeasibility claim: actual local generation/storage may have served the EV.
Observed departure service is still reported independently. Deferred charging
that is recovered before departure is not missed service. Tests include both
recovery and an actual departure shortfall with physical invariants enabled.

## Validation scope

Regression cases cover each missing-data reason, analytical mixed populations,
threshold differences, zero denominators, censored and partial sessions,
activation boundaries, back-to-back sessions, removed chargers, reset,
non-mutating evaluation, outage recovery/shortfall and CSV/manifest exports.
Historical KPI tests remain part of the compatibility check. This iteration
does not modify datasets, physical dispatch, rewards or market pricing, and
does not claim an annual campaign or calibration against real equipment.

# Auditing demand response, thermal service and equity

Iteration 5, included from v1.9.8, introduces `kpi-families-audit-v1` and
`demand-response-audit-v1`. These definitions are absent from v1.9.7. These are
model-consistency checks and definitions, not calibration against real equipment.
Datasets, dispatch, market rules and reward functions are unchanged.

## Interfaces and evidence

`env.get_kpi_family_audit()` returns member/district thermal populations and
district equity coverage. It accepts the same optional `comfort_band`,
`control_condition` and `baseline_condition` as evaluation. It is read-only.
V2 adds `building_thermal_service_audit_*`, `district_thermal_service_audit_*`
and `district_equity_audit_*`. The default scorecard pairs service rates with
coverage. The export manifest identifies definitions, DR preparation/aggregation
and the resolved reward class, configured attributes and available public
parameters. Custom reward semantics still require inspection. Core settlement
does not automatically add DR cash to the grid bill or reward.

Thermal/equity evaluation supports flat and entity interfaces. Demand response
continues to require entity. Simultaneous overlapping DR requests remain explicitly
unsupported; the existing rejection is preserved. No UI application is modified.

## Demand response

The baseline is the mean of the **completed** pre-event net-power samples. The
preparation interval count is `ceil(baseline_window_seconds / step_seconds)`, with
at least one interval, so the requested duration is never rounded down. A 3600 s
window needs four 900 s samples or two 2500 s samples. The effective duration is
reported. This does not resample a dataset or create history before an episode.

The preparation period and settled event interval must have stable structural
topology. A member/asset change makes the reference invalid, including after an
initially valid cached baseline. Invalid intervals retain requested energy but
receive no inferred delivery, revenue or penalty. They are excluded from valid
service denominators and expose a reason: insufficient history, nonfinite history,
no members or topology change. A lower penalty due to missing reference is not
evidence of better control. A rolling baseline remains an operational reference,
not a counterfactual causal model; previous load/DR behaviour can affect it.

For step duration `h`, baseline power `B`, actual power `P`, target `T` and
tolerance `epsilon`, in the declared **load perspective**:

- `up`: delivered power is `P - B`; `down`: `B - P`.
- Signed delivered energy is delivered power times `h`.
- Credited energy is `min(max(delivered_power, 0), T) * h`.
- Shortfall is `max(T - delivered_power - epsilon, 0) * h`.
- Revenue and penalty multiply their corresponding energy by the declared prices;
  net revenue is revenue minus penalty. Compliance is credited / valid requested
  energy. No valid requested energy gives an undefined rate.

Member signed delivery, requested energy and cash are additive. Member credit is
the **uncapped positive contribution**; member shortfall uses an equal share of
the target and does not apply district tolerance. They are allocation diagnostics,
not additive versions of the district credited energy or shortfall. For example,
two members delivering +2 and -3 kW against a 2 kW downward request over 15 minutes
give district delivery -0.25 kWh, credit 0, shortfall 0.75 kWh. Individual credits
sum to 0.5 kWh and individual shortfalls to 1 kWh; cash still reconciles. These
established allocation rules are unchanged.

Request IDs must be unique. An optional request-file `purpose` column or
`demand_response.request_purposes` map declares `service`, `stress` or `unspecified`.
The map takes precedence and unknown IDs/labels are rejected. Without a
declaration the purpose is **unspecified**. Feasibility, purpose and observed
fulfilment are separate questions; a failed request is never automatically called
stress. This audit does not prove joint DR scheduling feasibility.

District `demand_response_audit_{all|service|stress|unspecified}_*` metrics expose
unique events, active/valid interval counts, exclusion reasons, requested/valid
requested energy, signed delivery, credit, shortfall and cash. Baseline coverage
is valid / active intervals; energy coverage is valid requested / requested energy.
Empty denominators are undefined, not an automatic success.

## Comfort and resilience

The evaluation population is **executed intervals while a member is active**.
An unexecuted current observation is excluded even during a partial episode.
Inactive gaps never become fictitious zero-temperature samples. An occupied
interval has finite occupancy greater than zero. Temperature, both setpoints and
band must be finite, band nonnegative and heating setpoint no greater than cooling.
Unknown occupancy and unevaluable occupied samples have explicit counts/coverage.

Cold means `temperature < heating_setpoint - band`; hot means
`temperature > cooling_setpoint + band`. Equality is comfortable. Rates divide
discomfort intervals by evaluated occupied intervals, not person count. Native
per-interval building bands are honoured unless an explicit evaluation override
is supplied. The legacy rolling ratios now use cumulative observed occupancy;
cold/hot deviation min/max/mean condition on actual occupied violations. No
violation gives undefined severity, not a fabricated zero deviation.

The audit exposes occupied/evaluated/discomfort hours and cold/hot degree-hours.
Degree-hours use absolute deviation from the corresponding setpoint, restricted
to violating intervals. During outages, the same definitions apply to the
occupied outage population. No evaluated occupied outage sample gives an
undefined outage rate. District audit rates pool exposure and energy before
division; historical district comfort aggregates retain their member-mean form.

Service energy covers space cooling, space heating, hot water and base electricity.
It sums demand energy across those services; it is **not primary electrical
acquisition**. Device/demand series are converted from dataset resolution to
control-step kWh; thermal storage balances already use control-step kWh. Deficit
is calculated separately for each service before summing, so oversupply of one
cannot cancel an unmet demand elsewhere. The legacy unserved-energy caller also
uses these corrected units/component caps. This is reporting, not a dispatch fix.

`unserved_service_ratio` is total deficit / total known expected service energy;
the outage variant restricts both to known outage intervals. Missing/negative
energy inputs exclude that interval and reduce energy coverage. No expected
energy gives undefined audit ratios. The historical low-level zero-demand
unserved function still returns zero for compatibility; use the audit's population
and coverage to distinguish zero, unknown and not applicable. Appliances and EV
departure service retain their dedicated audits from iterations 1 and 2.

## Equity

Relative benefit remains `100 * (reference_cost - actual_cost) / reference_cost`.
Finite costs and strictly positive reference cost are required. The audit reports
eligible population, excluded unknown/nonpositive-reference counts and coverage.
Costs use each member's executed active participation and the **actual evaluation
conditions**, which appear in the V2 attributes. Different participation periods
remain different exposures; this is relative bill benefit, not equal opportunity.

Losers include eligible negative benefits. Gini and top-20 concentration use the
nonnegative part, retaining zero positions in the eligible population. Top-20 uses
`ceil(0.2 * eligible_members)`. No positive benefit mass makes both undefined.
The audit independently uses a sorted-rank Gini identity and is checked against
the legacy pairwise calculation and hand examples. Read concentration with losers
and coverage. BPR retains manually declared rich/poor groups and positive-part
group means; absent/invalid groups or a nonpositive rich mean are undefined.

## Validation and bounded real runs

```bash
python scripts/audit/audit_kpi_families.py \
  --schema data/datasets/rec_pt_2023_premium_100_v2/evaluation/premium_100_allin.json \
  --request-id development_dr_1 --policy normal --seed 0 \
  --output /tmp/dr-audit.json
```

The CLI includes the required preparation and terminal observation. It independently
reconstructs DR validity, energy/cash and additive allocations from completed raw
power samples; it compares exported district DR totals. Thermal/equity evidence is
a snapshot of the separately tested audit populations, not a second physical model.
Tolerance is `1e-5 + 1e-7 * max(abs(actual), abs(expected))` in the compared units.
An invalid reference can pass accounting consistency while its evaluation quality
remains incomplete; the report keeps these statuses separate.

Regression tests cover independent examples, zero/unknown denominators, negative
benefits, partial/inactive participation, native bands/overrides, 60/900/3600 s
units, already-scaled thermal tanks, duplicate IDs, invalidated cached references,
purpose groups, read-only queries and reward preservation. The eight Premium
development requests are exercised without editing inputs. The 2023 fourth request
coincides with retirement of Member_093's PV at index 30696: all its 16 intervals
have an invalid structural reference. The other seven have complete references.
The inputs declare no purpose, so all remain unspecified.

These corrections require rerunning affected comfort, resilience and DR analyses.
They do not claim annual validation, equal controller service, global optimality,
an empirical thermal model or calibration. Full temporal/data alignment and
reproducibility remain iteration 6; long-run resource validation remains iteration 7.

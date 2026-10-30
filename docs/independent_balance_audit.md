# Independent balance audit

Iteration 4 introduces `independent-flow-audit-v1` as an optional offline audit.
It is included from v1.9.7 and is absent from v1.9.6.

The reference calculation is implemented in `citylearn/internal/balance_audit.py`.
It does not call the production settlement, provenance or KPI formulas to compute
expected values. It compares independently calculated results with those APIs.
This demonstrates consistency of the simulated model under the exercised cases;
it does not calibrate that model against real equipment.

## Run a bounded window

```bash
python scripts/audit/audit_balances.py \
  --schema data/datasets/rec_pt_2023_premium_100_v2/evaluation/premium_100_allin.json \
  --start 4176 --transitions 14 --policy normal --seed 0 \
  --output /tmp/balances-4176-4190.json
```

The example executes indexes 4176 through 4189 and retains observation 4190 as
the terminal boundary. To include the 12 hours of preparation, use `--start 4132
--transitions 58`. Both windows keep the runtime physical checks active, use the
unaltered dataset and disable automatic KPI export. The CLI also supports zero
actions or bounded seeded actions. A failed balance gives a nonzero exit code
and an error in the JSON report. It does not relax checks to finish a window.

For a programmatic audit, reset the environment, attach before the first action,
and finish before another reset:

```python
from citylearn.internal.balance_audit import IndependentBalanceAudit

env.reset(seed=0)
with IndependentBalanceAudit(env) as audit:
    while not (env.terminated or env.truncated):
        env.step(actions_for_this_interval())
    report = audit.finish()
```

The context instruments completed transitions, including the no-feedback path,
then restores the original methods on exit. It does not choose or modify actions.
An audit must begin at episode-local index zero; a nonzero dataset start is valid.

## Reference calculations

- Electrical demand is summed from served base, thermal, appliance and escalator
  loads and actual storage/charger terminals. Islanded PV surplus is a separate
  curtailment sink. Each bus must close before the accounting comparison.
- Storage journals provide ordered terminal and inventory measurements. Each
  operation is checked against its initial/final energy, losses, device terminal
  sum and interval power-time budget. Nameplate duty is a necessary bound, not a
  proof of detailed feasibility within an interval.
- A directed flow network solves Kirchhoff concentration balances with a dense
  least-squares calculation. It uses explicit conversion, storage and destination
  nodes instead of the production ledger's affine mixing implementation. Charge
  conversion losses are removed before mixing with existing stored inventory.
- Community imports use an independent progressive-filling calculation at demand
  saturation breakpoints. Seller contributions remain proportional. Disabled
  financial settlement still allows the declared physical community netting;
  disabled local trading within an enabled market does not invent a local trade.
- Tariffs are independently read from configuration/CSV, including the global
  offset of partial episodes. Member bills, local buy/sell cash and operator
  balances are checked against external grid payments. With asymmetric or
  heterogeneous tariffs the operator balance can be nonzero.
- Grid acquisition carbon travels with its stored cohort. Initial/arrival energy
  and acquisitions without finite carbon intensity remain explicitly unknown.
  A later grid intensity does not relabel old stored energy as cleaner energy.
- Source/destination totals are compared after transitions; final V2 member and
  district energy/cost/provenance and gross BESS/EV totals are reconciled.
  Delivered appliance energy is checked against the request service audit.

Initial/final inventory is reported explicitly. No invented terminal price credit
is added to bills. Stored energy, losses, curtailment and asset exits are not useful
building consumption. Existing service audits retain their own denominators and
feasible/infeasible/unknown classifications.

## Accounting boundaries

The boundary contains active member AC buses, their BESS and connected EVs.
Thermal equipment is electrical end use; thermal tank provenance is not tracked.
Community exchange is net over a control interval. Storage operations retain
order, but the proportional bus convention does not reconstruct subinterval
chronology or identify individual electrons.

Departure/removal retires the last settled inventory into mobility or asset exit.
External SOC replacement retires the old cohort and creates an unknown cohort.
Re-entry preserves completed energy/cost and retired-device history, while new
physical equipment still starts from its configured initial state. Energy totals
exclude inactive gaps and terminal observations. A terminal arrival may contribute
real boundary inventory without contributing another interval of energy or cost.

The reference independently derives accounting from physical measurements; it
still trusts the model's raw served loads, event journals and configuration.
It cross-checks journals against terminals and stored energy, but is not an
independent electrochemical or charging-controller implementation. Custom models
with missing journals are outside this audit contract. Legacy custom KPI facades
without gross traces keep their existing net-series fallback, which cannot prove
gross throughput. Direct replacement of terminal readings likewise cannot recover
unrecorded operations and is not a physically validated dispatch.

## Tolerances, evidence and migration

Every comparison uses an explicit absolute tolerance of `1e-5` plus `1e-7` times
the larger magnitude, in the compared quantity's units (kWh, kgCO2 or EUR). Reports
include comparison count, maximum absolute error and maximum fraction of allowed
error. These maxima combine different quantities; inspect a failed label before
interpreting its units. Least-squares relative rank cutoff is `1e-12`. Unlabelled
source-free circulation is reported separately rather than assigned a PV origin.

Regression coverage includes 60/900/3600-second steps, fixed seeded action
sequences, outages, multiple storage ports, three-phase import/export limits,
member/asset changes, negative/asymmetric tariffs, unknown carbon, partial
appliance service, terminal-only PV and opposing storage movements. Instrumented
and uninstrumented runs are compared to check that the audit remains passive.

Four reporting corrections require rerunning affected experiments:

1. Member re-entry previously erased completed per-member energy/cost and some
   equipment history, while district records could still contain the old flows.
2. Some totals included an unexecuted terminal observation or inactive PV gaps.
3. Islanded unused PV was attributed to useful load. Provenance definition **1.1**
   adds `curtailment`; absorption excludes that rejected generation too.
4. Opposing BESS/charger calls in the same interval could cancel gross charge and
   discharge in reported throughput. Gross traces now retain both directions.

These corrections do not alter dispatch, datasets, tariff rules or the native
reference objective. They can change reported energy, cost, solar-use and cycle
totals. Do not mix provenance 1.0 and 1.1 exports as if their definitions matched.

The audit retains flow/cohort evidence and solves dense systems; it is intended
for bounded offline runs, not default online operation. Per-device gross traces
are bounded episode arrays. A long-run memory/performance campaign remains
iteration 7. Full demand-response, comfort, resilience and equity definitions are
iteration 5; dataset time alignment and full reproducibility are iteration 6.
No annual controller ranking, network power-flow proof, battery-aging calibration
or field validation is claimed here.

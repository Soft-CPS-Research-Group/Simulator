# Continuous campaign and resource evidence

Release v1.9.10 records iteration 7 and adds the offline `long-run-audit-v1` CLI. It keeps runtime physical
checks active throughout a continuous episode and records the exact schema,
loaded-input hashes, software versions, NormalPolicy parameters and audit-script
hash. It does not change datasets, dispatch, market pricing or rewards.

## Run

Run this repository script from a source checkout of v1.9.10 or later, with the
required dataset available. It is not an installed PyPI console command.

```bash
python scripts/audit/audit_long_run.py \
  --schema data/datasets/rec_pt_2023_micro_4_v2/evaluation/core_4.json \
  --start 0 --transitions 35040 --block-steps 960 \
  --seed 0 --output /tmp/micro4-year2023
```

The output directory must be empty. The requested window must contain its terminal
observation: 35,040 dispatched 900-second intervals require 35,041 observations.
The CLI preserves the declared native market: internal price 80% of purchase
price, zero export price. Other market contracts require their own audit.

NormalPolicy is identified as a privileged-state policy. Declared robustness seeds
and events remain enabled; `--seed` does not silently rewrite explicit schema
seeds. The campaign is not a comparison between controllers.

## What is checked

Measurements are captured after dispatch/settlement and before advancing topology
to the next interval, so equipment replacement cannot rewrite an executed sample.
Every executed interval checks finite reward, active member identities, measured
device terminals against the member bus, excess local energy as curtailment in a
grid outage, net import/export against settlement, internal energy/cash balance
and native prices. Independent streaming totals are retained by member identity,
including removed/re-entered members, and compared with final member/district
KPIs. Appliance physical energy is reconciled with the service audit. Runtime
checks independently enforce storage energy changes, SOC and equipment powers.

The final report also checks the production provenance ledger's cumulative
energy/carbon residuals, records source/destination inventories and distinguishes
service coverage/unknowns from successful departures. Capacity changes are the
model's simplified degradation, not empirically validated ageing.

This CLI does not solve a second independent producer/recipient flow network at
every annual interval. The independent-flow audit from iteration 4 and the
physical CI matrix provide that separate, bounded evidence. The long campaign
extends duration and checks aggregate/meter consistency; it does not relabel
production provenance residuals as an independent provenance proof.

## Files and resources

`intervals.csv` contains executed global indexes, active member counts, energy,
cost and outage membership. `summary.json` is replaced atomically after each
block and on completion/failure. Errors identify the failed phase and dispatch
index where applicable. Progress is evidence, not a resumable environment snapshot.

`kpis.csv`, `service.json` and `manifest.json` accompany successful runs. Undefined
EV ratios are JSON null; physical non-finite values fail the audit. Failed runs
are not accepted even if they have partial exports.

RSS and peak RSS are Linux process measurements, including the interpreter,
loaded inputs, runtime histories and audit overhead. Blocks record duration,
active population, settlement-history length and retained provenance aggregate
bytes. These distinguish expected episode-history growth from aggregate growth;
they do not establish leak freedom for every configuration or reset pattern.
Resource limits are measured after load/reset, at blocks and after final
evaluation/export; they are not an OS-level hard memory/time reservation.

The declared campaign uses an 8,192 MiB RSS threshold and 14,400 seconds per case.
The report records actual observed consumption and whether each threshold passed.
Extrapolating an intermediate Premium window to a year is indicative: changes of
topology, seasonal activity, allocator behaviour and final evaluation can change
both time and memory. A projection is not annual validation.

## Selected campaign

The prepared sequence starts with a full 2023 year in Micro-4, then a full 2023
year in Core-30 with dynamic topology/robustness, then 60 continuous days in the
nominal Premium-100 configuration from global index 39,360. The latter crosses
29 February and the spring 2024 DST transition. Actual active members, not nominal
configuration names, determine reported participation.

The machine-readable plan and final case reports identify precisely which stages
passed, failed or were not run. Prior iteration 6 windows supply separately scoped
calendar/replay evidence, including autumn 2024. No claim of continuous execution
of both years in every Portuguese configuration follows from these selected runs.
Empirical calibration remains iteration 8 and requires appropriate measurements.

The `topology_events` field counts topology log entries, including replay before
the selected window. It is not a count of structural events inside that window.
The campaign summary separately reports declared events inside executed indexes.
Service shortfalls are reported alongside coverage; a feasible individual replay
does not certify joint scheduling feasibility or policy success.

## Local selected-campaign result (2026-10-30)

| Caso | Intervalos | Dias | Membros ativos | Minutos | Pico observado MiB |
| --- | ---: | ---: | --- | ---: | ---: |
| micro4-year2023 | 35040 | 365 | 4–4 | 3.89 | 692.3 |
| core30-year2023 | 35040 | 365 | 26–27 | 28.81 | 2029.9 |
| premium60days2024 | 5760 | 60 | 88–88 | 19.62 | 1708.3 |

All three selected cases passed, with runtime physics enabled, independent meter/cash
reconciliation and service/KPI exports. The final local suite passed 767 tests
(10 new). Premium annual execution was not performed; projections are indicative.
These selected-campaign results are recorded in v1.9.10; they were measured on
the identical physical runtime in v1.9.9. The source-checkout audit CLI and these
reports accompany the release; the PyPI wheel contains the runtime, not repository
audit scripts or datasets.

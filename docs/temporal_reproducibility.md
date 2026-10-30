# Temporal coherence and reproducibility

Release v1.9.9 introduces `temporal-evidence-v1` in iteration 6. Existing datasets are not rewritten. The work verifies declared
model timing and selected deterministic replays; it is not field calibration or
an annual certification of every controller.

## One source-row clock

An optional schema `time_axis.calendar_file` points to a CSV/Parquet calendar with
contiguous `time_step` indexes starting at zero and unique, increasing, regular
`timestamp_utc` instants. A supplied `timestamp_local` must describe the same
instants. Portuguese datasets automatically use their existing
`shared/calendar.parquet`. The IANA `timezone`, when supplied, determines display
time. No timezone or DST information is inferred from repeated local clock hours.

The canonical UTC spacing overrides inference from the first two local hour/minute
labels. Otherwise a spring jump can look like a 75-minute sample and an autumn
fold like a 23-hour-15-minute sample. Native datasets remain on 900-second UTC
intervals through both transitions. Default exported timestamps use the canonical
sample index and include the local offset. The two occurrences of a repeated local
hour remain different instants. Leap-day and year boundaries use the same axis.

An explicit render `start_date` override retains its historical display behaviour
and is marked in metadata. Without a canonical calendar, legacy naive timestamps
remain available; the manifest does not invent a verified UTC origin. An optional
`dataset_seconds_per_time_step` declares spacing where no calendar is available.
It must agree with a supplied calendar. Declaration alone does not independently
verify the civil labels of source rows.

Changing `seconds_per_time_step` still **does not resample rows or events**. The
source row clock, control duration, per-member inferred spacing and consistency
flag are explicit. Runtime energy scaling is labelled
`none_runtime_energy_scaling_only`. Such a run is not a resolution-equivalence
experiment simply because its total energy has been scaled.

## Observations and executed intervals

The environment's current observation has not yet been dispatched. The manifest
separates its global index/UTC timestamp from the executed global interval range.
Daily member averages now use executed **active** participation, excluding an
unexecuted current observation and removal/re-entry gaps. Zero elapsed duration
has undefined daily averages. Historical custom facades without an environment
clock retain their compatibility convention.
These are rates per 86,400 seconds of active exposure, not means over civil dates
that can contain 23 or 25 hours.

With the usual N observations, an episode executes N-1 intervals. A full input
file with N interval values therefore needs an explicit terminal-observation
policy to execute the last interval. The existing
`terminal_observation_padding=true` holds the last observation and permits only
the terminal calendar index N to extend by one source interval. That extra
observation is not another energy dispatch or an invented future tariff sample.
No padding is silently added. This boundary matters when planning an annual run.

## Seeds and reset

`reset(seed=s)` now restarts the away-EV stream with s, independently of the
episode counter. Ordinary unseeded resets retain per-episode variation. Calendar
split selection still advances unless explicitly selected:

```python
env.reset(seed=7, options={"episode_index": 3})
```

Repeating that seed/index selects the same random or sequential split. Reset does
not freeze the split index by default. Random splits use the bounded
`sha256_seed_episode_v1` stream so seed zero varies across episodes and large valid
seeds cannot overflow on later episodes. This changes historical random split
sequences; rerun affected studies rather than mixing the two protocols. Reset does
not reconstruct equipment defaults or reload input noise: the **initialization
seed** and current environment/reset seed are recorded separately. A native BAU
loads with the source initialization seed, then uses the current environment and
away-EV seeds. It therefore retains initial defaults/noisy inputs after an explicit
reset seed change. Default robustness noise follows the reset seed; a declared
robustness seed remains its own independent choice.

Loader-created observation noise uses owned indexed streams. Component labels,
initialization seed, channel and absolute source-row offset determine the draws.
The same overlap of two input windows gets the same noise, and loading does not
consume NumPy's global RNG. Direct low-level data constructors without a noise
seed retain their legacy behaviour; custom controller/global Python/Torch RNGs
are not silently reseeded. Controllers must identify their own randomness.

## File and environment evidence

Before loading, the environment snapshots SHA-256 hashes and file identities of
declared supported input files, including model weights, plus the canonical clock.
Construction checks for intervening content changes. Exports retain the **loaded
hashes**, with changed/missing files reported separately. A metadata-only touch
with identical bytes does not invalidate an input. The native BAU rejects changed
inputs instead of comparing the old controlled data with a newly loaded reference.
The comparator does not treat that evidence as safely paired.

The manifest includes initialization/reset/away-EV seeds, episode index, clock
contract, padding, Python/platform and relevant installed dependency versions.
Pairing evidence includes temporal initialization conditions and runtime versions;
different noisy initialization cannot become paired merely by later resetting both
environments to the same seed. Historical paired-v1 exports without the temporal
extension do not certify the new temporal contract. Installed wheels still report
an unavailable Git revision rather than an unrelated checkout revision.

The file snapshot covers schema-declared dependencies and the discovered clock,
not arbitrary external reads by custom plugins. It records ordinary filesystem
content consistency, not an adversarial transactional filesystem guarantee.

## Explicit resampling utilities

`citylearn.internal.temporal.resample_series(values, source_seconds,
target_seconds, kind=..., weights=None)` requires a complete common duration:

| Kind | Rule |
|---|---|
| `energy` | Distribute interval kWh by overlap; sum on coarsening. |
| `power` | Average kW by overlap duration; preserve integrated energy. |
| `price`, `intensity` | Time mean by default; optional nonnegative energy weights preserve acquisition cost/carbon for those weights. |

Zero energy weight produces an undefined weighted value. Time averaging prices
does not generally preserve a bill with varying load. Categorical states and
events are rejected by the averaging API. `map_interval` requires exactly
representable boundaries and an explicit inclusive-end convention; arrivals,
departures and short appliance cycles cannot be silently rounded away.

`rescale_step_loss` converts a per-control-step loss fraction while preserving
retention over equal physical time. It does not calibrate a battery model.
Changing resolution does not automatically adjust model loss coefficients,
normalised capacity actions, short profiles or event schedules. These utilities
prepare explicit conversions; they do not automatically rewrite a whole dataset.
Resample interval values, not an unexecuted terminal observation, and declare how
the target terminal observation is supplied. Coarsening loses peaks and within-step
ordering even when total energy is conserved.

An analytical constant 2 kW load over one hour is exercised through the simulator
at 900 and 300 seconds. Both give 2 kWh, EUR 0.50 and 0.50 kgCO2 under the declared
piecewise tariffs and carbon intensity. That equivalence does not extend to
arbitrary storage/controller dynamics without additional assumptions.

## Bounded audit and limitations

```bash
python scripts/audit/audit_temporal.py \
  --schema data/datasets/rec_pt_2023_micro_4_v2/evaluation/core_4.json \
  --start 43683 --transitions 8 --seed 0 --output /tmp/time-audit.json
```

Without `--start`, the CLI checks source clocks, common row coverage, local labels,
structural event timestamps and EV catalogue boundaries. With `--start`, it also
compares repeated, fresh-environment and KPI-query rollouts with NormalPolicy and
physical checks. Exact trace equality is scoped to the exercised software/platform
and deterministic policy; it is not cross-hardware bitwise certification.

Validation covers nine native Portuguese configurations and selected windows
through New Year, 29 February, both DST transitions, Core-30 robustness and Premium
operation. Calendar checks verify 96 leap-day intervals and 92/100 local intervals
on the spring/autumn transition days. Event/session boundaries use their declared
inclusive/exclusive conventions; open terminal sessions remain unknown.

Rerun affected DST-started windows, explicit-seed repeated episodes, noisy inputs,
partial/re-entry daily rates and reference comparisons after release. Long-run
memory/performance and cumulative balances remain iteration 7; calibration remains
iteration 8. No annual controller ranking or empirical validation is claimed here.

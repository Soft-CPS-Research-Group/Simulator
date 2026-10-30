# Paired comparisons and reference interpretation

Iteration 3 introduces `paired-experiment-v1` evidence alongside KPI exports.
It is released in version 1.9.6.

## Identity and pairing

Before a custom-controller run, declare its implementation, version and resolved
parameters using `env.set_experiment_identity(implementation, version, parameters,
observation_access='agent_payload')`. Native operational agents provide their
actual identity and parameters when passed as `model` to `export_final_kpis`.
An unspecified controller remains unknown; a file name is not proof of identity.

The manifest records data hashes, a normalized physical configuration hash, actual
episode window and completed intervals, time resolution, effective away-EV seed,
initial inventory, service tolerances, resolved market, source-code fingerprint,
observation contract and reference definition. Moving the same data to another
root does not change the normalized contract. Changing an input, seed, target,
asset or reference cannot silently become a paired controller comparison.

Native BAU is identified by implementation, simulator version,
`immediate-full-charge-v1` and resolved parameters. It uses privileged simulator
state. The reference reuses the source episode's away-EV random stream, including
after repeated resets, and its resolved EV service tolerances. Initial inventory
must agree before reference KPIs are produced. Definition agreement with Algorithms
is not proof of numerical equivalence between independent implementations.

Use `citylearn.internal.comparison.compare_exports(left, right, mode='controller')`
or the repository CLI:

```
python scripts/audit/compare_experiments.py --left run_a/kpis.csv \
  --right run_b/kpis.csv --mode controller --output comparison.json
```

Each CSV/Parquet export must have its adjacent `.manifest.json`. V2 wide CSVs and
long-form name/cost_function/value exports are supported. Duplicate KPI/name rows
are rejected. The CLI returns 0 only for paired conditions; 2 means incompatible
or insufficient evidence. Numerical deltas remain available for inspection, but
`permitted_claim` is empty unless the pairing checks pass. Older exports without
the contract are unknown and must not be silently labelled matched. The existing
`compare_equity_runs.py` CLI uses the same gate, accepts V2 wide exports and reports
raw differences without a universal higher/lower-is-better assertion.

`controller` comparisons require equal market and observation conditions. Policy
parameters can differ. `sharing` comparisons require the same controller and
allow only the market/local-trading enable flags to differ. Tariffs, internal
price ratio, export price and settlement rules must still agree. Combined policy
and sharing changes are not isolated sharing benefits. Observation access must
agree: privileged internal-state access is not the same as agent payload access.

## Read outcomes together

The comparison report exposes district cost, audited EV/appliance service and
coverage, useful solar use, external purchase emissions and per-asset final
inventory differences. Active stationary storage and the whole EV fleet,
including away vehicles, have separate scopes. Final snapshots include boundary
events after the last transition. They are not an automatic monetary credit.
Lower cost with lower service or a depleted battery is a joint outcome, not proof
of equal-service operational savings. No arbitrary terminal-energy price is added.

Historical solar absorption can include charging a battery. Provenance-based
useful solar use counts final building load or mobility and reports stored energy
separately. External carbon purchases differ from internal carbon allocation;
internal trading does not erase purchased carbon. Read the actual zero-export
price and 80% local-price configuration from the resolved market, rather than
assuming every historical experiment used it.

## Hidden-fault agent observations

Compatibility defaults to `known_faults`: existing diagnostic metadata is visible
and the manifest says so. For a study without event labels, explicitly configure:

```json
"robustness": {
  "enabled": true,
  "events_file": "robustness_events.csv",
  "agent_observation_contract": {
    "mode": "hidden_faults",
    "features": {"building": ["non_shiftable_load", "hour"]}
  }
}
```

The entity interface is currently required. Every unlisted numeric feature is
masked to -9999 for the entire run, including fault-free intervals. Shapes and
entity identities remain stable. Entity IDs, edges and connection masks remain
visible as structural inputs. Metadata uses a safe temporal/topology allowlist;
event IDs, fault causes, runtime status and fault diagnostic features are absent.
Automatic action-execution diagnostics are also withheld. Audit metadata remains
available to the researcher through `get_metadata` and the manifest.

The contract rejects unknown features, fault diagnostic features, duplicate direct
measurement aliases and a selected clean alternate of a targeted noisy channel.
For example, targeting `non_shiftable_load` while exposing `load_power_kw` cannot
silently bypass the perturbation; the experiment fails configuration validation.
PV energy/power/building aliases and SOC aliases are also checked. Hidden-mode
observation bounds allow perturbed values and the sentinel. Physics, rewards and
KPI accounting use the underlying simulator state; masking alone does not change
those trajectories under identical actions.

The allowlist is part of the scientific experiment. Hiding labels does not prove
that a fault is statistically undetectable through independent measurements or
physical inference. Review derived signals before claiming a particular sensing
model. In-process agents with direct environment access, including native BAU,
remain privileged; the API payload contract is not a process-isolation boundary.
External analysis UIs must consume the pairing report before claiming a matched
comparison. This repository adds the packaged API and CLI, not an update to a
separately maintained UI. General re-entry energy-history reconciliation, annual
controller campaigns and empirical calibration remain later iterations.

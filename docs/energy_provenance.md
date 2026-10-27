# Energy provenance and carbon accounting (1.9.0)

The new `evaluate_v2()` family is `building_energy_provenance_*` and
`district_energy_provenance_*`. The definition version is **1.0**; the complete
KPI contract version is **1.9.0**. Both are present in environment metadata,
DataFrame attributes and the JSON manifest accompanying exported KPIs.

## Boundary and convention

The accounting boundary contains active member AC buses, stationary electrical
batteries and connected vehicle batteries. It follows actual terminal transfers
and changes in stored energy. Thermal tanks remain part of electrical end use;
the ledger does not claim to trace heat into subsequent thermal services.

Each member's AC bus is mixed proportionally over the control interval. Each
battery is mixed proportionally and all its operations retain their order. A
linear balance resolves energy charged and discharged within the same interval,
including transfers between several storages. This is an accounting convention,
not a claim to identify electrons or the chronology inside a sampled interval.
No counter changes actions, physical SOC, degradation, rewards or prices.

PV generated after reset is the current generation cohort. Grid energy carries
its acquisition carbon mass. Initial SOC and imposed arrival SOC have unknown
external origin and unknown carbon, even if historically renewable. Producer
identity survives local sharing and storage. The export manifest contains sparse
source/producer/recipient/destination totals; no full producer-by-member time
series is retained.

With financial settlement enabled, the ledger follows its actual local energy
allocation, including a disabled local-trading setting. With financial settlement
disabled, it uses same-interval community netting: imports are allocated with
member weights and demand caps; exports are allocated proportionally. The
metadata states which convention applied. No financial transaction is invented.

## Metric suffixes

Prepend `district_energy_provenance_` or `building_energy_provenance_` to the
following suffixes. Energy is in kWh, carbon mass in kgCO2 and ratios are unitless.

| Suffix | Meaning |
|---|---|
| `pv_generated_kwh` | PV produced within the episode and active boundary. |
| `pv_load_direct_kwh` | PV delivered to electrical end use without passing through a battery. Includes direct community sharing. |
| `pv_load_after_storage_kwh` | PV delivered to electrical end use after stationary or EV storage. |
| `pv_load_kwh` | Sum of direct and after-storage useful building end use. |
| `pv_mobility_kwh` | Known PV inventory leaving in a departing connected vehicle. This is delivery to mobility, not measured trip consumption. |
| `pv_grid_export_kwh` | PV reaching the external grid, including delayed battery or V2G export. |
| `pv_loss_kwh` | PV lost in battery/charger conversion and standby. |
| `pv_stored_kwh` | PV remaining inside connected/active storage. Not final consumption. |
| `pv_asset_exit_kwh` | Inventory leaving with a removed stationary asset/member. |
| `pv_external_reset_kwh` | Old inventory retired when an external SOC assignment replaces its provenance. |
| `unknown_input_kwh` | Initial inventory and imposed external SOC entering the accounting boundary. |
| `unknown_{destination}_kwh` | Energy with unknown origin at the corresponding destination. |
| `grid_{destination}_kwh` | Energy acquired from the grid at the corresponding destination. |
| `external_import_kwh` | External acquisition after the declared sharing/netting convention. |
| `external_purchase_emissions_kgco2` | External imports multiplied by the recipient's contemporaneous carbon intensity. No export credit. |
| `carbon_{destination}_kgco2` | Known acquisition carbon carried to load, mobility, export, loss, stored stock, asset exit or external reset. |
| `carbon_unknown_{destination}_kwh` | Energy whose acquisition carbon is unknown; it is not claimed to have zero carbon. |
| `local_import_kwh`, `local_export_kwh` | Internal transfers, also split by `pv_` and `grid_` prefixes. These are not additional final destinations. |

Destination substitutions are `load`, `mobility`, `grid_export`, `loss`, `stored`,
`asset_exit` and `external_reset`. Recipient-level PV use can exceed that member's
own generation because energy can be received from neighbours. Ratios and cohort
closure are therefore published at district level:

- `pv_useful_building_ratio = pv_load_kwh / pv_generated_kwh`.
- `pv_useful_with_mobility_ratio = (pv_load_kwh + pv_mobility_kwh) / pv_generated_kwh`.
- `pv_balance_residual_kwh`: generation minus all seven final destinations above.
- `grid_balance_residual_kwh` and `unknown_balance_residual_kwh`: acquisition or
  unknown input minus the corresponding destinations.
- `carbon_balance_residual_kgco2`: external acquisition emissions minus known
  carbon at all destinations.
- `storage_max_residual_kwh` and `bus_max_residual_kwh`: largest reconciliation
  discrepancy observed during accounting.

A zero PV denominator produces an undefined ratio (`None`/NaN), not an artificial
zero or 100%. Stored energy, losses and asset exits are excluded from useful-use
ratios. Carbon acquisition and carbon use answer different questions; **do not
add them together**. Unknown-carbon energy is explicitly reported separately.

BAU values append `business_as_usual` before the unit. Differences append
`delta_to_business_as_usual`. The sidecar uses the same ledger, boundary,
configuration, timestep and definition version as the controlled simulation.

## Compatibility

Existing PV self-consumption remains **instantaneous absorption**, which includes
energy entering storage. It does not prove final useful consumption. Existing
emissions remain the sum of positive member imports before sharing. These old
names and values were not silently replaced. Recompute using the new family to
answer provenance or external-boundary questions. Historical aggregated exports
alone cannot reconstruct storage origins; rerun those experiments.

For example, 1 kWh PV charged and later exported gives 1 kWh of
`pv_grid_export_kwh`, zero useful PV consumption and zero PV stock at the end.
An external purchase of 1 kWh at 0.2 kgCO2/kWh gives 0.2 kgCO2 even if gross member
imports before sharing total 3.5 kWh.

## EV feasibility

Departure feasibility now replays the concrete battery implementation in
isolated, session-sized buffers. It respects asymmetric power, maximum SOC,
charger AC losses, battery curves, capacity at the beginning of the session,
service headroom, deadbands and standby losses. A blocked interval does not end
the search. The final connected SOC is evaluated rather than an earlier peak.
The query leaves actual battery arrays, degradation, RNG and episode time intact.

This is an individual maximum-charging replay under the declared service and
base-load constraints. It is not a joint schedule for all vehicles, nor a proof
of global optimality for arbitrary custom non-monotone efficiency models. Legacy
adapters without a native physical battery retain a compatibility approximation;
missing physical/arrival data retains the legacy permissive classification.

## Reproducibility and limitations

Each KPI export writes a sibling `.manifest.json` containing the effective schema,
market settings, seed, episode window, timestep, definitions, package version,
Git revision/dirty state when available, SHA-256 hashes of referenced input files,
accounting totals and source/destination evidence. A wheel outside a Git checkout
reports the commit as unavailable rather than borrowing an unrelated revision.

Provenance does not add electrochemical battery temperature, detailed road-trip
energy, distribution power flow or field validation. Vehicle returns with imposed
SOC start a new unknown cohort. Escalators remain unsupported in the entity
interface; DR and dynamic topology require entity. Robustness catalogues targeting
entity features are not automatically translated into flat observations. Future-data access by some
reference policies and controller-quality comparisons are separate work.

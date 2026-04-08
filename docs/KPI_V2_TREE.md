# KPI v2 Naming Tree

## Naming Contract

All KPI names in `evaluate_v2()` follow:

`level_family_subfamily_metric_variant_unit`

- `level`: `building` or `district`
- `family`: `cost`, `energy_grid`, `emissions`, `solar_self_consumption`, `ev`, `battery`, `electrical_service_phase`, `equity`, `comfort_resilience`
- `subfamily`: e.g. `total`, `daily_average`, `ratio_to_baseline`, `shape_quality`, `community_market`, `events`, `performance`, `health`, `violations`, `imbalance`, `phase_peaks`, `benefit`, `distribution`, `discomfort`, `resilience`
- `variant`: optional (e.g. `control`, `baseline`, `delta`, `min`, `max`, `average`, `total`, `daily_average`)
- `unit`: optional and always at the end (e.g. `eur`, `kwh`, `kgco2`, `kw`, `count`, `percent`, `ratio`, `c`)

Examples:
- `district_cost_total_control_eur`
- `district_cost_daily_average_delta_eur`
- `district_cost_ratio_to_baseline_total_ratio`
- `building_energy_grid_total_import_control_kwh`
- `district_energy_grid_shape_quality_ramping_average_to_baseline_ratio`
- `district_energy_grid_community_market_local_traded_total_kwh`
- `district_solar_self_consumption_total_generation_kwh`
- `district_solar_self_consumption_community_market_import_share_ratio`

---

## Core Equations

- `total_import_kwh = Σ_t max(net_t, 0)`
- `total_export_kwh = Σ_t max(-net_t, 0)`
- `total_net_exchange_kwh = Σ_t net_t`
- `daily_average_x = total_x / simulated_days`
- `delta_x = control_x - baseline_x`
- `ratio_to_baseline_x = control_x / baseline_x` (safe division)
- `solar_self_consumption = (generation_total - export_total) / generation_total`

---

## Community (District KPIs)

District KPIs are all `district_*`.

### `cost`
- `district_cost_total_control_eur`
- `district_cost_total_baseline_eur`
- `district_cost_total_delta_eur`
- `district_cost_daily_average_control_eur`
- `district_cost_daily_average_baseline_eur`
- `district_cost_daily_average_delta_eur`
- `district_cost_ratio_to_baseline_total_ratio`

### `energy_grid`
- `district_energy_grid_total_import_control_kwh`
- `district_energy_grid_total_import_baseline_kwh`
- `district_energy_grid_total_import_delta_kwh`
- `district_energy_grid_daily_average_import_control_kwh`
- `district_energy_grid_daily_average_import_baseline_kwh`
- `district_energy_grid_daily_average_import_delta_kwh`
- `district_energy_grid_ratio_to_baseline_import_total_ratio`

- `district_energy_grid_total_export_control_kwh`
- `district_energy_grid_total_export_baseline_kwh`
- `district_energy_grid_total_export_delta_kwh`
- `district_energy_grid_daily_average_export_control_kwh`
- `district_energy_grid_daily_average_export_baseline_kwh`
- `district_energy_grid_daily_average_export_delta_kwh`
- `district_energy_grid_ratio_to_baseline_export_total_ratio`

- `district_energy_grid_total_net_exchange_control_kwh`
- `district_energy_grid_total_net_exchange_baseline_kwh`
- `district_energy_grid_total_net_exchange_delta_kwh`
- `district_energy_grid_daily_average_net_exchange_control_kwh`
- `district_energy_grid_daily_average_net_exchange_baseline_kwh`
- `district_energy_grid_daily_average_net_exchange_delta_kwh`
- `district_energy_grid_ratio_to_baseline_net_exchange_total_ratio`

Shape/quality:
- `district_energy_grid_shape_quality_ramping_average_to_baseline_ratio`
- `district_energy_grid_shape_quality_load_factor_penalty_daily_average_to_baseline_ratio`
- `district_energy_grid_shape_quality_load_factor_penalty_monthly_average_to_baseline_ratio`
- `district_energy_grid_shape_quality_peak_daily_average_to_baseline_ratio`
- `district_energy_grid_shape_quality_peak_all_time_average_to_baseline_ratio`

Community market (conditional):
- `district_energy_grid_community_market_local_traded_total_kwh`
- `district_energy_grid_community_market_local_traded_daily_average_kwh`

### `emissions`
- `district_emissions_total_control_kgco2`
- `district_emissions_total_baseline_kgco2`
- `district_emissions_total_delta_kgco2`
- `district_emissions_daily_average_control_kgco2`
- `district_emissions_daily_average_baseline_kgco2`
- `district_emissions_daily_average_delta_kgco2`
- `district_emissions_ratio_to_baseline_total_ratio`

### `solar_self_consumption`
- `district_solar_self_consumption_total_generation_kwh`
- `district_solar_self_consumption_total_export_kwh`
- `district_solar_self_consumption_daily_average_generation_kwh`
- `district_solar_self_consumption_daily_average_export_kwh`
- `district_solar_self_consumption_ratio_self_consumption_ratio`

Community market (conditional):
- `district_solar_self_consumption_community_market_import_share_ratio`

### `ev`
- `district_ev_events_departure_count`
- `district_ev_events_departure_met_count`
- `district_ev_events_departure_within_tolerance_count`
- `district_ev_performance_departure_success_ratio`
- `district_ev_performance_departure_within_tolerance_ratio`
- `district_ev_performance_departure_soc_deficit_mean_ratio`
- `district_ev_total_charge_kwh`
- `district_ev_total_v2g_export_kwh`

### `battery`
- `district_battery_total_charge_kwh`
- `district_battery_total_discharge_kwh`
- `district_battery_total_throughput_kwh`
- `district_battery_health_equivalent_full_cycles_count`
- `district_battery_health_capacity_fade_ratio`

### `electrical_service_phase`
- `district_electrical_service_phase_violations_energy_total_kwh`
- `district_electrical_service_phase_violations_event_count`
- `district_electrical_service_phase_imbalance_phase_average_ratio`
- `district_electrical_service_phase_phase_peaks_import_peak_l1_kw`
- `district_electrical_service_phase_phase_peaks_import_peak_l2_kw`
- `district_electrical_service_phase_phase_peaks_import_peak_l3_kw`
- `district_electrical_service_phase_phase_peaks_export_peak_l1_kw`
- `district_electrical_service_phase_phase_peaks_export_peak_l2_kw`
- `district_electrical_service_phase_phase_peaks_export_peak_l3_kw`

### `equity`
- `district_equity_distribution_gini_benefit_ratio`
- `district_equity_distribution_top20_benefit_ratio`
- `district_equity_distribution_losers_percent`
- `district_equity_distribution_bpr_asset_poor_over_rich_ratio`

### `comfort_resilience`
- `district_comfort_resilience_discomfort_overall_ratio`
- `district_comfort_resilience_discomfort_cold_ratio`
- `district_comfort_resilience_discomfort_hot_ratio`
- `district_comfort_resilience_discomfort_cold_delta_min_c`
- `district_comfort_resilience_discomfort_cold_delta_max_c`
- `district_comfort_resilience_discomfort_cold_delta_average_c`
- `district_comfort_resilience_discomfort_hot_delta_min_c`
- `district_comfort_resilience_discomfort_hot_delta_max_c`
- `district_comfort_resilience_discomfort_hot_delta_average_c`
- `district_comfort_resilience_resilience_one_minus_thermal_ratio`
- `district_comfort_resilience_resilience_unserved_energy_outage_normalized_ratio`
- `district_comfort_resilience_resilience_unserved_energy_annual_normalized_ratio`

---

## B1 (Single Building)

Building KPIs are all `building_*` and have the same family structure as district, except district-only community indicators:
- no `building_energy_grid_community_market_local_traded_*`
- no `building_solar_self_consumption_community_market_import_share_ratio`

Main pattern examples:
- `building_cost_total_control_eur`
- `building_energy_grid_total_import_control_kwh`
- `building_solar_self_consumption_total_generation_kwh`
- `building_equity_benefit_relative_percent`
- `building_ev_events_departure_within_tolerance_count`
- `building_ev_performance_departure_within_tolerance_ratio`

---

## Bn (All Buildings)

`Bn` uses the same KPI names as B1; building identity is in the `name` column (`Building_1`, ..., `Building_n`).

---

## Difference: Normal Self-Consumption vs Community Market Import Share

These are different KPIs and should not be merged.

### 1) Normal solar self-consumption
- KPI: `*_solar_self_consumption_ratio_self_consumption_ratio`
- Formula: `(generation_total - export_total) / generation_total`
- Meaning: fraction of PV generation consumed locally (solar-centric KPI)
- Availability: independent of community market (exists with market ON or OFF)

### 2) Community market import share
- KPI: `district_solar_self_consumption_community_market_import_share_ratio`
- Formula: `community_local_traded_total / district_energy_grid_total_import_control`
- Meaning: fraction of district import demand covered by local community trading (market-centric KPI)
- Availability: district-only, only when:
  - `community_market.enabled = true`
  - `community_market.kpis.community_self_consumption_enabled = true`

---

## EV Tolerance KPI (±5%)

- KPI (count): `*_ev_events_departure_within_tolerance_count`
- KPI (ratio): `*_ev_performance_departure_within_tolerance_ratio`
- Condition per departure event:
  - `abs(soc_departure - soc_target_departure) <= 0.05`
- Meaning:
  - counts/ratio of departures where SOC is within ±5 percentage points of target.
- Relation with strict success:
  - `departure_within_tolerance_count <= departure_count`
  - `departure_met_count` and `departure_within_tolerance_count` are not ordered in general.

---

## Important interpretation note

`district_energy_grid_total_import_*` is not necessarily equal to the sum of `building_energy_grid_total_import_*`.

- District is computed on aggregated net at each timestep:
  - `Σ_t max(Σ_b net_b,t, 0)`
- Building sum is computed per-building before aggregation:
  - `Σ_t Σ_b max(net_b,t, 0)`

This difference is expected and represents simultaneous import/export compensation across buildings.

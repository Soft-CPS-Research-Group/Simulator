# KPIs Reference

The simulator exposes two KPI APIs.

Portuguese version: [pt/kpis_reference.md](pt/kpis_reference.md).

| API | Status | Format | Recommended use |
|---|---|---|---|
| `evaluate()` | legacy | DataFrame with historical cost-function names | Compatibility with older workflows. |
| `evaluate_v2()` | primary | DataFrame with structured KPI names | Production, dashboards and scientific comparison. |

For the full v2 naming tree, see [KPI_V2_TREE.md](KPI_V2_TREE.md).

## Units

| Unit suffix | Meaning |
|---|---|
| `eur` | Monetary cost. If the dataset uses another currency, interpret as dataset currency. |
| `kwh` | Accumulated energy. |
| `kgco2` | Accumulated emissions. |
| `kw` | Power. |
| `ratio` | Dimensionless ratio. |
| `percent` | Percentage. |
| `count` | Event/cycle count. |
| `hours` | Hours. |
| `c` | Celsius. |

## KPI v2 Naming Contract

```text
level_family_subfamily_metric_variant_unit
```

| Part | Examples |
|---|---|
| `level` | `building`, `district`. |
| `family` | `cost`, `energy_grid`, `emissions`, `solar_self_consumption`, `ev`, `battery`, `electrical_service_phase`, `equity`, `comfort_resilience`, `deferrable_appliance`. |
| `subfamily` | `total`, `daily_average`, `ratio_to_baseline`, `shape_quality`, `service`. |
| `metric` | `import`, `export`, `charge`, `completed_cycles`. |
| `variant` | `control`, `baseline`, `delta`, `total`, `average`, `l1`. |
| `unit` | `eur`, `kwh`, `ratio`, etc. |

Examples:

| KPI | Meaning |
|---|---|
| `district_cost_total_control_eur` | Total control cost. |
| `building_energy_grid_total_import_control_kwh` | Building import energy. |
| `district_energy_grid_shape_quality_ramping_average_to_baseline_ratio` | Ramping relative to baseline. |
| `district_solar_self_consumption_ratio_self_consumption_ratio` | District solar self-consumption ratio. |
| `building_deferrable_appliance_service_completed_cycles_count` | Completed deferrable cycles. |

## Core Equations

For net energy `net_t`, where import is positive and export is negative:

```text
total_import_kwh = sum(max(net_t, 0))
total_export_kwh = sum(max(-net_t, 0))
total_net_exchange_kwh = sum(net_t)
daily_average_x = total_x / simulated_days
delta_x = control_x - baseline_x
ratio_to_baseline_x = control_x / baseline_x
self_consumption = (solar_generation_total - export_total) / solar_generation_total
```

Safe division returns `None` or a safe placeholder when the denominator is not physically meaningful.

## KPI Families

| Family | Level | Measures |
|---|---|---|
| `cost` | building, district | Cost control/baseline/delta/ratio. |
| `energy_grid` | building, district | Import, export, net exchange and shape quality. |
| `emissions` | building, district | Emissions control/baseline/delta/ratio. |
| `solar_self_consumption` | building, district | Generation, export and self-consumption. |
| `ev` | building, district | Departures, success rate, deficits, charge and V2G. |
| `battery` | building, district | Charge, discharge, throughput, cycles, capacity fade. |
| `electrical_service_phase` | building, district | Violations, imbalance and phase peaks. |
| `equity` | building, district | Relative benefits and benefit distribution. |
| `comfort_resilience` | building, district | Discomfort and outage/resilience events. |
| `deferrable_appliance` | building, district | Completed/missed cycles, service level and served energy. |

## EV KPIs

| Concept | Unit | Meaning |
|---|---:|---|
| departure count | count | Number of EV departures observed. |
| departure met count | count | Departures where required SOC was met. |
| departure within tolerance count | count | Departures within configured tolerance. |
| departure success ratio | ratio | `met / departures`. |
| departure SOC deficit mean | ratio | Mean SOC deficit on failures. |
| charge total | kWh | Energy charged into EVs. |
| V2G export total | kWh | Energy exported by EVs. |

## BESS KPIs

| Concept | Unit | Meaning |
|---|---:|---|
| charge total | kWh | Energy charged into BESS. |
| discharge total | kWh | Energy discharged. |
| throughput total | kWh | Absolute charge plus discharge. |
| equivalent full cycles | count | Throughput relative to capacity. |
| capacity fade | ratio | Relative capacity degradation. |

## Deferrable Appliance KPIs

| Concept | Unit | Meaning |
|---|---:|---|
| completed cycles | count | Cycles completed successfully. |
| missed cycles | count | Cycles that missed the start window. |
| service level | ratio | `completed / (completed + missed)`. |
| served energy | kWh | Energy of completed cycles. |
| unserved energy | kWh | Energy of missed cycles. |
| average start delay | hours | Mean `start - earliest_start`. |

## Electrical Service and Phase KPIs

| Concept | Unit | Meaning |
|---|---:|---|
| violation total | kWh | Energy above service/phase limits. |
| violation timestep count | count | Number of timesteps with a violation. |
| phase imbalance average | ratio | Average phase imbalance. |
| phase import/export peak L1/L2/L3 | kW | Import/export peaks per phase. |

## Community Market KPIs

Available when `community_market.enabled=true`.

| Concept | Level | Unit | Meaning |
|---|---|---:|---|
| local traded total | district | kWh | Locally traded energy. |
| local traded daily average | district | kWh/day | Daily average local trade. |
| community import share | district | ratio | Share of demand served locally. |
| settled cost | building/district | eur | Cost after local settlement. |
| counterfactual cost | building/district | eur | Cost without local market settlement. |
| market savings | building/district | eur | Counterfactual minus settled cost. |

## Export

| Mode | How |
|---|---|
| In memory | `env.evaluate_v2()` |
| End-of-episode export | `CityLearnEnv(..., export_kpis_on_episode_end=True)` |
| With render | `render_mode="end"` or `render_mode="during"` |

New production KPIs should be registered in [releases.md](releases.md) and, when part of the v2 naming contract, in [KPI_V2_TREE.md](KPI_V2_TREE.md).

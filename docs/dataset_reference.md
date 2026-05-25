# Dataset Reference

This page explains how to build simulator-compatible datasets, including CSV, Parquet, real power data in kW, EVs, absolute PV and normalized deferrable appliances.

Portuguese version: [pt/dataset_reference.md](pt/dataset_reference.md).

## Supported File Formats

| Format | Extensions | Loader | When to use |
|---|---|---|---|
| CSV | `.csv` | `pandas.read_csv` with `skiprows/nrows` for windows | Small/medium datasets and easy inspection. |
| Parquet | `.parquet`, `.pq`, `.parq` | `pyarrow.parquet` batches for windows | Large datasets, 15s data, many assets or full-year runs. |

CSV and Parquet are interchangeable when the schema path is updated and columns, units and types remain equivalent.

## Unit Contract

| Data | Expected unit |
|---|---:|
| Load and consumption series | kWh/step |
| PV with `generation_mode="absolute"` | kWh/step |
| PV with `generation_mode="per_kw"` | W/kW |
| BESS and charger power limits | kW |
| EV required/estimated SOC in charger files | percent in the raw file, converted to ratio internally |
| Prices | currency/kWh |
| Carbon intensity | kgCO2/kWh |
| Weather temperature | C |
| Irradiance | W/m2 |

Real power data must be converted before writing energy columns:

```text
kWh_per_step = kW * seconds_per_time_step / 3600
kW = kWh_per_step * 3600 / seconds_per_time_step
```

For 15 seconds:

```text
1 kW during 15s = 1 * 15 / 3600 = 0.0041666667 kWh
```

## `energy_simulation` Columns

| Column | Required | Unit | Meaning |
|---|---:|---:|---|
| `month` | yes | 1-12 | Month. |
| `hour` | yes | 1-24 | Hour. |
| `day_type` | yes | 1-8 | Day of week or special day. |
| `minutes` | recommended sub-hour | 0-59 | Minute. |
| `seconds` | recommended sub-minute | 0-59 | Second. |
| `indoor_dry_bulb_temperature` | yes | C | Indoor temperature. |
| `non_shiftable_load` | yes | kWh/step | Non-flexible load. |
| `dhw_demand` | yes | kWh/step | Domestic hot water demand. |
| `cooling_demand` | yes | kWh/step | Cooling demand. |
| `heating_demand` | yes | kWh/step | Heating demand. |
| `solar_generation` | yes | depends on PV mode | PV input. |
| `daylight_savings_status` | no | 0/1 | DST. |
| `average_unmet_cooling_setpoint_difference` | no | C | Cooling discomfort. |
| `indoor_relative_humidity` | no | percent | Indoor humidity. |
| `occupant_count` | no | people | Occupancy. |
| `indoor_dry_bulb_temperature_cooling_set_point` | no | C | Cooling setpoint. |
| `indoor_dry_bulb_temperature_heating_set_point` | no | C | Heating setpoint. |
| `hvac_mode` | no | enum | 0 off, 1 cooling, 2 heating, 3 auto. |
| `power_outage` | no | 0/1 | Outage flag. |
| `comfort_band` | no | C | Comfort band. |

Cooling and heating demand cannot both be positive in the same timestep.

## Weather, Pricing and Carbon Files

| File | Required columns | Units |
|---|---|---|
| `weather` | `outdoor_dry_bulb_temperature`, `outdoor_relative_humidity`, `diffuse_solar_irradiance`, `direct_solar_irradiance`, plus `*_predicted_1/2/3` | C, percent, W/m2 |
| `pricing` | `electricity_pricing`, `electricity_pricing_predicted_1/2/3` | currency/kWh |
| `carbon_intensity` | `carbon_intensity` | kgCO2/kWh |

## Charger Simulation Columns

| Column | Unit/format | Meaning |
|---|---:|---|
| `electric_vehicle_charger_state` | enum | 1 connected, 2 incoming, 3 away/commuting. |
| `electric_vehicle_id` | string | EV ID. |
| `electric_vehicle_departure_time` | steps | Steps until departure. Internal default `-1`. |
| `electric_vehicle_required_soc_departure` | percent | Required departure SOC. Converted to ratio. |
| `electric_vehicle_estimated_arrival_time` | steps | Steps until arrival. Internal default `-1`. |
| `electric_vehicle_estimated_soc_arrival` | percent | Estimated arrival SOC. Converted to ratio. |
| `electric_vehicle_current_soc` | percent or ratio | Optional measured/estimated current SOC. |

For sub-hourly datasets, countdown fields must be expressed in timesteps at the dataset resolution. Example: 1 hour at 15s is 240 steps.

## Entity Observation Bundles in Packaged Datasets

The packaged 15-second entity datasets and `citylearn_challenge_2022_phase_all_plus_evs` opt in to all entity observation bundles:

| Bundle | Purpose |
|---|---|
| `entity_core_electrical` | Physical power, energy, SOC and asset capability descriptors. |
| `entity_community_operational` | Community power, headroom and flexible capacity aggregates. |
| `entity_forecasts_existing` | Existing dataset `*_predicted_*` observations. |
| `entity_forecasts_derived` | Compact simulator-perfect point forecasts for price, load, PV and net demand. |
| `entity_temporal_derived` | Robust calendar and short lag features. |
| `entity_action_feedback` | Requested, limited and applied action feedback with clipping reasons. |

Other schemas keep the default-compatible behavior unless they declare `observation_bundles`.

## Deferrable Appliances

The official format is sparse: a cycle profile catalog plus a flexibility request schedule. Do not repeat the full `load_profile` at every timestep.

### `cycle_profiles_file`

| Column | Unit | Meaning |
|---|---:|---|
| `profile_id` | string | Profile ID. |
| `duration_steps` | steps | Cycle duration. |
| `total_energy_kwh` | kWh | Sum of the profile. |
| `load_profile` | list of kWh/step | Step energy profile. |

Validation:

| Check | Rule |
|---|---|
| `profile_id` | Non-empty and unique. |
| `duration_steps` | Integer > 0. |
| `load_profile` | Non-empty, finite and non-negative. |
| Sum | `sum(load_profile) == total_energy_kwh` within tolerance. |

### `flexibility_schedule_file`

| Column | Unit | Meaning |
|---|---:|---|
| `cycle_id` | string | Unique request/cycle ID. |
| `profile_id` | string | Reference to the profile catalog. |
| `earliest_start_time_step` | global timestep | First allowed start. |
| `latest_start_time_step` | global timestep | Last allowed start. |
| `deadline_time_step` | global timestep | Completion deadline. |
| `priority` | ratio | Priority 0-1, clipped. |
| `must_run` | bool | Mandatory request flag. |

Validation:

| Check | Rule |
|---|---|
| `cycle_id` | Unique and non-empty. |
| `profile_id` | Must exist in the catalog. |
| Windows | `earliest <= latest`. |
| Deadline | `latest + duration_steps - 1 <= deadline`. |
| Timesteps | Global integer indices >= 0. |

## PV Datasets

| Case | Schema | `solar_generation` column |
|---|---|---|
| Real/measured generation | `"generation_mode": "absolute"` | `kWh/step`. |
| Normalized profile | `"generation_mode": "per_kw"` | `W/kW` per 1 kW installed. |

For real production datasets, `absolute` is the recommended mode.

## 15-Second Datasets

| Topic | Recommendation |
|---|---|
| `seconds_per_time_step` | `15`. |
| Real power data | Convert kW to kWh/step before writing dataset energy columns. |
| Real PV | Write kWh/step and use `generation_mode="absolute"`. |
| EV countdowns | Express `*_departure_time` and `*_arrival_time` in 15s steps. |
| Large files | Prefer Parquet. |
| Pre-training smoke test | Run a short window and call `evaluate_v2()`. |
| Compact dynamic-assets example | `data/datasets/citylearn_three_phase_dynamic_asset_changes_demo_15s_parquet/schema.json` contains 7 days at 15s with charger, PV and BESS add/remove events. |

## Performance and Loader Behavior

| Optimization | Effect |
|---|---|
| Windowed CSV | Uses `skiprows/nrows` to load only the requested window. |
| Windowed Parquet | Reads batches and slices rows. |
| Shared cache | Reuses weather/pricing/carbon when several buildings point to the same file and `noise_std=0`. |
| Parquet | Smaller files, typed reads and better large-dataset behavior. |

## New Dataset Checklist

1. Define `seconds_per_time_step`.
2. Convert measured power to `kWh/step` where the simulator expects energy.
3. Choose PV `absolute` or `per_kw`.
4. Ensure EV schedules use countdowns in dataset timesteps.
5. Write deferrables as catalog plus schedule.
6. Prefer Parquet for annual or sub-minute datasets.
7. Run a smoke episode and `evaluate_v2()`.
8. Run `audit_physics.py` for new critical datasets.

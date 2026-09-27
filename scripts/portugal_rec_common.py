"""Calendar, serialization and schema primitives for the Portugal REC suite.

Physical populations, mobility and topology are defined by
``generate_portugal_rec_suite.py``; there is no previous-suite generator here.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple
import hashlib
import json
import numpy as np
import pandas as pd

YEAR = 2023


SECONDS_PER_TIME_STEP = 900


N_STEPS = 35_040


TIMEZONE = "Europe/Lisbon"


DATASET_CONTRACT_VERSION = "2023-pt-q15-v2.0-candidate"


SEED = 2023


@dataclass(frozen=True)
class FamilyConfig:
    key: str
    directory: str
    label: str
    member_categories: Tuple[str, ...]
    pv_members: Tuple[int, ...]
    battery_members: Tuple[int, ...]
    charger_counts: Mapping[int, int]
    deferrable_members: Tuple[int, ...]
    variants: Tuple[str, ...]
    default_variant: str


OBSERVATIONS = {
    name: {"active": active, "shared_in_central_agent": shared}
    for name, active, shared in (
        ("month", True, True),
        ("day_type", True, True),
        ("hour", True, True),
        ("minutes", True, True),
        ("daylight_savings_status", True, True),
        ("outdoor_dry_bulb_temperature", True, True),
        ("outdoor_dry_bulb_temperature_predicted_1", True, True),
        ("outdoor_dry_bulb_temperature_predicted_2", True, True),
        ("outdoor_dry_bulb_temperature_predicted_3", True, True),
        ("outdoor_relative_humidity", True, True),
        ("diffuse_solar_irradiance", True, True),
        ("direct_solar_irradiance", True, True),
        ("carbon_intensity", True, True),
        ("non_shiftable_load", True, False),
        ("solar_generation", True, False),
        ("electrical_storage_soc", True, False),
        ("net_electricity_consumption", True, False),
        ("electricity_pricing", True, False),
        ("electricity_pricing_predicted_1", True, False),
        ("electricity_pricing_predicted_2", True, False),
        ("electricity_pricing_predicted_3", True, False),
        ("power_outage", True, False),
        ("electric_vehicle_charger_connected_state", True, False),
        ("connected_electric_vehicle_at_charger_battery_capacity", True, False),
        ("connected_electric_vehicle_at_charger_departure_time", True, False),
        ("connected_electric_vehicle_at_charger_required_soc_departure", True, False),
        ("connected_electric_vehicle_at_charger_soc", True, False),
        ("electric_vehicle_charger_incoming_state", True, False),
        ("incoming_electric_vehicle_at_charger_estimated_arrival_time", True, False),
        ("deferrable_appliance_pending", True, False),
        ("deferrable_appliance_running", True, False),
        ("deferrable_appliance_can_start", True, False),
        ("deferrable_appliance_deadline_missed", True, False),
        ("deferrable_appliance_earliest_start_time_step", True, False),
        ("deferrable_appliance_latest_start_time_step", True, False),
        ("deferrable_appliance_deadline_time_step", True, False),
        ("deferrable_appliance_hours_until_latest_start", True, False),
        ("deferrable_appliance_hours_until_deadline", True, False),
        ("deferrable_appliance_slack_steps", True, False),
        ("deferrable_appliance_slack_ratio", True, False),
        ("deferrable_appliance_urgency_ratio", True, False),
        ("deferrable_appliance_cycle_duration_steps", True, False),
        ("deferrable_appliance_cycle_energy_kwh", True, False),
        ("deferrable_appliance_remaining_energy_kwh", True, False),
        ("deferrable_appliance_current_step_energy_kwh", True, False),
        ("deferrable_appliance_priority", True, False),
    )
}


def _member_id(index: int) -> str:
    return f"Member_{index:03d}"


def _charger_id(member_index: int, charger_index: int) -> str:
    return f"CHG-M{member_index:03d}-{charger_index:02d}"


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_file_checksums(root: Path) -> None:
    checksum_path = root / "file_checksums.sha256"
    excluded = {"file_checksums.sha256", "validation_report.json"}
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append(f"{digest.hexdigest()}  {relative}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def build_calendar() -> pd.DataFrame:
    utc = pd.date_range(f"{YEAR}-01-01", periods=N_STEPS, freq="15min", tz="UTC")
    local = utc.tz_convert(TIMEZONE)
    offsets = np.array(
        [int(ts.utcoffset().total_seconds() // 60) for ts in local], dtype="int16"
    )
    return pd.DataFrame(
        {
            "time_step": np.arange(N_STEPS, dtype="int32"),
            "timestamp_utc": utc,
            "timestamp_local": local.astype(str),
            "timezone": TIMEZONE,
            "utc_offset_minutes": offsets,
            "month_local": local.month.astype("int8"),
            "day_local": local.day.astype("int8"),
            "day_of_week_local": (local.dayofweek + 1).astype("int8"),
            "hour_local": local.hour.astype("int8"),
            "minute_local": local.minute.astype("int8"),
            "daylight_savings_status": (offsets != 0).astype("int8"),
        }
    )


def _periodic_gaussian(
    hour: np.ndarray,
    centre: float | np.ndarray,
    width: float,
) -> np.ndarray:
    delta = np.abs(hour - centre)
    delta = np.minimum(delta, 24.0 - delta)
    return np.exp(-0.5 * (delta / width) ** 2)


def _local_at(
    day: pd.Timestamp, minute_of_day: int, day_offset: int = 0
) -> pd.Timestamp:
    """Construct a civil-time instant without elapsed-time DST drift.

    Adding a duration to a timezone-aware midnight shifts an intended wall
    clock routine by one hour after either Lisbon DST transition.  Build the
    naive local clock first and localize it instead.  The missing spring hour
    is moved forward by exactly one hour while the first occurrence of the
    repeated autumn hour is selected deterministically.
    """

    target_day = day.date() + timedelta(days=day_offset)
    local_clock = pd.Timestamp(target_day) + pd.Timedelta(minutes=int(minute_of_day))
    return local_clock.tz_localize(
        TIMEZONE,
        ambiguous=True,
        nonexistent=pd.Timedelta(hours=1),
    )


def _electrical_service(contract: Mapping) -> Mapping:
    return {
        "mode": contract["simulator_mode"],
        "default_split": contract["default_split"],
        "connection_type": contract["connection_type"],
        "contracted_power_kva": contract["contracted_power_kva"],
        "original_sampled_contracted_power_kva": contract[
            "original_sampled_contracted_power_kva"
        ],
        "native_peak_import_kw": contract["native_peak_import_kw"],
        "native_peak_export_kw": contract["native_peak_export_kw"],
        "native_feasibility_margin": contract["native_feasibility_margin"],
        "contract_level_upgraded_for_native_feasibility": contract[
            "contract_level_upgraded_for_native_feasibility"
        ],
        "active_power_factor_assumption": contract["active_power_factor_assumption"],
        "limits": {
            "total": {
                "import_kw": contract["total_import_limit_kw"],
                "export_kw": contract["total_export_limit_kw"],
            },
            "per_phase": {
                phase: {
                    "import_kw": contract["per_phase_import_limits_kw"][phase],
                    "export_kw": contract["per_phase_export_limits_kw"][phase],
                }
                for phase in ("L1", "L2", "L3")
            },
        },
        "observations": {
            "headroom": True,
            "headroom_export": True,
            "violation": True,
            "phase_encoding": True,
        },
    }


def _robustness_events(family: str) -> pd.DataFrame:
    unavailable_member = 6 if family == "core30" else 25
    unavailable_charger = 2
    common = [
        {
            "event_id": "obs_missing_member_004",
            "module": "observation",
            "target_type": "building",
            "target_id": _member_id(4),
            "target_feature": "non_shiftable_load",
            "start_time_step": 7600,
            "end_time_step": 7631,
            "mode": "missing",
            "replacement_value": -9999.0,
        },
        {
            "event_id": "forecast_price_bias",
            "module": "forecast",
            "target_type": "district",
            "target_id": "*",
            "target_feature": "electricity_pricing_predicted_1",
            "start_time_step": 15100,
            "end_time_step": 15195,
            "mode": "bias",
            "value": 0.025,
        },
        {
            "event_id": "storage_action_delay",
            "module": "action",
            "target_type": "storage",
            "target_id": "*",
            "target_feature": "electrical_storage",
            "start_time_step": 20100,
            "end_time_step": 20147,
            "mode": "delay",
            "delay_steps": 4,
        },
        {
            "event_id": "charger_unavailable",
            "module": "asset",
            "target_type": "charger",
            "target_id": f"{_member_id(unavailable_member)}/{_charger_id(unavailable_member, unavailable_charger)}",
            "target_feature": "both",
            "start_time_step": 27000,
            "end_time_step": 27047,
            "mode": "unavailable",
            "replacement_value": -9999.0,
        },
    ]
    if family == "premium":
        common.extend(
            [
                {
                    "event_id": "coincident_pv_telemetry_loss",
                    "module": "observation",
                    "target_type": "pv",
                    "target_id": "*",
                    "target_feature": "generation_energy_kwh_step",
                    "start_time_step": 27016,
                    "end_time_step": 27031,
                    "mode": "missing",
                    "replacement_value": -9999.0,
                },
                {
                    "event_id": "coincident_charger_dropout",
                    "module": "action",
                    "target_type": "charger",
                    "target_id": "*",
                    "target_feature": "electric_vehicle_storage",
                    "start_time_step": 27016,
                    "end_time_step": 27031,
                    "mode": "dropout",
                },
                {
                    "event_id": "service_charger_outage",
                    "module": "asset",
                    "target_type": "charger",
                    "target_id": f"{_member_id(71)}/{_charger_id(71, 1)}",
                    "target_feature": "both",
                    "start_time_step": 27008,
                    "end_time_step": 27047,
                    "mode": "unavailable",
                    "replacement_value": -9999.0,
                },
                {
                    "event_id": "stationary_storage_outage",
                    "module": "asset",
                    "target_type": "storage",
                    "target_id": f"{_member_id(1)}/electrical_storage",
                    "target_feature": "both",
                    "start_time_step": 27016,
                    "end_time_step": 27047,
                    "mode": "unavailable",
                    "replacement_value": -9999.0,
                },
            ]
        )
    columns = [
        "event_id",
        "module",
        "target_type",
        "target_id",
        "target_feature",
        "start_time_step",
        "end_time_step",
        "mode",
        "value",
        "std",
        "min_value",
        "max_value",
        "replacement_value",
        "delay_steps",
    ]
    return pd.DataFrame(common).reindex(columns=columns)


def _dr_requests(family: str) -> pd.DataFrame:
    scale = 1.0 if family == "core30" else 3.0
    rows = [
        (
            "dr_winter_evening_down",
            "dso",
            "down",
            2500,
            2515,
            18.0 * scale,
            0.35,
            1.20,
            1.0,
        ),
        (
            "dr_spring_midday_up",
            "tso",
            "up",
            12500,
            12511,
            12.0 * scale,
            0.24,
            0.80,
            0.5,
        ),
        (
            "dr_summer_evening_down",
            "dso",
            "down",
            22500,
            22523,
            24.0 * scale,
            0.42,
            1.50,
            1.0,
        ),
        (
            "dr_winter_stress_down",
            "dso",
            "down",
            27016,
            27031,
            30.0 * scale,
            0.55,
            1.80,
            1.5,
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "request_id",
            "issuer",
            "direction",
            "start_time_step",
            "end_time_step",
            "target_power_kw",
            "activation_price_eur_per_kwh",
            "shortfall_penalty_eur_per_kwh",
            "tolerance_power_kw",
        ],
    )


def _variant_flags(variant: str) -> Mapping[str, bool]:
    return {
        "safety": variant
        in {
            "CORE-30-SAFETY",
            "CORE-30-COMBINED",
            "PREMIUM-100-CLEAN",
            "PREMIUM-100-ALLIN",
        },
        "health": variant
        in {"CORE-30-HEALTH", "CORE-30-COMBINED", "PREMIUM-100-ALLIN"},
        "dynamic": variant
        in {
            "CORE-30-DYNAMIC",
            "CORE-30-COMBINED",
            "PREMIUM-100-CLEAN",
            "PREMIUM-100-ALLIN",
        },
        "dr": variant in {"CORE-30-COMBINED", "PREMIUM-100-CLEAN", "PREMIUM-100-ALLIN"},
        "grid_outage": variant == "PREMIUM-100-ALLIN",
    }


def build_schema(
    config: FamilyConfig,
    variant: str,
    members: Sequence[Mapping],
    chargers: Sequence[Mapping],
    evs: Sequence[Mapping],
    deferrables: Sequence[Mapping],
    electrical_contracts: Mapping[str, Mapping],
) -> Mapping:
    flags = _variant_flags(variant)
    chargers_by_member: Dict[str, List[Mapping]] = {}
    for charger in chargers:
        chargers_by_member.setdefault(charger["member_id"], []).append(charger)
    deferrable_by_member = {item["member_id"]: item for item in deferrables}
    member_by_id = {item["member_id"]: item for item in members}

    buildings = {}
    for member in members:
        member_id = member["member_id"]
        member_index = int(member["member_index"])
        electrical_contract = electrical_contracts[member_id]
        building = {
            "include": True,
            "type": "citylearn.building.Building",
            "energy_simulation": f"timeseries/{member_id}.parquet",
            "weather": "shared/weather.parquet",
            "carbon_intensity": "shared/carbon_intensity.parquet",
            "pricing": "shared/pricing_btn_simple.parquet",
            "inactive_observations": [],
            "inactive_actions": [] if member["has_battery"] else ["electrical_storage"],
            "equity_group": member["equity_group"],
        }
        if member["has_pv"]:
            building["pv"] = {
                "type": "citylearn.energy_model.PV",
                "autosize": False,
                "attributes": {
                    "nominal_power": member["pv_capacity_kw"],
                    "generation_mode": "absolute",
                },
            }
        if member["has_battery"]:
            building["electrical_storage"] = {
                "type": "citylearn.energy_model.Battery",
                "autosize": False,
                "attributes": {
                    "capacity": member["battery_capacity_kwh"],
                    "nominal_power": member["battery_nominal_power_kw"],
                    "efficiency": member["battery_efficiency"],
                    "initial_soc": member["battery_initial_soc"],
                    "depth_of_discharge": member["battery_depth_of_discharge"],
                    "loss_coefficient": member["battery_loss_coefficient"],
                    "capacity_loss_coefficient": member[
                        "battery_capacity_loss_coefficient"
                    ],
                    "phase_connection": electrical_contract["asset_phase_connection"],
                },
            }
        member_chargers = chargers_by_member.get(member_id, [])
        if member_chargers:
            building["chargers"] = {}
            for item in member_chargers:
                building["chargers"][item["charger_id"]] = {
                    "type": "citylearn.electric_vehicle_charger.Charger",
                    "charger_simulation": f"chargers/{item['charger_id']}.parquet",
                    "autosize": False,
                    "noise_std": 0.0,
                    "attributes": {
                        "nominal_power": item["max_charging_power_kw"],
                        "efficiency": 0.95,
                        "charger_type": (
                            1 if item["max_discharging_power_kw"] > 0 else 0
                        ),
                        "max_charging_power": item["max_charging_power_kw"],
                        "min_charging_power": 1.4,
                        "max_discharging_power": item["max_discharging_power_kw"],
                        "min_discharging_power": 0.0,
                        "phase_connection": item["phase_connection"],
                    },
                }
        if member_id in deferrable_by_member:
            item = deferrable_by_member[member_id]
            building["deferrable_appliances"] = {
                item["deferrable_id"]: {
                    "type": "citylearn.energy_model.DeferrableAppliance",
                    "autosize": False,
                    "cycle_profiles_file": f"deferrables/{item['deferrable_id']}_profiles.parquet",
                    "flexibility_schedule_file": f"deferrables/{item['deferrable_id']}_schedule.parquet",
                    "attributes": {"trigger_threshold": 0.5},
                }
            }
        if flags["safety"]:
            building["electrical_service"] = _electrical_service(electrical_contract)
        if flags["grid_outage"]:
            building["power_outage"] = {
                "simulate_power_outage": member_index in {4, 25, 71, 91},
                "stochastic_power_outage": False,
            }
        buildings[member_id] = building

    ev_definitions = {}
    for ev in evs:
        ev_definitions[ev["ev_id"]] = {
            "type": "citylearn.electric_vehicle.ElectricVehicle",
            "include": True,
            "battery": {
                "type": "citylearn.energy_model.Battery",
                "autosize": False,
                "attributes": {
                    "capacity": ev["battery_capacity_kwh"],
                    "nominal_power": ev["nominal_power_kw"],
                    "initial_soc": ev["initial_soc"],
                    "depth_of_discharge": ev["depth_of_discharge"],
                    "loss_coefficient": 0.0,
                },
            },
        }

    schema = {
        "dataset_id": variant,
        "dataset_family": config.label,
        "dataset_contract_version": DATASET_CONTRACT_VERSION,
        "random_seed": SEED,
        "root_directory": None,
        "central_agent": False,
        "interface": "entity",
        "simulation_start_time_step": 0,
        "simulation_end_time_step": N_STEPS - 1,
        "episode_time_steps": None,
        "rolling_episode_split": False,
        "random_episode_split": False,
        "seconds_per_time_step": SECONDS_PER_TIME_STEP,
        "start_date": f"{YEAR}-01-01",
        "timezone": TIMEZONE,
        "observations": OBSERVATIONS,
        "actions": {
            "cooling_storage": {"active": False},
            "heating_storage": {"active": False},
            "dhw_storage": {"active": False},
            "electrical_storage": {"active": True},
            "electric_vehicle_storage": {"active": True},
            "deferrable_appliance": {"active": True},
        },
        "reward_function": {
            "type": "citylearn.reward_function.RewardFunction",
            "attributes": {},
        },
        "observation_bundles": {
            "entity_core_electrical": True,
            "entity_community_operational": True,
            "entity_forecasts_existing": True,
            "entity_forecasts_derived": True,
            "entity_demand_response": flags["dr"],
            "entity_robustness": flags["health"],
            "entity_action_feedback": True,
            "entity_temporal_derived": True,
        },
        "derived_forecasts": {
            "load_pv_method": "daily_persistence",
            "persistence_period_seconds": 86400,
            "cold_start": "current_step",
            "price_source": "publication_aware_day_ahead_market_input",
            "price_publication_time_local": "13:00",
            "price_unpublished_fallback": "daily_persistence",
            "price_horizon_steps": [4, 24, 96],
        },
        "community_market": {
            "enabled": True,
            "local_price_ratio_to_grid_import": 0.8,
            "intra_community_sell_ratio": 0.8,
            "grid_export_price": 0.0,
            "import_member_weights": {member_id: 1.0 for member_id in member_by_id},
            "counterfactual": "grid_only",
            "kpis": {
                "community_local_traded_enabled": True,
                "community_self_consumption_enabled": True,
            },
        },
        "electric_vehicles_def": ev_definitions,
        "buildings": buildings,
        "render_file_format": "parquet",
        "export_kpis_on_episode_end": False,
        "physics_invariant_checks": True,
    }
    # The generator sets physical membership and asset transitions explicitly.
    schema["topology_mode"] = "static"
    if flags["health"]:
        schema["robustness"] = {
            "enabled": True,
            "events_file": "events/robustness_events.parquet",
            "random_seed": SEED,
            "missing_replacement_value": -9999.0,
            "modules": {
                "observations": {"enabled": True},
                "forecasts": {"enabled": True},
                "actions": {"enabled": True},
                "assets": {"enabled": True},
            },
        }
    if flags["dr"]:
        schema["demand_response"] = {
            "enabled": True,
            "requests_file": "events/demand_response_requests.parquet",
            "baseline_method": "rolling_pre_event_average",
            "baseline_window_seconds": 3600,
            "allow_overlapping_requests": False,
        }
    return schema

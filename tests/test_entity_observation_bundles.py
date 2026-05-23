from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from citylearn.citylearn import CityLearnEnv


BASE_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
MINUTE_SCHEMA_PATH = Path(__file__).resolve().parent / "data/minute_ev_demo/schema.json"
PACKAGED_ALL_BUNDLE_SCHEMA_PATHS = [
    *sorted((Path(__file__).resolve().parents[1] / "data/datasets").glob("*15s*/schema.json")),
    BASE_SCHEMA_PATH,
]
ALL_ENTITY_BUNDLES = {
    "entity_core_electrical",
    "entity_community_operational",
    "entity_forecasts_existing",
    "entity_forecasts_derived",
    "entity_temporal_derived",
    "entity_action_feedback",
}


def _schema_with_bundles(
    *,
    core: bool = False,
    community: bool = False,
    forecast: bool = False,
    derived_forecast: bool = False,
    temporal: bool = False,
    action_feedback: bool = False,
) -> Mapping[str, Any]:
    schema = json.loads(BASE_SCHEMA_PATH.read_text())
    schema["root_directory"] = str(BASE_SCHEMA_PATH.parent)
    schema["interface"] = "entity"
    schema["topology_mode"] = "static"
    schema["observation_bundles"] = {
        "entity_core_electrical": {"active": core},
        "entity_community_operational": {"active": community},
        "entity_forecasts_existing": {"active": forecast},
        "entity_forecasts_derived": {"active": derived_forecast},
        "entity_temporal_derived": {"active": temporal},
        "entity_action_feedback": {"active": action_feedback},
    }
    return schema


def _minute_schema_with_bundles(tmp_path: Path, *, derived_forecast: bool = False, action_feedback: bool = False) -> Path:
    dataset_dir = tmp_path / "minute_ev_demo_entity"
    shutil.copytree(MINUTE_SCHEMA_PATH.parent, dataset_dir)
    schema_path = dataset_dir / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["root_directory"] = str(dataset_dir)
    schema["interface"] = "entity"
    schema["observation_bundles"] = {
        "entity_core_electrical": {"active": True},
        "entity_community_operational": {"active": True},
        "entity_forecasts_existing": {"active": False},
        "entity_forecasts_derived": {"active": derived_forecast},
        "entity_temporal_derived": {"active": True},
        "entity_action_feedback": {"active": action_feedback},
    }
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return schema_path


def _zero_entity_actions(env: CityLearnEnv):
    tables = env.action_space["tables"]
    return {
        "tables": {
            name: np.zeros(space.shape, dtype="float32")
            for name, space in tables.items()
            if name in {"building", "charger", "deferrable_appliance"}
        }
    }


def test_observation_bundles_default_to_disabled():
    schema = json.loads(BASE_SCHEMA_PATH.read_text())
    schema["root_directory"] = str(BASE_SCHEMA_PATH.parent)
    schema.pop("observation_bundles", None)
    env = CityLearnEnv(schema, interface="entity", central_agent=True, episode_time_steps=6, random_seed=0)

    try:
        env.reset(seed=0)
        bundles = env.entity_specs["observation_bundles"]
        assert bundles["entity_base"]["active"] is True
        assert bundles["entity_core_electrical"]["active"] is False
        assert bundles["entity_community_operational"]["active"] is False
        assert bundles["entity_forecasts_existing"]["active"] is False
        assert bundles["entity_forecasts_derived"]["active"] is False
        assert bundles["entity_temporal_derived"]["active"] is False
        assert bundles["entity_action_feedback"]["active"] is False
    finally:
        env.close()


def test_packaged_15s_and_ev_2022_schemas_enable_all_entity_bundles():
    for schema_path in PACKAGED_ALL_BUNDLE_SCHEMA_PATHS:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        bundles = schema.get("observation_bundles", {})
        assert set(bundles) >= ALL_ENTITY_BUNDLES, schema_path
        for bundle_name in ALL_ENTITY_BUNDLES:
            assert bundles[bundle_name]["active"] is True, (schema_path, bundle_name)


def test_core_bundle_adds_canonical_features_and_metadata():
    env = CityLearnEnv(_schema_with_bundles(core=True), interface="entity", central_agent=True, episode_time_steps=6, random_seed=0)

    try:
        env.reset(seed=0)
        building_features = env.entity_specs["tables"]["building"]["features"]
        assert "net_power_kw" in building_features
        assert "net_energy_kwh_step" in building_features
        assert "load_power_kw" in building_features
        assert "load_energy_kwh_step" in building_features

        building_meta = env.entity_specs["tables"]["building"]["feature_metadata"]["net_power_kw"]
        assert building_meta["unit"] == "kw"
        assert building_meta["bundle"] == "entity_core_electrical"
        assert building_meta["legacy"] is False

        assert "pv" in env.entity_specs["tables"]
        assert "building_to_pv" in env.entity_specs["edges"]
    finally:
        env.close()


def test_core_bundle_adds_ev_charging_flexibility_features():
    env = CityLearnEnv(_schema_with_bundles(core=True), interface="entity", central_agent=True, episode_time_steps=6, random_seed=0)

    try:
        obs, _ = env.reset(seed=0)
        charger_features = env.entity_specs["tables"]["charger"]["features"]
        charger_meta = env.entity_specs["tables"]["charger"]["feature_metadata"]
        expected_features = [
            "hours_until_departure",
            "time_until_departure_ratio",
            "energy_to_required_soc_kwh",
            "required_average_power_kw",
            "avg_power_to_departure_kw",
            "charging_slack_kw",
            "charging_priority_ratio",
            "connected_ev_soc_min_ratio",
            "connected_ev_energy_available_kwh",
            "connected_ev_energy_to_full_kwh",
            "can_charge",
            "can_discharge",
            "available_charge_power_kw",
            "available_discharge_power_kw",
            "available_charge_action_normalized",
            "available_discharge_action_normalized",
            "max_deliverable_energy_until_departure_kwh",
            "departure_energy_margin_kwh",
            "departure_feasibility_ratio",
            "min_required_action_normalized",
            "charge_efficiency_at_max_ratio",
            "discharge_efficiency_at_max_ratio",
            "incoming_ev_required_soc_departure",
            "incoming_ev_departure_time_step",
            "incoming_ev_hours_until_departure",
            "incoming_ev_time_until_departure_ratio",
        ]

        for feature in expected_features:
            assert feature in charger_features
            assert charger_meta[feature]["bundle"] == "entity_core_electrical"
            assert charger_meta[feature]["legacy"] is False

        assert charger_meta["hours_until_departure"]["unit"] == "h"
        assert charger_meta["required_average_power_kw"]["unit"] == "kw"
        assert charger_meta["charging_slack_kw"]["unit"] == "kw"
        assert charger_meta["charging_priority_ratio"]["unit"] == "ratio"
        assert charger_meta["departure_feasibility_ratio"]["unit"] == "ratio"
        assert charger_meta["can_charge"]["unit"] == "flag"
        assert charger_meta["charge_efficiency_at_max_ratio"]["unit"] == "ratio"

        connected_col = charger_features.index("connected_state")
        connected_rows = np.flatnonzero(obs["tables"]["charger"][:, connected_col] == 1.0)
        assert len(connected_rows) > 0
        row = int(connected_rows[0])

        def value(name: str) -> float:
            return float(obs["tables"]["charger"][row, charger_features.index(name)])

        step_hours = env.seconds_per_time_step / 3600.0
        departure_steps = value("connected_ev_departure_time_step")
        hours = value("hours_until_departure")
        required_soc = value("connected_ev_required_soc_departure")
        current_soc = value("connected_ev_soc")
        capacity = value("connected_ev_battery_capacity_kwh")
        max_charging_power = value("max_charging_power_kw")
        charge_efficiency = value("charge_efficiency_at_max_ratio")
        expected_energy = max((required_soc - current_soc) * max(capacity, 0.0), 0.0)
        expected_required_power = expected_energy / max(hours, 1.0e-6) if expected_energy > 0.0 else 0.0
        expected_energy_to_full = max((1.0 - current_soc) * max(capacity, 0.0), 0.0)
        expected_max_deliverable = min(
            value("available_charge_power_kw") * max(hours, 0.0) * max(charge_efficiency, 1.0e-6),
            expected_energy_to_full,
        )
        if expected_energy <= 0.0:
            expected_priority = 0.0
        elif hours <= 0.0 or max_charging_power <= 0.0:
            expected_priority = 1.0
        else:
            expected_priority = np.clip(expected_required_power / max(max_charging_power, 1.0e-6), 0.0, 1.0)

        assert hours == pytest.approx(departure_steps * step_hours)
        assert value("time_until_departure_ratio") == pytest.approx(np.clip(hours / 24.0, 0.0, 1.0))
        assert value("energy_to_required_soc_kwh") == pytest.approx(expected_energy)
        assert value("required_average_power_kw") == pytest.approx(expected_required_power)
        assert value("avg_power_to_departure_kw") == pytest.approx(expected_required_power)
        assert value("charging_slack_kw") == pytest.approx(max_charging_power - expected_required_power)
        assert value("charging_priority_ratio") == pytest.approx(expected_priority)
        assert value("connected_ev_energy_to_full_kwh") == pytest.approx(expected_energy_to_full)
        assert value("max_deliverable_energy_until_departure_kwh") == pytest.approx(expected_max_deliverable)
        assert value("departure_energy_margin_kwh") == pytest.approx(expected_max_deliverable - expected_energy)
        expected_feasibility = expected_energy / max(expected_max_deliverable, 1.0e-6) if expected_energy > 0.0 else 0.0
        assert value("departure_feasibility_ratio") == pytest.approx(expected_feasibility)
        expected_min_action = expected_required_power / max(max_charging_power, 1.0e-6) if expected_required_power > 0.0 else 0.0
        assert value("min_required_action_normalized") == pytest.approx(expected_min_action)
    finally:
        env.close()


def test_base_entity_features_add_static_asset_capabilities_without_core_bundle():
    env = CityLearnEnv(_schema_with_bundles(core=False), interface="entity", central_agent=True, episode_time_steps=6, random_seed=0)

    try:
        obs, _ = env.reset(seed=0)
        charger_features = env.entity_specs["tables"]["charger"]["features"]
        charger_meta = env.entity_specs["tables"]["charger"]["feature_metadata"]
        for feature in ["min_charging_power_kw", "min_discharging_power_kw", "charger_efficiency_ratio"]:
            assert feature in charger_features
            assert charger_meta[feature]["bundle"] == "entity_base"
            assert charger_meta[feature]["legacy"] is False

        first_charger = next(
            charger
            for building in env.buildings
            for charger in (building.electric_vehicle_chargers or [])
        )
        charger_row = 0
        assert obs["tables"]["charger"][charger_row, charger_features.index("min_charging_power_kw")] == pytest.approx(first_charger.min_charging_power)
        assert obs["tables"]["charger"][charger_row, charger_features.index("min_discharging_power_kw")] == pytest.approx(first_charger.min_discharging_power)
        assert obs["tables"]["charger"][charger_row, charger_features.index("charger_efficiency_ratio")] == pytest.approx(first_charger.efficiency)

        storage_features = env.entity_specs["tables"]["storage"]["features"]
        storage_meta = env.entity_specs["tables"]["storage"]["feature_metadata"]
        for feature in ["min_charge_power_kw", "min_discharge_power_kw", "efficiency_ratio", "round_trip_efficiency_ratio"]:
            assert feature in storage_features
            assert storage_meta[feature]["bundle"] == "entity_base"
            assert storage_meta[feature]["legacy"] is False
    finally:
        env.close()


def test_core_bundle_exposes_incoming_ev_departure_requirements():
    env = CityLearnEnv(_schema_with_bundles(core=True), interface="entity", central_agent=True, episode_time_steps=24, random_seed=0)

    try:
        obs, _ = env.reset(seed=0)
        charger_features = env.entity_specs["tables"]["charger"]["features"]
        incoming_col = charger_features.index("incoming_state")

        for _ in range(24):
            incoming_rows = np.flatnonzero(obs["tables"]["charger"][:, incoming_col] == 1.0)
            if len(incoming_rows) > 0:
                row = int(incoming_rows[0])
                break
            obs, *_ = env.step(_zero_entity_actions(env))
        else:
            pytest.skip("Dataset does not expose an incoming EV in the test window.")

        def value(name: str) -> float:
            return float(obs["tables"]["charger"][row, charger_features.index(name)])

        step_hours = env.seconds_per_time_step / 3600.0
        required_soc = value("incoming_ev_required_soc_departure")
        departure_steps = value("incoming_ev_departure_time_step")
        hours = value("incoming_ev_hours_until_departure")
        assert required_soc == pytest.approx(-0.1) or 0.0 <= required_soc <= 1.0
        if departure_steps < 0.0:
            assert hours == pytest.approx(-1.0)
            assert value("incoming_ev_time_until_departure_ratio") == pytest.approx(-1.0)
        else:
            assert hours == pytest.approx(departure_steps * step_hours)
            assert value("incoming_ev_time_until_departure_ratio") == pytest.approx(np.clip(hours / 24.0, 0.0, 1.0))
    finally:
        env.close()


def test_core_bundle_uses_charger_efficiency_curves_at_max_power():
    schema = _schema_with_bundles(core=True)
    attrs = schema["buildings"]["Building_1"]["chargers"]["charger_1_1"]["attributes"]
    attrs["charge_efficiency_curve"] = [[0.0, 0.80], [1.0, 0.91]]
    attrs["discharge_efficiency_curve"] = [[0.0, 0.75], [1.0, 0.88]]
    env = CityLearnEnv(schema, interface="entity", central_agent=True, episode_time_steps=6, random_seed=0)

    try:
        obs, _ = env.reset(seed=0)
        features = env.entity_specs["tables"]["charger"]["features"]
        assert obs["tables"]["charger"][0, features.index("charge_efficiency_at_max_ratio")] == pytest.approx(0.91)
        assert obs["tables"]["charger"][0, features.index("discharge_efficiency_at_max_ratio")] == pytest.approx(0.88)
    finally:
        env.close()


def test_temporal_bundle_toggle_changes_presence_of_temporal_features():
    off_env = CityLearnEnv(_schema_with_bundles(temporal=False), interface="entity", central_agent=True, episode_time_steps=6, random_seed=0)
    on_env = CityLearnEnv(_schema_with_bundles(temporal=True), interface="entity", central_agent=True, episode_time_steps=6, random_seed=0)

    try:
        off_env.reset(seed=0)
        on_env.reset(seed=0)
        off_features = set(off_env.entity_specs["tables"]["building"]["features"])
        on_features = set(on_env.entity_specs["tables"]["building"]["features"])
        assert "net_energy_prev_1_kwh_step" not in off_features
        assert "net_energy_prev_1_kwh_step" in on_features
        assert "net_energy_prev_3_mean_kwh_step" in on_features
        for feature in ["hour_sin", "hour_cos", "seconds_of_day_sin", "seconds_of_day_cos", "is_weekend"]:
            assert feature in on_features
        assert "time_step" not in on_features
        assert "time_step" not in set(on_env.entity_specs["tables"]["district"]["features"])
        temporal_meta = on_env.entity_specs["tables"]["building"]["feature_metadata"]["net_energy_prev_1_kwh_step"]
        assert temporal_meta["unit"] == "kwh_step"
        district_temporal_meta = on_env.entity_specs["tables"]["district"]["feature_metadata"]["community_net_prev_1_kwh_step"]
        assert district_temporal_meta["unit"] == "kwh_step"
        obs, _ = on_env.reset(seed=0)
        assert obs["meta"]["time_step"] == 0
        assert obs["meta"]["seconds_per_time_step"] == pytest.approx(on_env.seconds_per_time_step)
    finally:
        off_env.close()
        on_env.close()


def test_derived_forecast_bundle_uses_physical_horizons(tmp_path: Path):
    schema_path = _minute_schema_with_bundles(tmp_path, derived_forecast=True)
    env = CityLearnEnv(str(schema_path), interface="entity", central_agent=True, episode_time_steps=8, random_seed=0)

    try:
        obs, _ = env.reset(seed=0)
        district_features = env.entity_specs["tables"]["district"]["features"]
        building_features = env.entity_specs["tables"]["building"]["features"]
        assert "forecast_price_min_next_15m" in district_features
        assert "forecast_load_mean_next_15m_kw" in building_features

        price_15m = float(obs["tables"]["district"][0, district_features.index("forecast_price_min_next_15m")])
        price_1h = float(obs["tables"]["district"][0, district_features.index("forecast_price_mean_next_1h")])
        load_15m = float(obs["tables"]["building"][0, building_features.index("forecast_load_mean_next_15m_kw")])
        net_15m = float(obs["tables"]["building"][0, building_features.index("forecast_net_mean_next_15m_kw")])

        assert price_15m == pytest.approx(0.10)
        assert price_1h == pytest.approx(np.mean([0.10, 0.11, 0.12, 0.13]))
        assert load_15m == pytest.approx(1.1 / 0.25)
        assert net_15m == pytest.approx(load_15m)
        assert obs["meta"]["forecast_config"]["source"] == "actual_future"
    finally:
        env.close()


def test_action_feedback_bundle_distinguishes_requested_limited_and_applied(tmp_path: Path):
    schema_path = _minute_schema_with_bundles(tmp_path, action_feedback=True)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["buildings"]["Building_1"]["charging_constraints"] = {
        "building_limit_kw": 1.0,
        "observations": {"headroom": True, "violation": True},
    }
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    env = CityLearnEnv(str(schema_path), interface="entity", central_agent=True, episode_time_steps=8, random_seed=0)

    try:
        env.reset(seed=0)
        actions = _zero_entity_actions(env)
        charger_action_features = env.entity_specs["actions"]["charger"]["features"]
        actions["tables"]["charger"][0, charger_action_features.index("electric_vehicle_storage")] = 1.0

        obs, *_ = env.step(actions)
        charger_features = env.entity_specs["tables"]["charger"]["features"]

        def charger_value(name: str) -> float:
            return float(obs["tables"]["charger"][0, charger_features.index(name)])

        assert charger_value("last_requested_action_normalized") == pytest.approx(1.0)
        assert 0.0 < charger_value("last_limited_action_normalized") < 1.0
        assert charger_value("last_requested_power_kw") == pytest.approx(7.2)
        assert charger_value("last_limited_power_kw") == pytest.approx(1.0, abs=1.0e-5)
        assert 0.0 < charger_value("last_applied_power_kw") <= charger_value("last_limited_power_kw")
        assert charger_value("clip_reason_building_headroom") == pytest.approx(1.0)

        storage_features = env.entity_specs["tables"]["storage"]["features"]
        assert "last_requested_action_normalized" in storage_features
        assert "last_applied_power_kw" in storage_features
    finally:
        env.close()


def test_power_energy_kwh_step_consistency_for_core_metrics():
    env = CityLearnEnv(
        _schema_with_bundles(core=True, community=True),
        interface="entity",
        central_agent=True,
        episode_time_steps=6,
        random_seed=0,
    )

    try:
        obs, _ = env.reset(seed=0)
        step_hours = env.seconds_per_time_step / 3600.0

        building_features = env.entity_specs["tables"]["building"]["features"]
        pairs = [
            ("net_power_kw", "net_energy_kwh_step"),
            ("import_power_kw", "import_energy_kwh_step"),
            ("export_power_kw", "export_energy_kwh_step"),
            ("pv_power_kw", "pv_energy_kwh_step"),
            ("pv_surplus_power_kw", "pv_surplus_energy_kwh_step"),
            ("flexible_charge_power_capacity_kw", "flexible_charge_energy_capacity_kwh_step"),
            ("flexible_discharge_power_capacity_kw", "flexible_discharge_energy_capacity_kwh_step"),
        ]

        for power_name, energy_name in pairs:
            p_ix = building_features.index(power_name)
            e_ix = building_features.index(energy_name)
            power = float(obs["tables"]["building"][0, p_ix])
            energy = float(obs["tables"]["building"][0, e_ix])
            assert energy == pytest.approx(power * step_hours, rel=1.0e-5, abs=1.0e-5)

        obs, *_ = env.step(_zero_entity_actions(env))
        district_features = env.entity_specs["tables"]["district"]["features"]
        p_ix = district_features.index("community_net_power_kw")
        e_ix = district_features.index("community_net_energy_kwh_step")
        power = float(obs["tables"]["district"][0, p_ix])
        energy = float(obs["tables"]["district"][0, e_ix])
        assert energy == pytest.approx(power * step_hours, rel=1.0e-5, abs=1.0e-5)
        for feature in [
            "building_import_headroom_kw",
            "building_export_headroom_kw",
            "import_phase_headroom_kw",
            "export_phase_headroom_kw",
        ]:
            assert feature in building_features
        for feature in [
            "community_flexible_charge_power_capacity_kw",
            "community_flexible_discharge_power_capacity_kw",
            "community_flexible_energy_to_full_kwh",
            "community_flexible_energy_available_kwh",
        ]:
            assert feature in district_features
    finally:
        env.close()


def test_core_bundle_exposes_storage_pv_and_phase_power_descriptors():
    schema = _schema_with_bundles(core=True)
    env = CityLearnEnv(schema, interface="entity", central_agent=True, episode_time_steps=6, random_seed=0)

    try:
        obs, _ = env.reset(seed=0)
        storage_features = env.entity_specs["tables"]["storage"]["features"]
        storage_meta = env.entity_specs["tables"]["storage"]["feature_metadata"]
        for feature in [
            "current_efficiency_ratio",
            "degraded_capacity_kwh",
            "soc_min_ratio",
            "can_charge",
            "can_discharge",
            "available_charge_power_kw",
            "available_discharge_power_kw",
            "available_charge_action_normalized",
            "available_discharge_action_normalized",
            "available_charge_energy_kwh_step",
            "available_discharge_energy_kwh_step",
            "max_charge_energy_kwh_step",
            "max_discharge_energy_kwh_step",
            "charge_headroom_ratio",
            "discharge_available_ratio",
            "usable_soc_ratio",
        ]:
            assert feature in storage_features
            assert storage_meta[feature]["bundle"] == "entity_core_electrical"
            assert storage_meta[feature]["legacy"] is False

        if obs["tables"]["storage"].shape[0] > 0:
            row = 0

            def storage_value(name: str) -> float:
                return float(obs["tables"]["storage"][row, storage_features.index(name)])

            capacity = storage_value("capacity_kwh")
            nominal_power = storage_value("nominal_power_kw")
            assert storage_value("available_charge_action_normalized") == pytest.approx(
                storage_value("available_charge_power_kw") / max(nominal_power, 1.0e-6)
            )
            assert storage_value("available_discharge_action_normalized") == pytest.approx(
                storage_value("available_discharge_power_kw") / max(nominal_power, 1.0e-6)
            )
            assert storage_value("available_charge_energy_kwh_step") == pytest.approx(
                storage_value("available_charge_power_kw") * env.seconds_per_time_step / 3600.0
            )
            assert storage_value("available_discharge_energy_kwh_step") == pytest.approx(
                storage_value("available_discharge_power_kw") * env.seconds_per_time_step / 3600.0
            )
            assert storage_value("max_charge_energy_kwh_step") == pytest.approx(
                storage_value("max_charge_power_kw") * env.seconds_per_time_step / 3600.0
            )
            assert storage_value("max_discharge_energy_kwh_step") == pytest.approx(
                storage_value("max_discharge_power_kw") * env.seconds_per_time_step / 3600.0
            )
            assert 0.0 <= storage_value("charge_headroom_ratio") <= 1.0
            assert 0.0 <= storage_value("discharge_available_ratio") <= 1.0
            assert capacity >= 0.0

        pv_features = env.entity_specs["tables"]["pv"]["features"]
        assert "generation_capacity_factor_ratio" in pv_features
        if obs["tables"]["pv"].shape[0] > 0:
            row = 0
            generation_power = float(obs["tables"]["pv"][row, pv_features.index("generation_power_kw")])
            installed_power = float(obs["tables"]["pv"][row, pv_features.index("installed_power_kw")])
            capacity_factor = float(obs["tables"]["pv"][row, pv_features.index("generation_capacity_factor_ratio")])
            expected_capacity_factor = generation_power / installed_power if installed_power > 0.0 else 0.0
            assert capacity_factor == pytest.approx(expected_capacity_factor)

        building_features = env.entity_specs["tables"]["building"]["features"]
        assert "charging_total_service_power_kw" in building_features
    finally:
        env.close()

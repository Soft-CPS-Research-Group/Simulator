from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from citylearn.citylearn import CityLearnEnv


BASE_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"


def _schema_with_bundles(
    *,
    core: bool = False,
    community: bool = False,
    forecast: bool = False,
    temporal: bool = False,
) -> Mapping[str, Any]:
    schema = json.loads(BASE_SCHEMA_PATH.read_text())
    schema["root_directory"] = str(BASE_SCHEMA_PATH.parent)
    schema["interface"] = "entity"
    schema["topology_mode"] = "static"
    schema["observation_bundles"] = {
        "entity_core_electrical": {"active": core},
        "entity_community_operational": {"active": community},
        "entity_forecasts_existing": {"active": forecast},
        "entity_temporal_derived": {"active": temporal},
    }
    return schema


def _zero_entity_actions(env: CityLearnEnv):
    tables = env.action_space["tables"]
    return {
        "tables": {
            "building": np.zeros(tables["building"].shape, dtype="float32"),
            "charger": np.zeros(tables["charger"].shape, dtype="float32"),
        }
    }


def test_observation_bundles_default_to_disabled():
    env = CityLearnEnv(str(BASE_SCHEMA_PATH), interface="entity", central_agent=True, episode_time_steps=6, random_seed=0)

    try:
        env.reset(seed=0)
        bundles = env.entity_specs["observation_bundles"]
        assert bundles["entity_core_electrical"]["active"] is False
        assert bundles["entity_community_operational"]["active"] is False
        assert bundles["entity_forecasts_existing"]["active"] is False
        assert bundles["entity_temporal_derived"]["active"] is False
    finally:
        env.close()


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
        ]

        for feature in expected_features:
            assert feature in charger_features
            assert charger_meta[feature]["bundle"] == "entity_core_electrical"
            assert charger_meta[feature]["legacy"] is False

        assert charger_meta["hours_until_departure"]["unit"] == "h"
        assert charger_meta["required_average_power_kw"]["unit"] == "kw"
        assert charger_meta["charging_slack_kw"]["unit"] == "kw"
        assert charger_meta["charging_priority_ratio"]["unit"] == "ratio"

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
        expected_energy = max((required_soc - current_soc) * max(capacity, 0.0), 0.0)
        expected_required_power = expected_energy / max(hours, 1.0e-6) if expected_energy > 0.0 else 0.0
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
        temporal_meta = on_env.entity_specs["tables"]["building"]["feature_metadata"]["net_energy_prev_1_kwh_step"]
        assert temporal_meta["unit"] == "kwh_step"
        district_temporal_meta = on_env.entity_specs["tables"]["district"]["feature_metadata"]["community_net_prev_1_kwh_step"]
        assert district_temporal_meta["unit"] == "kwh_step"
    finally:
        off_env.close()
        on_env.close()


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
    finally:
        env.close()

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

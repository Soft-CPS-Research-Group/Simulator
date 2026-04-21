from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from citylearn.citylearn import CityLearnEnv


SCHEMA = Path(__file__).resolve().parents[1] / "data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"


def _zero_entity_actions(env: CityLearnEnv):
    tables = env.action_space["tables"]
    return {
        "tables": {
            "building": np.zeros(tables["building"].shape, dtype="float32"),
            "charger": np.zeros(tables["charger"].shape, dtype="float32"),
        }
    }


def _entity_storage_action(env: CityLearnEnv, value: float):
    tables = env.action_space["tables"]
    building = np.zeros(tables["building"].shape, dtype="float32")
    charger = np.zeros(tables["charger"].shape, dtype="float32")
    features = env.entity_specs["actions"]["building"]["features"]

    if "electrical_storage" not in features:
        pytest.skip("Dataset does not expose electrical_storage action in entity mode.")

    building[0, features.index("electrical_storage")] = value
    return {"tables": {"building": building, "charger": charger}}


def test_entity_interface_shapes_and_specs_are_consistent():
    env = CityLearnEnv(str(SCHEMA), interface="entity", central_agent=True, episode_time_steps=6, random_seed=0)

    try:
        observations, _ = env.reset()
        specs = env.entity_specs

        assert isinstance(observations, dict)
        assert "tables" in observations
        assert "edges" in observations
        assert observations["tables"]["district"].shape[1] == len(specs["tables"]["district"]["features"])
        assert observations["tables"]["building"].shape[0] == len(env.buildings)
        assert observations["tables"]["building"].shape[1] == len(specs["tables"]["building"]["features"])
        assert observations["tables"]["charger"].shape[1] == len(specs["tables"]["charger"]["features"])
        assert observations["tables"]["ev"].shape[1] == len(specs["tables"]["ev"]["features"])
    finally:
        env.close()


def test_entity_interface_has_deterministic_id_and_edge_indexing():
    env = CityLearnEnv(str(SCHEMA), interface="entity", central_agent=True, episode_time_steps=6, random_seed=0)

    try:
        first_obs, _ = env.reset(seed=0)
        first_specs = env.entity_specs
        first_building_to_charger = first_obs["edges"]["building_to_charger"].copy()

        second_obs, _ = env.reset(seed=0)
        second_specs = env.entity_specs
        second_building_to_charger = second_obs["edges"]["building_to_charger"].copy()

        assert first_specs["tables"]["building"]["ids"] == second_specs["tables"]["building"]["ids"]
        assert first_specs["tables"]["charger"]["ids"] == second_specs["tables"]["charger"]["ids"]
        assert first_specs["tables"]["ev"]["ids"] == second_specs["tables"]["ev"]["ids"]
        assert np.array_equal(first_building_to_charger, second_building_to_charger)
    finally:
        env.close()


def test_entity_interface_includes_ev_and_charger_tables_and_edges():
    env = CityLearnEnv(str(SCHEMA), interface="entity", central_agent=True, episode_time_steps=6, random_seed=0)

    try:
        observations, _ = env.reset()
        expected_chargers = sum(len(building.electric_vehicle_chargers or []) for building in env.buildings)
        expected_evs = len(env.electric_vehicles)

        assert observations["tables"]["charger"].shape[0] == expected_chargers
        assert observations["tables"]["ev"].shape[0] == expected_evs
        assert observations["edges"]["building_to_charger"].shape[0] == expected_chargers
    finally:
        env.close()


def test_entity_observations_use_latest_settled_endogenous_transition():
    env = CityLearnEnv(str(SCHEMA), interface="entity", central_agent=True, episode_time_steps=6, random_seed=0)

    try:
        env.reset()
        obs, _, terminated, truncated, _ = env.step(_zero_entity_actions(env))
        assert not terminated
        assert not truncated

        feature_names = env.entity_specs["tables"]["building"]["features"]
        feature_index = feature_names.index("net_electricity_consumption")
        settled_t = max(env.time_step - 1, 0)
        expected_current = float(env.buildings[0].net_electricity_consumption[settled_t])

        assert obs["meta"]["endogenous_time_step"] == settled_t
        assert obs["tables"]["building"][0, feature_index] == pytest.approx(expected_current)
    finally:
        env.close()


def test_entity_soc_uses_settled_index_after_nonzero_storage_action():
    env = CityLearnEnv(str(SCHEMA), interface="entity", central_agent=True, episode_time_steps=6, random_seed=0)

    try:
        env.reset(seed=0)
        obs, _, terminated, truncated, _ = env.step(_entity_storage_action(env, 0.5))
        assert not terminated
        assert not truncated

        feature_names = env.entity_specs["tables"]["building"]["features"]
        soc_index = feature_names.index("electrical_storage_soc")
        settled_t = max(env.time_step - 1, 0)
        expected_soc = float(env.buildings[0].electrical_storage.soc[settled_t])

        assert obs["meta"]["endogenous_time_step"] == settled_t
        assert obs["tables"]["building"][0, soc_index] == pytest.approx(expected_soc)
    finally:
        env.close()

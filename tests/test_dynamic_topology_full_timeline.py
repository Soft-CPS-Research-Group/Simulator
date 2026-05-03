from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from citylearn.citylearn import CityLearnEnv


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "data/datasets/citylearn_three_phase_dynamic_topology_demo/schema.json"
EVENT_STEPS = [2, 3, 4, 5, 6, 7, 8, 9]


EXPECTED_TIMELINE = {
    0: {
        "version": 0,
        "events": [],
        "counts": {"building": 17, "charger": 8, "storage": 16, "pv": 16},
    },
    1: {
        "version": 0,
        "events": [],
        "counts": {"building": 17, "charger": 8, "storage": 16, "pv": 16},
    },
    2: {
        "version": 1,
        "events": ["evt_add_member_18"],
        "counts": {"building": 18, "charger": 9, "storage": 17, "pv": 17},
    },
    3: {
        "version": 2,
        "events": ["evt_add_member_18", "evt_add_charger_to_b2"],
        "counts": {"building": 18, "charger": 10, "storage": 17, "pv": 17},
    },
    4: {
        "version": 3,
        "events": ["evt_add_member_18", "evt_add_charger_to_b2", "evt_remove_charger_b5"],
        "counts": {"building": 18, "charger": 9, "storage": 17, "pv": 17},
    },
    5: {
        "version": 4,
        "events": ["evt_add_member_18", "evt_add_charger_to_b2", "evt_remove_charger_b5", "evt_remove_pv_b11"],
        "counts": {"building": 18, "charger": 9, "storage": 17, "pv": 16},
    },
    6: {
        "version": 5,
        "events": [
            "evt_add_member_18",
            "evt_add_charger_to_b2",
            "evt_remove_charger_b5",
            "evt_remove_pv_b11",
            "evt_add_pv_b6",
        ],
        "counts": {"building": 18, "charger": 9, "storage": 17, "pv": 17},
    },
    7: {
        "version": 6,
        "events": [
            "evt_add_member_18",
            "evt_add_charger_to_b2",
            "evt_remove_charger_b5",
            "evt_remove_pv_b11",
            "evt_add_pv_b6",
            "evt_remove_batt_b12",
        ],
        "counts": {"building": 18, "charger": 9, "storage": 16, "pv": 17},
    },
    8: {
        "version": 7,
        "events": [
            "evt_add_member_18",
            "evt_add_charger_to_b2",
            "evt_remove_charger_b5",
            "evt_remove_pv_b11",
            "evt_add_pv_b6",
            "evt_remove_batt_b12",
            "evt_add_batt_b3",
        ],
        "counts": {"building": 18, "charger": 9, "storage": 17, "pv": 17},
    },
    9: {
        "version": 8,
        "events": [
            "evt_add_member_18",
            "evt_add_charger_to_b2",
            "evt_remove_charger_b5",
            "evt_remove_pv_b11",
            "evt_add_pv_b6",
            "evt_remove_batt_b12",
            "evt_add_batt_b3",
            "evt_remove_member_18",
        ],
        "counts": {"building": 17, "charger": 8, "storage": 16, "pv": 16},
    },
}


def _load_schema() -> Mapping[str, Any]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    schema["root_directory"] = str(SCHEMA_PATH.parent)
    schema["interface"] = "entity"
    schema["topology_mode"] = "dynamic"

    for event, step in zip(schema.get("topology_events", []), EVENT_STEPS):
        event["time_step"] = step

    return schema


def _zero_entity_actions(env: CityLearnEnv):
    tables = env.action_space["tables"]
    return {
        "tables": {
            "building": np.zeros(tables["building"].shape, dtype="float32"),
            "charger": np.zeros(tables["charger"].shape, dtype="float32"),
        }
    }


def _snapshot(env: CityLearnEnv) -> Mapping[str, Any]:
    obs = env.observations
    spec = env.entity_specs
    return {
        "time_step": env.time_step,
        "topology_version": env.topology_version,
        "event_ids": [entry["id"] for entry in env.topology_event_log],
        "table_ids": {
            "building": list(spec["tables"]["building"]["ids"]),
            "charger": list(spec["tables"]["charger"]["ids"]),
            "storage": list(spec["tables"]["storage"]["ids"]),
            "pv": list(spec["tables"]["pv"]["ids"]),
        },
        "action_ids": {
            "building": list(spec["actions"]["building"]["ids"]),
            "charger": list(spec["actions"]["charger"]["ids"]),
        },
        "action_features": {
            "building": list(spec["actions"]["building"]["features"]),
            "charger": list(spec["actions"]["charger"]["features"]),
        },
        "obs": obs,
        "observation_space": env.observation_space,
        "action_space": env.action_space,
    }


def _assert_snapshot_contract(env: CityLearnEnv, snapshot: Mapping[str, Any]):
    t = snapshot["time_step"]
    expected = EXPECTED_TIMELINE[t]
    obs = snapshot["obs"]
    observation_space = snapshot["observation_space"]
    action_space = snapshot["action_space"]

    assert snapshot["topology_version"] == expected["version"]
    assert obs["meta"]["time_step"] == t
    assert obs["meta"]["topology_version"] == expected["version"]
    assert snapshot["event_ids"] == expected["events"]

    for table_name, expected_count in expected["counts"].items():
        ids = snapshot["table_ids"][table_name]
        assert len(ids) == expected_count
        assert obs["tables"][table_name].shape[0] == expected_count
        assert obs["tables"][table_name].shape == observation_space["tables"][table_name].shape
        assert np.all(np.isfinite(obs["tables"][table_name]))

    assert snapshot["action_ids"]["building"] == snapshot["table_ids"]["building"]
    assert snapshot["action_space"]["tables"]["building"].shape[0] == expected["counts"]["building"]
    assert snapshot["action_space"]["tables"]["charger"].shape[0] == expected["counts"]["charger"]
    assert snapshot["action_space"]["tables"]["building"].shape == action_space["tables"]["building"].shape
    assert snapshot["action_space"]["tables"]["charger"].shape == action_space["tables"]["charger"].shape

    assert obs["edges"]["district_to_building"].shape[0] == expected["counts"]["building"]
    assert obs["edges"]["building_to_charger"].shape[0] == expected["counts"]["charger"]
    assert obs["edges"]["building_to_storage"].shape[0] == expected["counts"]["storage"]
    assert obs["edges"]["building_to_pv"].shape[0] == expected["counts"]["pv"]

    _assert_edge_bounds(obs["edges"]["building_to_charger"], expected["counts"]["building"], expected["counts"]["charger"])
    _assert_edge_bounds(obs["edges"]["building_to_storage"], expected["counts"]["building"], expected["counts"]["storage"])
    _assert_edge_bounds(obs["edges"]["building_to_pv"], expected["counts"]["building"], expected["counts"]["pv"])

    parsed = env._entity_service.parse_actions(_zero_entity_actions(env))
    assert len(parsed) == expected["counts"]["building"]


def _assert_edge_bounds(edge: np.ndarray, source_count: int, target_count: int):
    if edge.size == 0:
        return

    assert np.all((edge[:, 0] >= 0) & (edge[:, 0] < max(source_count, 1)))
    assert np.all((edge[:, 1] >= 0) & (edge[:, 1] < max(target_count, 1)))


def _building_action_bound(env: CityLearnEnv, building_name: str, action_name: str) -> tuple[float, float]:
    spec = env.entity_specs
    row = spec["actions"]["building"]["ids"].index(building_name)
    col = spec["actions"]["building"]["features"].index(action_name)
    space = env.action_space["tables"]["building"]
    return float(space.low[row, col]), float(space.high[row, col])


def _assert_building_action_map(env: CityLearnEnv, building_name: str, action_name: str, value: float):
    payload = _zero_entity_actions(env)
    payload["map"] = {f"building:{building_name}": {action_name: value}}
    parsed = env._entity_service.parse_actions(payload)
    building_index = [building.name for building in env.buildings].index(building_name)
    assert parsed[building_index][f"{action_name}_action"] == pytest.approx(value)


def _assert_charger_action_map(env: CityLearnEnv, charger_global_id: str, value: float):
    payload = _zero_entity_actions(env)
    payload["map"] = {f"charger:{charger_global_id}": {"electric_vehicle_storage": value}}
    parsed = env._entity_service.parse_actions(payload)
    building_name, charger_id = charger_global_id.split("/", 1)
    building_index = [building.name for building in env.buildings].index(building_name)
    assert parsed[building_index]["electric_vehicle_storage_actions"][charger_id] == pytest.approx(value)


def test_dynamic_topology_full_event_timeline_entity_contract_and_action_spaces():
    env = CityLearnEnv(_load_schema(), interface="entity", topology_mode="dynamic", episode_time_steps=12, random_seed=0)

    try:
        env.reset(seed=0)
        snapshots = {}

        while env.time_step <= 9:
            previous_payload = _zero_entity_actions(env)
            previous_action_shape = {
                name: space.shape for name, space in env.action_space["tables"].items()
            }
            snapshot = _snapshot(env)
            snapshots[env.time_step] = snapshot
            _assert_snapshot_contract(env, snapshot)
            _assert_event_specific_state(env, env.time_step)

            if env.time_step == 9:
                break

            env.step(previous_payload)
            current_action_shape = {
                name: space.shape for name, space in env.action_space["tables"].items()
            }
            if current_action_shape != previous_action_shape:
                with pytest.raises(AssertionError, match="shape mismatch"):
                    env._entity_service.parse_actions(previous_payload)

        _assert_before_after_entity_observation_changes(snapshots)
    finally:
        env.close()


def _assert_event_specific_state(env: CityLearnEnv, t: int):
    spec = env.entity_specs
    buildings = set(spec["tables"]["building"]["ids"])
    chargers = set(spec["tables"]["charger"]["ids"])
    storage = set(spec["tables"]["storage"]["ids"])
    pv = set(spec["tables"]["pv"]["ids"])
    charger_actions = set(spec["actions"]["charger"]["ids"])

    if t < 2:
        assert "Building_18" not in buildings
        assert "Building_18/charger_18_1" not in chargers
        assert "Building_18/electrical_storage" not in storage
        assert "Building_18/pv" not in pv
    else:
        expected_active = t < 9
        assert ("Building_18" in buildings) is expected_active
        assert ("Building_18/charger_18_1" in chargers) is expected_active
        assert ("Building_18/electrical_storage" in storage) is expected_active
        assert ("Building_18/pv" in pv) is expected_active

    assert ("Building_2/charger_2_dyn_1" in chargers) is (t >= 3)
    assert ("Building_2/charger_2_dyn_1" in charger_actions) is (t >= 3)
    if t >= 3:
        _assert_charger_action_map(env, "Building_2/charger_2_dyn_1", 0.25)

    assert ("Building_5/charger_5_1" in chargers) is (t < 4)
    if t >= 4:
        payload = _zero_entity_actions(env)
        payload["map"] = {"charger:Building_5/charger_5_1": {"electric_vehicle_storage": 0.2}}
        with pytest.raises(AssertionError, match="Unknown charger id"):
            env._entity_service.parse_actions(payload)

    assert ("Building_11/pv" in pv) is (t < 5)
    assert ("Building_6/pv" in pv) is (t >= 6)

    assert ("Building_12/electrical_storage" in storage) is (t < 7)
    if t < 7:
        _low, high = _building_action_bound(env, "Building_12", "electrical_storage")
        assert high > 0.0
    else:
        _low, high = _building_action_bound(env, "Building_12", "electrical_storage")
        assert high == pytest.approx(0.0)
        payload = _zero_entity_actions(env)
        payload["map"] = {"building:Building_12": {"electrical_storage": 0.2}}
        with pytest.raises(AssertionError, match="Unknown building action keys"):
            env._entity_service.parse_actions(payload)

    assert ("Building_3/electrical_storage" in storage) is (t >= 8)
    _low, high = _building_action_bound(env, "Building_3", "electrical_storage")
    if t < 8:
        assert high == pytest.approx(0.0)
    else:
        assert high > 0.0
        _assert_building_action_map(env, "Building_3", "electrical_storage", 0.2)


def _assert_before_after_entity_observation_changes(snapshots: Mapping[int, Mapping[str, Any]]):
    def count(t: int, table: str) -> int:
        return snapshots[t]["obs"]["tables"][table].shape[0]

    assert count(1, "building") + 1 == count(2, "building")
    assert count(1, "charger") + 1 == count(2, "charger")
    assert count(1, "storage") + 1 == count(2, "storage")
    assert count(1, "pv") + 1 == count(2, "pv")

    assert count(2, "charger") + 1 == count(3, "charger")
    assert count(3, "charger") - 1 == count(4, "charger")
    assert count(4, "pv") - 1 == count(5, "pv")
    assert count(5, "pv") + 1 == count(6, "pv")
    assert count(6, "storage") - 1 == count(7, "storage")
    assert count(7, "storage") + 1 == count(8, "storage")

    assert count(8, "building") - 1 == count(9, "building")
    assert count(8, "charger") - 1 == count(9, "charger")
    assert count(8, "storage") - 1 == count(9, "storage")
    assert count(8, "pv") - 1 == count(9, "pv")

    assert "Building_18" not in snapshots[1]["table_ids"]["building"]
    assert "Building_18" in snapshots[2]["table_ids"]["building"]
    assert "Building_18" not in snapshots[9]["table_ids"]["building"]
    assert "Building_2/charger_2_dyn_1" not in snapshots[2]["table_ids"]["charger"]
    assert "Building_2/charger_2_dyn_1" in snapshots[3]["table_ids"]["charger"]
    assert "Building_5/charger_5_1" in snapshots[3]["table_ids"]["charger"]
    assert "Building_5/charger_5_1" not in snapshots[4]["table_ids"]["charger"]
    assert "Building_11/pv" in snapshots[4]["table_ids"]["pv"]
    assert "Building_11/pv" not in snapshots[5]["table_ids"]["pv"]
    assert "Building_6/pv" not in snapshots[5]["table_ids"]["pv"]
    assert "Building_6/pv" in snapshots[6]["table_ids"]["pv"]
    assert "Building_12/electrical_storage" in snapshots[6]["table_ids"]["storage"]
    assert "Building_12/electrical_storage" not in snapshots[7]["table_ids"]["storage"]
    assert "Building_3/electrical_storage" not in snapshots[7]["table_ids"]["storage"]
    assert "Building_3/electrical_storage" in snapshots[8]["table_ids"]["storage"]

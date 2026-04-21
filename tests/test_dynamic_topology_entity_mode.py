from __future__ import annotations

import csv
import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from citylearn.citylearn import CityLearnEnv


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "data/datasets/citylearn_three_phase_dynamic_topology_demo/schema.json"


def _load_schema(*, tie_first_two_events: bool = False) -> Mapping[str, Any]:
    schema = json.loads(SCHEMA_PATH.read_text())
    schema["root_directory"] = str(SCHEMA_PATH.parent)
    schema["interface"] = "entity"
    schema["topology_mode"] = "dynamic"

    # Keep event order but move them early to keep tests fast.
    remapped_steps = [2, 3, 4, 5, 6, 7, 8, 9]
    for event, step in zip(schema.get("topology_events", []), remapped_steps):
        event["time_step"] = step

    if tie_first_two_events and len(schema.get("topology_events", [])) >= 2:
        schema["topology_events"][0]["time_step"] = 2
        schema["topology_events"][1]["time_step"] = 2

    return schema


def _zero_entity_actions(env: CityLearnEnv):
    tables = env.action_space["tables"]
    return {
        "tables": {
            "building": np.zeros(tables["building"].shape, dtype="float32"),
            "charger": np.zeros(tables["charger"].shape, dtype="float32"),
        }
    }


def _step_until(env: CityLearnEnv, target_time_step: int):
    while env.time_step < target_time_step:
        env.step(_zero_entity_actions(env))


def _building_index_by_name(env: CityLearnEnv, building_name: str) -> int:
    return [b.name for b in env.buildings].index(building_name)


def test_dynamic_mode_requires_entity_interface():
    with pytest.raises(ValueError):
        CityLearnEnv(
            str(SCHEMA_PATH),
            interface="flat",
            topology_mode="dynamic",
            episode_time_steps=8,
            random_seed=0,
        )


def test_event_boundary_semantics_and_topology_version_increment():
    env = CityLearnEnv(_load_schema(), interface="entity", topology_mode="dynamic", episode_time_steps=14, random_seed=0)

    try:
        obs, _ = env.reset(seed=0)
        initial_buildings = obs["tables"]["building"].shape[0]
        assert obs["meta"]["topology_version"] == 0

        # Event at t=2 must not apply while transitioning to t=1.
        obs, *_ = env.step(_zero_entity_actions(env))
        assert env.time_step == 1
        assert obs["meta"]["topology_version"] == 0
        assert obs["tables"]["building"].shape[0] == initial_buildings

        # Event at t=2 applies after transition 1->2 and before obs at t=2.
        obs, *_ = env.step(_zero_entity_actions(env))
        assert env.time_step == 2
        assert obs["meta"]["topology_version"] == 1
        assert obs["tables"]["building"].shape[0] == initial_buildings + 1
        assert "Building_18" in env.entity_specs["tables"]["building"]["ids"]
    finally:
        env.close()


def test_dynamic_shape_changes_and_action_availability_expand_and_shrink():
    env = CityLearnEnv(_load_schema(), interface="entity", topology_mode="dynamic", episode_time_steps=16, random_seed=0)

    try:
        env.reset(seed=0)
        b3_idx = _building_index_by_name(env, "Building_3")
        b12_idx = _building_index_by_name(env, "Building_12")

        initial_charger_rows = env.entity_specs["tables"]["charger"]["ids"]
        assert len(initial_charger_rows) > 0

        _step_until(env, 2)  # add member
        assert "Building_18" in [b.name for b in env.buildings]

        _step_until(env, 3)  # add charger to Building_2
        charger_rows_after_add = len(env.entity_specs["tables"]["charger"]["ids"])

        _step_until(env, 4)  # remove charger from Building_5
        charger_rows_after_remove = len(env.entity_specs["tables"]["charger"]["ids"])
        assert charger_rows_after_add - charger_rows_after_remove == 1

        _step_until(env, 7)  # remove battery from Building_12
        assert "electrical_storage" not in env.buildings[b12_idx].active_actions

        _step_until(env, 8)  # add battery to Building_3
        assert "electrical_storage" in env.buildings[b3_idx].active_actions

        _step_until(env, 9)  # remove member
        assert "Building_18" not in [b.name for b in env.buildings]
    finally:
        env.close()


def test_hybrid_action_tables_plus_map_precedence_and_unknown_ids_validation():
    env = CityLearnEnv(_load_schema(), interface="entity", topology_mode="dynamic", episode_time_steps=12, random_seed=0)

    try:
        env.reset(seed=0)

        building = env.buildings[0]
        building_idx = 0
        non_ev_actions = [
            name for name in building.active_actions
            if not name.startswith("electric_vehicle_storage_") and "washing_machine" not in name
        ]
        if not non_ev_actions:
            pytest.skip("No non-EV building actions available in this scenario.")

        action_name = non_ev_actions[0]
        payload = _zero_entity_actions(env)
        payload["map"] = {
            f"building:{building.name}": {
                action_name: 0.5,
            }
        }

        parsed = env._entity_service.parse_actions(payload)
        assert parsed[building_idx][f"{action_name}_action"] == pytest.approx(0.5)

        with pytest.raises(AssertionError):
            env._entity_service.parse_actions(
                {
                    "tables": payload["tables"],
                    "map": {
                        "building:unknown_member": {action_name: 0.1},
                    },
                }
            )
    finally:
        env.close()


def test_dynamic_entity_layout_normalization_and_encoding_contract_stays_consistent():
    env = CityLearnEnv(_load_schema(), interface="entity", topology_mode="dynamic", episode_time_steps=16, random_seed=0)

    try:
        obs, _ = env.reset(seed=0)

        for _ in range(11):
            spec = env.entity_specs
            space = env.observation_space

            assert obs["tables"]["building"].shape == space["tables"]["building"].shape
            assert obs["tables"]["charger"].shape == space["tables"]["charger"].shape
            assert obs["tables"]["storage"].shape == space["tables"]["storage"].shape
            assert obs["tables"]["ev"].shape == space["tables"]["ev"].shape

            assert obs["tables"]["building"].shape[0] == len(spec["tables"]["building"]["ids"])
            assert obs["tables"]["charger"].shape[0] == len(spec["tables"]["charger"]["ids"])
            assert obs["tables"]["storage"].shape[0] == len(spec["tables"]["storage"]["ids"])

            # Numerical stability contract for dynamic layouts: no NaN/inf while topology changes.
            for table_name in ("district", "building", "charger", "ev", "storage"):
                assert np.all(np.isfinite(obs["tables"][table_name]))

            # Phase encodings must remain binary during topology mutations.
            building_features = spec["tables"]["building"]["features"]
            phase_columns = [idx for idx, name in enumerate(building_features) if str(name).startswith("phase_")]
            if phase_columns:
                phase_values = obs["tables"]["building"][:, phase_columns]
                assert np.all((phase_values == 0.0) | (phase_values == 1.0))

            # Basic edge encoding integrity for graph models.
            b_count = obs["tables"]["building"].shape[0]
            c_count = obs["tables"]["charger"].shape[0]
            s_count = obs["tables"]["storage"].shape[0]

            if c_count > 0:
                building_to_charger = obs["edges"]["building_to_charger"]
                assert np.all((building_to_charger[:, 0] >= 0) & (building_to_charger[:, 0] < max(b_count, 1)))
                assert np.all((building_to_charger[:, 1] >= 0) & (building_to_charger[:, 1] < max(c_count, 1)))

            if s_count > 0:
                building_to_storage = obs["edges"]["building_to_storage"]
                assert np.all((building_to_storage[:, 0] >= 0) & (building_to_storage[:, 0] < max(b_count, 1)))
                assert np.all((building_to_storage[:, 1] >= 0) & (building_to_storage[:, 1] < max(s_count, 1)))

            obs, *_ = env.step(_zero_entity_actions(env))
    finally:
        env.close()


def test_event_order_is_deterministic_for_same_time_step():
    env = CityLearnEnv(_load_schema(tie_first_two_events=True), interface="entity", topology_mode="dynamic", episode_time_steps=12, random_seed=0)

    try:
        env.reset(seed=0)
        _step_until(env, 2)
        event_ids = [entry["id"] for entry in env.topology_event_log if entry["time_step"] == 2]
        assert event_ids[:2] == ["evt_add_member_18", "evt_add_charger_to_b2"]
    finally:
        env.close()


def test_dynamic_non_central_agent_reward_summary_handles_variable_member_count():
    env = CityLearnEnv(
        _load_schema(),
        interface="entity",
        topology_mode="dynamic",
        central_agent=False,
        episode_time_steps=12,
        random_seed=0,
    )

    try:
        env.reset(seed=0)
        while not env.terminated and not env.truncated:
            env.step(_zero_entity_actions(env))

        assert len(env.episode_rewards) == 1
        summary = env.episode_rewards[0]
        for key in ("min", "max", "sum", "mean"):
            assert key in summary
            # At least one aggregated reward entry must exist.
            value = summary[key]
            if isinstance(value, list):
                assert len(value) > 0
            else:
                assert np.isfinite(float(value))
    finally:
        env.close()


def test_end_mode_exports_respect_dynamic_asset_activity_windows(tmp_path):
    env = CityLearnEnv(
        _load_schema(),
        interface="entity",
        topology_mode="dynamic",
        central_agent=True,
        episode_time_steps=12,
        random_seed=0,
        render_mode="end",
        render=True,
        render_directory=tmp_path,
    )

    def _timestamps(path: Path):
        with path.open(newline="") as handle:
            return [row["timestamp"] for row in csv.DictReader(handle)]

    try:
        env.reset(seed=0)
        while not env.terminated and not env.truncated:
            env.step(_zero_entity_actions(env))

        outputs_path = Path(env.new_folder_path)
        episode_num = env.episode_tracker.episode

        b18_file = outputs_path / f"exported_data_building_18_ep{episode_num}.csv"
        add_charger_file = outputs_path / f"exported_data_building_2_charger_2_dyn_1_ep{episode_num}.csv"
        removed_charger_file = outputs_path / f"exported_data_building_5_charger_5_1_ep{episode_num}.csv"
        removed_battery_file = outputs_path / f"exported_data_building_12_battery_ep{episode_num}.csv"
        added_battery_file = outputs_path / f"exported_data_building_3_battery_ep{episode_num}.csv"

        assert b18_file.is_file()
        assert add_charger_file.is_file()
        assert removed_charger_file.is_file()
        assert removed_battery_file.is_file()
        assert added_battery_file.is_file()

        start_dt = datetime.datetime.combine(env.render_start_date, datetime.time())
        start_dt += datetime.timedelta(seconds=int(env.episode_tracker.episode_start_time_step) * env.seconds_per_time_step)

        def _step_timestamp(step: int) -> str:
            return (start_dt + datetime.timedelta(seconds=step * env.seconds_per_time_step)).strftime("%Y-%m-%dT%H:%M:%S")

        b18_ts = _timestamps(b18_file)
        add_charger_ts = _timestamps(add_charger_file)
        removed_charger_ts = _timestamps(removed_charger_file)
        removed_battery_ts = _timestamps(removed_battery_file)
        added_battery_ts = _timestamps(added_battery_file)

        assert b18_ts[0] >= _step_timestamp(2)
        assert add_charger_ts[0] >= _step_timestamp(3)
        assert removed_charger_ts[-1] < _step_timestamp(4)
        assert removed_battery_ts[-1] < _step_timestamp(7)
        assert added_battery_ts[0] >= _step_timestamp(8)
    finally:
        env.close()

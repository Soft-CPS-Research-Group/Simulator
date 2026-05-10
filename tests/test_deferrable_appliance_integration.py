import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gymnasium")

from citylearn.citylearn import CityLearnEnv


SOURCE_DATASET = Path(__file__).resolve().parent / "data/minute_ev_demo"
DEFERRABLE_OBSERVATIONS = [
    "deferrable_appliance_pending",
    "deferrable_appliance_running",
    "deferrable_appliance_can_start",
    "deferrable_appliance_deadline_missed",
    "deferrable_appliance_earliest_start_time_step",
    "deferrable_appliance_latest_start_time_step",
    "deferrable_appliance_deadline_time_step",
    "deferrable_appliance_hours_until_latest_start",
    "deferrable_appliance_hours_until_deadline",
    "deferrable_appliance_slack_steps",
    "deferrable_appliance_slack_ratio",
    "deferrable_appliance_urgency_ratio",
    "deferrable_appliance_cycle_duration_steps",
    "deferrable_appliance_cycle_energy_kwh",
    "deferrable_appliance_remaining_energy_kwh",
    "deferrable_appliance_current_step_energy_kwh",
    "deferrable_appliance_priority",
    "deferrable_appliance_must_run",
    "deferrable_appliance_cycle_average_power_kw",
    "deferrable_appliance_cycle_peak_power_kw",
    "deferrable_appliance_cycle_load_factor_ratio",
    "deferrable_appliance_cycle_peak_step_offset_ratio",
    "deferrable_appliance_remaining_duration_steps",
    "deferrable_appliance_remaining_average_power_kw",
    "deferrable_appliance_current_step_power_kw",
]


def _schema_with_deferrable_appliance(tmp_path: Path, *, earliest=1, latest=1) -> Path:
    dataset_dir = tmp_path / "minute_ev_demo"
    shutil.copytree(SOURCE_DATASET, dataset_dir)

    pd.DataFrame(
        [
            {
                "profile_id": "normal",
                "duration_steps": 2,
                "total_energy_kwh": 0.3,
                "load_profile": "[0.1,0.2]",
            }
        ]
    ).to_csv(dataset_dir / "washer_cycle_profiles.csv", index=False)
    pd.DataFrame(
        [
            {
                "cycle_id": "cycle_1",
                "profile_id": "normal",
                "earliest_start_time_step": earliest,
                "latest_start_time_step": latest,
                "deadline_time_step": latest + 1,
                "priority": 0.75,
                "must_run": True,
            }
        ]
    ).to_csv(dataset_dir / "washer_flexibility_schedule.csv", index=False)

    schema_path = dataset_dir / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["root_directory"] = str(dataset_dir)
    for observation in DEFERRABLE_OBSERVATIONS:
        schema["observations"][observation] = {"active": True, "shared_in_central_agent": True}
    schema["actions"]["deferrable_appliance"] = {"active": True}
    schema["buildings"]["Building_1"]["deferrable_appliances"] = {
        "washer_1": {
            "type": "citylearn.energy_model.DeferrableAppliance",
            "cycle_profiles_file": "washer_cycle_profiles.csv",
            "flexibility_schedule_file": "washer_flexibility_schedule.csv",
        }
    }
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return schema_path


def _zero_flat_actions(env: CityLearnEnv):
    return [np.zeros(space.shape, dtype="float32") for space in env.action_space]


def _kpi_value(df, *, level, name, cost_function):
    row = df[(df["level"] == level) & (df["name"] == name) & (df["cost_function"] == cost_function)]
    assert len(row) == 1
    return float(row["value"].iloc[0])


def test_flat_start_action_consumption_and_service_kpis_are_golden(tmp_path: Path):
    schema_path = _schema_with_deferrable_appliance(tmp_path, earliest=1, latest=1)
    env = CityLearnEnv(str(schema_path), central_agent=True, episode_time_steps=4, random_seed=0)

    try:
        env.reset(seed=0)
        action_index = env.action_names[0].index("deferrable_appliance_washer_1")

        env.step(_zero_flat_actions(env))

        start_actions = _zero_flat_actions(env)
        start_actions[0][action_index] = 1.0
        env.step(start_actions)
        building = env.buildings[0]

        assert building.deferrable_appliances_electricity_consumption[1] == pytest.approx(0.1)

        env.step(_zero_flat_actions(env))
        assert building.deferrable_appliances_electricity_consumption[2] == pytest.approx(0.2)

        kpis = env.evaluate_v2()
        assert _kpi_value(
            kpis,
            level="building",
            name="Building_1",
            cost_function="building_deferrable_appliance_service_completed_cycles_count",
        ) == 1.0
        assert _kpi_value(
            kpis,
            level="building",
            name="Building_1",
            cost_function="building_deferrable_appliance_service_missed_cycles_count",
        ) == 0.0
        assert _kpi_value(
            kpis,
            level="district",
            name="District",
            cost_function="district_deferrable_appliance_service_service_level_ratio",
        ) == 1.0
        assert _kpi_value(
            kpis,
            level="district",
            name="District",
            cost_function="district_deferrable_appliance_service_served_energy_total_kwh",
        ) == pytest.approx(0.3)
    finally:
        env.close()


def test_missed_cycle_service_kpis_are_golden(tmp_path: Path):
    schema_path = _schema_with_deferrable_appliance(tmp_path, earliest=1, latest=1)
    env = CityLearnEnv(str(schema_path), central_agent=True, episode_time_steps=4, random_seed=0)

    try:
        env.reset(seed=0)
        while not env.terminated:
            env.step(_zero_flat_actions(env))

        kpis = env.evaluate_v2()
        assert _kpi_value(
            kpis,
            level="building",
            name="Building_1",
            cost_function="building_deferrable_appliance_service_completed_cycles_count",
        ) == 0.0
        assert _kpi_value(
            kpis,
            level="building",
            name="Building_1",
            cost_function="building_deferrable_appliance_service_missed_cycles_count",
        ) == 1.0
        assert _kpi_value(
            kpis,
            level="district",
            name="District",
            cost_function="district_deferrable_appliance_service_service_level_ratio",
        ) == 0.0
        assert _kpi_value(
            kpis,
            level="district",
            name="District",
            cost_function="district_deferrable_appliance_service_unserved_energy_total_kwh",
        ) == pytest.approx(0.3)
    finally:
        env.close()


def test_entity_table_action_edge_and_start(tmp_path: Path):
    schema_path = _schema_with_deferrable_appliance(tmp_path, earliest=0, latest=0)
    env = CityLearnEnv(str(schema_path), interface="entity", central_agent=True, episode_time_steps=3, random_seed=0)

    try:
        observations, _ = env.reset(seed=0)
        specs = env.entity_specs
        assert "deferrable_appliance" in specs["tables"]
        assert specs["tables"]["deferrable_appliance"]["ids"] == ["Building_1/washer_1"]
        assert specs["actions"]["deferrable_appliance"]["features"] == ["start"]
        assert "building_to_deferrable_appliance" in specs["edges"]
        assert observations["tables"]["deferrable_appliance"].shape == (1, len(DEFERRABLE_OBSERVATIONS))
        features = specs["tables"]["deferrable_appliance"]["features"]

        def value(name: str) -> float:
            return float(observations["tables"]["deferrable_appliance"][0, features.index(name)])

        step_hours = env.seconds_per_time_step / 3600.0
        assert value("must_run") == 1.0
        assert value("cycle_average_power_kw") == pytest.approx(0.3 / (2 * step_hours))
        assert value("cycle_peak_power_kw") == pytest.approx(0.2 / step_hours)
        assert value("cycle_load_factor_ratio") == pytest.approx(0.75)
        assert value("cycle_peak_step_offset_ratio") == pytest.approx(1.0)
        assert value("remaining_duration_steps") == pytest.approx(2.0)
        assert value("remaining_average_power_kw") == pytest.approx(0.3 / (2 * step_hours))
        assert value("current_step_power_kw") == pytest.approx(0.0)

        payload = {
            "tables": {
                name: np.zeros(space.shape, dtype="float32")
                for name, space in env.action_space["tables"].items()
                if name in {"building", "charger", "deferrable_appliance"}
            }
        }
        payload["tables"]["deferrable_appliance"][0, 0] = 1.0
        env.step(payload)

        assert env.buildings[0].deferrable_appliances_electricity_consumption[0] == pytest.approx(0.1)
    finally:
        env.close()

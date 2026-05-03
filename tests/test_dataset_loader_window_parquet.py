import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("pyarrow")

from citylearn.citylearn import CityLearnEnv


SOURCE_DATASET = Path(__file__).resolve().parent / "data/minute_ev_demo"


def _copy_minute_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "minute_ev_demo"
    shutil.copytree(SOURCE_DATASET, dataset_dir)

    washing_machine = pd.DataFrame(
        {
            "day_type": [1] * 8,
            "hour": list(range(8)),
            "wm_start_time_step": [-1, -1, 2, 2, 4, 4, -1, -1],
            "wm_end_time_step": [-1, -1, 3, 3, 5, 5, -1, -1],
            "load_profile": ["-1", "-1", "[0.1, 0.2]", "[0.1, 0.2]", "[0.3]", "[0.3]", "-1", "-1"],
        }
    )
    washing_machine.to_csv(dataset_dir / "washing_machine_1.csv", index=False)

    schema_path = dataset_dir / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["root_directory"] = str(dataset_dir)
    schema["observations"]["washing_machine_start_time_step"] = {
        "active": True,
        "shared_in_central_agent": True,
    }
    schema["observations"]["washing_machine_end_time_step"] = {
        "active": True,
        "shared_in_central_agent": True,
    }
    schema["actions"]["washing_machine"] = {"active": True}
    schema["buildings"]["Building_1"]["washing_machines"] = {
        "washing_machine_1": {
            "type": "citylearn.energy_model.WashingMachine",
            "autosize": False,
            "washing_machine_energy_simulation": "washing_machine_1.csv",
        }
    }
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return schema_path


def _convert_schema_sources_to_parquet(schema_path: Path) -> Path:
    dataset_dir = schema_path.parent
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    converted = copy.deepcopy(schema)

    def convert(filename: str) -> str:
        source = dataset_dir / filename
        target = source.with_suffix(".parquet")
        pd.read_csv(source).to_parquet(target, index=False)
        return target.name

    for building in converted["buildings"].values():
        building["energy_simulation"] = convert(building["energy_simulation"])
        building["weather"] = convert(building["weather"])
        building["carbon_intensity"] = convert(building["carbon_intensity"])
        building["pricing"] = convert(building["pricing"])

        for charger in (building.get("chargers") or {}).values():
            charger["charger_simulation"] = convert(charger["charger_simulation"])

        for washing_machine in (building.get("washing_machines") or {}).values():
            washing_machine["washing_machine_energy_simulation"] = convert(
                washing_machine["washing_machine_energy_simulation"]
            )

    parquet_schema_path = dataset_dir / "schema_parquet.json"
    parquet_schema_path.write_text(json.dumps(converted, indent=2), encoding="utf-8")
    return parquet_schema_path


def _zero_actions(env: CityLearnEnv):
    return [np.zeros(space.shape, dtype="float32") for space in env.action_space]


def _rollout_trace(schema_path: Path):
    env = CityLearnEnv(
        str(schema_path),
        central_agent=True,
        simulation_start_time_step=2,
        simulation_end_time_step=6,
        episode_time_steps=3,
        rolling_episode_split=False,
        random_episode_split=False,
        random_seed=0,
    )

    try:
        observations, _ = env.reset(seed=0)
        building = env.buildings[0]

        raw_energy_rows = len(building.energy_simulation.__dict__["_month"])
        raw_weather_rows = len(building.weather.__dict__["_outdoor_dry_bulb_temperature"])
        raw_pricing_rows = len(building.pricing.__dict__["_electricity_pricing"])
        raw_carbon_rows = len(building.carbon_intensity.__dict__["_carbon_intensity"])
        raw_charger_rows = len(
            building.electric_vehicle_chargers[0].charger_simulation.__dict__["_electric_vehicle_charger_state"]
        )
        raw_washing_rows = len(
            building.washing_machines[0].washing_machine_simulation.__dict__["_wm_start_time_step"]
        )

        trace = {
            "reset_observations": np.asarray(observations[0], dtype="float64"),
            "raw_rows": (
                raw_energy_rows,
                raw_weather_rows,
                raw_pricing_rows,
                raw_carbon_rows,
                raw_charger_rows,
                raw_washing_rows,
            ),
            "episode_starts": [env.episode_tracker.episode_start_time_step],
            "net": [],
            "cost": [],
            "emission": [],
            "charger_state": [],
            "washing_start": [],
        }

        for _ in range(2):
            observations, *_ = env.step(_zero_actions(env))
            t = env.time_step - 1
            trace["net"].append(float(building.net_electricity_consumption[t]))
            trace["cost"].append(float(building.net_electricity_consumption_cost[t]))
            trace["emission"].append(float(building.net_electricity_consumption_emission[t]))
            trace["charger_state"].append(
                float(building.electric_vehicle_chargers[0].charger_simulation.electric_vehicle_charger_state[t])
            )
            trace["washing_start"].append(
                int(building.washing_machines[0].washing_machine_simulation.wm_start_time_step[t])
            )

        env.reset(seed=0)
        trace["episode_starts"].append(env.episode_tracker.episode_start_time_step)
        return trace
    finally:
        env.close()


def test_windowed_loader_reads_only_simulation_window_and_parquet_matches_csv(tmp_path: Path):
    csv_schema_path = _copy_minute_dataset(tmp_path)
    parquet_schema_path = _convert_schema_sources_to_parquet(csv_schema_path)

    csv_trace = _rollout_trace(csv_schema_path)
    parquet_trace = _rollout_trace(parquet_schema_path)

    assert csv_trace["raw_rows"] == (5, 5, 5, 5, 5, 5)
    assert parquet_trace["raw_rows"] == csv_trace["raw_rows"]
    assert parquet_trace["episode_starts"] == csv_trace["episode_starts"]
    np.testing.assert_allclose(parquet_trace["reset_observations"], csv_trace["reset_observations"])
    np.testing.assert_allclose(parquet_trace["net"], csv_trace["net"])
    np.testing.assert_allclose(parquet_trace["cost"], csv_trace["cost"])
    np.testing.assert_allclose(parquet_trace["emission"], csv_trace["emission"])
    assert parquet_trace["charger_state"] == pytest.approx(csv_trace["charger_state"])
    assert parquet_trace["washing_start"] == csv_trace["washing_start"]


def test_shared_weather_pricing_and_carbon_sources_are_cached_between_buildings(tmp_path: Path):
    schema_path = _copy_minute_dataset(tmp_path)
    dataset_dir = schema_path.parent
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    schema["buildings"]["Building_2"] = copy.deepcopy(schema["buildings"]["Building_1"])
    schema["buildings"]["Building_2"]["energy_simulation"] = "Building_2.csv"
    schema["buildings"]["Building_2"].pop("chargers", None)
    schema["buildings"]["Building_2"].pop("washing_machines", None)
    pd.read_csv(dataset_dir / "Building_1.csv").to_csv(dataset_dir / "Building_2.csv", index=False)
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    env = CityLearnEnv(str(schema_path), central_agent=True, episode_time_steps=3, random_seed=0)

    try:
        env.reset(seed=0)
        assert env.buildings[0].weather is env.buildings[1].weather
        assert env.buildings[0].pricing is env.buildings[1].pricing
        assert env.buildings[0].carbon_intensity is env.buildings[1].carbon_intensity
        assert env.buildings[0].energy_simulation is not env.buildings[1].energy_simulation
    finally:
        env.close()

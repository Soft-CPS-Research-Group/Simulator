from pathlib import Path

import pytest

pytest.importorskip("gymnasium")

from citylearn.citylearn import CityLearnEnv
from citylearn.internal.units import (
    normalized_capacity_action_to_energy_kwh,
    normalized_power_action_to_energy_kwh,
    power_kw_to_energy_kwh,
)


SCHEMA = Path(__file__).resolve().parents[2] / "data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"


def test_thermal_storage_actions_use_matching_storage_capacities(monkeypatch):
    env = CityLearnEnv(str(SCHEMA), central_agent=True, episode_time_steps=4, random_seed=0)

    try:
        env.reset()
        building = env.buildings[0]

        building.cooling_storage.capacity = 100.0
        building.heating_storage.capacity = 10.0
        building.dhw_storage.capacity = 20.0

        for storage in [building.heating_storage, building.dhw_storage]:
            storage.efficiency = 1.0
            storage.loss_coefficient = 0.0
            storage.initial_soc = 0.0
            storage.reset()

        monkeypatch.setattr(building.heating_device, "get_max_output_power", lambda *args, **kwargs: 1.0e6)
        monkeypatch.setattr(building.heating_device, "get_input_power", lambda *args, **kwargs: 0.0)
        monkeypatch.setattr(building.dhw_device, "get_max_output_power", lambda *args, **kwargs: 1.0e6)
        monkeypatch.setattr(building.dhw_device, "get_input_power", lambda *args, **kwargs: 0.0)

        building.update_heating_storage(0.1)
        building.update_dhw_storage(0.1)

        expected_heating_kwh = normalized_capacity_action_to_energy_kwh(
            0.1,
            building.heating_storage.capacity,
            seconds_per_time_step=building.seconds_per_time_step,
            scale_with_time=True,
        )
        expected_dhw_kwh = normalized_capacity_action_to_energy_kwh(
            0.1,
            building.dhw_storage.capacity,
            seconds_per_time_step=building.seconds_per_time_step,
            scale_with_time=True,
        )

        assert building.heating_storage.energy_balance[building.time_step] == pytest.approx(expected_heating_kwh)
        assert building.dhw_storage.energy_balance[building.time_step] == pytest.approx(expected_dhw_kwh)
    finally:
        env.close()


def test_action_to_energy_helpers_are_consistent():
    assert normalized_power_action_to_energy_kwh(0.5, 4.0, 900.0) == pytest.approx(0.5)
    assert power_kw_to_energy_kwh(2.0, 1800.0) == pytest.approx(1.0)
    assert normalized_capacity_action_to_energy_kwh(
        0.25,
        8.0,
        seconds_per_time_step=1800.0,
        scale_with_time=True,
    ) == pytest.approx(1.0)

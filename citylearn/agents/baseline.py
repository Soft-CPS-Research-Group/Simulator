from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from citylearn.agents.base import Agent
from citylearn.citylearn import CityLearnEnv


class BusinessAsUsualAgent(Agent):
    """Deterministic operational baseline for day-to-day equipment use.

    The policy represents a conservative non-optimizing operator:
    connected EVs charge toward 100%, deferrable appliances start as soon as
    they can, and electrical storage performs simple PV self-consumption.
    """

    def __init__(
        self,
        env: CityLearnEnv,
        ev_target_soc: float = 1.0,
        storage_min_soc: float = 0.20,
        storage_max_soc: float = 0.90,
        storage_deadband_kw: float = 0.05,
        deferrable_start_action: float = 1.0,
        **kwargs: Any,
    ):
        self.ev_target_soc = float(np.clip(1.0 if ev_target_soc is None else ev_target_soc, 0.0, 1.0))
        self.storage_min_soc = float(np.clip(0.20 if storage_min_soc is None else storage_min_soc, 0.0, 1.0))
        self.storage_max_soc = float(np.clip(0.90 if storage_max_soc is None else storage_max_soc, 0.0, 1.0))
        if self.storage_max_soc < self.storage_min_soc:
            self.storage_min_soc, self.storage_max_soc = self.storage_max_soc, self.storage_min_soc
        self.storage_deadband_kw = max(float(0.05 if storage_deadband_kw is None else storage_deadband_kw), 0.0)
        self.deferrable_start_action = float(np.clip(1.0 if deferrable_start_action is None else deferrable_start_action, 0.0, 1.0))
        super().__init__(env, **kwargs)

        # Entity-interface envs still accept flat-like action payloads. Keep
        # agent bookkeeping in flat space so the same policy works in both modes.
        self.observation_space = self.env.unwrapped.flat_observation_space
        self.action_space = self.env.unwrapped.flat_action_space
        self.reset()

    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        env = self.env.unwrapped
        per_building_actions = [
            self._building_action_vector(building)
            for building in env.buildings
        ]

        if env.central_agent:
            actions = [[value for vector in per_building_actions for value in vector]]
        else:
            actions = per_building_actions

        self.actions = actions
        self.next_time_step()
        return actions

    def _building_action_vector(self, building) -> List[float]:
        bounds = list(zip(building.action_space.low, building.action_space.high))
        preliminary = {}
        ev_requested_kw = 0.0
        deferrable_requested_kw = 0.0

        for action_name, (low, high) in zip(building.active_actions, bounds):
            if action_name.startswith('electric_vehicle_storage_'):
                charger_id = action_name.replace('electric_vehicle_storage_', '', 1)
                value = self._ev_action(building, charger_id, low, high)
                preliminary[action_name] = value
                ev_requested_kw += self._charger_requested_kw(building, charger_id, value)

            elif action_name.startswith('deferrable_appliance_'):
                value = self._deferrable_action(building, action_name, low, high)
                preliminary[action_name] = value
                deferrable_requested_kw += self._deferrable_start_power_kw(building, action_name, value)

        actions = []
        for action_name, (low, high) in zip(building.active_actions, bounds):
            if action_name == 'electrical_storage':
                value = self._storage_action(
                    building,
                    low,
                    high,
                    prospective_ev_kw=ev_requested_kw,
                    prospective_deferrable_kw=deferrable_requested_kw,
                )
            elif action_name in preliminary:
                value = preliminary[action_name]
            else:
                value = 0.0

            actions.append(float(np.clip(value, low, high)))

        return actions

    def _ev_action(self, building, charger_id: str, low: float, high: float) -> float:
        charger = self._charger(building, charger_id)
        ev = None if charger is None else getattr(charger, 'connected_electric_vehicle', None)

        if ev is None:
            return 0.0

        soc = self._current_soc(getattr(ev, 'battery', None))
        if soc is None or soc >= self.ev_target_soc - 1.0e-6:
            return 0.0

        max_power = self._safe_scalar(getattr(charger, 'max_charging_power', 0.0), 0.0)
        if max_power <= 0.0:
            return 0.0

        return float(np.clip(1.0, max(0.0, low), max(0.0, high)))

    def _deferrable_action(self, building, action_name: str, low: float, high: float) -> float:
        appliance = self._deferrable_appliance(building, action_name)
        if appliance is None:
            return 0.0

        observations = appliance.observations()
        pending = self._safe_scalar(observations.get('pending'), 0.0)
        running = self._safe_scalar(observations.get('running'), 0.0)
        can_start = self._safe_scalar(observations.get('can_start'), 0.0)
        deadline_missed = self._safe_scalar(observations.get('deadline_missed'), 0.0)

        if pending > 0.5 and running <= 0.5 and can_start > 0.5 and deadline_missed <= 0.5:
            return float(np.clip(self.deferrable_start_action, low, high))

        return 0.0

    def _storage_action(
        self,
        building,
        low: float,
        high: float,
        *,
        prospective_ev_kw: float,
        prospective_deferrable_kw: float,
    ) -> float:
        storage = getattr(building, 'electrical_storage', None)
        nominal_power_kw = self._safe_scalar(getattr(storage, 'nominal_power', 0.0), 0.0)
        capacity = self._safe_scalar(getattr(storage, 'capacity', 0.0), 0.0)

        if storage is None or nominal_power_kw <= 0.0 or capacity <= 0.0:
            return 0.0

        soc = self._current_soc(storage)
        if soc is None:
            return 0.0

        load_kw = self._base_load_kw(building) + max(prospective_ev_kw, 0.0) + max(prospective_deferrable_kw, 0.0)
        pv_kw = self._pv_generation_kw(building)
        net_kw = load_kw - pv_kw

        if net_kw < -self.storage_deadband_kw and soc < self.storage_max_soc:
            return float(np.clip(min(-net_kw, nominal_power_kw) / nominal_power_kw, low, high))

        if net_kw > self.storage_deadband_kw and soc > self.storage_min_soc:
            return float(np.clip(-min(net_kw, nominal_power_kw) / nominal_power_kw, low, high))

        return float(np.clip(0.0, low, high))

    def _base_load_kw(self, building) -> float:
        t = int(building.time_step)
        step_hours = max(float(building.seconds_per_time_step), 1.0) / 3600.0
        load_kwh = building._dataset_energy_to_control_step(self._series_value(building.non_shiftable_load, t))

        try:
            temperature = self._series_value(building.weather.outdoor_dry_bulb_temperature, t)
            cooling = building._dataset_energy_to_control_step(self._series_value(building.cooling_demand, t))
            heating = building._dataset_energy_to_control_step(self._series_value(building.heating_demand, t))
            dhw = building._dataset_energy_to_control_step(self._series_value(building.dhw_demand, t))
            load_kwh += max(self._safe_scalar(building.cooling_device.get_input_power(cooling, temperature, heating=False), 0.0), 0.0)

            try:
                heating_load = building.heating_device.get_input_power(heating, temperature, heating=True)
            except TypeError:
                heating_load = building.heating_device.get_input_power(heating)
            load_kwh += max(self._safe_scalar(heating_load, 0.0), 0.0)

            try:
                dhw_load = building.dhw_device.get_input_power(dhw, temperature, heating=True)
            except TypeError:
                dhw_load = building.dhw_device.get_input_power(dhw)
            load_kwh += max(self._safe_scalar(dhw_load, 0.0), 0.0)
        except Exception:
            pass

        return max(load_kwh, 0.0) / step_hours

    def _pv_generation_kw(self, building) -> float:
        t = int(building.time_step)
        step_hours = max(float(building.seconds_per_time_step), 1.0) / 3600.0
        return max(abs(self._series_value(building.solar_generation, t)), 0.0) / step_hours

    def _charger_requested_kw(self, building, charger_id: str, action: float) -> float:
        if action <= 0.0:
            return 0.0

        charger = self._charger(building, charger_id)
        if charger is None:
            return 0.0

        return max(action, 0.0) * max(self._safe_scalar(getattr(charger, 'max_charging_power', 0.0), 0.0), 0.0)

    def _deferrable_start_power_kw(self, building, action_name: str, action: float) -> float:
        if action <= 0.0:
            return 0.0

        appliance = self._deferrable_appliance(building, action_name)
        if appliance is None:
            return 0.0

        energy_kwh = self._safe_scalar(appliance.preview_start_energy_kwh(action), 0.0)
        step_hours = max(float(building.seconds_per_time_step), 1.0) / 3600.0
        return max(energy_kwh, 0.0) / step_hours

    @staticmethod
    def _charger(building, charger_id: str):
        for charger in getattr(building, 'electric_vehicle_chargers', []) or []:
            if getattr(charger, 'charger_id', None) == charger_id:
                return charger
        return None

    @staticmethod
    def _deferrable_appliance(building, action_name: str):
        appliance_name = action_name.replace('deferrable_appliance_', '', 1)
        for appliance in getattr(building, 'deferrable_appliances', []) or []:
            if getattr(appliance, 'name', None) == appliance_name or action_name == getattr(appliance, 'name', None):
                return appliance
        return None

    @staticmethod
    def _current_soc(storage) -> Optional[float]:
        if storage is None:
            return None

        try:
            t = int(storage.time_step)
            soc_series = getattr(storage, 'soc')
            if hasattr(soc_series, '__len__') and not np.isscalar(soc_series):
                value = soc_series[min(max(t, 0), len(soc_series) - 1)]
            else:
                value = soc_series
        except Exception:
            return None

        try:
            soc = float(value)
        except (TypeError, ValueError):
            return None

        if not np.isfinite(soc):
            return None

        if abs(soc) > 1.5:
            soc /= 100.0

        return float(np.clip(soc, 0.0, 1.0))

    @staticmethod
    def _series_value(values, index: int) -> float:
        try:
            return float(values[min(max(index, 0), len(values) - 1)])
        except Exception:
            return 0.0

    @staticmethod
    def _safe_scalar(value, default: float = 0.0) -> float:
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return float(default)

        if not np.isfinite(scalar):
            return float(default)

        return scalar

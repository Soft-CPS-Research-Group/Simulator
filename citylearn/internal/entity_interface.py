"""Canonical entity observation/action contract for CityLearnEnv."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from gymnasium import spaces
import numpy as np


@dataclass(frozen=True)
class ChargerRef:
    """Resolved charger location inside the environment."""

    row: int
    building_index: int
    building_name: str
    charger_id: str
    global_id: str


@dataclass(frozen=True)
class StorageRef:
    """Resolved storage location inside the environment."""

    row: int
    building_index: int
    building_name: str
    storage_id: str
    global_id: str


@dataclass(frozen=True)
class PVRef:
    """Resolved PV location inside the environment."""

    row: int
    building_index: int
    building_name: str
    pv_id: str
    global_id: str


class CityLearnEntityInterfaceService:
    """Build and parse canonical entity payloads."""

    CORE_BUNDLE = "entity_core_electrical"
    COMMUNITY_BUNDLE = "entity_community_operational"
    FORECAST_BUNDLE = "entity_forecasts_existing"
    TEMPORAL_BUNDLE = "entity_temporal_derived"
    DEFAULT_OBSERVATION_BUNDLES = {
        CORE_BUNDLE: False,
        COMMUNITY_BUNDLE: False,
        FORECAST_BUNDLE: False,
        TEMPORAL_BUNDLE: False,
    }

    LEGACY_CHARGER_FEATURES = [
        "connected_state",
        "incoming_state",
        "connected_ev_soc",
        "connected_ev_required_soc_departure",
        "connected_ev_battery_capacity_kwh",
        "connected_ev_departure_time_step",
        "incoming_ev_estimated_soc_arrival",
        "incoming_ev_estimated_arrival_time_step",
        "last_charged_kwh",
        "max_charging_power_kw",
        "max_discharging_power_kw",
    ]
    EXTRA_CHARGER_FEATURES = [
        "commanded_power_kw",
        "applied_power_kw",
        "applied_energy_kwh_step",
        "energy_to_required_soc_kwh",
        "avg_power_to_departure_kw",
    ]
    LEGACY_EV_FEATURES = [
        "soc",
        "battery_capacity_kwh",
        "depth_of_discharge_ratio",
    ]
    EXTRA_EV_FEATURES = [
        "soc_ratio",
        "soc_min_ratio",
        "soc_max_ratio",
        "energy_available_kwh",
        "energy_to_full_kwh",
    ]
    LEGACY_STORAGE_FEATURES = [
        "soc",
        "capacity_kwh",
        "nominal_power_kw",
        "electricity_consumption_kwh",
    ]
    EXTRA_STORAGE_FEATURES = [
        "electrical_storage_soc_ratio",
        "max_charge_power_kw",
        "max_discharge_power_kw",
        "energy_to_full_kwh",
        "energy_available_kwh",
    ]
    PV_FEATURES = [
        "generation_power_kw",
        "generation_energy_kwh_step",
        "installed_power_kw",
    ]
    CORE_BUILDING_EXTRA_FEATURES = [
        "net_power_kw",
        "net_energy_kwh_step",
        "import_power_kw",
        "import_energy_kwh_step",
        "export_power_kw",
        "export_energy_kwh_step",
        "load_power_kw",
        "load_energy_kwh_step",
        "pv_power_kw",
        "pv_energy_kwh_step",
        "bess_power_kw",
        "bess_energy_kwh_step",
        "ev_charging_power_kw",
        "ev_charging_energy_kwh_step",
        "electrical_storage_soc_ratio",
    ]
    COMMUNITY_DISTRICT_EXTRA_FEATURES = [
        "community_net_power_kw",
        "community_net_energy_kwh_step",
        "community_import_power_kw",
        "community_import_energy_kwh_step",
        "community_export_power_kw",
        "community_export_energy_kwh_step",
        "community_pv_power_kw",
        "community_pv_energy_kwh_step",
        "community_bess_power_kw",
        "community_bess_energy_kwh_step",
        "community_ev_power_kw",
        "community_ev_energy_kwh_step",
        "community_building_headroom_kw",
        "community_building_export_headroom_kw",
        "community_phase_headroom_kw",
        "community_phase_export_headroom_kw",
        "active_buildings_count",
        "active_chargers_count",
        "active_evs_count",
        "topology_version",
    ]
    TEMPORAL_BUILDING_FEATURES = [
        "net_energy_prev_1_kwh_step",
        "net_energy_prev_3_mean_kwh_step",
        "import_energy_prev_1_kwh_step",
        "export_energy_prev_1_kwh_step",
    ]
    TEMPORAL_DISTRICT_FEATURES = [
        "community_net_prev_1_kwh_step",
        "community_net_prev_3_mean_kwh_step",
    ]

    def __init__(self, env):
        self.env = env
        self._initialized = False
        self._layout_topology_version = -1
        self._observation_bundles = self._resolve_observation_bundles()

    def reset(self):
        """Rebuild specs and reusable buffers for the current episode layout."""

        self._build_entity_layout()
        self._build_spaces()
        self._build_specs()
        self._initialized = True
        self._layout_topology_version = int(getattr(self.env, 'topology_version', 0))

    def invalidate(self):
        """Mark specs/layout as stale and rebuild on next access."""

        self._initialized = False

    @property
    def observation_space(self) -> spaces.Dict:
        self._ensure_initialized()
        return self._observation_space

    @property
    def action_space(self) -> spaces.Dict:
        self._ensure_initialized()
        return self._action_space

    @property
    def specs(self) -> Mapping[str, Any]:
        self._ensure_initialized()
        return self._specs

    def observation_payload(self) -> Mapping[str, Any]:
        """Return canonical entity observations for current env time step."""

        self._ensure_initialized()
        env = self.env
        t = int(env.time_step)
        endogenous_t = max(t - 1, 0)
        step_hours = self._step_hours()
        core_bundle_enabled = self._bundle_enabled(self.CORE_BUNDLE)
        community_bundle_enabled = self._bundle_enabled(self.COMMUNITY_BUNDLE)
        forecast_bundle_enabled = self._bundle_enabled(self.FORECAST_BUNDLE)
        temporal_bundle_enabled = self._bundle_enabled(self.TEMPORAL_BUNDLE)
        requires_building_electrical_metrics = core_bundle_enabled or community_bundle_enabled

        self._district_obs.fill(0.0)
        self._building_obs.fill(0.0)
        self._charger_obs.fill(0.0)
        self._ev_obs.fill(0.0)
        self._storage_obs.fill(0.0)
        self._pv_obs.fill(0.0)

        building_feature_maps: List[Mapping[str, float]] = []
        building_electrical_metrics: List[Mapping[str, float]] = []
        first_building_data: Mapping[str, Any] = {}

        for i, building in enumerate(env.buildings):
            # Exogenous signals are read at current t, while endogenous values come from
            # the latest settled transition (max(t-1, 0)) to avoid uninitialized buffers.
            data = building._get_observations_data(include_all=False)

            if i == 0:
                first_building_data = data

            values = dict(data)
            electrical_metrics = None
            if requires_building_electrical_metrics:
                electrical_metrics = self._build_electrical_building_metrics(
                    building=building,
                    endogenous_t=endogenous_t,
                    control_t=t,
                    step_hours=step_hours,
                )

            if core_bundle_enabled and electrical_metrics is not None:
                values.update(electrical_metrics)

            if community_bundle_enabled and electrical_metrics is not None:
                building_electrical_metrics.append(electrical_metrics)

            if temporal_bundle_enabled:
                values.update(
                    self._build_temporal_building_metrics(
                        building=building,
                        endogenous_t=endogenous_t,
                    )
                )

            building_feature_maps.append(values)

        district_values = dict(first_building_data)

        if forecast_bundle_enabled:
            district_values.update(self._build_forecast_district_metrics(first_building_data))

        if community_bundle_enabled:
            district_values.update(self._build_community_district_metrics(building_electrical_metrics))

        if temporal_bundle_enabled:
            district_values.update(
                self._build_temporal_district_metrics(
                    endogenous_t=endogenous_t,
                )
            )

        for j, name in enumerate(self._district_features):
            self._district_obs[0, j] = self._safe_scalar(district_values.get(name, 0.0))

        for i, values in enumerate(building_feature_maps):
            for j, name in enumerate(self._building_features):
                self._building_obs[i, j] = self._safe_scalar(values.get(name, 0.0))

        for ref in self._charger_refs:
            charger = env.buildings[ref.building_index]._charger_lookup[ref.charger_id]
            sim = charger.charger_simulation
            state = self._safe_index(sim.electric_vehicle_charger_state, t, np.nan)
            connected = charger.connected_electric_vehicle is not None and state == 1
            incoming = charger.incoming_electric_vehicle is not None and state == 2
            commanded_energy = self._safe_index(charger.past_charging_action_values_kwh, endogenous_t, 0.0)
            applied_energy = (
                self._safe_index(charger.electricity_consumption, endogenous_t, 0.0)
                if core_bundle_enabled else 0.0
            )
            required_soc = self._safe_index(sim.electric_vehicle_required_soc_departure, t, -0.1)
            departure_steps = self._safe_index(sim.electric_vehicle_departure_time, t, -1.0)

            if connected:
                connected_ev = charger.connected_electric_vehicle
                current_soc = self._safe_scalar(connected_ev.battery.soc[endogenous_t], -0.1)
                battery_capacity = self._safe_scalar(connected_ev.battery.capacity, -1.0)
                if core_bundle_enabled:
                    energy_to_required_soc = max((required_soc - current_soc) * max(battery_capacity, 0.0), 0.0)
                    avg_power_to_departure = (
                        energy_to_required_soc / max(departure_steps * step_hours, 1.0e-6)
                        if departure_steps > 0 else 0.0
                    )
                else:
                    energy_to_required_soc = 0.0
                    avg_power_to_departure = 0.0
            else:
                current_soc = -0.1
                battery_capacity = -1.0
                energy_to_required_soc = 0.0
                avg_power_to_departure = 0.0

            values = {
                "connected_state": 1.0 if connected else 0.0,
                "incoming_state": 1.0 if incoming else 0.0,
                "connected_ev_soc": current_soc,
                "connected_ev_required_soc_departure": required_soc if connected else -0.1,
                "connected_ev_battery_capacity_kwh": battery_capacity,
                "connected_ev_departure_time_step": self._safe_index(sim.electric_vehicle_departure_time, t, -1.0) if connected else -1.0,
                "incoming_ev_estimated_soc_arrival": self._safe_index(sim.electric_vehicle_estimated_soc_arrival, t, -0.1) if incoming else -0.1,
                "incoming_ev_estimated_arrival_time_step": self._safe_index(sim.electric_vehicle_estimated_arrival_time, t, -1.0) if incoming else -1.0,
                "last_charged_kwh": commanded_energy,
                "max_charging_power_kw": self._safe_scalar(charger.max_charging_power, 0.0),
                "max_discharging_power_kw": self._safe_scalar(charger.max_discharging_power, 0.0),
            }
            if core_bundle_enabled:
                values.update(
                    {
                        "commanded_power_kw": commanded_energy / step_hours,
                        "applied_power_kw": applied_energy / step_hours,
                        "applied_energy_kwh_step": applied_energy,
                        "energy_to_required_soc_kwh": energy_to_required_soc,
                        "avg_power_to_departure_kw": avg_power_to_departure,
                    }
                )
            for col, feature_name in enumerate(self._charger_features):
                self._charger_obs[ref.row, col] = self._safe_scalar(values.get(feature_name, 0.0))

        for i, ev in enumerate(env.electric_vehicles):
            soc = self._safe_scalar(ev.battery.soc[endogenous_t], 0.0)
            capacity = self._safe_scalar(ev.battery.capacity, 0.0)
            depth_of_discharge = self._safe_scalar(ev.battery.depth_of_discharge, 0.0)
            values = {
                "soc": soc,
                "battery_capacity_kwh": capacity,
                "depth_of_discharge_ratio": depth_of_discharge,
            }
            if core_bundle_enabled:
                soc_min = max(1.0 - depth_of_discharge, 0.0)
                energy_available = max((soc - soc_min) * max(capacity, 0.0), 0.0)
                energy_to_full = max((1.0 - soc) * max(capacity, 0.0), 0.0)
                values.update(
                    {
                        "soc_ratio": soc,
                        "soc_min_ratio": soc_min,
                        "soc_max_ratio": 1.0,
                        "energy_available_kwh": energy_available,
                        "energy_to_full_kwh": energy_to_full,
                    }
                )
            for col, feature_name in enumerate(self._ev_features):
                self._ev_obs[i, col] = self._safe_scalar(values.get(feature_name, 0.0))

        for ref in self._storage_refs:
            building = env.buildings[ref.building_index]
            storage = building.electrical_storage
            capacity = self._safe_scalar(getattr(storage, "capacity", 0.0), 0.0)
            nominal_power = self._safe_scalar(getattr(storage, "nominal_power", 0.0), 0.0)
            soc = self._safe_scalar(storage.soc[endogenous_t], 0.0)
            values = {
                "soc": soc,
                "capacity_kwh": capacity,
                "nominal_power_kw": nominal_power,
                "electricity_consumption_kwh": self._safe_scalar(
                    building.electrical_storage_electricity_consumption[endogenous_t],
                    0.0,
                ),
            }
            if core_bundle_enabled:
                depth_of_discharge = self._safe_scalar(getattr(storage, "depth_of_discharge", 1.0), 1.0)
                soc_min = max(1.0 - depth_of_discharge, 0.0)
                current_energy = soc * max(capacity, 0.0)
                energy_to_full = max(capacity - current_energy, 0.0)
                energy_available = max((soc - soc_min) * max(capacity, 0.0), 0.0)
                max_charge_power = (
                    self._safe_scalar(storage.get_max_input_power(), nominal_power)
                    if hasattr(storage, "get_max_input_power")
                    else nominal_power
                )
                max_discharge_power = (
                    self._safe_scalar(storage.get_max_output_power(), nominal_power)
                    if hasattr(storage, "get_max_output_power")
                    else nominal_power
                )
                values.update(
                    {
                        "electrical_storage_soc_ratio": soc,
                        "max_charge_power_kw": max_charge_power,
                        "max_discharge_power_kw": max_discharge_power,
                        "energy_to_full_kwh": energy_to_full,
                        "energy_available_kwh": energy_available,
                    }
                )
            for col, feature_name in enumerate(self._storage_features):
                self._storage_obs[ref.row, col] = self._safe_scalar(values.get(feature_name, 0.0))

        if self._pv_features:
            for ref in self._pv_refs:
                building = env.buildings[ref.building_index]
                generation_energy = abs(self._safe_index(building.solar_generation, endogenous_t, 0.0))
                values = {
                    "generation_power_kw": generation_energy / step_hours,
                    "generation_energy_kwh_step": generation_energy,
                    "installed_power_kw": self._safe_scalar(getattr(building.pv, "nominal_power", 0.0), 0.0),
                }
                for col, feature_name in enumerate(self._pv_features):
                    self._pv_obs[ref.row, col] = self._safe_scalar(values.get(feature_name, 0.0))

        self._charger_to_ev_connected.fill(-1)
        self._charger_to_ev_connected_mask.fill(0.0)
        self._charger_to_ev_incoming.fill(-1)
        self._charger_to_ev_incoming_mask.fill(0.0)
        for ref in self._charger_refs:
            charger = env.buildings[ref.building_index]._charger_lookup[ref.charger_id]
            sim = charger.charger_simulation
            state = self._safe_index(sim.electric_vehicle_charger_state, t, np.nan)
            if state == 1 and charger.connected_electric_vehicle is not None:
                ev_row = self._ev_row_by_name.get(charger.connected_electric_vehicle.name)
                if ev_row is not None:
                    self._charger_to_ev_connected[ref.row] = np.array([ref.row, ev_row], dtype=np.int32)
                    self._charger_to_ev_connected_mask[ref.row] = 1.0

            if state == 2 and charger.incoming_electric_vehicle is not None:
                ev_row = self._ev_row_by_name.get(charger.incoming_electric_vehicle.name)
                if ev_row is not None:
                    self._charger_to_ev_incoming[ref.row] = np.array([ref.row, ev_row], dtype=np.int32)
                    self._charger_to_ev_incoming_mask[ref.row] = 1.0

        return {
            "tables": {
                "district": self._district_obs,
                "building": self._building_obs,
                "charger": self._charger_obs,
                "ev": self._ev_obs,
                "storage": self._storage_obs,
                "pv": self._pv_obs,
            },
            "edges": {
                "district_to_building": self._district_to_building,
                "building_to_charger": self._building_to_charger,
                "building_to_storage": self._building_to_storage,
                "building_to_pv": self._building_to_pv,
                "charger_to_ev_connected": self._charger_to_ev_connected,
                "charger_to_ev_connected_mask": self._charger_to_ev_connected_mask,
                "charger_to_ev_incoming": self._charger_to_ev_incoming,
                "charger_to_ev_incoming_mask": self._charger_to_ev_incoming_mask,
            },
            "meta": {
                "time_step": t,
                "endogenous_time_step": endogenous_t,
                "spec_version": "entity_v1",
                "temporal_semantics": {
                    "exogenous": "t",
                    "endogenous": "t_minus_1_settled",
                },
                "topology_version": int(getattr(env, "topology_version", 0)),
            },
        }

    def parse_actions(self, actions: Any) -> List[Mapping[str, float]]:
        """Parse canonical entity action payload into per-building action dicts."""

        self._ensure_initialized()
        if self._is_flat_like(actions):
            vectors = self._parse_flat_like(actions)
            return self._map_vectors_to_action_dicts(vectors)

        if not isinstance(actions, Mapping):
            raise AssertionError("Entity interface expects mapping payload with action tables.")

        tables = actions.get("tables", actions)
        building_table = tables.get("building")
        charger_table = tables.get("charger")

        building_table = self._to_2d_array(building_table, rows=len(self._building_ids), cols=len(self._building_action_features))
        charger_table = self._to_2d_array(charger_table, rows=len(self._charger_refs), cols=len(self._charger_action_features))
        building_overrides, charger_overrides = self._resolve_map_overrides(actions)

        vectors: List[List[float]] = []
        for building_idx, building in enumerate(self.env.buildings):
            building_values = []
            per_building_map = building_overrides.get(building.name, {})

            for action_name, low, high in zip(building.active_actions, building.action_space.low, building.action_space.high):
                value = 0.0
                if action_name in self._building_action_col_by_name:
                    col = self._building_action_col_by_name[action_name]
                    if building_table is not None:
                        value = building_table[building_idx, col]
                    if isinstance(per_building_map, Mapping) and action_name in per_building_map:
                        value = self._safe_scalar(per_building_map[action_name], float(value))

                elif action_name.startswith("electric_vehicle_storage_"):
                    row = self._charger_row_by_ev_action_name.get((building_idx, action_name))
                    if row is not None and row >= 0 and charger_table is not None and charger_table.shape[1] > 0:
                        value = charger_table[row, 0]
                    charger_ref = self._charger_refs[row] if row is not None and row >= 0 else None
                    if charger_ref is not None:
                        charger_payload = charger_overrides.get(charger_ref.global_id, {})
                        if isinstance(charger_payload, Mapping) and "electric_vehicle_storage" in charger_payload:
                            value = self._safe_scalar(charger_payload["electric_vehicle_storage"], float(value))

                elif "washing_machine" in action_name and isinstance(per_building_map, Mapping):
                    if action_name in per_building_map:
                        value = self._safe_scalar(per_building_map[action_name])

                value = float(np.clip(value, low, high))
                building_values.append(value)

            vectors.append(building_values)

        return self._map_vectors_to_action_dicts(vectors)

    def _resolve_map_overrides(self, actions: Mapping[str, Any]) -> Tuple[Mapping[str, Mapping[str, Any]], Mapping[str, Mapping[str, Any]]]:
        raw_map = actions.get("map")
        if raw_map is None:
            raw_map = {}
            legacy_building_map = actions.get("building") or {}
            legacy_charger_map = actions.get("charger") or {}
            if isinstance(legacy_building_map, Mapping):
                for key, payload in legacy_building_map.items():
                    raw_map[f"building:{key}"] = payload
            if isinstance(legacy_charger_map, Mapping):
                for key, payload in legacy_charger_map.items():
                    raw_map[f"charger:{key}"] = payload

        if not isinstance(raw_map, Mapping):
            raise AssertionError("Entity action 'map' must be a mapping keyed by canonical entity ids.")

        building_ids = set(self._building_ids)
        charger_global_ids = {ref.global_id for ref in self._charger_refs}
        charger_raw_to_global = {ref.charger_id: ref.global_id for ref in self._charger_refs}
        building_overrides: Dict[str, Mapping[str, Any]] = {}
        charger_overrides: Dict[str, Mapping[str, Any]] = {}

        for raw_id, payload in raw_map.items():
            if not isinstance(payload, Mapping):
                raise AssertionError(f"Entity action map entry for '{raw_id}' must be a mapping of action values.")

            raw_key = str(raw_id)
            entity_kind, entity_id = self._split_map_id(raw_key)

            if entity_kind is None:
                if raw_key in building_ids:
                    entity_kind, entity_id = "building", raw_key
                elif raw_key in charger_global_ids:
                    entity_kind, entity_id = "charger", raw_key
                elif raw_key in charger_raw_to_global:
                    entity_kind, entity_id = "charger", charger_raw_to_global[raw_key]
                else:
                    raise AssertionError(f"Unknown entity id '{raw_key}' in action map.")

            if entity_kind == "building":
                if entity_id not in building_ids:
                    raise AssertionError(f"Unknown building id '{entity_id}' in action map.")
                building_overrides[entity_id] = dict(payload)
                continue

            if entity_kind == "charger":
                if entity_id in charger_raw_to_global:
                    entity_id = charger_raw_to_global[entity_id]
                if entity_id not in charger_global_ids:
                    raise AssertionError(f"Unknown charger id '{entity_id}' in action map.")
                charger_overrides[entity_id] = dict(payload)
                continue

            raise AssertionError(f"Unknown entity kind '{entity_kind}' in action map id '{raw_key}'.")

        self._validate_override_keys(building_overrides, charger_overrides)
        return building_overrides, charger_overrides

    @staticmethod
    def _split_map_id(raw_key: str) -> Tuple[Optional[str], Optional[str]]:
        if ":" not in raw_key:
            return None, None
        entity_kind, entity_id = raw_key.split(":", 1)
        entity_kind = entity_kind.strip().lower()
        entity_id = entity_id.strip()
        if entity_kind == "" or entity_id == "":
            return None, None
        return entity_kind, entity_id

    def _validate_override_keys(
        self,
        building_overrides: Mapping[str, Mapping[str, Any]],
        charger_overrides: Mapping[str, Mapping[str, Any]],
    ):
        for building in self.env.buildings:
            payload = building_overrides.get(building.name, {})
            if not payload:
                continue
            unknown = [key for key in payload.keys() if key not in set(building.active_actions)]
            if unknown:
                raise AssertionError(
                    f"Unknown building action keys for '{building.name}': {sorted(unknown)}."
                )

        for charger_global_id, payload in charger_overrides.items():
            unknown = [key for key in payload.keys() if key != "electric_vehicle_storage"]
            if unknown:
                raise AssertionError(
                    f"Unknown charger action keys for '{charger_global_id}': {sorted(unknown)}."
                )

    def _parse_flat_like(self, actions: Any) -> List[List[float]]:
        env = self.env

        def _is_scalar(value: Any) -> bool:
            return bool(np.isscalar(value))

        def _to_vector(value: Any, *, context: str) -> List[float]:
            if isinstance(value, np.ndarray):
                if value.ndim == 1:
                    return value.tolist()
                if value.ndim == 2 and value.shape[0] == 1:
                    return value[0].tolist()
                raise AssertionError(f"{context} must be a 1D action vector.")

            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    return []
                if all(_is_scalar(v) for v in value):
                    return list(value)
                if len(value) == 1 and isinstance(value[0], (list, tuple, np.ndarray)):
                    return _to_vector(value[0], context=context)
                raise AssertionError(f"{context} must be a 1D action vector.")

            raise AssertionError(f"{context} must be a 1D action vector.")

        vectors: List[List[float]]
        if env.central_agent:
            merged = _to_vector(actions, context="central_agent actions")
            expected = env._expected_central_action_count
            if len(merged) != expected:
                raise AssertionError(f"Expected {expected} actions but {len(merged)} were parsed to env.step.")
            vectors = []
            cursor = 0
            for building in env.buildings:
                size = building.action_space.shape[0]
                vectors.append(merged[cursor:cursor + size])
                cursor += size
        else:
            if isinstance(actions, np.ndarray):
                if actions.ndim != 2:
                    raise AssertionError("Expected one action vector per building when central_agent=False.")
                vectors = [row.tolist() for row in actions]
            elif isinstance(actions, (list, tuple)):
                vectors = [_to_vector(row, context=f"building action vector at index {idx}") for idx, row in enumerate(actions)]
            else:
                raise AssertionError("Expected one action vector per building when central_agent=False.")

            expected = len(env.buildings)
            if len(vectors) != expected:
                raise AssertionError(f"Expected {expected} building action vectors but {len(vectors)} were provided.")

        for building, vector in zip(env.buildings, vectors):
            expected = building.action_space.shape[0]
            if len(vector) != expected:
                raise AssertionError(f"Expected {expected} for {building.name} but {len(vector)} actions were provided.")

        return vectors

    def _map_vectors_to_action_dicts(self, vectors: Sequence[Sequence[float]]) -> List[Mapping[str, float]]:
        parsed_actions = []
        active_actions = self.env._active_actions_cache

        for i, _building in enumerate(self.env.buildings):
            action_dict = {}
            electric_vehicle_actions = {}
            washing_machine_actions = {}

            for action_name, action in zip(active_actions[i], vectors[i]):
                if "electric_vehicle_storage" in action_name:
                    charger_id = action_name.replace("electric_vehicle_storage_", "")
                    electric_vehicle_actions[charger_id] = action
                elif "washing_machine" in action_name:
                    washing_machine_actions[action_name] = action
                else:
                    action_dict[f"{action_name}_action"] = action

            if electric_vehicle_actions:
                action_dict["electric_vehicle_storage_actions"] = electric_vehicle_actions
            if washing_machine_actions:
                action_dict["washing_machine_actions"] = washing_machine_actions

            parsed_actions.append(action_dict)

        return parsed_actions

    def _resolve_observation_bundles(self) -> Mapping[str, bool]:
        schema = self.env.schema if isinstance(self.env.schema, Mapping) else {}
        raw = schema.get("observation_bundles", {}) if isinstance(schema, Mapping) else {}
        if not isinstance(raw, Mapping):
            raw = {}

        resolved = dict(self.DEFAULT_OBSERVATION_BUNDLES)
        for bundle_name in resolved:
            value = raw.get(bundle_name, False)
            if isinstance(value, Mapping):
                value = value.get("active", False)
            if isinstance(value, str):
                resolved[bundle_name] = value.strip().lower() in {"1", "true", "yes", "on"}
            else:
                resolved[bundle_name] = bool(value)

        return resolved

    def _bundle_enabled(self, bundle_name: str) -> bool:
        return bool(self._observation_bundles.get(bundle_name, False))

    def _step_hours(self) -> float:
        return max(float(self.env.seconds_per_time_step) / 3600.0, 1.0e-6)

    @staticmethod
    def _append_unique(target: List[str], values: Sequence[str]):
        for value in values:
            if value not in target:
                target.append(value)

    @staticmethod
    def _series_mean(values: Sequence[float], indices: Sequence[int], fallback: float) -> float:
        filtered: List[float] = []
        for idx in indices:
            if idx < 0:
                continue
            try:
                value = float(values[idx])
            except Exception:
                continue
            if np.isfinite(value):
                filtered.append(value)
        if len(filtered) == 0:
            return float(fallback)
        return float(np.mean(filtered))

    @classmethod
    def _infer_unit(cls, name: str) -> str:
        key = str(name).lower()
        if key.endswith("_power_kw") or "headroom_kw" in key:
            return "kw"
        if key.endswith("_kwh_step"):
            return "kwh_step"
        if key.endswith("_capacity_kwh") or key.endswith("_kwh"):
            return "kwh"
        if key.endswith("_soc_ratio") or key.endswith("_ratio") or key.endswith("_soc"):
            return "ratio"
        if key.endswith("_time_step"):
            return "time_step"
        if key.endswith("_count"):
            return "count"
        if key in {"hour", "day_type", "month", "minutes"}:
            return "index"
        if "consumption" in key or "generation" in key or "demand" in key:
            return "kwh"
        return "scalar"

    def _feature_metadata_for_table(self, table_name: str, features: Sequence[str]) -> Mapping[str, Mapping[str, Any]]:
        bundle_map = self._table_feature_bundle.get(table_name, {})
        legacy_map = self._table_feature_legacy.get(table_name, {})
        metadata: Dict[str, Mapping[str, Any]] = {}

        for feature in features:
            metadata[feature] = {
                "unit": self._infer_unit(feature),
                "bundle": bundle_map.get(feature, "legacy"),
                "legacy": bool(legacy_map.get(feature, True)),
            }

        return metadata

    def _build_electrical_building_metrics(
        self,
        *,
        building,
        endogenous_t: int,
        control_t: int,
        step_hours: float,
    ) -> Mapping[str, float]:
        net_energy = self._safe_index(building.net_electricity_consumption, endogenous_t, 0.0)
        import_energy = max(net_energy, 0.0)
        export_energy = max(-net_energy, 0.0)
        pv_energy = abs(self._safe_index(building.solar_generation, endogenous_t, 0.0))
        bess_energy = self._safe_index(building.electrical_storage_electricity_consumption, endogenous_t, 0.0)
        ev_energy = self._safe_index(building.chargers_electricity_consumption, endogenous_t, 0.0)
        cooling = self._safe_index(building.cooling_electricity_consumption, endogenous_t, 0.0)
        heating = self._safe_index(building.heating_electricity_consumption, endogenous_t, 0.0)
        dhw = self._safe_index(building.dhw_electricity_consumption, endogenous_t, 0.0)
        non_shiftable = self._safe_index(building.non_shiftable_load_electricity_consumption, endogenous_t, 0.0)
        washing = self._safe_index(building.washing_machines_electricity_consumption, endogenous_t, 0.0)
        load_energy = max(cooling + heating + dhw + non_shiftable + washing + ev_energy, 0.0)

        constraint_state = getattr(building, "_charging_constraints_state", {}) or {}
        building_headroom = constraint_state.get("building_headroom_kw")
        building_export_headroom = constraint_state.get("building_export_headroom_kw")
        phase_headroom = constraint_state.get("phase_headroom_kw") or {}
        phase_export_headroom = constraint_state.get("phase_export_headroom_kw") or {}
        phase_headroom_sum = sum(
            self._safe_scalar(value, 0.0)
            for value in phase_headroom.values()
            if value is not None
        )
        phase_export_headroom_sum = sum(
            self._safe_scalar(value, 0.0)
            for value in phase_export_headroom.values()
            if value is not None
        )

        metrics = {
            "net_power_kw": net_energy / step_hours,
            "net_energy_kwh_step": net_energy,
            "import_power_kw": import_energy / step_hours,
            "import_energy_kwh_step": import_energy,
            "export_power_kw": export_energy / step_hours,
            "export_energy_kwh_step": export_energy,
            "load_power_kw": load_energy / step_hours,
            "load_energy_kwh_step": load_energy,
            "pv_power_kw": pv_energy / step_hours,
            "pv_energy_kwh_step": pv_energy,
            "bess_power_kw": bess_energy / step_hours,
            "bess_energy_kwh_step": bess_energy,
            "ev_charging_power_kw": ev_energy / step_hours,
            "ev_charging_energy_kwh_step": ev_energy,
            "electrical_storage_soc_ratio": self._safe_index(
                building.electrical_storage.soc,
                endogenous_t,
                0.0,
            ),
            "_building_headroom_kw": self._safe_scalar(building_headroom, np.nan),
            "_building_export_headroom_kw": self._safe_scalar(building_export_headroom, np.nan),
            "_phase_headroom_kw": phase_headroom_sum,
            "_phase_export_headroom_kw": phase_export_headroom_sum,
            "_control_time_step": float(control_t),
        }
        return metrics

    def _build_temporal_building_metrics(self, *, building, endogenous_t: int) -> Mapping[str, float]:
        prev_1 = max(endogenous_t - 1, 0)
        net_prev_1 = self._safe_index(building.net_electricity_consumption, prev_1, 0.0)
        import_prev_1 = max(net_prev_1, 0.0)
        export_prev_1 = max(-net_prev_1, 0.0)
        indices = list(range(max(endogenous_t - 3, 0), endogenous_t))
        if len(indices) == 0:
            indices = [prev_1]

        net_prev_3_mean = self._series_mean(
            building.net_electricity_consumption,
            indices,
            fallback=net_prev_1,
        )

        return {
            "net_energy_prev_1_kwh_step": net_prev_1,
            "net_energy_prev_3_mean_kwh_step": net_prev_3_mean,
            "import_energy_prev_1_kwh_step": import_prev_1,
            "export_energy_prev_1_kwh_step": export_prev_1,
        }

    def _build_forecast_district_metrics(self, first_building_data: Mapping[str, Any]) -> Mapping[str, float]:
        forecasts: Dict[str, float] = {}
        for key, value in first_building_data.items():
            if "_predicted_" not in str(key):
                continue
            forecasts[str(key)] = self._safe_scalar(value, 0.0)
        return forecasts

    def _build_community_district_metrics(
        self,
        building_electrical_metrics: Sequence[Mapping[str, float]],
    ) -> Mapping[str, float]:
        step_hours = self._step_hours()
        net_energy = sum(self._safe_scalar(m.get("net_energy_kwh_step", 0.0), 0.0) for m in building_electrical_metrics)
        import_energy = sum(self._safe_scalar(m.get("import_energy_kwh_step", 0.0), 0.0) for m in building_electrical_metrics)
        export_energy = sum(self._safe_scalar(m.get("export_energy_kwh_step", 0.0), 0.0) for m in building_electrical_metrics)
        pv_energy = sum(self._safe_scalar(m.get("pv_energy_kwh_step", 0.0), 0.0) for m in building_electrical_metrics)
        bess_energy = sum(self._safe_scalar(m.get("bess_energy_kwh_step", 0.0), 0.0) for m in building_electrical_metrics)
        ev_energy = sum(self._safe_scalar(m.get("ev_charging_energy_kwh_step", 0.0), 0.0) for m in building_electrical_metrics)

        building_headroom = [
            self._safe_scalar(m.get("_building_headroom_kw"), np.nan) for m in building_electrical_metrics
        ]
        building_export_headroom = [
            self._safe_scalar(m.get("_building_export_headroom_kw"), np.nan) for m in building_electrical_metrics
        ]
        community_building_headroom = sum(v for v in building_headroom if np.isfinite(v))
        community_building_export_headroom = sum(v for v in building_export_headroom if np.isfinite(v))
        community_phase_headroom = sum(
            self._safe_scalar(m.get("_phase_headroom_kw", 0.0), 0.0) for m in building_electrical_metrics
        )
        community_phase_export_headroom = sum(
            self._safe_scalar(m.get("_phase_export_headroom_kw", 0.0), 0.0) for m in building_electrical_metrics
        )

        return {
            "community_net_power_kw": net_energy / step_hours,
            "community_net_energy_kwh_step": net_energy,
            "community_import_power_kw": import_energy / step_hours,
            "community_import_energy_kwh_step": import_energy,
            "community_export_power_kw": export_energy / step_hours,
            "community_export_energy_kwh_step": export_energy,
            "community_pv_power_kw": pv_energy / step_hours,
            "community_pv_energy_kwh_step": pv_energy,
            "community_bess_power_kw": bess_energy / step_hours,
            "community_bess_energy_kwh_step": bess_energy,
            "community_ev_power_kw": ev_energy / step_hours,
            "community_ev_energy_kwh_step": ev_energy,
            "community_building_headroom_kw": community_building_headroom,
            "community_building_export_headroom_kw": community_building_export_headroom,
            "community_phase_headroom_kw": community_phase_headroom,
            "community_phase_export_headroom_kw": community_phase_export_headroom,
            "active_buildings_count": float(len(self.env.buildings)),
            "active_chargers_count": float(sum(len(b.electric_vehicle_chargers or []) for b in self.env.buildings)),
            "active_evs_count": float(len(self.env.electric_vehicles)),
            "topology_version": float(getattr(self.env, "topology_version", 0)),
        }

    def _build_temporal_district_metrics(
        self,
        *,
        endogenous_t: int,
    ) -> Mapping[str, float]:
        prev_1 = max(endogenous_t - 1, 0)
        community_net_prev_1 = sum(
            self._safe_index(building.net_electricity_consumption, prev_1, 0.0)
            for building in self.env.buildings
        )
        indices = list(range(max(endogenous_t - 3, 0), endogenous_t))
        if len(indices) == 0:
            indices = [prev_1]

        aggregate_window = []
        for idx in indices:
            aggregate_window.append(
                sum(self._safe_index(building.net_electricity_consumption, idx, 0.0) for building in self.env.buildings)
            )

        community_net_prev_3_mean = float(np.mean(aggregate_window)) if len(aggregate_window) > 0 else community_net_prev_1
        return {
            "community_net_prev_1_kwh_step": community_net_prev_1,
            "community_net_prev_3_mean_kwh_step": community_net_prev_3_mean,
        }

    def _build_entity_layout(self):
        env = self.env
        self._building_ids = [b.name for b in env.buildings]
        self._ev_ids = [ev.name for ev in env.electric_vehicles]
        self._ev_row_by_name = {name: i for i, name in enumerate(self._ev_ids)}

        self._charger_refs: List[ChargerRef] = []
        row = 0
        for b_idx, building in enumerate(env.buildings):
            for charger in building.electric_vehicle_chargers or []:
                global_id = f"{building.name}/{charger.charger_id}"
                self._charger_refs.append(
                    ChargerRef(
                        row=row,
                        building_index=b_idx,
                        building_name=building.name,
                        charger_id=charger.charger_id,
                        global_id=global_id,
                    )
                )
                row += 1

        self._storage_refs: List[StorageRef] = []
        storage_row = 0
        for b_idx, building in enumerate(env.buildings):
            if self._has_electrical_storage(building):
                storage_id = "electrical_storage"
                global_id = f"{building.name}/{storage_id}"
                self._storage_refs.append(
                    StorageRef(
                        row=storage_row,
                        building_index=b_idx,
                        building_name=building.name,
                        storage_id=storage_id,
                        global_id=global_id,
                    )
                )
                storage_row += 1

        self._pv_refs: List[PVRef] = []
        pv_row = 0
        for b_idx, building in enumerate(env.buildings):
            if not self._has_pv_asset(building):
                continue
            pv_id = "pv"
            global_id = f"{building.name}/{pv_id}"
            self._pv_refs.append(
                PVRef(
                    row=pv_row,
                    building_index=b_idx,
                    building_name=building.name,
                    pv_id=pv_id,
                    global_id=global_id,
                )
            )
            pv_row += 1

        self._charger_row_by_global_id = {ref.global_id: ref.row for ref in self._charger_refs}
        self._charger_row_by_raw_id = {ref.charger_id: ref.row for ref in self._charger_refs}
        self._charger_row_by_building_and_raw_id = {
            (ref.building_index, ref.charger_id): ref.row for ref in self._charger_refs
        }
        self._charger_row_by_ev_action_name = {}

        shared = set(env.shared_observations)
        self._district_features = []
        if len(env.buildings) > 0:
            for name in env.buildings[0].active_observations:
                if name in shared and not self._is_entity_specific_observation(name):
                    self._district_features.append(name)

        self._building_features = []
        for building in env.buildings:
            for name in building.active_observations:
                if name in shared or self._is_entity_specific_observation(name):
                    continue
                if name not in self._building_features:
                    self._building_features.append(name)

        self._table_feature_bundle: Dict[str, Dict[str, str]] = {
            "district": {},
            "building": {},
            "charger": {},
            "ev": {},
            "storage": {},
            "pv": {},
        }
        self._table_feature_legacy: Dict[str, Dict[str, bool]] = {
            "district": {},
            "building": {},
            "charger": {},
            "ev": {},
            "storage": {},
            "pv": {},
        }

        for name in self._district_features:
            bundle_name = self.FORECAST_BUNDLE if "_predicted_" in str(name) else "legacy"
            self._table_feature_bundle["district"][name] = bundle_name
            self._table_feature_legacy["district"][name] = True

        for name in self._building_features:
            self._table_feature_bundle["building"][name] = "legacy"
            self._table_feature_legacy["building"][name] = True

        if self._bundle_enabled(self.CORE_BUNDLE):
            self._append_unique(self._building_features, self.CORE_BUILDING_EXTRA_FEATURES)
            for name in self.CORE_BUILDING_EXTRA_FEATURES:
                self._table_feature_bundle["building"][name] = self.CORE_BUNDLE
                self._table_feature_legacy["building"][name] = False

        if self._bundle_enabled(self.TEMPORAL_BUNDLE):
            self._append_unique(self._building_features, self.TEMPORAL_BUILDING_FEATURES)
            for name in self.TEMPORAL_BUILDING_FEATURES:
                self._table_feature_bundle["building"][name] = self.TEMPORAL_BUNDLE
                self._table_feature_legacy["building"][name] = False

        if self._bundle_enabled(self.COMMUNITY_BUNDLE):
            self._append_unique(self._district_features, self.COMMUNITY_DISTRICT_EXTRA_FEATURES)
            for name in self.COMMUNITY_DISTRICT_EXTRA_FEATURES:
                self._table_feature_bundle["district"][name] = self.COMMUNITY_BUNDLE
                self._table_feature_legacy["district"][name] = False

        if self._bundle_enabled(self.TEMPORAL_BUNDLE):
            self._append_unique(self._district_features, self.TEMPORAL_DISTRICT_FEATURES)
            for name in self.TEMPORAL_DISTRICT_FEATURES:
                self._table_feature_bundle["district"][name] = self.TEMPORAL_BUNDLE
                self._table_feature_legacy["district"][name] = False

        if self._bundle_enabled(self.FORECAST_BUNDLE) and len(env.buildings) > 0:
            current = env.buildings[0]._get_observations_data(include_all=False)
            forecast_keys = [key for key in current if "_predicted_" in str(key)]
            self._append_unique(self._district_features, forecast_keys)
            for name in forecast_keys:
                self._table_feature_bundle["district"][name] = self.FORECAST_BUNDLE
                self._table_feature_legacy["district"][name] = True

        self._charger_features = list(self.LEGACY_CHARGER_FEATURES)
        for name in self.LEGACY_CHARGER_FEATURES:
            self._table_feature_bundle["charger"][name] = "legacy"
            self._table_feature_legacy["charger"][name] = True
        if self._bundle_enabled(self.CORE_BUNDLE):
            self._append_unique(self._charger_features, self.EXTRA_CHARGER_FEATURES)
            for name in self.EXTRA_CHARGER_FEATURES:
                self._table_feature_bundle["charger"][name] = self.CORE_BUNDLE
                self._table_feature_legacy["charger"][name] = False

        self._ev_features = list(self.LEGACY_EV_FEATURES)
        for name in self.LEGACY_EV_FEATURES:
            self._table_feature_bundle["ev"][name] = "legacy"
            self._table_feature_legacy["ev"][name] = True
        if self._bundle_enabled(self.CORE_BUNDLE):
            self._append_unique(self._ev_features, self.EXTRA_EV_FEATURES)
            for name in self.EXTRA_EV_FEATURES:
                self._table_feature_bundle["ev"][name] = self.CORE_BUNDLE
                self._table_feature_legacy["ev"][name] = False

        self._storage_features = list(self.LEGACY_STORAGE_FEATURES)
        for name in self.LEGACY_STORAGE_FEATURES:
            self._table_feature_bundle["storage"][name] = "legacy"
            self._table_feature_legacy["storage"][name] = True
        if self._bundle_enabled(self.CORE_BUNDLE):
            self._append_unique(self._storage_features, self.EXTRA_STORAGE_FEATURES)
            for name in self.EXTRA_STORAGE_FEATURES:
                self._table_feature_bundle["storage"][name] = self.CORE_BUNDLE
                self._table_feature_legacy["storage"][name] = False

        self._pv_features = list(self.PV_FEATURES) if self._bundle_enabled(self.CORE_BUNDLE) else []
        for name in self._pv_features:
            self._table_feature_bundle["pv"][name] = self.CORE_BUNDLE
            self._table_feature_legacy["pv"][name] = False

        self._building_action_features = []
        self._charger_action_features = []
        for b_idx, building in enumerate(env.buildings):
            for action_name in building.active_actions:
                if action_name.startswith("electric_vehicle_storage_"):
                    charger_id = action_name.replace("electric_vehicle_storage_", "")
                    resolved_row = self._charger_row_by_building_and_raw_id.get((b_idx, charger_id))
                    if resolved_row is None:
                        resolved_row = self._charger_row_by_raw_id.get(charger_id)
                    if resolved_row is not None:
                        self._charger_row_by_ev_action_name[(b_idx, action_name)] = resolved_row
                    if "electric_vehicle_storage" not in self._charger_action_features:
                        self._charger_action_features.append("electric_vehicle_storage")
                elif "washing_machine" in action_name:
                    continue
                elif action_name not in self._building_action_features:
                    self._building_action_features.append(action_name)

        self._building_action_col_by_name = {name: i for i, name in enumerate(self._building_action_features)}
        self._charger_action_col_by_name = {name: i for i, name in enumerate(self._charger_action_features)}

        self._district_obs = np.zeros((1, len(self._district_features)), dtype=np.float32)
        self._building_obs = np.zeros((len(self._building_ids), len(self._building_features)), dtype=np.float32)
        self._charger_obs = np.zeros((len(self._charger_refs), len(self._charger_features)), dtype=np.float32)
        self._ev_obs = np.zeros((len(self._ev_ids), len(self._ev_features)), dtype=np.float32)
        self._storage_obs = np.zeros((len(self._storage_refs), len(self._storage_features)), dtype=np.float32)
        self._pv_obs = np.zeros((len(self._pv_refs), len(self._pv_features)), dtype=np.float32)

        self._district_to_building = np.zeros((len(self._building_ids), 2), dtype=np.int32)
        if len(self._building_ids) > 0:
            self._district_to_building[:, 1] = np.arange(len(self._building_ids), dtype=np.int32)

        self._building_to_charger = np.full((len(self._charger_refs), 2), -1, dtype=np.int32)
        for ref in self._charger_refs:
            self._building_to_charger[ref.row] = np.array([ref.building_index, ref.row], dtype=np.int32)

        self._building_to_storage = np.full((len(self._storage_refs), 2), -1, dtype=np.int32)
        for ref in self._storage_refs:
            self._building_to_storage[ref.row] = np.array([ref.building_index, ref.row], dtype=np.int32)

        self._building_to_pv = np.full((len(self._pv_refs), 2), -1, dtype=np.int32)
        for ref in self._pv_refs:
            self._building_to_pv[ref.row] = np.array([ref.building_index, ref.row], dtype=np.int32)

        self._charger_to_ev_connected = np.full((len(self._charger_refs), 2), -1, dtype=np.int32)
        self._charger_to_ev_connected_mask = np.zeros((len(self._charger_refs),), dtype=np.float32)
        self._charger_to_ev_incoming = np.full((len(self._charger_refs), 2), -1, dtype=np.int32)
        self._charger_to_ev_incoming_mask = np.zeros((len(self._charger_refs),), dtype=np.float32)

    def _build_spaces(self):
        district_low, district_high = self._observation_bounds_for_features(self._district_features, owner="district")
        building_low, building_high = self._observation_bounds_for_features(self._building_features, owner="building", per_building=True)
        charger_low, charger_high = self._charger_observation_bounds()
        ev_low, ev_high = self._ev_observation_bounds()
        storage_low, storage_high = self._storage_observation_bounds()
        pv_low, pv_high = self._pv_observation_bounds()

        self._observation_space = spaces.Dict(
            {
                "tables": spaces.Dict(
                    {
                        "district": spaces.Box(low=district_low, high=district_high, dtype=np.float32),
                        "building": spaces.Box(low=building_low, high=building_high, dtype=np.float32),
                        "charger": spaces.Box(low=charger_low, high=charger_high, dtype=np.float32),
                        "ev": spaces.Box(low=ev_low, high=ev_high, dtype=np.float32),
                        "storage": spaces.Box(low=storage_low, high=storage_high, dtype=np.float32),
                        "pv": spaces.Box(low=pv_low, high=pv_high, dtype=np.float32),
                    }
                ),
                "edges": spaces.Dict(
                    {
                        "district_to_building": spaces.Box(
                            low=np.zeros((len(self._building_ids), 2), dtype=np.int32),
                            high=np.maximum(np.array([[0, max(len(self._building_ids) - 1, 0)]], dtype=np.int32), 0).repeat(len(self._building_ids), axis=0),
                            dtype=np.int32,
                        ),
                        "building_to_charger": spaces.Box(
                            low=np.full((len(self._charger_refs), 2), -1, dtype=np.int32),
                            high=np.array(
                                [[max(len(self._building_ids) - 1, 0), max(len(self._charger_refs) - 1, 0)]],
                                dtype=np.int32,
                            ).repeat(len(self._charger_refs), axis=0),
                            dtype=np.int32,
                        ),
                        "building_to_storage": spaces.Box(
                            low=np.full((len(self._storage_refs), 2), -1, dtype=np.int32),
                            high=np.array(
                                [[max(len(self._building_ids) - 1, 0), max(len(self._storage_refs) - 1, 0)]],
                                dtype=np.int32,
                            ).repeat(len(self._storage_refs), axis=0),
                            dtype=np.int32,
                        ),
                        "building_to_pv": spaces.Box(
                            low=np.full((len(self._pv_refs), 2), -1, dtype=np.int32),
                            high=np.array(
                                [[max(len(self._building_ids) - 1, 0), max(len(self._pv_refs) - 1, 0)]],
                                dtype=np.int32,
                            ).repeat(len(self._pv_refs), axis=0),
                            dtype=np.int32,
                        ),
                        "charger_to_ev_connected": spaces.Box(
                            low=np.full((len(self._charger_refs), 2), -1, dtype=np.int32),
                            high=np.array(
                                [[max(len(self._charger_refs) - 1, 0), max(len(self._ev_ids) - 1, 0)]],
                                dtype=np.int32,
                            ).repeat(len(self._charger_refs), axis=0),
                            dtype=np.int32,
                        ),
                        "charger_to_ev_connected_mask": spaces.Box(
                            low=np.zeros((len(self._charger_refs),), dtype=np.float32),
                            high=np.ones((len(self._charger_refs),), dtype=np.float32),
                            dtype=np.float32,
                        ),
                        "charger_to_ev_incoming": spaces.Box(
                            low=np.full((len(self._charger_refs), 2), -1, dtype=np.int32),
                            high=np.array(
                                [[max(len(self._charger_refs) - 1, 0), max(len(self._ev_ids) - 1, 0)]],
                                dtype=np.int32,
                            ).repeat(len(self._charger_refs), axis=0),
                            dtype=np.int32,
                        ),
                        "charger_to_ev_incoming_mask": spaces.Box(
                            low=np.zeros((len(self._charger_refs),), dtype=np.float32),
                            high=np.ones((len(self._charger_refs),), dtype=np.float32),
                            dtype=np.float32,
                        ),
                    }
                ),
            }
        )

        building_action_low, building_action_high = self._building_action_bounds()
        charger_action_low, charger_action_high = self._charger_action_bounds()
        self._action_space = spaces.Dict(
            {
                "tables": spaces.Dict(
                    {
                        "building": spaces.Box(low=building_action_low, high=building_action_high, dtype=np.float32),
                        "charger": spaces.Box(low=charger_action_low, high=charger_action_high, dtype=np.float32),
                    }
                )
            }
        )

    def _build_specs(self):
        def _units_for(names: Sequence[str]) -> List[str]:
            return [self._infer_unit(name) for name in names]

        def _table_spec(table_name: str, ids: Sequence[str], features: Sequence[str]) -> Mapping[str, Any]:
            feature_metadata = self._feature_metadata_for_table(table_name, features)
            return {
                "ids": list(ids),
                "features": list(features),
                "units": _units_for(features),
                "feature_metadata": feature_metadata,
            }

        self._specs = {
            "version": "entity_v1",
            "temporal_semantics": {
                "exogenous": "t",
                "endogenous": "t_minus_1_settled",
                "event_boundary": "events_at_k_apply_after_transition_k_minus_1_to_k_before_observation_k",
            },
            "normalization": {
                "simulator_applies_normalization": False,
                "policy": "external_running_stats",
                "encoding": {
                    "ids": "stable_canonical_ids",
                    "categorical": "numeric_or_binary_from_dataset",
                    "time": "raw_indices_unwrapped",
                },
                "dynamic_topology": {
                    "variable_size_tables": bool(getattr(self.env, "topology_mode", "static") == "dynamic"),
                    "stable_ids": True,
                    "relation_masks": [
                        "charger_to_ev_connected_mask",
                        "charger_to_ev_incoming_mask",
                    ],
                    "running_stats_recommendation": "maintain_per_feature_stats_externally_and_ignore_inactive_ids",
                },
            },
            "observation_bundles": {
                bundle_name: {"active": bool(active)}
                for bundle_name, active in self._observation_bundles.items()
            },
            "tables": {
                "district": _table_spec("district", ["district_0"], self._district_features),
                "building": _table_spec("building", self._building_ids, self._building_features),
                "charger": _table_spec("charger", [ref.global_id for ref in self._charger_refs], self._charger_features),
                "ev": _table_spec("ev", self._ev_ids, self._ev_features),
                "storage": _table_spec("storage", [ref.global_id for ref in self._storage_refs], self._storage_features),
                "pv": _table_spec("pv", [ref.global_id for ref in self._pv_refs], self._pv_features),
            },
            "actions": {
                "building": {
                    "ids": list(self._building_ids),
                    "features": list(self._building_action_features),
                    "units": _units_for(self._building_action_features),
                },
                "charger": {
                    "ids": [ref.global_id for ref in self._charger_refs],
                    "features": list(self._charger_action_features),
                    "units": _units_for(self._charger_action_features),
                },
            },
            "edges": {
                "district_to_building": {
                    "source": "district",
                    "target": "building",
                },
                "building_to_charger": {
                    "source": "building",
                    "target": "charger",
                },
                "building_to_storage": {
                    "source": "building",
                    "target": "storage",
                },
                "building_to_pv": {
                    "source": "building",
                    "target": "pv",
                },
                "charger_to_ev_connected": {
                    "source": "charger",
                    "target": "ev",
                },
                "charger_to_ev_incoming": {
                    "source": "charger",
                    "target": "ev",
                },
            },
            "topology": {
                "mode": getattr(self.env, "topology_mode", "static"),
                "version": int(getattr(self.env, "topology_version", 0)),
                "active_ids": {
                    "building": list(self._building_ids),
                    "ev": list(self._ev_ids),
                    "charger": [ref.global_id for ref in self._charger_refs],
                    "storage": [ref.global_id for ref in self._storage_refs],
                    "pv": [ref.global_id for ref in self._pv_refs],
                },
                "lifecycle_fields": ["born_at", "removed_at", "active"],
                "member_lifecycle": getattr(self.env, "topology_member_lifecycle", {}),
            },
        }

    def _default_bounds_for_feature(self, name: str) -> Tuple[float, float]:
        key = str(name).lower()
        if key.endswith("_soc_ratio") or key.endswith("_ratio"):
            return 0.0, 1.0
        if key.endswith("_time_step"):
            return -1.0, float(max(self.env.time_steps, 1))
        if key.endswith("_count"):
            upper = float(max(len(self._building_ids), len(self._charger_refs), len(self._ev_ids), 1))
            return 0.0, upper
        if key.endswith("_power_kw") or "headroom_kw" in key:
            return -1.0e6, 1.0e6
        if key.endswith("_kwh_step"):
            return -1.0e6, 1.0e6
        if key.endswith("_capacity_kwh") or key.endswith("_kwh"):
            return -1.0e6, 1.0e6
        return -1.0e6, 1.0e6

    def _observation_bounds_for_features(self, names: Sequence[str], *, owner: str, per_building: bool = False):
        if owner == "district":
            if len(names) == 0:
                return np.zeros((1, 0), dtype=np.float32), np.zeros((1, 0), dtype=np.float32)
            if len(self.env.buildings) == 0:
                zeros = np.zeros((1, len(names)), dtype=np.float32)
                return zeros.copy(), zeros
            building = self.env.buildings[0]
            low_map, high_map = building.estimate_observation_space_limits(include_all=True, periodic_normalization=False)
            low_values: List[float] = []
            high_values: List[float] = []
            for name in names:
                default_low, default_high = self._default_bounds_for_feature(name)
                low_values.append(self._safe_scalar(low_map.get(name, default_low), default_low))
                high_values.append(self._safe_scalar(high_map.get(name, default_high), default_high))
            low = np.array([low_values], dtype=np.float32)
            high = np.array([high_values], dtype=np.float32)
            return low, high

        if not per_building:
            return np.zeros((len(self._building_ids), 0), dtype=np.float32), np.zeros((len(self._building_ids), 0), dtype=np.float32)

        rows = len(self._building_ids)
        cols = len(names)
        low = np.full((rows, cols), -1.0e6, dtype=np.float32)
        high = np.full((rows, cols), 1.0e6, dtype=np.float32)
        if cols == 0:
            return low, high

        for i, building in enumerate(self.env.buildings):
            low_map, high_map = building.estimate_observation_space_limits(include_all=True, periodic_normalization=False)
            for j, name in enumerate(names):
                default_low, default_high = self._default_bounds_for_feature(name)
                low[i, j] = self._safe_scalar(low_map.get(name, default_low), default_low)
                high[i, j] = self._safe_scalar(high_map.get(name, default_high), default_high)

        return low, high

    def _charger_observation_bounds(self):
        rows = len(self._charger_refs)
        cols = len(self._charger_features)
        low = np.full((rows, cols), -1.0e6, dtype=np.float32)
        high = np.full((rows, cols), 1.0e6, dtype=np.float32)
        if rows == 0:
            return low, high

        step_hours = self._step_hours()
        max_departure = float(self.env.time_steps)
        max_capacity = max([self._safe_scalar(ev.battery.capacity, 0.0) for ev in self.env.electric_vehicles] + [0.0])
        charging_limits = [
            self._safe_scalar(charger.max_charging_power, 0.0)
            for building in self.env.buildings
            for charger in (building.electric_vehicle_chargers or [])
        ]
        discharging_limits = [
            self._safe_scalar(charger.max_discharging_power, 0.0)
            for building in self.env.buildings
            for charger in (building.electric_vehicle_chargers or [])
        ]
        max_charge_kw = max(charging_limits + [0.0])
        max_discharge_kw = max(discharging_limits + [0.0])
        max_abs_power_kw = max(max_charge_kw, max_discharge_kw, 1.0)
        max_energy_step = max_abs_power_kw * step_hours

        bounds = {
            "connected_state": (0.0, 1.0),
            "incoming_state": (0.0, 1.0),
            "connected_ev_soc": (-0.1, 1.0),
            "connected_ev_required_soc_departure": (-0.1, 1.0),
            "connected_ev_battery_capacity_kwh": (-1.0, max(max_capacity, 1.0)),
            "connected_ev_departure_time_step": (-1.0, max_departure),
            "incoming_ev_estimated_soc_arrival": (-0.1, 1.0),
            "incoming_ev_estimated_arrival_time_step": (-1.0, max_departure),
            "last_charged_kwh": (-max_energy_step, max_energy_step),
            "max_charging_power_kw": (0.0, max(max_charge_kw, 1.0)),
            "max_discharging_power_kw": (0.0, max(max_discharge_kw, 1.0)),
            "commanded_power_kw": (-max_abs_power_kw, max_abs_power_kw),
            "applied_power_kw": (-max_abs_power_kw, max_abs_power_kw),
            "applied_energy_kwh_step": (-max_energy_step, max_energy_step),
            "energy_to_required_soc_kwh": (0.0, max(max_capacity, 1.0)),
            "avg_power_to_departure_kw": (0.0, max(max_abs_power_kw, 1.0)),
        }

        for j, feature in enumerate(self._charger_features):
            feature_low, feature_high = bounds.get(feature, self._default_bounds_for_feature(feature))
            low[:, j] = feature_low
            high[:, j] = feature_high

        return low, high

    def _ev_observation_bounds(self):
        rows = len(self._ev_ids)
        cols = len(self._ev_features)
        low = np.full((rows, cols), -1.0e6, dtype=np.float32)
        high = np.full((rows, cols), 1.0e6, dtype=np.float32)
        if rows == 0:
            return low, high

        max_capacity = max([self._safe_scalar(ev.battery.capacity, 0.0) for ev in self.env.electric_vehicles] + [0.0])

        bounds = {
            "soc": (0.0, 1.0),
            "soc_ratio": (0.0, 1.0),
            "soc_min_ratio": (0.0, 1.0),
            "soc_max_ratio": (0.0, 1.0),
            "battery_capacity_kwh": (0.0, max(max_capacity, 1.0)),
            "depth_of_discharge_ratio": (0.0, 1.0),
            "energy_available_kwh": (0.0, max(max_capacity, 1.0)),
            "energy_to_full_kwh": (0.0, max(max_capacity, 1.0)),
        }

        for j, feature in enumerate(self._ev_features):
            feature_low, feature_high = bounds.get(feature, self._default_bounds_for_feature(feature))
            low[:, j] = feature_low
            high[:, j] = feature_high

        return low, high

    def _storage_observation_bounds(self):
        rows = len(self._storage_refs)
        cols = len(self._storage_features)
        low = np.full((rows, cols), -1.0e6, dtype=np.float32)
        high = np.full((rows, cols), 1.0e6, dtype=np.float32)

        if rows == 0:
            return low, high

        step_hours = self._step_hours()
        for ref in self._storage_refs:
            building = self.env.buildings[ref.building_index]
            storage = building.electrical_storage
            capacity = self._safe_scalar(getattr(storage, "capacity", 0.0), 0.0)
            nominal_power = self._safe_scalar(getattr(storage, "nominal_power", 0.0), 0.0)
            energy_limit = max(nominal_power * step_hours, 1.0)
            row = ref.row

            for col, feature in enumerate(self._storage_features):
                feature_low, feature_high = self._default_bounds_for_feature(feature)
                if feature in {"soc", "electrical_storage_soc_ratio"}:
                    feature_low, feature_high = 0.0, 1.0
                elif feature in {"capacity_kwh", "energy_to_full_kwh", "energy_available_kwh"}:
                    feature_low, feature_high = 0.0, max(capacity, 1.0)
                elif feature in {"nominal_power_kw", "max_charge_power_kw", "max_discharge_power_kw"}:
                    feature_low, feature_high = 0.0, max(nominal_power, 1.0)
                elif feature == "electricity_consumption_kwh":
                    feature_low, feature_high = -energy_limit, energy_limit
                low[row, col] = feature_low
                high[row, col] = feature_high

        return low, high

    def _pv_observation_bounds(self):
        rows = len(self._pv_refs)
        cols = len(self._pv_features)
        low = np.full((rows, cols), -1.0e6, dtype=np.float32)
        high = np.full((rows, cols), 1.0e6, dtype=np.float32)
        if rows == 0:
            return low, high

        step_hours = self._step_hours()
        for ref in self._pv_refs:
            building = self.env.buildings[ref.building_index]
            installed_power = self._safe_scalar(getattr(building.pv, "nominal_power", 0.0), 0.0)
            row = ref.row
            for col, feature in enumerate(self._pv_features):
                if feature == "generation_power_kw":
                    low[row, col], high[row, col] = 0.0, max(installed_power, 1.0)
                elif feature == "generation_energy_kwh_step":
                    low[row, col], high[row, col] = 0.0, max(installed_power * step_hours, 1.0)
                elif feature == "installed_power_kw":
                    low[row, col], high[row, col] = 0.0, max(installed_power, 1.0)
                else:
                    low[row, col], high[row, col] = self._default_bounds_for_feature(feature)
        return low, high

    def _building_action_bounds(self):
        rows = len(self._building_ids)
        cols = len(self._building_action_features)
        low = np.zeros((rows, cols), dtype=np.float32)
        high = np.zeros((rows, cols), dtype=np.float32)
        for i, building in enumerate(self.env.buildings):
            for action_name, a_low, a_high in zip(building.active_actions, building.action_space.low, building.action_space.high):
                col = self._building_action_col_by_name.get(action_name)
                if col is None:
                    continue
                low[i, col] = float(a_low)
                high[i, col] = float(a_high)
        return low, high

    def _charger_action_bounds(self):
        rows = len(self._charger_refs)
        cols = len(self._charger_action_features)
        low = np.zeros((rows, cols), dtype=np.float32)
        high = np.zeros((rows, cols), dtype=np.float32)
        if cols == 0:
            return low, high

        for ref in self._charger_refs:
            building = self.env.buildings[ref.building_index]
            for action_name, a_low, a_high in zip(building.active_actions, building.action_space.low, building.action_space.high):
                expected = f"electric_vehicle_storage_{ref.charger_id}"
                if action_name == expected:
                    low[ref.row, 0] = float(a_low)
                    high[ref.row, 0] = float(a_high)
                    break

        return low, high

    def _ensure_initialized(self):
        if not self._initialized:
            self.reset()
            return

        current_topology_version = int(getattr(self.env, 'topology_version', 0))
        if current_topology_version != self._layout_topology_version:
            self.reset()

    @staticmethod
    def _safe_scalar(value: Any, default: float = 0.0) -> float:
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not np.isfinite(scalar):
            return float(default)
        return scalar

    @classmethod
    def _safe_index(cls, values: Any, index: int, default: float = 0.0) -> float:
        try:
            return cls._safe_scalar(values[index], default)
        except Exception:
            return float(default)

    @staticmethod
    def _to_2d_array(value: Any, *, rows: int, cols: int) -> Optional[np.ndarray]:
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 1 and cols > 0 and arr.shape[0] == cols and rows == 1:
            return arr.reshape(1, cols)
        if arr.ndim != 2:
            raise AssertionError("Entity action tables must be 2D arrays.")
        if arr.shape[0] != rows or arr.shape[1] != cols:
            raise AssertionError(f"Entity action table shape mismatch: expected ({rows}, {cols}) got {arr.shape}.")
        return arr

    @staticmethod
    def _has_electrical_storage(building) -> bool:
        storage = getattr(building, "electrical_storage", None)
        if storage is None:
            return False
        capacity = float(getattr(storage, "capacity", 0.0))
        nominal_power = float(getattr(storage, "nominal_power", 0.0))
        return capacity > 0.0 and nominal_power > 0.0

    @staticmethod
    def _has_pv_asset(building) -> bool:
        pv = getattr(building, "pv", None)
        if pv is None:
            return False
        nominal_power = float(getattr(pv, "nominal_power", 0.0))
        return nominal_power > 0.0

    @staticmethod
    def _is_entity_specific_observation(name: str) -> bool:
        key = str(name)
        return (
            key.startswith("electric_vehicle_charger_")
            or key.startswith("connected_electric_vehicle_at_charger_")
            or key.startswith("incoming_electric_vehicle_at_charger_")
            or "washing_machine" in key
        )

    @staticmethod
    def _is_flat_like(actions: Any) -> bool:
        if isinstance(actions, np.ndarray):
            return True
        if isinstance(actions, (list, tuple)):
            if len(actions) == 0:
                return True
            first = actions[0]
            return np.isscalar(first) or isinstance(first, (list, tuple, np.ndarray))
        return False

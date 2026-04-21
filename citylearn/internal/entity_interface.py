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


class CityLearnEntityInterfaceService:
    """Build and parse canonical entity payloads."""

    CHARGER_FEATURES = [
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
    EV_FEATURES = [
        "soc",
        "battery_capacity_kwh",
        "depth_of_discharge_ratio",
    ]
    STORAGE_FEATURES = [
        "soc",
        "capacity_kwh",
        "nominal_power_kw",
        "electricity_consumption_kwh",
    ]

    def __init__(self, env):
        self.env = env
        self._initialized = False
        self._layout_topology_version = -1

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

        self._district_obs.fill(0.0)
        self._building_obs.fill(0.0)
        self._charger_obs.fill(0.0)
        self._ev_obs.fill(0.0)
        self._storage_obs.fill(0.0)

        for i, building in enumerate(env.buildings):
            # Exogenous signals are read at current t, while endogenous values come from
            # the latest settled transition (max(t-1, 0)) to avoid uninitialized buffers.
            data = building._get_observations_data(include_all=False)

            if i == 0:
                for j, name in enumerate(self._district_features):
                    self._district_obs[0, j] = self._safe_scalar(data.get(name, 0.0))

            for j, name in enumerate(self._building_features):
                self._building_obs[i, j] = self._safe_scalar(data.get(name, 0.0))

        for ref in self._charger_refs:
            charger = env.buildings[ref.building_index]._charger_lookup[ref.charger_id]
            sim = charger.charger_simulation
            state = self._safe_index(sim.electric_vehicle_charger_state, t, np.nan)
            connected = charger.connected_electric_vehicle is not None and state == 1
            incoming = charger.incoming_electric_vehicle is not None and state == 2

            values = {
                "connected_state": 1.0 if connected else 0.0,
                "incoming_state": 1.0 if incoming else 0.0,
                "connected_ev_soc": self._safe_scalar(charger.connected_electric_vehicle.battery.soc[endogenous_t], -0.1) if connected else -0.1,
                "connected_ev_required_soc_departure": self._safe_index(sim.electric_vehicle_required_soc_departure, t, -0.1) if connected else -0.1,
                "connected_ev_battery_capacity_kwh": self._safe_scalar(charger.connected_electric_vehicle.battery.capacity, -1.0) if connected else -1.0,
                "connected_ev_departure_time_step": self._safe_index(sim.electric_vehicle_departure_time, t, -1.0) if connected else -1.0,
                "incoming_ev_estimated_soc_arrival": self._safe_index(sim.electric_vehicle_estimated_soc_arrival, t, -0.1) if incoming else -0.1,
                "incoming_ev_estimated_arrival_time_step": self._safe_index(sim.electric_vehicle_estimated_arrival_time, t, -1.0) if incoming else -1.0,
                "last_charged_kwh": self._safe_index(charger.past_charging_action_values_kwh, endogenous_t, 0.0),
                "max_charging_power_kw": self._safe_scalar(charger.max_charging_power, 0.0),
                "max_discharging_power_kw": self._safe_scalar(charger.max_discharging_power, 0.0),
            }
            for col, feature_name in enumerate(self.CHARGER_FEATURES):
                self._charger_obs[ref.row, col] = self._safe_scalar(values.get(feature_name, 0.0))

        for i, ev in enumerate(env.electric_vehicles):
            self._ev_obs[i, 0] = self._safe_scalar(ev.battery.soc[endogenous_t], 0.0)
            self._ev_obs[i, 1] = self._safe_scalar(ev.battery.capacity, 0.0)
            self._ev_obs[i, 2] = self._safe_scalar(ev.battery.depth_of_discharge, 0.0)

        for ref in self._storage_refs:
            building = env.buildings[ref.building_index]
            storage = building.electrical_storage
            values = {
                "soc": self._safe_scalar(storage.soc[endogenous_t], 0.0),
                "capacity_kwh": self._safe_scalar(getattr(storage, "capacity", 0.0), 0.0),
                "nominal_power_kw": self._safe_scalar(getattr(storage, "nominal_power", 0.0), 0.0),
                "electricity_consumption_kwh": self._safe_scalar(
                    building.electrical_storage_electricity_consumption[endogenous_t],
                    0.0,
                ),
            }
            for col, feature_name in enumerate(self.STORAGE_FEATURES):
                self._storage_obs[ref.row, col] = self._safe_scalar(values.get(feature_name, 0.0))

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
            },
            "edges": {
                "district_to_building": self._district_to_building,
                "building_to_charger": self._building_to_charger,
                "building_to_storage": self._building_to_storage,
                "charger_to_ev_connected": self._charger_to_ev_connected,
                "charger_to_ev_connected_mask": self._charger_to_ev_connected_mask,
                "charger_to_ev_incoming": self._charger_to_ev_incoming,
                "charger_to_ev_incoming_mask": self._charger_to_ev_incoming_mask,
            },
            "meta": {
                "time_step": t,
                "endogenous_time_step": endogenous_t,
                "spec_version": "entity_v1",
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
        self._charger_obs = np.zeros((len(self._charger_refs), len(self.CHARGER_FEATURES)), dtype=np.float32)
        self._ev_obs = np.zeros((len(self._ev_ids), len(self.EV_FEATURES)), dtype=np.float32)
        self._storage_obs = np.zeros((len(self._storage_refs), len(self.STORAGE_FEATURES)), dtype=np.float32)

        self._district_to_building = np.zeros((len(self._building_ids), 2), dtype=np.int32)
        if len(self._building_ids) > 0:
            self._district_to_building[:, 1] = np.arange(len(self._building_ids), dtype=np.int32)

        self._building_to_charger = np.full((len(self._charger_refs), 2), -1, dtype=np.int32)
        for ref in self._charger_refs:
            self._building_to_charger[ref.row] = np.array([ref.building_index, ref.row], dtype=np.int32)

        self._building_to_storage = np.full((len(self._storage_refs), 2), -1, dtype=np.int32)
        for ref in self._storage_refs:
            self._building_to_storage[ref.row] = np.array([ref.building_index, ref.row], dtype=np.int32)

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

        self._observation_space = spaces.Dict(
            {
                "tables": spaces.Dict(
                    {
                        "district": spaces.Box(low=district_low, high=district_high, dtype=np.float32),
                        "building": spaces.Box(low=building_low, high=building_high, dtype=np.float32),
                        "charger": spaces.Box(low=charger_low, high=charger_high, dtype=np.float32),
                        "ev": spaces.Box(low=ev_low, high=ev_high, dtype=np.float32),
                        "storage": spaces.Box(low=storage_low, high=storage_high, dtype=np.float32),
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
        def _unit_hint(name: str) -> str:
            key = str(name).lower()
            if key.endswith("_soc") or key.endswith("_ratio"):
                return "ratio"
            if key.endswith("_kw") or "headroom_kw" in key:
                return "kw"
            if key.endswith("_kwh") or "demand" in key or "consumption" in key or "generation" in key:
                return "kwh"
            if key.endswith("_time_step"):
                return "time_step"
            if key in {"hour", "day_type", "month", "minutes"}:
                return "index"
            return "scalar"

        def _units_for(names: Sequence[str]) -> List[str]:
            return [_unit_hint(name) for name in names]

        self._specs = {
            "version": "entity_v1",
            "tables": {
                "district": {
                    "ids": ["district_0"],
                    "features": list(self._district_features),
                    "units": _units_for(self._district_features),
                },
                "building": {
                    "ids": list(self._building_ids),
                    "features": list(self._building_features),
                    "units": _units_for(self._building_features),
                },
                "charger": {
                    "ids": [ref.global_id for ref in self._charger_refs],
                    "features": list(self.CHARGER_FEATURES),
                    "units": _units_for(self.CHARGER_FEATURES),
                },
                "ev": {
                    "ids": list(self._ev_ids),
                    "features": list(self.EV_FEATURES),
                    "units": _units_for(self.EV_FEATURES),
                },
                "storage": {
                    "ids": [ref.global_id for ref in self._storage_refs],
                    "features": list(self.STORAGE_FEATURES),
                    "units": _units_for(self.STORAGE_FEATURES),
                },
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
                },
                "lifecycle_fields": ["born_at", "removed_at", "active"],
                "member_lifecycle": getattr(self.env, "topology_member_lifecycle", {}),
            },
        }

    def _observation_bounds_for_features(self, names: Sequence[str], *, owner: str, per_building: bool = False):
        if owner == "district":
            if len(names) == 0:
                return np.zeros((1, 0), dtype=np.float32), np.zeros((1, 0), dtype=np.float32)
            building = self.env.buildings[0]
            low_map, high_map = building.estimate_observation_space_limits(include_all=True, periodic_normalization=False)
            low = np.array([[self._safe_scalar(low_map.get(name, -1.0e6), -1.0e6) for name in names]], dtype=np.float32)
            high = np.array([[self._safe_scalar(high_map.get(name, 1.0e6), 1.0e6) for name in names]], dtype=np.float32)
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
                low[i, j] = self._safe_scalar(low_map.get(name, -1.0e6), -1.0e6)
                high[i, j] = self._safe_scalar(high_map.get(name, 1.0e6), 1.0e6)

        return low, high

    def _charger_observation_bounds(self):
        rows = len(self._charger_refs)
        cols = len(self.CHARGER_FEATURES)
        low = np.full((rows, cols), -1.0e6, dtype=np.float32)
        high = np.full((rows, cols), 1.0e6, dtype=np.float32)
        if rows == 0:
            return low, high

        max_departure = float(self.env.time_steps)
        max_capacity = max([self._safe_scalar(ev.battery.capacity, 0.0) for ev in self.env.electric_vehicles] + [0.0])
        max_last_charge = 0.0
        for ref in self._charger_refs:
            charger = self.env.buildings[ref.building_index]._charger_lookup[ref.charger_id]
            max_power = max(self._safe_scalar(charger.max_charging_power, 0.0), self._safe_scalar(charger.max_discharging_power, 0.0))
            max_last_charge = max(max_last_charge, max_power * (self.env.seconds_per_time_step / 3600.0))

        bounds = {
            "connected_state": (0.0, 1.0),
            "incoming_state": (0.0, 1.0),
            "connected_ev_soc": (-0.1, 1.0),
            "connected_ev_required_soc_departure": (-0.1, 1.0),
            "connected_ev_battery_capacity_kwh": (-1.0, max_capacity if max_capacity > 0.0 else 1.0),
            "connected_ev_departure_time_step": (-1.0, max_departure),
            "incoming_ev_estimated_soc_arrival": (-0.1, 1.0),
            "incoming_ev_estimated_arrival_time_step": (-1.0, max_departure),
            "last_charged_kwh": (-max_last_charge, max_last_charge),
            "max_charging_power_kw": (0.0, max([self._safe_scalar(c.max_charging_power, 0.0) for b in self.env.buildings for c in (b.electric_vehicle_chargers or [])] + [0.0])),
            "max_discharging_power_kw": (0.0, max([self._safe_scalar(c.max_discharging_power, 0.0) for b in self.env.buildings for c in (b.electric_vehicle_chargers or [])] + [0.0])),
        }

        for j, feature in enumerate(self.CHARGER_FEATURES):
            feature_low, feature_high = bounds.get(feature, (-1.0e6, 1.0e6))
            low[:, j] = feature_low
            high[:, j] = feature_high

        return low, high

    def _ev_observation_bounds(self):
        rows = len(self._ev_ids)
        cols = len(self.EV_FEATURES)
        low = np.full((rows, cols), -1.0e6, dtype=np.float32)
        high = np.full((rows, cols), 1.0e6, dtype=np.float32)
        if rows == 0:
            return low, high

        capacities = [self._safe_scalar(ev.battery.capacity, 0.0) for ev in self.env.electric_vehicles]
        max_capacity = max(capacities + [0.0])
        low[:, 0] = 0.0
        high[:, 0] = 1.0
        low[:, 1] = 0.0
        high[:, 1] = max_capacity if max_capacity > 0 else 1.0
        low[:, 2] = 0.0
        high[:, 2] = 1.0
        return low, high

    def _storage_observation_bounds(self):
        rows = len(self._storage_refs)
        cols = len(self.STORAGE_FEATURES)
        low = np.full((rows, cols), -1.0e6, dtype=np.float32)
        high = np.full((rows, cols), 1.0e6, dtype=np.float32)

        if rows == 0:
            return low, high

        for ref in self._storage_refs:
            building = self.env.buildings[ref.building_index]
            storage = building.electrical_storage
            capacity = self._safe_scalar(getattr(storage, "capacity", 0.0), 0.0)
            nominal_power = self._safe_scalar(getattr(storage, "nominal_power", 0.0), 0.0)
            row = ref.row

            low[row, self.STORAGE_FEATURES.index("soc")] = 0.0
            high[row, self.STORAGE_FEATURES.index("soc")] = 1.0
            low[row, self.STORAGE_FEATURES.index("capacity_kwh")] = 0.0
            high[row, self.STORAGE_FEATURES.index("capacity_kwh")] = max(capacity, 1.0)
            low[row, self.STORAGE_FEATURES.index("nominal_power_kw")] = 0.0
            high[row, self.STORAGE_FEATURES.index("nominal_power_kw")] = max(nominal_power, 1.0)
            low[row, self.STORAGE_FEATURES.index("electricity_consumption_kwh")] = -max(nominal_power, 1.0)
            high[row, self.STORAGE_FEATURES.index("electricity_consumption_kwh")] = max(nominal_power, 1.0)

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

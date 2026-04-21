from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from citylearn.energy_model import Battery, PV
from citylearn.utilities import parse_bool

if TYPE_CHECKING:
    from citylearn.building import Building
    from citylearn.citylearn import CityLearnEnv
    from citylearn.electric_vehicle import ElectricVehicle
    from citylearn.electric_vehicle_charger import Charger


SUPPORTED_OPERATIONS = {
    'add_member',
    'remove_member',
    'add_asset',
    'remove_asset',
}
SUPPORTED_ASSET_TYPES = {
    'charger',
    'pv',
    'electrical_storage',
}


@dataclass(frozen=True)
class TopologyEvent:
    """Canonical in-memory topology event."""

    event_id: str
    time_step: int
    operation: str
    target_member_id: Optional[str]
    target_asset_type: Optional[str]
    target_asset_id: Optional[str]
    source_member_id: Optional[str]
    source_asset_id: Optional[str]
    overrides: Mapping[str, Any]
    order: int


class CityLearnTopologyService:
    """Schema-driven dynamic topology lifecycle and mutation service."""

    def __init__(self, env: "CityLearnEnv"):
        self.env = env
        self._member_pool: Dict[str, Building] = {}
        self._member_order: List[str] = []
        self._ev_pool: Dict[str, ElectricVehicle] = {}
        self._ev_order: List[str] = []
        self._initial_active_member_ids: List[str] = []
        self._active_member_ids: List[str] = []
        self._active_ev_ids: List[str] = []
        self._member_lifecycle: Dict[str, Dict[str, Any]] = {}
        self._events: List[TopologyEvent] = []
        self._event_cursor: int = 0
        self._topology_version: int = 0
        self._event_log: List[Mapping[str, Any]] = []
        self._active_member_history: Dict[int, List[str]] = {}
        self._active_ev_history: Dict[int, List[str]] = {}
        self._topology_version_history: Dict[int, int] = {}
        self._active_charger_history: Dict[int, Dict[str, Dict[str, Charger]]] = {}
        self._active_storage_history: Dict[int, Dict[str, Optional[Battery]]] = {}
        self._charger_observation_flags = self._collect_charger_observation_flags()
        self._charger_action_enabled = self._is_schema_action_active('electric_vehicle_storage')

    @property
    def enabled(self) -> bool:
        return getattr(self.env, 'topology_mode', 'static') == 'dynamic'

    @property
    def member_pool(self) -> Mapping[str, Building]:
        return self._member_pool

    @property
    def ev_pool(self) -> Mapping[str, ElectricVehicle]:
        return self._ev_pool

    @property
    def active_member_ids(self) -> Sequence[str]:
        return tuple(self._active_member_ids)

    @property
    def active_ev_ids(self) -> Sequence[str]:
        return tuple(self._active_ev_ids)

    @property
    def topology_version(self) -> int:
        return int(self._topology_version)

    @property
    def event_log(self) -> Sequence[Mapping[str, Any]]:
        return tuple(self._event_log)

    @property
    def member_lifecycle(self) -> Mapping[str, Mapping[str, Any]]:
        return {
            member_id: dict(state)
            for member_id, state in self._member_lifecycle.items()
        }

    def initialize(self, buildings: Sequence[Building], electric_vehicles: Sequence[ElectricVehicle]):
        """Initialize static pools and parse deterministic event stream."""

        self._member_pool = {building.name: building for building in buildings}
        self._member_order = [building.name for building in buildings]
        self._ev_pool = {ev.name: ev for ev in electric_vehicles}
        self._ev_order = [ev.name for ev in electric_vehicles]
        self._events = self._parse_events()

        include_flags = self._initial_member_include_flags()
        self._initial_active_member_ids = [
            member_id for member_id in self._member_order if include_flags.get(member_id, True)
        ]

        if len(self._initial_active_member_ids) == 0 and len(self._member_order) > 0:
            self._initial_active_member_ids = [self._member_order[0]]

    def reset(self):
        """Reset topology and pools for a fresh episode."""

        if not self.enabled:
            return

        for building in self._member_pool.values():
            self._bind_building_runtime_context(building)
            building.reset()

        for ev in self._ev_pool.values():
            self._bind_ev_runtime_context(ev)
            ev.reset()

        self._active_member_ids = list(self._initial_active_member_ids)
        self._active_ev_ids = []
        self._member_lifecycle = {
            member_id: {
                'born_at': 0 if member_id in set(self._active_member_ids) else None,
                'removed_at': None,
                'active': member_id in set(self._active_member_ids),
            }
            for member_id in self._member_order
        }

        self._event_cursor = 0
        self._topology_version = 0
        self._event_log = []
        self._active_member_history = {}
        self._active_ev_history = {}
        self._topology_version_history = {}
        self._active_charger_history = {}
        self._active_storage_history = {}

        self._set_active_views()
        self.apply_events_for_time_step(0)

    def apply_events_for_time_step(self, time_step: int) -> bool:
        """Apply all schema events scheduled at `time_step`."""

        if not self.enabled:
            return False

        changed = False

        while self._event_cursor < len(self._events):
            event = self._events[self._event_cursor]

            if event.time_step > time_step:
                break

            if event.time_step < time_step:
                # Defensive skip if caller advances in larger increments.
                self._event_cursor += 1
                continue

            event_changed = self._apply_event(event, time_step)
            changed = changed or event_changed
            self._event_log.append(
                {
                    'id': event.event_id,
                    'time_step': event.time_step,
                    'operation': event.operation,
                    'target_member_id': event.target_member_id,
                    'target_asset_type': event.target_asset_type,
                    'target_asset_id': event.target_asset_id,
                    'source_member_id': event.source_member_id,
                    'source_asset_id': event.source_asset_id,
                    'applied': bool(event_changed),
                    'topology_version': int(self._topology_version),
                }
            )
            self._event_cursor += 1

        if changed:
            self._set_active_views()

        self._record_history(time_step)

        return changed

    def active_member_ids_at(self, time_step: int) -> List[str]:
        return self._history_lookup(self._active_member_history, time_step, list(self._active_member_ids))

    def active_ev_ids_at(self, time_step: int) -> List[str]:
        return self._history_lookup(self._active_ev_history, time_step, list(self._active_ev_ids))

    def topology_version_at(self, time_step: int) -> int:
        values = self._history_lookup(self._topology_version_history, time_step, int(self._topology_version))
        return int(values)

    def active_chargers_at(self, time_step: int, member_id: str) -> Mapping[str, Charger]:
        snapshot = self._history_lookup_reference(self._active_charger_history, time_step, {})
        member_chargers = snapshot.get(member_id, {})
        return dict(member_chargers)

    def active_storage_at(self, time_step: int, member_id: str) -> Optional[Battery]:
        snapshot = self._history_lookup_reference(self._active_storage_history, time_step, {})
        return snapshot.get(member_id)

    @staticmethod
    def _history_lookup(history: Mapping[int, Any], time_step: int, default: Any):
        if time_step in history:
            return deepcopy(history[time_step])

        valid = [k for k in history.keys() if k <= time_step]

        if len(valid) == 0:
            return deepcopy(default)

        return deepcopy(history[max(valid)])

    @staticmethod
    def _history_lookup_reference(history: Mapping[int, Any], time_step: int, default: Any):
        if time_step in history:
            return history[time_step]

        valid = [k for k in history.keys() if k <= time_step]
        if len(valid) == 0:
            return default

        return history[max(valid)]

    def _record_history(self, time_step: int):
        self._active_member_history[int(time_step)] = list(self._active_member_ids)
        self._active_ev_history[int(time_step)] = list(self._active_ev_ids)
        self._topology_version_history[int(time_step)] = int(self._topology_version)
        charger_snapshot: Dict[str, Dict[str, Charger]] = {}
        storage_snapshot: Dict[str, Optional[Battery]] = {}

        for member_id in self._active_member_ids:
            building = self._member_pool.get(member_id)
            if building is None:
                continue

            charger_snapshot[member_id] = {
                charger.charger_id: charger for charger in (building.electric_vehicle_chargers or [])
            }
            storage_snapshot[member_id] = (
                building.electrical_storage if self._has_electrical_storage_asset(building) else None
            )

        self._active_charger_history[int(time_step)] = charger_snapshot
        self._active_storage_history[int(time_step)] = storage_snapshot

    def _set_active_views(self):
        env = self.env
        active_set = set(self._active_member_ids)
        active_buildings = [
            self._member_pool[member_id]
            for member_id in self._member_order
            if member_id in active_set
        ]
        env.buildings = active_buildings

        t = int(getattr(env, 'time_step', 0))
        for building in active_buildings:
            self._bind_building_runtime_context(building)
            self._set_building_time_step(building, t)

        active_ev_ids = self._collect_active_ev_ids(active_buildings)
        self._active_ev_ids = [ev_id for ev_id in self._ev_order if ev_id in active_ev_ids]
        env.electric_vehicles = [self._ev_pool[ev_id] for ev_id in self._active_ev_ids]

        for ev in env.electric_vehicles:
            self._bind_ev_runtime_context(ev)
            self._set_ev_time_step(ev, t)

    @staticmethod
    def _set_building_time_step(building: Building, time_step: int):
        building.time_step = time_step

        for attr_name in (
            'cooling_device',
            'heating_device',
            'dhw_device',
            'non_shiftable_load_device',
            'cooling_storage',
            'heating_storage',
            'dhw_storage',
            'electrical_storage',
            'pv',
        ):
            obj = getattr(building, attr_name, None)
            if obj is not None and hasattr(obj, 'time_step'):
                obj.time_step = time_step

        for charger in building.electric_vehicle_chargers or []:
            charger.time_step = time_step

        for washing_machine in building.washing_machines or []:
            washing_machine.time_step = time_step

    @staticmethod
    def _set_ev_time_step(ev: ElectricVehicle, time_step: int):
        ev.time_step = time_step
        if getattr(ev, 'battery', None) is not None:
            ev.battery.time_step = time_step

    def _bind_ev_runtime_context(self, ev: ElectricVehicle):
        env = self.env
        ev.episode_tracker = env.episode_tracker
        ev.random_seed = env.random_seed
        ev.time_step_ratio = env.time_step_ratio

    def _bind_building_runtime_context(self, building: Building):
        env = self.env
        building.episode_tracker = env.episode_tracker
        building.random_seed = env.random_seed
        building.time_step_ratio = env.time_step_ratio

        for charger in building.electric_vehicle_chargers or []:
            charger.episode_tracker = env.episode_tracker
            charger.random_seed = env.random_seed
            charger.time_step_ratio = building.time_step_ratio

        for washing_machine in building.washing_machines or []:
            washing_machine.episode_tracker = env.episode_tracker
            washing_machine.random_seed = env.random_seed
            washing_machine.time_step_ratio = building.time_step_ratio

    def _collect_active_ev_ids(self, buildings: Iterable[Building]) -> Set[str]:
        active_ev_ids: Set[str] = set()

        for building in buildings:
            for charger in building.electric_vehicle_chargers or []:
                sim_ids = getattr(charger.charger_simulation, 'electric_vehicle_id', None)
                if sim_ids is not None:
                    for ev_id in sim_ids:
                        if self._is_valid_ev_id(ev_id) and ev_id in self._ev_pool:
                            active_ev_ids.add(ev_id)

                for ev_obj in (
                    getattr(charger, 'connected_electric_vehicle', None),
                    getattr(charger, 'incoming_electric_vehicle', None),
                ):
                    ev_name = getattr(ev_obj, 'name', None)
                    if isinstance(ev_name, str) and ev_name in self._ev_pool:
                        active_ev_ids.add(ev_name)

        return active_ev_ids

    @staticmethod
    def _is_valid_ev_id(value: Any) -> bool:
        if not isinstance(value, str):
            return False

        text = value.strip()
        return text not in {'', 'nan'}

    def _apply_event(self, event: TopologyEvent, time_step: int) -> bool:
        op = event.operation

        if op == 'add_member':
            return self._add_member(event, time_step)

        if op == 'remove_member':
            return self._remove_member(event, time_step)

        if op == 'add_asset':
            return self._add_asset(event, time_step)

        if op == 'remove_asset':
            return self._remove_asset(event, time_step)

        raise ValueError(f'Unsupported topology operation: {op}')

    def _add_member(self, event: TopologyEvent, time_step: int) -> bool:
        target_member_id = event.target_member_id
        if target_member_id is None:
            raise ValueError('add_member requires target_member_id.')

        if target_member_id not in self._member_pool:
            source_member_id = event.source_member_id
            if source_member_id is None or source_member_id not in self._member_pool:
                raise ValueError(
                    f"add_member target '{target_member_id}' is not preloaded and source_member_id is invalid."
                )

            cloned = deepcopy(self._member_pool[source_member_id])
            cloned.name = target_member_id
            self._bind_building_runtime_context(cloned)
            self._member_pool[target_member_id] = cloned
            self._member_order.append(target_member_id)
            self._member_lifecycle[target_member_id] = {
                'born_at': None,
                'removed_at': None,
                'active': False,
            }

        if target_member_id in self._active_member_ids:
            return False

        building = self._member_pool[target_member_id]
        self._bind_building_runtime_context(building)
        building.reset()
        self._warm_start_member_state(building, time_step)

        self._active_member_ids.append(target_member_id)
        lifecycle = self._member_lifecycle.setdefault(target_member_id, {'born_at': None, 'removed_at': None, 'active': False})
        lifecycle['active'] = True
        lifecycle['removed_at'] = None
        if lifecycle.get('born_at') is None:
            lifecycle['born_at'] = int(time_step)

        self._apply_building_overrides(building, event.overrides)
        self._refresh_building_after_mutation(building)
        self._topology_version += 1
        return True

    def _warm_start_member_state(self, building: Building, target_time_step: int):
        """Advance a newly inserted member to current env time with zero actions."""

        target = max(int(target_time_step), 0)
        self._set_building_time_step(building, 0)
        if target == 0:
            return

        zero_kwargs = self._zero_action_kwargs(building)
        for _ in range(target):
            building.apply_actions(**zero_kwargs)
            building.update_variables()
            building.next_time_step()

    @staticmethod
    def _zero_action_kwargs(building: Building) -> Mapping[str, Any]:
        active_actions = set(list(getattr(building, 'active_actions', []) or []))
        kwargs: Dict[str, Any] = {}

        if 'cooling_or_heating_device' in active_actions:
            kwargs['cooling_or_heating_device_action'] = 0.0
        if 'cooling_device' in active_actions:
            kwargs['cooling_device_action'] = 0.0
        if 'heating_device' in active_actions:
            kwargs['heating_device_action'] = 0.0
        if 'cooling_storage' in active_actions:
            kwargs['cooling_storage_action'] = 0.0
        if 'heating_storage' in active_actions:
            kwargs['heating_storage_action'] = 0.0
        if 'dhw_storage' in active_actions:
            kwargs['dhw_storage_action'] = 0.0
        if 'electrical_storage' in active_actions:
            kwargs['electrical_storage_action'] = 0.0

        ev_actions: Dict[str, float] = {}
        washing_actions: Dict[str, float] = {}
        for action_name in active_actions:
            if action_name.startswith('electric_vehicle_storage_'):
                charger_id = action_name.replace('electric_vehicle_storage_', '')
                ev_actions[charger_id] = 0.0
            elif 'washing_machine' in action_name:
                washing_actions[action_name] = 0.0

        if ev_actions:
            kwargs['electric_vehicle_storage_actions'] = ev_actions
        if washing_actions:
            kwargs['washing_machine_actions'] = washing_actions

        return kwargs

    def _remove_member(self, event: TopologyEvent, time_step: int) -> bool:
        target_member_id = event.target_member_id
        if target_member_id is None:
            raise ValueError('remove_member requires target_member_id.')

        if target_member_id not in self._active_member_ids:
            return False

        self._active_member_ids = [member_id for member_id in self._active_member_ids if member_id != target_member_id]
        lifecycle = self._member_lifecycle.setdefault(target_member_id, {'born_at': None, 'removed_at': None, 'active': False})
        lifecycle['active'] = False
        lifecycle['removed_at'] = int(time_step)
        self._topology_version += 1
        return True

    def _add_asset(self, event: TopologyEvent, time_step: int) -> bool:
        building = self._resolve_target_building(event)
        asset_type = event.target_asset_type

        if asset_type == 'charger':
            return self._add_charger_asset(building, event, time_step)

        if asset_type == 'pv':
            source_building = self._resolve_source_building(event)
            building.pv = deepcopy(source_building.pv)
            self._apply_object_overrides(building.pv, event.overrides)
            self._bind_building_runtime_context(building)
            building.pv.reset()
            self._set_building_time_step(building, time_step)
            self._refresh_building_after_mutation(building)
            self._topology_version += 1
            return True

        if asset_type == 'electrical_storage':
            source_building = self._resolve_source_building(event)
            building.electrical_storage = deepcopy(source_building.electrical_storage)
            self._apply_object_overrides(building.electrical_storage, event.overrides)
            self._bind_building_runtime_context(building)
            building.electrical_storage.reset()
            self._set_building_time_step(building, time_step)
            self._sync_electrical_storage_metadata(building)
            self._refresh_building_after_mutation(building)
            self._topology_version += 1
            return True

        raise ValueError(f'Unsupported target_asset_type for add_asset: {asset_type}')

    def _remove_asset(self, event: TopologyEvent, time_step: int) -> bool:
        building = self._resolve_target_building(event)
        asset_type = event.target_asset_type

        if asset_type == 'charger':
            charger_id = event.target_asset_id
            if charger_id is None:
                raise ValueError('remove_asset for charger requires target_asset_id.')

            chargers = list(building.electric_vehicle_chargers or [])
            remaining = [charger for charger in chargers if charger.charger_id != charger_id]

            if len(remaining) == len(chargers):
                return False

            building.electric_vehicle_chargers = remaining
            self._sync_charger_metadata(building)
            self._refresh_building_after_mutation(building)
            self._set_building_time_step(building, time_step)
            self._topology_version += 1
            return True

        if asset_type == 'pv':
            building.pv = PV(0.0, seconds_per_time_step=building.seconds_per_time_step)
            self._bind_building_runtime_context(building)
            building.pv.reset()
            self._set_building_time_step(building, time_step)
            self._refresh_building_after_mutation(building)
            self._topology_version += 1
            return True

        if asset_type == 'electrical_storage':
            building.electrical_storage = Battery(0.0, 0.0, seconds_per_time_step=building.seconds_per_time_step)
            self._bind_building_runtime_context(building)
            building.electrical_storage.reset()
            self._set_building_time_step(building, time_step)
            self._sync_electrical_storage_metadata(building)
            self._refresh_building_after_mutation(building)
            self._topology_version += 1
            return True

        raise ValueError(f'Unsupported target_asset_type for remove_asset: {asset_type}')

    def _add_charger_asset(self, building: Building, event: TopologyEvent, time_step: int) -> bool:
        target_asset_id = event.target_asset_id
        if target_asset_id is None:
            raise ValueError('add_asset for charger requires target_asset_id.')

        if any(charger.charger_id == target_asset_id for charger in building.electric_vehicle_chargers or []):
            return False

        source_building = self._resolve_source_building(event)
        source_asset_id = event.source_asset_id if event.source_asset_id is not None else target_asset_id
        source_charger = self._resolve_source_charger(source_building, source_asset_id)
        cloned: Charger = deepcopy(source_charger)
        cloned.charger_id = target_asset_id
        self._apply_object_overrides(cloned, event.overrides)
        cloned.episode_tracker = building.episode_tracker
        cloned.random_seed = building.random_seed
        cloned.time_step_ratio = building.time_step_ratio
        cloned.reset()
        cloned.time_step = time_step

        chargers = list(building.electric_vehicle_chargers or [])
        chargers.append(cloned)
        building.electric_vehicle_chargers = chargers
        self._sync_charger_metadata(building)
        self._refresh_building_after_mutation(building)
        self._set_building_time_step(building, time_step)

        self._topology_version += 1
        return True

    def _resolve_target_building(self, event: TopologyEvent) -> Building:
        member_id = event.target_member_id
        if member_id is None:
            raise ValueError(f'{event.operation} requires target_member_id.')

        building = self._member_pool.get(member_id)
        if building is None:
            raise ValueError(f"Unknown target_member_id '{member_id}' in topology event '{event.event_id}'.")

        if member_id not in self._active_member_ids:
            raise ValueError(f"target_member_id '{member_id}' is inactive. Activate member before asset mutation.")

        return building

    def _resolve_source_building(self, event: TopologyEvent) -> Building:
        source_member_id = event.source_member_id if event.source_member_id is not None else event.target_member_id

        if source_member_id is None:
            raise ValueError(f"{event.operation} requires source_member_id or target_member_id.")

        source_building = self._member_pool.get(source_member_id)
        if source_building is None:
            raise ValueError(f"Unknown source_member_id '{source_member_id}' in topology event '{event.event_id}'.")

        return source_building

    @staticmethod
    def _resolve_source_charger(source_building: Building, source_asset_id: str):
        if source_asset_id is None:
            raise ValueError('source_asset_id is required for charger add_asset operations.')

        for charger in source_building.electric_vehicle_chargers or []:
            if charger.charger_id == source_asset_id:
                return charger

        raise ValueError(f"Source charger '{source_asset_id}' was not found in member '{source_building.name}'.")

    def _refresh_building_after_mutation(self, building: Building):
        self._bind_building_runtime_context(building)

        if hasattr(building, '_update_charger_lookup'):
            building._update_charger_lookup()

        if hasattr(building, '_initialize_charging_constraints'):
            building._initialize_charging_constraints(
                getattr(building, '_charging_constraints_config', {}) or {},
                electrical_service=getattr(building, '_electrical_service_config', {}) or {},
                electrical_storage_phase_connection=getattr(building, '_electrical_storage_phase_connection', None),
            )

        building.observation_space = building.estimate_observation_space(include_all=False, normalize=False)
        building.action_space = building.estimate_action_space()

    def _sync_charger_metadata(self, building: Building):
        if not hasattr(building, 'observation_metadata') or not hasattr(building, 'action_metadata'):
            return

        charger_observation_prefixes = (
            'electric_vehicle_charger_',
            'connected_electric_vehicle_at_charger_',
            'incoming_electric_vehicle_at_charger_',
        )
        for key in list(building.observation_metadata.keys()):
            if key.startswith(charger_observation_prefixes):
                building.observation_metadata[key] = False

        for key in list(building.action_metadata.keys()):
            if key.startswith('electric_vehicle_storage_'):
                building.action_metadata[key] = False

        for charger in building.electric_vehicle_chargers or []:
            charger_id = charger.charger_id

            if self._charger_observation_flags.get('electric_vehicle_charger_connected_state', False):
                building.observation_metadata[f'electric_vehicle_charger_{charger_id}_connected_state'] = True

            if self._charger_observation_flags.get('connected_electric_vehicle_at_charger_departure_time', False):
                building.observation_metadata[f'connected_electric_vehicle_at_charger_{charger_id}_departure_time'] = True

            if self._charger_observation_flags.get('connected_electric_vehicle_at_charger_required_soc_departure', False):
                building.observation_metadata[f'connected_electric_vehicle_at_charger_{charger_id}_required_soc_departure'] = True

            if self._charger_observation_flags.get('connected_electric_vehicle_at_charger_soc', False):
                building.observation_metadata[f'connected_electric_vehicle_at_charger_{charger_id}_soc'] = True

            if self._charger_observation_flags.get('connected_electric_vehicle_at_charger_battery_capacity', False):
                building.observation_metadata[f'connected_electric_vehicle_at_charger_{charger_id}_battery_capacity'] = True

            if self._charger_observation_flags.get('electric_vehicle_charger_incoming_state', False):
                building.observation_metadata[f'electric_vehicle_charger_{charger_id}_incoming_state'] = True

            if self._charger_observation_flags.get('incoming_electric_vehicle_at_charger_estimated_arrival_time', False):
                building.observation_metadata[f'incoming_electric_vehicle_at_charger_{charger_id}_estimated_arrival_time'] = True

            if self._charger_observation_flags.get('incoming_electric_vehicle_at_charger_estimated_soc_arrival', False):
                building.observation_metadata[f'incoming_electric_vehicle_at_charger_{charger_id}_estimated_soc_arrival'] = True

            if self._charger_action_enabled:
                building.action_metadata[f'electric_vehicle_storage_{charger_id}'] = True

    def _sync_electrical_storage_metadata(self, building: Building):
        has_storage = self._has_electrical_storage_asset(building)
        action_enabled = self._is_schema_action_active('electrical_storage')

        if 'electrical_storage' in building.action_metadata:
            building.action_metadata['electrical_storage'] = bool(has_storage and action_enabled)

        if 'electrical_storage_soc' in building.observation_metadata:
            building.observation_metadata['electrical_storage_soc'] = bool(
                has_storage and self._is_schema_observation_active('electrical_storage_soc')
            )

        if 'electrical_storage_electricity_consumption' in building.observation_metadata:
            building.observation_metadata['electrical_storage_electricity_consumption'] = bool(
                has_storage and self._is_schema_observation_active('electrical_storage_electricity_consumption')
            )

    @staticmethod
    def _has_electrical_storage_asset(building: Building) -> bool:
        battery = getattr(building, 'electrical_storage', None)
        if battery is None:
            return False

        capacity = getattr(battery, 'capacity', 0.0)
        nominal_power = getattr(battery, 'nominal_power', 0.0)
        return float(capacity) > 0.0 and float(nominal_power) > 0.0

    @staticmethod
    def _apply_object_overrides(obj: Any, overrides: Mapping[str, Any]):
        if not isinstance(overrides, Mapping):
            return

        for key, value in overrides.items():
            if hasattr(obj, key):
                try:
                    setattr(obj, key, value)
                except Exception:
                    continue

    @staticmethod
    def _apply_building_overrides(building: Building, overrides: Mapping[str, Any]):
        if not isinstance(overrides, Mapping):
            return

        for key, value in overrides.items():
            if key in {'name', 'chargers', 'electrical_storage', 'pv'}:
                continue
            if hasattr(building, key):
                try:
                    setattr(building, key, value)
                except Exception:
                    continue

    def _parse_events(self) -> List[TopologyEvent]:
        raw_events = []
        if isinstance(self.env.schema, Mapping):
            raw_events = self.env.schema.get('topology_events', []) or []

        events: List[TopologyEvent] = []

        for order, item in enumerate(raw_events):
            if not isinstance(item, Mapping):
                raise ValueError(f'topology_events[{order}] must be an object.')

            event_id = str(item.get('id', f'topology_event_{order}'))
            try:
                time_step = int(item.get('time_step'))
            except Exception as exc:
                raise ValueError(f"topology_events[{order}].time_step must be an integer.") from exc

            operation = str(item.get('operation', '')).strip().lower()
            if operation not in SUPPORTED_OPERATIONS:
                raise ValueError(
                    f"topology_events[{order}].operation='{operation}' is not supported."
                )

            target_asset_type = item.get('target_asset_type')
            if target_asset_type is not None:
                target_asset_type = str(target_asset_type).strip().lower()

            if operation in {'add_asset', 'remove_asset'} and target_asset_type not in SUPPORTED_ASSET_TYPES:
                raise ValueError(
                    f"topology_events[{order}] target_asset_type must be one of {sorted(SUPPORTED_ASSET_TYPES)}."
                )

            events.append(
                TopologyEvent(
                    event_id=event_id,
                    time_step=time_step,
                    operation=operation,
                    target_member_id=self._normalize_optional_str(item.get('target_member_id')),
                    target_asset_type=target_asset_type,
                    target_asset_id=self._normalize_optional_str(item.get('target_asset_id')),
                    source_member_id=self._normalize_optional_str(item.get('source_member_id')),
                    source_asset_id=self._normalize_optional_str(item.get('source_asset_id')),
                    overrides=deepcopy(item.get('overrides', {}) or {}),
                    order=order,
                )
            )

        events.sort(key=lambda e: (e.time_step, e.order, e.event_id))

        seen_ids: Set[str] = set()
        for event in events:
            if event.event_id in seen_ids:
                raise ValueError(f"Duplicate topology event id '{event.event_id}'.")
            seen_ids.add(event.event_id)

        return events

    @staticmethod
    def _normalize_optional_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return None if text == '' else text

    def _initial_member_include_flags(self) -> Mapping[str, bool]:
        flags: Dict[str, bool] = {}
        schema_buildings = {}

        if isinstance(self.env.schema, Mapping):
            schema_buildings = self.env.schema.get('buildings', {}) or {}

        for member_id in self._member_order:
            include = True
            schema_building = schema_buildings.get(member_id, {}) if isinstance(schema_buildings, Mapping) else {}
            include = parse_bool(
                schema_building.get('include', True),
                default=True,
                path=f'buildings.{member_id}.include',
            )
            flags[member_id] = bool(include)

        return flags

    def _collect_charger_observation_flags(self) -> Mapping[str, bool]:
        observations = {}
        if isinstance(self.env.schema, Mapping):
            observations = self.env.schema.get('observations', {}) or {}

        flags = {}
        for key in (
            'electric_vehicle_charger_connected_state',
            'connected_electric_vehicle_at_charger_departure_time',
            'connected_electric_vehicle_at_charger_required_soc_departure',
            'connected_electric_vehicle_at_charger_soc',
            'connected_electric_vehicle_at_charger_battery_capacity',
            'electric_vehicle_charger_incoming_state',
            'incoming_electric_vehicle_at_charger_estimated_arrival_time',
            'incoming_electric_vehicle_at_charger_estimated_soc_arrival',
        ):
            value = observations.get(key, {}) if isinstance(observations, Mapping) else {}
            active = parse_bool(
                value.get('active', False) if isinstance(value, Mapping) else False,
                default=False,
                path=f'observations.{key}.active',
            )
            flags[key] = bool(active)

        return flags

    def _is_schema_action_active(self, action_name: str) -> bool:
        actions = {}
        if isinstance(self.env.schema, Mapping):
            actions = self.env.schema.get('actions', {}) or {}

        action_data = actions.get(action_name, {}) if isinstance(actions, Mapping) else {}
        return bool(
            parse_bool(
                action_data.get('active', False) if isinstance(action_data, Mapping) else False,
                default=False,
                path=f'actions.{action_name}.active',
            )
        )

    def _is_schema_observation_active(self, observation_name: str) -> bool:
        observations = {}
        if isinstance(self.env.schema, Mapping):
            observations = self.env.schema.get('observations', {}) or {}

        obs_data = observations.get(observation_name, {}) if isinstance(observations, Mapping) else {}
        return bool(
            parse_bool(
                obs_data.get('active', False) if isinstance(obs_data, Mapping) else False,
                default=False,
                path=f'observations.{observation_name}.active',
            )
        )

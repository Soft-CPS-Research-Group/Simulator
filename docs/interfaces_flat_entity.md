# Flat and Entity Interfaces

CityLearn supports two I/O contracts: `flat` for classic Gymnasium-style RL workflows and `entity` for ORL, GraphRL, Transformers and dynamic topology.

Portuguese version: [pt/interfaces_flat_entity.md](pt/interfaces_flat_entity.md).

## Quick Comparison

| Topic | Flat | Entity |
|---|---|---|
| Observation | List of vectors per building or one central vector. | Dict with entity tables and relational edges. |
| Action | List/array ordered by `env.action_names`. | Dict with action tables or an ID-keyed map. |
| Best for | RBC, MARL, wrappers, SB3-style workflows. | GNNs, transformers, dynamic input/output networks. |
| Dynamic topology | Not supported. | Supported. |
| Stable IDs | Implicit through vector order. | Explicit in `env.entity_specs`. |
| EV/charger relation | Expanded flat names. | Separate tables plus `charger_to_ev_*` edges. |

## Flat Mode

```python
env = CityLearnEnv(schema, interface="flat")
```

With `central_agent=False`, observations are one vector per building:

```python
observations = [
    [building_1_feature_1, building_1_feature_2],
    [building_2_feature_1, building_2_feature_2],
]
```

With `central_agent=True`, observations are concatenated into a single vector:

```python
observations = [[all_active_features_for_all_buildings]]
```

`shared_observations` are included once in the central vector to avoid repeated weather, pricing and calendar features.

Flat actions follow the same convention:

```python
actions = [
    [building_1_action_1, building_1_action_2],
    [building_2_action_1, building_2_action_2],
]
```

The exact order is available through:

```python
env.observation_names
env.action_names
```

## Entity Mode

```python
env = CityLearnEnv(schema, interface="entity")
```

Dynamic topology requires entity mode:

```python
env = CityLearnEnv(schema, interface="entity", topology_mode="dynamic")
```

Entity observations have this structure:

```python
{
  "tables": {
    "district": np.ndarray,
    "building": np.ndarray,
    "charger": np.ndarray,
    "ev": np.ndarray,
    "storage": np.ndarray,
    "pv": np.ndarray,
    "deferrable_appliance": np.ndarray
  },
  "edges": {
    "district_to_building": np.ndarray,
    "building_to_charger": np.ndarray,
    "building_to_storage": np.ndarray,
    "building_to_pv": np.ndarray,
    "building_to_deferrable_appliance": np.ndarray,
    "charger_to_ev_connected": np.ndarray,
    "charger_to_ev_connected_mask": np.ndarray,
    "charger_to_ev_incoming": np.ndarray,
    "charger_to_ev_incoming_mask": np.ndarray
  },
  "meta": {
    "time_step": int,
    "endogenous_time_step": int,
    "spec_version": "entity_v1",
    "topology_version": int
  }
}
```

## `entity_specs`

`env.entity_specs` is the machine-readable schema for tables, columns, IDs, units, bundles and edges:

```python
specs = env.entity_specs
building_features = specs["tables"]["building"]["features"]
charger_ids = specs["tables"]["charger"]["ids"]
charger_units = specs["tables"]["charger"]["units"]
```

| Field | Contents |
|---|---|
| `ids` | Stable canonical row IDs. |
| `features` | Column names. |
| `units` | Inferred units. |
| `feature_metadata` | Unit, bundle and legacy flag per feature. |
| `actions` | Action table IDs, columns and units. |
| `edges` | Source/target table metadata. |
| `topology` | Active IDs, lifecycle and topology version. |

## Temporal Semantics

| Field | Meaning |
|---|---|
| Exogenous observations | Read at timestep `t`. Examples: weather, pricing, schedules. |
| Endogenous observations | Read at settled `t-1`. Examples: previous consumption and SOC after the last action. |
| Topology events | Events at `k` are applied after transition `k-1 -> k` and before observation `k`. |

## Entity Bundles

| Bundle | Default | Tables | Purpose |
|---|---:|---|---|
| `entity_base` | always on | charger, storage, deferrable | Essential static and service features. |
| `entity_core_electrical` | off | building, charger, ev, storage, pv | Power, step energy, efficiency, derived SOC and PV. |
| `entity_community_operational` | off | district | Aggregates, headroom, counts and topology version. |
| `entity_forecasts_existing` | off | district | Forecasts already present in the dataset. |
| `entity_temporal_derived` | off | district, building | Short lags and rolling means. |

## Entity Actions

Recommended table payload:

```python
actions = {
    "tables": {
        "building": building_action_array,
        "charger": charger_action_array,
        "deferrable_appliance": deferrable_action_array
    }
}
```

ID-keyed overrides are also supported:

```python
actions = {
    "map": {
        "building:Building_1": {"electrical_storage": 0.2},
        "charger:Building_1:AC001": {"electric_vehicle_storage": 0.5},
        "deferrable_appliance:Building_1:washer_1": {"start": 1.0}
    }
}
```

Use prefixed IDs for robust GraphRL and Transformer integrations.

## Dynamic Topology Guidance

| Element | Agent guidance |
|---|---|
| `topology_version` | Re-read `entity_specs` when it changes. |
| EV masks | Use `charger_to_ev_connected_mask` and `charger_to_ev_incoming_mask`. |
| Running statistics | Maintain stats per feature, not per fixed row. |
| Removed assets | Ignore IDs that disappear from `active_ids`. |
| Added assets | Initialize model memory/hidden state for new IDs. |

# Running Simulations

This page is the operational reference for installing the simulator, creating environments, running episodes, using the CLI and understanding the main execution parameters.

Portuguese version: [pt/running_simulations.md](pt/running_simulations.md).

## Installation

| Case | Command | Notes |
|---|---|---|
| Standard install | `pip install softcpsrecsimulator` | The Python import path is still `citylearn`. |
| Parquet datasets | `pip install pyarrow` | Required only when schemas point to `.parquet`, `.pq` or `.parq`. |
| PV autosizing | `pip install "softcpsrecsimulator[pysam]"` | Required only for EPW/PySAM autosizing. |
| Local development | `.venv/bin/pip install -e .` | Use the repo `.venv` when available. |

## Python Quickstart

```python
import numpy as np
from citylearn.citylearn import CityLearnEnv

schema = "data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
env = CityLearnEnv(schema, interface="flat", episode_time_steps=24, render_mode="none")

observations, info = env.reset()
terminated = truncated = False

while not (terminated or truncated):
    actions = [np.zeros(space.shape, dtype="float32") for space in env.action_space]
    observations, reward, terminated, truncated, info = env.step(actions)

kpis_v2 = env.evaluate_v2()
```

## Entity Interface Quickstart

```python
from citylearn.citylearn import CityLearnEnv

env = CityLearnEnv(
    "data/datasets/citylearn_three_phase_dynamic_topology_demo/schema.json",
    interface="entity",
    topology_mode="dynamic",
)

obs, info = env.reset()
specs = env.entity_specs

actions = {
    "tables": {
        "building": env.action_space["tables"]["building"].sample(),
        "charger": env.action_space["tables"]["charger"].sample(),
        "deferrable_appliance": env.action_space["tables"]["deferrable_appliance"].sample(),
    }
}

obs, reward, terminated, truncated, info = env.step(actions)
```

## `CityLearnEnv` Parameters

| Parameter | Type | Default | Purpose | Notes |
|---|---:|---:|---|---|
| `schema` | `str`, `Path`, `Mapping` | required | Dataset name, schema path or preloaded dict. | Relative paths are resolved from `root_directory`. |
| `root_directory` | path | schema | Base folder for dataset files. | Overrides schema value. |
| `buildings` | list | schema | Subset of buildings to load. | Names or indices. |
| `electric_vehicles` | list | schema | Subset of EVs to load. | Usually from `electric_vehicles_def`. |
| `simulation_start_time_step` | int | schema | First global timestep. | Inclusive. |
| `simulation_end_time_step` | int | schema | Last global timestep. | Inclusive. |
| `episode_time_steps` | int/list | schema | Episode size or explicit windows. | Can be used with rolling/random splits. |
| `rolling_episode_split` | bool | schema | Sequential window episodes. | Useful for training. |
| `random_episode_split` | bool | schema | Random window episodes. | Uses `random_seed`. |
| `seconds_per_time_step` | float | schema | Physical duration of each step. | Examples: 15, 60, 300, 900, 3600. |
| `time_step_ratio` | int/float | inferred | Ratio between control step and dataset spacing. | Normally dataset and schema should match. |
| `reward_function` | class/path | schema | Reward used by `step`. | Supports `reward_function_kwargs`. |
| `reward_function_kwargs` | dict | `{}` | Constructor kwargs for the reward. | Pass-through. |
| `central_agent` | bool | schema | Single controller for all buildings. | `False` returns one vector per building. |
| `shared_observations` | list | schema | Shared observations in central mode. | Included once in central vectors. |
| `active_observations` | list/list[list] | schema | Enable only these observations. | Global or per-building override. |
| `inactive_observations` | list/list[list] | schema/building | Disable observations. | Applied after active selection. |
| `active_actions` | list/list[list] | schema | Enable only these actions. | Global or per-building override. |
| `inactive_actions` | list/list[list] | schema/building | Disable actions. | Applied after active selection. |
| `simulate_power_outage` | bool | schema/building | Enable outage simulation. | Uses data series or stochastic model. |
| `solar_generation` | bool | schema | Compatibility switch for solar generation. | Kept for original CityLearn compatibility. |
| `random_seed` | int | schema | Random seed. | Affects splits and stochastic attributes. |
| `offline` | bool | `False` | Disable network fallbacks. | Requires local datasets. |
| `interface` | `flat`/`entity` | schema/flat | Observation/action contract. | Entity returns tables and edges. |
| `topology_mode` | `static`/`dynamic` | schema/static | Enable dynamic topology events. | Dynamic requires `interface="entity"`. |
| `start_date` | date/string | schema/2024-01-01 | Base date for render/export timestamps. | Does not change physics. |
| `render_mode` | `none`/`during`/`end` | `none` | CSV export policy. | `end` is preferred for performance. |
| `render_session_name` | string | schema/None | Export session subfolder. | Must be relative. |
| `export_kpis_on_episode_end` | bool | render flag | Export KPIs at episode end. | Can be enabled without full render. |

## Extra `**kwargs`

| Parameter | Type | Default | Purpose |
|---|---:|---:|---|
| `render_directory` | path | internal output | Base export folder. |
| `render_directory_name` | string | `render_logs` | Legacy export folder name. |
| `render` | bool | derived | Legacy render switch. |
| `debug_timing` | bool | schema/False | Runtime timing logs. |
| `check_observation_limits` | bool | schema/False | Validate observations against estimated bounds. |
| `physics_invariant_checks` | bool | schema/False | Run physical invariant checks at runtime. |
| `metrics_log_interval` | int | schema/0 | Runtime metric log cadence. |

## CLI

```bash
citylearn --version
citylearn list_datasets
citylearn list_default_time_series_variables
citylearn simulate data/datasets/my_dataset/schema.json train -e 3
citylearn simulate data/datasets/my_dataset/schema.json evaluate
```

| Option | Example | Purpose |
|---|---|---|
| `schema` | dataset name or `schema.json` path | Dataset to run. |
| `-a`, `--agent_name` | `citylearn.agents.baseline.BusinessAsUsualAgent` | Agent class path. |
| `-ke`, `--env_kwargs` | `'{"interface":"entity"}'` | JSON kwargs for `CityLearnEnv`. |
| `-ka`, `--agent_kwargs` | `'{"x":1}'` | JSON kwargs for the agent. |
| `-w`, `--wrappers` | wrapper class paths | Gymnasium wrappers. |
| `-tv`, `--time_series_variables` | `net_electricity_consumption` | Series stored after evaluation. |
| `-sid`, `--simulation_id` | `run_001` | Output naming ID. |
| `-fa`, `--agent_filepath` | `outputs/agent.pkl` | Load/save agent path. |
| `-d`, `--output_directory` | `outputs/run_001` | Output folder. |
| `-te`, `--evaluation_episode_time_steps` | `0 2879` | Evaluation window. Can repeat. |
| `-p`, `--append` | flag | Do not overwrite existing output. |
| `-rs`, `--random_seed` | `42` | Seed. |
| `--offline` | flag | Require local files. |
| `train -e` | `train -e 10` | Number of training episodes. |
| `train --save_agent` | flag | Save agent at the end. |
| `train --evaluate` | flag | Evaluate after training. |
| `evaluate` | subcommand | Deterministic evaluation. |

## Render and Export

| `render_mode` | Runtime cost | Output | Use case |
|---|---:|---|---|
| `none` | lowest | No render CSV | Training. |
| `during` | high | Writes rows each step | Short debugging runs. |
| `end` | medium | Writes full episode at end | Long episodes with final CSV output. |

## Recommended Validation

```bash
.venv/bin/pytest -q
.venv/bin/python scripts/audit/audit_entity_contract.py --strict
.venv/bin/python scripts/audit/audit_physics.py
.venv/bin/python -m ruff check citylearn tests scripts/manual scripts/ci --select E9,F821
```

For large 15s datasets, always smoke-test a short window first:

```bash
.venv/bin/python - <<'PY'
import numpy as np
from citylearn.citylearn import CityLearnEnv

env = CityLearnEnv(
    "data/datasets/citylearn_three_phase_electrical_service_demo_15s_parquet/schema.json",
    simulation_start_time_step=0,
    simulation_end_time_step=120,
    episode_time_steps=120,
)
obs, info = env.reset()
for _ in range(100):
    if isinstance(env.action_space, list):
        actions = [np.zeros(space.shape, dtype="float32") for space in env.action_space]
    else:
        actions = {
            "tables": {
                name: np.zeros(space.shape, dtype="float32")
                for name, space in env.action_space["tables"].spaces.items()
            }
        }
    obs, reward, terminated, truncated, info = env.step(actions)
    if terminated or truncated:
        break
print(env.evaluate_v2().head())
PY
```

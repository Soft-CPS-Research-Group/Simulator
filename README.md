# CityLearn
CityLearn is an open source Farama Foundation Gymnasium environment for the implementation of Multi-Agent Reinforcement Learning (RL) for building energy coordination and demand response in cities. A major challenge for RL in demand response is the ability to compare algorithm performance. Thus, CityLearn facilitates and standardizes the evaluation of RL agents such that different algorithms can be easily compared with each other.

![Demand-response](https://github.com/intelligent-environments-lab/CityLearn/blob/master/assets/images/dr.jpg)

## Environment Overview

CityLearn includes energy models of buildings and distributed energy resources (DER) including air-to-water heat pumps, electric heaters and batteries. A collection of building energy models makes up a virtual district (a.k.a neighborhood or community). In each building, space cooling, space heating and domestic hot water end-use loads may be independently satisfied through air-to-water heat pumps. Alternatively, space heating and domestic hot water loads can be satisfied through electric heaters.

![Citylearn](https://github.com/intelligent-environments-lab/CityLearn/blob/master/assets/images/environment.jpg)

## Documentation Map

This fork extends CityLearn with sub-hourly physics, EVs, deferrable appliances, entity observations, dynamic topology, three-phase electrical service, community KPIs and large-dataset performance work. The documentation is organized as reference pages that are easy to scan by humans and external agents.

Default documentation is in English. Portuguese versions are available under [docs/pt](docs/pt/).

| Page | Portuguese | Use it for |
|---|---|---|
| [How to run simulations](docs/running_simulations.md) | [PT](docs/pt/running_simulations.md) | Install, Python quickstart, CLI, every `CityLearnEnv` parameter, render/export and validation commands. |
| [Schema reference](docs/schema_reference.md) | [PT](docs/pt/schema_reference.md) | Every major schema section: observations, actions, buildings, devices, PV modes, EVs, chargers, deferrables, topology and community market. |
| [Flat and entity interfaces](docs/interfaces_flat_entity.md) | [PT](docs/pt/interfaces_flat_entity.md) | Difference between vector mode and entity-table mode, payload format, entity specs and dynamic topology semantics. |
| [Actions reference](docs/actions_reference.md) | [PT](docs/pt/actions_reference.md) | Flat/entity action names, ranges, physical meaning and conversion to energy. |
| [Observations reference](docs/observations_reference.md) | [PT](docs/pt/observations_reference.md) | Observation dictionary with units, bundles, sentinels, entity tables and edges. |
| [Dataset reference](docs/dataset_reference.md) | [PT](docs/pt/dataset_reference.md) | Required files/columns, CSV vs Parquet, 15s datasets, real-data conversion from kW to kWh/step. |
| [KPIs reference](docs/kpis_reference.md) | [PT](docs/pt/kpis_reference.md) | `evaluate()` vs `evaluate_v2()`, KPI units, equations, EV/BESS/deferrable/community/phase KPIs. |
| [KPI v2 naming tree](docs/KPI_V2_TREE.md) | - | Full naming convention and tree for structured KPI names. |
| [Data unit contract](docs/data_unit_contract.md) | [PT](docs/pt/data_unit_contract.md) | Formal unit contract for data, power limits and prices/emissions. |
| [Simulator features](docs/features.md) | [PT](docs/pt/features.md) | Capability inventory, including hidden/less obvious features. |
| [Releases](docs/releases.md) | [PT](docs/pt/releases.md) | Release policy, changelog template and version history for this fork. |

## Current Capability Snapshot

| Area | Supported |
|---|---|
| Time resolution | Hourly and sub-hourly, including 15min, 5min, 1min and 15s fixtures. |
| Dataset formats | CSV and Parquet with equivalent schema columns. |
| Real data | Power data can be converted to `kWh/step`; PV supports absolute measured generation. |
| PV | `per_kw` normalized profile mode and `absolute` measured-energy mode. |
| EVs | Charger-centric schedules, connected/incoming EVs, SOC requirements, V2G-capable action path. |
| Deferrable appliances | Normalized cycle catalog plus sparse flexibility schedule. |
| Interfaces | Flat Gymnasium-style vectors and entity tables/edges for ORL, GraphRL and Transformers. |
| Dynamic topology | Add/remove buildings and assets during simulation in entity mode. |
| Three phase | Phase connections, headroom, current phase power, violations and phase KPIs. |
| Community market | Local settlement, import weights, savings and self-consumption KPIs. |
| Performance | Windowed loading, shared weather/pricing/carbon cache, Parquet for large 15s datasets. |
| Validation | Unit tests, golden KPI tests, physics audit and strict entity contract audit. |

## Installation

Install latest release in PyPI with `pip`:
```console
pip install softcpsrecsimulator
```

Optional dependency for PV autosizing (`PySAM`):
```console
pip install "softcpsrecsimulator[pysam]"
```

Python import path remains:
```python
from citylearn.citylearn import CityLearnEnv
```

Optional dependency for Parquet datasets:
```console
pip install pyarrow
```

## Quickstart

```python
import numpy as np
from citylearn.citylearn import CityLearnEnv

env = CityLearnEnv(
    "data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json",
    interface="flat",
    episode_time_steps=24,
    render_mode="none",
)

observations, info = env.reset()
terminated = truncated = False

while not (terminated or truncated):
    actions = [np.zeros(space.shape, dtype="float32") for space in env.action_space]
    observations, reward, terminated, truncated, info = env.step(actions)

kpis = env.evaluate_v2()
```

Entity interface quickstart:

```python
from citylearn.citylearn import CityLearnEnv

env = CityLearnEnv(
    "data/datasets/citylearn_three_phase_dynamic_topology_demo/schema.json",
    interface="entity",
    topology_mode="dynamic",
)

observations, info = env.reset()
specs = env.entity_specs
```

## Unit Contract

| Quantity | Unit |
|---|---|
| Dataset energy columns | `kWh/step` |
| PV `generation_mode="absolute"` | `kWh/step` |
| PV `generation_mode="per_kw"` | `W/kW` profile multiplied by installed power |
| Power limits and device ratings | `kW` |
| Prices | currency/kWh |
| Carbon intensity | kgCO2/kWh |
| Deferrable cycle `load_profile` | `kWh/step` |

For real power data:

```text
kWh_per_step = kW * seconds_per_time_step / 3600
```

## Developer Commands
Use the repository virtual environment when available:

```console
.venv/bin/pytest -q
```

Critical lint checks used in CI:

```console
.venv/bin/python -m ruff check citylearn tests scripts/manual scripts/ci --select E9,F821
```

Manual utility scripts live in `scripts/manual` and are excluded from default pytest collection:

```console
python scripts/manual/demo_ev_rbc.py
python scripts/manual/demo_ev_rbc_export_end.py
```

Runtime benchmark (main training resolutions 5s/60s):

```console
python scripts/manual/bench_runtime.py --seconds 5 60 --render-modes none end --episode-steps 1200
```

CI performance smoke check command:

```console
python scripts/ci/perf_smoke.py --episode-steps 600 --seconds 60 --baseline-file scripts/ci/perf_baseline.json
```

## Publish This Fork to PyPI
This fork is configured to publish the distribution name `softcpsrecsimulator`.

1. Create the package on PyPI (project name: `softcpsrecsimulator`).
2. In GitHub repo settings, add secret `PYPI_API_TOKEN` with a PyPI token that can publish this project.
3. Bump `citylearn/__init__.py` version.
4. Push commit and create a GitHub Release (or run `Publish Python Package` workflow manually).
5. Workflow `.github/workflows/pypi_deploy.yml` builds `dist/*` and uploads to PyPI.

Optional local build check:
```console
python -m pip install --upgrade pip build twine
python -m build
python -m twine check dist/*
```

## Internal Architecture
Public APIs remain in `CityLearnEnv` and `Building`, while internal orchestration is split into service modules under `citylearn/internal`:

- `loading.py`: schema-driven loading/build assembly (`_load*`, metadata processing).
- `runtime.py`: episode runtime orchestration (`step`, action parsing, time progression, EV/charger association).
- `building_ops.py`: building observation/action orchestration.
- `kpi.py`: KPI/evaluation pipeline.

## Export and Render Modes
`CityLearnEnv` keeps export off by default:

- `render_mode='none'`: no CSV export and minimal runtime overhead.
- `render_mode='during'`: writes CSV rows at each environment step.
- `render_mode='end'`: keeps the rollout fast and writes full episode CSVs when the episode ends.

Optional location controls:

- `render_directory`: base folder for exports.
- `render_session_name`: session subfolder name under `render_directory` (or `render_directory_name`).
- `render_directory_name`: legacy fallback folder under project root when `render_directory` is not set.
- `render`: legacy boolean flag; still supported for compatibility.

Examples:

```python
from citylearn.citylearn import CityLearnEnv

# Fast training/no export
env = CityLearnEnv(schema, render_mode='none')

# Stream CSVs every step
env = CityLearnEnv(
    schema,
    render_mode='during',
    render_directory='outputs',
    render_session_name='run_during'
)

# Export once at episode end
env = CityLearnEnv(
    schema,
    render_mode='end',
    render_directory='outputs',
    render_session_name='run_end'
)
```

## Release Discipline

Every release should update [docs/releases.md](docs/releases.md) and any affected reference page.

| Version type | When to use |
|---|---|
| Patch | Additive compatible features, fixes, docs and tests. |
| Minor | New simulator capability or schema/API change. |
| Major | Broad breaking changes. |

Recommended pre-release checks:

```console
.venv/bin/pytest -q
.venv/bin/python scripts/audit/audit_entity_contract.py --strict
.venv/bin/python scripts/audit/audit_physics.py
```

## Upstream Documentation

The original CityLearn documentation remains useful for base concepts and examples: [official docs](https://intelligent-environments-lab.github.io/CityLearn/). This fork's local reference pages above document the additional simulator contracts.

## CityLearn UI

CityLearn UI is a visual dashboard for exploring simulation data generated by the CityLearn framework. It was developed to simplify the analysis of results from smart energy communities, district energy coordination, demand response (among other applications), allowing users to visually inspect building-level components, compare simulation KPIs, and create simulation schemas with ease.

The interface is available in two options:

* Web app: https://citylearnui.netlify.app/ (free hosted version — not recommended for sensitive/personal data)
* Open-source code: https://github.com/Soft-CPS-Research-Group/citylearn-ui

You can check a tutorial at the official CityLearn [website](https://intelligent-environments-lab.github.io/CityLearn/ui.html), in the CityLearn UI repository [README](https://github.com/Soft-CPS-Research-Group/citylearn-ui), or at the help [tooltip of the oficial webapp](https://citylearn-ui.netlify.app/admin/help).

**Compatibility:** This version of the UI currently supports CityLearn v2.5.0 simulation data.

The KPI export consumed by the UI is generated from `evaluate_v2()` by default, and `evaluate()` remains available only for legacy workflows.

**Developed by:** José, a member of the [SoftCPS](https://www2.isep.ipp.pt/softcps/), Software for Cyber-Physical Systems research group (ISEP, Portugal) in collaboration with the Intelligent Environments Lab, University of Texas at Austin.

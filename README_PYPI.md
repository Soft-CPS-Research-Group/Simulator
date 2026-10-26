# SoftCPS REC Simulator

`softcpsrecsimulator` is a simulation framework maintained by the SoftCPS Research Group for
energy-community studies with reinforcement learning.

It is a fork-based project used for research and experimentation on:

- electric vehicles (EVs) and chargers,
- stationary batteries (BESS),
- photovoltaic generation (PV),
- electrical-service constraints (single/three-phase),
- local community market settlement and KPI analysis.
- entity-mode RL observation contracts with derived forecasts, physical deadlines,
  feasible action capacity and requested/limited/applied action feedback.
- dynamic member and asset topology with repeatable multi-episode reset semantics.

Current source release: `1.8.3`. EV arrival SOC is visible immediately in flat and
entity observations. Runtime community datasets preserve the flexibility known
at each historical timestamp through their exported charger time series.

## Project Positioning

This package is a **fork project** built from the CityLearn codebase and extended for SoftCPS REC
simulation needs.

## Installation

```bash
pip install softcpsrecsimulator
```

## Python Usage

For compatibility with existing ecosystems, the Python module path currently remains:

```python
from citylearn.citylearn import CityLearnEnv
```

## Source and Documentation

- Source: https://github.com/Soft-CPS-Research-Group/Simulator
- Original upstream project: https://github.com/intelligent-environments-lab/CityLearn

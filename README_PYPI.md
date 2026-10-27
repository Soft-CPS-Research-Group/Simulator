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

Current source release: `1.9.0`. This release adds passive energy provenance
through stationary batteries, EVs and community sharing, useful solar-energy
indicators and carbon accounting at the external grid boundary. EV departure
feasibility uses the concrete battery physics. KPI exports include reproducibility
manifests. Existing instantaneous PV and gross-import emissions remain available
with their historical meanings. See the release notes for validation and limits.

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

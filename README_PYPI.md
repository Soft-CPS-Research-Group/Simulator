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

Current source release: `2.2.0`. It aligns the simulator with CityLearn 3.5.0:
new business-as-usual (BAU), Basic, Smart and Community baselines, unlimited
electrical headroom published as 1,000,000 kW, and nested dataset downloads.
BAU-relative KPIs change; see the
[migration guide](https://github.com/Soft-CPS-Research-Group/Simulator/blob/master/docs/migration.md#upgrade-from-softcps-211-to-220)
and the [release history](https://github.com/Soft-CPS-Research-Group/Simulator/blob/master/docs/releases.md).

## Project Positioning

This package distributes the same simulator code as
[CityLearn](https://github.com/citylearn-project/CityLearn), under the
`softcpsrecsimulator` name, with additional SoftCPS datasets.

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
- CityLearn: https://github.com/citylearn-project/CityLearn

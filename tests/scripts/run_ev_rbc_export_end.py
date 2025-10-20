#!/usr/bin/env python3
"""EV reference controller run with exports at the end of the episode."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from citylearn.agents.rbc import BasicElectricVehicleRBC_ReferenceController as Agent  # noqa: E402
from citylearn.citylearn import CityLearnEnv  # noqa: E402

SCHEMA = ROOT / "data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"


def main() -> None:
    render_root = ROOT / "SimulationData"
    env = CityLearnEnv(
        str(SCHEMA),
        central_agent=True,
        episode_time_steps=96,
        render_mode="end",
        render_directory=render_root,
        render_session_name="rbc_export_end_example2",
        random_seed=0,
    )

    try:
        controller = Agent(env)
        observations, _ = env.reset()

        while not env.terminated:
            actions = controller.predict(observations, deterministic=True)
            observations, _, terminated, truncated, _ = env.step(actions)
            if terminated or truncated:
                break

        env.export_final_kpis(controller)

        outputs_path = Path(env.new_folder_path)
        print(f"Exports written to: {outputs_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

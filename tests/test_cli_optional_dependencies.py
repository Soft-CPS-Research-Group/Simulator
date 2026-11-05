"""The public CLI must start without optional reinforcement-learning packages."""
import subprocess
import sys
import json
from pathlib import Path

import pytest


def test_cli_starts_without_stable_baselines3():
    code = """
import importlib.abc
import sys
class BlockOptionalAgent(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'stable_baselines3' or fullname.startswith('stable_baselines3.'):
            raise ModuleNotFoundError('Optional package deliberately unavailable')
sys.meta_path.insert(0, BlockOptionalAgent())
from citylearn.__main__ import main
sys.argv = ['sphinx-build']
parser = main()
simulate = next(a for a in parser._actions if a.dest == 'subcommands').choices['simulate']
agent = next(a for a in simulate._actions if a.dest == 'agent_name')
assert agent.default == 'citylearn.agents.baseline.BusinessAsUsualAgent'
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("operation", ["train", "evaluate"])
def test_cli_runs_short_baseline_episode_on_supported_python(tmp_path, operation):
    from citylearn.__main__ import Simulator

    schema = Path(__file__).parent / "data/minute_ev_demo/schema.json"
    kwargs = dict(schema=str(schema), offline=True,
                  agent_name="citylearn.agents.baseline.BusinessAsUsualAgent",
                  output_directory=tmp_path, simulation_id="cli-smoke", random_seed=0,
                  env_kwargs=dict(episode_time_steps=4, render_mode="none",
                                  export_kpis_on_episode_end=False))
    getattr(Simulator, operation)(**kwargs)
    report = json.loads((tmp_path / f"cli-smoke-{'evaluation' if operation == 'evaluate' else 'train'}.json").read_text())
    assert report[f"{'evaluation' if operation == 'evaluate' else 'train'}_start_timestamp"]
    assert report[f"{'evaluation' if operation == 'evaluate' else 'train'}_end_timestamp"]
    if operation == "evaluate":
        assert report["time_series"]["time_step"] == [0, 1, 2]
        assert all(len(values) == 3 for values in report["time_series"].values())

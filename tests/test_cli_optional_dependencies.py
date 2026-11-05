"""The public CLI must start without optional reinforcement-learning packages."""
import subprocess
import sys


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

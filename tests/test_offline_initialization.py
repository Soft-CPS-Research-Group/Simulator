import shutil
from pathlib import Path

import pytest

from citylearn.citylearn import CityLearnEnv, UnknownSchemaError
from citylearn.data import DataSet

pytestmark = pytest.mark.use_real_dataset_resolution

ROOT = Path(__file__).resolve().parents[1]
MINUTE_SCHEMA = ROOT / "tests" / "data" / "minute_ev_demo" / "schema.json"


def _raise_if_network_used(*args, **kwargs):
    raise AssertionError("Network access should not be attempted in offline tests.")


def test_offline_initialization_uses_local_dataset_name_without_network(tmp_path, monkeypatch):
    local_datasets_root = tmp_path / "datasets"
    dataset_name = "offline_demo"
    dataset_dir = local_datasets_root / dataset_name
    local_datasets_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(MINUTE_SCHEMA.parent, dataset_dir)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("CITYLEARN_LOCAL_DATASETS_PATH", str(local_datasets_root))
    monkeypatch.setattr(DataSet, "get_requests_session", staticmethod(_raise_if_network_used))

    env = CityLearnEnv(dataset_name, offline=True, central_agent=True, episode_time_steps=2, render_mode="none")

    assert len(env.buildings) == 1
    assert env.root_directory.endswith(dataset_name)


def test_offline_initialization_raises_clear_error_for_missing_dataset(tmp_path, monkeypatch):
    local_datasets_root = tmp_path / "datasets"
    local_datasets_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("CITYLEARN_LOCAL_DATASETS_PATH", str(local_datasets_root))
    monkeypatch.setattr(DataSet, "get_requests_session", staticmethod(_raise_if_network_used))

    with pytest.raises(UnknownSchemaError, match="offline mode is enabled"):
        CityLearnEnv("missing_offline_dataset", offline=True, central_agent=True, episode_time_steps=2, render_mode="none")


def test_offline_initialization_skips_sizing_data_when_autosize_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(DataSet, "get_requests_session", staticmethod(_raise_if_network_used))

    def _unexpected_pv(*args, **kwargs):
        raise AssertionError("PV sizing data should not be loaded when autosize is disabled.")

    def _unexpected_battery(*args, **kwargs):
        raise AssertionError("Battery sizing data should not be loaded when autosize is disabled.")

    monkeypatch.setattr(DataSet, "get_pv_sizing_data", _unexpected_pv)
    monkeypatch.setattr(DataSet, "get_battery_sizing_data", _unexpected_battery)

    env = CityLearnEnv(str(MINUTE_SCHEMA), offline=True, central_agent=True, episode_time_steps=2, render_mode="none")

    assert len(env.buildings) == 1

"""Online datasets must include nested files and never cache partial downloads."""
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from citylearn.data import DataSet

pytestmark = pytest.mark.use_real_dataset_resolution


@pytest.fixture
def remote_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(DataSet, "_find_local_dataset_root", lambda *args: None)
    dataset = DataSet(offline=False)
    root = "data/datasets/nested_demo"
    # The parent listing gives the dataset's Git tree, which one recursive
    # trees request lists in full.
    listing = {"data/datasets": [dict(type="dir", name="nested_demo", path=root, sha="tree-sha")]}
    tree = [
        dict(type="blob", path="schema.json"),
        dict(type="tree", path="shared"),
        dict(type="blob", path="shared/pricing.parquet"),
        dict(type="tree", path="shared/deeper"),
        dict(type="blob", path="shared/deeper/events.json"),
    ]
    contents = Mock(side_effect=lambda path: listing[path])
    monkeypatch.setattr(dataset, "get_github_contents", contents)
    monkeypatch.setattr(dataset, "get_github_tree", Mock(side_effect=lambda sha: list(tree)))
    payloads = {"schema.json": b'{"pricing": "shared/pricing.parquet"}', "pricing.parquet": b"PAR1-test", "events.json": b"[]"}
    statuses = {}

    def respond(url, **kwargs):
        response = requests.Response()
        key = url.rsplit("/", 1)[-1]
        response.status_code = statuses.get(key, 200)
        response._content = payloads[key]
        response.url = url
        return response

    session = requests.Session()
    monkeypatch.setattr(session, "get", Mock(side_effect=respond))
    monkeypatch.setattr(dataset, "get_requests_session", lambda: session)
    return dataset, tree, statuses, contents


def test_download_includes_all_nested_files_and_reuses_complete_cache(remote_dataset):
    dataset, _, _, contents = remote_dataset
    schema = Path(dataset.get_dataset("nested_demo"))
    assert schema.is_file()
    assert (schema.parent / "shared/pricing.parquet").read_bytes() == b"PAR1-test"
    assert (schema.parent / "shared/deeper/events.json").read_bytes() == b"[]"
    contents.reset_mock()
    assert dataset.get_dataset("nested_demo") == str(schema)
    contents.assert_not_called()


def test_failed_nested_download_leaves_no_partial_cache_and_can_retry(remote_dataset):
    dataset, _, statuses, _ = remote_dataset
    statuses["pricing.parquet"] = 503
    with pytest.raises(requests.HTTPError):
        dataset.get_dataset("nested_demo")
    cache = Path(dataset.cache_directory) / "datasets"
    assert not list(cache.iterdir())
    statuses.clear()
    assert Path(dataset.get_dataset("nested_demo")).is_file()


def test_missing_schema_does_not_create_valid_cache(remote_dataset):
    dataset, tree, _, _ = remote_dataset
    tree.pop(0)
    with pytest.raises(FileNotFoundError, match="schema.json"):
        dataset.get_dataset("nested_demo")
    assert not list((Path(dataset.cache_directory) / "datasets").iterdir())


@pytest.mark.parametrize("path", ["../escape/schema.json", "shared/../../schema.json", "/etc/schema.json"])
def test_remote_paths_cannot_escape_dataset_directory(remote_dataset, path):
    dataset, tree, _, _ = remote_dataset
    tree[:] = [dict(type="blob", path=path)]
    with pytest.raises(ValueError):
        dataset.get_dataset("nested_demo")
    cache = Path(dataset.cache_directory) / "datasets"
    assert not cache.exists() or not list(cache.iterdir())

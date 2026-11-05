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
    listing = {
        root: [
            dict(type="file", path=f"{root}/schema.json", download_url="https://example.test/schema"),
            dict(type="dir", path=f"{root}/shared"),
        ],
        f"{root}/shared": [
            dict(type="file", path=f"{root}/shared/pricing.parquet", download_url="https://example.test/pricing"),
            dict(type="dir", path=f"{root}/shared/deeper"),
        ],
        f"{root}/shared/deeper": [
            dict(type="file", path=f"{root}/shared/deeper/events.json", download_url="https://example.test/events"),
        ],
    }
    contents = Mock(side_effect=lambda path: listing[path])
    monkeypatch.setattr(dataset, "get_github_contents", contents)
    payloads = {"schema": b'{"pricing": "shared/pricing.parquet"}', "pricing": b"PAR1-test", "events": b"[]"}
    statuses = {}

    def respond(url):
        response = requests.Response()
        key = url.rsplit("/", 1)[-1]
        response.status_code = statuses.get(key, 200)
        response._content = payloads[key]
        response.url = url
        return response

    session = requests.Session()
    monkeypatch.setattr(session, "get", Mock(side_effect=respond))
    monkeypatch.setattr(dataset, "get_requests_session", lambda: session)
    return dataset, listing, statuses, contents


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
    statuses["pricing"] = 503
    with pytest.raises(requests.HTTPError):
        dataset.get_dataset("nested_demo")
    cache = Path(dataset.cache_directory) / "datasets"
    assert not list(cache.iterdir())
    statuses.clear()
    assert Path(dataset.get_dataset("nested_demo")).is_file()


def test_missing_schema_does_not_create_valid_cache(remote_dataset):
    dataset, listing, _, _ = remote_dataset
    listing["data/datasets/nested_demo"].pop(0)
    with pytest.raises(FileNotFoundError, match="schema.json"):
        dataset.get_dataset("nested_demo")
    assert not list((Path(dataset.cache_directory) / "datasets").iterdir())


@pytest.mark.parametrize("path", ["data/datasets/nested_demo/../escape", "data/datasets/another/file"])
def test_remote_paths_cannot_escape_dataset_directory(remote_dataset, path):
    dataset, listing, _, _ = remote_dataset
    listing["data/datasets/nested_demo"] = [dict(type="file", path=path, download_url="https://example.test/schema")]
    with pytest.raises(ValueError):
        dataset.get_dataset("nested_demo")
    assert not list((Path(dataset.cache_directory) / "datasets").iterdir())

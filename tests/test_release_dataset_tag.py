"""Release tag collisions must not switch online data to upstream CityLearn."""
import json
import os
import subprocess
import sys

import pytest

from citylearn import __version__


@pytest.mark.parametrize("override", [None, "v1.9.13", "custom-pinned-data"])
def test_dataset_release_tag_preserves_fork_and_explicit_overrides(override):
    env = dict(os.environ)
    if override is None:
        env.pop("CITYLEARN_DATASET_TAG", None)
    else:
        env["CITYLEARN_DATASET_TAG"] = override
    code = """
import json
from citylearn.data import DataSet
print(json.dumps({"default": DataSet(offline=True).tag,
                  "explicit": DataSet(tag="explicit-data-ref", offline=True).tag}))
"""
    result = subprocess.run([sys.executable, "-c", code], env=env,
                            capture_output=True, text=True, check=True)
    actual = json.loads(result.stdout)
    assert actual["default"] == (override or f"softcpsrecsimulator-v{__version__}")
    assert actual["explicit"] == "explicit-data-ref"

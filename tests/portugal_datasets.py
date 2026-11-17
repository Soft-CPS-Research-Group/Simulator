"""Locate the Portugal REC datasets for the tests.

The repository ships CityLearn's ``pt_rec_*`` datasets, which hold only the
files the schemas read. The SoftCPS generator scripts still build the full
``rec_pt_2023_*_v2`` packages (catalogues, time series, mobility ledgers,
preparation views, checksums) into ``data/datasets`` as a local, untracked
working copy (see ``.gitignore``). Tests of those construction artefacts run
when such a build is present and are skipped otherwise.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / 'data' / 'datasets'

# Generator family -> shipped dataset and the prefix of its scenario names.
FAMILIES = {
    'micro_4': ('pt_rec_4', 'rec_4'),
    'core_15': ('pt_rec_15', 'rec_15'),
    'core_30': ('pt_rec_30', 'rec_30'),
    'premium_100': ('pt_rec_100', 'rec_100'),
}
# Generator scenario prefixes, which differ from the family name for Micro-4.
_SCENARIO_PREFIXES = {'micro_4': 'core_4'}

SHIPPED = [DATASETS / name for name, _ in FAMILIES.values()]
SHIPPED_SCHEMAS = [p for root in SHIPPED for p in sorted((root / 'evaluation').glob('*.json'))]

BUILD = sorted(DATASETS.glob('rec_pt_2023_*_v2'))
requires_build = pytest.mark.skipif(
    not BUILD,
    reason='needs a Portugal generator build (rec_pt_2023_*_v2), which is not shipped; '
           'build it into data/datasets with the scripts/*portugal* generator',
)


def shipped_root(family: str) -> Path:
    """Shipped dataset of a generator family, e.g. ``core_30`` -> ``pt_rec_30``."""
    return DATASETS / FAMILIES[family][0]


def shipped_variant(variant: str) -> str:
    """Shipped scenario name of a generator scenario, e.g. ``core_30_topology`` -> ``rec_30_topology``."""
    for family, (_, prefix) in FAMILIES.items():
        for old in (_SCENARIO_PREFIXES.get(family, family), family):
            if variant == old or variant.startswith(old + '_'):
                return prefix + variant[len(old):]
    raise KeyError(variant)


def shipped_path(path) -> Path:
    """Map a ``data/datasets/rec_pt_2023_<family>_v2/...`` path to the shipped dataset."""
    parts = Path(path).parts
    index = next(i for i, p in enumerate(parts) if p.startswith('rec_pt_2023_') and p.endswith('_v2'))
    family = parts[index][len('rec_pt_2023_'):-len('_v2')]
    mapped = []
    for part in parts[index + 1:]:
        stem, dot, suffix = part.partition('.')
        try:
            mapped.append(shipped_variant(stem) + dot + suffix)
        except KeyError:
            mapped.append(part)
    return shipped_root(family).joinpath(*mapped)

"""Apply the common nonnegative OMIE tariff and enable native local trading.

This is the final, offline step after building the two-year Portugal inputs.
It changes prices, market configuration and their metadata, not physical inputs.
Archived retail references remain provenance and are never read as active prices.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'portugal-native-2023-2024-v3.2'
PRICING = 'shared/pricing_omie.parquet'
POLICY = 'omie_nonnegative'
MARKET = {'enabled': True, 'local_price_ratio_to_grid_import': 0.8,
          'grid_export_price': 0.0}
HORIZONS = (4, 24, 96)
TOTAL = 70176
UTC = pd.date_range('2023-01-01', periods=TOTAL, freq='15min', tz='UTC')
STORY = ('Tarifa de referência comum a todos os membros: OMIE em EUR/MWh dividido '
         'por 1000, com negativos a zero; mercado comunitário ativo a 80% desse preço.')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def load_market(reference_root=ROOT):
    """Read Portuguese wholesale EUR/MWh with one external forecast-support day."""
    pieces, paths = [], []
    for year in (2023, 2024):
        path = reference_root / f'data/portugal_reference_{year}/omie_{year}_hourly.parquet'
        frame = pd.read_parquet(path)
        # The archived 2024 builder emitted the UTC index as a column named index.
        frame = frame.set_index('timestamp_utc' if 'timestamp_utc' in frame else 'index')
        pieces.append(frame.price_portugal_eur_per_mwh)
        paths.append(path)
    support = reference_root / 'data/portugal_reference_2024/forecast_support'
    support_paths = [*(reference_root/'data/portugal_reference_2024/raw_2024').glob('omie_2025-*.txt'),
                     *support.glob('omie_*.txt')]
    for path in sorted(support_paths):
        day = path.stem.removeprefix('omie_')
        rows = [line.split(';') for line in path.read_text().splitlines()]
        rows = [r for r in rows if len(r) >= 6 and r[0].isdigit()]
        start = pd.Timestamp(day).tz_localize('Europe/Madrid')
        end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize('Europe/Madrid')
        clock = pd.date_range(start, end, freq='h', inclusive='left').tz_convert('UTC')
        if [int(r[3]) for r in rows] != list(range(1, len(clock)+1)):
            raise ValueError(f'Incomplete OMIE market day: {path}')
        pieces.append(pd.Series([float(r[5]) for r in rows], index=clock))
        paths.append(path)
    raw = pd.concat(pieces).sort_index()
    if raw.groupby(level=0).nunique().gt(1).any():
        raise ValueError('Conflicting OMIE observations for the same timestamp')
    raw = raw[~raw.index.duplicated()]
    hours = pd.date_range('2023-01-01', '2025-01-02', freq='h', inclusive='left', tz='UTC')
    raw = raw.reindex(hours)
    if not np.isfinite(raw.to_numpy()).all():
        raise ValueError('OMIE must cover every simulation and forecast-support hour')
    clock = pd.date_range('2023-01-01', '2025-01-02', freq='15min', inclusive='left', tz='UTC')
    return raw.reindex(clock, method='ffill'), paths


def build_prices(market, issue_clock=UTC):
    """Convert once and clip; forecasts use published day-ahead or prior-day OMIE."""
    if not market.index.is_unique or not market.index.is_monotonic_increasing:
        raise ValueError('Market timestamps must be unique and increasing')
    values = market.reindex(issue_clock).to_numpy(dtype=float)
    result = {'electricity_pricing': np.maximum(values / 1000.0, 0.0)}
    boundary = []
    for k, horizon in enumerate(HORIZONS, 1):
        target = issue_clock + pd.Timedelta(minutes=15*horizon)
        madrid = target.tz_convert('Europe/Madrid')
        publication = (madrid.tz_localize(None).normalize() - pd.Timedelta(days=1)
                       + pd.Timedelta(hours=14)).tz_localize('Europe/Madrid').tz_convert('UTC')
        available = issue_clock >= publication
        fallback = target - pd.Timedelta(days=1)
        # At the start of the archive use the first available observation.
        fallback = pd.DatetimeIndex(np.maximum(fallback.asi8, market.index[0].value), tz='UTC')
        source = pd.DatetimeIndex(np.where(available, target, fallback))
        forecast = np.maximum(market.reindex(source).to_numpy(dtype=float)/1000.0, 0.0)
        result[f'electricity_pricing_predicted_{k}'] = forecast
        mask = target > issue_clock[-1]
        for i in np.flatnonzero(mask):
            boundary.append({'issue_utc': str(issue_clock[i]), 'target_utc': str(target[i]),
                             'horizon_steps': horizon, 'publication_utc': str(publication[i]),
                             'omie_source_utc': str(source[i]), 'day_ahead_available': bool(available[i]),
                             'forecast_eur_kwh': float(forecast[i])})
    frame = pd.DataFrame(result).astype('float32')
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Missing OMIE observation for a realised price or forecast')
    return frame, boundary


def refresh_checksums(root):
    target = root / 'file_checksums.sha256'
    target.write_text(''.join(f'{digest(p)}  {p.relative_to(root)}\n'
                              for p in sorted(root.rglob('*')) if p.is_file() and p != target))


def apply(repo=ROOT, reference_root=ROOT):
    market, sources = load_market(reference_root)
    prices, boundary = build_prices(market)
    roots = sorted((repo / 'data/datasets').glob('rec_pt_2023_*_v2'))
    if len(roots) != 4:
        raise ValueError('Expected the four Portuguese dataset families')
    manifest_path = repo / 'results/portugal_rec_native/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    entries = {e['variant']: e for e in manifest['variants']}
    for root in roots:
        legacy = root / 'shared/pricing_btn_bi.parquet'
        archive = root / 'sources/legacy_retail_pricing_2023_2024.parquet'
        if legacy.exists():
            archive.parent.mkdir(exist_ok=True)
            if archive.exists() and digest(archive) != digest(legacy):
                raise ValueError(f'Existing archived tariff differs: {archive}')
            shutil.move(str(legacy), str(archive))
        prices.to_parquet(root / PRICING, index=False, compression='zstd')
        for path in [root/'schema.json', *sorted((root/'evaluation').glob('*.json'))]:
            schema = json.loads(path.read_text())
            if schema['simulation_end_time_step'] != TOTAL-1:
                raise ValueError(f'Expected a complete two-year schema: {path}')
            for building in schema['buildings'].values():
                building['pricing'] = PRICING
            schema['community_market'] = dict(MARKET)
            schema['dataset_contract_version'] = VERSION
            schema['experimental_design'].update(version=VERSION, tariff_policy=POLICY)
            dump(path, schema)
            if path.stem in entries:
                entries[path.stem]['sha256'] = digest(path)
        members_path = root/'catalogs/members.parquet'
        members = pd.read_parquet(members_path)
        members['tariff_id'] = POLICY
        members['story'] = members.story.str.replace(
            r'Tarifa de referência comum a todos os membros: [^.]+\.', STORY, regex=True)
        members.to_parquet(members_path, index=False, compression='zstd')
        # Preserve existing lifecycle wording while updating the tariff sentence.
        for path in [root/'MEMBER_STORIES.md', root/'evaluation/MEMBER_STORIES.md']:
            if path.exists():
                import re
                path.write_text(re.sub(r'Tarifa de referência comum a todos os membros: [^.]+\.', STORY, path.read_text()))
        dm_path = root/'dataset_manifest.json'
        dm = json.loads(dm_path.read_text())
        dm.update(version=VERSION, tariff_policy=POLICY, active_pricing_file=PRICING,
                  market_configuration_scope='enabled in all dataset and evaluation schemas',
                  price_formula='max(OMIE_PT_EUR_MWh / 1000, 0)',
                  community_market=dict(MARKET),
                  validation='See results/portugal_rec_native/pricing_validation.json for current checks; historical checks retain their original scope.')
        dump(dm_path, dm)
        contract_path = root/'settlement_contract.json'
        contract = json.loads(contract_path.read_text())
        contract.update(version=VERSION, tariff_policy=POLICY, active_pricing_file=PRICING,
                        dataset_market_enabled=True, local_price_ratio_to_grid_import=0.8,
                        configuration_scope='enabled in dataset and evaluation schemas',
                        native_rule='local price = 0.8 * max(OMIE_PT_EUR_MWh / 1000, 0)')
        dump(contract_path, contract)
        current_path = root/'CURRENT_EVALUATION.json'
        current = json.loads(current_path.read_text())
        current.update(version=VERSION, tariff_policy=POLICY)
        for entry in current['variants']:
            entry['sha256'] = digest(root/'evaluation'/f"{entry['variant']}.json")
        dump(current_path, current)
        readme = root/'README.md'
        text = readme.read_text().replace('O mercado continua desligado.',
            'O mercado comunitário está ativo: preço local = 80% do preço de compra à rede; exportação à rede = zero.')
        marker = '\n## Preços OMIE e mercado comunitário\n'
        text = text.split(marker)[0].rstrip() + marker + '\n'
        text += (f'Versão {VERSION}. Série ativa: `{PRICING}`. '
                 'Compra à rede = máximo(OMIE português em EUR/MWh / 1000, 0), em EUR/kWh. '
                 'Sem margem, perdas, MIBEL, TAR, taxas ou impostos. Previsões a 1/6/24 h seguem a mesma regra, '
                 'com OMIE publicado ou persistência diária. As referências de retalho em sources/ são apenas históricas.\n')
        readme.write_text(text)
        refresh_checksums(root)
    manifest.update(version=VERSION, tariff_policy=POLICY,
                    pricing=f'Common {PRICING}: max(OMIE_PT_EUR_MWh / 1000, 0). No retail additions. Native community market enabled at 0.8; grid export zero.',
                    validation_scope='Pricing and market revision; current evidence in pricing_validation.json. Previous construction reports remain historical.')
    dump(manifest_path, manifest)
    stats = {}
    for year in (2023, 2024):
        mask = UTC.year == year
        raw = market.reindex(UTC[mask]).to_numpy()
        actual = prices.electricity_pricing[mask]
        stats[str(year)] = {'intervals': int(mask.sum()), 'negative_omie_clipped': int((raw < 0).sum()),
                            'original_zero_intervals': int((raw == 0).sum()), 'final_zero_intervals': int((actual == 0).sum()),
                            'minimum_eur_kwh': float(actual.min()), 'mean_eur_kwh': float(actual.astype(float).mean()),
                            'maximum_eur_kwh': float(actual.max())}
    evidence = {'version': VERSION, 'formula': 'max(OMIE_PT_EUR_MWh / 1000, 0)', 'community_market': MARKET,
                'source_files': {str(p.relative_to(reference_root)): digest(p) for p in sources},
                'forecast_method': 'Target OMIE after conservative publication at 14:00 Europe/Madrid; otherwise same target time one day earlier, clipped to first archive observation at initialisation. Full 1/6/24 h horizons, including 2025-01-01 support.',
                'annual': stats, 'boundary_forecasts': boundary}
    dump(repo/'results/portugal_rec_native/pricing_policy.json', evidence)
    print(json.dumps({'version': VERSION, 'families': len(roots), 'configurations': len(entries), 'annual': stats}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--reference-root', type=Path, default=ROOT)
    args = parser.parse_args()
    apply(args.repo.resolve(), args.reference_root.resolve())

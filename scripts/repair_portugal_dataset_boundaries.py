"""Final offline integrity step, after the historical build and OMIE conversion.

The home-parking generator takes the complement of trips within the observation
year. Its last boundary is not a scheduled departure. Preserve the observed
interval and target, but publish an unknown departure for those partial visits.
Frozen source archives and historical validation reports are left intact.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from apply_portugal_omie_pricing import ROOT, dump, refresh_checksums

REVISION = 'portugal-boundaries-1'
TOTAL = 70176


def member_presence(schema, member, rows):
    present = np.full(rows, schema['buildings'][member].get('include', True), dtype=bool)
    for event in sorted(schema.get('topology_events', []), key=lambda e: e['time_step']):
        if event.get('target_member_id') != member:
            continue
        if event['operation'] in ('add_member', 'remove_member'):
            present[int(event['time_step']):] = event['operation'] == 'add_member'
    return present


def censor_sessions(frame, end=TOTAL):
    """Only the generated terminal home complement is right-censored."""
    frame = frame.copy()
    if 'observed_end_time_step' not in frame:
        frame['observed_end_time_step'] = frame.departure_time_step.astype('int64')
    partial = frame.observed_end_time_step.eq(end) & frame.purpose.eq('home')
    frame['right_censored'] = partial
    frame.loc[partial, 'departure_time_step'] = np.nan
    return frame


def repair(repo=ROOT):
    repo = Path(repo)
    roots = sorted((repo/'data/datasets').glob('rec_pt_2023_*_v2'))
    if len(roots) != 4:
        raise ValueError('Expected four Portuguese dataset families')
    manifest_path = repo/'results/portugal_rec_native/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    report_path = repo/'results/portugal_rec_native/boundary_integrity.json'
    prior = json.loads(report_path.read_text()) if report_path.exists() else {}
    outages = {r['path']: r for r in prior.get('removed_inactive_outages', [])}
    families = []
    changed = []
    for root in roots:
        path = root/'catalogs/charging_sessions.parquet'
        original = pd.read_parquet(path)
        sessions = censor_sessions(original)
        if not original.equals(sessions):
            sessions.to_parquet(path, index=False, compression='zstd')
            changed.append(str(path.relative_to(repo)))
        partial = sessions.loc[sessions.right_censored]
        ids = set(partial.session_id)
        schedule_files = 0
        for path in sorted(root.rglob('CHG-*.parquet')):
            frame = pd.read_parquet(path)
            mask = frame.electric_vehicle_session_id.isin(ids) & frame.electric_vehicle_charger_state.eq(1)
            if not mask.any():
                continue
            schedule_files += 1
            if frame.loc[mask, 'electric_vehicle_departure_time'].notna().any():
                frame.loc[mask, 'electric_vehicle_departure_time'] = np.nan
                frame.to_parquet(path, index=False, compression='zstd')
                changed.append(str(path.relative_to(repo)))
        for path in sorted((root/'evaluation').glob('*.json')):
            schema = json.loads(path.read_text())
            for member, building in schema['buildings'].items():
                source = root/building['energy_simulation']
                frame = pd.read_parquet(source)
                if 'power_outage' not in frame:
                    continue
                inactive = ~member_presence(schema, member, len(frame))
                remove = inactive & frame.power_outage.gt(0).to_numpy()
                if not remove.any():
                    continue
                rel = str(source.relative_to(repo))
                outages[rel] = dict(path=rel, variant=path.stem, member=member,
                                    time_steps=np.flatnonzero(remove).tolist())
                frame.loc[remove, 'power_outage'] = 0
                frame.to_parquet(source, index=False, compression='zstd')
                changed.append(rel)
        info = dict(revision=REVISION, right_censored_sessions=partial.session_id.tolist(),
                    schedule_files_with_partial_sessions=schedule_files,
                    session_contract='observed_end_time_step is exclusive; departure_time_step is null for right-censored home parking; target and arrival SOC unchanged',
                    runtime_contract='null departure countdown loads as unknown (-1); a connected vehicle with unknown departure is not a terminal departure',
                    source='portugal_profiles_2024.mobility: home parking is the complement of observed trips, ending at the observation boundary')
        dump(root/'boundary_integrity.json', info)
        for name in ('dataset_manifest.json', 'CURRENT_EVALUATION.json', 'extension_2024.json'):
            path = root/name
            metadata = json.loads(path.read_text())
            metadata['integrity_revision'] = REVISION
            dump(path, metadata)
        refresh_checksums(root)
        families.append(dict(family=root.name, sessions=len(sessions),
                             complete_sessions=int((~sessions.right_censored).sum()),
                             right_censored_sessions=len(partial),
                             schedule_files_with_partial_sessions=schedule_files))
    # Rebuild only the executable outage index. Keep historical programmes and
    # the DR/robustness records byte-for-byte equivalent as JSON values.
    events = [event for event in manifest['events'] if event['kind'] != 'grid']
    for entry in manifest['variants']:
        path = repo/entry['schema']
        root = path.parent.parent
        schema = json.loads(path.read_text())
        for member, building in schema['buildings'].items():
            frame = pd.read_parquet(root/building['energy_simulation'])
            if 'power_outage' not in frame:
                continue
            indices = np.flatnonzero(frame.power_outage.gt(0))
            for group in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
                if len(group):
                    events.append(dict(variant=entry['variant'], family=entry['family'],
                                       phase='development', kind='grid', member_id=member,
                                       start_time_step=int(group[0]), end_time_step=int(group[-1])))
    # Preserve the index order as well as all unaffected event records.
    key = lambda event: json.dumps(event, sort_keys=True)
    remaining = {key(event): event for event in events}
    ordered = []
    for event in manifest['events']:
        if key(event) in remaining:
            ordered.append(remaining.pop(key(event)))
    manifest['events'] = ordered + list(remaining.values())
    manifest['integrity_revision'] = REVISION
    manifest['boundary_integrity_report'] = str(report_path.relative_to(repo))
    dump(manifest_path, manifest)
    report = dict(revision=REVISION, families=families,
                  removed_inactive_outages=list(outages.values()),
                  removed_outage_steps=sum(len(r['time_steps']) for r in outages.values()),
                  preserved='Prices, forecasts, market, load/PV, EV identities, arrival SOC, targets, DR requests and intentional faults are unchanged.',
                  runtime_requirement='Use the accompanying unknown-departure KPI/reward handling; historical simulator commits do not correctly handle terminal unknown deadlines.')
    dump(report_path, report)
    return dict(changed_data_files=changed, report=report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT)
    args = parser.parse_args()
    result = repair(args.repo)
    print(json.dumps(dict(changed_data_files=len(result['changed_data_files']),
                          families=result['report']['families'],
                          removed_outage_steps=result['report']['removed_outage_steps']), indent=2))

"""Validate reserved lifecycle histories, matched inputs and executed interfaces."""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT))
from prepare_portugal_heldout_2024 import (VERSION, REVISION, N23, TOTAL, DYNAMIC,
    read_json, dump, digest, resource_state, active_masks)


def runtime_fingerprint(repo=ROOT):
    """Bind reusable execution evidence to engine, executable schemas and inputs.

    Descriptive metadata is excluded; CityLearn does not consume these fields.
    File hashes are resolved against each schema's actual root directory.
    """
    repo = Path(repo).resolve()
    manifest = read_json(repo/'results/portugal_rec_native/manifest.json')
    schemas, files = {}, {}
    def collect(value, root):
        if isinstance(value, dict):
            for v in value.values():
                collect(v, root)
        elif isinstance(value, list):
            for v in value:
                collect(v, root)
        elif isinstance(value, str) and value.endswith('.parquet'):
            path = (root/value).resolve()
            if path.is_file():
                files[str(path.relative_to(repo))] = digest(path)
    for entry in manifest['variants']+manifest['references']:
        path = repo/entry['schema']
        schema = read_json(path)
        root = (path.parent/schema['root_directory']).resolve()
        for key in ('dataset_id', 'dataset_family', 'dataset_contract_version', 'experimental_design', 'native_compatibility', 'root_directory'):
            schema.pop(key, None)
        collect(schema, root)
        schemas[entry['schema']] = schema
    engine = {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT/'citylearn').rglob('*.py'))}
    payload = json.dumps(dict(schemas=schemas, inputs=files, engine=engine), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def comparable(schema):
    result = deepcopy(schema)
    # Descriptive identity and path depth do not affect execution. All other
    # fields, including seeds, state, services and observation flags, are checked.
    result.pop('dataset_id', None)
    result.pop('root_directory', None)
    design = result['experimental_design']
    for k in ('name', 'label', 'purpose', 'paired_reference', 'factors'):
        design.pop(k, None)
    return result


def verify_pair(repo, pair):
    path = repo/pair['scenario_schema']
    root = path.parent.parent
    a, b = read_json(path), read_json(repo/pair['reference_schema'])
    factor = pair['factor']
    left, right = comparable(a), comparable(b)
    changed_files, changed_limits = [], []
    if factor == 'topology_2024':
        archived = read_json(root/'evaluation/campaign_2024/source_schemas'/path.name)
        archived['dataset_contract_version'] = VERSION
        archived['experimental_design']['version'] = VERSION
        assert comparable(archived) == right, 'Frozen reference must exactly reproduce the previous trajectory'
        assert [e for e in a['topology_events'] if e['time_step'] < N23] == b['topology_events']
        before = min(e['time_step'] for e in a['topology_events'] if e['time_step'] >= N23)-1
        assert resource_state(a, before) == resource_state(b, before)
        assert a['random_seed'] == b['random_seed']
        for key in ('community_market', 'derived_forecasts', 'electric_vehicles_def', 'observations', 'actions', 'observation_bundles'):
            assert a[key] == b[key], key
        for key in ('robustness', 'demand_response'):
            assert a.get(key) == b.get(key), key
        for m, original in b['buildings'].items():
            current = a['buildings'][m]
            for key in ('pricing', 'weather', 'carbon_intensity', 'electrical_service', 'chargers', 'deferrable_appliances'):
                if key == 'chargers':
                    for cid, ch in original.get(key, {}).items():
                        other = current[key][cid]
                        assert {k:v for k,v in ch.items() if k!='charger_simulation'} == {k:v for k,v in other.items() if k!='charger_simulation'}
                        pd.testing.assert_frame_equal(pd.read_parquet(root/ch['charger_simulation']).iloc[:before+1], pd.read_parquet(root/other['charger_simulation']).iloc[:before+1])
                else:
                    assert original.get(key) == current.get(key), (m, key)
            old = pd.read_parquet(root/original['energy_simulation'])
            new = pd.read_parquet(root/current['energy_simulation'])
            pd.testing.assert_frame_equal(old.iloc[:before+1], new.iloc[:before+1])
        return dict(reference=pair['reference'], factor=factor, status='pass', identical_through_step=before)
    if factor == 'dr_activation':
        assert left['demand_response']['enabled'] is True and right['demand_response']['enabled'] is False
        right['demand_response']['enabled'] = True
    elif factor == 'electrical_limits':
        normal = read_json(root/'evaluation/core_30_normal.json')
        for m, building in left['buildings'].items():
            other = right['buildings'][m]
            if building['electrical_service'] != other['electrical_service']:
                assert other['electrical_service'] == normal['buildings'][m]['electrical_service']
                changed_limits.append(m)
                other['electrical_service'] = deepcopy(building['electrical_service'])
        assert changed_limits == ['Member_001', 'Member_006', 'Member_025'], changed_limits
    elif factor == 'disturbances':
        assert left['robustness']['enabled'] is True and right['robustness']['enabled'] is False
        right['robustness']['enabled'] = True
        assert not right['experimental_design']['sensor_noise_stress']
        right['experimental_design']['sensor_noise_stress'] = left['experimental_design']['sensor_noise_stress']
        for m, building in left['buildings'].items():
            other = right['buildings'][m]
            if building['energy_simulation'] == other['energy_simulation']:
                continue
            original = pd.read_parquet(root/building['energy_simulation'])
            actual = pd.read_parquet(root/other['energy_simulation'])
            outage = original.power_outage.gt(0)
            # A stochastic programme can fall entirely after this owner's exit.
            # The paired reference must still be identical outside actual cuts.
            assert actual.power_outage.eq(0).all()
            pd.testing.assert_frame_equal(original.drop(columns=['power_outage', 'solar_generation']), actual.drop(columns=['power_outage', 'solar_generation']))
            np.testing.assert_array_equal(original.loc[~outage, 'solar_generation'], actual.loc[~outage, 'solar_generation'])
            live, pv, _ = active_masks(a, m)
            physical = pd.read_parquet(root/f'timeseries/{m}.parquet').solar_generation.to_numpy()
            np.testing.assert_array_equal(actual.loc[outage, 'solar_generation'], np.where(live & pv, physical, 0)[outage])
            changed_files.append(other['energy_simulation'])
            other['energy_simulation'] = building['energy_simulation']
    else:
        raise AssertionError(factor)
    assert left == right, f'Undeclared schema change in {pair["reference"]}'
    return dict(reference=pair['reference'], factor=factor, status='pass', changed_energy_files=changed_files, changed_limit_members=changed_limits)


def offered_sessions(root, schema):
    seen = {}
    for m, building in schema['buildings'].items():
        for cid, ch in building.get('chargers', {}).items():
            f = pd.read_parquet(root/ch['charger_simulation'])
            connected = f.electric_vehicle_charger_state.eq(1)
            on = f.loc[connected].copy()
            on['time_step'] = np.flatnonzero(connected)
            for sid, group in on.groupby('electric_vehicle_session_id', sort=False):
                assert sid not in seen, ('duplicate session at different connectors', sid)
                seen[sid] = (int(group.time_step.min()), int(group.time_step.max())+1, m, cid)
    return seen


def static_audit(repo=ROOT, write_cohorts=False):
    repo = Path(repo)
    record = repo/'results/portugal_rec_native/heldout_2024'
    campaign = read_json(record/'campaign.json')
    manifest = read_json(repo/'results/portugal_rec_native/manifest.json')
    assert manifest['version'] == VERSION
    for relative, expected in read_json(record/'source_input_checksums.json').items():
        actual = digest(repo/relative)
        if actual != expected:
            root = repo/Path(*Path(relative).parts[:3])
            local = str((repo/relative).relative_to(root))
            realism_path = root/'realism_revision.json'
            realism = read_json(realism_path) if realism_path.exists() else {}
            if (realism.get('revision') in {'portugal-realism-1', 'portugal-realism-2', 'portugal-realism-3', 'portugal-realism-4'}
                    and realism.get('revision') == manifest.get('realism_revision')
                    and realism.get('output_checksums', {}).get(local) == actual):
                continue
            # The EV revision explicitly supersedes mobility inputs in both
            # arms of the matched experiment. Other source inputs stay frozen.
            root = repo/Path(*Path(relative).parts[:3])
            local = str((repo/relative).relative_to(root))
            allowed = (Path(local).name.startswith('CHG-') or local in {
                'catalogs/charging_sessions.parquet', 'catalogs/electric_vehicles.parquet',
                'catalogs/chargers.parquet'})
            revision = read_json(root/'ev_mobility_revision.json') if allowed and (root/'ev_mobility_revision.json').exists() else {}
            assert revision.get('revision') == 'portugal-ev-mobility-1' and revision.get('output_checksums', {}).get(local) == actual, ('Previous physical/exogenous input changed without a matching mobility revision', relative)
    assert len(manifest['variants']) == 9 and len(manifest['references']) == 6
    pairs = [verify_pair(repo, p) for p in campaign['pairs']]
    preparation = []
    for entry in manifest['preparation_schemas']:
        path = repo/entry['schema']
        s = read_json(path)
        root = (path.parent/s['root_directory']).resolve()
        old = read_json(root/'evaluation/campaign_2024/source_schemas'/path.name)
        assert s['simulation_end_time_step'] == N23-1
        assert s['experimental_design']['years'] == [2023]
        assert s.get('topology_events', []) == old.get('topology_events', [])
        assert s['buildings'] == old['buildings'], 'Preparation must not preload held-out members or equipment'
        assert s['electric_vehicles_def'] == old['electric_vehicles_def']
        for config, key in [(s.get('robustness', {}), 'events_file'), (s.get('demand_response', {}), 'requests_file'),
                            (s['experimental_design'], 'native_confirmation_robustness_file'),
                            (s['experimental_design'].get('measurement_noise', {}), 'channel_file'),
                            (s['experimental_design'].get('measurement_noise', {}), 'fault_file')]:
            if key in config:
                f = pd.read_parquet(root/config[key])
                assert f.start_time_step.ge(0).all() and f.end_time_step.lt(N23).all()
        preparation.append(dict(variant=entry['variant'], status='pass', final_step=N23-1,
                                future_member_pool=False, future_event_programmes=False))
    schemas, cohort_summary = [], []
    for family, variants in DYNAMIC.items():
        root = repo/f'data/datasets/rec_pt_2023_{family}_v2'
        sessions = pd.read_parquet(root/'catalogs/charging_sessions.parquet')
        for variant in variants:
            path = root/f'evaluation/{variant}.json'
            s = read_json(path)
            old = read_json(root/'evaluation/campaign_2024/source_schemas'/path.name)
            evs = pd.read_parquet(root/'catalogs/electric_vehicles.parquet')
            owned = set(evs.loc[evs.home_member_id.eq('Member_020'), 'ev_id'])
            events = [e for e in s['topology_events'] if e['time_step'] >= N23]
            assert len(events) == 6 and [e for e in s['topology_events'] if e['time_step'] < N23] == old['topology_events']
            assert resource_state(s, N23-1) == resource_state(old, N23-1)
            expected_initial = 27 if family == 'core_30' else 88
            assert resource_state(s, N23)['counts']['building'] == expected_initial
            assert resource_state(s, TOTAL-1)['counts']['building'] == expected_initial
            departed, admitted = set(), set(m for m,b in s['buildings'].items() if b.get('include', True))
            for e in s['topology_events']:
                if e['operation'] == 'remove_member':
                    assert e['target_member_id'] not in departed
                    departed.add(e['target_member_id'])
                if e['operation'] == 'add_member':
                    assert e['target_member_id'] not in departed | admitted, 'Readmission or ineffective admission'
                    admitted.add(e['target_member_id'])
            for m, b in s['buildings'].items():
                frame = pd.read_parquet(root/b['energy_simulation'])
                assert len(frame) == TOTAL and np.isfinite(frame.to_numpy()).all()
                live, pv, _ = active_masks(s, m)
                assert not (frame.power_outage.gt(0).to_numpy() & ~live).any()
                assert frame.loc[~live & (np.arange(TOTAL)>=N23), ['non_shiftable_load', 'solar_generation', 'occupant_count']].eq(0).all().all()
                if m not in old['buildings']:
                    assert frame.loc[live, 'non_shiftable_load'].sum() > 0, 'New admission must introduce real demand'
                    assert not any(b.get(k) for k in ('chargers', 'deferrable_appliances', 'pv', 'electrical_storage'))
                if m in old['buildings']:
                    original = pd.read_parquet(root/old['buildings'][m]['energy_simulation'])
                    pd.testing.assert_frame_equal(original.iloc[:N23], frame.iloc[:N23])
                    for key in ('pricing', 'weather', 'carbon_intensity', 'deferrable_appliances'):
                        assert b.get(key) == old['buildings'][m].get(key), (m, key)
                for cid, ch in b.get('chargers', {}).items():
                    f = pd.read_parquet(root/ch['charger_simulation'])
                    assert len(f) == TOTAL
                    if m in old['buildings']:
                        initial = pd.read_parquet(root/old['buildings'][m]['chargers'][cid]['charger_simulation'])
                        pd.testing.assert_frame_equal(initial.iloc[:N23], f.iloc[:N23])
            before, after = offered_sessions(root, old), offered_sessions(root, s)
            assert set(after) <= set(before), 'New member has no charging; do not invent or reidentify existing visits'
            removed = set(before)-set(after)
            exit_t = next(e['time_step'] for e in events if e['operation']=='remove_member')
            assert all(before[sid][0] >= exit_t for sid in removed)
            assert set(sessions.loc[sessions.session_id.isin(removed), 'ev_id']) <= owned
            assert all(after[sid] == before[sid] for sid in after), 'No visit is shortened or silently censored by departure'
            rows = sessions[['session_id', 'ev_id', 'member_id', 'charger_id', 'arrival_time_step', 'departure_time_step', 'observed_end_time_step', 'right_censored', 'requested_energy_reference_kwh']].copy()
            rows['offered_previous_trajectory'] = rows.session_id.isin(before)
            rows['offered_current_trajectory'] = rows.session_id.isin(after)
            rows['cohort'] = np.where(rows.session_id.isin(removed), 'excluded_by_2024_permanent_owner_exit', np.where(rows.session_id.isin(after), 'shared_service', 'outside_previous_and_current_trajectory'))
            rows['eligible_known_departure'] = rows.offered_current_trajectory & ~rows.right_censored
            # Eligibility is input exposure, not controller success. Runtime
            # censoring/eligibility remains part of the exported service audit.
            if write_cohorts:
                rows.to_parquet(record/f'{variant}_service_cohorts.parquet', index=False, compression='zstd')
            cohort_summary.append(dict(variant=variant, source_sessions=len(rows), previous_offered=len(before), current_offered=len(after),
                excluded_by_2024_departure=len(removed), excluded_requested_energy_reference_kwh=float(rows.loc[rows.session_id.isin(removed), 'requested_energy_reference_kwh'].sum()),
                affected_inflight_sessions=0, shared_service_requests_unchanged=True,
                appliances='Current appliance realization is shared across matched arms; exiting member owns no controllable appliance'))
            schemas.append(dict(variant=variant, status='pass', events_2024=len(events), inherited_members=expected_initial,
                peak_members=expected_initial+1, final_members=expected_initial))
    for entry in manifest['variants']+manifest['references']+manifest['preparation_schemas']:
        assert digest(repo/entry['schema']) == entry['sha256']
    # Every derivative explicitly records a source and preserves the entire
    # preparation-year series, including EV identities and year-end services.
    for row in campaign['derived_files']:
        root = repo/f'data/datasets/rec_pt_2023_{row["family"]}_v2'
        pd.testing.assert_frame_equal(pd.read_parquet(root/row['path']).iloc[:N23], pd.read_parquet(root/row['source']).iloc[:N23])
    return dict(status='pass', pairs=pairs, scenarios=schemas, cohorts=cohort_summary, preparation=preparation,
                preparation_year_preserved=True,
                preparation_preservation_scope='Matched 2023 inputs across current campaign and references; declared mobility/realism revisions apply to both.',
                mobility_revision=manifest.get('mobility_revision'), declared_2024_events=len(campaign['events']))


def runtime_window(path, start, end, label):
    from citylearn.citylearn import CityLearnEnv
    from citylearn.agents.baseline import NormalPolicy
    schema = read_json(path)
    root = (path.parent/schema['root_directory']).resolve()
    original = deepcopy(schema)
    schema.update(root_directory=str(root), simulation_start_time_step=start, simulation_end_time_step=end,
        episode_time_steps=end-start+1, rolling_episode_split=False, random_episode_split=False,
        physics_invariant_checks=True)
    env = CityLearnEnv(schema, render_mode='none', offline=True, export_kpis_on_episode_end=False)
    ticks = {start, end} | {e['time_step']+d for e in schema.get('topology_events', []) if start <= e['time_step'] <= end for d in (-1, 0, 1)}
    snapshots = []
    begin = time.monotonic()
    try:
        obs, _ = env.reset(seed=schema['random_seed'])
        policy = NormalPolicy(env)
        while True:
            global_t = start+env.time_step
            expected = resource_state(original, global_t)
            for table, ids in expected['ids'].items():
                actual = env.entity_specs['tables'].get(table, {}).get('ids', [])
                assert sorted(actual) == ids, (path.stem, global_t, table, actual, ids)
                if table in obs['tables']:
                    assert obs['tables'][table].shape[0] == len(ids)
            for table in ('building', 'charger', 'deferrable_appliance'):
                if table in env.entity_specs['actions']:
                    assert sorted(env.entity_specs['actions'][table]['ids']) == expected['ids'][table]
                    assert env.action_space['tables'][table].shape[0] == len(expected['ids'][table])
            departed_20 = any(e['operation']=='remove_member' and e['target_member_id']=='Member_020'
                             and e['time_step']<=global_t for e in original.get('topology_events', []))
            if departed_20:
                assert not any('EV-M020-' in eid for eid in env.entity_specs['tables'].get('ev', {}).get('ids', []))
            if global_t in ticks:
                snapshots.append(dict(time_step=global_t, counts=expected['counts'], topology_version=env.topology_version))
            if env.terminated:
                break
            obs, reward, terminated, truncated, _ = env.step(policy.predict(obs, deterministic=True))
            assert np.isfinite(reward).all()
            rows = env._last_community_market_settlement
            assert abs(sum(r['local_import_kwh']-r['local_export_kwh'] for r in rows)) < 1e-4
            for r in rows:
                assert r['grid_import_price'] >= 0 and r['grid_export_price'] == 0
                assert abs(r['local_buy_price']-.8*r['grid_import_price']) < 1e-8
            if truncated:
                raise AssertionError('Unexpected truncation')
        executed = [e for e in env.topology_event_log if start <= e['time_step'] <= end]
        declared = [e for e in original.get('topology_events', []) if start <= e['time_step'] <= end]
        assert [e['id'] for e in executed] == [e['id'] for e in declared]
        assert all(e['applied'] for e in executed), 'A recorded event did not change the running community'
        # Check actual action bounds after commissioning, not just presence of a
        # storage object. The pre-installation inactive mask must be lifted.
        for e in declared:
            if e['operation']=='add_asset' and e.get('target_asset_type')=='electrical_storage':
                building = next(b for b in env.buildings if b.name==e['target_member_id'])
                assert building.electrical_storage.capacity == e['overrides']['capacity']
                assert building.action_metadata['electrical_storage']
        kpis = env.evaluate_v2(include_business_as_usual=False)
        assert not kpis.duplicated(['level', 'name', 'cost_function']).any()
        return dict(schema=path.name, label=label, status='pass', start=start, end=end, transitions=env.time_step,
                    snapshots=snapshots, executed_events=executed, kpi_rows=len(kpis), seconds=round(time.monotonic()-begin, 3))
    finally:
        env.close()
        del env
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument('--runtime', action='store_true')
    execution.add_argument('--reuse-runtime', action='store_true', help='Reuse completed windows only if executable inputs and engine hashes match')
    execution.add_argument('--resume-runtime', action='store_true', help='Resume partial event windows with identical executable inputs')
    parser.add_argument('--half-window', type=int, default=48, help='Transitions either side of each actual event (default 12 h)')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    record = args.repo/'results/portugal_rec_native/heldout_2024'
    output = args.output or record/'validation.json'
    fingerprint = runtime_fingerprint(args.repo)
    previous = read_json(output) if args.reuse_runtime or args.resume_runtime else None
    if previous:
        assert previous.get('runtime_input_fingerprint') == fingerprint, 'Execution evidence is stale'
        assert all(r['status']=='pass' for r in previous['runtime'])
        if args.reuse_runtime:
            assert previous.get('status') == 'pass' and len(previous['runtime']) == 72
    report = dict(version=VERSION, revision=REVISION, static=static_audit(args.repo, write_cohorts=True), runtime=[])
    report['runtime_input_fingerprint'] = fingerprint
    if previous:
        report['runtime'] = previous['runtime']
        report['runtime_evidence_reused_with_identical_executable_inputs'] = True
    dump(output, report)
    if args.runtime or args.resume_runtime:
        campaign = read_json(record/'campaign.json')
        manifest = read_json(args.repo/'results/portugal_rec_native/manifest.json')
        for entry in manifest['variants']+manifest['references']:
            path = args.repo/entry['schema']
            schema = read_json(path)
            events = [e for e in schema.get('topology_events', []) if e['time_step'] >= N23]
            windows = [('year_boundary', N23-8, N23+8), ('terminal_boundary', TOTAL-17, TOTAL-1)]
            windows += [(e['id'], e['time_step']-args.half_window, e['time_step']+args.half_window) for e in events]
            for label, start, end in windows:
                if any(r['schema']==path.name and r['label']==label and r['start']==start and r['end']==end for r in report['runtime']):
                    continue
                row = runtime_window(path, start, end, label)
                report['runtime'].append(row)
                dump(output, report)
                print(path.stem, label, row['status'], flush=True)
    assert runtime_fingerprint(args.repo) == fingerprint, 'Inputs or engine changed during the audit'
    report['status'] = 'pass'
    report['dataset_checksums'] = {str(p.relative_to(args.repo)): digest(p) for p in
        sorted((args.repo/'data/datasets').glob('rec_pt_2023_*_v2/file_checksums.sha256'))}
    report['runtime_scope'] = 'Actual event and annual-boundary windows; no annual controller benchmark or training. Window resets preserve topology, not unsimulated physical state.'
    dump(output, report)
    print(json.dumps({'static': report['static']['status'], 'runtime_windows': len(report['runtime']), 'output': str(output)}, indent=2))


if __name__ == '__main__':
    main()

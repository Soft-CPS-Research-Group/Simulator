"""Add a reserved 2024 lifecycle campaign and matched scenario references.

Offline, additive and idempotent: original physical inputs are never overwritten.
Run after pricing and boundary repair. Archived input schemas and hashes identify
the previous edition; they are also the frozen-2024 lifecycle counterfactuals.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from apply_portugal_omie_pricing import refresh_checksums
from portugal_lifecycle_stories import lifecycle

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'portugal-native-2023-2024-v3.3'
REVISION = 'portugal-heldout-2024-1'
PREVIOUS_COMMIT = '8157f605e26fb75651037451fb453b1a48056787'
N23, TOTAL = 35040, 70176
DYNAMIC = {'core_30': ['core_30_topology', 'core_30_combined'],
           'premium_100': ['premium_100_allin']}


def read_json(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def step(local):
    stamp = pd.Timestamp(local, tz='Europe/Lisbon').tz_convert('UTC')
    return int((stamp-pd.Timestamp('2023-01-01', tz='UTC'))/pd.Timedelta(minutes=15))


def stamp(t):
    return (pd.Timestamp('2023-01-01', tz='UTC')+pd.Timedelta(minutes=15*t)).isoformat()


def write_frame(root, path, frame):
    dest = root/path
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(dest, index=False, compression='zstd')
    return str(path)


def active_masks(schema, member):
    b = schema['buildings'][member]
    live = np.full(TOTAL, b.get('include', True), dtype=bool)
    pv = np.full(TOTAL, b.get('pv', {}).get('attributes', {}).get('nominal_power', 0) > 0)
    battery = np.full(TOTAL, b.get('electrical_storage', {}).get('attributes', {}).get('capacity', 0) > 0)
    for e in sorted(schema.get('topology_events', []), key=lambda x: x['time_step']):
        if e['target_member_id'] != member:
            continue
        op, t = e['operation'], e['time_step']
        if op in ('add_member', 'remove_member'):
            live[t:] = op == 'add_member'
        elif e.get('target_asset_type') == 'pv':
            pv[t:] = op == 'add_asset'
        elif e.get('target_asset_type') == 'electrical_storage':
            battery[t:] = op == 'add_asset'
    return live, pv, battery


def resource_state(schema, t):
    """Independent schema interpreter; runtime checks compare exact identities."""
    active = {m for m, b in schema['buildings'].items() if b.get('include', True)}
    assets = {}
    for m, b in schema['buildings'].items():
        assets[m] = {'charger': set(b.get('chargers', {})),
                     'deferrable_appliance': set(b.get('deferrable_appliances', {})),
                     'pv': bool(b.get('pv', {}).get('attributes', {}).get('nominal_power', 0)),
                     'storage': bool(b.get('electrical_storage', {}).get('attributes', {}).get('capacity', 0))}
    for e in sorted(schema.get('topology_events', []), key=lambda x: x['time_step']):
        if e['time_step'] > t:
            break
        m, op = e['target_member_id'], e['operation']
        if op == 'add_member':
            active.add(m)
        elif op == 'remove_member':
            active.remove(m)
        else:
            kind = {'electrical_storage': 'storage'}.get(e['target_asset_type'], e['target_asset_type'])
            if kind in ('pv', 'storage'):
                assets[m][kind] = op == 'add_asset'
            elif op == 'add_asset':
                assets[m][kind].add(e['target_asset_id'])
            else:
                assets[m][kind].remove(e['target_asset_id'])
    ids = {'building': sorted(active)}
    for kind in ('charger', 'deferrable_appliance'):
        ids[kind] = sorted(f'{m}/{a}' for m in active for a in assets[m][kind])
    for kind in ('pv', 'storage'):
        suffix = 'electrical_storage' if kind == 'storage' else kind
        ids[kind] = sorted(f'{m}/{suffix}' for m in active if assets[m][kind])
    return {'ids': ids, 'counts': {k: len(v) for k, v in ids.items()}}


def campaign_events(family, exit_step):
    core = family == 'core_30'
    newcomer, install, retired = ('031', '010', '023') if core else ('101', '029', '099')
    # M025 has a schema-loaded battery in both bases, including after its exit
    # from neither campaign. Explicit attributes avoid copying its operating SoC.
    source = 'Member_025' if core else 'Member_004'
    storage = dict(capacity=5.12, nominal_power=2.56, efficiency=.94, initial_soc=.1,
                   depth_of_discharge=.9, loss_coefficient=2.805892205981486e-05,
                   capacity_loss_coefficient=0., power_efficiency_curve=[[0, .94], [1, .94]],
                   capacity_power_curve=[[0, 1], [1, 1]])
    definitions = [
        ('admission', step('2024-02-12 12:00'), 'add_member', f'Member_{newcomer}', None, None, {}),
        ('commissioning', step('2024-05-13 12:00'), 'add_asset', f'Member_{install}', 'electrical_storage', f'BESS-M{install}-01', storage),
        ('replacement_out', step('2024-05-20 12:00'), 'remove_asset', 'Member_014', 'electrical_storage', 'BESS-M014-01', {}),
        ('replacement_in', step('2024-05-22 12:00'), 'add_asset', 'Member_014', 'electrical_storage', 'BESS-M014-02', storage),
        ('permanent_departure', exit_step, 'remove_member', 'Member_020', None, None, {}),
        ('permanent_retirement', step('2024-11-11 12:00'), 'remove_asset', f'Member_{retired}', 'pv', f'PV-M{retired}-01', {}),
    ]
    events = []
    for purpose, t, op, member, kind, aid, overrides in definitions:
        e = dict(id=f'heldout2024_{purpose}_{member}', time_step=t, operation=op,
                 target_member_id=member, purpose=purpose, campaign=REVISION,
                 timestamp_utc=stamp(t), timestamp_local=pd.Timestamp(stamp(t)).tz_convert('Europe/Lisbon').isoformat())
        if kind:
            e.update(target_asset_type=kind, target_asset_id=aid)
        if overrides:
            e.update(source_member_id=source, overrides=deepcopy(overrides))
        events.append(e)
    return events


def choose_exit(roots):
    """A shared calendar date in a service-free gap, chosen without controller runs."""
    busy = np.zeros(TOTAL, dtype=bool)
    for root in roots:
        sessions = pd.read_parquet(root/'catalogs/charging_sessions.parquet')
        evs = pd.read_parquet(root/'catalogs/electric_vehicles.parquet')
        owned = set(evs.loc[evs.home_member_id.eq('Member_020'), 'ev_id'])
        for r in sessions.loc[sessions.ev_id.isin(owned)].itertuples():
            busy[int(r.arrival_time_step):int(r.observed_end_time_step)] = True
    start = step('2024-08-12 12:00')
    for t in range(start, start+7*96):
        if not busy[t-2:t+5].any():
            return t
    raise ValueError('No common service-free departure window in the reserved week')


def extend_dynamic(root, original, family, events):
    schema = deepcopy(original)
    variant = original['experimental_design']['name']
    prefix = Path('evaluation/campaign_2024/inputs')/variant
    new_id = 'Member_031' if family == 'core_30' else 'Member_101'
    donor = 'Member_027' if family == 'core_30' else 'Member_088'
    b = deepcopy(original['buildings'][donor])
    assert not any(b.get(k) for k in ('pv', 'electrical_storage', 'chargers', 'deferrable_appliances'))
    b['include'] = False
    # A separately identified synthetic service site, using an explicit existing
    # archetype. No EV or controllable-cycle obligations are invented or shared.
    raw = pd.read_parquet(root/f'timeseries/{donor}.parquet')
    raw.loc[:events[0]['time_step']-1, ['non_shiftable_load', 'solar_generation', 'occupant_count', 'power_outage']] = 0.
    b['energy_simulation'] = write_frame(root, prefix/f'{new_id}.parquet', raw)
    schema['buildings'][new_id] = b
    schema['topology_events'] = original['topology_events'] + deepcopy(events)
    install_member = events[1]['target_member_id']
    # Permit the newly commissioned storage action. The dynamic interface still
    # masks this action while the member has no installed battery before May.
    schema['buildings'][install_member]['inactive_actions'] = [
        a for a in schema['buildings'][install_member].get('inactive_actions', [])
        if a != 'electrical_storage']
    exit_t = next(e['time_step'] for e in events if e['operation'] == 'remove_member')
    evs = pd.read_parquet(root/'catalogs/electric_vehicles.parquet')
    owned = set(evs.loc[evs.home_member_id.eq('Member_020'), 'ev_id'])
    lineage = []
    for m, b in schema['buildings'].items():
        if m == new_id:
            continue
        old = pd.read_parquet(root/b['energy_simulation'])
        frame = old.copy()
        live, pv, battery = active_masks(schema, m)
        _, oldpv, oldbattery = active_masks(original, m)
        delta = battery.astype(int)-oldbattery.astype(int)
        # Standby remains an exogenous 8 W load, consistent with the original
        # construction; each changed battery is the same modular product class.
        if delta.any():
            frame['non_shiftable_load'] = np.maximum(0., frame.non_shiftable_load + delta*.002).astype(old.non_shiftable_load.dtype)
        changed_supply = (pv != oldpv) | ((battery != oldbattery) & frame.power_outage.gt(0).to_numpy())
        if changed_supply.any():
            physical = pd.read_parquet(root/f'timeseries/{m}.parquet').solar_generation.to_numpy()
            supply = live & pv & (~frame.power_outage.gt(0).to_numpy() | battery)
            frame.loc[changed_supply, 'solar_generation'] = np.where(supply, physical, 0.)[changed_supply]
        frame.loc[~live & (np.arange(TOTAL) >= N23), ['non_shiftable_load', 'solar_generation', 'occupant_count', 'power_outage']] = 0.
        if not frame.equals(old):
            pd.testing.assert_frame_equal(frame.iloc[:N23], old.iloc[:N23])
            previous = b['energy_simulation']
            b['energy_simulation'] = write_frame(root, prefix/f'{m}.parquet', frame)
            lineage.append(dict(path=b['energy_simulation'], source=previous, operation='2024 availability and storage standby'))
        for cid, ch in b.get('chargers', {}).items():
            old = pd.read_parquet(root/ch['charger_simulation'])
            mask = (np.arange(TOTAL) >= exit_t) & (old.electric_vehicle_id.isin(owned).to_numpy() | (m == 'Member_020'))
            if not mask.any():
                continue
            frame = old.copy()
            for col in frame:
                frame.loc[mask, col] = 0 if col == 'electric_vehicle_charger_state' else ('' if col in ('electric_vehicle_id', 'electric_vehicle_session_id') else np.nan)
            previous = ch['charger_simulation']
            ch['charger_simulation'] = write_frame(root, prefix/f'{cid}.parquet', frame)
            lineage.append(dict(path=ch['charger_simulation'], source=previous, operation='permanent owner exit; no later private services'))
    schema['experimental_design']['member_departures'].append(dict(member_id='Member_020', exit_step=exit_t,
        owned_ev_ids=sorted(owned), post_exit_demand='none', pending_cycles='none; no controllable appliances',
        boundary_policy='service-free gap; all previously admitted EV visits complete before exit'))
    schema['experimental_design']['continuation'] = 'Preserve all 2023 lifecycle events; add the explicitly reserved 2024 campaign, with no readmission or resurrection.'
    schema['experimental_design']['heldout_campaign'] = dict(revision=REVISION,
        supplemental_member_catalogue='evaluation/campaign_2024/catalogue.json',
        information_boundary='2023 only for training, validation and preprocessing; 2024 event schedule is evaluator-only, not a policy input')
    members = pd.read_parquet(root/'catalogs/members.parquet')
    template = members.loc[members.member_id.eq(donor)].iloc[0]
    supplemental = dict(member_id=new_id, family=family, include_initially=False, admission_step=events[0]['time_step'],
        profile_template_member=donor, profile_method='Existing service archetype, deterministic reuse; a distinct fictional site, not an independent statistical draw',
        category=str(template.category), routine_archetype=str(template.routine_archetype),
        electrical_service=schema['buildings'][new_id]['electrical_service'],
        assets=[], vehicles=[], controllable_service_requests=[], base_catalogue_unchanged=True)
    return schema, lineage, supplemental


def remove_disturbances(root, scenario, name):
    ref = deepcopy(scenario)
    ref['robustness']['enabled'] = False
    ref['experimental_design']['sensor_noise_stress'] = False
    for m, b in ref['buildings'].items():
        frame = pd.read_parquet(root/b['energy_simulation'])
        outage = frame.power_outage.gt(0).to_numpy()
        if not outage.any():
            continue
        live, pv, _ = active_masks(ref, m)
        physical = pd.read_parquet(root/f'timeseries/{m}.parquet').solar_generation.to_numpy()
        frame.loc[outage, 'solar_generation'] = np.where(live & pv, physical, 0.)[outage]
        frame['power_outage'] = np.zeros(len(frame), dtype=frame.power_outage.dtype)
        b['energy_simulation'] = write_frame(root, Path('evaluation/campaign_2024/inputs')/name/f'{m}.parquet', frame)
    return ref


def update_identity(schema, name=None, label=None, factors=None):
    schema['dataset_contract_version'] = VERSION
    schema['experimental_design']['version'] = VERSION
    if name:
        schema['dataset_id'] = name.upper().replace('_', '-')
        schema['experimental_design'].update(name=name, label=label or name,
            purpose='Matched input-history reference; not a BAU or RBC controller')
    if factors is not None:
        schema['experimental_design']['factors'] = factors
    return schema


def preparation_view(root, original, variant):
    """A 2023 entry point without the held-out member pool or future programmes."""
    schema = update_identity(deepcopy(original))
    schema.update(root_directory='../..', simulation_end_time_step=N23-1)
    design = schema['experimental_design']
    design.update(years=[2023], year_windows={'2023': [0, N23-1]},
                  continuation='Preparation view ends in 2023; no held-out lifecycle calendar or supplemental member pool.',
                  preparation_only=True)
    schedules = design.get('measurement_noise', {}).get('schedules', {})
    if 'measurement_noise' in design:
        design['measurement_noise']['temporal_exposure'] = 'Preparation view: 2023 only; reserved 2024 programmes are absent.'
    for phase, windows in schedules.items():
        schedules[phase] = [dict(w, end_time_step=min(w['end_time_step'], N23-1)) for w in windows if w['start_time_step'] < N23]
    prefix = Path('evaluation/preparation_2023/inputs')/variant
    for config, key in [(schema.get('robustness', {}), 'events_file'),
                        (schema.get('demand_response', {}), 'requests_file'),
                        (design, 'native_confirmation_robustness_file'),
                        (design.get('measurement_noise', {}), 'channel_file'),
                        (design.get('measurement_noise', {}), 'fault_file')]:
        if key not in config:
            continue
        oldpath = Path(config[key])
        frame = pd.read_parquet(root/oldpath)
        frame = frame.loc[frame.start_time_step.lt(N23)].copy()
        # Do not erase a real service crossing the year boundary. None of the
        # supplied DR requests crosses it; noise exposure can be clipped safely.
        if key == 'requests_file':
            assert frame.end_time_step.lt(N23).all()
        else:
            frame['end_time_step'] = frame.end_time_step.clip(upper=N23-1)
        config[key] = write_frame(root, prefix/oldpath.name, frame)
    return schema


def prepare(repo=ROOT):
    repo = Path(repo)
    results = repo/'results/portugal_rec_native'
    record = results/'heldout_2024'
    manifest = read_json(results/'manifest.json')
    if not (record/'previous_manifest.json').exists():
        if manifest['version'] != 'portugal-native-2023-2024-v3.2':
            raise ValueError('Expected the boundary-repaired v3.2 source edition')
        dump(record/'previous_manifest.json', manifest)
    original_manifest = read_json(record/'previous_manifest.json')
    inputs_path = record/'source_input_checksums.json'
    if inputs_path.exists():
        for relative, expected in read_json(inputs_path).items():
            if digest(repo/relative) != expected:
                raise ValueError(f'Previous-edition input changed: {relative}; review before rebuilding the campaign')
    roots = {f: repo/f'data/datasets/rec_pt_2023_{f}_v2' for f in DYNAMIC}
    exit_t = choose_exit(roots.values())
    pairs, timelines, event_index, lineage, catalogues = [], [], [], [], {}
    originals, scenarios = {}, {}
    for entry in original_manifest['variants']:
        path = repo/entry['schema']
        root = path.parent.parent
        archived = root/'evaluation/campaign_2024/source_schemas'/path.name
        if not archived.exists():
            dump(archived, read_json(path))
        original = read_json(archived)
        name, family = entry['variant'], entry['family']
        originals[name] = original
        schema = deepcopy(original)
        if name in DYNAMIC.get(family, []):
            events = campaign_events(family, exit_t)
            schema, files, supplement = extend_dynamic(root, original, family, events)
            lineage += [dict(family=family, variant=name, **x) for x in files]
            catalogues[family] = supplement
            event_index += [dict(variant=name, **e) for e in events]
        update_identity(schema)
        scenarios[name] = schema
        dump(path, schema)
    if not inputs_path.exists():
        paths = set()
        def collect(value, root):
            if isinstance(value, dict):
                for v in value.values():
                    collect(v, root)
            elif isinstance(value, list):
                for v in value:
                    collect(v, root)
            elif isinstance(value, str) and value.endswith('.parquet') and (root/value).is_file():
                paths.add(root/value)
        for entry in original_manifest['variants']:
            root = (repo/entry['schema']).parent.parent
            collect(originals[entry['variant']], root)
            for folder in ('catalogs', 'timeseries', 'chargers', 'deferrables', 'shared'):
                paths.update((root/folder).glob('*.parquet'))
        dump(inputs_path, {str(p.relative_to(repo)): digest(p) for p in sorted(paths)})
    reference_specs = [
        ('core_30_combined', 'core_30_combined_no_disturbances', 'disturbances'),
        ('core_30_combined', 'core_30_combined_nominal_limits', 'electrical_limits'),
        ('premium_100_allin', 'premium_100_allin_no_disturbances', 'disturbances'),
        ('premium_100_allin', 'premium_100_allin_no_dr', 'dr_activation'),
        ('core_30_topology', 'core_30_topology_frozen_2024', 'topology_2024'),
        ('premium_100_allin', 'premium_100_allin_frozen_2024', 'topology_2024'),
    ]
    preparation = []
    for entry in original_manifest['variants']:
        root = (repo/entry['schema']).parent.parent
        relative = root.relative_to(repo)/'evaluation/preparation_2023'/f"{entry['variant']}.json"
        dump(repo/relative, preparation_view(root, originals[entry['variant']], entry['variant']))
        preparation.append(dict(variant=entry['variant'], schema=str(relative), sha256=digest(repo/relative),
                                interval=[0, N23-1], role='Preparation entry point, not an additional experimental scenario'))
    references = []
    for source, name, factor in reference_specs:
        family = 'core_30' if source.startswith('core') else 'premium_100'
        root = roots[family]
        scenario = scenarios[source]
        ref = deepcopy(scenario)
        factors = deepcopy(scenario['experimental_design']['factors'])
        if factor == 'disturbances':
            ref = remove_disturbances(root, scenario, name)
            factors.update(H=False, G=False)
        elif factor == 'electrical_limits':
            normal = scenarios['core_30_normal']
            for m, b in ref['buildings'].items():
                if m in normal['buildings']:
                    b['electrical_service'] = deepcopy(normal['buildings'][m]['electrical_service'])
            factors['E'] = False
        elif factor == 'dr_activation':
            ref['demand_response']['enabled'] = False
            factors['D'] = False
        else:
            ref = deepcopy(originals[source])
            # T remains true: the 2023 history is retained, only the new campaign is absent.
        update_identity(ref, name, name.replace('_', ' ').title(), factors)
        # References live one directory deeper than principal evaluation files.
        ref['root_directory'] = '../..'
        rel = f'data/datasets/rec_pt_2023_{family}_v2/evaluation/references/{name}.json'
        pair = dict(scenario=source, reference=name, factor=factor,
                    scenario_schema=f'data/datasets/rec_pt_2023_{family}_v2/evaluation/{source}.json',
                    reference_schema=rel, seed=scenario['random_seed'], scored_interval=[N23, TOTAL-1],
                    contrast_scope='2024 lifecycle campaign' if factor == 'topology_2024' else 'whole factor campaign; not attribution to one individual fault',
                    initial_state='Same run start and state/history convention; pair controller seeds and checkpoints explicitly',
                    cohort_policy='Full offered services plus common-member exposure; report excluded post-exit requests separately, never as fulfilled')
        ref['experimental_design']['paired_reference'] = pair
        dump(repo/rel, ref)
        references.append(dict(family=family, variant=name, schema=rel, sha256=digest(repo/rel), factors=factors))
        pairs.append(pair)
    manifest = deepcopy(original_manifest)
    manifest.update(version=VERSION, campaign_revision=REVISION, references=references, preparation_schemas=preparation,
        previous_edition=dict(version=original_manifest['version'], integrity_revision=original_manifest.get('integrity_revision'),
            git_commit=PREVIOUS_COMMIT, manifest='results/portugal_rec_native/heldout_2024/previous_manifest.json'),
        validation_scope='Current lifecycle and paired-reference evidence: heldout_2024/validation.json. Earlier reports retain their original edition and scope.',
        catalogue_description='Four unchanged physical base inventories; nine principal configurations and six matched references. Dynamic histories add one separately catalogued service member per affected base.',
        heldout_protocol=dict(preparation_year=2023, final_test_year=2024, controller_information='No future event calendar or test outcomes for training/selection/preprocessing',
            legacy_phase_names='development/confirmation are inherited disturbance programme names, not permission to use any 2024 record during preparation',
            initial_state='2024 continues the 2023 topology. Continuous runs carry physical state; window probes initialize declared state and do not prove annual state carry-over.',
            event_windows='14 days before and 14 days after each campaign cluster; May includes installation, two-day replacement gap and recovery. Annual score remains all 2024.'),
        campaign_events=event_index)
    for entry in manifest['variants']:
        entry['sha256'] = digest(repo/entry['schema'])
    dump(results/'manifest.json', manifest)
    for family in DYNAMIC:
        root = roots[family]
        rows = []
        for variant in DYNAMIC[family]:
            schema = scenarios[variant]
            ticks = [0, N23-1, N23, TOTAL-1] + [e['time_step']+d for e in schema['topology_events'] if e['time_step'] >= N23 for d in (-1, 0, 1)]
            for t in sorted(set(ticks)):
                rows.append(dict(variant=variant, time_step=t, timestamp_utc=stamp(t), **resource_state(schema, t)))
        timelines += rows
        dump(root/'evaluation/campaign_2024/catalogue.json', dict(version=VERSION,
            base_member_inventory=30 if family == 'core_30' else 100,
            supplemental_members=[catalogues[family]],
            added_storage_assets=[dict(member_id=e['target_member_id'], asset_id=e['target_asset_id'], installed_at=e['time_step'], attributes=e['overrides'])
                                  for e in campaign_events(family, exit_t) if e['operation']=='add_asset'],
            timeline=rows))
    for root in sorted((repo/'data/datasets').glob('rec_pt_2023_*_v2')):
        plan = root/'catalogs/lifecycle_plan.json'
        archive = root/'evaluation/campaign_2024/source_lifecycle_plan.json'
        if plan.exists() and not archive.exists():
            dump(archive, read_json(plan))
        dump(plan, dict(**lifecycle(root), edition=VERSION, campaign_revision=REVISION,
                       historical_member_narratives='The base member parquet stories are preserved source metadata from v3.2; current executable lifecycle is recorded here.'))
        for filename in ['dataset_manifest.json', 'CURRENT_EVALUATION.json']:
            data = read_json(root/filename)
            data.update(version=VERSION, campaign_revision=REVISION)
            data['member_catalogue_scope'] = 'Original static physical inventory; prose fields retain v3.2 source histories. Current per-variant lifecycle is in catalogs/lifecycle_plan.json; supplemental identities are in evaluation/campaign_2024/catalogue.json where applicable.'
            entries = [v for v in manifest['variants'] if Path(v['schema']).parts[2] == root.name]
            data['references'] = [v for v in references if Path(v['schema']).parts[2] == root.name]
            data['preparation_schemas'] = [v for v in preparation if Path(v['schema']).parts[2] == root.name]
            if filename == 'CURRENT_EVALUATION.json':
                data['variants'] = entries
            dump(root/filename, data)
        base = read_json(root/'schema.json')
        base['dataset_contract_version'] = VERSION
        if 'experimental_design' in base:
            base['experimental_design']['version'] = VERSION
        dump(root/'schema.json', base)
        refresh_checksums(root)
    windows = [dict(name=name, start_time_step=step(start), end_time_step_exclusive=step(end),
                    start_local=start, end_local_exclusive=end, purpose='reserved event analysis; not preparation')
               for name, start, end in [
                   ('admission', '2024-01-29 12:00', '2024-02-26 12:00'),
                   ('storage_installation_and_replacement', '2024-04-29 12:00', '2024-06-05 12:00'),
                   ('permanent_departure', '2024-07-29 12:00', '2024-08-26 12:00'),
                   ('pv_retirement', '2024-10-28 12:00', '2024-11-25 12:00')]]
    dump(record/'campaign.json', dict(version=VERSION, revision=REVISION, events=event_index, pairs=pairs,
        timelines=timelines, derived_files=lineage, supplemental_members=catalogues, reserved_analysis_windows=windows))
    return dict(version=VERSION, events=len(event_index), references=len(pairs), exit_step=exit_t, exit_utc=stamp(exit_t))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT)
    print(json.dumps(prepare(parser.parse_args().repo), indent=2))

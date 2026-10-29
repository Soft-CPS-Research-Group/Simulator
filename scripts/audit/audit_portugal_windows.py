"""Portuguese winter/summer windows with the user's actual market convention."""
import gc
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import argparse
import resource
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('outputs/audit/portugal_windows.json'))
parser.add_argument('--long-days', type=int, default=0)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from citylearn.citylearn import CityLearnEnv
from citylearn.agents.baseline import NormalPolicy

rows = []
for path in sorted((ROOT/'data/datasets').glob('rec_pt_*/evaluation/*.json')):
    if args.long_days and path.stem not in {'core_4', 'core_30_combined', 'premium_100_allin'}:
        continue
    windows = [('long2023', 0)] if args.long_days else [('winter2023', 96*30), ('summer2024', 35040+96*180)]
    steps = args.long_days*96+1 if args.long_days else 192
    for label, start in windows:
        schema = json.loads(path.read_text())
        schema.update(root_directory=str(path.parent.parent), simulation_start_time_step=start,
            simulation_end_time_step=start+steps-1, episode_time_steps=steps, random_episode_split=False,
            rolling_episode_split=False, physics_invariant_checks=True)
        env = None
        row = dict(schema=str(path), window=label, start=start, transitions=0, status='pass')
        start_clock = time.perf_counter()
        try:
            assert schema['community_market'] == dict(enabled=True, local_price_ratio_to_grid_import=.8, grid_export_price=0.), 'Dataset market must already be enabled'
            env = CityLearnEnv(schema, render_mode='none', offline=True, random_seed=0,
                               export_kpis_on_episode_end=False)
            obs, _ = env.reset(seed=0)
            policy = NormalPolicy(env)
            row['members'] = len(env.buildings)
            ledger_seconds = [0.0]
            for method in ('begin_step', 'settle', 'synchronize'):
                original = getattr(env._provenance_service, method)
                def timed(*a, _original=original, **kw):
                    started = time.perf_counter()
                    try:
                        return _original(*a, **kw)
                    finally:
                        ledger_seconds[0] += time.perf_counter()-started
                # begin_step calls synchronize; time only the two leaf methods.
                if method != 'begin_step':
                    setattr(env._provenance_service, method, timed)
            row['chargers'] = sum(len(b.electric_vehicle_chargers or []) for b in env.buildings)
            row['appliances'] = sum(len(b.deferrable_appliances or []) for b in env.buildings)
            sums = dict(local=0., external_import=0., external_export=0., cost=0.)
            while not env.terminated:
                obs, reward, terminated, truncated, info = env.step(policy.predict(obs, deterministic=True))
                row['transitions'] += 1
                assert np.isfinite(reward).all(), 'Non-finite reward'
                settlement = env._last_community_market_settlement
                imported = sum(r['local_import_kwh'] for r in settlement)
                exported = sum(r['local_export_kwh'] for r in settlement)
                assert abs(imported-exported) < 1e-4, 'Internal energy mismatch'
                for r in settlement:
                    assert r['grid_import_price'] >= 0., 'Negative grid purchase price'
                    assert abs(r['local_buy_price']-.8*r['grid_import_price']) < 1e-8
                    assert abs(r['local_sell_price']-.8*r['grid_import_price']) < 1e-8
                    assert r['grid_export_price'] == 0.
                assert abs(sum(r['local_import_kwh']*r['local_buy_price']-r['local_export_kwh']*r['local_sell_price'] for r in settlement)) < 1e-6, 'Internal cash mismatch'
                sums['local'] += imported
                sums['external_import'] += sum(r['grid_import_kwh'] for r in settlement)
                sums['external_export'] += sum(r['grid_export_kwh'] for r in settlement)
                sums['cost'] += sum(r['settled_cost_eur'] for r in settlement)
                if terminated or truncated: break
            ledger = env._provenance_service
            _, accounting = ledger.summary()
            for key in ('pv_balance_residual_kwh', 'grid_balance_residual_kwh',
                        'unknown_balance_residual_kwh', 'carbon_balance_residual_kgco2',
                        'storage_max_residual_kwh', 'bus_max_residual_kwh'):
                assert abs(accounting[key]) < 1e-3, (key, accounting[key])
            def retained_bytes(obj, seen=None):
                seen = set() if seen is None else seen
                if id(obj) in seen: return 0
                seen.add(id(obj))
                size = sys.getsizeof(obj)
                if isinstance(obj, dict):
                    size += sum(retained_bytes(k, seen)+retained_bytes(v, seen) for k,v in obj.items())
                elif isinstance(obj, (list, tuple)):
                    size += sum(retained_bytes(v, seen) for v in obj)
                return size
            state = [ledger.sources, ledger.sinks, ledger.internal_transfers,
                     [stock.bag for stock in ledger.stocks.values()]]
            row.update(accounting=accounting, ledger_seconds=ledger_seconds[0],
                       ledger_aggregate_bytes=retained_bytes(state),
                       process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.)
            frame = env.evaluate_v2(include_business_as_usual=False)
            assert not frame.duplicated(['level','name','cost_function']).any()
            district = frame[frame['name']=='District'].set_index('cost_function')['value']
            pairs = {
                'local':'district_energy_grid_community_market_local_traded_total_kwh',
                'external_import':'district_energy_grid_community_market_grid_import_after_local_total_kwh',
                'external_export':'district_energy_grid_community_market_grid_export_after_local_total_kwh',
                'cost':'district_cost_community_market_settled_total_eur',
            }
            row['max_kpi_error'] = max(abs(float(district[key])-sums[name]) for name,key in pairs.items())
            assert row['max_kpi_error'] < 1e-3
            row.update(totals=sums, kpi_rows=len(frame), topology_events=len(env.topology_event_log))
        except Exception as exc:
            row.update(status='fail', error=str(exc), traceback=traceback.format_exc(limit=8))
        finally:
            if env is not None: env.close()
        row['seconds'] = round(time.perf_counter()-start_clock, 3)
        rows.append(row)
        args.output.write_text(json.dumps(rows, indent=2))
        print(path.stem, label, row['status'], row['transitions'], row.get('error',''), flush=True)
        del env
        gc.collect()

sys.exit(0 if all(row['status'] == 'pass' for row in rows) else 1)

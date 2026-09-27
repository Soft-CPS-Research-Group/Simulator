"""Feasibility witnesses for two corrected EV visits, using the original engine.

This is a short input-consistency check, not a controller or annual evaluation.
Each witness specifies charging powers in advance and checks native SOC/limits.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd

COMMIT = '978be68b0e0740deab15ecf536805f323b86fa57'


def verify(repo, datasets_root, output):
    repo = repo.resolve()
    sys.path.insert(0, str(repo))
    from citylearn.citylearn import CityLearnEnv
    assert Path(sys.modules['citylearn.citylearn'].__file__).resolve() == repo/'citylearn/citylearn.py'
    assert not subprocess.check_output(['git', 'diff', COMMIT, '--', 'citylearn'], cwd=repo)
    assert not subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', 'citylearn'], cwd=repo)
    root = datasets_root.resolve()/'data/datasets/rec_pt_2023_core_30_v2'
    fixes = json.loads((root/'native_input_corrections.json').read_text())['visitor_sessions']
    sessions = pd.read_parquet(root/'catalogs/charging_sessions.parquet')
    bounds, witnesses, traces = [], [], []
    for path in sorted((root/'evaluation').glob('*.json')):
        schema = json.loads(path.read_text())
        for fix in fixes:
            start, stop = fix['arrival_time_step'], fix['departure_time_step']
            site = schema['buildings'][fix['member_id']]
            charger = site['chargers'][fix['charger_id']]['attributes']
            session = sessions.loc[sessions.session_id.eq(fix['session_id'])].iloc[0]
            battery = schema['electric_vehicles_def'][session.ev_id]['battery']['attributes']
            assert battery['loss_coefficient'] == battery['capacity_loss_coefficient'] == 0
            assert {v[1] for v in battery['power_efficiency_curve']} == {0.96}
            eta = charger['efficiency'] * np.sqrt(0.96)
            requested = (fix['target_soc']-fix['arrival_soc'])*battery['capacity']
            # Generous bound: no site sharing and no battery taper. Failure
            # against this bound proves the old request physically impossible.
            old_upper = charger['max_charging_power']*.25*5*eta
            assert requested > old_upper + .1
            if path.stem == 'core_30_normal':
                bounds.append(dict(session_id=fix['session_id'],requested_kwh=requested,
                    old_minutes=75,new_minutes=90,old_max_stored_kwh=old_upper,
                    old_unavoidable_deficit_kwh=requested-old_upper,
                    new_max_stored_kwh=charger['max_charging_power']*.25*6*eta,
                    charger_power_kw=charger['max_charging_power'],charger_efficiency=charger['efficiency'],
                    battery_one_way_efficiency=float(np.sqrt(.96)),target_soc=fix['target_soc'],arrival_soc=fix['arrival_soc']))
            overlapping = sessions.loc[(sessions.member_id==fix['member_id']) &
                (sessions.arrival_time_step < stop) & (sessions.departure_time_step > start)]
            schedules = {}
            for row in overlapping.to_dict('records'):
                assert row['arrival_time_step'] == start
                attrs = schema['electric_vehicles_def'][row['ev_id']]['battery']['attributes']
                ca = site['chargers'][row['charger_id']]['attributes']
                gain = max(0.,row['target_soc']-row['arrival_soc_reference'])*attrs['capacity']
                slots = 6 if row['session_id']==fix['session_id'] else 2
                power = (gain+.001)/(slots*.25*ca['efficiency']*np.sqrt(.96))
                assert ca['min_charging_power'] <= power <= ca['max_charging_power']
                schedules[row['charger_id']] = dict(power=power,slots=slots,**row)
            schema.update(root_directory=str(root),simulation_start_time_step=start,
                simulation_end_time_step=stop,episode_time_steps=stop-start+1,physics_invariant_checks=True)
            schema_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            with CityLearnEnv(schema) as env:
                env.reset()
                member = next(b for b in env.buildings if b.name==fix['member_id'])
                evs = {}
                for ch in member.electric_vehicle_chargers:
                    if ch.charger_id not in schedules:continue
                    row = schedules[ch.charger_id]
                    assert ch.connected_electric_vehicle.name == row['ev_id']
                    evs[ch.charger_id] = ch.connected_electric_vehicle
                    assert abs(float(evs[ch.charger_id].battery.soc[0])-row['arrival_soc_reference']) < 1e-6
                max_violation = 0.
                while not env.terminated:
                    t = env.time_step
                    actions = []
                    for b in env.buildings:
                        ca = {}
                        for ch in b.electric_vehicle_chargers or []:
                            p = schedules.get(ch.charger_id) if b.name==member.name else None
                            ca[ch.charger_id] = p['power']/ch.max_charging_power if p and t < p['slots'] else 0.
                        actions.append(dict(electrical_storage_action=0.,electric_vehicle_storage_actions=ca,
                            deferrable_appliance_actions={'deferrable_appliance_'+a.name:0. for a in b.deferrable_appliances or []}))
                    env._runtime_service._step_once(parsed_actions=actions,collect_observations=False,collect_info=False)
                    limits = member._electrical_service_config['limits']
                    total = float(member._charging_total_power_history_kw[t])
                    phases = {k:float(v[t]) for k,v in member._charging_phase_power_history_kw.items()}
                    violation = member._ops_service._electrical_service_violation_kw(total,phases,limits['total'],limits['per_phase'],('L1','L2','L3'))
                    max_violation = max(max_violation,violation)
                    for ch in member.electric_vehicle_chargers:
                        if ch.charger_id not in evs:continue
                        traces.append(dict(variant=path.stem,case=fix['session_id'],step=start+t,charger_id=ch.charger_id,
                            soc=float(evs[ch.charger_id].battery.soc[t]),grid_energy_kwh=float(ch.electricity_consumption[t]),
                            site_power_kw=total,max_phase_power_kw=max(phases.values()),violation_kw=violation))
                assert env.time_step == 6
                assert max_violation < 1e-5
                for cid,ev in evs.items():
                    target = schedules[cid]['target_soc']
                    final_soc = float(ev.battery.soc[5])
                    assert final_soc >= target-1e-6,(path.stem,cid,target,final_soc)
                    witnesses.append(dict(variant=path.stem,case=fix['session_id'],session_id=schedules[cid]['session_id'],
                        charger_id=cid,target_soc=target,final_soc=final_soc,deficit_kwh=max(0.,target-final_soc)*ev.battery.capacity,
                        site_max_violation_kw=max_violation,schema_sha256=schema_sha,action_steps=6))
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(traces).to_parquet(output/'core30_session_witness_trace.parquet',index=False)
    result = dict(scope='Two session feasibility certificates, five Core-30 variants, fixed input actions; no algorithm comparison or annual certification',
        original_engine_commit=COMMIT,core_unmodified=True,short_windows=10,annual_runs=0,bounds=bounds,witnesses=witnesses,
        inputs_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [root/'native_input_corrections.json',root/'catalogs/charging_sessions.parquet',root/'chargers/CHG-M028-01.parquet']},passed=True)
    (output/'core30_session_feasibility.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(passed=True,short_windows=10,bounds=bounds),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--datasets-root',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    verify(a.repo,a.datasets_root or a.repo,a.output or a.repo/'results/portugal_rec_native')

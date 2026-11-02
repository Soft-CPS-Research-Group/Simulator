"""References and measurements against the original, unpatched simulator."""
from pathlib import Path
import argparse,json,sys,time,hashlib,subprocess
import numpy as np
import pandas as pd

def phase_split(connection):
    return np.array([float(connection==p) for p in ('L1','L2','L3')]) if connection in ('L1','L2','L3') else np.ones(3)/3

def controller(env,payload,kind,request):
    t=env.time_step;obs={}
    for tab,arr in payload['tables'].items():
        sp=env.entity_specs['tables'][tab]
        obs.update({(tab,i):dict(zip(sp['features'],arr[j])) for j,i in enumerate(sp['ids'])})
    actions=[]
    for b in env.buildings:
        row=obs[('building',b.name)];apps={};down=request is not None and request['direction']=='down'
        for app in b.deferrable_appliances or []:
            ao=app.observations();defer=down and ao.get('pending',0) and ao.get('latest_start_time_step',0)>request['end_time_step']
            apps['deferrable_appliance_'+app.name]=float(not defer)
        base,phases=b._ops_service._estimate_non_controllable_base_power(apps)
        load=float(row['non_shiftable_load']);bad=not np.isfinite(load) or load<0
        if bad:load=0.
        pv=float(obs.get(('pv',b.name+'/pv'),{}).get('generation_power_kw',0.));badpv=not np.isfinite(pv) or pv<0
        if badpv:pv=0.
        true_native=float(b.energy_simulation.non_shiftable_load[t])*4
        true_pv=float(b._pv_generation_to_control_step(np.array([b.energy_simulation.solar_generation[t]]))[0])*4
        delta=load*4-pv-(true_native-true_pv);base+=delta
        split=phase_split(env.schema['buildings'][b.name]['electrical_service'].get('default_split','balanced'))
        phases={p:phases[p]+delta*split[j] for j,p in enumerate(('L1','L2','L3'))}
        limits=b._electrical_service_config['limits'];remaining=np.array([limits['per_phase'][p]['import_kw']-phases[p] for p in ('L1','L2','L3')]);total=limits['total']['import_kw']-base
        ca={};estimated=0.
        for ch in sorted(b.electric_vehicle_chargers or [],key=lambda c:float(c.charger_simulation.electric_vehicle_departure_time[t]) if c.connected_electric_vehicle else 1e9):
            ev=ch.connected_electric_vehicle;power=0.
            if ev is not None:
                cap=ev.battery.capacity;soc=float(obs[('ev',ev.name)]['soc']);target=min(1.,float(ch.charger_simulation.electric_vehicle_required_soc_departure[t])+1./cap);need=max(0.,target-soc)*cap
                ph=phase_split(ch.phase_connection);nz=ph>0
                maxp=max(0.,min(total,float(np.min(remaining[nz]/ph[nz])),ch.max_charging_power))
                hours=float(ch.charger_simulation.electric_vehicle_departure_time[t])*.25
                defer=bad and hours>2+need/max(maxp*.90,.01)
                if down and hours>(request['end_time_step']-t+1)*.25+need/max(maxp*.90,.01)+1:defer=True
                if need>.01 and not defer and not b.power_outage:
                    power=maxp
                    if power<ch.min_charging_power:power=0.
                if power:total-=power;remaining-=power*ph
            ca[ch.charger_id]=power/ch.max_charging_power;estimated+=power
        bat=b.electrical_storage
        storage=float(np.clip(-(base+estimated)/bat.nominal_power,-1,1)) if bat.nominal_power and kind=='battery' and not badpv else 0.
        if bat.nominal_power and request is not None and kind=='battery':storage=-1. if down else 1.
        if b.power_outage and bat.nominal_power:storage=-1.
        actions.append(dict(electrical_storage_action=storage,electric_vehicle_storage_actions=ca,deferrable_appliance_actions=apps))
    return actions

def run(repo,name,kind,steps,start=0):
    sys.path.insert(0,str(repo));from citylearn.citylearn import CityLearnEnv
    commit='978be68b0e0740deab15ecf536805f323b86fa57'
    assert Path(sys.modules['citylearn.citylearn'].__file__).resolve()==repo/'citylearn/citylearn.py', 'Wrong runtime imported'
    assert not subprocess.check_output(['git','diff',commit,'--','citylearn'],cwd=repo), 'Engine differs from the original pinned commit'
    assert not subprocess.check_output(['git','ls-files','--others','--exclude-standard','citylearn'],cwd=repo), 'Extra runtime modules'
    manifest=json.loads((repo/'results/portugal_rec_native/manifest.json').read_text());entry=next(v for v in manifest['variants'] if v['variant']==name)
    path=repo/entry['schema'];root=path.parent.parent;s=json.loads(path.read_text());s.update(root_directory=str(root),simulation_start_time_step=start,simulation_end_time_step=start+steps-1,episode_time_steps=steps,physics_invariant_checks=True)
    requests=pd.read_parquet(root/s['demand_response']['requests_file']).to_dict('records') if s.get('demand_response',{}).get('enabled') else []
    out=repo/'results/portugal_rec_native/annual';out.mkdir(parents=True,exist_ok=True)
    key=f'{name}_{kind}'+(f'_window_{start}_{steps}' if steps!=35040 or start else '')
    rows=[];departures=[];limits=[];declared_phase_issues=[];cut_issues=[];boundary=[];tic=time.monotonic();max_balance=max_cash=max_vehicle_overpower=0.
    catalog=pd.read_parquet(root/'catalogs/electric_vehicles.parquet').set_index('ev_id')
    exits={r['member_id']:r['exit_step'] for r in s['experimental_design'].get('member_departures',[])}
    with CityLearnEnv(s) as env:
        env.reset()
        while not env.terminated:
            t=env.time_step;absolute=t+start;buildings=list(env.buildings)
            req=next((r for r in requests if r['start_time_step']<=absolute<=r['end_time_step']),None)
            # Controller dates use episode-local slots; native DR uses absolute time.
            localreq=None if req is None else {**req,'end_time_step':req['end_time_step']-start}
            action=controller(env,env.observations,kind,localreq)
            assets={b.name:(b.electrical_storage,b._charging_total_power_history_kw,dict(b._charging_phase_power_history_kw),b._electrical_service_config['limits'],list(b.deferrable_appliances or []),list(b.electric_vehicle_chargers or [])) for b in buildings}
            native={b.name:float(b.energy_simulation.non_shiftable_load[t]) for b in buildings};outages={b.name:bool(b.power_outage) for b in buildings}
            leaving=[]
            for b in buildings:
                for ch in b.electric_vehicle_chargers or []:
                    ev=ch.connected_electric_vehicle
                    if ev is None:continue
                    sim=ch.charger_simulation
                    if t+1<len(sim.electric_vehicle_charger_state) and (sim.electric_vehicle_charger_state[t+1]!=1 or sim.electric_vehicle_id[t+1]!=ev.name or sim.electric_vehicle_session_id[t+1]!=sim.electric_vehicle_session_id[t]):
                        if absolute+1<exits.get(b.name,35040):leaving.append((b.name,ch.charger_id,ev,float(sim.electric_vehicle_required_soc_departure[t])))
            env._runtime_service._step_once(parsed_actions=action,collect_observations=False,collect_info=False)
            for mid,cid,ev,target in leaving:departures.append(dict(step=absolute,member=mid,charger=cid,ev_id=ev.name,gap_kwh=max(0.,target-float(ev.battery.soc[t]))*ev.battery.capacity))
            settle=env._last_community_market_settlement
            cash=sum((x['local_import_kwh']-x['local_export_kwh'])*x['local_price'] for x in settle);max_cash=max(max_cash,abs(cash))
            imp=exp=pv=nload=unserved=appload=evload=cut_residual=0.
            for b in buildings:
                bat,totalhist,phasehist,lim,apps,chargers=assets[b.name]
                net=float(b.net_electricity_consumption[t]);imp+=max(net,0);exp+=max(-net,0);pv+=abs(float(b.solar_generation[t]));nload+=native[b.name]
                unserved+=max(0.,native[b.name]-float(b.non_shiftable_load_electricity_consumption[t]))
                appload+=sum(float(a.electricity_consumption[t]) for a in apps);evload+=sum(float(c.electricity_consumption[t]) for c in chargers)
                terms=float(bat.electricity_consumption[t])+sum(float(getattr(b,f)[t]) for f in ('cooling_electricity_consumption','heating_electricity_consumption','dhw_electricity_consumption','non_shiftable_load_electricity_consumption','solar_generation','chargers_electricity_consumption','deferrable_appliances_electricity_consumption'))
                if not outages[b.name]:
                    max_balance=max(max_balance,abs(net-terms))
                    residual=b._ops_service._electrical_service_violation_kw(float(totalhist[t]),{p:float(v[t]) for p,v in phasehist.items()},lim['total'],lim['per_phase'],('L1','L2','L3'))
                    if residual>1e-4:limits.append(dict(step=absolute,member=b.name,residual_kw=residual))
                    actual_phases={p:float(v[t]) for p,v in phasehist.items()}
                    default=phase_split(s['buildings'][b.name]['electrical_service'].get('default_split','balanced'))
                    for appliance in apps:
                        connection=s['native_compatibility']['declared_appliance_phases'].get(appliance.name)
                        if connection is None:continue
                        correction=(phase_split(connection)-default)*float(appliance.electricity_consumption[t])*4
                        for j,p in enumerate(('L1','L2','L3')):actual_phases[p]+=correction[j]
                    physical_residual=b._ops_service._electrical_service_violation_kw(float(totalhist[t]),actual_phases,lim['total'],lim['per_phase'],('L1','L2','L3'))
                    if physical_residual>1e-4:declared_phase_issues.append(dict(step=absolute,member=b.name,residual_kw=physical_residual))
                elif terms>5e-4:
                    cut_residual+=terms;cut_issues.append(dict(step=absolute,member=b.name,unbacked_served_energy_kwh=terms))
                for ch in chargers:
                    ev=ch._Charger__past_connected_evs[t]
                    if ev is not None and ev.name in catalog.index:
                        over=float(ch.electricity_consumption[t])*4-float(catalog.loc[ev.name,'max_ac_charging_power'])
                        max_vehicle_overpower=max(max_vehicle_overpower,over)
            sr=env._demand_response_service._settlement_by_time_step.get(t,{})
            rows.append(dict(step=absolute,members=len(buildings),import_kwh=imp,export_kwh=exp,pv_kwh=pv,native_kwh=nload,unserved_kwh=unserved,ev_kwh=evload,appliance_kwh=appload,outage_members=sum(outages.values()),grid_cost_eur=sum(x['grid_import_kwh']*x['grid_import_price'] for x in settle),cash_residual_eur=cash,unbacked_served_energy_kwh=cut_residual,dr_active=req is not None,dr_baseline_valid=bool(sr.get('baseline_valid',False)),dr_credited_kwh=float(sr.get('credited_kwh',0))))
            if t%2000==0:print(json.dumps(dict(case=key,step=t,seconds=round(time.monotonic()-tic,1))),flush=True)
        topo=env._topology_service
        apps=[a for mid in topo.member_pool for a in topo.historical_deferrable_appliances(mid)] if env.topology_mode=='dynamic' else [a for b in env.buildings for a in b.deferrable_appliances or []]
        result=dict(case=key,configuration=name,reference=kind,source_commit=manifest['simulator_commit'],schema_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),records=steps,action_intervals=env.time_step,seconds=time.monotonic()-tic,tariff_policy=manifest['tariff_policy'],max_site_balance_residual_kwh=max_balance,max_cash_residual_eur=max_cash,max_ev_ac_excess_kw=max_vehicle_overpower,residual_limit_intervals=len(limits),worst_limit_residual_kw=max((x['residual_kw'] for x in limits),default=0),departures=len(departures),departure_gap_over_0_1kwh=sum(x['gap_kwh']>.1 for x in departures),departure_gap_kwh=sum(x['gap_kwh'] for x in departures),appliances=[dict(name=a.name,**a.service_summary()) for a in apps],topology_events=list(env.topology_event_log),unbacked_cut_intervals=len(cut_issues),native_mobility='prescribed session arrivals; trip energy balance not simulated',totals=pd.DataFrame(rows).select_dtypes('number').sum().to_dict())
        (out/(key+'_dr.json')).write_text(json.dumps(list(env._demand_response_service.settlement_history),indent=2,default=str)+'\n')
    result['declared_appliance_phase_violation_intervals']=len(declared_phase_issues)
    result['worst_declared_appliance_phase_residual_kw']=max((x['residual_kw'] for x in declared_phase_issues),default=0.)
    for suffix,data in [('trace',rows),('departures',departures),('limits',limits),('declared_phase_issues',declared_phase_issues),('cut_issues',cut_issues)]:pd.DataFrame(data).to_parquet(out/(key+'_'+suffix+'.parquet'),index=False)
    (out/(key+'.json')).write_text(json.dumps(result,indent=2,default=str)+'\n')
    print(json.dumps(dict(completed=key,seconds=result['seconds'])),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--repo',type=Path,required=True);p.add_argument('--reference',choices=['direct','battery'],default='battery');p.add_argument('--steps',type=int,default=35040);p.add_argument('--start',type=int,default=0);a=p.parse_args();run(a.repo.resolve(),a.name,a.reference,a.steps,a.start)

"""Append 2024 to the same four Portugal bases / nine original-engine schemas.

Input is a frozen 2023 directory; output must be a separate staging directory.
No runtime imports, controller evaluation, equipment resizing or topology replay.
"""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import argparse,json,hashlib,shutil
import numpy as np
import pandas as pd
import pvlib
import portugal_profiles_2024 as profiles
from portugal_site_calendar import opening_series
from portugal_lifecycle_stories import synchronize
ROOT=Path(__file__).resolve().parents[1]
N23=35040;N24=35136;TOTAL=N23+N24
VERSION='portugal-native-2023-2024-v3.1'
COMMIT='978be68b0e0740deab15ecf536805f323b86fa57'
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+'\n')
def write(p,f):p.parent.mkdir(parents=True,exist_ok=True);f.to_parquet(p,index=False,compression='zstd')
def read(p):return pd.read_parquet(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def joined(p,f):write(p,pd.concat([read(p),f],ignore_index=True))
def shift(t):
    # Preserve weekday and Lisbon civil hour for synthetic seasonal campaigns.
    d=pd.Timestamp('2023-01-01',tz='UTC')+pd.Timedelta(minutes=int(t)*15)
    local=d.tz_convert('Europe/Lisbon').tz_localize(None)+pd.Timedelta(weeks=52)
    stamp=local.tz_localize('Europe/Lisbon',ambiguous=True,nonexistent='shift_forward').tz_convert('UTC')
    return int((stamp-pd.Timestamp('2023-01-01',tz='UTC')).total_seconds()/900)
def blank_charger(f):
    f=f.copy();f['electric_vehicle_charger_state']=np.int8(0)
    for col in f:
        if col in ('electric_vehicle_id','electric_vehicle_session_id'):f[col]=''
        elif col!='electric_vehicle_charger_state':f[col]=np.nan
    return f

def append_events(p):
    f=read(p)
    if 'start_time_step' not in f or 'end_time_step' not in f:return
    extra=[]
    for i,row in f.iterrows():
        d=row.to_dict();a=int(d['start_time_step']);b=int(d['end_time_step'])
        if a==0 and b==N23-1:
            f.at[i,'end_time_step']=TOTAL-1;continue
        d['start_time_step']=shift(a);d['end_time_step']=d['start_time_step']+b-a
        if d['start_time_step']<N23 or d['end_time_step']>=TOTAL:raise ValueError((p,d))
        for k in ('event_id','request_id','id'):
            if k in d and pd.notna(d[k]):d[k]=str(d[k])+'_2024'
        for k in list(d):
            if 'timestamp' in k:d[k]=str(pd.Timestamp('2023-01-01',tz='UTC')+pd.Timedelta(minutes=d['start_time_step' if 'start' in k else 'end_time_step']*15))
        extra.append(d)
    write(p,pd.concat([f,pd.DataFrame(extra)],ignore_index=True))

def state_at_end(schema,mid,m):
    b=schema['buildings'][mid];live=b.get('include',True)
    presence={'pv':b.get('pv',{}).get('attributes',{}).get('nominal_power',0)>0,'electrical_storage':b.get('electrical_storage',{}).get('attributes',{}).get('capacity',0)>0}
    removed=set()
    for e in sorted(schema.get('topology_events',[]),key=lambda x:x['time_step']):
        if e.get('target_member_id')!=mid:continue
        op=e['operation'];kind=e.get('target_asset_type')
        if op in ('add_member','remove_member'):live=op=='add_member'
        if kind in presence and op in ('add_asset','remove_asset'):presence[kind]=op=='add_asset'
        if op=='remove_asset':removed.add(e.get('target_asset_id'))
        if op=='add_asset':removed.discard(e.get('target_asset_id'))
    return live,presence,removed

def family(src,dst,ref,entries):
    shutil.copytree(src,dst,dirs_exist_ok=True)
    members=read(src/'catalogs/members.parquet');apps=read(src/'catalogs/deferrables.parquet');old_ev=read(src/'catalogs/electric_vehicles.parquet').set_index('ev_id')
    contracts=json.loads((src/'catalogs/electrical_services.json').read_text())
    key={'micro_4':'micro','core_15':'core15','core_30':'core30','premium_100':'premium'}[entries[0]['family']]
    cfg=SimpleNamespace(key=key,charger_counts={m['member_index']:int(m['physical_charger_count']) for m in members.to_dict('records') if m['has_charging']})
    c=profiles.calendar();cal=c.copy();cal['time_step']+=N23;joined(dst/'shared/calendar.parquet',cal)
    for name in ['weather','carbon_intensity','pricing_btn_bi']:joined(dst/f'shared/{name}.parquet',read(ref/f'{name}.parquet'))
    price=read(dst/'shared/pricing_btn_bi.parquet')
    components=pd.concat([read(ROOT/'data/portugal_reference_2023/price_components_btn_bi.parquet'),read(ref/'price_components_btn_bi.parquet')],ignore_index=True)
    clock=pd.DatetimeIndex(components.timestamp_utc);madrid=clock.tz_convert('Europe/Madrid')
    for k,h in enumerate((4,24,96),1):
        issue=np.arange(N23-h,N23);target=issue+h
        publication=(madrid[target].tz_localize(None).normalize()-pd.Timedelta(days=1)+pd.Timedelta(hours=14)).tz_localize('Europe/Madrid').tz_convert('UTC')
        ix=np.where(clock[issue]>=publication,target,np.minimum(target-96,issue))
        price.loc[issue,f'electricity_pricing_predicted_{k}']=((components.omie_eur_kwh.to_numpy()[ix]+.014)*1.16+components.access_energy_eur_kwh.to_numpy()[target]+.001).astype('float32')
    write(dst/'shared/pricing_btn_bi.parquet',price)
    # Old heterogeneous tariffs are historical source records, not selectable inputs.
    for p in (dst/'shared').glob('pricing_*.parquet'):
        if p.name!='pricing_btn_bi.parquet':
            dest=dst/'sources/2023_unused_tariffs'/p.name;dest.parent.mkdir(exist_ok=True,parents=True);shutil.move(p,dest)
    wf=read(ref/'weather_physics.parquet');mid=pd.DatetimeIndex(wf.timestamp_utc);w={'mid':mid,'position':pvlib.solarposition.get_solarposition(mid,38.72,-9.14),**{k:wf[k].to_numpy() for k in ['temperature','wind','dni','dhi']}}
    prior=read(ref/'eredes_profiles_2024.parquet');oldtemp=read(src/'shared/weather.parquet').outdoor_dry_bulb_temperature.to_numpy()
    thermal={'heating_proxy_kwh':np.maximum(18-w['temperature'],0).sum()/np.maximum(18-oldtemp,0).sum(),'cooling_proxy_kwh':np.maximum(w['temperature']-24,0).sum()/np.maximum(oldtemp-24,0).sum()}
    newframes={};schedules={};new_annual=[];operational_members=[]
    for rawm in members.to_dict('records'):
        m=deepcopy(rawm);mid=m['member_id']
        # Keep the same vacation civil date across the leap year.
        if m['vacation_start_day']>59:m['vacation_start_day']+=1
        operational_members.append(m)
        f,l=profiles.member_load(m,c,w,prior,cfg);oldledger=read(src/f'end_uses/{mid}.parquet')
        for col in ['base_excluding_cycles_kwh','dhw_proxy_kwh','heating_proxy_kwh','cooling_proxy_kwh']:
            goal=oldledger[col].sum()*thermal.get(col,366/365)
            l[col]*=goal/max(l[col].sum(),1e-12)
        for a in apps.loc[apps.member_id==mid].to_dict('records'):
            aid=a['deferrable_id'];pro,sch,_,fixed=profiles.appliance(m,c,a['appliance_type'],bool(a['controlled']),cfg)
            oldpro=read(src/f'deferrables/{aid}_profiles.parquet')
            assert pro.profile_id.tolist()==oldpro.profile_id.tolist()
            for p,q in zip(pro.load_profile,oldpro.load_profile):assert np.allclose(json.loads(p),json.loads(q)),(mid,aid)
            # Device identity, rating and profile stay frozen; only use schedule varies.
            sch['year']=2024
            for col in ('earliest_start_time_step','latest_start_time_step','deadline_time_step'):sch[col]+=N23
            schedules[aid]=sch;joined(dst/f'deferrables/{aid}_schedule.parquet',sch)
            l['fixed_appliances_kwh']+=fixed
        f['non_shiftable_load']=l.sum(axis=1);newframes[mid]=f
        joined(dst/f'timeseries/{mid}.parquet',f.astype('float32'))
        joined(dst/f'end_uses/{mid}.parquet',l.astype('float32'))
        if m['category']!='residential':joined(dst/f'operations/{mid}.parquet',pd.DataFrame({'time_step':cal.time_step,'is_open':opening_series(m,c)}))
        oldframe=read(src/f'timeseries/{mid}.parquet')
        new_annual.append({'member_id':mid,'year':2023,'non_shiftable_load_kwh':float(oldframe.non_shiftable_load.sum()),'pv_generation_kwh':float(oldframe.solar_generation.sum()),'heating_proxy_kwh':float(oldledger.heating_proxy_kwh.sum()),'cooling_proxy_kwh':float(oldledger.cooling_proxy_kwh.sum())})
        new_annual.append({'member_id':mid,'year':2024,'non_shiftable_load_kwh':float(f.non_shiftable_load.sum()),'pv_generation_kwh':float(f.solar_generation.sum()),'heating_proxy_kwh':float(l.heating_proxy_kwh.sum()),'cooling_proxy_kwh':float(l.cooling_proxy_kwh.sum())})
    chargers,sessions,vehicles,charging=profiles.mobility(cfg,operational_members,contracts,c)
    for ev in vehicles:
        eid=ev['ev_id']
        for col in ['battery_capacity_kwh','max_ac_charging_power','consumption_kwh_per_km','soc_preference']:
            assert ev[col]==old_ev.loc[eid,col],(eid,col,ev[col],old_ev.loc[eid,col])
        joined(dst/f'mobility/{eid}.parquet',ev.pop('_ledger'))
    sess=pd.DataFrame(sessions);old_sessions=read(src/'catalogs/charging_sessions.parquet');bridges=[]
    # Continue the last home parking session across New Year. Keep its target
    # and session identity. Update only the formerly censored departure countdown.
    for cid,f in charging.items():
        old=read(src/f'chargers/{cid}.parquet');last=old.iloc[-1]
        if last.electric_vehicle_charger_state==1 and f.iloc[0].electric_vehicle_charger_state==1 and last.electric_vehicle_id==f.iloc[0].electric_vehicle_id:
            firstid=f.iloc[0].electric_vehicle_session_id;oldid=last.electric_vehicle_session_id
            first=sess.loc[sess.session_id==firstid].iloc[0];b=int(first.departure_time_step)
            mask=f.electric_vehicle_session_id.eq(firstid);f.loc[mask,'electric_vehicle_session_id']=oldid;f.loc[mask,'electric_vehicle_required_soc_departure']=last.electric_vehicle_required_soc_departure
            old_sessions.loc[old_sessions.session_id==oldid,'departure_time_step']=N23+b
            sess=sess.loc[sess.session_id!=firstid]
            bridges.append(dict(charger_id=cid,session_id=oldid,continued_slots_2024=b,scope='Uncensor year-end departure countdown only; preserve 2023 state/SOC/target'))
        joined(dst/f'chargers/{cid}.parquet',f)
    for col in ['arrival_time_step','departure_time_step']:sess[col]+=N23
    sess['year']=2024;old_sessions['year']=2023;write(dst/'catalogs/charging_sessions.parquet',pd.concat([old_sessions,sess],ignore_index=True))
    # All native charger paths share the same continuation, with permanent
    # member/vehicle/charger departures applied independently for each variant.
    for entry in entries:
        path=dst/'evaluation'/f"{entry['variant']}.json"
        s=json.loads(path.read_text());variant=entry['variant'];out=dst/'evaluation/native_inputs'/variant
        departed={e['target_member_id'] for e in s.get('topology_events',[]) if e['operation']=='remove_member'}
        owned=set(old_ev.index[old_ev.home_member_id.isin(departed)])
        for mid,b in s['buildings'].items():
            m=members.loc[members.member_id==mid].iloc[0];live,present,removed=state_at_end(s,mid,m)
            f=newframes[mid].copy();oldfile=src/b['energy_simulation'];old=read(oldfile)
            if not present['electrical_storage']:f.non_shiftable_load=np.maximum(0,f.non_shiftable_load-float(m.battery_standby_w)/4000)
            if not present['pv']:f.solar_generation=0.
            outage_indices=np.flatnonzero(old.power_outage.to_numpy()>0)
            for a in outage_indices:
                t=shift(a)-N23
                if 0<=t<N24:f.loc[t,'power_outage']=1
            if not present['electrical_storage']:f.loc[f.power_outage.gt(0),'solar_generation']=0.
            if not live:f[['non_shiftable_load','solar_generation','occupant_count','power_outage']]=0.
            write(dst/b['energy_simulation'],pd.concat([old,f.astype('float32')],ignore_index=True))
            for cid,ch in b.get('chargers',{}).items():
                actual=cid.removeprefix('NATIVE_TEMPLATE__');tail=charging[actual].copy()
                if not live or actual in removed:tail=blank_charger(tail)
                else:
                    bad=tail.electric_vehicle_id.isin(owned)
                    if bad.any():tail.loc[bad]=blank_charger(tail.loc[bad])
                oldpath=ch['charger_simulation'];dest=out/(actual+'.parquet');initial=read(src/oldpath)
                write(dest,pd.concat([initial,tail],ignore_index=True));ch['charger_simulation']=str(dest.relative_to(dst))
            for aid,app in b.get('deferrable_appliances',{}).items():
                for field,value in list(app.items()):
                    if isinstance(value,str) and value.endswith('_schedule.parquet'):
                        tail=schedules[aid] if live and aid not in removed else schedules[aid].iloc[:0]
                        dest=out/(aid+'_schedule.parquet');write(dest,pd.concat([read(src/value),tail],ignore_index=True));app[field]=str(dest.relative_to(dst))
        s['simulation_end_time_step']=TOTAL-1;s['episode_time_steps']=None
        s['dataset_contract_version']=VERSION;s['experimental_design'].update(version=VERSION,years=[2023,2024],year_windows={'2023':[0,N23-1],'2024':[N23,TOTAL-1]},continuation='Topology is not replayed. Permanent exits and retired assets remain absent; same equipment and members.')
        s['experimental_design'].pop('source_physical_fingerprint',None)
        s['experimental_design']['frozen_2023_source_fingerprint']=entry.get('source_physical_fingerprint')
        noise=s['experimental_design'].get('measurement_noise',{})
        if noise:
            noise['temporal_exposure']='Seasonal campaigns in both years; raw telemetry only; no commercial DR baseline qualification.'
            for phase,windows in noise.get('schedules',{}).items():
                noise['schedules'][phase]=windows+[{'start_time_step':shift(v['start_time_step']),'end_time_step':shift(v['start_time_step'])+v['end_time_step']-v['start_time_step']} for v in windows]
        dump(path,s)
    # Extend data event files, including reserved confirmation events. Topology
    # is a one-time history and is intentionally never copied into 2024.
    for p in (dst/'evaluation').rglob('*.parquet'):
        if p.name in ('health.parquet','observations_and_health.parquet','measurement_noise.parquet','robustness.parquet','confirmation_robustness.parquet','requests.parquet'):append_events(p)
    # Repair the cross-year countdown in every runtime and descriptive schedule.
    for br in bridges:
        for p in dst.rglob(br['charger_id']+'.parquet'):
            f=read(p)
            if len(f)!=TOTAL:continue
            mask=f.iloc[:N23].electric_vehicle_session_id.eq(br['session_id']) & f.iloc[:N23].electric_vehicle_charger_state.eq(1)
            if mask.any() and f.iloc[N23].electric_vehicle_session_id==br['session_id']:
                f.loc[np.flatnonzero(mask),'electric_vehicle_departure_time']+=br['continued_slots_2024'];write(p,f)
    write(dst/'catalogs/member_year_totals.parquet',pd.DataFrame(new_annual))
    dump(dst/'extension_2024.json',{'version':VERSION,'years':[2023,2024],'records':TOTAL,'frozen_2023_records':N23,'continuity':'same members/equipment; 2023 topology permanent; no fresh admissions or topology replay in 2024','year_end_session_bridges':bridges,'physical_equipment_resized':False,'profile_generation':'New 2024 operating draws; frozen appliance profiles and EV specs; true 2024 source weather/prices/calendar. Thermal annual budgets respond to degree-hour ratio.','source_reference':'data/portugal_reference_2024','source_history':'2023 annual_* catalogue fields and mobility ledgers are frozen/synthetic diagnostics, not runtime trajectories.'})
    for p in dst.rglob('*.md'):
        t=p.read_text().replace('indexada a OMIE 2023','indexada a OMIE 2023–2024');p.write_text(t)
    for col in ['story']:
        members[col]=members[col].str.replace('indexada a OMIE 2023','indexada a OMIE 2023–2024',regex=False)
    write(dst/'catalogs/members.parquet',members)
    synchronize(dst)
    # Mark historical catalogue/event sources explicitly; runtime schemas rule.
    (dst/'README.md').write_text(f'''# Portugal — 2023 e 2024\n\nMesma comunidade e equipamentos, {TOTAL} registos de 15 minutos.\nOs caminhos com «2023» conservam o nome histórico; os inputs correntes abrangem os dois anos.\nOs schemas em evaluation/ são a fonte executável. As saídas de membros/ativos\nmantêm-se definitivas em 2024; não se repetem os eventos de topologia.\nVer extension_2024.json e docs/portugal_rec_native/Extensao_2024.md.\nOs campos annual_* dos catálogos e os ficheiros antigos em events/ são proveniência de 2023,\nnão contagens atualizadas nem eventos executados. O mercado continua desligado.\n''')
    for entry in entries:
        entry['sha256']=sha(dst/'evaluation'/f"{entry['variant']}.json")
        entry['frozen_2023_source_fingerprint']=entry.pop('source_physical_fingerprint',None)
    default=next((e for e in entries if e['variant'].endswith('_normal')),entries[0])
    shutil.copy2(dst/'evaluation'/f"{default['variant']}.json",dst/'schema.json')
    dump(dst/'CURRENT_EVALUATION.json',{'version':VERSION,'simulator_commit':COMMIT,'status':'dataset_construction_only','years':[2023,2024],'variants':entries})
    dm=json.loads((dst/'dataset_manifest.json').read_text());dm.update(version=VERSION,years=[2023,2024],records=TOTAL,year_windows={'2023':[0,N23-1],'2024':[N23,TOTAL-1]},frozen_2023_commit='393cd200bb66b671d8904eba00688d56d91f08d3');dump(dst/'dataset_manifest.json',dm)
    (dst/'file_checksums.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(dst)}\n' for p in sorted(dst.rglob('*')) if p.is_file() and p!=dst/'file_checksums.sha256'))
    print(dst.name,'built',len(sess),'new sessions',len(bridges),'New Year continuations',flush=True)

def native_event_index(repo, manifest):
    events=[]
    for entry in manifest['variants']:
        schema=repo/entry['schema'];root=schema.parent.parent;s=json.loads(schema.read_text());base={'variant':entry['variant'],'family':entry['family']}
        for phase,file in [('development',s.get('robustness',{}).get('events_file')),('confirmation',s['experimental_design'].get('native_confirmation_robustness_file'))]:
            if file:events.extend(dict(**base,phase=phase,kind='robustness',**row) for row in json.loads(read(root/file).to_json(orient='records')))
        for mid,b in s['buildings'].items():
            f=read(root/b['energy_simulation']);idx=np.flatnonzero(f.power_outage.gt(0));chunks=np.split(idx,np.flatnonzero(np.diff(idx)>1)+1)
            for g in chunks:
                if len(g):events.append(dict(**base,phase='development',kind='grid',member_id=mid,start_time_step=int(g[0]),end_time_step=int(g[-1])))
        if s.get('demand_response',{}).get('enabled'):events.extend(dict(**base,phase='development',kind='dr',**row) for row in json.loads(read(root/s['demand_response']['requests_file']).to_json(orient='records')))
    return events

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();src=args.source.resolve();out=args.output.resolve()
    if src==out or src in out.parents or out in src.parents:raise ValueError('Use separate input and output trees')
    if out.exists():raise FileExistsError('Refusing to overwrite staging output; use a new directory')
    manifest=json.loads((src/'results/portugal_rec_native/manifest.json').read_text())
    for root in sorted((src/'data/datasets').glob('rec_pt_2023_*_v2')):
        entries=[x for x in manifest['variants'] if Path(x['schema']).parts[2]==root.name]
        family(root,out/'data/datasets'/root.name,ROOT/'data/portugal_reference_2024',entries)
    manifest.update(version=VERSION,years=[2023,2024],records=TOTAL,annual_records={'2023':N23,'2024':N24},annual_action_steps=None,year_windows={'2023':[0,N23-1],'2024':[N23,TOTAL-1]},status='dataset_construction_only',frozen_2023_commit='393cd200bb66b671d8904eba00688d56d91f08d3')
    manifest['windows_2023_source']=deepcopy(manifest['windows'])
    for phase,windows in manifest['windows'].items():
        manifest['windows'][phase]=windows+[[shift(a),shift(a)+b-a] for a,b in windows]
    manifest['events_2023_source']=manifest.pop('events');manifest['event_catalogue_scope']='events indexes native event files and native outage columns for both years; events_2023_source preserves the previous index.'
    manifest['events']=native_event_index(out,manifest)
    dump(out/'results/portugal_rec_native/manifest.json',manifest)
if __name__=='__main__':main()

"""Static 2023–2024 input checks. No CityLearn imports or controller evaluation."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
import numpy as np
import pandas as pd
N23=35040;TOTAL=70176
COMMIT='978be68b0e0740deab15ecf536805f323b86fa57'

def verify(repo,source,engine):
 checks=[];stats=[];bridges=[];isolated=0;errors=[];prices=None
 def check(name,ok):
  checks.append({'name':name,'passed':bool(ok)})
  if not ok:errors.append(name)
 def read(p):return pd.read_parquet(p)
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 m=json.loads((repo/'results/portugal_rec_native/manifest.json').read_text());check('exactly nine configurations',len(m['variants'])==9)
 check('core equals pinned original',not subprocess.check_output(['git','diff',COMMIT,'--','citylearn'],cwd=engine))
 check('no extra runtime modules',not subprocess.check_output(['git','ls-files','--others','--exclude-standard','citylearn'],cwd=engine))
 for root in sorted((repo/'data/datasets').glob('rec_pt_2023_*_v2')):
  label=root.name;oldroot=source/'data/datasets'/root.name
  members=read(root/'catalogs/members.parquet');oldm=read(oldroot/'catalogs/members.parquet')
  check(label+': same members and equipment',members.drop(columns=['story']).equals(oldm.drop(columns=['story'])))
  cal=read(root/'shared/calendar.parquet');expected=pd.date_range('2023-01-01','2025-01-01',freq='15min',inclusive='left',tz='UTC')
  check(label+': complete regular UTC axis',pd.DatetimeIndex(cal.timestamp_utc).equals(expected) and np.array_equal(cal.time_step,np.arange(TOTAL)))
  local=expected.tz_convert('Europe/Lisbon');check(label+': leap day',sum(local.strftime('%Y-%m-%d')=='2024-02-29')==96)
  for date,n in [('2024-03-31',92),('2024-10-27',100)]:check(label+': DST '+date,sum(local.strftime('%Y-%m-%d')==date)==n)
  check(label+': holidays 2024',all(cal.loc[local.strftime('%Y-%m-%d')==d,'is_national_holiday'].all() for d in ['2024-03-29','2024-03-31','2024-05-30']))
  check(label+': school summer break starts after June 28',cal.loc[local.strftime('%Y-%m-%d')=='2024-06-29','is_school_break'].all())
  price=read(root/'shared/pricing_btn_bi.parquet');check(label+': finite common price',len(price)==TOTAL and np.isfinite(price.to_numpy()).all())
  if prices is None:prices=price
  check(label+': identical prices across families',prices.equals(price))
  oldprice=read(oldroot/'shared/pricing_btn_bi.parquet');check(label+': 2023 actual prices unchanged',np.array_equal(price.electricity_pricing[:N23],oldprice.electricity_pricing))
  for k,h in enumerate((4,24,96),1):check(label+': 2023 forecasts before boundary unchanged '+str(h),np.array_equal(price[f'electricity_pricing_predicted_{k}'][:N23-h],oldprice[f'electricity_pricing_predicted_{k}'][:N23-h]))
  weather=read(root/'shared/weather.parquet');check(label+': finite weather',len(weather)==TOTAL and np.isfinite(weather.to_numpy()).all())
  check(label+': 2023 weather unchanged',weather.iloc[:N23].equals(read(oldroot/'shared/weather.parquet')))
  carbon=read(root/'shared/carbon_intensity.parquet');check(label+': carbon complete nonnegative',len(carbon)==TOTAL and np.isfinite(carbon.to_numpy()).all() and (carbon>=0).all().all())
  sessions=read(root/'catalogs/charging_sessions.parquet');check(label+': unique sessions',sessions.session_id.is_unique)
  for field in ['ev_id','charger_id']:
   ok=True
   for _,g in sessions.sort_values('arrival_time_step').groupby(field):ok &= not (g.arrival_time_step.to_numpy()[1:]<g.departure_time_step.to_numpy()[:-1]).any()
   check(label+': no double bookings '+field,ok)
  check(label+': SOC targets 50–100%',sessions.target_soc.between(.5,1).all())
  for member in members.loc[members.category!='residential'].to_dict('records'):
   op=read(root/f"operations/{member['member_id']}.parquet");opened=op.is_open.to_numpy();subset=sessions.loc[(sessions.member_id==member['member_id'])&(sessions.arrival_time_step>=N23)]
   check(label+': visits within site opening '+member['member_id'],all(opened[int(row.arrival_time_step):int(row.departure_time_step)].all() for row in subset.itertuples()))
   if member['routine_archetype']=='office':check(label+': office weekends/holidays closed '+member['member_id'],not opened[(local.year==2024)&(cal.is_rest_day.to_numpy())].any())

  normal=json.loads((root/'schema.json').read_text());chs={cid:c for b in normal['buildings'].values() for cid,c in b.get('chargers',{}).items()}
  deficits=[]
  for v in sessions.loc[(sessions.arrival_time_step>=N23)&(sessions.departure_time_step<TOTAL)].itertuples():
   a=normal['electric_vehicles_def'][v.ev_id]['battery']['attributes'];ch=chs[v.charger_id]['attributes'];cap=a['capacity'];soc=v.arrival_soc_reference;curve=np.array(a['capacity_power_curve'])
   for t in range(v.departure_time_step-v.arrival_time_step):
    power=min(ch['max_charging_power']*ch['efficiency'],a['nominal_power']*float(np.interp(soc,curve[:,0],curve[:,1])))
    soc=min(1,soc+power*.25*np.sqrt(a['efficiency'])/cap)
    if soc>=v.target_soc-1e-9:break
   if (v.target_soc-soc)*cap>1e-5:deficits.append(v.session_id)
   isolated+=1
  check(label+': isolated EV duration/power/efficiency/taper bounds',not deficits)
  for member in members.to_dict('records'):
   mid=member['member_id'];f=read(root/f'timeseries/{mid}.parquet');old=read(oldroot/f'timeseries/{mid}.parquet')
   check(label+': base input '+mid,len(f)==TOTAL and np.isfinite(f.to_numpy()).all() and (f[['solar_generation','non_shiftable_load']]>=0).all().all())
   check(label+': 2023 physical profile unchanged '+mid,np.array_equal(f.iloc[:N23].to_numpy(),old.to_numpy()))
   check(label+': inverter bound '+mid,float(f.solar_generation.max())*4<=member['pv_inverter_ac_kw']+1e-4)
  for p in (root/'evaluation').glob('*.json'):
   s=json.loads(p.read_text());old=json.loads((oldroot/'evaluation'/p.name).read_text());v=p.stem
   check(v+': original engine and horizon',s['native_compatibility']['no_runtime_modifications'] and s['simulation_end_time_step']==TOTAL-1)
   check(v+': original equipment',s['electric_vehicles_def']==old['electric_vehicles_def'])
   check(v+': no topology replay',s.get('topology_events',[])==old.get('topology_events',[]))
   check(v+': market/export configuration',s['community_market']=={'enabled':False,'grid_export_price':0.0})
   check(v+': DR only All-in',not s.get('demand_response',{}).get('enabled') or v=='premium_100_allin')
   departed={e['target_member_id'] for e in s.get('topology_events',[]) if e['operation']=='remove_member'}
   evs=read(root/'catalogs/electric_vehicles.parquet');owned=set(evs.loc[evs.home_member_id.isin(departed),'ev_id'])
   retired={e.get('target_asset_id') for e in s.get('topology_events',[]) if e['operation']=='remove_asset' and e.get('purpose')!='native_schema_template_only'}
   final_sessions={};changed2023=[]
   for mid,b in s['buildings'].items():
    f=read(root/b['energy_simulation']);orig=read(oldroot/old['buildings'][mid]['energy_simulation'])
    check(v+': input rows '+mid,len(f)==TOTAL and np.isfinite(f.to_numpy()).all())
    check(v+': 2023 native physics unchanged '+mid,np.array_equal(f.iloc[:N23].to_numpy(),orig.to_numpy()))
    if mid in departed:check(v+': member permanently absent '+mid,f.iloc[N23:][['non_shiftable_load','solar_generation','occupant_count','power_outage']].eq(0).all().all())
    check(v+': common tariff '+mid,b['pricing']=='shared/pricing_btn_bi.parquet')
    for cid,ch in b.get('chargers',{}).items():
     cf=read(root/ch['charger_simulation']);connected=cf.electric_vehicle_charger_state.eq(1);tail=cf.iloc[N23:]
     check(v+': charger rows '+cid,len(cf)==TOTAL)
     if mid in departed or cid in retired:check(v+': removed charger inactive '+cid,not tail.electric_vehicle_charger_state.isin([1,2]).any())
     check(v+': departed owned vehicles absent '+cid,not tail.electric_vehicle_id.isin(owned).any())
     use=tail.loc[tail.electric_vehicle_charger_state.eq(1)]
     for eid,g in use.groupby('electric_vehicle_id'):
      previous=final_sessions.setdefault(eid,set());new=set(g.index);check(v+': EV at one charger '+eid+' '+cid,not(previous&new));previous.update(new)
     for br in json.loads((root/'extension_2024.json').read_text())['year_end_session_bridges']:
      if cid!=br['charger_id'] or cf.iloc[N23].electric_vehicle_session_id!=br['session_id']:continue
      ok=cf.iloc[N23-1].electric_vehicle_session_id==cf.iloc[N23].electric_vehicle_session_id and cf.iloc[N23-1].electric_vehicle_required_soc_departure==cf.iloc[N23].electric_vehicle_required_soc_departure and cf.iloc[N23-1].electric_vehicle_departure_time==cf.iloc[N23].electric_vehicle_departure_time+1
      check(v+': continuous New Year parking '+cid,ok);bridges.append({'variant':v,**br})
    for aid,app in b.get('deferrable_appliances',{}).items():
     schedule=read(root/app['flexibility_schedule_file']);pro=read(root/app['cycle_profiles_file']).set_index('profile_id')
     ok=True
     for row in schedule.itertuples():
      length=len(json.loads(pro.loc[row.profile_id,'load_profile']));ok &= 0<=row.earliest_start_time_step<=row.latest_start_time_step and row.latest_start_time_step+length-1<=row.deadline_time_step<TOTAL
     check(v+': appliance windows '+aid,ok)
     if aid in retired or mid in departed:check(v+': removed appliance no 2024 requests '+aid,not schedule.earliest_start_time_step.ge(N23).any())
   for field in [s.get('robustness',{}).get('events_file'),s.get('experimental_design',{}).get('native_confirmation_robustness_file'),s.get('demand_response',{}).get('requests_file')]:
    if not field:continue
    f=read(root/field);check(v+': event bounds '+field,((f.start_time_step>=0)&(f.end_time_step<TOTAL)&(f.end_time_step>=f.start_time_step)).all())
    check(v+': event coverage in 2024 '+field,f.end_time_step.ge(N23).any())
   if s.get('demand_response',{}).get('enabled'):
    f=read(root/s['demand_response']['requests_file']);stress=f.loc[f.stratum=='impossible_stress'];check(v+': impossible stress retained both years',len(stress)==2 and stress.target_power_kw.eq(1500).all() and (stress.end_time_step-stress.start_time_step+1).eq(16).all())
   stats.append({'configuration':v,'members_catalogue':len(s['buildings']),'members_active_end':sum(b.get('include',True) for b in s['buildings'].values())+sum(e['operation']=='add_member' for e in s.get('topology_events',[]))-len(departed),'records':TOTAL})
  check(label+': checksums',all(sha(root/line.split('  ',1)[1])==line.split('  ',1)[0] for line in (root/'file_checksums.sha256').read_text().splitlines()))
 report={'version':m['version'],'scope':'static input validation only; no annual controller or usage evaluation; isolated EV energy bounds do not establish joint site/event feasibility','simulator_core':COMMIT,'annual_runs':0,'checks':checks,'failed':errors,'isolated_2024_sessions_checked':isolated,'new_year_continuations':bridges,'configurations':stats}
 dest=repo/'results/portugal_rec_native/two_year_integrity.json';dest.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'checks':len(checks),'failed':errors,'isolated_sessions':isolated,'bridges':len(bridges)},indent=2));return not errors
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--engine',type=Path,default=Path(__file__).resolve().parents[2]);a=p.parse_args();raise SystemExit(0 if verify(a.repo,a.source,a.engine) else 1)

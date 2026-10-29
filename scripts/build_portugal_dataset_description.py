"""Describe the published Portugal datasets. Reads inputs; never runs CityLearn.

Produces figure-ready tables, deterministic charts and descriptive statistics.
PDF/Markdown composition is handled by render_portugal_dataset_report.py.
"""
from pathlib import Path
import hashlib
import json
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap

ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'docs/portugal_rec_native'
OUT=DOCS/'dataset_description'
FIG=DOCS/'figures/descriptive'
ORDER=['Core-4','Core-15','Core-30','Premium-100']
KEYS=dict(zip(ORDER,['micro_4','core_15','core_30','premium_100']))
COLOR=dict(zip(ORDER,['#446778','#228b85','#ca8a31','#934561']))
INK='#203b4b';TEAL='#228b85';GOLD='#ca8a31';RED='#934561';GREY='#9daab1'
N=70176; CLOCK=pd.date_range('2023-01-01',periods=N,freq='15min',tz='UTC')
LOCAL=CLOCK.tz_convert('Europe/Lisbon'); YEARS=CLOCK.year.to_numpy()
ROUTINES={'commuter':'Daytime commuter','hybrid_worker':'Hybrid worker','night_shift':'Night-shift worker','school_runs':'School-run household','retired':'Retired / daytime presence','late_worker':'Late-working household','office':'Office','retail':'Retail shop','clinic':'Clinic','cafe':'Cafe','workshop':'Workshop','sports_centre':'Sports centre','grocery':'Grocery store','care_home':'Care home'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'axes.titlesize':12,'axes.titleweight':'semibold','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#8a999f','xtick.color':INK,'ytick.color':INK,'text.color':INK,'axes.labelcolor':INK,'axes.titlecolor':INK,'legend.frameon':False,'savefig.facecolor':'white','figure.facecolor':'white','axes.grid':False,'svg.fonttype':'none'})

def records(f):return json.loads(f.to_json(orient='records'))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
def save(name,fig):
    fig.savefig(FIG/f'{name}.png',dpi=190,bbox_inches='tight',pad_inches=.12)
    fig.savefig(FIG/f'{name}.svg',bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
def grid(ax):ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
def compact_axis(ax):
    ax.set_xlim(CLOCK[0],CLOCK[-1]);ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]));ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
def quantiles(x):return {key:float(val) for key,val in zip(['min','p10','median','p90','max'],np.quantile(x,[0,.1,.5,.9,1]))}
def barlabels(ax,bs,fmt='{:.0f}'):
    for b in bs:ax.text(b.get_x()+b.get_width()/2,b.get_height(),fmt.format(b.get_height()),ha='center',va='bottom',fontsize=8)


def extract():
    data={};summary=[];members_all=[];variants=[];profiles_all=[];cycles_all=[];sessions_all=[];event_rows=[];outages=[];timelines=[];health=[];drs=[];limits=[]
    for name,key in KEYS.items():
        root=ROOT/f'data/datasets/rec_pt_2023_{key}_v2'
        cat={x:pd.read_parquet(root/f'catalogs/{x}.parquet') for x in ['members','chargers','electric_vehicles','deferrables']}
        m=cat['members']; schema=json.loads((root/'schema.json').read_text());cat['schema']=schema;cat['root']=root
        cat['calendar']=pd.read_parquet(root/'shared/calendar.parquet')
        member_ids=m.member_id.tolist();cat['load']=np.column_stack([pd.read_parquet(root/schema['buildings'][mid]['energy_simulation']).non_shiftable_load.to_numpy() for mid in member_ids])
        cat['pv']=np.column_stack([pd.read_parquet(root/schema['buildings'][mid]['energy_simulation']).solar_generation.to_numpy() for mid in member_ids])
        cat['end_uses']=sum(pd.read_parquet(root/f'end_uses/{mid}.parquet') for mid in member_ids)
        native_sessions=[];apps=[];profiles=[]
        for mid,b in schema['buildings'].items():
            for cid,ch in b.get('chargers',{}).items():
                f=pd.read_parquet(root/ch['charger_simulation'])
                on=f.loc[f.electric_vehicle_charger_state.eq(1)].copy();on['start']=on.index
                for sid,g in on.groupby('electric_vehicle_session_id',sort=False):
                    row=g.iloc[0];a=int(row.start);dep=int(a+row.electric_vehicle_departure_time);eid=row.electric_vehicle_id
                    capacity=schema['electric_vehicles_def'][eid]['battery']['attributes']['capacity']
                    target=float(row.electric_vehicle_required_soc_departure);soc=float(row.electric_vehicle_current_soc)
                    native_sessions.append(dict(family=name,session_id=sid,member_id=mid,charger_id=cid,ev_id=eid,arrival=a,departure=dep,year=int(YEARS[a]),duration_h=(dep-a)/4,
                        arrival_hour=float(LOCAL[a].hour+LOCAL[a].minute/60),departure_hour=float((CLOCK[0]+pd.Timedelta(minutes=15*dep)).tz_convert('Europe/Lisbon').hour+((dep%4)*.25)),
                        target_soc=target,arrival_soc=soc,gap_kwh=max(0,(target-soc)*capacity),left_censored=a==0,right_censored=dep>=N))
            for aid,app in b.get('deferrable_appliances',{}).items():
                f=pd.read_parquet(root/app['flexibility_schedule_file']).copy();p=pd.read_parquet(root/app['cycle_profiles_file']).copy()
                f['member_id']=mid;f['asset_id']=aid;f['family']=name;f['year']=YEARS[f.earliest_start_time_step.to_numpy(int)]
                f=f.merge(p[['profile_id','total_energy_kwh','duration_steps']],on='profile_id',validate='many_to_one')
                f['start_slack_h']=(f.latest_start_time_step-f.earliest_start_time_step)/4;f['duration_h']=f.duration_steps/4
                f['category']=m.set_index('member_id').loc[mid,'category'];f['appliance_type']=cat['deferrables'].set_index('deferrable_id').loc[aid,'appliance_type']
                apps.append(f);p['asset_id']=aid;p['family']=name;p['member_id']=mid;p['appliance_type']=f.appliance_type.iloc[0];profiles.append(p)
            es=b['electrical_service'];limits.append(dict(family=name,member_id=mid,connection=es['connection_type'],contract_kva=es['contracted_power_kva'],import_kw=es['limits']['total']['import_kw'],export_kw=es['limits']['total']['export_kw'],phase=b['electrical_service']['default_split']))
        ss=pd.DataFrame(native_sessions).merge(m[['member_id','category','routine_archetype']],on='member_id',validate='many_to_one')
        ss['site_group']=np.where(ss.category=='residential','Residential','Services')
        ss['day_kind']=cat['calendar'].day_kind.to_numpy()[ss.arrival.to_numpy(int)]
        aa=pd.concat(apps,ignore_index=True);pp=pd.concat(profiles,ignore_index=True)
        cat['sessions']=ss;cat['cycles']=aa;cat['profiles']=pp
        sessions_all.append(ss);cycles_all.append(aa);profiles_all.append(pp)
        totals=[]
        for year in (2023,2024):
            sel=YEARS==year
            for i,mid in enumerate(member_ids):totals.append(dict(family=name,member_id=mid,year=year,load_kwh=float(cat['load'][sel,i].sum()),pv_kwh=float(cat['pv'][sel,i].sum())))
        cat['totals']=pd.DataFrame(totals).merge(m[['member_id','category','routine_archetype']],on='member_id')
        m2=m.copy();m2['family']=name;members_all.append(m2)
        summary.append(dict(family=name,members=len(m),residential=int((m.category=='residential').sum()),services=int((m.category!='residential').sum()),residents=int(m.occupants.sum()),pv_sites=int(m.has_pv.sum()),pv_modules=int(m.pv_module_count.sum()),pv_kwp=float(m.pv_capacity_kw.sum()),battery_sites=int(m.has_battery.sum()),battery_kwh=float(m.battery_capacity_kwh.sum()),battery_kw=float(m.battery_nominal_power_kw.sum()),charger_sites=int(m.has_charging.sum()),chargers=len(cat['chargers']),evs=len(cat['electric_vehicles']),resident_evs=int(cat['electric_vehicles'].vehicle_pool.eq('resident').sum()),visitor_evs=int(cat['electric_vehicles'].vehicle_pool.eq('external_visitors').sum()),controlled_apps=int(cat['deferrables'].controlled.sum()),fixed_apps=int((~cat['deferrables'].controlled).sum()),flexible_sites=int(m.has_deferrable.sum()),pure_consumers=int(m.pure_consumer.sum()),sessions=len(ss),complete_sessions=int((~ss.left_censored&~ss.right_censored).sum()),cycles=len(aa),load_2023_mwh=float(cat['load'][YEARS==2023].sum()/1000),load_2024_mwh=float(cat['load'][YEARS==2024].sum()/1000),pv_2023_mwh=float(cat['pv'][YEARS==2023].sum()/1000),pv_2024_mwh=float(cat['pv'][YEARS==2024].sum()/1000),cycles_2023_kwh=float(aa.loc[aa.year==2023,'total_energy_kwh'].sum()),cycles_2024_kwh=float(aa.loc[aa.year==2024,'total_energy_kwh'].sum())))
        for sp in sorted((root/'evaluation').glob('*.json')):
            s=json.loads(sp.read_text());v=sp.stem;events=[e for e in s.get('topology_events',[]) if e.get('purpose')!='native_schema_template_only']
            for e in events:event_rows.append({**e,'variant':v,'family':name,'timestamp_local':str(LOCAL[e['time_step']])})
            for phase,path in [('development',s.get('robustness',{}).get('events_file')),('confirmation',s['experimental_design'].get('native_confirmation_robustness_file'))]:
                if path:
                    f=pd.read_parquet(root/path).copy();f['variant']=v;f['phase']=phase
                    inferred=np.where(f['mode'].isin(['noise','clip']),'measurement_noise','health')
                    if 'category' not in f:f['category']=inferred
                    else:f['category']=f['category'].fillna(pd.Series(inferred,index=f.index))
                    health.append(f)
            if s.get('demand_response',{}).get('enabled'):
                f=pd.read_parquet(root/s['demand_response']['requests_file']).copy();f['variant']=v;drs.append(f)
            outage_members=set();outage_slots=set()
            for mid,b in s['buildings'].items():
                f=pd.read_parquet(root/b['energy_simulation'],columns=['power_outage']);idx=np.flatnonzero(f.power_outage.to_numpy()>0)
                if len(idx):outage_members.add(mid);outage_slots.update(idx)
                chunks=np.split(idx,np.flatnonzero(np.diff(idx)>1)+1)
                for chunk in chunks:
                    if len(chunk):outages.append(dict(variant=v,member_id=mid,start=int(chunk[0]),end=int(chunk[-1]),duration_h=len(chunk)/4))
            # Physical asset presence from schema state and chronological events.
            active={mid:b.get('include',True) for mid,b in s['buildings'].items()}
            assets={mid:{'pv':int(b.get('pv',{}).get('attributes',{}).get('nominal_power',0)>0),'battery':int(b.get('electrical_storage',{}).get('attributes',{}).get('capacity',0)>0),'charger':set(cid for cid in b.get('chargers',{}) if not cid.startswith('NATIVE_TEMPLATE__')),'appliance':set(b.get('deferrable_appliances',{}))} for mid,b in s['buildings'].items()}
            def snapshot(t):return dict(variant=v,family=name,time_step=t,members=sum(active.values()),pv=sum(x['pv'] for mid,x in assets.items() if active[mid]),batteries=sum(x['battery'] for mid,x in assets.items() if active[mid]),chargers=sum(len(x['charger']) for mid,x in assets.items() if active[mid]),appliances=sum(len(x['appliance']) for mid,x in assets.items() if active[mid]))
            first=snapshot(0);timelines.append(first)
            for e in sorted(events,key=lambda e:e['time_step']):
                mid=e['target_member_id'];op=e['operation'];kind=e.get('target_asset_type');aid=e.get('target_asset_id')
                if op in ('add_member','remove_member'):active[mid]=op=='add_member'
                elif kind in ('pv','electrical_storage'):assets[mid]['pv' if kind=='pv' else 'battery']=int(op=='add_asset')
                elif kind in ('charger','deferrable_appliance'):
                    collection=assets[mid]['charger' if kind=='charger' else 'appliance']
                    if op=='add_asset':collection.add(aid)
                    else:collection.discard(aid)
                timelines.append(snapshot(e['time_step']))
            last=snapshot(N-1);timelines.append(last)
            label=s['experimental_design']['label'].replace('Combinado','Combined').replace('Topologia','Topology')
            variants.append(dict(variant=v,family=name,label=label,source_label=s['experimental_design']['label'],**s['experimental_design']['factors'],initial_members=first['members'],final_members=last['members'],admissions=sum(e['operation']=='add_member' for e in events),departures=sum(e['operation']=='remove_member' for e in events),events=len(events),outage_sites=len(outage_members),outage_hours=len(outage_slots)/4,initial_assets={k:first[k] for k in ['pv','batteries','chargers','appliances']},final_assets={k:last[k] for k in ['pv','batteries','chargers','appliances']}))
        data[name]=cat
        print('Described',name,'sessions',len(ss),'cycles',len(aa),flush=True)
    tables={'inventory':pd.DataFrame(summary),'members':pd.concat(members_all,ignore_index=True),'sessions':pd.concat(sessions_all,ignore_index=True),'cycles':pd.concat(cycles_all,ignore_index=True),'cycle_profiles':pd.concat(profiles_all,ignore_index=True),'variants':pd.DataFrame(variants),'topology_events':pd.DataFrame(event_rows),'topology_timeline':pd.DataFrame(timelines),'health':pd.concat(health,ignore_index=True),'outages':pd.DataFrame(outages),'dr':pd.concat(drs,ignore_index=True),'electrical_limits':pd.DataFrame(limits),'annual_member_energy':pd.concat([d['totals'] for d in data.values()],ignore_index=True)}
    return data,tables


def charts(data,t):
    inv=t['inventory'].set_index('family').loc[ORDER];premium=data['Premium-100'];m=t['members'];y=YEARS==2024
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.3),layout='constrained')
    x=np.arange(4);axs[0].bar(x,inv.residential,color=TEAL,label='Residential');axs[0].bar(x,inv.services,bottom=inv.residential,color=GOLD,label='Services');axs[0].set(xticks=x,xticklabels=ORDER,ylabel='Members',title='Four community sizes');axs[0].legend()
    for i,name in enumerate(ORDER):
        axs[1].scatter(inv.loc[name,'pv_kwp'],inv.loc[name,'battery_kwh'],s=40+inv.loc[name,'chargers']*7,color=COLOR[name])
        axs[1].annotate(name,(inv.loc[name,'pv_kwp'],inv.loc[name,'battery_kwh']),xytext=([(4,-15),(4,15),(6,4),(6,4)][i]),textcoords='offset points',fontsize=9)
    axs[1].set(xlabel='PV capacity (kWp)',ylabel='Stationary storage (kWh)',title='Scale of the installed assets');axs[1].margins(.2);save('01_overview',fig)
    # Calendar cover graphic: mean Premium PV power by local hour and day.
    cal=premium['calendar'];p=pd.DataFrame({'date':LOCAL[y].strftime('%Y-%m-%d'),'hour':LOCAL[y].hour+LOCAL[y].minute/60,'pv':premium['pv'][y].sum(axis=1)*4})
    heat=p.pivot_table(index='hour',columns='date',values='pv',aggfunc='mean')
    fig,ax=plt.subplots(figsize=(10,3.1),layout='constrained');im=ax.imshow(heat.to_numpy(),aspect='auto',origin='lower',extent=[0,366,0,24],cmap='YlOrBr');ax.set(yticks=[0,6,12,18,24],ylabel='Local hour',xticks=[0,91,182,274],xticklabels=['Jan','Apr','Jul','Oct']);fig.colorbar(im,ax=ax,label='Premium PV power (kW)',shrink=.8);save('00_cover',fig)
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.3),layout='constrained')
    for i,name in enumerate(ORDER):
        mm=data[name]['members'];res=mm[mm.category=='residential'];vals=res.occupants.value_counts().reindex(range(1,6),fill_value=0)/len(res)*100
        axs[0].bar(np.arange(5)+i*.19,vals,width=.18,label=name,color=COLOR[name])
    axs[0].set(xticks=np.arange(5)+.28,xticklabels=range(1,6),xlabel='Residents per household',ylabel='Share of households (%)');axs[0].legend(fontsize=8)
    vals=[data[n]['members'].query("category=='residential'").floor_area_m2 for n in ORDER]
    b=axs[1].boxplot(vals,labels=ORDER,patch_artist=True,showfliers=False,widths=.5)
    for box,c in zip(b['boxes'],COLOR.values()):box.set(facecolor=c,alpha=.75)
    axs[1].set(ylabel='Residential floor area (m²)');save('02_households',fig)
    # Routines and operating calendar of non-residential Premium members.
    fig,axs=plt.subplots(1,2,figsize=(9.6,4.4),layout='constrained');mm=premium['members'];res=mm[mm.category=='residential'];counts=res.routine_archetype.value_counts().reindex(list(ROUTINES)[:6]);axs[0].barh([ROUTINES[k] for k in counts.index],counts.values,color=TEAL);axs[0].invert_yaxis();axs[0].set(xlabel='Premium households',title='Residential routines')
    service=mm[mm.category!='residential'];matrix=[]
    for routine in service.routine_archetype.unique():
        mids=service.loc[service.routine_archetype==routine,'member_id'];daily=[]
        for mid in mids:
            opened=pd.read_parquet(premium['root']/f'operations/{mid}.parquet').is_open.to_numpy();daily.append([opened[y&(LOCAL.dayofweek==day)].mean()*24 for day in range(7)])
        matrix.append(np.mean(daily,axis=0))
    im=axs[1].imshow(matrix,aspect='auto',cmap='YlGnBu',vmin=0,vmax=24);axs[1].set(xticks=range(7),xticklabels=['M','T','W','T','F','S','S'],yticks=range(len(matrix)),yticklabels=[ROUTINES[k] for k in service.routine_archetype.unique()],title='Mean open hours per civil day');fig.colorbar(im,ax=axs[1],label='hours/day',shrink=.7);save('03_routines',fig)
    fig,axs=plt.subplots(1,2,figsize=(9.6,4.6),layout='constrained');own=['has_pv','has_battery','has_charging','has_deferrable'];mat=premium['members'][own].to_numpy(int);axs[0].imshow(mat,aspect='auto',cmap=ListedColormap(['#edf1f3',TEAL]),interpolation='nearest');axs[0].set(xticks=range(4),xticklabels=['PV','Battery','EVSE','Flexible\ncycles'],yticks=[0,19,39,59,79,99],yticklabels=['001','020','040','060','080','100'],ylabel='Premium member ID',title='Assets overlap within a site');axs[0].axhline(79.5,color=GOLD,lw=1.5)
    for i,name in enumerate(ORDER):
        vals=data[name]['members'][own+['pure_consumer']].mean()*100;axs[1].barh(np.arange(5)+i*.18,vals,height=.17,color=COLOR[name],label=name)
    axs[1].set(yticks=np.arange(5)+.27,yticklabels=['PV','Stationary battery','EV charging','Flexible cycles','Consumption only'],xlabel='Share of members (%)',xlim=(0,100));axs[1].legend(fontsize=8,loc='lower right');save('04_ownership',fig)
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.5),layout='constrained');endcols=['base_excluding_cycles_kwh','dhw_proxy_kwh','heating_proxy_kwh','cooling_proxy_kwh','fixed_appliances_kwh','battery_standby_kwh'];endlabels=['Base','Hot water','Heating','Cooling','Fixed cycles','Standby'];endcolors=[INK,TEAL,GOLD,RED,'#638eb0',GREY];bottom=np.zeros(4)
    for col,label,color in zip(endcols,endlabels,endcolors):
        values=np.array([data[n]['end_uses'].loc[y,col].sum()/1000 for n in ORDER]);axs[0].bar(ORDER,values,bottom=bottom,color=color,label=label);bottom+=values
    axs[0].set(ylabel='Non-shiftable demand (MWh)',title='2024 exogenous demand');axs[0].legend(fontsize=7,ncol=2)
    for i,name in enumerate(ORDER):
        a=data[name]['totals'].query('year==2024');axs[1].scatter(a.load_kwh/1000,a.pv_kwh/1000,s=16,alpha=.55,color=COLOR[name],label=name)
    axs[1].set(xlabel='Annual non-shiftable load (MWh/site)',ylabel='Annual PV generation (MWh/site)',title='Members have different energy roles');axs[1].legend(fontsize=8);save('05_energy',fig)
    fig,axs=plt.subplots(2,2,figsize=(9.6,4.8),layout='constrained')
    for ax,name in zip(axs.flat,ORDER):
        series=pd.Series(data[name]['load'].sum(axis=1),index=CLOCK)
        for year,ls in [(2023,'--'),(2024,'-')]:
            days=366 if year==2024 else 365;s=series.loc[str(year)].resample('MS').sum();avg=s.to_numpy()/s.index.days_in_month/len(data[name]['members']);ax.plot(range(1,13),avg,ls,color=COLOR[name],label=str(year),lw=2)
        ax.set(title=name,ylabel='kWh/member/day',xticks=[1,4,7,10,12]);grid(ax);ax.legend(fontsize=8)
    save('06_seasonal_load',fig)
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.6),layout='constrained')
    for ax,category in zip(axs,['residential','services']):
        mask=premium['members'].category.eq('residential').to_numpy();mask=mask if category=='residential' else ~mask
        power=premium['load'][:,mask].mean(axis=1)*4
        holiday=cal.is_national_holiday.to_numpy()|cal.is_municipal_holiday.to_numpy();weekend=cal.is_weekend.to_numpy()
        for label,sel,color in [('Weekday',~holiday&~weekend,TEAL),('Weekend',~holiday&weekend,GOLD),('Public holiday',holiday,RED)]:
            f=pd.DataFrame({'h':LOCAL[y&sel].hour+LOCAL[y&sel].minute/60,'kw':power[y&sel]});q=f.groupby('h').kw.mean();ax.plot(q.index,q.values,label=label,color=color,lw=2)
        ax.set(title='Premium '+('households' if category=='residential' else 'services'),xlabel='Local hour',ylabel='Mean non-shiftable power/site (kW)',xticks=[0,6,12,18,24]);ax.legend(fontsize=8);grid(ax)
    save('07_daily_load',fig)
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.5),layout='constrained')
    for name in ORDER:
        vals=np.sort(data[name]['load'][y].sum(axis=0))[::-1];axs[0].plot(np.r_[0,np.arange(1,len(vals)+1)/len(vals)*100],np.r_[0,np.cumsum(vals)/sum(vals)*100],label=name,color=COLOR[name],lw=2)
    axs[0].plot([0,100],[0,100],':',color=GREY);axs[0].set(xlabel='Members, ranked by demand (%)',ylabel='Cumulative non-shiftable demand (%)');axs[0].legend(fontsize=8);grid(axs[0])
    a=premium['totals'].query('year==2024').nlargest(8,'load_kwh').iloc[::-1];axs[1].barh([x.member_id[-3:]+' '+ROUTINES[x.routine_archetype] for x in a.itertuples()],a.load_kwh/1000,color=GOLD);axs[1].set(xlabel='2024 non-shiftable demand (MWh)',title='Largest Premium demand sites');save('08_concentration',fig)
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.6),layout='constrained')
    for i,name in enumerate(ORDER):
        p=data[name]['members'].query('has_pv');axs[0].scatter(np.full(len(p),i)+np.linspace(-.12,.12,len(p)),p.pv_capacity_kw,s=28,color=COLOR[name],alpha=.8)
    axs[0].set(xticks=range(4),xticklabels=ORDER,ylabel='Array nameplate capacity (kWp)',title='Installed PV arrays')
    p=premium['members'].query('has_pv');axs[1].scatter(p.pv_capacity_kw,p.pv_inverter_ac_kw,color=TEAL,s=40,alpha=.85);axs[1].plot([0,p.pv_capacity_kw.max()],[0,p.pv_capacity_kw.max()],':',color=GREY);axs[1].set(xlabel='DC array capacity (kWp)',ylabel='AC inverter rating (kW)',title='Premium array and inverter ratings');save('09_pv_fleet',fig)
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.5),layout='constrained')
    for year,ls in [(2023,'--'),(2024,'-')]:
        s=pd.Series(premium['pv'].sum(axis=1),index=CLOCK).loc[str(year)].resample('MS').sum()/premium['members'].pv_capacity_kw.sum();axs[0].plot(range(1,13),s,ls,color=GOLD,lw=2,label=str(year))
    axs[0].set(xticks=range(1,13),xlabel='Month',ylabel='PV yield (kWh/kWp/month)');axs[0].legend();grid(axs[0])
    for month,col,label in [(1,INK,'January'),(7,GOLD,'July')]:
        sel=y&(LOCAL.month==month);f=pd.DataFrame({'hour':LOCAL[sel].hour+LOCAL[sel].minute/60,'PV':premium['pv'][sel].sum(axis=1)*4,'Load':premium['load'][sel].sum(axis=1)*4});q=f.groupby('hour').mean()
        axs[1].plot(q.index,q.PV,color=col,label=label+' PV');axs[1].plot(q.index,q.Load,'--',color=col,label=label+' load')
    axs[1].set(xlabel='Local hour',ylabel='Community power (kW)',xticks=[0,6,12,18,24]);axs[1].legend(fontsize=8,ncol=2);grid(axs[1]);save('10_pv_seasons',fig)
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.4),layout='constrained')
    for i,name in enumerate(ORDER):
        b=data[name]['members'].query('has_battery');axs[0].scatter(np.full(len(b),i)+np.linspace(-.15,.15,len(b)),b.battery_capacity_kwh,color=COLOR[name],s=32)
        axs[1].scatter(b.battery_capacity_kwh,b.battery_nominal_power_kw,color=COLOR[name],label=name,s=45,alpha=.7)
    axs[0].set(xticks=range(4),xticklabels=ORDER,ylabel='Stationary battery capacity (kWh)');axs[1].set(xlabel='Energy capacity (kWh)',ylabel='Rated power (kW)');axs[1].legend(fontsize=8);save('11_batteries',fig)
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.5),layout='constrained');lim=t['electrical_limits']
    for i,name in enumerate(ORDER):
        g=lim[lim.family==name];a=(g.connection=='single_phase').mean()*100;axs[0].bar(i,a,color=TEAL);axs[0].bar(i,100-a,bottom=a,color=GOLD)
    axs[0].set(xticks=range(4),xticklabels=ORDER,ylabel='Share of sites (%)');axs[0].legend(['Single phase','Three phase'],fontsize=8)
    normal=data['Core-30']['schema'];safe=json.loads((data['Core-30']['root']/'evaluation/core_30_safety.json').read_text());a=np.array([b['electrical_service']['limits']['total']['import_kw'] for b in normal['buildings'].values()]);b=np.array([b['electrical_service']['limits']['total']['import_kw'] for b in safe['buildings'].values()]);axs[1].scatter(a,b,c=np.where(a==b,INK,RED),s=35);axs[1].plot([0,max(a)],[0,max(a)],':',color=GREY);axs[1].set(xlabel='Core-30 Normal import limit (kW)',ylabel='Safety import limit (kW)',title='Safety variant connection settings');save('12_connections',fig)
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.4),layout='constrained')
    for i,name in enumerate(ORDER):
        e=data[name]['electric_vehicles'];vals=np.sort(e.battery_capacity_kwh);axs[0].step(vals,np.arange(1,len(vals)+1)/len(vals)*100,where='post',color=COLOR[name],label=name,lw=2)
    axs[0].set(xlabel='EV usable battery capacity (kWh)',ylabel='Cumulative share of vehicles (%)');axs[0].legend(fontsize=8);grid(axs[0])
    x=np.arange(4);axs[1].bar(x,inv.resident_evs,color=TEAL,label='Resident vehicles');axs[1].bar(x,inv.visitor_evs,bottom=inv.resident_evs,color=GOLD,label='External visitor pool');axs[1].set(xticks=x,xticklabels=ORDER,ylabel='Distinct vehicle identities');axs[1].legend(fontsize=8);save('13_ev_fleet',fig)
    fig,axs=plt.subplots(1,2,figsize=(9.6,4.0),layout='constrained');c=premium['chargers'];counts=premium['members'].query('has_charging').nlargest(10,'physical_charger_count').iloc[::-1]
    axs[0].barh([x.member_id[-3:]+' '+ROUTINES[x.routine_archetype] for x in counts.itertuples()],counts.physical_charger_count,color=TEAL);axs[0].set(xlabel='Physical charging connectors',title='Premium multi-connector sites')
    grouped=c.groupby(['max_charging_power_kw','phase_connection']).size().unstack(fill_value=0);x=np.arange(len(grouped));bottom=np.zeros(len(grouped))
    for ph,color in zip(grouped.columns,[TEAL,GOLD,RED,INK]):
        vals=grouped[ph].to_numpy();axs[1].bar(x,vals,bottom=bottom,color=color,label=ph);bottom+=vals
    axs[1].set(xticks=x,xticklabels=[f'{v:g}' for v in grouped.index],xlabel='Maximum charging power (kW)',ylabel='Connectors',title='Power and phase allocation');axs[1].legend(fontsize=8);save('14_chargers',fig)
    sessions=t['sessions'];s=sessions.query("family=='Premium-100' and year==2024 and not left_censored and not right_censored")
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.5),layout='constrained')
    for ax,group in zip(axs,['Residential','Services']):
        g=s[s.site_group==group]
        for col,label,color in [('arrival_hour','Arrivals',TEAL),('departure_hour','Departures',GOLD)]:
            h,_=np.histogram(g[col],bins=np.arange(25));ax.step(np.arange(24)+.5,h/len(g)*100,where='mid',label=label,color=color,lw=2)
        ax.set(title=f'{group} ({len(g):,} complete sessions)',xlabel='Local hour',ylabel='Share of sessions (%)',xticks=[0,6,12,18,24]);ax.legend(fontsize=8);grid(ax)
    save('15_ev_times',fig)
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.4),layout='constrained')
    for j,(group,color) in enumerate([('Residential',TEAL),('Services',GOLD)]):
        g=s[s.site_group==group];hist,_=np.histogram(g.target_soc*100,bins=[50-1e-7,60,70,80,90,99.999,100.001]);axs[0].bar(np.arange(6)+(j-.5)*.38,hist/len(g)*100,width=.36,label=group,color=color)
        vals=np.sort(g.duration_h);axs[1].plot(vals,np.arange(1,len(vals)+1)/len(vals)*100,color=color,label=group,lw=2)
    axs[0].set(xlabel='Requested SOC at departure (%)',ylabel='Share of sessions (%)',xticks=range(6),xticklabels=['50-59','60-69','70-79','80-89','90-<100','100']);axs[0].tick_params(axis='x',labelsize=8);axs[0].legend(fontsize=8)
    axs[1].set(xlabel='Connection duration (hours)',ylabel='Cumulative share of sessions (%)',xlim=(0,36));axs[1].legend(fontsize=8);grid(axs[1]);save('16_soc_dwell',fig)
    # Scheduled connector occupancy: an input property, not delivered charging.
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.5),layout='constrained');root=premium['root'];sch=premium['schema']
    for ax,group in zip(axs,['Residential','Services']):
        mids=set(premium['members'].loc[premium['members'].category.eq('residential') if group=='Residential' else premium['members'].category.ne('residential'),'member_id']);arrays=[pd.read_parquet(root/c['charger_simulation']).electric_vehicle_charger_state.eq(1).to_numpy() for mid,b in sch['buildings'].items() if mid in mids for c in b.get('chargers',{}).values()];occ=np.stack(arrays).mean(axis=0)*100
        f=pd.DataFrame({'weekday':LOCAL[y].dayofweek,'hour':LOCAL[y].hour,'occupancy':occ[y]});a=f.pivot_table(index='weekday',columns='hour',values='occupancy');im=ax.imshow(a,aspect='auto',vmin=0,vmax=100,cmap='YlGnBu');ax.set(title=group,xticks=[0,6,12,18,23],yticks=range(7),yticklabels=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],xlabel='Local hour');fig.colorbar(im,ax=ax,label='Connected connectors (%)',shrink=.8)
    save('17_connection_calendar',fig)
    fig,ax=plt.subplots(figsize=(9.6,3.8),layout='constrained');ev=premium['electric_vehicles'];linked=ev[(ev.work_member_id!='')&(ev.home_member_id!='')];homes=linked.home_member_id.unique();works=linked.work_member_id.unique();hy={v:i for i,v in enumerate(homes)};wy={v:i*(len(homes)-1)/max(1,len(works)-1) for i,v in enumerate(works)}
    for e in linked.itertuples():ax.plot([0,1],[hy[e.home_member_id],wy[e.work_member_id]],color=TEAL,alpha=.45,lw=1.5)
    for mid,ypos in hy.items():ax.scatter(0,ypos,c=TEAL,s=30);ax.text(-.025,ypos,mid[-3:],ha='right',va='center',fontsize=8)
    for mid,ypos in wy.items():ax.scatter(1,ypos,c=GOLD,s=90);ax.text(1.025,ypos,mid[-3:]+' '+ROUTINES[premium['members'].set_index('member_id').loc[mid,'routine_archetype']],va='center',fontsize=9)
    ax.set(xlim=(-.13,1.35),ylim=(-1,len(homes)),xticks=[0,1],xticklabels=['Home member','Work destination'],yticks=[],title=f'{len(linked)} resident vehicle identities with a home-work link');ax.spines[['left','bottom']].set_visible(False);save('18_mobility_links',fig)
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.4),layout='constrained');profiles=premium['profiles']
    for kind,col in [('washing_machine',TEAL),('dishwasher',GOLD)]:
        row=profiles[profiles.appliance_type==kind].iloc[0];arr=np.array(json.loads(row.load_profile))*4;axs[0].step(np.arange(len(arr)+1)*.25,np.r_[arr,arr[-1]],where='post',label=kind.replace('_',' ').title(),color=col,lw=2)
        g=profiles[profiles.appliance_type==kind];axs[1].scatter(g.duration_steps*.25,g.total_energy_kwh,s=13,color=col,alpha=.6,label=kind.replace('_',' ').title())
    axs[0].set(xlabel='Elapsed cycle time (h)',ylabel='Average power per interval (kW)',title='Example synthetic programmes');axs[0].legend(fontsize=8)
    axs[1].set(xlabel='Programme duration (h)',ylabel='Cycle energy (kWh)',title='All controlled Premium programmes');axs[1].legend(fontsize=8);save('19_appliances',fig)
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.4),layout='constrained');cy=t['cycles'];groups=[cy.loc[(cy.family==name)&(cy.year==2024),'start_slack_h'] for name in ORDER];bp=axs[0].boxplot(groups,labels=ORDER,patch_artist=True,showfliers=False)
    for b,col in zip(bp['boxes'],COLOR.values()):b.set(facecolor=col,alpha=.75)
    axs[0].set(ylabel='Allowed start-time slack (h)',title='Controlled requests in 2024')
    g=cy[(cy.family=='Premium-100')&(cy.year==2024)];categories=['weekday','weekend','holiday'];vals=[g.loc[g.day_kind==k,'start_slack_h'] for k in categories];b=axs[1].boxplot(vals,labels=['Weekday','Weekend','Holiday'],patch_artist=True,showfliers=False)
    for box,col in zip(b['boxes'],[TEAL,GOLD,RED]):box.set(facecolor=col,alpha=.75)
    axs[1].set(ylabel='Allowed start-time slack (h)',title='Premium calendar-dependent windows');save('20_flexibility',fig)
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.4),layout='constrained');timeline=t['topology_timeline']
    for variant,color,label in [('core_30_topology',COLOR['Core-30'],'Core-30 Topology'),('premium_100_allin',COLOR['Premium-100'],'Premium All-in')]:
        g=timeline[timeline.variant==variant];axs[0].step(CLOCK[g.time_step],g.members,where='post',color=color,lw=2,label=label)
    axs[0].set(ylabel='Active members');compact_axis(axs[0]);axs[0].legend(fontsize=8);grid(axs[0]);g=timeline[timeline.variant=='premium_100_allin']
    for kind,col,label in [('pv',GOLD,'PV sites'),('batteries',TEAL,'Batteries'),('chargers',INK,'Connectors'),('appliances',RED,'Flexible appliances')]:axs[1].step(CLOCK[g.time_step],g[kind],where='post',color=col,lw=1.7,label=label)
    axs[1].set(ylabel='Assets at active members',title='Premium All-in');compact_axis(axs[1]);axs[1].legend(fontsize=8,ncol=2);grid(axs[1]);save('21_topology',fig)
    fig,axs=plt.subplots(2,1,figsize=(9.6,3.9),layout='constrained')
    events=t['topology_events']
    for ax,variant,label in zip(axs,['core_30_topology','premium_100_allin'],['Core-30 Topology / Combined','Premium All-in']):
        g=events[(events.variant==variant)&(events.target_member_id=='Member_001')]
        for i,row in enumerate(g.itertuples()):
            x=.03+i*.34;desc={'remove_asset':'Battery retired','add_asset':'Replacement installed','remove_member':'Member leaves permanently'}[row.operation]
            desc+='\n'+(str(row.target_asset_id) if isinstance(row.target_asset_id,str) else 'All assets leave with the member')
            ax.text(x,.48,desc+'\n'+LOCAL[row.time_step].strftime('%d %b %Y, %H:%M'),transform=ax.transAxes,fontsize=10,va='center',linespacing=1.65)
            if i<2:ax.annotate('',xy=(x+.32,.48),xytext=(x+.27,.48),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':GREY})
        ax.set_title(label,loc='left');ax.axis('off')
    save('22_member_history',fig)
    fig,axs=plt.subplots(2,1,figsize=(9.6,4.7),layout='constrained');he=t['health']
    for ax,variant,label in zip(axs,['core_30_robustness','premium_100_allin'],['Core-30 Robustness','Premium All-in']):
        g=he[(he.variant==variant)&(he.phase=='development')];keys=[('measurement_noise','observation','noise','Measurement noise'),('health','observation','missing','Missing observations'),('health','forecast','bias','Price forecast bias'),('health','action','delay','Action delay'),('health','action','dropout','Action dropout'),('health','asset','unavailable','Asset unavailable')]
        for i,(category,module,mode,lab) in enumerate(keys):
            rows=g[(g.category==category)&(g.module==module)&(g['mode']==mode)][['start_time_step','end_time_step']].drop_duplicates()
            for a,b in rows.itertuples(index=False,name=None):
                ax.broken_barh([(mdates.date2num(CLOCK[a]),(b-a+1)/96)],(i-.28,.56),facecolors=TEAL if category=='measurement_noise' else RED)
                if b-a<96:ax.scatter(CLOCK[a],i,s=14,color=RED,zorder=3)
        ax.set(yticks=range(len(keys)),yticklabels=[x[3] for x in keys],title=label,xlim=(CLOCK[0],CLOCK[-1]));ax.invert_yaxis();compact_axis(ax)
    save('23_health_calendar',fig)
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.6),layout='constrained');o=t['outages'];g=o[o.variant=='premium_100_allin'];mids=g.member_id.unique();mapping={mid:i for i,mid in enumerate(mids)}
    for row in g.itertuples():axs[0].scatter(CLOCK[row.start],mapping[row.member_id],s=20+row.duration_h*18,color=RED,alpha=.7)
    axs[0].set(yticks=range(len(mids)),yticklabels=[x[-3:] for x in mids],ylabel='Affected member ID',title='Premium All-in outage episodes');compact_axis(axs[0])
    variants=['core_30_robustness','core_30_combined','premium_100_allin'];hours=[o.loc[o.variant==v,'duration_h'].sum() for v in variants];bs=axs[1].bar(['Core-30\nRobustness','Core-30\nCombined','Premium\nAll-in'],hours,color=[GOLD,TEAL,RED]);barlabels(axs[1],bs);axs[1].set(ylabel='Scheduled site-hours of outage',title='Input exposure across both years',ylim=(0,max(hours)*1.15));grid(axs[1]);save('24_outages',fig)
    fig,ax=plt.subplots(figsize=(9.6,3.1),layout='constrained');dr=t['dr'];regular=dr[dr.stratum=='service_candidate']
    for row in regular.itertuples():
        val=row.target_power_kw*(1 if row.direction=='up' else -1);ax.vlines(CLOCK[row.start_time_step],0,val,color=TEAL if val>0 else GOLD,lw=3);ax.scatter(CLOCK[row.start_time_step],val,color=TEAL if val>0 else GOLD,s=38)
    ax.axhline(0,color=GREY,lw=.7);ax.set(ylabel='Requested change in demand (kW)',title='Operational-candidate DR requests: 1 hour each');compact_axis(ax);grid(ax);save('25_dr',fig)
    shared=premium['root']/'shared';weather=pd.read_parquet(shared/'weather.parquet');price=pd.read_parquet(shared/'pricing_omie.parquet');carbon=pd.read_parquet(shared/'carbon_intensity.parquet')
    fig,axs=plt.subplots(2,1,figsize=(9.6,4.8),layout='constrained')
    for ax,arr,label,color in [(axs[0],price.electricity_pricing.to_numpy()*100,'Grid purchase price (cents/kWh)',TEAL),(axs[1],carbon.carbon_intensity.to_numpy()*1000,'Grid carbon intensity (gCO2e/kWh)',INK)]:
        s=pd.Series(arr,index=CLOCK);daily=s.resample('D').agg(['mean',lambda x:x.quantile(.1),lambda x:x.quantile(.9)]);ax.fill_between(daily.index,daily.iloc[:,1],daily.iloc[:,2],color=color,alpha=.15,label='Within-day 10th-90th percentile');ax.plot(daily.index,daily['mean'],color=color,lw=.8,label='Daily mean');ax.set(ylabel=label);compact_axis(ax);ax.legend(fontsize=8);grid(ax)
    save('26_price_carbon',fig)
    fig,axs=plt.subplots(1,2,figsize=(9.6,3.4),layout='constrained')
    for year,col in [(2023,TEAL),(2024,GOLD)]:
        sel=YEARS==year;s=pd.Series(weather.outdoor_dry_bulb_temperature.to_numpy()[sel],index=CLOCK[sel]);daily=s.resample('D').mean();axs[0].plot(np.arange(len(daily))+1,daily,color=col,lw=.8,label=str(year))
    axs[0].set(xlabel='Day of year',ylabel='Daily mean air temperature (°C)');axs[0].legend(fontsize=8);grid(axs[0])
    q=pd.Series(premium['pv'][y].sum(axis=1),index=CLOCK[y]).resample('D').sum();axs[1].plot(q.index,q,color=GOLD,lw=.8);axs[1].set(ylabel='Premium daily PV generation (kWh)',title='Day-to-day resource variation, 2024');axs[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2));axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%b'));grid(axs[1]);save('27_weather',fig)


def statistics(data,t):
    metrics={};inv=t['inventory'];m=t['members'];s=t['sessions'];cy=t['cycles']
    for name,d in data.items():
        mm=d['members'];res=mm[mm.category=='residential'];pv=mm[mm.has_pv];bs=mm[mm.has_battery];ss=d['sessions'];complete=ss[~ss.left_censored&~ss.right_censored];es=t['electrical_limits'].query('family==@name');annual=d['totals'].query('year==2024');sortedload=np.sort(annual.load_kwh)[::-1]
        metrics[name]={'inventory':inv.set_index('family').loc[name].to_dict(),'household_size':quantiles(res.occupants),'home_area':quantiles(res.floor_area_m2),'routines':mm.routine_archetype.value_counts().to_dict(),'pv_sizes':quantiles(pv.pv_capacity_kw),'pv_tilts':pv.pv_tilt_deg.value_counts().to_dict(),'pv_azimuths':pv.pv_azimuth_deg.value_counts().to_dict(),'batteries_by_size':bs.battery_capacity_kwh.value_counts().sort_index().to_dict(),'ev_capacities':d['electric_vehicles'].battery_capacity_kwh.value_counts().sort_index().to_dict(),'ev_ac_powers':d['electric_vehicles'].max_ac_charging_power.value_counts().sort_index().to_dict(),'chargers_by_power':d['chargers'].max_charging_power_kw.value_counts().sort_index().to_dict(),'single_phase_sites':int(es.connection.eq('single_phase').sum()),'three_phase_sites':int(es.connection.eq('three_phase').sum()),'top_one_load_share':float(sortedload[0]/sum(sortedload)*100),'top_five_load_share':float(sortedload[:5].sum()/sum(sortedload)*100),'top_quintile_load_share':float(sortedload[:int(np.ceil(len(sortedload)/5))].sum()/sum(sortedload)*100),'home_work_links':int(d['electric_vehicles'].work_member_id.ne('').sum()),'cycle_slack':quantiles(d['cycles'].start_slack_h),'cycle_duration':quantiles(d['cycles'].duration_h),'cycle_energy':quantiles(d['cycles'].total_energy_kwh),'session_duration':quantiles(complete.duration_h),'target_100_pct':float(np.isclose(complete.target_soc,1).mean()*100),'target_below_50_count':int((ss.target_soc<.5-1e-8).sum()),'soc_target':quantiles(complete.target_soc*100),'sessions_by_year':ss.year.value_counts().sort_index().to_dict(),'cycles_by_year':d['cycles'].year.value_counts().sort_index().to_dict(),'sessions_by_purpose':complete.site_group.value_counts().to_dict(),'pv_yield_2023':float(d['pv'][YEARS==2023].sum()/mm.pv_capacity_kw.sum()),'pv_yield_2024':float(d['pv'][YEARS==2024].sum()/mm.pv_capacity_kw.sum())}
    h=t['health'];health_summary=[]
    for (v,phase,cat,mod,mode),g in h.groupby(['variant','phase','category','module','mode']):
        intervals=set(i for row in g.itertuples() for i in range(row.start_time_step,row.end_time_step+1))
        health_summary.append(dict(variant=v,phase=phase,category=cat,module=mod,mode=mode,channel_rows=len(g),campaigns=len(g[['start_time_step','end_time_step']].drop_duplicates()),hours=len(intervals)/4,exposure_pct=len(intervals)/N*100,std=sorted(g['std'].dropna().unique().tolist()),values=sorted(g.value.dropna().unique().tolist())))
    root=data['Premium-100']['root'];p=pd.read_parquet(root/'shared/pricing_omie.parquet').electricity_pricing.to_numpy();c=pd.read_parquet(root/'shared/carbon_intensity.parquet').carbon_intensity.to_numpy();weather=pd.read_parquet(root/'shared/weather.parquet');market=[]
    for year in (2023,2024):
        sel=YEARS==year;market.append(dict(year=year,price_min=float(p[sel].min()),price_mean=float(p[sel].mean()),price_max=float(p[sel].max()),price_negative_hours=float((p[sel]<0).sum()/4),carbon_mean_g=float(c[sel].mean()*1000),carbon_min_g=float(c[sel].min()*1000),carbon_max_g=float(c[sel].max()*1000),temperature_mean=float(weather.outdoor_dry_bulb_temperature[sel].mean())))
    return {'families':metrics,'health_summary':health_summary,'market':market}


def main():
    OUT.mkdir(exist_ok=True,parents=True);FIG.mkdir(exist_ok=True,parents=True)
    data,tables=extract();charts(data,tables);metrics=statistics(data,tables)
    checks=[]
    def check(label,ok):
        checks.append({'name':label,'passed':bool(ok)})
        if not ok:raise AssertionError(label)
    for name,d in data.items():
        m=d['members'];s=d['sessions'];a=d['cycles'];base=next(v for v in records(tables['variants']) if v['family']==name and (v['variant'].endswith('_normal') or v['variant'] in ('core_4','core_15')))
        check(name+': static inventory equals schema',base['initial_assets']=={'pv':int(m.has_pv.sum()),'batteries':int(m.has_battery.sum()),'chargers':len(d['chargers']),'appliances':int(d['deferrables'].controlled.sum())})
        check(name+': consumer-only classification',m.pure_consumer.equals(~m[['has_pv','has_battery','has_charging','has_deferrable']].any(axis=1)))
        check(name+': unique session identities',s.session_id.is_unique)
        check(name+': sessions within physical SOC range',s.target_soc.between(.5-1e-8,1+1e-8).all() and (s.duration_h>0).all())
        check(name+': complete record count',len(d['load'])==N and len(d['pv'])==N)
        check(name+': end-use ledger sums to non-shiftable series',np.allclose(d['end_uses'].sum(axis=1),d['load'].sum(axis=1),atol=2e-5,rtol=1e-6))
        check(name+': controlled cycle profile lookup',a.total_energy_kwh.notna().all() and a.duration_steps.notna().all())
        check(name+': cycle windows',((a.earliest_start_time_step<=a.latest_start_time_step)&(a.latest_start_time_step+a.duration_steps-1<=a.deadline_time_step)).all())
    check('exactly nine configurations',len(tables['variants'])==9)
    check('exactly 28 descriptive figures',len(list(FIG.glob('*.png')))==28)
    metrics['descriptive_checks']=checks
    for name,table in tables.items():
        # Drop prose/JSON waveforms from machine-readable descriptive tables.
        table=table.drop(columns=['story','dynamic_membership_story','load_profile'],errors='ignore')
        table.to_csv(OUT/f'{name}.csv',index=False)
    metrics['variants']=records(tables['variants']);metrics['dr']=records(tables['dr']);metrics['version']=json.loads((ROOT/'results/portugal_rec_native/manifest.json').read_text())['version'];metrics['period']={'years':[2023,2024],'records':N,'interval_minutes':15,'days':731}
    metrics['input_checksums']={str(d['root'].relative_to(ROOT)):hashlib.sha256((d['root']/'file_checksums.sha256').read_bytes()).hexdigest() for d in data.values()}
    metrics['method']='Static description of catalogues, original-engine schemas and selected input series. No CityLearn imports, actions, controller policies or experiment outputs.'
    metrics['counting']='Inventory is the full static base; session and controlled-cycle distributions are read from the Normal schemas. Dynamic active counts are reconstructed from schema topology events. Energy is non-shiftable load and PV separately; no EV/controlled-cycle double-counting or performance KPI.'
    dump(OUT/'statistics.json',metrics)
    print(json.dumps({'inventory':records(tables['inventory']),'market':metrics['market'],'variants':metrics['variants']},indent=2))


if __name__=='__main__':main()

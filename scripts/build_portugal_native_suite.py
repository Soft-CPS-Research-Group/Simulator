"""Compile existing Portugal inputs for unchanged Simulator 978be68b.

No CityLearn import, patch, wrapper, policy or runtime override is used here.
"""
from pathlib import Path
from copy import deepcopy
import argparse, json, shutil, hashlib, re
import numpy as np
import pandas as pd

COMMIT = '978be68b0e0740deab15ecf536805f323b86fa57'
VERSION = 'portugal-native-978be68b-v2.1'
COMMON_PRICING = 'shared/pricing_btn_bi.parquet'
COMMON_TARIFF_STORY = 'Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023.'

def common_tariff_story(text):
    """Keep active prose consistent with pricing, without rewriting provenance."""
    return re.sub(r'Tarifa de referência(?: comum a todos os membros)?: [^,\n]+, indexada a OMIE 2023\.', COMMON_TARIFF_STORY, text)

def native_story(text):
    """Describe only the mobility and exit semantics of the original engine."""
    text = common_tariff_story(text)
    text = text.replace('com energia das viagens contabilizada',
        'com itinerários descritos nos dados; o SOC é prescrito em cada chegada, sem balanço longitudinal das viagens no motor original')
    text = text.replace('fora da adesão à REC mantém o carregamento privado, com o mesmo veículo e sem reiniciar o SOC',
        'quem sai deixa o cenário com todos os seus ativos; não existe carregamento privado posterior no cenário')
    return text

def correct_core30_visitor_durations(root):
    """Two 75-minute cafe visits need one more 15-minute interval.

    Arrival SOC, requested SOC, charger and vehicle limits remain unchanged.
    The 15:00 local departure remains within the cafe's Sunday opening hours.
    """
    sessions_path=root/'catalogs/charging_sessions.parquet'
    charger_path=root/'chargers/CHG-M028-01.parquet'
    sessions=pd.read_parquet(sessions_path);charger=pd.read_parquet(charger_path)
    corrections=[]
    for sid in ('EV-V028-01-1-S0092','EV-V028-01-1-S0109'):
        selected=sessions.session_id.eq(sid)
        assert selected.sum()==1, sid
        row=sessions.loc[selected].iloc[0]
        arrival,departure=int(row.arrival_time_step),int(row.departure_time_step)
        assert departure-arrival==5 and charger.loc[departure,'electric_vehicle_charger_state']==3, sid
        conflict=sessions.loc[~selected & ((sessions.charger_id==row.charger_id)|(sessions.ev_id==row.ev_id))]
        assert not ((conflict.arrival_time_step<departure+1)&(conflict.departure_time_step>arrival)).any(),sid
        charger.loc[departure]=charger.loc[departure-1]
        charger.loc[arrival:departure,'electric_vehicle_departure_time']=np.arange(6,0,-1,dtype=float)
        # This column is an illustrative pre-action SOC; native arrivals alone
        # initialize the battery. Preserve existing rows and extend that trace.
        step_gain=11.04*.25*.95*np.sqrt(.96)/64.
        charger.loc[departure,'electric_vehicle_current_soc']=min(float(row.target_soc),float(charger.loc[departure-1,'electric_vehicle_current_soc'])+step_gain)
        sessions.loc[selected,'departure_time_step']=departure+1
        corrections.append(dict(session_id=sid,member_id=row.member_id,charger_id=row.charger_id,arrival_time_step=arrival,original_departure_time_step=departure,departure_time_step=departure+1,arrival_soc=float(row.arrival_soc_reference),target_soc=float(row.target_soc),reason='75 minutes cannot deliver requested energy at native charger/battery efficiencies; extend cafe visit to 90 minutes, ending at 15:00 local'))
    charger.to_parquet(charger_path,index=False);sessions.to_parquet(sessions_path,index=False)
    dump(root/'native_input_corrections.json',{'version':VERSION,'visitor_sessions':corrections,'mobility_reference_scope':'Unchanged mobility/reference_soc is historical source metadata, not a native longitudinal SOC trajectory.'})
    return corrections

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')

def compile_suite(source, output, tariff):
    source, output = source.resolve(), output.resolve()
    if tariff != 'common':
        raise ValueError('The current catalogue uses one common grid tariff.')
    if source == output or output in source.parents or source in output.parents:
        raise ValueError('Source and output must be separate directory trees.')
    manifest = json.loads((source/'results/portugal_rec_v2/evaluation_v2/manifest.json').read_text())
    changes=[];native_event_catalog=[];session_corrections=[]
    families=set(v['family'] for v in manifest['variants'])
    for family in families:
        name='rec_pt_2023_'+family+'_v2'
        dst=output/'data/datasets'/name
        if dst.exists(): shutil.rmtree(dst)
        shutil.copytree(source/'data/datasets'/name,dst)
        # Old executable schemas are superseded, not advertised as compatible.
        shutil.rmtree(dst/'schemas',ignore_errors=True)
        for p in (dst/'evaluation').glob('*.json'):p.unlink()
        if family=='core_30':session_corrections=correct_core30_visitor_durations(dst)
    for entry in manifest['variants']:
        name=entry['variant'];srcpath=source/entry['schema'];srcroot=srcpath.parent.parent
        root=output/entry['schema'];root=root.parent.parent
        s=json.loads(srcpath.read_text());original=deepcopy(s)
        out=root/'evaluation/native_inputs'/name;out.mkdir(parents=True,exist_ok=True)
        def save(frame, filename):
            p=out/filename;frame.to_parquet(p,index=False);return str(p.relative_to(root))
        events=s.get('topology_events',[])
        absent={(i['member_id'],i['asset_type']) for i in s.get('initially_absent_assets',[])}
        exits={e['target_member_id']:e['time_step'] for e in events if e['operation']=='remove_member'}
        for mid,b in s['buildings'].items():
            raw=pd.read_parquet(srcroot/b['energy_simulation'])
            # Standby is an exogenous known asset-presence load in the dataset.
            standby=float(b.get('battery_standby_power_w',0))/1000*.25
            present=np.full(len(raw),b.get('electrical_storage',{}).get('attributes',{}).get('capacity',0)>0)
            if (mid,'electrical_storage') in absent:present[:]=False
            for e in events:
                if e.get('target_member_id')==mid and e.get('target_asset_type')=='electrical_storage':
                    present[e['time_step']:]=e['operation']=='add_asset'
            live=np.arange(len(raw))<exits.get(mid,len(raw))
            raw['non_shiftable_load']=np.maximum(0.,raw.non_shiftable_load.to_numpy()-standby*(~present & live))
            # Native outage input is the energy-simulation column.
            outage=b.get('power_outage',{})
            raw['power_outage']=0
            for lo,hi in outage.get('outage_windows',[]):raw.loc[lo:hi,'power_outage']=1
            no_supply=raw.power_outage.eq(1)&~present
            raw.loc[no_supply,'solar_generation']=0.
            b['energy_simulation']=save(raw,mid+'.parquet')
            outage.pop('outage_windows',None)
            for key in ('islanding','battery_standby_power_w','physical_asset_ids'):b.pop(key,None)
            for app in b.get('deferrable_appliances',{}).values():
                app.get('attributes',{}).pop('phase_connection',None)
            # Use a real existing template and native property overrides for
            # first commissioning. No fictitious removal/re-entry at t=0.
            for kind in ('pv','electrical_storage'):
                if (mid,kind) not in absent:continue
                measure='nominal_power' if kind=='pv' else 'capacity'
                attrs=deepcopy(b[kind]['attributes'])
                donor=next(m for m,x in original['buildings'].items() if m!=mid and x.get(kind,{}).get('attributes',{}).get(measure,0)>0 and (m,kind) not in absent)
                for e in events:
                    if e.get('target_member_id')==mid and e.get('target_asset_type')==kind and e['operation']=='add_asset':
                        e['source_member_id']=donor
                        e['overrides']={**attrs,**e.get('overrides',{})}
                        e['overrides'].pop('phase_connection',None)
                b[kind]['attributes'][measure]=0.
                # Absolute generation ignores nominal_power in the original PV.
                # A zero-capacity per-kW placeholder genuinely produces zero;
                # the native add_asset event restores the absolute-mode asset.
                if kind=='pv':b[kind]['attributes']['generation_mode']='per_kw'
                if kind=='electrical_storage':b[kind]['attributes']['nominal_power']=0.
        # Native EVs reset SOC at each arrival. Mobility remains source metadata.
        # Battery input-power limit models AC acceptance after charger losses.
        efficiency={eid:1. for eid in s['electric_vehicles_def']}
        for mid,b in s['buildings'].items():
            for cid,ch in b.get('chargers',{}).items():
                a=ch['attributes'];eta=min([a.get('efficiency',1.)]+[x[1] for x in a.get('charge_efficiency_curve',[]) or []])
                frame=pd.read_parquet(srcroot/ch['charger_simulation'])
                for eid in frame.electric_vehicle_id.dropna().unique():
                    if eid in efficiency:efficiency[eid]=min(efficiency[eid],eta)
                # A retired charger has no subsequent sessions in this REC.
                end=min([e['time_step'] for e in events if e['operation']=='remove_asset' and e.get('target_asset_type')=='charger' and e.get('target_asset_id')==cid]+[35040])
                if end<35040:
                    frame.loc[end:,'electric_vehicle_charger_state']=0
                    frame.loc[end:,['electric_vehicle_id','electric_vehicle_session_id']]=''
                    for col in frame.columns:
                        if col not in ('electric_vehicle_charger_state','electric_vehicle_id','electric_vehicle_session_id'):frame.loc[end:,col]=np.nan
                    ch['charger_simulation']=save(frame,cid+'.parquet')
        for eid,ev in s['electric_vehicles_def'].items():
            limit=ev.pop('max_ac_charging_power',None);ev.pop('mobility_file',None)
            if limit:ev['battery']['attributes']['nominal_power']=min(ev['battery']['attributes']['nominal_power'],limit*efficiency[eid])
        native_templates=[]
        for item in original.get('initially_absent_assets',[]):
            if item['asset_type']!='charger':continue
            mid,cid=item['member_id'],item['asset_id']
            template='NATIVE_TEMPLATE__'+cid
            s['buildings'][mid]['chargers'][template]=s['buildings'][mid]['chargers'].pop(cid)
            for event in events:
                if event['operation']=='add_asset' and event.get('target_asset_id')==cid:
                    event['source_member_id']=mid;event['source_asset_id']=template
            # Native initialization removes the configuration template before
            # the first observation. A distinct real ID is commissioned later.
            # The original engine retains this independent schedule in its
            # template snapshot. No real asset is removed/re-entered at t=0.
            events.insert(0,dict(id='native_template_registration_'+cid,time_step=0,operation='remove_asset',target_member_id=mid,target_asset_type='charger',target_asset_id=template,purpose='native_schema_template_only'))
            native_templates.append(dict(member_id=mid,template_id=template,physical_asset_id=cid,visible_in_observations=False,physical_retirement=False))
        # Explicit dependency rows use the native per-feature event interface.
        rb=s.get('robustness',{})
        if rb.get('enabled'):
            dynamic_name='core_30_combined' if entry['family']=='core_30' else 'premium_100_allin'
            dynamic=json.loads((srcroot/'evaluation'/f'{dynamic_name}.json').read_text())
            dynamic_exits={e['target_member_id'] for e in dynamic.get('topology_events',[]) if e['operation']=='remove_member'}
            changed_assets={(e['target_member_id'],e.get('target_asset_type')) for e in dynamic.get('topology_events',[]) if e['operation'] in ('add_asset','remove_asset')}
            stable=lambda m:dynamic['buildings'][m].get('include',True) and m not in dynamic_exits
            # Native target validation checks all event targets at every layout
            # change, including past events. Use the same persistent sensor
            # cohort in static and dynamic variants, recording the restriction.
            pv_members=[m for m,b in original['buildings'].items() if stable(m) and (m,'pv') not in changed_assets and b.get('pv',{}).get('attributes',{}).get('nominal_power',0)>0]
            storage_target=next(m for m,b in original['buildings'].items() if stable(m) and (m,'electrical_storage') not in changed_assets and b.get('electrical_storage',{}).get('attributes',{}).get('capacity',0)>0)
            def translate(frame):
                rows=[]
                for row in frame.to_dict('records'):
                    group=row.pop('dependency_group',None)
                    if pd.isna(group):group=None
                    if group=='native_load':targets=[('building',row['target_id'],f) for f in ('non_shiftable_load','load_power_kw','load_energy_kwh_step','pv_surplus_power_kw','pv_surplus_energy_kwh_step')]
                    elif group=='pv_generation':
                        targets=[('pv',m+'/pv',f) for m in pv_members for f in ('generation_power_kw','generation_energy_kwh_step','generation_capacity_factor_ratio')]
                        targets += [('building',m,f) for m in pv_members for f in ('solar_generation','pv_power_kw','pv_energy_kwh_step','pv_surplus_power_kw','pv_surplus_energy_kwh_step')]
                        targets += [('district','*',f) for f in ('community_pv_power_kw','community_pv_energy_kwh_step')]
                    elif group=='price_forecast':
                        targets=[(tab,'*','electricity_pricing_predicted_'+str(i)) for tab in ('building','district') for i in (1,2,3)]
                        targets += [('district','*','forecast_price_next_'+h) for h in ('15m','1h','3h','6h','24h')]
                    elif row['target_type']=='storage' and row['mode']=='unavailable':targets=[('storage',storage_target,row['target_feature'])]
                    else:targets=[(row['target_type'],row['target_id'],row['target_feature'])]
                    for i,(tab,ident,feat) in enumerate(targets):
                        rows.append({**row,'event_id':row['event_id']+(f'_channel_{i}' if group else ''),'target_type':tab,'target_id':ident,'target_feature':feat})
                return pd.DataFrame(rows)
            source_events=rb['events_file']
            rb['events_file']=save(translate(pd.read_parquet(srcroot/source_events)),'robustness.parquet')
            confirmation=source_events.replace('/development/','/confirmation/')
            s['experimental_design']['native_confirmation_robustness_file']=save(translate(pd.read_parquet(srcroot/confirmation)),'confirmation_robustness.parquet')
            rb.pop('expose_fault_metadata',None)
            s['experimental_design']['native_health_persistent_pv_members']=pv_members
            s['experimental_design']['native_health_unavailable_storage_member']=storage_target
        s['observation_bundles']['entity_robustness']=False
        # The native meta/health payload can still reveal faults: documented.
        if s.get('demand_response',{}).get('enabled'):
            dr=s['demand_response']
            s['demand_response']={'enabled':True,'requests_file':dr['requests_file'],'baseline_method':'rolling_pre_event_average','baseline_window_seconds':3600,'allow_overlapping_requests':False}
        for key in ('permanent_asset_retirements','initially_absent_assets','outside_community_service','minimum_simulator_features'):s.pop(key,None)
        # Keep the full grid purchase price in the data. Local trading and its
        # ratio are an experiment-schema choice, using the unchanged engine.
        for b in s['buildings'].values():b['pricing']=COMMON_PRICING
        s['community_market']={'enabled':False,'grid_export_price':0.0}
        s['experimental_design'].update(version=VERSION,agent_fault_labels='native metadata may expose events',mobility_semantics='independent arrivals; SOC prescribed per session; no longitudinal trip-energy claim',tariff_policy=tariff,simulator_commit=COMMIT)
        s['native_compatibility']={'engine_commit':COMMIT,'no_runtime_modifications':True,'appliance_phases':'native building default split','backup':'native aggregate supply; no extra hybrid inverter cap','dr_reference':'native rolling pre-event mean; no calibration/qualification claim','external_service':'source mobility is descriptive, not physically simulated','original_schema_sha256':hashlib.sha256(srcpath.read_bytes()).hexdigest()}
        s['native_compatibility']['declared_appliance_phases']={a:cfg.get('attributes',{}).get('phase_connection') for b in original['buildings'].values() for a,cfg in b.get('deferrable_appliances',{}).items()}
        if native_templates:s['native_compatibility']['charger_configuration_templates']=native_templates
        if 'physical_fingerprint' in s['experimental_design']:
            s['experimental_design']['source_physical_fingerprint']=s['experimental_design'].pop('physical_fingerprint')
        if 'physical_fingerprint' in entry:entry['source_physical_fingerprint']=entry.pop('physical_fingerprint')
        if rb.get('enabled'):
            for phase,file in [('development',rb['events_file']),('confirmation',s['experimental_design']['native_confirmation_robustness_file'])]:
                native_event_catalog.extend(dict(variant=name,family=entry['family'],phase=phase,kind='robustness',**row) for row in json.loads(pd.read_parquet(root/file).to_json(orient='records')))
        for mid,b in original['buildings'].items():
            for lo,hi in b.get('power_outage',{}).get('outage_windows',[]):native_event_catalog.append(dict(variant=name,family=entry['family'],phase='development',kind='grid',member_id=mid,start_time_step=lo,end_time_step=hi))
        if s.get('demand_response',{}).get('enabled'):
            native_event_catalog.extend(dict(variant=name,family=entry['family'],phase='development',kind='dr',**row) for row in json.loads(pd.read_parquet(root/s['demand_response']['requests_file']).to_json(orient='records')))
        s['root_directory']='..'
        dump(output/entry['schema'],s)
        entry['sha256']=hashlib.sha256((output/entry['schema']).read_bytes()).hexdigest()
        changes.append({'variant':name,'initially_absent_assets':len(absent),'native_events':len(events),'tariff':tariff})
    for family in families:
        root=output/'data/datasets'/('rec_pt_2023_'+family+'_v2')
        entries=[x for x in manifest['variants'] if x['family']==family]
        base=next((x for x in entries if x['variant'].endswith('_normal')),entries[0])
        s=json.loads((output/base['schema']).read_text());s['root_directory']='.'
        dump(root/'schema.json',s)
        dump(root/'CURRENT_EVALUATION.json',{'version':VERSION,'simulator_commit':COMMIT,'status':'dataset_construction_only','tariff_policy':tariff,'variants':entries})
        (root/'README.md').write_text(f'# Portugal — {family}\n\nVersão {VERSION}. Apenas dados; motor original {COMMIT}.\n\nAs nove configurações do catálogo usam quatro bases de 2023, com intervalos de 15 minutos. Os schemas correntes desta família estão em evaluation/. A tarifa de compra é comum: {COMMON_PRICING}, em EUR/kWh, sem desconto de mercado. O mercado está desligado nos schemas de dados; os 80% configuram-se depois no schema da experiência, através de community_market.local_price_ratio_to_grid_import. Exportação à rede: zero.\n\nOs catálogos e ficheiros de mobilidade conservam os dados de origem; a execução nativa usa SOC por sessão e não simula o registo longitudinal de viagens. Ver native_compatibility em cada schema. As séries tarifárias alternativas em shared/ e as atribuições source_tariff_id são proveniência, não tarifas ativas. Esta revisão só constrói dados: não foram feitas novas avaliações de utilização.\n')
        members_path=root/'catalogs/members.parquet'
        members=pd.read_parquet(members_path)
        members['source_tariff_id']=members['tariff_id']
        members['tariff_id']='btn_bi'
        for column in ('story','dynamic_membership_story'):
            if column in members:members[column]=members[column].map(lambda v:native_story(v) if isinstance(v,str) else v)
        members.to_parquet(members_path,index=False)
        for story_path in root.rglob('MEMBER_STORIES.md'):
            story_path.write_text(native_story(story_path.read_text()))
        # Manifests of original generation remain provenance, not current checksums.
        for old in ('file_checksums.sha256','validation_report.json','dataset_manifest.json'):
            if (root/old).exists():
                target=root/'provenance_original'/old;target.parent.mkdir(exist_ok=True)
                shutil.move(root/old,target)
        dump(root/'dataset_manifest.json',{'version':VERSION,'engine_commit':COMMIT,'source_version':'portugal-v2.3','tariff_policy':tariff,'active_pricing_file':COMMON_PRICING,'market_configuration_scope':'experiment schema; disabled in dataset schemas','current_configurations':[v['variant'] for v in entries],'validation':'construction only; static integrity checks, no controller or usage evaluation for this revision'})
        old = root/'settlement_contract.json'
        if old.exists():
            shutil.move(old,root/'provenance_original/settlement_contract.json')
        dump(old,{'runtime_commit':COMMIT,'tariff_policy':tariff,'active_pricing_file':COMMON_PRICING,'grid_export_price':0,'dataset_market_enabled':False,'price_series_contains_local_discount':False,'configuration_scope':'enable community_market and set local_price_ratio_to_grid_import=0.8 in the experiment schema','native_rule':'local price = common grid purchase price times experiment ratio','cash_conservation':'common price for buyers and sellers; use original simulator settlement','not_external_rebalanced':True})
        checks={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and p!=root/'file_checksums.sha256'}
        (root/'file_checksums.sha256').write_text(''.join(f'{h}  {p}\n' for p,h in sorted(checks.items())))
    manifest.update(version=VERSION,simulator_commit=COMMIT,physical_version='portugal-v2.3 with explicit native input adaptations',tariff_policy=tariff,events=native_event_catalog)
    manifest['pricing']='One common full grid purchase tariff: shared/pricing_btn_bi.parquet, EUR/kWh. Grid export 0. Market disabled in dataset schemas; enable original settlement and configure 0.8 in experiment schemas. No local discount in price series. Explicit DR only in Premium All-in.'
    manifest['source_history_not_executed']=manifest.pop('history',{})
    manifest['history']={'baseline_method':'rolling_pre_event_average','baseline_window_seconds':3600,'qualified_commercial_baseline':False}
    manifest['validation_scope']='Dataset construction only. Static integrity checks; no usage/controller evaluation for this common-tariff revision. Previous individual-tariff simulations are historical.'
    manifest['status']='dataset_construction_only'
    manifest['input_corrections']={'core_30_visitor_sessions':session_corrections,'tariff_narratives':'Common BTN bi-hourly reference in member catalogues and stories.'}
    dest=output/'results/portugal_rec_native'
    dump(dest/'manifest.json',manifest);dump(dest/'adaptations.json',changes)
    print(json.dumps({'configurations':len(changes),'output':str(output),'version':VERSION}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tariff',choices=['common'],default='common');a=p.parse_args()
    compile_suite(a.source,a.output,a.tariff)

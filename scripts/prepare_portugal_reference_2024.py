"""Offline 2024 source alignment; writes dataset inputs, never simulator code.

Sources are archived in data/portugal_reference_2024. Hourly satellite/reanalysis
weather is interpolated, not presented as measured quarter-hour observations.
"""
from pathlib import Path
import gzip,json,zipfile,xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import pvlib
from prepare_portugal_reference import civil_day
ROOT=Path(__file__).resolve().parents[1]
UTC=pd.date_range('2024-01-01','2025-01-01',freq='15min',inclusive='left',tz='UTC')

def eredes(root):
    # E-REDES rows are chronological interval ends, including 92/100 slots
    # on Lisbon DST transition days. Preserve row order on the regular UTC axis.
    ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(root/'E-REDES_Perfil_Consumo_2024.xlsx') as z:
        sheet=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    rows=[];days=[]
    for row in sheet.findall('m:sheetData/m:row',ns)[4:]:
        vals={c.attrib['r'].rstrip('0123456789'):c.findtext('m:v',namespaces=ns) for c in row}
        rows.append([float(vals[k]) for k in ['D','E','F','G']]);days.append(int(float(vals['A'])))
    raw=np.array(rows);assert raw.shape==(35136,4)
    local=UTC.tz_convert('Europe/Lisbon');expected=(local.tz_localize(None).normalize()-pd.Timestamp('1899-12-30')).days
    assert np.array_equal(days,expected), 'E-REDES civil date / DST alignment'
    out=pd.DataFrame(raw,columns=['BTN_A','BTN_B','BTN_C','IP']);out.insert(0,'timestamp_utc',UTC)
    out.to_parquet(root/'eredes_profiles_2024.parquet',index=False)
    return out

def market(root):
    blocks=[]
    for p in sorted([*(root/'raw_2024').glob('omie_*.txt'), *(root/'forecast_support').glob('omie_*.txt')]):
        day=p.stem.removeprefix('omie_');rows=[]
        for line in p.read_text().splitlines():
            f=line.split(';')
            if len(f)>=6 and f[0].isdigit():rows.append([int(f[3]),float(f[4]),float(f[5])])
        if not rows:continue
        f=pd.DataFrame(rows,columns=['market_period','price_spain_eur_per_mwh','price_portugal_eur_per_mwh'])
        idx=civil_day(day,'Europe/Madrid','h');assert len(f)==len(idx),(day,len(f))
        assert f.market_period.tolist()==list(range(1,len(idx)+1))
        f['timestamp_utc']=idx.tz_convert('UTC');f['market_date']=day;blocks.append(f)
    hourly=pd.concat(blocks).set_index('timestamp_utc').sort_index()
    assert hourly.index.is_unique
    h=hourly.reindex(UTC[::4]);assert h.notna().all().all()
    h.reset_index().to_parquet(root/'omie_2024_hourly.parquet',index=False)
    c=pd.DataFrame(index=UTC);c['omie_eur_kwh']=h.price_portugal_eur_per_mwh.reindex(UTC,method='ffill')/1000;c['mibel_eur_kwh']=0.
    local=UTC.tz_convert('Europe/Lisbon');off=(local.hour<8)|(local.hour>=22)
    jun=local>=pd.Timestamp('2024-06-01',tz='Europe/Lisbon');apr=local>=pd.Timestamp('2024-04-01',tz='Europe/Lisbon')
    c['access_energy_eur_kwh']=np.where(jun,np.where(off,.0157,.0860),np.where(off,.0092,.0502))
    c['access_peak_power_eur_kwh']=0.;c['cgs_eur_kwh']=.004;c['supplier_margin_eur_kwh']=.010
    c['tse_eur_kwh']=np.where(apr,.002893,0.);c['loss_multiplier']=1.16;c['iec_eur_kwh']=.001
    c['electricity_pricing']=(c.omie_eur_kwh+.014+c.tse_eur_kwh)*1.16+c.access_energy_eur_kwh+.001
    c.reset_index(names='timestamp_utc').to_parquet(root/'price_components_btn_bi.parquet',index=False)
    past23=pd.read_parquet(ROOT/'data/portugal_reference_2023/price_components_btn_bi.parquet').set_index('timestamp_utc')
    # One support day outside the simulation window. January 2's first Madrid
    # market hour supplies January 1 at 23:00 UTC; no 2025 simulation rows added.
    support_clock=pd.date_range('2025-01-01','2025-01-02',freq='15min',inclusive='left',tz='UTC')
    support=pd.DataFrame(index=support_clock)
    support['omie_eur_kwh']=hourly.price_portugal_eur_per_mwh.reindex(support_clock,method='ffill')/1000
    assert hourly.reindex(support_clock[::4]).notna().all().all(), 'Missing support market hours'
    off=(support_clock.hour<8)|(support_clock.hour>=22)
    support['access_energy_eur_kwh']=np.where(off,.0149,.0830)
    support['tse_eur_kwh']=.0016574
    support['electricity_pricing']=(support.omie_eur_kwh+.014+support.tse_eur_kwh)*1.16+support.access_energy_eur_kwh+.001
    support.reset_index(names='timestamp_utc').to_parquet(root/'forecast_support/price_components.parquet',index=False)
    both=pd.concat([past23,c,support]);offset=len(past23);idx=pd.DatetimeIndex(both.index);madrid=idx.tz_convert('Europe/Madrid')
    out=pd.DataFrame({'electricity_pricing':c.electricity_pricing.to_numpy()});issue=np.arange(len(c))+offset
    boundary=[]
    for k,horizon in enumerate((4,24,96),1):
        target=issue+horizon
        assert target.max()<len(both)
        pub=(madrid[target].tz_localize(None).normalize()-pd.Timedelta(days=1)+pd.Timedelta(hours=14)).tz_localize('Europe/Madrid').tz_convert('UTC')
        earlier=np.minimum(target-96,issue)
        omie=np.where(UTC>=pub,both.omie_eur_kwh.to_numpy()[target],both.omie_eur_kwh.to_numpy()[earlier])
        # The retroactive TSE was announced 30 April. A causal forecast cannot
        # know that surcharge during April; conservatively use it from May 1.
        tse=np.where(UTC>=pd.Timestamp('2024-05-01',tz='Europe/Lisbon'),both.tse_eur_kwh.fillna(0).to_numpy()[target],0.)
        out[f'electricity_pricing_predicted_{k}']=(omie+.014+tse)*1.16+both.access_energy_eur_kwh.to_numpy()[target]+.001
        mask=target>=offset+len(c)
        boundary.append(pd.DataFrame({'issue_utc':UTC[mask], 'target_utc':idx[target[mask]],
            'horizon_steps':horizon, 'publication_utc':pub[mask],
            'omie_source_utc':idx[np.where(UTC>=pub,target,earlier)[mask]],
            'day_ahead_available':(UTC>=pub)[mask],
            'forecast_eur_kwh':out.loc[mask,f'electricity_pricing_predicted_{k}'].to_numpy()}))
    pd.concat(boundary,ignore_index=True).to_parquet(root/'forecast_support/boundary_forecasts.parquet',index=False)
    out.astype('float32').to_parquet(root/'pricing_btn_bi.parquet',index=False)
    return c

def carbon(root):
    essential=['Hydro','Solar','Consumption','Pumping','Import','Export','Wind','Natural Gas','Other Thermal','Biomass','Coal']
    blocks=[]
    for p in sorted((root/'raw_2024').glob('ren_*.json.gz')):
        day=p.name[4:-8];j=json.loads(gzip.decompress(p.read_bytes()))['data'];idx=civil_day(day,'Europe/Lisbon','15min')
        assert list(idx.strftime('%H:%M'))==j['xAxis']['categories'],day
        d={x['name']:x['data'] for x in j['series']};b=pd.DataFrame({key:d[key] for key in essential},index=idx.tz_convert('UTC'),dtype=float)
        b['Wave']=pd.Series(d.get('Wave',[0]*len(idx)),index=b.index).fillna(0);blocks.append(b)
    out=pd.concat(blocks);assert out.index.equals(UTC) and out.notna().all().all() and (out>=0).all().all()
    vals=[]
    for year in (2024,2025):
        j=json.loads((root/f'ree_emissions_{year}_daily.json').read_text())['data'];vals.extend(next(x for x in j['included'] if x['type']=='tCO2 eq./MWh')['attributes']['values'])
    es=pd.Series([x['value'] for x in vals],index=pd.to_datetime([x['datetime'] for x in vals],utc=True)).sort_index()
    es=es[~es.index.duplicated()];out['es_daily_kg_co2e_kwh']=es.reindex(UTC,method='ffill')
    factor=json.loads((ROOT/'data/portugal_reference_2023/aligned_reference_metadata.json').read_text())['carbon']['effective_thermal_factor_kg_kwh']
    generation=out[['Hydro','Solar','Wind','Natural Gas','Other Thermal','Biomass','Coal','Wave']].sum(axis=1);fossil=out[['Natural Gas','Other Thermal','Coal']].sum(axis=1)
    out['pt_generation_mw']=generation;out['pt_production_kg_co2e_kwh']=factor*fossil/generation
    out['pt_import_blend_kg_co2e_kwh']=(factor*fossil+out.Import*out.es_daily_kg_co2e_kwh)/(generation+out.Import)
    out['carbon_intensity']=out.pt_import_blend_kg_co2e_kwh*1.16
    out['balance_residual_mw']=generation+out.Import-out.Export-out.Consumption-out.Pumping
    assert np.isfinite(out.to_numpy()).all()
    out.reset_index(names='timestamp_utc').to_parquet(root/'carbon_components_2024.parquet',index=False)
    out[['carbon_intensity']].astype('float32').reset_index(drop=True).to_parquet(root/'carbon_intensity.parquet',index=False)
    return {'effective_thermal_factor_kg_kwh':factor,'calibration_year':2023,'method':'Fixed 2023 APA-calibrated fossil coefficient; actual 2024 REN generation and daily REE imports; delivery multiplier 1.16. Derived operational estimate, not an APA-certified 2024 factor or marginal emissions.','delivered_mean_kg_kwh':float(out.carbon_intensity.mean())}

def weather(root):
    rows=[]
    for p in sorted((root/'raw_2024').glob('pvgis_2024_*.json.gz')):
        d=json.loads(gzip.decompress(p.read_bytes()))['data'];rows.append(pd.DataFrame({'timestamp_utc':pd.to_datetime(d['timestamps'],utc=True),'beam_horizontal':d['direct_inclined_irradiance_before_reflectivity'],'diffuse_horizontal':d['diffuse_inclined_irradiance_before_reflectivity']}))
    raw=pd.concat(rows,ignore_index=True);assert len(raw)==8784 and raw.timestamp_utc.is_unique
    raw.to_parquet(root/'pvgis_horizontal_2024.parquet',index=False)
    # SARAH contains seven missing daylight records in this year. Preserve the
    # gaps in the raw archive and fill only missing components using ERA5.
    solar=json.loads(gzip.decompress((root/'era5_open_meteo_solar_2024.json.gz').read_bytes()))['data']['hourly']
    si=pd.to_datetime(solar['time'],utc=True)-pd.Timedelta(minutes=30)
    filled=[]
    for col,key in [('beam_horizontal','direct_radiation'),('diffuse_horizontal','diffuse_radiation')]:
        bad=raw[col].isna();fallback=np.interp(pd.DatetimeIndex(raw.timestamp_utc).asi8.astype(float),pd.DatetimeIndex(si).asi8.astype(float),solar[key])
        for i in np.flatnonzero(bad):filled.append(dict(timestamp_utc=str(raw.timestamp_utc.iloc[i]),component=col,value=float(fallback[i]),source='ERA5 hourly preceding-interval mean, assigned to interval midpoint'))
        raw.loc[bad,col]=fallback[bad]
    (root/'weather_gap_fills.json').write_text(json.dumps(filled,indent=2)+'\n')

    er=json.loads(gzip.decompress((root/'era5_open_meteo_2024.json.gz').read_bytes()))['data']['hourly'];met=pd.DataFrame(er);met['time']=pd.to_datetime(met.time,utc=True)
    mid=UTC+pd.Timedelta(minutes=7.5);pos=pvlib.solarposition.get_solarposition(mid,38.72,-9.14);hp=pvlib.solarposition.get_solarposition(pd.DatetimeIndex(raw.timestamp_utc),38.72,-9.14)
    mu=np.clip(np.cos(np.radians(hp.zenith.to_numpy())),.05,1)
    interp=lambda idx,v:np.interp(mid.asi8.astype(float),pd.DatetimeIndex(idx).asi8.astype(float),np.array(v,float))
    daylight=pos.elevation.to_numpy()>0
    dni=np.where(daylight,interp(raw.timestamp_utc,np.minimum(raw.beam_horizontal/mu,1100)),0)
    dhi=np.where(daylight,interp(raw.timestamp_utc,raw.diffuse_horizontal),0)
    temp=interp(met.time,met.temperature_2m);wind=interp(met.time,met.wind_speed_10m)
    out=pd.DataFrame({'outdoor_dry_bulb_temperature':temp,'outdoor_relative_humidity':interp(met.time,met.relative_humidity_2m),'diffuse_solar_irradiance':dhi,'direct_solar_irradiance':dni})
    previous=pd.read_parquet(ROOT/'data/datasets/rec_pt_2023_micro_4_v2/shared/weather.parquet').iloc[:35040]
    for col in list(out):
        history=np.r_[previous[col].to_numpy(),out[col].to_numpy()];issue=np.arange(len(out))+len(previous)
        for k,h in enumerate((4,24,96),1):
            pred=history[np.minimum(issue+h-96,issue)].copy()
            if 'solar' in col:pred[pvlib.solarposition.get_solarposition(mid+pd.Timedelta(minutes=15*h),38.72,-9.14).elevation.to_numpy()<=0]=0
            out[f'{col}_predicted_{k}']=pred
    assert np.isfinite(out.to_numpy()).all()
    out.astype('float32').to_parquet(root/'weather.parquet',index=False)
    pd.DataFrame({'timestamp_utc':mid,'temperature':temp,'wind':wind,'dni':dni,'dhi':dhi}).to_parquet(root/'weather_physics.parquet',index=False)

def main():
    root=ROOT/'data/portugal_reference_2024';eredes(root);c=market(root);co=carbon(root);weather(root)
    m={'year':2024,'records':35136,'formula':'(OMIE + 0.004 + 0.010 + TSE)*1.16 + TAR_BTN_bi + 0.001; EUR/kWh pre-VAT; TSE=0.002893 from April 1 retrospectively; zero MIBEL','tariff':'Common reference, variable energy only; excludes fixed charges, VAT and audiovisual/DGEG fees. Not a reconstruction of an individual bill. Negative prices preserved.','carbon':co,'weather':'PVGIS6 SARAH-3 hourly horizontal irradiation; ERA5 temperature, humidity and wind via Open-Meteo; 38.72N -9.14E. 15-minute values interpolated, night masked astronomically; no fabricated weather noise.','eredes':'Published chronological local interval ends (92/100 slots on DST transition days), aligned by row order to UTC after checking every civil date. Profile is a synthetic demand prior, not measured household consumption.','forecast':'Causal persistence for weather; published day-ahead OMIE (14:00 Madrid cutoff), prior-day fallback; retroactive TSE excluded until May 1 publication known.','grid_price_min':float(c.electricity_pricing.min()),'grid_price_max':float(c.electricity_pricing.max())}
    m['forecast_support']={'period_utc':'2025-01-01 00:00 through 23:45','records':96,'simulated':False,'scope':'Market forecast targets only; meteorological persistence already uses full horizons. Internal native load/PV boundary clipping is unchanged.','tar_2025_eur_kwh':{'off_peak':.0149,'other':.0830},'tse_2025_eur_kwh':.0016574,'contract':'Same common BTN benchmark margin, CGS, losses and IEC; future TAR/TSE known from ERSE announcements of 16 and 20 December 2024.','sources':'forecast_support/sources.json'}
    (root/'aligned_reference_metadata.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n');print(json.dumps(m,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

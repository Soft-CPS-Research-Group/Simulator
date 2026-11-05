"""Measured commercial/appliance benchmarks, with held-out physical identities.

Download large originals into RAM only. Retain the selected commercial meters,
appliance cycle summaries and source provenance, never raw archives or clones.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
from revise_portugal_ev_mobility import dump,write

REFERENCE=ROOT/"data/portugal_validation_reference"
PROVENANCE=REFERENCE/"commercial_appliance_sources.json"
OUTPUT=ROOT/"results/portugal_rec_native/commercial_appliance_validation.json"
BDG_SHA="9b97ccbe90096aff42ed4fd6493bf7ae692d7118"
NILMTK_SHA="8967bd1fde24d0c8d89408179dd2499d8e60b4c3"
REFIT_HOUSES={2:"heldout",3:"development",5:"development",11:"heldout"}
# NILMTK REFIT converter metadata: meter 1 is aggregate; IAM column=meter-1.
REFIT_CHANNELS={2:{"washing_machine":2,"dishwasher":3},
                3:{"washing_machine":6,"dishwasher":5},
                5:{"washing_machine":3,"dishwasher":4},
                11:{"washing_machine":3,"dishwasher":4}}


def get(url):
    response=requests.get(url,timeout=(30,180));response.raise_for_status()
    return response.content


def get_large(url,size):
    """Bounded public byte-range requests avoid gateway timeouts on large CSVs."""
    payload=io.BytesIO();block=32*1024*1024
    for start in range(0,size,block):
        end=min(size-1,start+block-1)
        for attempt in range(3):
            try:
                response=requests.get(url,headers={"Range":f"bytes={start}-{end}"},timeout=(30,90))
                response.raise_for_status()
                if response.status_code==200:
                    assert start==0 and len(response.content)==size
                    return response.content
                assert response.status_code==206 and response.headers["Content-Range"].startswith(f"bytes {start}-{end}/")
                assert len(response.content)==end-start+1
                payload.write(response.content);break
            except (requests.Timeout,requests.HTTPError) as error:
                if attempt==2:raise
                if isinstance(error,requests.HTTPError) and error.response.status_code<500:raise
        print("Downloaded",end+1,"of",size,"bytes",flush=True)
    return payload.getvalue()


def sha(data):return hashlib.sha256(data).hexdigest()


def source_record(url,data,license):
    return dict(url=url,bytes=len(data),sha256=sha(data),license=license)


def provenance():
    return json.loads(PROVENANCE.read_text()) if PROVENANCE.exists() else dict(revision="portugal-commercial-appliance-1",sources=[])


def save_reference(name,frame,record):
    path=REFERENCE/name;write(path,frame)
    record.setdefault("derived_sha256",{})[name]=sha(path.read_bytes())


def bdg_cohort(metadata):
    """Selection uses published labels/areas, never meter outcomes."""
    f=metadata.loc[metadata.electricity.eq("Yes") & metadata.sqm.ge(100)].copy()
    f["sector"]=""
    f.loc[f.primaryspaceusage.eq("Office") & f.sqm.le(3000),"sector"]="office"
    f.loc[f.primaryspaceusage.eq("Retail"),"sector"]="retail"
    f.loc[f.primaryspaceusage.eq("Food sales and service"),"sector"]="cafe"
    f.loc[f.sub_primaryspaceusage.eq("Market"),"sector"]="grocery"
    f.loc[f.sub_primaryspaceusage.eq("Maintenance Repair Shop"),"sector"]="workshop"
    f.loc[f.sub_primaryspaceusage.isin(["Health Services","Urgent Care Center/Clinic/Other Outpatient Office"]),"sector"]="clinic"
    f.loc[f.sub_primaryspaceusage.eq("Senior Care Community"),"sector"]="care_home"
    f.loc[f.sub_primaryspaceusage.isin(["Gymnasium","Fitness Center","Sports Facility"]),"sector"]="sports_centre"
    f.loc[f.primaryspaceusage.eq("Manufacturing/industrial") & f.sub_primaryspaceusage.isin(["Industrial","Manufacturing Facility"]),"sector"]="industrial_reference_only"
    f=f.loc[f.sector.ne("")]
    f["selection_hash"]=f.building_id.map(lambda value:sha(value.encode()))
    selected=[]
    for sector,group in f.groupby("sector"):
        group=group.sort_values("selection_hash").head(24 if sector=="office" else 12)
        group["split"]=["heldout" if i%2==0 else "development" for i in range(len(group))]
        selected.append(group)
    return pd.concat(selected,ignore_index=True)


def acquire_commercial():
    record=provenance()
    base=f"https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/{BDG_SHA}/"
    meta_url=base+"data/metadata/metadata.csv";raw_meta=get(meta_url)
    selected=bdg_cohort(pd.read_csv(io.BytesIO(raw_meta)))
    record["commercial_protocol"]=dict(commit=BDG_SHA,
        selection="Metadata labels/areas and SHA-256 rank only, frozen before reading meters. Alternate physical buildings within sector; first heldout. Office <=3000 sqm. Up to 24 offices, 12 other sector buildings. Single-building sectors are heldout diagnostics only.",
        selected=selected[["building_id","site_id","sector","split","sqm","timezone"]].to_dict("records"),
        limitations="North American/European 2016-2017 buildings, generally larger than Portuguese fictional sites; electricity meters can exclude district heating/cooling. Building split, not independent site/climate split. Factories are context only: current inventory contains workshops, not factories.")
    dump(PROVENANCE,record)  # Freeze split before requesting meter observations.
    source="data/meters/raw/electricity.csv"
    pointer=get(f"https://raw.githubusercontent.com/buds-lab/building-data-genome-project-2/{BDG_SHA}/"+source).decode()
    expected=next(x.split(":",1)[1] for x in pointer.splitlines() if x.startswith("oid "))
    url=base+source;print("Downloading BDG2 electricity into memory",flush=True);raw=get(url)
    assert sha(raw)==expected
    columns=pd.read_csv(io.BytesIO(raw),nrows=0).columns
    meters=pd.read_csv(io.BytesIO(raw),usecols=[columns[0],*selected.building_id.tolist()])
    meters=meters.rename(columns={columns[0]:"timestamp_local"})
    meters["timestamp_local"]=pd.to_datetime(meters.timestamp_local)
    record["sources"]=[x for x in record["sources"] if x.get("dataset")!="BDG2"]
    for u,data in ((meta_url,raw_meta),(url,raw)):
        record["sources"].append(dict(dataset="BDG2",**source_record(u,data,"CC-BY-SA-4.0")))
    save_reference("bdg2_commercial_metadata.parquet",selected.drop(columns="selection_hash"),record)
    save_reference("bdg2_commercial_hourly.parquet",meters,record)
    record["commercial_units"]="Published hourly electricity kWh per interval; same numeric value as mean kW over one hour. Local civil timestamps; compare at hourly resolution, without invented 15-minute observations."
    dump(PROVENANCE,record);print("BDG2 selected buildings",len(selected),flush=True)


def extract_cycles(unix,power,issues,threshold=10.,gap_seconds=900.):
    """Threshold/gap candidates, not annotated cycles; discard censored/gappy runs."""
    unix=np.asarray(unix,dtype=np.int64);power=np.asarray(power,dtype=float)
    dt=np.r_[np.diff(unix),8.].astype(float)
    good=np.isfinite(power)&(power>=0)&(power<=4000)&(dt>0)&(dt<=120)
    energy=np.where(good,power*np.minimum(dt,16)/3.6e6,0.)
    bad=np.r_[0,np.cumsum(~good)]
    cumulative=np.r_[0,np.cumsum(energy)]
    active=np.flatnonzero(good & (power>threshold))
    if not len(active):return [],dict(active_points=0,accepted=0,rejected=0)
    cuts=np.r_[0,np.flatnonzero(np.diff(unix[active])>gap_seconds)+1,len(active)]
    result=[];rejected=0
    for left,right in zip(cuts[:-1],cuts[1:]):
        a,b=int(active[left]),int(active[right-1]+1)
        duration=float(unix[b-1]+min(dt[b-1],16)-unix[a])
        kwh=float(cumulative[b]-cumulative[a]);peak=float(power[a:b].max())
        # Require observed quiet context on both sides; no partial first/last cycle.
        if (unix[a]-unix[0]<gap_seconds or unix[-1]-unix[b-1]<gap_seconds or
            bad[b]-bad[a] or np.minimum(dt[a:b],16).sum()/duration<.98 or
            duration<600 or duration>21600 or kwh<.02 or peak<100):
            rejected+=1;continue
        slot=((unix[a:b]-unix[a])//900).astype(int)
        profile=np.bincount(slot,weights=energy[a:b]).tolist()
        result.append(dict(start_utc=pd.Timestamp(unix[a],unit="s",tz="UTC"),duration_hours=duration/3600,
            energy_kwh=kwh,peak_kw=peak/1000,profile_15min_kwh=json.dumps(profile),threshold_w=threshold,gap_seconds=gap_seconds,
            issues_flagged_fraction=float(np.mean(np.asarray(issues)[a:b]!=0))))
    return result,dict(active_points=len(active),accepted=len(result),rejected=rejected)


def acquire_appliances():
    import yaml
    record=provenance()
    api=json.loads(get("https://zenodo.org/api/records/5063428"))
    assert api["metadata"]["license"]["id"]=="cc-by-4.0"
    record["appliance_protocol"]=dict(house_split=REFIT_HOUSES,channels=REFIT_CHANNELS,
        metadata_commit=NILMTK_SHA,
        selection="Four complete published house files, selected before power inspection: development houses 3/5, heldout 2/11. Distinct physical households; only washing machine and dishwasher channels.",
        extraction="10 W threshold / 15-minute within-program gaps; sensitivity 5/20 W and 5/30-minute gaps. Discard censored, >120-second-gap, <98% observed duration or implausible cycles. Integrate measured W with observed intervals, cap held samples at 16 seconds. Issues flags the sum of IAMs exceeding aggregate, not a per-channel failure; preserve its fraction for separate quality sensitivity rather than discarding a whole multi-hour cycle for one flagged row. Retain aggregate cycle/day summaries, not raw CSV copies.",
        limits="UK 2013-2015 appliances, not 2024 Portuguese equipment. Cleaned REFIT filled short gaps and zeroed long gaps upstream; unflagged zero gaps cannot be recovered. Detected candidates are not ground-truth program labels or measured user deadlines.")
    dump(PROVENANCE,record)
    existing=record.setdefault("refit_house_evidence",{})
    for house,split in REFIT_HOUSES.items():
        cycle_path=REFERENCE/f"refit_house_{house}_cycles.parquet"
        if str(house) in existing and existing[str(house)].get("extraction_revision")==2 and cycle_path.exists() and sha(cycle_path.read_bytes())==record.get("derived_sha256",{}).get(cycle_path.name):
            print("REFIT already verified",house,flush=True);continue
        url=f"https://raw.githubusercontent.com/nilmtk/nilmtk/{NILMTK_SHA}/nilmtk/dataset_converters/refit/metadata/building{house}.yaml"
        metadata=get(url);mapping=yaml.safe_load(metadata)
        for kind,channel in REFIT_CHANNELS[house].items():
            label="washing machine" if kind=="washing_machine" else "dish washer"
            assert any(a["type"]==label and a["meters"]==[channel+1] for a in mapping["appliances"])
        file=next(x for x in api["files"] if x["key"]==f"CLEAN_House{house}.csv")
        print("Downloading REFIT house",house,"into memory",flush=True)
        raw=get_large(file["links"]["self"],file["size"])
        assert len(raw)==file["size"] and "md5:"+hashlib.md5(raw).hexdigest()==file["checksum"]
        digest=sha(raw)
        f=pd.read_csv(io.BytesIO(raw),usecols=["Unix","Issues",*[f"Appliance{x}" for x in REFIT_CHANNELS[house].values()]])
        del raw
        assert not f.Unix.duplicated().any() and f.Unix.is_monotonic_increasing
        rows=[];quality=[];daily=[]
        for kind,channel in REFIT_CHANNELS[house].items():
            power=f[f"Appliance{channel}"].to_numpy()
            for threshold,gap in ((10,900),(5,900),(20,900),(10,300),(10,1800)):
                cycles,q=extract_cycles(f.Unix.to_numpy(),power,f.Issues.to_numpy(),threshold,gap)
                for row in cycles:row.update(house=house,split=split,appliance_type=kind)
                rows.extend(cycles);quality.append(dict(appliance_type=kind,threshold_w=threshold,gap_seconds=gap,**q))
            dt=np.r_[np.diff(f.Unix),8.];good=(dt>0)&(dt<=120)&(power>=0)&(power<=4000)
            days=pd.DataFrame(dict(date=pd.to_datetime(f.Unix,unit="s",utc=True).dt.tz_convert("Europe/London").dt.date,
                kwh=np.where(good,power*np.minimum(dt,16)/3.6e6,0.),observed_seconds=np.where(good,np.minimum(dt,16),0.),
                issue_flagged_seconds=np.where(f.Issues.ne(0),np.minimum(dt,16),0.)))
            days=days.groupby("date",as_index=False).sum();days["date"]=days.date.astype(str)
            days["house"]=house;days["split"]=split;days["appliance_type"]=kind;daily.append(days)
        save_reference(cycle_path.name,pd.DataFrame(rows),record)
        save_reference(f"refit_house_{house}_days.parquet",pd.concat(daily,ignore_index=True),record)
        existing[str(house)]=dict(filename=file["key"],url=file["links"]["self"],bytes=file["size"],md5=file["checksum"],sha256=digest,license="CC-BY-4.0",metadata_url=url,metadata_sha256=sha(metadata),raw_rows=len(f),quality=quality,extraction_revision=2)
        dump(PROVENANCE,record);print("REFIT",house,"cycles incl sensitivity",len(rows),flush=True)


def acquire_kitchens():
    """Portuguese professional kitchens: external structural diagnostic only.

    Freeze this role before reading power. No fitted appliance cycle labels:
    heater bursts and enabled service blocks are different from user jobs.
    """
    record=provenance()
    record["kitchen_protocol"]=dict(role="External structural diagnostic only; no numerical fitting or population representativeness claim. Three Portuguese professional kitchens for 3-4 weeks. One dishwasher and one glasswasher in kitchen 1: not independent-device heldout validation.",
        integration="Published active watts, UTC timestamps converted to Europe/Lisbon for civil comparisons. Piecewise constant integration, held intervals bounded at twice the median sampling interval; retain observed seconds per 15-minute bin. No zero-fill/interpolation of absent samples.",
        limitations="Kitchen aggregate is not whole-cafe electricity; unmonitored circuits are not classified as non-shiftable. Heater bursts are not annotated washing cycles. Professional machines differ in size/rating from the fictional cafe.")
    dump(PROVENANCE,record)
    files={"kitchen_1":("652445f79907550056a1d414","https://osf.io/download/b9x84/"),
           "kitchen_2":("652447819907550051a1d807","https://osf.io/download/bgyvf/"),
           "kitchen_3":("6524478b7bfe18005f5bb066","https://osf.io/download/3ahsq/"),
           "dishwasher":("65252ff87bfe1801015bb43c","https://osf.io/download/dmj46/"),
           "glasswasher":("6525302c7bfe1801015bb457","https://osf.io/download/x3wpd/")}
    evidence=record.setdefault("kitchen_evidence",{})
    for name,(fid,url) in files.items():
        target=REFERENCE/f"fik_{name}_15min.parquet"
        if name in evidence and target.exists() and sha(target.read_bytes())==record["derived_sha256"].get(target.name):continue
        meta=json.loads(get(f"https://api.osf.io/v2/files/{fid}/"))["data"]["attributes"]
        print("Downloading FIK",name,flush=True)
        raw=get_large(url,meta["size"])
        digest=sha(raw);assert digest==meta["extra"]["hashes"]["sha256"]
        columns=pd.read_csv(io.BytesIO(raw),nrows=0).columns
        time_column=next(c for c in columns if c.lower()=="timestamp")
        f=pd.read_csv(io.BytesIO(raw),usecols=[time_column,"P_Total"]).rename(columns={time_column:"Timestamp"});del raw
        t=pd.to_datetime(f.Timestamp,utc=True).astype("int64").to_numpy()/1e9
        assert (np.diff(t)>0).all()
        median=float(np.median(np.diff(t)));duration=np.minimum(np.r_[np.diff(t),median],2*median)
        power=f.P_Total.to_numpy(dtype=float)
        good=np.isfinite(power)&(power>=0)
        duration=np.where(good,duration,0.);power=np.where(good,power,0.)
        origin=int(t[0]//900)*900;slot=((t-origin)//900).astype(int)
        first=np.minimum(duration,origin+(slot+1)*900-t);rest=duration-first
        size=int(slot[-1])+2
        energy=np.bincount(slot,weights=power*first/3.6e6,minlength=size)+np.bincount(slot+1,weights=power*rest/3.6e6,minlength=size)
        coverage=np.bincount(slot,weights=first,minlength=size)+np.bincount(slot+1,weights=rest,minlength=size)
        out=pd.DataFrame(dict(timestamp_utc=pd.to_datetime(origin+np.arange(size)*900,unit="s",utc=True),energy_kwh=energy,observed_seconds=coverage))
        save_reference(target.name,out,record)
        evidence[name]=dict(url=url,osf_file_id=fid,bytes=meta["size"],sha256=digest,license="CC-BY-4.0",rows=len(f),median_sample_seconds=median,
            positive_power_kw_quantiles=np.quantile(power[power>100],[.1,.5,.9,.99]).tolist() if (power>100).any() else [])
        # Correct units of the diagnostic quantiles; original columns are watts.
        evidence[name]["positive_power_kw_quantiles"]=[x/1000 for x in evidence[name]["positive_power_kw_quantiles"]]
        dump(PROVENANCE,record);print("FIK compact reference saved",name,len(out),flush=True)


def hourly_days(times,kw,timezone="Europe/Lisbon",measured=False):
    times=pd.DatetimeIndex(times)
    if times.tz is not None:times=times.tz_convert(timezone).tz_localize(None)
    values=np.asarray(kw,dtype=float)
    f=pd.DataFrame(dict(date=times.normalize(),hour=times.hour,kw=values))
    f.loc[~np.isfinite(values)|(values<0),"kw"]=np.nan
    # Follow the source's electricity-zero caution, without interpolation.
    if measured:f.loc[f.kw.eq(0),"kw"]=np.nan
    ambiguous=times.tz_localize(timezone,ambiguous="NaT",nonexistent="NaT").isna()
    bad=set(times[ambiguous].normalize())
    count=f.groupby("date").agg(rows=("kw","size"),valid=("kw","count"),hours=("hour","nunique"))
    valid=count.index[(count.rows==24)&(count.valid==24)&(count.hours==24)]
    valid=valid[~valid.isin(bad)]
    matrix=f.loc[f.date.isin(valid)].pivot(index="date",columns="hour",values="kw").sort_index()
    return matrix.loc[matrix.sum(axis=1)>0]


def commercial_metrics(matrix,sqm):
    if len(matrix)<60:return dict(valid=False,complete_days=len(matrix))
    values=matrix.to_numpy();dates=matrix.index;mean=values.mean()
    energy=pd.Series(values.sum(axis=1),index=dates)
    monthly=energy.groupby([dates.year,dates.month]).transform("mean")
    conditional=energy.groupby([dates.year,dates.month,dates.dayofweek]).transform("mean")
    residual=np.log(energy/conditional)
    lag=residual.reindex(dates-pd.Timedelta(days=1)).to_numpy()
    good=np.isfinite(lag)
    normalized=values/values.mean(axis=1)[:,None]
    typical=pd.DataFrame(normalized,index=dates).groupby(dates.dayofweek).transform("mean").to_numpy()
    previous=matrix.reindex(dates-pd.Timedelta(days=7)).to_numpy();paired=np.isfinite(previous).all(axis=1)
    return dict(valid=True,complete_days=len(matrix),mean_day_kwh=float(energy.mean()),
        annualized_electricity_kwh_m2=float(energy.mean()*365/sqm),
        within_month_day_cv=float((energy/monthly).std()),conditional_daily_log_sigma=float(residual.std()),
        conditional_daily_lag1=float(np.corrcoef(residual.to_numpy()[good],lag[good])[0,1]),
        weekday_night_day_ratio=float(values[dates.dayofweek<5,:6].mean()/values[dates.dayofweek<5,9:18].mean()),
        weekend_weekday_ratio=float(energy[dates.dayofweek>=5].mean()/energy[dates.dayofweek<5].mean()),
        hourly_ramp_p95_to_mean=float(np.quantile(np.abs(np.diff(values,axis=1)),.95)/mean),
        p99_to_mean=float(np.quantile(values,.99)/mean),daily_shape_deviation=float(np.abs(normalized-typical).mean()),
        lag7_day_correlation=float(np.corrcoef(values[paired].ravel(),previous[paired].ravel())[0,1]),
        peak_slot_mode_share=float(pd.Series(values.argmax(axis=1)).value_counts(normalize=True).max()),
        identical_daily_shapes=int(pd.DataFrame(np.round(normalized,6)).duplicated().sum()))


def measured_commercial(split=None):
    metadata=pd.read_parquet(REFERENCE/"bdg2_commercial_metadata.parquet")
    if split:metadata=metadata.loc[metadata.split.eq(split)]
    source=pd.read_parquet(REFERENCE/"bdg2_commercial_hourly.parquet",columns=["timestamp_local",*metadata.building_id])
    rows=[]
    for m in metadata.itertuples():
        matrix=hourly_days(source.timestamp_local,source[m.building_id],m.timezone,True)
        rows.append(dict(building_id=m.building_id,site_id=m.site_id,sector=m.sector,split=m.split,
            sqm=m.sqm,source_hours=len(source),covered_calendar_months=int(matrix.index.month.nunique()),
            usable_days_by_year_month={str(month):int(count) for month,count in pd.Series(1,index=matrix.index).groupby(matrix.index.to_period("M")).sum().items()},
            **commercial_metrics(matrix,m.sqm)))
    return rows


def synthetic_commercial():
    from portugal_empirical_validation import synthetic_household
    rows=[]
    for root in sorted((ROOT/"data/datasets").glob("rec_pt_2023_*_v2")):
        members=pd.read_parquet(root/"catalogs/members.parquet")
        clock=pd.DatetimeIndex(pd.read_parquet(root/"shared/calendar.parquet").timestamp_utc)
        for m in members.loc[members.category.ne("residential")].itertuples():
            power=synthetic_household(root,m.member_id)
            hourly=pd.Series(power,index=clock).resample("h").mean()
            for year in (2023,2024):
                selected=hourly.loc[hourly.index.year==year]
                rows.append(dict(family=root.name,member_id=m.member_id,sector=m.routine_archetype,year=year,sqm=m.floor_area_m2,
                    **commercial_metrics(hourly_days(selected.index,selected.to_numpy()),m.floor_area_m2)))
    return rows


def synthetic_appliances():
    rows=[]
    for root in sorted((ROOT/"data/datasets").glob("rec_pt_2023_*_v2")):
        apps=pd.read_parquet(root/"catalogs/deferrables.parquet")
        for a in apps.itertuples():
            profiles=pd.read_parquet(root/f"deferrables/{a.deferrable_id}_profiles.parquet").set_index("profile_id")
            schedule=pd.read_parquet(root/f"deferrables/{a.deferrable_id}_schedule.parquet")
            for year,g in schedule.groupby("year"):
                used=profiles.loc[g.profile_id]
                energies=used.total_energy_kwh.to_numpy();durations=used.duration_steps.to_numpy()*.25
                rows.append(dict(family=root.name,appliance_id=a.deferrable_id,member_id=a.member_id,appliance_type=a.appliance_type,
                    equipment_use=a.equipment_use,controlled=bool(a.controlled),year=int(year),cycles=len(g),
                    weekly_frequency=float(len(g)*7/(365 if year==2023 else 366)),
                    energy_quantiles=np.quantile(energies,[.1,.5,.9]).tolist(),duration_quantiles=np.quantile(durations,[.1,.5,.9]).tolist(),
                    peak_kw_quantiles=np.quantile([max(json.loads(p))*4 for p in used.load_profile],[.1,.5,.9]).tolist(),
                    mode_duration_share=float(pd.Series(durations).value_counts(normalize=True).max()),
                    unique_energy_values=len(np.unique(energies)),unique_profile_share=float(g.profile_id.nunique()/len(g))))
    return rows


def measured_appliances(split=None):
    rows=[]
    for house,group in REFIT_HOUSES.items():
        if split and group!=split:continue
        cycles=pd.read_parquet(REFERENCE/f"refit_house_{house}_cycles.parquet")
        days=pd.read_parquet(REFERENCE/f"refit_house_{house}_days.parquet")
        cycles["date"]=cycles.start_utc.dt.tz_convert("Europe/London").dt.strftime("%Y-%m-%d")
        for (kind,threshold,gap),g in cycles.groupby(["appliance_type","threshold_w","gap_seconds"]):
            # A denominator of adequately observed dates, including zero-use
            # dates. DST days are excluded rather than treating them as 24 h.
            valid=days.loc[days.appliance_type.eq(kind)&days.observed_seconds.ge(.9*86400),"date"]
            dates=pd.DatetimeIndex(valid).tz_localize("Europe/London")
            valid=valid.loc[(dates+pd.DateOffset(days=1)-dates).total_seconds()==86400]
            g=g.loc[g.date.isin(valid)]
            for quality in ("all_candidates","issues_fraction_le_0.1"):
                selected=g if quality=="all_candidates" else g.loc[g.issues_flagged_fraction.le(.1)]
                if not len(selected):continue
                rows.append(dict(house=house,split=group,appliance_type=kind,threshold_w=float(threshold),gap_seconds=float(gap),
                    quality=quality,cycles=len(selected),adequately_observed_days=len(valid),
                    detected_cycles_per_week=float(len(selected)*7/len(valid)),
                    energy_quantiles=selected.energy_kwh.quantile([.1,.5,.9]).tolist(),
                    duration_quantiles=selected.duration_hours.quantile([.1,.5,.9]).tolist(),
                    peak_kw_quantiles=selected.peak_kw.quantile([.1,.5,.9]).tolist(),
                    energy_duration_spearman=float(selected.energy_kwh.corr(selected.duration_hours,method="spearman")),
                    issues_flagged_median_fraction=float(selected.issues_flagged_fraction.median())))
    return rows


def kitchen_diagnostics():
    rows=[]
    for name,meta in provenance()["kitchen_evidence"].items():
        f=pd.read_parquet(REFERENCE/f"fik_{name}_15min.parquet")
        local=f.timestamp_utc.dt.tz_convert("Europe/Lisbon")
        f["date"]=local.dt.strftime("%Y-%m-%d");f["hour"]=local.dt.hour
        f["usable"]=f.observed_seconds.ge(.8*900)
        days=f.groupby("date").agg(slots=("usable","size"),usable=("usable","sum"),energy=("energy_kwh","sum"))
        valid=days.index[(days.slots==96)&(days.usable>=.9*96)]
        g=f.loc[f.date.isin(valid)&f.usable]
        power=g.energy_kwh*3600/g.observed_seconds
        active=power>.1
        day_energy=days.loc[valid,"energy"]
        rows.append(dict(name=name,complete_enough_days=len(valid),usable_15min_fraction=float(f.usable.mean()),
            daily_kwh_quantiles=day_energy.quantile([.1,.5,.9]).tolist(),
            observed_mean_kw_quantiles=power.quantile([.1,.5,.9,.99]).tolist(),
            active_hours_per_observed_day=float(active.sum()*.25/max(1,len(valid))),
            night_day_ratio=float(power[g.hour.lt(6)].mean()/power[g.hour.between(9,18)].mean()),
            scope="Observed-duration normalized powers and >=80%-covered bins; days require >=90% usable bins. Not extrapolated annual demand, annotated cycles or national statistics."))
    return rows


def inventory_audit():
    rows=[]
    for root in sorted((ROOT/"data/datasets").glob("rec_pt_2023_*_v2")):
        members=pd.read_parquet(root/"catalogs/members.parquet")
        chargers=pd.read_parquet(root/"catalogs/chargers.parquet")
        sessions=pd.read_parquet(root/"catalogs/charging_sessions.parquet")
        schema=json.loads((root/"schema.json").read_text())
        for m in members.loc[members.category.ne("residential")].itertuples():
            opened=pd.read_parquet(root/f"operations/{m.member_id}.parquet").is_open.to_numpy()
            f=pd.read_parquet(root/f"timeseries/{m.member_id}.parquet")
            ch=chargers.loc[chargers.member_id.eq(m.member_id)]
            visits=sessions.loc[sessions.member_id.eq(m.member_id)]
            occupied=np.zeros(len(opened),dtype=int)
            outside=0
            for s in visits.itertuples():
                a,b=int(s.arrival_time_step),int(s.observed_end_time_step)
                outside+=int((~opened[a:b]).sum());occupied[a:b]+=1
            assert len(ch)==m.physical_charger_count and len(ch)<=m.parking_spaces
            assert outside==0 and occupied.max()<=len(ch)
            limit=schema["buildings"][m.member_id]["electrical_service"]["limits"]["total"]["import_kw"]
            assert f.non_shiftable_load.max()*4<=limit+1e-5
            rows.append(dict(family=root.name,member_id=m.member_id,sector=m.routine_archetype,sqm=m.floor_area_m2,
                parking_spaces=int(m.parking_spaces),chargers=len(ch),charger_to_parking_ratio=float(len(ch)/m.parking_spaces),
                rated_charger_sum_kw=float(ch.max_charging_power_kw.sum()),service_import_kw=schema["buildings"][m.member_id]["electrical_service"]["limits"]["total"]["import_kw"],
                open_hours_2024=float(opened[35040:].sum()*.25),sessions_2024=int(visits.year.eq(2024).sum()),
                occupied_open_port_fraction=float(occupied[35040:].sum()/max(1,opened[35040:].sum()*len(ch))),
                max_simultaneous_occupied_ports=int(occupied.max()),visits_outside_opening_slots=outside,
                maximum_expected_people=float(f.occupant_count.max()),
                nonshiftable_peak_kw=float(f.non_shiftable_load.max()*4),
                closed_energy_share=float(f.non_shiftable_load[~opened].sum()/f.non_shiftable_load.sum())))
    return rows


def freeze_protocol():
    from portugal_activity import CONFIG
    report=json.loads(OUTPUT.read_text())
    assert "evaluation_protocol" not in report,"Choices already frozen; record any later revision explicitly"
    report["evaluation_protocol"]=dict(configuration=CONFIG,
        development_appliances=measured_appliances("development"),
        commercial_adjustment="Development BDG2 sector median hourly anchors. Energy: geometric half-transfer from the original EUI; shape: 0.65 linear transfer of night/day ratio into background prior. Neither weight is estimated from Portugal; sensitivity includes energy transfer 0/1 and shape 0.4/0.9. Sector activity peaks, attendance density and sub-hour fluctuations are explicit priors. Grocery has no development meter.",
        appliance_adjustment="House-balanced whole detected REFIT candidates from houses 3/5, jointly retaining energy/duration/shape, 48 variants per domestic/staff machine; bounded small energy perturbation. Installed nominal rating respected without deleting energy. Frequency/deadlines remain household inventory priors, professional laundry remains an unvalidated equipment prior.",
        evaluation="Hold out BDG2 physical buildings and REFIT houses 2/11. Compare all sectors and both years, by building/house rather than pooling every sample. Report every discrepancy; no post-hoc fit or pass thresholds based on heldout observations.",
        sources_sha256=sha(PROVENANCE.read_bytes()))
    dump(OUTPUT,report)


def comparison_summary(report):
    commercial={}
    for label,rows in (("before",report["before"]["commercial"]),("after",report["after"]["commercial"]),("heldout",report["commercial_heldout"])):
        frame=pd.DataFrame(rows)
        if "year" in frame:frame=frame.loc[frame.year.eq(2024)]
        metrics=["annualized_electricity_kwh_m2","weekday_night_day_ratio","weekend_weekday_ratio","hourly_ramp_p95_to_mean","daily_shape_deviation"]
        commercial[label]=frame.loc[frame.valid].groupby("sector")[metrics].median().to_dict("index")
    appliances={}
    for label in ("before","after"):
        frame=pd.DataFrame(report[label]["appliances"]);frame=frame.loc[frame.year.eq(2024)].copy()
        frame["energy_kwh"]=frame.energy_quantiles.map(lambda q:q[1]);frame["duration_hours"]=frame.duration_quantiles.map(lambda q:q[1])
        appliances[label]=frame.groupby("equipment_use")[["energy_kwh","duration_hours","weekly_frequency","mode_duration_share"]].median().to_dict("index")
    frame=pd.DataFrame(report["appliance_measured"])
    primary=frame.loc[frame.split.eq("heldout")&frame.quality.eq("all_candidates")&frame.threshold_w.eq(10)&frame.gap_seconds.eq(900)].copy()
    primary["energy_kwh"]=primary.energy_quantiles.map(lambda q:q[1]);primary["duration_hours"]=primary.duration_quantiles.map(lambda q:q[1])
    appliances["heldout"]=primary.groupby("appliance_type")[["energy_kwh","duration_hours","detected_cycles_per_week"]].median().to_dict("index")
    sensitivity=[]
    for (house,kind),g in frame.groupby(["house","appliance_type"]):
        g=g.loc[g.quality.eq("all_candidates")]
        energies=np.array([q[1] for q in g.energy_quantiles]);duration=np.array([q[1] for q in g.duration_quantiles])
        sensitivity.append(dict(house=int(house),appliance_type=kind,detector_settings=len(g),
            median_energy_range_kwh=[float(energies.min()),float(energies.max())],
            median_duration_range_hours=[float(duration.min()),float(duration.max())],
            cycle_count_range=[int(g.cycles.min()),int(g.cycles.max())]))
    return dict(commercial_2024= commercial,appliances_2024=appliances,detector_sensitivity=sensitivity,
        interpretation="Medians of building/appliance summaries; not pooled-time estimates, population confidence intervals or a percent-realism score. Heldout commercial EUI remains discrepant; individual appliances have heterogeneous use frequency and detector sensitivity. No heldout retuning.")


def commercial_coverage_sensitivity(rows):
    """Post-hoc coverage diagnostic; never reselect development anchors."""
    f=pd.DataFrame(rows);comparisons=[]
    metrics=["annualized_electricity_kwh_m2","weekday_night_day_ratio","hourly_ramp_p95_to_mean"]
    for (split,sector),group in f.loc[f.valid].groupby(["split","sector"]):
        covered=group.loc[group.complete_days.ge(180)&group.covered_calendar_months.ge(10)]
        comparisons.append(dict(split=split,sector=sector,primary_buildings=len(group),covered_buildings=len(covered),
            primary_medians=group[metrics].median().to_dict(),covered_medians=covered[metrics].median().to_dict() if len(covered) else None,
            excluded_buildings=group.loc[~group.building_id.isin(covered.building_id),"building_id"].tolist()))
    return dict(method="Post-hoc sensitivity: >=180 complete days and >=10 distinct calendar months across the two years. Original split, primary comparison and frozen development anchors retained; no retuning.",
        invalid_buildings=f.loc[~f.valid,["building_id","sector","split","complete_days"]].to_dict("records"),comparisons=comparisons)


def final_comparison():
    from audit_portugal_heldout_2024 import runtime_fingerprint
    from portugal_activity import CONFIG
    report=json.loads(OUTPUT.read_text());assert report["evaluation_protocol"]["configuration"]==CONFIG
    fingerprint=runtime_fingerprint()
    for name,digest in provenance()["derived_sha256"].items():assert sha((REFERENCE/name).read_bytes())==digest
    report.update(commercial_development=measured_commercial("development"),commercial_heldout=measured_commercial("heldout"),appliance_measured=measured_appliances(),
        kitchen_diagnostics=kitchen_diagnostics(),after=dict(commercial=synthetic_commercial(),appliances=synthetic_appliances(),inventory=inventory_audit()))
    report["commercial_coverage_sensitivity"]=commercial_coverage_sensitivity(report["commercial_development"]+report["commercial_heldout"])
    report["runtime_input_fingerprint"]=fingerprint
    report["generator_sha256"]={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in [ROOT/"scripts"/n for n in
        ("portugal_activity.py","portugal_commercial_activity.py","portugal_stochastic_demand.py","portugal_realism_config.json","revise_portugal_realism.py","revise_portugal_ev_mobility.py")]}
    report["source_manifest_sha256"]=sha(PROVENANCE.read_bytes())
    report["audit_sha256"]=sha(Path(__file__).read_bytes())
    report["summary"]=comparison_summary(report)
    report["portuguese_revision"]=portuguese_diagnostics(report)
    report["status"]="completed_comparison_with_reported_discrepancies"
    assert runtime_fingerprint()==fingerprint
    json.dumps(report,allow_nan=False)
    dump(OUTPUT,report)
    for title,rows in (("before",report["before"]["commercial"]),("after",report["after"]["commercial"]),("heldout",report["commercial_heldout"])):
        f=pd.DataFrame(rows)
        if "year" in f:f=f.loc[f.year.eq(2024)]
        print(title,f.loc[f.valid].groupby("sector")[["annualized_electricity_kwh_m2","weekday_night_day_ratio","weekend_weekday_ratio","hourly_ramp_p95_to_mean","daily_shape_deviation"]].median().round(3).to_string())


def portuguese_diagnostics(report):
    """Compare to the immediately preceding revision, with explicit source roles."""
    from portugal_activity import CONFIG
    before=pd.DataFrame(report["before_portuguese_revision"]["commercial"])
    after=pd.DataFrame(report["after"]["commercial"])
    summaries=[]
    for kind,g in after.loc[after.year.eq(2024)].groupby("sector"):
        old=before.loc[before.year.eq(2024)&before.sector.eq(kind)]
        anchor=CONFIG["commercial"]["portuguese_annual_anchors"].get(kind)
        summaries.append(dict(sector=kind,sites=len(g),before_eui=float(old.annualized_electricity_kwh_m2.median()),
            after_eui=float(g.annualized_electricity_kwh_m2.median()),reference=anchor,
            after_eui_range=[float(g.annualized_electricity_kwh_m2.min()),float(g.annualized_electricity_kwh_m2.max())]))
    people=[];budgets=[];professional=[]
    for root in sorted((ROOT/"data/datasets").glob("rec_pt_2023_*_v2")):
        members=pd.read_parquet(root/"catalogs/members.parquet")
        apps=pd.read_parquet(root/"catalogs/deferrables.parquet")
        totals=pd.read_parquet(root/"catalogs/member_year_totals.parquet")
        for m in members.loc[members.category.ne("residential")].itertuples():
            ops=pd.read_parquet(root/f"operations/{m.member_id}.parquet")
            f=pd.read_parquet(root/f"timeseries/{m.member_id}.parquet")
            components=ops[["staff_count","client_count","resident_count"]].sum(axis=1)
            assert np.array_equal(components.to_numpy(),f.occupant_count.to_numpy())
            assert ops.client_count.loc[~ops.is_open].eq(0).all()
            g=ops.iloc[35040:];opened=g.loc[g.is_open]
            people.append(dict(family=root.name,member_id=m.member_id,sector=m.routine_archetype,
                peak_people=int(components.max()),open_mean_staff=float(opened.staff_count.mean()),
                open_mean_clients=float(opened.client_count.mean()),min_residents=int(g.resident_count.min()),
                clients_outside_hours=int(ops.client_count.loc[~ops.is_open].sum())))
            for year,lo,hi in ((2023,0,35040),(2024,35040,70176)):
                energy=float(f.non_shiftable_load.iloc[lo:hi].astype(float).sum())
                controlled=0.
                for app in apps.loc[apps.member_id.eq(m.member_id)&apps.controlled].itertuples():
                    p=pd.read_parquet(root/f"deferrables/{app.deferrable_id}_profiles.parquet").set_index("profile_id")
                    jobs=pd.read_parquet(root/f"deferrables/{app.deferrable_id}_schedule.parquet")
                    controlled+=jobs.loc[jobs.year.eq(year),"profile_id"].map(p.total_energy_kwh).sum()
                target=totals.loc[totals.member_id.eq(m.member_id)&totals.year.eq(year),"building_budget_excluding_ev_kwh"].item()
                residual=energy+controlled-target;assert abs(residual)<.01
                budgets.append(dict(family=root.name,member_id=m.member_id,year=year,total_building_kwh=energy+controlled,
                    controlled_appliances_kwh=controlled,target_kwh=target,residual_kwh=residual))
        for app in apps.loc[apps.equipment_use.isin(["linen_laundry","catering_dishwashing_batch"])].itertuples():
            p=pd.read_parquet(root/f"deferrables/{app.deferrable_id}_profiles.parquet").set_index("profile_id")
            jobs=pd.read_parquet(root/f"deferrables/{app.deferrable_id}_schedule.parquet")
            selected=p.loc[jobs.loc[jobs.year.eq(2024),"profile_id"]]
            professional.append(dict(family=root.name,member_id=app.member_id,equipment_use=app.equipment_use,
                cycles_2024=len(selected),median_energy_kwh=float(selected.total_energy_kwh.median()),
                median_duration_hours=float(selected.duration_steps.median()*.25),
                processed_linen_kg=float(selected.load_kg.sum()) if "load_kg" in selected else None))
    return dict(status="pass",annual_comparison=summaries,occupancy=people,building_budgets=budgets,professional=professional,
        inference_limit="Published Portuguese summaries inform scenario assumptions, not independent population validation. No measured commercial occupancy or professional workload series were fitted. Clinics, workshops, sports centres and care homes still lack comparable Portuguese annual anchors. Grocery reference sites are much smaller; generic health/beauty is not a clinic and a print shop is not an industrial factory.")


def commercial_seed_sensitivity():
    from copy import deepcopy
    from portugal_activity import CONFIG
    from portugal_commercial_activity import commercial_end_uses
    from portugal_stochastic_demand import bounded_energy_allocation
    from revise_portugal_ev_mobility import source_frame
    from revise_portugal_realism import profiles_for
    from audit_portugal_heldout_2024 import runtime_fingerprint
    fingerprint=runtime_fingerprint();baseline=deepcopy(CONFIG)
    root=ROOT/"data/datasets/rec_pt_2023_premium_100_v2"
    members=pd.read_parquet(root/"catalogs/members.parquet")
    members=members.loc[members.category.ne("residential")].groupby("routine_archetype",group_keys=False).head(1)
    calendar=pd.read_parquet(root/"shared/calendar.parquet");clock=pd.DatetimeIndex(calendar.timestamp_utc)
    schema=json.loads((root/"schema.json").read_text())
    cases=[(m,source_frame(root/f"end_uses/{m.member_id}.parquet").astype(float)) for m in members.itertuples()]
    apps=pd.read_parquet(root/"catalogs/deferrables.parquet").groupby("equipment_use",group_keys=False).head(1)
    seeds=[baseline["seed"]+1009*(i+1) for i in range(20)]
    result=dict(status="running",seeds=seeds,draws=[],sensitivity=[],appliance_program_draws=[],
        scope="20 seeds x 8 commercial sectors, two-year exogenous commercial end-use generator, physical normal-service envelopes; no native simulation. 72 paired OAT configurations x 8 sectors, including Portuguese transfer weights and attendance. Separate 20-seed checks for each appliance use class; full coupled appliance/EV communities are in ensemble_validation.json.")
    def draw(seed,allow_infeasible=False):
        CONFIG["seed"]=seed;rows=[]
        for m,original in cases:
            f,people,days,ops=commercial_end_uses(root,m,calendar,original)
            cap=schema["buildings"][m.member_id]["electrical_service"]["limits"]["total"]["import_kw"]*.25
            other=f.drop(columns="base_excluding_cycles_kwh").sum(axis=1).to_numpy()
            assert (other<=cap).all()
            impossible=[dict(year=year,base_requested_kwh=float(f.base_excluding_cycles_kwh.iloc[lo:hi].sum()),
                base_available_kwh=float((cap-other[lo:hi]).sum()))
                for year,lo,hi in ((2023,0,35040),(2024,35040,70176))
                if f.base_excluding_cycles_kwh.iloc[lo:hi].sum()>(cap-other[lo:hi]).sum()+1e-8]
            if impossible:
                assert allow_infeasible,(m.member_id,"Default seed exceeds the physical annual envelope",impossible)
                rows.append(dict(member_id=m.member_id,sector=m.routine_archetype,
                    status="infeasible_annual_budget",years=impossible))
                continue
            for lo,hi in ((0,35040),(35040,70176)):
                values=f.base_excluding_cycles_kwh.iloc[lo:hi].to_numpy()
                f.loc[lo:hi-1,"base_excluding_cycles_kwh"]=bounded_energy_allocation(values,cap-other[lo:hi],values.sum())
            values=f.sum(axis=1).to_numpy()*4
            assert np.isfinite(values).all() and values.min()>=0 and values.max()<=4*cap+1e-6
            assert np.isfinite(people).all() and people.max()>0
            assert len(days)==731 and not days.date.duplicated().any()
            series=pd.Series(values[35040:],index=clock[35040:]).resample("h").mean()
            metrics=commercial_metrics(hourly_days(series.index,series.to_numpy()),m.floor_area_m2)
            repeated=metrics["identical_daily_shapes"]>0
            assert allow_infeasible or not repeated,(m.member_id,"Repeated default daily shapes",metrics)
            rows.append(dict(member_id=m.member_id,sector=m.routine_archetype,
                status="rejected_service_saturated_profile" if repeated else "feasible",
                fraction_at_service_limit=float(np.mean(values>=cap*4-1e-5)),**metrics))
        return dict(seed=seed,sectors=rows)
    try:
        for seed in seeds:
            result["draws"].append(draw(seed))
            for app in apps.itertuples():
                p=profiles_for(root,app)
                for row in p.itertuples():
                    values=np.array(json.loads(row.load_profile))
                    assert (values>=0).all() and values.max()*4<=app.nominal_power_kw+1e-8
                    assert abs(values.sum()-row.total_energy_kwh)<1e-9
                result["appliance_program_draws"].append(dict(seed=seed,equipment_use=app.equipment_use,
                    median_energy_kwh=float(p.total_energy_kwh.median()),median_duration_hours=float(p.duration_steps.median()*.25),
                    distinct_durations=int(p.duration_steps.nunique()),sha256=sha(pd.util.hash_pandas_object(p,index=False).values.tobytes())))
            print("commercial/appliance seed",seed,"pass",flush=True)
        settings={"energy_transfer_weight":(0.,1.),"shape_transfer_weight":(.4,.9),
            "daily_sigma_multiplier":(.75,1.25),"timing_jitter_hours":(.375,.625),"intraday_log_sigma":(.09,.15),
            "office_between_building_log_sigma":(.125,.5),"presence.attendance_mean":(.7,.95),
            "presence.load_coupling_weight":(0.,.6)}
        settings.update({f"portuguese_annual_anchors.{kind}.transfer_weight":(0.,1.)
            for kind in ("office","retail","cafe","grocery")})
        for parameter,values in settings.items():
            for value in values:
                CONFIG.clear();CONFIG.update(deepcopy(baseline))
                container=CONFIG["commercial"];parts=parameter.split(".")
                for part in parts[:-1]:container=container[part]
                container[parts[-1]]=value
                for seed in seeds[:3]:
                    row=draw(seed,allow_infeasible=True);row.update(parameter=parameter,value=value);result["sensitivity"].append(row)
                print("commercial sensitivity",parameter,value,"pass",flush=True)
        result["status"]="pass"
        result["infeasible_sensitivity_cases"]=[dict(parameter=r["parameter"],value=r["value"],seed=r["seed"],**s)
            for r in result["sensitivity"] for s in r["sectors"] if s["status"]=="infeasible_annual_budget"]
        result["rejected_saturated_sensitivity_cases"]=[dict(parameter=r["parameter"],value=r["value"],seed=r["seed"],**s)
            for r in result["sensitivity"] for s in r["sectors"] if s["status"]=="rejected_service_saturated_profile"]
        result["infeasibility_policy"]="Default seeds must be feasible and have no exact daily copies. Extreme sensitivity settings may exceed annual capacity or collapse daily shapes at the service ceiling: both are reported as rejected settings, never claimed as accepted realistic generations. No hardware changes."
        assert runtime_fingerprint()==fingerprint
    finally:
        CONFIG.clear();CONFIG.update(baseline)
        report=json.loads(OUTPUT.read_text());report["seed_sensitivity"]=result;dump(OUTPUT,report)


def make_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report=json.loads(OUTPUT.read_text())
    if "portuguese_revision" in report:
        fig,axes=plt.subplots(2,2,figsize=(14,9),layout="constrained")
        rows=report["portuguese_revision"]["annual_comparison"]
        labels={"office":"Escritório","retail":"Loja","cafe":"Café","grocery":"Mercearia",
            "clinic":"Clínica","workshop":"Oficina","sports_centre":"Desporto","care_home":"Lar"}
        x=np.arange(len(rows));ax=axes[0,0]
        ax.bar(x-.18,[r["before_eui"] for r in rows],.36,label="Revisão anterior",color="#b7bbc3")
        ax.bar(x+.18,[r["after_eui"] for r in rows],.36,label="Revisão atual",color="#087e8b")
        refx=[i for i,r in enumerate(rows) if r["reference"]]
        ax.scatter(refx,[rows[i]["reference"]["electricity_kwh_m2_year"] for i in refx],
            color="#c7633a",marker="D",label="Média PT usada como referência",zorder=4)
        ax.set_xticks(x,[labels[r["sector"]] for r in rows],rotation=25,ha="right")
        ax.set_ylabel("Eletricidade anualizada (kWh/m² · 365 dias)")
        ax.set_title("Mediana por atividade, incluindo aparelhos; exclui EV")
        ax.legend(fontsize=8)
        root=ROOT/"data/datasets/rec_pt_2023_premium_100_v2"
        clock=pd.DatetimeIndex(pd.read_parquet(root/"shared/calendar.parquet").timestamp_utc).tz_convert("Europe/Lisbon")
        members=pd.read_parquet(root/"catalogs/members.parquet")
        selection=(clock>=pd.Timestamp("2024-02-05",tz="Europe/Lisbon"))&(clock<pd.Timestamp("2024-02-12",tz="Europe/Lisbon"))
        for ax,kind in zip((axes[0,1],axes[1,0]),("office","cafe")):
            m=members.loc[members.routine_archetype.eq(kind)].iloc[0]
            ops=pd.read_parquet(root/f"operations/{m.member_id}.parquet").loc[selection]
            days=(clock[selection]-clock[selection][0]).total_seconds()/86400
            ax.stackplot(days,ops.staff_count,ops.client_count,labels=["Trabalhadores","Clientes"],colors=["#087e8b","#dda45c"],step="post")
            ax.set_xticks(np.arange(7)+.5,["Seg","Ter","Qua","Qui","Sex","Sáb","Dom"])
            ax.set_xlim(0,7);ax.set_ylabel("Pessoas presentes")
            ax.set_title(labels[kind]+": presenças sintéticas, não medições")
            ax.legend(fontsize=8,loc="upper right")
        ax=axes[1,1];apps=pd.read_parquet(root/"catalogs/deferrables.parquet")
        app=apps.loc[apps.equipment_use.eq("linen_laundry")].iloc[0]
        pro=pd.read_parquet(root/f"deferrables/{app.deferrable_id}_profiles.parquet")
        for temp,g in pro.groupby("wash_temperature_c"):
            ax.scatter(g.load_kg,g.total_energy_kwh,label=f"{temp} °C — modelo",alpha=.7)
        ax.scatter([12],[1.68],s=100,marker="D",color="#c7633a",label="Ficha: 12 kg / 60 °C / água fria")
        ax.set_xlabel("Carga de roupa (kg)");ax.set_ylabel("Energia por programa (kWh)")
        ax.set_title("Lavandaria: especificação e transferência de engenharia")
        ax.legend(fontsize=8)
        for ax in axes.flat:ax.spines[["top","right"]].set_visible(False)
        fig.suptitle("Portugal — consumos comerciais, ocupação e equipamentos",fontsize=16)
        fig.text(.5,-.025,"As referências PT informam pressupostos; não são validação independente. Ocupação, frequências e transferências de equipamento permanecem hipóteses.",ha="center",fontsize=9)
        fig.savefig(ROOT/"docs/portugal_rec_native/figures/commercial_appliance_validation.png",dpi=150,bbox_inches="tight")
        plt.close(fig);return
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout="constrained")
    sectors=["office","retail","clinic","cafe","workshop","sports_centre","care_home","grocery"]
    labels=["Escritório","Loja","Clínica","Café","Oficina","Desporto","Lar","Supermercado"]
    colors=["#b7bbc3","#087e8b","#c7633a"]
    frames=[]
    for key in ("before","after"):
        f=pd.DataFrame(report[key]["commercial"]);frames.append(f.loc[f.year.eq(2024)].groupby("sector").median(numeric_only=True))
    f=pd.DataFrame(report["commercial_heldout"]);frames.append(f.loc[f.valid].groupby("sector").median(numeric_only=True))
    for ax,metric,title in ((axes[0,0],"weekday_night_day_ratio","Consumo noturno / diurno"),
                            (axes[0,1],"hourly_ramp_p95_to_mean","Variação horária: P95 / média")):
        for i,(frame,label) in enumerate(zip(frames,["Antes","Revisto","Referência reservada BDG2"])):
            ax.bar(np.arange(len(sectors))+(i-1)*.25,frame[metric].reindex(sectors),.24,color=colors[i],label=label)
        ax.set_xticks(range(len(sectors)),labels,rotation=30,ha="right");ax.set_title(title);ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    axes[0,0].legend(fontsize=8)
    kinds=["household_laundry","household_dishes"]
    for ax,col,title in ((axes[1,0],"energy_quantiles","Energia por programa (kWh)"),
                         (axes[1,1],"duration_quantiles","Duração por programa (horas)")):
        for i,key in enumerate(("before","after")):
            f=pd.DataFrame(report[key]["appliances"]);f=f.loc[f.year.eq(2024)]
            for j,kind in enumerate(kinds):
                values=np.array(f.loc[f.equipment_use.eq(kind),col].tolist());q=np.median(values,axis=0)
                ax.errorbar(j+(i-1)*.18,q[1],yerr=[[q[1]-q[0]],[q[2]-q[1]]],fmt="o",capsize=5,color=colors[i],label=["Antes","Revisto"][i] if j==0 else None)
        measured=pd.DataFrame(report["appliance_measured"])
        for j,kind in enumerate(("washing_machine","dishwasher")):
            g=measured.loc[measured.split.eq("heldout")&measured.appliance_type.eq(kind)&measured.quality.eq("all_candidates")&measured.threshold_w.eq(10)&measured.gap_seconds.eq(900)]
            q=np.median(np.array(g[col].tolist()),axis=0)
            ax.errorbar(j+.18,q[1],yerr=[[q[1]-q[0]],[q[2]-q[1]]],fmt="o",capsize=5,color=colors[2],label="Referência reservada REFIT" if j==0 else None)
        ax.set_xticks([0,1],["Máquina de roupa","Máquina de loiça"]);ax.set_title(title);ax.grid(axis="y",alpha=.15)
        ax.set_ylim(bottom=0)
    axes[1,0].legend(fontsize=8)
    fig.suptitle("Portugal: validação comercial e de aparelhos\nReferências externas com diferenças de escala, geografia e equipamento",fontsize=15)
    fig.text(.5,-.035,"BDG2: medianas por edifício. REFIT: mediana dos P10/P50/P90 por aparelho; barras não são intervalos de confiança.\nSupermercado: sem edifício de desenvolvimento. Equipamento profissional continua identificado como pressuposto.",ha="center",fontsize=9)
    target=ROOT/"docs/portugal_rec_native/figures/commercial_appliance_validation.png"
    fig.savefig(target,dpi=150,bbox_inches="tight");plt.close(fig)


def regenerate_and_reproduce():
    import subprocess
    import time
    for iteration in (1,2):
        start=__import__("time").perf_counter()
        subprocess.run([sys.executable,str(ROOT/"scripts/revise_portugal_realism.py"),"--apply"],cwd=ROOT,
            stdout=subprocess.PIPE,check=True,text=True)
        hashes={str(p.relative_to(ROOT)):sha(p.read_bytes()) for root in sorted((ROOT/"data/datasets").glob("rec_pt_2023_*_v2")) for p in root.rglob("*.parquet")}
        print("Full regeneration",iteration,"complete",len(hashes),"Parquet files",round(time.perf_counter()-start,1),"seconds",flush=True)
        if iteration==1:before=hashes
        else:assert hashes==before,[p for p in hashes if hashes[p]!=before.get(p)]
    report=json.loads(OUTPUT.read_text())
    report["reproducibility"]=dict(status="pass",parquet_files=len(hashes),identical_bytes=True,
        scope="Two complete four-family generations from pinned Git source, including matched propagation and EV planning. Original raw downloads, archives and second dataset trees are not created.",
        generator_sha256={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in [ROOT/"scripts"/n for n in
            ("portugal_activity.py","portugal_commercial_activity.py","portugal_stochastic_demand.py","portugal_realism_config.json","revise_portugal_realism.py","revise_portugal_ev_mobility.py")]})
    dump(OUTPUT,report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire-commercial",action="store_true")
    parser.add_argument("--acquire-appliances",action="store_true")
    parser.add_argument("--acquire-kitchens",action="store_true")
    parser.add_argument("--baseline",action="store_true")
    parser.add_argument("--development",action="store_true")
    parser.add_argument("--freeze-protocol",action="store_true")
    parser.add_argument("--evaluate",action="store_true")
    parser.add_argument("--seeds",action="store_true")
    parser.add_argument("--figure",action="store_true")
    parser.add_argument("--regenerate-and-reproduce",action="store_true")
    args=parser.parse_args()
    if args.acquire_commercial:acquire_commercial()
    if args.acquire_appliances:acquire_appliances()
    if args.acquire_kitchens:acquire_kitchens()
    if args.regenerate_and_reproduce:regenerate_and_reproduce()
    if args.freeze_protocol:freeze_protocol()
    if args.evaluate:final_comparison()
    if args.seeds:commercial_seed_sensitivity()
    if args.figure:make_figure()
    if args.baseline:
        report=json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
        assert "before" not in report,"Baseline already frozen"
        report["before"]=dict(revision="portugal-realism-2",commercial=synthetic_commercial(),appliances=synthetic_appliances(),inventory=inventory_audit())
        dump(OUTPUT,report);print("Baseline frozen",flush=True)
    if args.development:
        report=json.loads(OUTPUT.read_text())
        report["commercial_development"]=measured_commercial("development")
        dump(OUTPUT,report)
        frame=pd.DataFrame(report["commercial_development"])
        cols=["annualized_electricity_kwh_m2","conditional_daily_log_sigma","conditional_daily_lag1","weekday_night_day_ratio","weekend_weekday_ratio","hourly_ramp_p95_to_mean","daily_shape_deviation","lag7_day_correlation"]
        print(frame.loc[frame.valid].groupby("sector")[cols].median().round(3).to_string())
        print("Counts",frame.groupby(["sector","valid"]).size().to_dict())
        synthetic=pd.DataFrame(report["before"]["commercial"])
        print("Synthetic baseline",synthetic.loc[synthetic.year.eq(2024)].groupby("sector")[cols].median().round(3).to_string())


if __name__=="__main__":main()

"""Independent measured-data benchmarks and rigidity diagnostics for Portugal.

Raw SHEERM meter measurements (not gap-filled files) and measured SINTEF
charging reports are retained once as compact, lossless Parquet inputs. No
estimated EV SOC or inferred batteries are treated as measured observations.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from revise_portugal_ev_mobility import dump, write

REFERENCE = ROOT / "data/portugal_validation_reference"
OUTPUT = ROOT / "results/portugal_rec_native/empirical_validation.json"


def acquire():
    """Fetch only measured load/session columns; pin every original checksum."""
    REFERENCE.mkdir(parents=True, exist_ok=True)
    items = []
    for record in ("13972210", "13896176"):
        response = requests.get(f"https://zenodo.org/api/records/{record}", timeout=60)
        response.raise_for_status()
        record_data = response.json()
        assert record_data["metadata"]["license"]["id"] == "cc-by-4.0"
        for f in record_data["files"]:
            if (record == "13972210" and f["key"].startswith("Load House ") and f["key"].endswith("_raw.csv")) or f["key"] == "Dataset1_charging_reports.csv":
                items.append((record, f))

    def fetch(item):
        record, f = item
        response = requests.get(f["links"]["self"], timeout=90)
        response.raise_for_status()
        raw = response.content
        assert len(raw) == f["size"]
        assert "md5:" + hashlib.md5(raw).hexdigest() == f["checksum"]
        source = dict(record=record, filename=f["key"], url=f["links"]["self"],
                      bytes=len(raw), checksum=f["checksum"], sha256=hashlib.sha256(raw).hexdigest(), license="CC-BY-4.0")
        if record == "13972210":
            frame = pd.read_csv(io.BytesIO(raw))
            house = int(f["key"].split()[2].split("_")[0])
            # Civil timestamps are kept as published. DST/duplicate and missing
            # intervals are excluded by complete-day validation, never filled.
            result = pd.DataFrame(dict(house=house, timestamp_local=pd.to_datetime(frame.Date+" "+frame.Hour),
                                       power_kw=pd.to_numeric(frame["Consumption (kW)"], errors="coerce")))
        else:
            result = pd.read_csv(io.BytesIO(raw), sep=";", decimal=",")
            for c in ("plugin_time", "plugout_time"):
                result[c] = pd.to_datetime(result[c], errors="coerce")
        return record, source, result

    results = list(ThreadPoolExecutor(3).map(fetch, items))
    assert sum(r[0] == "13972210" for r in results) == 13
    for record, filename in (("13972210", "sheerm_raw_load.parquet"), ("13896176", "sintef_measured_sessions.parquet")):
        write(REFERENCE / filename, pd.concat([r[2] for r in results if r[0] == record], ignore_index=True))
    manifest = dict(revision="portugal-empirical-1", sources=[r[1] for r in results],
        acquisition_date="2026-11-05", load_units="Published mean kW per 15-minute interval; kWh = kW/4",
        load_quality="Raw-tagged files, no gap-filled version or interpolation added by this audit. Upstream meter filtering cannot be independently excluded. Complete 96-slot civil days only; Lisbon DST dates excluded.",
        load_cohort_limitations="Thirteen mainland Portuguese households; exact equipment/occupants not fully known. Non-electric heating reported for houses 8-10. SHEERM PV output is modelled, not measured; excluded from PV validation.",
        split="Declared before metric inspection: odd SHEERM houses for development, even houses for held-out validation. SINTEF users split by SHA-256 parity.",
        scope="Small, nonrepresentative household cohorts; Portuguese meters and Norwegian residential EV sessions are separate populations. No empirical arrival-SOC validation.")
    manifest["derived_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in REFERENCE.glob("*.parquet")}
    dump(REFERENCE / "sources.json", manifest)
    return manifest


def complete_days(times, powers):
    """Return only unambiguous, entirely observed 96-slot days and their dates."""
    times = pd.DatetimeIndex(times)
    if times.tz is not None:
        times = times.tz_convert("Europe/Lisbon").tz_localize(None)
    frame = pd.DataFrame({"date": times.normalize(), "slot": times.hour*4+times.minute//15,
                          "kw": np.asarray(powers, dtype=float), "stamp": times})
    frame.loc[~np.isfinite(frame.kw) | frame.kw.lt(0), "kw"] = np.nan
    ambiguous = times.tz_localize("Europe/Lisbon", ambiguous="NaT", nonexistent="NaT").isna()
    bad_dates = set(times[ambiguous].normalize())
    count = frame.groupby("date").agg(rows=("kw", "size"), valid=("kw", "count"), slots=("slot", "nunique"))
    valid = count.index[(count.rows == 96) & (count.valid == 96) & (count.slots == 96)]
    valid = valid[~valid.isin(bad_dates)]
    frame = frame.loc[frame.date.isin(valid)]
    matrix = frame.pivot(index="date", columns="slot", values="kw").sort_index()
    matrix = matrix.loc[matrix.sum(axis=1) > 0]
    return matrix


def load_metrics(matrix):
    values = matrix.to_numpy()
    assert len(values) > 10 and values.shape[1] == 96
    mean = values.mean()
    energy = values.sum(axis=1)*.25
    normalized = values/values.mean(axis=1)[:, None]
    dates = matrix.index
    typical = pd.DataFrame(normalized, index=dates).groupby(dates.dayofweek).transform("mean").to_numpy()
    def corr_lag(days):
        aligned = matrix.reindex(dates-pd.Timedelta(days=days)).to_numpy()
        valid = np.isfinite(aligned).all(axis=1)
        return float(np.corrcoef(values[valid].ravel(), aligned[valid].ravel())[0, 1]) if valid.sum() > 2 else None
    daily = pd.Series(energy, index=dates)
    monthly_mean = daily.groupby([dates.year, dates.month]).transform("mean")
    return dict(complete_days=len(matrix), mean_kw=float(mean), mean_day_kwh=float(energy.mean()),
        daily_energy_cv=float(energy.std()/energy.mean()), within_month_day_cv=float((daily/monthly_mean).std()),
        p99_to_mean=float(np.quantile(values,.99)/mean), ramp_p95_to_mean=float(np.quantile(np.abs(np.diff(values,axis=1)),.95)/mean),
        daily_shape_deviation=float(np.abs(normalized-typical).mean()),
        lag1_day_correlation=corr_lag(1), lag7_day_correlation=corr_lag(7),
        identical_daily_shapes=int(pd.DataFrame(np.round(normalized, 6)).duplicated().sum()),
        largest_daily_peak_slot_share=float(pd.Series(values.argmax(axis=1)).value_counts(normalize=True).max()),
        unchanged_quarter_hour_share=float(np.mean(np.diff(values,axis=1)==0)),
        weekend_to_weekday_energy=float(energy[dates.dayofweek>=5].mean()/energy[dates.dayofweek<5].mean()))


def synthetic_household(root, mid, include_controlled=True):
    frame = pd.read_parquet(root/f"timeseries/{mid}.parquet")
    energy = frame.non_shiftable_load.to_numpy(dtype=float).copy()
    if include_controlled:
        apps = pd.read_parquet(root/"catalogs/deferrables.parquet")
        for a in apps.loc[apps.member_id.eq(mid) & apps.controlled].itertuples():
            pro = pd.read_parquet(root/f"deferrables/{a.deferrable_id}_profiles.parquet").set_index("profile_id")
            schedules = pd.read_parquet(root/f"deferrables/{a.deferrable_id}_schedule.parquet")
            for s in schedules.itertuples():
                values = np.asarray(json.loads(pro.loc[s.profile_id,"load_profile"]))
                start=int(s.earliest_start_time_step)
                energy[start:start+len(values)] += values
    return energy*4


def measured_load_rows():
    raw = pd.read_parquet(REFERENCE/"sheerm_raw_load.parquet")
    rows=[]
    for house, f in raw.groupby("house"):
        matrix = complete_days(f.timestamp_local, f.power_kw)
        rows.append(dict(house=int(house), split="development" if house % 2 else "heldout", raw_rows=len(f),
                         invalid_measurements=int((~np.isfinite(f.power_kw) | f.power_kw.lt(0)).sum()),
                         **load_metrics(matrix)))
    return rows


def measured_house_independence():
    """Detect copied traces across the declared household split."""
    raw=pd.read_parquet(REFERENCE/"sheerm_raw_load.parquet")
    series={int(h):g.drop_duplicates("timestamp_local",keep=False).set_index("timestamp_local").power_kw
            for h,g in raw.groupby("house")}
    pairs=[]
    for a,b in itertools.combinations(series,2):
        aligned=pd.concat([series[a],series[b]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
        aligned=aligned.loc[(aligned>=0).all(axis=1)]
        if len(aligned)<96:continue
        equal=float(np.isclose(aligned.iloc[:,0],aligned.iloc[:,1],rtol=1e-6,atol=1e-8).mean())
        pairs.append(dict(houses=[a,b],overlapping_slots=len(aligned),equal_value_share=equal,
            pearson=float(aligned.corr().iloc[0,1]),cross_split=bool(a%2!=b%2)))
    assert not any(r["equal_value_share"]>.99 for r in pairs),"Near-duplicate measured houses"
    return dict(pairs=pairs,near_duplicates=0,interpretation="Trace independence check only; distinct homes still share weather and population selection biases.")


def measured_quality_sensitivity(original_rows):
    """Audit raw spikes against the authors' independently published threshold.

    This is a reported post-hoc quality sensitivity, not a change to the fitted
    parameters, reserved split, primary raw comparison or retained source data.
    """
    raw=pd.read_parquet(REFERENCE/"sheerm_raw_load.parquet")
    threshold_kw=10/.25  # SHEERM methods: 10 kWh per 15-minute interval.
    flagged=raw.loc[raw.power_kw.gt(threshold_kw)]
    rows=[]
    for house,frame in raw.groupby("house"):
        power=frame.power_kw.mask(frame.power_kw.gt(threshold_kw))
        rows.append(dict(house=int(house),split="development" if house%2 else "heldout",
                         **load_metrics(complete_days(frame.timestamp_local,power))))
    before=pd.DataFrame(original_rows).query("split == 'heldout'")
    after=pd.DataFrame(rows).query("split == 'heldout'")
    metrics=["mean_day_kwh","within_month_day_cv","ramp_p95_to_mean","daily_shape_deviation","p99_to_mean","lag1_day_correlation","lag7_day_correlation"]
    comparisons={key:dict(raw_median=float(before[key].median()),screened_median=float(after[key].median())) for key in metrics}
    return dict(source="https://pmc.ncbi.nlm.nih.gov/articles/PMC11929752/",published_threshold_kwh_per_interval=10,
        threshold_kw=threshold_kw,flagged_observations=flagged.assign(timestamp_local=flagged.timestamp_local.astype(str)).to_dict("records"),
        complete_days_removed=int(before.complete_days.sum()-after.complete_days.sum()),heldout_medians=comparisons,
        method="Post-hoc source-quality sensitivity: remove whole days containing raw readings above the published acquisition threshold, without interpolation, source replacement or generator retuning. Primary comparison retains raw observations.")


def snapshot_loads():
    rows=[]
    for root in sorted((ROOT/"data/datasets").glob("rec_pt_2023_*_v2")):
        members=pd.read_parquet(root/"catalogs/members.parquet")
        clock=pd.DatetimeIndex(pd.read_parquet(root/"shared/calendar.parquet").timestamp_utc)
        for m in members.loc[members.category.eq("residential")].itertuples():
            power=synthetic_household(root,m.member_id)
            for year in (2023,2024):
                mask=clock.year==year
                rows.append(dict(family=root.name, member=m.member_id, year=year,
                                 **load_metrics(complete_days(clock[mask],power[mask]))))
    return rows


def ev_statistics(frame):
    result=dict(sessions=len(frame),users=frame.user.nunique())
    for col in ("energy_kwh","duration_hours","arrival_hour"):
        values=frame[col].dropna()
        result[col]={str(q):float(values.quantile(q)) for q in (.1,.5,.9,.99)}
    rates=[];gaps=[];duration_modes=[]
    for _,group in frame.sort_values("arrival").groupby("user"):
        span=(group.arrival.max()-group.arrival.min()).total_seconds()/86400
        if span>=60 and len(group)>=10:
            rates.append((len(group)-1)*7/span)
            gaps.extend(group.arrival.diff().dt.total_seconds().dropna()/3600)
        duration_modes.append((group.duration_hours.mul(4).round()/4).value_counts(normalize=True).max())
    result["observed_span_sessions_per_week"]={str(q):float(np.quantile(rates,q)) for q in (.1,.5,.9)}
    result["inter_session_hours"]={str(q):float(np.quantile(gaps,q)) for q in (.1,.5,.9)}
    result["median_user_duration_mode_share"]=float(np.nanmedian(duration_modes))
    result["weekend_session_share"]=float((frame.arrival.dt.dayofweek>=5).mean())
    result["energy_duration_spearman"]=float(frame[["energy_kwh","duration_hours"]].corr(method="spearman").iloc[0,1])
    return result


def ev_comparison():
    raw=pd.read_parquet(REFERENCE/"sintef_measured_sessions.parquet")
    measured=pd.DataFrame(dict(user=raw.user_id,arrival=raw.plugin_time,energy_kwh=raw.energy_session,
        duration_hours=raw.connection_time,arrival_hour=raw.plugin_time.dt.hour+raw.plugin_time.dt.minute/60))
    measured["split"]=measured.user.map(lambda s:"development" if int(hashlib.sha256(s.encode()).hexdigest()[:8],16)%2 else "heldout")
    measured=measured.loc[measured.energy_kwh.gt(.5) & measured.arrival.notna()]
    output=dict(measured={k:ev_statistics(v) for k,v in measured.groupby("split")},synthetic=[],
        scope="Measured Norwegian residential AC sessions, mixed/unknown vehicle inventory, 2018-2021; compare Portuguese synthetic home resident sessions only, not workplace/visitors. No inferred batteries or SOC used. Synthetic AC input includes charger and battery losses; metering boundary may differ slightly.",
        matched_filters="Energy >0.5 kWh; finite duration 2 minutes to 120 hours for duration/energy joint comparison; uncensored synthetic home sessions in 2024. Rates require >=10 sessions and >=60 observed days, use (n-1)/observed arrival span, not annual attendance. Duration-mode shares quantize BOTH populations to 15 minutes.",
        interpretation="Distribution discrepancies are reported, not erased by fitting a different fleet to Norwegian observations. SOC targets and resident profile proportions remain scenario design assumptions.")
    # Source already cleaned most timing violations; apply the same duration
    # filter to both populations, preserving missing/invalid counts separately.
    for split,group in measured.groupby("split"):
        valid=group.duration_hours.between(2/60,120)
        output["measured"][split]=ev_statistics(group.loc[valid])
        output["measured"][split]["excluded_timing_sessions"]=int((~valid).sum())
    for root in sorted((ROOT/"data/datasets").glob("rec_pt_2023_*_v2")):
        sessions=pd.read_parquet(root/"catalogs/charging_sessions.parquet")
        schema=json.loads((root/"schema.json").read_text())
        clock=pd.DatetimeIndex(pd.read_parquet(root/"shared/calendar.parquet").timestamp_utc)
        selected=sessions.loc[sessions.purpose.eq("home") & sessions.year.eq(2024) & ~sessions.right_censored].copy()
        rows=[]
        for s in selected.itertuples():
            battery=schema["electric_vehicles_def"][s.ev_id]["battery"]["attributes"]
            charger=schema["buildings"][s.member_id]["chargers"][s.charger_id]["attributes"]
            arrival=clock[s.arrival_time_step].tz_convert("Europe/Lisbon").tz_localize(None)
            rows.append(dict(user=s.ev_id,arrival=arrival,arrival_hour=arrival.hour+arrival.minute/60,
                energy_kwh=s.requested_energy_reference_kwh/np.sqrt(battery["efficiency"])/charger["efficiency"],
                duration_hours=(s.observed_end_time_step-s.arrival_time_step)*.25))
        f=pd.DataFrame(rows);valid=f.energy_kwh.gt(.5) & f.duration_hours.between(2/60,120)
        output["synthetic"].append(dict(family=root.name,excluded_sessions=int((~valid).sum()),**ev_statistics(f.loc[valid])))
    return output


def rigidity_audit():
    """Search deterministic artifacts separately from declared physical controls."""
    rows=[]
    for root in sorted((ROOT/"data/datasets").glob("rec_pt_2023_*_v2")):
        members=pd.read_parquet(root/"catalogs/members.parquet")
        cal=pd.read_parquet(root/"shared/calendar.parquet")
        clock=pd.DatetimeIndex(cal.timestamp_utc)
        assert len(clock)==70176 and np.all(np.diff(clock.asi8)==900*10**9)
        physical=[];pv=[];max_balance=0.;bad_values=0;temporal=[]
        for m in members.itertuples():
            f=pd.read_parquet(root/f"timeseries/{m.member_id}.parquet")
            end=pd.read_parquet(root/f"end_uses/{m.member_id}.parquet")
            values=f.non_shiftable_load.to_numpy()
            bad_values+=int((~np.isfinite(values)|(values<0)).sum())
            max_balance=max(max_balance,float(np.max(np.abs(end.sum(axis=1)-values))))
            physical.append(values)
            matrix=complete_days(clock[35040:],values[35040:]*4)
            diagnostics=load_metrics(matrix)
            temporal.append(dict(member=m.member_id,category=m.category,
                identical_daily_shapes=diagnostics["identical_daily_shapes"],
                largest_daily_peak_slot_share=diagnostics["largest_daily_peak_slot_share"],
                within_month_day_cv=diagnostics["within_month_day_cv"],
                lag7_day_correlation=diagnostics["lag7_day_correlation"],
                exact_52_week_repeated_intervals_share=float(np.mean(values[34944:]==values[:-34944]))))
            if f.solar_generation.max()>0:
                pv.append(f.solar_generation.to_numpy()/f.solar_generation.max())
        daily=pd.read_parquet(root/"catalogs/household_activity.parquet")
        leave_repeats=0;leave_pairs=0
        for mid,d in daily.groupby("member_id"):
            dates=[d.loc[d.year.eq(year)&d.on_leave,"date"].str[5:].tolist() for year in (2023,2024)]
            leave_pairs+=1;leave_repeats+=dates[0]==dates[1]
        s=pd.read_parquet(root/"catalogs/charging_sessions.parquet")
        v=pd.read_parquet(root/"catalogs/electric_vehicles.parquet")
        modes={}
        for pool,group in s.merge(v[["ev_id","vehicle_pool"]],on="ev_id").groupby("vehicle_pool"):
            modes[pool]=float(np.median([((g.observed_end_time_step-g.arrival_time_step).value_counts(normalize=True).max()) for _,g in group.groupby("ev_id")]))
        overlaps=[]
        for path in (root/"evaluation").glob("*.json"):
            schema=json.loads(path.read_text())
            if not schema.get("robustness",{}).get("enabled"):continue
            events=pd.read_parquet(root/schema["robustness"]["events_file"])
            local=events.loc[events.generation_regime.eq("local_fault")].drop_duplicates("cause_id")
            overlaps.append(dict(variant=path.stem,independent_local_causes=len(local),
                coincident_local_cause_starts=int(local.start_time_step.duplicated().sum())))
        assert bad_values==0 and max_balance<1e-5
        exact=sum(np.array_equal(a,b) for i,a in enumerate(physical) for b in physical[i+1:])
        assert exact==0
        corr=np.corrcoef(pv) if len(pv)>1 else np.eye(len(pv))
        revision=json.loads((root/"ev_mobility_revision.json").read_text())
        rows.append(dict(family=root.name,invalid_native_demand_values=bad_values,max_end_use_balance_error_kwh=max_balance,
            duplicate_member_loads=exact,identical_leave_calendar_year_pairs=leave_repeats,household_year_pairs=leave_pairs,
            median_ev_duration_mode_share=modes,pv_median_profile_correlation=float(np.median(corr[np.triu_indices(len(pv),1)])) if len(pv)>1 else None,
            local_fault_coincidences=overlaps,connection_jitter_rejected_ev_ids=revision.get("connection_jitter_rejected_ev_ids",[])))
        rows[-1]["temporal_structure_all_members"]=temporal
    return dict(families=rows,retained_structures=[
        "Installed equipment, annual driving budgets and resident charging-profile composition are fixed experimental controls, not resampled population estimates.",
        "Business opening hours and four-on/four-off night shifts are declared routines; external commercial/appliance comparisons are in commercial_appliance_validation.json, with explicit cohort and transfer limitations.",
        "Indoor 21 C/50% fields remain inactive compatibility placeholders; no thermal comfort validation.",
        "PV sites share Lisbon weather and have different tilt/azimuth/shading; strong correlation is expected, not independent climate evidence.",
        "Controlled topology histories, compulsory appliance jobs, full-charge preferences and impossible DR stress are benchmark assumptions.",
        "15-minute observations cannot establish sub-minute transients, interruptions shorter than 15 minutes or realistic 15-second control."],
        status="pass_internal_checks_with_declared_modelling_limits")


def load_comparison(report):
    measured=pd.DataFrame(report["measured_load"])
    current=pd.DataFrame(report["synthetic_load"])
    before=pd.DataFrame(report["before"]["synthetic_load"])
    metrics=["mean_day_kwh","within_month_day_cv","ramp_p95_to_mean","daily_shape_deviation",
             "p99_to_mean","lag1_day_correlation","lag7_day_correlation","largest_daily_peak_slot_share"]
    target={}
    for split,frame in measured.groupby("split"):
        values={}
        for key in metrics:
            a=frame[key].to_numpy();rng=np.random.default_rng(20261105)
            draws=np.median(rng.choice(a,(2000,len(a)),replace=True),axis=1)
            values[key]=dict(median=float(np.median(a)),p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9)),
                            bootstrap_household_median_95ci=np.quantile(draws,[.025,.975]).tolist())
        target[split]=dict(households=len(frame),complete_days=int(frame.complete_days.sum()),metrics=values)
    comparisons=[]
    score_metrics=report["load_calibration"]["target_median"].keys()
    for family,frame in current.loc[current.year.eq(2024)].groupby("family"):
        previous=before.loc[before.family.eq(family)&before.year.eq(2024)]
        target_heldout=pd.Series({k:target["heldout"]["metrics"][k]["median"] for k in metrics})
        med=frame[metrics].median();old=previous[metrics].median()
        comparisons.append(dict(family=family,synthetic_households=len(frame),before_median=old.to_dict(),revised_median=med.to_dict(),
            heldout_ratio=(med/target_heldout).to_dict(),
            before_mean_absolute_log_ratio=float(np.abs(np.log(old[list(score_metrics)]/target_heldout[list(score_metrics)])).mean()),
            revised_mean_absolute_log_ratio=float(np.abs(np.log(med[list(score_metrics)]/target_heldout[list(score_metrics)])).mean()),
            caution="Diagnostic distribution distance, not an empirical pass threshold or population hypothesis test"))
    return dict(measured=target,comparisons=comparisons,
        boundary="Meter demand versus synthetic nonshiftable demand plus controlled appliance cycles started at their earliest allowed time. EV charging excluded because measured fleet/equipment inventory is unknown; no controller behaviour is claimed.",
        uncertainty="Households, not quarter-hour rows, are the independent bootstrap units. Small nonrandom cohort; confidence intervals do not establish nationwide representativeness. Calendar periods, equipment, electric heating and occupancy differ.")


def figure(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    measured=pd.DataFrame(report["measured_load"]).query("split == 'heldout'")
    before=pd.DataFrame(report["before"]["synthetic_load"])
    current=pd.DataFrame(report["synthetic_load"])
    before=before.loc[before.family.str.contains("premium")&before.year.eq(2024)]
    current=current.loc[current.family.str.contains("premium")&current.year.eq(2024)]
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    panels=[("within_month_day_cv","Variação da energia entre dias","Coeficiente de variação dentro do mês"),
            ("ramp_p95_to_mean","Variações entre quartos de hora","P95 da variação absoluta / potência média"),
            ("daily_shape_deviation","Variação da forma do perfil diário","Desvio absoluto do perfil médio por dia da semana"),
            ("p99_to_mean","Picos de consumo","P99 da potência / potência média")]
    for ax,(key,title,label) in zip(axes.ravel(),panels):
        data=[measured[key],before[key],current[key]]
        plot=ax.boxplot(data,patch_artist=True,showfliers=False)
        ax.set_xticks([1,2,3],["Reais: 6 casas\nreservadas","Antes: 80 casas\nsintéticas","Revisto: 80 casas\nsintéticas"])
        for patch_,color in zip(plot['boxes'],['#3b526b','#b8bec5','#338b79']):patch_.set_facecolor(color);patch_.set_alpha(.7)
        ax.scatter(np.ones(len(measured)),measured[key],s=20,c='#25394e',zorder=3)
        ax.set_title(title,fontweight='bold');ax.set_ylabel(label,fontsize=9);ax.grid(axis='y',alpha=.2)
    fig.suptitle("Portugal — comparação com medições brutas independentes",fontsize=16,fontweight='bold')
    fig.text(.5,.015,"SHEERM: ajuste em 7 outras casas; avaliação nas 6 mostradas. Premium residencial 2024. Coortes e equipamentos distintos.",ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.94])
    fig.savefig(ROOT/'docs/portugal_rec_native/figures/empirical_validation.png',dpi=170)
    plt.close(fig)


def calibrate_load():
    """Small predeclared grid; only odd measured houses enter the objective."""
    from portugal_activity import CONFIG, build_calendar, presence
    from portugal_stochastic_demand import activity_intensity, residential_shape, bounded_energy_allocation
    from revise_portugal_ev_mobility import source_frame
    report=json.loads(OUTPUT.read_text())
    report.setdefault("before",dict(synthetic_load=report["synthetic_load"],
        scope="portugal-realism-1 generated inputs before empirical corrections"))
    measured=pd.DataFrame(report["measured_load"])
    development=measured.loc[measured.split.eq("development")]
    metrics=["within_month_day_cv","ramp_p95_to_mean","daily_shape_deviation","p99_to_mean","lag1_day_correlation"]
    target=development[metrics].median()
    root=ROOT/"data/datasets/rec_pt_2023_premium_100_v2"
    members=pd.read_parquet(root/"catalogs/members.parquet")
    selected=members.loc[members.category.eq("residential")].groupby("routine_archetype",group_keys=False).head(2)
    calendar=pd.read_parquet(root/"shared/calendar.parquet")
    clock=pd.DatetimeIndex(calendar.timestamp_utc)
    daily=build_calendar(root,selected,calendar)
    schema=json.loads((root/"schema.json").read_text())
    cases=[]
    for m in selected.itertuples():
        d=daily.loc[daily.member_id.eq(m.member_id)].copy()
        ledger=pd.read_parquet(root/f"end_uses/{m.member_id}.parquet")
        old=source_frame(root/f"end_uses/{m.member_id}.parquet")
        base=ledger.base_excluding_cycles_kwh.to_numpy()
        cases.append((m,d,presence(clock,d,m.occupants),old.base_excluding_cycles_kwh.to_numpy(),
            ledger.drop(columns="base_excluding_cycles_kwh").sum(axis=1).to_numpy(),
            synthetic_household(root,m.member_id)/4-base,
            [base[:35040].sum(),base[35040:].sum()],
            schema["buildings"][m.member_id]["electrical_service"]["limits"]["total"]["import_kw"]*.25))
    rows=[]
    grid=list(itertools.product((.25,.35,.45),(.35,.5,.65),(8,16,24),(.55,.7)))
    for sigma,fraction,rate,rho in grid:
        CONFIG["demand"].update(daily_log_sigma=sigma,episode_fraction=fraction,episodes_per_day=rate,episode_persistence=rho)
        observations=[]
        for m,d,occ,original,other,all_other,totals,limit in cases:
            for year,inds in d.groupby("year").groups.items():
                d.loc[inds,"activity_intensity"]=activity_intensity(root,m.member_id,year,len(inds))
            base=residential_shape(root,m,original,calendar,d,occ)
            for ix,(lo,hi) in enumerate(((0,35040),(35040,70176))):
                base[lo:hi]=bounded_energy_allocation(base[lo:hi],limit-other[lo:hi],totals[ix])
            observations.append(load_metrics(complete_days(clock[35040:],4*(base+all_other)[35040:])))
        med=pd.DataFrame(observations)[metrics].median()
        score=float(np.mean(np.abs(np.log(med/target))))
        row=dict(parameters=CONFIG["demand"].copy(),score=score,synthetic_median=med.to_dict())
        rows.append(row)
        print('load grid',len(rows),len(grid),'score',round(score,4),flush=True)
    winner=min(rows,key=lambda r:r["score"])
    CONFIG["demand"].update(winner["parameters"])
    config_path=ROOT/"scripts/portugal_realism_config.json"
    config=json.loads(config_path.read_text());config["demand"]=winner["parameters"]
    config["parameter_status"]="Load episode parameters developed on seven odd-ID raw SHEERM households; six even-ID households reserved for external evaluation. Other behavioural/event parameters remain explicit scenario priors. Not a Portuguese population fit."
    dump(config_path,config)
    report["load_calibration"]=dict(development_houses=development.house.tolist(),heldout_houses=measured.loc[measured.split.eq("heldout"),"house"].tolist(),
        selected_synthetic_members=selected.member_id.tolist(),year=2024,
        objective="Mean absolute log ratio of household median metrics; equal metric weights; no held-out measured house used for selection",
        target_median=target.to_dict(),grid=rows,selected=winner,
        limitation="Aggregate base-load adjustment; fixed inventory energy, existing thermal/appliance priors. Calibration grid uses existing appliance schedules; final regenerated schedules are independently evaluated.")
    dump(OUTPUT,report)
    return winner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acquire",action="store_true")
    p.add_argument("--calibrate-load",action="store_true")
    args=p.parse_args()
    if args.acquire: acquire()
    if args.calibrate_load:
        print(json.dumps(calibrate_load(),indent=2))
        return
    sources=json.loads((REFERENCE/"sources.json").read_text())
    from audit_portugal_heldout_2024 import runtime_fingerprint
    fingerprint=runtime_fingerprint()
    for path,digest in sources["derived_sha256"].items():
        assert hashlib.sha256((REFERENCE/path).read_bytes()).hexdigest()==digest
    report=json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    report.update(revision="portugal-empirical-1", scope=sources["scope"], measured_load=measured_load_rows(), synthetic_load=snapshot_loads(),
                  ev_comparison=ev_comparison(),rigidity_audit=rigidity_audit(),measured_house_independence=measured_house_independence())
    report["measured_quality_sensitivity"]=measured_quality_sensitivity(report["measured_load"])
    if "before" in report:
        report["load_comparison"]=load_comparison(report)
        figure(report)
    assert runtime_fingerprint()==fingerprint,"Inputs changed during empirical comparison"
    report["runtime_input_fingerprint"]=fingerprint
    report["status"]="completed_comparison_with_reported_discrepancies"
    report["source_manifest_sha256"]=hashlib.sha256((REFERENCE/"sources.json").read_bytes()).hexdigest()
    report["audit_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["generator_sha256"]={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        [ROOT/"scripts"/n for n in ("portugal_activity.py","portugal_stochastic_demand.py","portugal_commercial_activity.py","portugal_realism_config.json","revise_portugal_realism.py","revise_portugal_ev_mobility.py")]}
    # Reject non-finite numeric evidence instead of emitting non-standard JSON.
    json.dumps(report,allow_nan=False)
    dump(OUTPUT,report)
    print(json.dumps({"measured_houses":len(report["measured_load"]),"synthetic_house_years":len(report["synthetic_load"]),"output":str(OUTPUT)}))


if __name__ == "__main__": main()

"""Seed and one-at-a-time sensitivity experiments without dataset copies.

Component experiments use every family and residential routine. Coherent small
community replays run the complete physical/appliance/EV generator in memory;
they are generation witnesses, not simulator-controller experiments.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
from portugal_activity import CONFIG, build_calendar, presence, rng_for
from portugal_stochastic_demand import residential_shape, bounded_energy_allocation
import revise_portugal_realism as generator
import revise_portugal_ev_mobility as ev
from portugal_empirical_validation import complete_days, load_metrics, synthetic_household
from audit_portugal_heldout_2024 import runtime_fingerprint

OUTPUT=ROOT/"results/portugal_rec_native/ensemble_validation.json"
METRICS=("within_month_day_cv","ramp_p95_to_mean","daily_shape_deviation","p99_to_mean","lag1_day_correlation")


def summarize(report):
    """Describe seed spread and paired local sensitivity without pass thresholds."""
    components=pd.DataFrame(report["component_ensembles"])
    keys=(*METRICS,"multiday_absences_per_household_year",
          "nonstress_outages_per_exposed_member_year","nonstress_outage_minutes_per_exposed_member_year")
    spread={family:{key:{str(q):float(group[key].quantile(q)) for q in (0,.05,.5,.95,1)}
                    for key in keys} for family,group in components.groupby("family")}
    paired=[]
    for row in report["sensitivity"]:
        if "distributions" in row:
            base=next(r for r in report["coherent_ensembles"] if (r["family"],r["seed"])==(row["family"],row["seed"]))
            extract=lambda r:next(d for d in r["distributions"] if d["year"]==2024 and d["pool"]=="resident")
            left,right=extract(row),extract(base)
            metrics=("sessions_per_vehicle_week","request_mean_pp","full_target_pct","median_slack_hours")
        else:
            left=row
            right=next(r for r in report["component_ensembles"] if (r["family"],r["seed"])==(row["family"],row["seed"]))
            metrics=keys
        for metric in metrics:
            paired.append(dict(parameter=row["parameter"],factor=row["factor"],seed=row["seed"],metric=metric,
                absolute_change=float(left[metric]-right[metric]),
                relative_change=float(left[metric]/right[metric]-1) if right[metric]!=0 else None))
        if "appliance_metrics" in row:
            for metric,value in row["appliance_metrics"].items():
                reference=base["appliance_metrics"][metric]
                paired.append(dict(parameter=row["parameter"],factor=row["factor"],seed=row["seed"],metric="appliances."+metric,
                    absolute_change=float(value-reference),relative_change=float(value/reference-1) if reference!=0 else None))
    report["summary"]=dict(component_draws=len(components),seeds=components.seed.nunique(),
        coherent_draws=len(report["coherent_ensembles"]),sensitivity_draws=len(report["sensitivity"]),
        event_programme_draws=len(report.get("event_ensembles",[])),
        seed_quantiles=spread,paired_sensitivity=paired,
        interpretation="Empirical Monte Carlo ranges, not population confidence intervals. Paired differences use identical seed and family. Fixed equipment/budgets limit variability; one-at-a-time perturbations do not explore parameter interactions.")
    report["generator_sha256"]={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [ROOT/"scripts"/n for n in ("portugal_activity.py","portugal_stochastic_demand.py","portugal_commercial_activity.py","portugal_realism_config.json","revise_portugal_realism.py","revise_portugal_ev_mobility.py")]}
    report["audit_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def appliance_metrics(root,memory):
    energies=[];slacks=[];energy_cvs=[]
    apps=memory[str((root/"catalogs/deferrables.parquet").resolve())]
    for app in apps.itertuples():
        schedules=memory[str((root/f"deferrables/{app.deferrable_id}_schedule.parquet").resolve())]
        profiles=memory[str((root/f"deferrables/{app.deferrable_id}_profiles.parquet").resolve())].set_index("profile_id")
        values=schedules.profile_id.map(profiles.total_energy_kwh).to_numpy()
        assert np.isfinite(values).all() and (values>0).all()
        energies.extend(values)
        energy_cvs.append(float(np.std(values)/np.mean(values)))
        if app.controlled:
            slacks.extend((schedules.latest_start_time_step-schedules.earliest_start_time_step).to_numpy()*.25)
    return dict(cycles=len(energies),requested_kwh=float(np.sum(energies)),median_cycle_kwh=float(np.median(energies)),
        median_within_appliance_energy_cv=float(np.median(energy_cvs)),
        controlled_median_slack_hours=float(np.median(slacks)),controlled_p90_slack_hours=float(np.quantile(slacks,.9)))


def physical_appliance_draw(root,seed):
    CONFIG["seed"]=seed
    memory={}
    with patch.object(generator,"write",lambda path,frame:memory.update({str(path.resolve()):frame.copy(deep=True)})):
        generator.revise_physical(root,pd.read_parquet(root/"shared/calendar.parquet"))
    return appliance_metrics(root,memory)


def event_draw(root,seed):
    """Regenerate every sensor/DR event programme without writing datasets."""
    CONFIG["seed"]=seed
    captured={}
    with patch.object(generator,"write",lambda path,frame:captured.update({str(path.relative_to(root)):frame.copy(deep=True)})):
        generator.revise_events(root)
    diagnostics=[]
    for path,frame in captured.items():
        assert frame.start_time_step.ge(0).all() and frame.end_time_step.lt(70176).all()
        assert frame.end_time_step.ge(frame.start_time_step).all()
        if "request_id" in frame:
            assert frame.request_id.is_unique
            stress=frame.loc[frame.stratum.eq("impossible_stress")]
            assert len(stress)==2 and stress.target_power_kw.eq(1500).all()
            assert (stress.end_time_step-stress.start_time_step+1).eq(16).all()
            diagnostics.append(dict(file=path,requests=len(frame),stress_requests=len(stress)))
        else:
            assert frame.event_id.is_unique
            diagnostics.append(dict(file=path,channel_rows=len(frame),latent_causes=frame.cause_id.nunique()))
    return dict(family=root.name,seed=seed,status="pass",programmes=diagnostics,
        scope="Complete sensor/DR programme regeneration, identity/bounds/nonoverlap and retained stress checks; grid outage statistics are in component draws. No simulator execution.")


def cases():
    result=[]
    for root in sorted((ROOT/"data/datasets").glob("rec_pt_2023_*_v2")):
        members=pd.read_parquet(root/"catalogs/members.parquet")
        selected=members.loc[members.category.eq("residential")].groupby("routine_archetype",group_keys=False).head(1)
        calendar=pd.read_parquet(root/"shared/calendar.parquet")
        schema=json.loads((root/"schema.json").read_text())
        entries=[]
        for m in selected.itertuples():
            ledger=pd.read_parquet(root/f"end_uses/{m.member_id}.parquet")
            source=ev.source_frame(root/f"end_uses/{m.member_id}.parquet")
            base=ledger.base_excluding_cycles_kwh.to_numpy()
            entries.append(dict(member=m,original=source.base_excluding_cycles_kwh.to_numpy(),
                other=ledger.drop(columns="base_excluding_cycles_kwh").sum(axis=1).to_numpy(),
                all_other=synthetic_household(root,m.member_id)/4-base,
                annual=[base[:35040].sum(),base[35040:].sum()],
                limit=schema["buildings"][m.member_id]["electrical_service"]["limits"]["total"]["import_kw"]*.25))
        result.append((root,selected,calendar,entries))
    return result


def component_draw(case, seed):
    root,selected,calendar,entries=case
    CONFIG["seed"]=seed
    daily=build_calendar(root,selected,calendar)
    clock=pd.DatetimeIndex(calendar.timestamp_utc)
    measures=[]
    for item in entries:
        m=item["member"]
        d=daily.loc[daily.member_id.eq(m.member_id)]
        occupancy=presence(clock,d,m.occupants)
        base=residential_shape(root,m,item["original"],calendar,d,occupancy)
        for ix,(lo,hi) in enumerate(((0,35040),(35040,70176))):
            base[lo:hi]=bounded_energy_allocation(base[lo:hi],item["limit"]-item["other"][lo:hi],item["annual"][ix])
        values=load_metrics(complete_days(clock[35040:],4*(base+item["all_other"])[35040:]))
        assert np.isfinite(base).all() and (base>=0).all()
        violation=float(np.max((base+item["other"])*4-item["limit"]*4))
        # Stored end-use ledgers are float32; use the same 1e-5 kW numerical
        # tolerance as native input validation (observed roundoff ~1e-7 kW).
        assert violation<=1e-5,(root.name,m.member_id,seed,violation)
        measures.append(values)
    long=daily.loc[daily.absence_start.ge(0)].drop_duplicates(["member_id","absence_start"])
    # Include zero-event households in all rates. Do not count channel aliases
    # as independent faults; common-cause IDs identify joint exposures.
    events=[r for mid in selected.member_id for r in generator.outage_programme(root,mid)]
    nonstress=[r for r in events if r["generation_regime"]!="controlled_stress"]
    common={r["cause_id"] for r in nonstress if r["generation_regime"]=="common_cause_fault"}
    yearly_exposure=len(selected)*2
    summary={k:float(np.median([v[k] for v in measures])) for k in METRICS}
    summary.update(seed=seed,family=root.name,households=len(selected),
        calendar_sha256=hashlib.sha256(pd.util.hash_pandas_object(daily,index=False).values.tobytes()).hexdigest(),
        multiday_absences_per_household_year=len(long)/yearly_exposure,
        nonstress_outages_per_exposed_member_year=len(nonstress)/yearly_exposure,
        nonstress_outage_minutes_per_exposed_member_year=sum((r["end_time_step"]-r["start_time_step"]+1)*15 for r in nonstress)/yearly_exposure,
        common_cause_years=len(common))
    return summary


def coherent_draw(root, seed):
    """Complete normal physical/appliance/EV chain, with all writes in RAM."""
    CONFIG["seed"]=seed
    memory={}
    read_original=pd.read_parquet
    def key(path): return str(Path(path).resolve()) if isinstance(path,(str,Path)) else None
    def capture(path,frame): memory[key(path)]=frame.copy(deep=True)
    def read(path,*args,**kwargs):
        value=memory.get(key(path))
        if value is None: return read_original(path,*args,**kwargs)
        result=value.copy(deep=True)
        return result[kwargs["columns"]] if kwargs.get("columns") else result
    with patch.object(generator,"write",capture),patch.object(pd,"read_parquet",read):
        calendar=pd.read_parquet(root/"shared/calendar.parquet")
        generator.revise_physical(root,calendar)
        schema=json.loads((root/"schema.json").read_text())
        # Map normal executable inputs to this realization, exactly as the
        # generator's propagation does, without modifying matched scenarios.
        for mid,b in schema["buildings"].items():
            target=root/b["energy_simulation"]
            original=ev.source_frame(target)
            physical=pd.read_parquet(root/f"timeseries/{mid}.parquet")
            old_physical=ev.source_frame(root/f"timeseries/{mid}.parquet")
            original["non_shiftable_load"]=np.maximum(0,original.non_shiftable_load+physical.non_shiftable_load-old_physical.non_shiftable_load)
            original["occupant_count"]=physical.occupant_count
            capture(target,original)
            for aid,a in b.get("deferrable_appliances",{}).items():
                capture(root/a["flexibility_schedule_file"],pd.read_parquet(root/f"deferrables/{aid}_schedule.parquet"))
        report=ev.revise_family(root,apply=False)
        for r in report["annual_energy"]:
            assert abs(r["trip_energy_kwh"]-r["original_trip_energy_kwh"])<1e-7
        # Physical and temporal feasibility are asserted by the complete planner.
        return dict(seed=seed,family=root.name,status="pass",distributions=report["distributions"],
            appliance_metrics=appliance_metrics(root,memory),
            appliances=sum(len(frame) for path,frame in memory.items() if path.endswith("_schedule.parquet") and "/deferrables/" in path),
            annual_driving_kwh=float(sum(r["trip_energy_kwh"] for r in report["annual_energy"])),
            scope="Full regenerated physical loads, appliances, calendars and EV reference trajectories; normal topology, two years, no native simulator execution")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-count",type=int,default=20)
    parser.add_argument("--coherent-count",type=int,default=5)
    parser.add_argument("--resume",action="store_true",help="Continue checkpointed draws with identical inputs and configuration")
    parser.add_argument("--events-only",action="store_true",help="Complete event-programme draws for an existing matching audit")
    parser.add_argument("--large-coherent-only",action="store_true",help="Complete the two large-family draws after an existing matching audit")
    args=parser.parse_args()
    baseline=deepcopy(CONFIG)
    fingerprint=runtime_fingerprint()
    if args.events_only:
        report=json.loads(OUTPUT.read_text())
        assert report["status"]=="pass" and report["runtime_input_fingerprint"]==fingerprint
        assert report["configuration"]==baseline
        assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==value for path,value in report["generator_sha256"].items())
        done={(r["family"],r["seed"]) for r in report.setdefault("event_ensembles",[])}
        report["status"]="running"
        try:
            for seed in sorted({r["seed"] for r in report["component_ensembles"]}):
                for root in sorted((ROOT/"data/datasets").glob("rec_pt_2023_*_v2")):
                    if (root.name,seed) in done:continue
                    report["event_ensembles"].append(event_draw(root,seed))
                    ev.dump(OUTPUT,report)
                print('event seed',seed,'pass',flush=True)
            assert runtime_fingerprint()==fingerprint
            summarize(report)
            report["status"]="pass"
        except Exception as error:
            report["status"]="fail";report["error"]=repr(error)
            raise
        finally:
            CONFIG.clear();CONFIG.update(baseline);ev.dump(OUTPUT,report)
        return
    if args.large_coherent_only:
        report=json.loads(OUTPUT.read_text())
        assert report["status"]=="pass" and report["runtime_input_fingerprint"]==fingerprint
        assert report["configuration"]==baseline
        done={(r["family"],r["seed"]) for r in report["coherent_ensembles"]}
        report["status"]="running"
        try:
            for seed in [baseline["seed"]+1009*(i+1) for i in range(2)]:
                for family in ("core_30","premium_100"):
                    root=ROOT/f"data/datasets/rec_pt_2023_{family}_v2"
                    if (root.name,seed) in done:continue
                    report["coherent_ensembles"].append(coherent_draw(root,seed))
                    print('large coherent',family,seed,'pass',flush=True)
                    ev.dump(OUTPUT,report)
            assert runtime_fingerprint()==fingerprint
            summarize(report)
            report["status"]="pass"
        except Exception as error:
            report["status"]="fail";report["error"]=repr(error)
            raise
        finally:
            CONFIG.clear();CONFIG.update(baseline);ev.dump(OUTPUT,report)
        return
    selected=cases()
    report=dict(revision="portugal-ensemble-1",runtime_input_fingerprint=fingerprint,
        configuration=baseline,component_ensembles=[],coherent_ensembles=[],sensitivity=[],event_ensembles=[],
        scope="Monte Carlo component diagnostics on representative residential routines in all four families. One-at-a-time sensitivity is local, not global uncertainty quantification. Other end-use/appliance components held at final realization in component diagnostics; coherent runs regenerate the complete chain in RAM.")
    seeds=[baseline["seed"]+1009*(i+1) for i in range(args.seed_count)]
    if args.resume:
        previous=json.loads(OUTPUT.read_text())
        assert previous["runtime_input_fingerprint"]==fingerprint and previous["configuration"]==baseline
        assert all(r.get("status","pass")=="pass" for r in previous["coherent_ensembles"])
        for name in ("component_ensembles","coherent_ensembles","sensitivity"):
            report[name]=previous[name]
        report["event_ensembles"]=previous.get("event_ensembles",[])
        # Upgrade earlier checkpoint evidence with direct appliance outcomes,
        # without rerunning already verified EV plans or changing datasets.
        for row in report["coherent_ensembles"]+report["sensitivity"]:
            if "distributions" not in row or "appliance_metrics" in row:continue
            CONFIG.clear();CONFIG.update(deepcopy(baseline))
            if "parameter" in row:
                section,parameter=row["parameter"].split(".")
                CONFIG[section][parameter]=row["value"]
            row["appliance_metrics"]=physical_appliance_draw(ROOT/"data/datasets"/row["family"],row["seed"])
            print('appliance outcomes',row['family'],row['seed'],row.get('parameter','baseline'),row.get('factor',''),flush=True)
            ev.dump(OUTPUT,report)
        CONFIG.clear();CONFIG.update(deepcopy(baseline))
    component_done={(r["family"],r["seed"]) for r in report["component_ensembles"]}
    coherent_done={(r["family"],r["seed"]) for r in report["coherent_ensembles"]}
    sensitivity_done={(r["parameter"],r["factor"],r["seed"]) for r in report["sensitivity"]}
    event_done={(r["family"],r["seed"]) for r in report["event_ensembles"]}
    try:
        for seed in seeds:
            for case in selected:
                if (case[0].name,seed) not in event_done:
                    report["event_ensembles"].append(event_draw(case[0],seed))
                if (case[0].name,seed) in component_done:continue
                report["component_ensembles"].append(component_draw(case,seed))
            print('component seed',seed,'pass',flush=True)
            ev.dump(OUTPUT,report)
        # Same seeds and inventory for each +/-25% parameter experiment.
        parameters=[("demand","daily_log_sigma"),("demand","episode_fraction"),
            ("demand","episodes_per_day"),("demand","episode_persistence"),
            ("activity","weekend_trip_rate_gamma_scale"),("activity","leave_start_jitter_days"),
            ("events","local_outages_per_member_year"),("events","outage_median_minutes")]
        premium=next(c for c in selected if "premium" in c[0].name)
        for section,parameter in parameters:
            for factor in (.75,1.25):
                CONFIG.clear();CONFIG.update(deepcopy(baseline))
                CONFIG[section][parameter]=baseline[section][parameter]*factor
                for seed in seeds[:3]:
                    if (f"{section}.{parameter}",factor,seed) in sensitivity_done:continue
                    row=component_draw(premium,seed)
                    row.update(parameter=f"{section}.{parameter}",factor=factor,value=CONFIG[section][parameter])
                    report["sensitivity"].append(row)
                print('sensitivity',section,parameter,factor,'pass',flush=True)
                ev.dump(OUTPUT,report)
        CONFIG.clear();CONFIG.update(deepcopy(baseline))
        for seed in seeds[:args.coherent_count]:
            for family in ("micro_4","core_15"):
                if (f"rec_pt_2023_{family}_v2",seed) in coherent_done:continue
                row=coherent_draw(ROOT/f"data/datasets/rec_pt_2023_{family}_v2",seed)
                report["coherent_ensembles"].append(row)
                print('coherent',family,seed,'pass',flush=True)
                ev.dump(OUTPUT,report)
        for seed in seeds[:2]:
            for family in ("core_30","premium_100"):
                if (f"rec_pt_2023_{family}_v2",seed) in coherent_done:continue
                row=coherent_draw(ROOT/f"data/datasets/rec_pt_2023_{family}_v2",seed)
                report["coherent_ensembles"].append(row)
                print('coherent',family,seed,'pass',flush=True)
                ev.dump(OUTPUT,report)
        # Charging/appliance parameters need the coherent chain, not a load-only
        # proxy. Use identical small-community seeds for paired sensitivity.
        for section,parameter in (("charging","threshold_std"),("charging","delayed_connection_probability"),
                                  ("appliances","empirical_energy_jitter_sigma"),("appliances","slack_log_sigma")):
            for factor in (.75,1.25):
                CONFIG.clear();CONFIG.update(deepcopy(baseline))
                CONFIG[section][parameter]=baseline[section][parameter]*factor
                for seed in seeds[:3]:
                    if (f"{section}.{parameter}",factor,seed) in sensitivity_done:continue
                    row=coherent_draw(ROOT/"data/datasets/rec_pt_2023_micro_4_v2",seed)
                    row.update(parameter=f"{section}.{parameter}",factor=factor,value=CONFIG[section][parameter])
                    report["sensitivity"].append(row)
                print('coherent sensitivity',parameter,factor,'pass',flush=True)
                ev.dump(OUTPUT,report)
        assert runtime_fingerprint()==fingerprint,"Ensemble audit modified executable files"
        report["status"]="pass"
        summarize(report)
    except Exception as error:
        report["status"]="fail";report["error"]=repr(error)
        raise
    finally:
        CONFIG.clear();CONFIG.update(baseline)
        ev.dump(OUTPUT,report)


if __name__=="__main__": main()

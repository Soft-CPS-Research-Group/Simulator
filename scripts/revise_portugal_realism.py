"""Offline, reproducible revision of activity, appliances and event programmes.

Run with --apply; immutable inputs are read from the pinned Git source used by
the EV revision. No dataset copies, archives, simulator edits or commits. The
same keyed realizations feed every matched experimental arm. The config records
the distinction between empirical inputs, structural references and priors.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from portugal_activity import CONFIG, N, N23, DT, at, build_calendar, presence, rng_for
from revise_portugal_ev_mobility import ROOT, SOURCE, source_frame, write, dump
from apply_portugal_omie_pricing import refresh_checksums
from prepare_portugal_heldout_2024 import active_masks, preparation_view
from portugal_stochastic_demand import residential_shape, bounded_energy_allocation
from portugal_commercial_activity import commercial_end_uses


@lru_cache(maxsize=1)
def appliance_development_candidates():
    """Only predeclared development households may influence generation."""
    reference=ROOT/"data/portugal_validation_reference"
    rows=pd.concat([pd.read_parquet(reference/f"refit_house_{house}_cycles.parquet") for house in (3,5)],ignore_index=True)
    return rows.loc[rows.threshold_w.eq(10)&rows.gap_seconds.eq(900)&rows.split.eq("development")]


def empirical_programs(root,appliance):
    candidates=appliance_development_candidates()
    candidates=candidates.loc[candidates.appliance_type.eq(appliance.appliance_type)]
    rng=rng_for(root,appliance.deferrable_id,0,"measured_program_variants")
    cfg=CONFIG["appliances"];rows=[]
    for k in range(cfg["empirical_program_variants"]):
        house=int(rng.choice([3,5]));pool=candidates.loc[candidates.house.eq(house)]
        p=pool.iloc[int(rng.integers(len(pool)))]
        values=np.array(json.loads(p.profile_15min_kwh),dtype=float)
        sigma=cfg["empirical_energy_jitter_sigma"]
        values*=float(np.clip(rng.lognormal(-.5*sigma*sigma,sigma),.75,1.25))
        cap=appliance.nominal_power_kw*DT
        # Quarter-hour mean powers must respect the installed machine. Extend
        # only when necessary to retain the entire sampled programme energy.
        if values.sum()>len(values)*cap:
            length=int(np.ceil(values.sum()/cap))
            values=np.diff(np.interp(np.linspace(0,len(values),length+1),np.arange(len(values)+1),np.r_[0,values.cumsum()]))
        values=bounded_energy_allocation(values,np.full(len(values),cap),values.sum())
        rows.append(dict(profile_id=f"{appliance.deferrable_id}-REFIT-{k:02d}",program="measured_candidate_variant",
            duration_steps=len(values),total_energy_kwh=float(values.sum()),load_profile=json.dumps(values.tolist()),
            reference_house=house,reference_start_utc=str(p.start_utc)))
    return pd.DataFrame(rows)


def profiles_for(root, appliance):
    if appliance.equipment_use in ("household_laundry","household_dishes","staff_kitchen_dishes"):
        return empirical_programs(root,appliance)
    if appliance.equipment_use=="linen_laundry":
        cfg=CONFIG["appliances"]["linen_reference"]
        rng=rng_for(root,appliance.deferrable_id,0,"linen_engineering_programs");rows=[]
        for k in range(CONFIG["appliances"]["empirical_program_variants"]):
            fraction=1. if k%8==0 else .45+.55*rng.beta(5,2)
            load_kg=appliance.rated_capacity*fraction
            temperature=int(rng.choice(cfg["temperature_celsius"],p=cfg["temperature_probabilities"]))
            motor=cfg["motor_component_kwh_prior"]
            heating=(cfg["full_load_cold_60c_kwh"]-motor)*load_kg/cfg["full_load_kg"]*(temperature-15)/45
            energy=motor+heating
            available_heater=max(.1,appliance.nominal_power_kw-cfg["auxiliary_power_kw_prior"])
            extra_minutes=max(0.,heating*(1/available_heater-1/cfg["reference_heater_kw"])*60)
            minutes=cfg["cold_60c_minutes"]*rng.uniform(.9,1.1)+extra_minutes
            length=max(int(np.ceil(minutes/15)),int(np.ceil(energy/(appliance.nominal_power_kw*DT))))
            # Engineering stage envelope, not a measured sub-hour trace.
            x=np.arange(length,dtype=float)
            weights=.15+np.exp(-.5*((x-.3*length)/max(.6,.18*length))**2)
            values=bounded_energy_allocation(weights,np.full(length,appliance.nominal_power_kw*DT),energy)
            rows.append(dict(profile_id=f"{appliance.deferrable_id}-LINEN-{k:02d}",
                program="cold_water_professional_linen_engineering_prior",duration_steps=length,
                total_energy_kwh=energy,load_profile=json.dumps(values.tolist()),load_kg=load_kg,
                wash_temperature_c=temperature,reference="Miele-PWM912-PT",external_hot_water_kwh=0.))
        return pd.DataFrame(rows)
    if appliance.equipment_use=="catering_dishwashing_batch":
        rng=rng_for(root,appliance.deferrable_id,0,"catering_batches");rows=[]
        for k in range(CONFIG["appliances"]["empirical_program_variants"]):
            length=int(rng.choice([1,2,3],p=[.7,.25,.05]))
            energy=float(np.clip(rng.lognormal(np.log(.45),.35),.12,1.1))
            values=np.exp(-np.arange(length,dtype=float));values*=energy/values.sum()
            assert values.max()<=appliance.nominal_power_kw*DT
            rows.append(dict(profile_id=f"{appliance.deferrable_id}-BATCH-{k:02d}",program="professional_batch_15min_aggregate_prior",
                duration_steps=length,total_energy_kwh=energy,load_profile=json.dumps(values.tolist())))
        return pd.DataFrame(rows)
    old = source_frame(root/f"deferrables/{appliance.deferrable_id}_profiles.parquet")
    rng = rng_for(root, appliance.deferrable_id, 0, "program_variants")
    cfg = CONFIG["appliances"]
    rows = []
    for p in old.itertuples():
        original = np.array(json.loads(p.load_profile))
        for k in range(cfg["variants_per_program"]):
            length = max(2, int(round(len(original)*rng.uniform(*cfg["duration_factors"]))))
            # Resample the cumulative energy, conserving integrated heater/motor
            # stages. Perturb complete programme energy, not independent samples.
            values = np.diff(np.interp(np.linspace(0, len(original), length+1),
                                      np.arange(len(original)+1), np.r_[0, np.cumsum(original)]))
            values *= np.clip(rng.lognormal(-.5*cfg["energy_log_sigma"]**2, cfg["energy_log_sigma"]), .7, 1.3)
            cap = appliance.nominal_power_kw*DT
            assert values.sum() < length*cap
            for _ in range(length+1):
                excess = np.maximum(values-cap, 0.).sum()
                if excess < 1e-12:
                    break
                values = np.minimum(values, cap)
                spare = values < cap-1e-12
                values[spare] += excess/spare.sum()
            rows.append(dict(profile_id=f"{p.profile_id}-V{k:02d}", program=p.program,
                             duration_steps=length, total_energy_kwh=float(values.sum()),
                             load_profile=json.dumps(values.tolist())))
    return pd.DataFrame(rows)


def appliance_schedule(root, m, app, profiles, calendar, daily, occupancy):
    clock = pd.DatetimeIndex(calendar.timestamp_utc)
    old = source_frame(root/f"deferrables/{app.deferrable_id}_schedule.parquet")
    residential = m.category == "residential"
    catering=app.equipment_use=="catering_dishwashing_batch"
    opened = occupancy > 0 if residential else pd.read_parquet(root/f"operations/{m.member_id}.parquet").is_open.to_numpy()
    fixed = np.zeros(N)
    rows = []
    last = 0
    intensity_by_date = daily.set_index("date").activity_intensity.to_dict()
    if not residential:
        # Keep the declared annual use-rate prior, but vary actual workload with
        # the relevant people, rather than borrowing electrical sensor noise.
        column=("staff_person_hours" if app.equipment_use=="staff_kitchen_dishes" else
                "resident_person_hours" if m.routine_archetype=="care_home" else "client_person_hours")
        for _,group in daily.groupby("year"):
            normal=group.loc[group.is_service_day,column].mean()
            if normal>0:
                for row in group.itertuples():
                    intensity_by_date[row.date]=.3*row.activity_intensity+.7*getattr(row,column)/normal
    for year in (2023, 2024):
        rng = rng_for(root, app.deferrable_id, year, "cycle_schedule")
        lo, hi = (0, N23) if year == 2023 else (N23, N)
        old_year = old.loc[old.earliest_start_time_step.between(lo, hi-1)]
        # Original inventory/use intensity is a prior. Daily realization is new.
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", tz="Europe/Lisbon")
        possible = sum(opened[at(clock, day, 0):at(clock, day, 24)].any() for day in dates)
        rate = CONFIG["appliances"]["catering_batches_per_service_day"] if catering else len(old_year)/max(possible, 1)
        for day in dates:
            a, b = at(clock, day, 0), at(clock, day, 24)
            rest = bool(calendar.is_rest_day.iloc[a])
            intensity = intensity_by_date.get(day.strftime("%Y-%m-%d"), 1.)
            count = min(12 if catering else 3, int(rng.poisson(rate*intensity*(1.15 if rest and residential else 1.))))
            if not count:
                continue
            for job in range(count):
                probabilities=None if "reference_house" in profiles or "load_kg" in profiles else np.r_[np.full(len(profiles)//2,.7),np.full(len(profiles)//2,.3)]/(len(profiles)//2)
                p = profiles.iloc[int(rng.choice(len(profiles),p=probabilities))]
                values = np.array(json.loads(p.load_profile))
                length = len(values)
                earliest = max(a, last)
                close = min(b, at(clock, day, 23.5)) if residential else b
                slots = np.arange(earliest, max(earliest, close-length+1))
                # Loading requires presence/opening; fixed operation also ends
                # before absence/closure. Delayed starts may run while at work.
                valid = np.array([opened[t] and (opened[t:t+length].all() if not app.controlled or not residential else True) for t in slots], dtype=bool)
                slots = slots[valid]
                if catering:
                    # The first open quarter-hour heats the tank. It cannot
                    # simultaneously offer a wash batch at the full power limit.
                    service_slots=np.flatnonzero(opened[a:b])+a
                    if len(service_slots):slots=slots[slots>service_slots[0]]
                if not len(slots):
                    continue
                hours = clock[slots].tz_convert("Europe/Lisbon").hour.to_numpy()+clock[slots].tz_convert("Europe/Lisbon").minute.to_numpy()/60
                weights = np.exp(-.5*((hours-(12 if rest else 19)) / (3 if rest else 2))**2)
                if residential and app.controlled:
                    weights += .35*np.exp(-.5*((hours-7)/.7)**2)
                if not residential:
                    weights = np.ones(len(slots))
                    if catering:
                        service=np.flatnonzero(opened[a:b])+a
                        preferred=service[0]+(job+rng.uniform(.2,.8))*len(service)/count
                        weights=np.exp(-.5*((slots-preferred)/2)**2)+1e-100
                start = int(rng.choice(slots, p=weights/weights.sum()))
                center = {"narrow": 4, "moderate": 12, "wide": 24}[m.flexibility_preference]
                slack = max(0, int(round(rng.lognormal(np.log(center), CONFIG["appliances"]["slack_log_sigma"]))))
                if not residential:
                    slack = int(rng.integers(0, 3 if catering else 9))
                    closed = np.flatnonzero(~opened[start:close])
                    if len(closed):
                        close = start+int(closed[0])
                else:
                    # Never leave a cycle pending across a multi-day absence.
                    absences = daily.loc[(daily.absence_start > start) & (daily.absence_start < close), "absence_start"]
                    if len(absences):
                        close = int(absences.min())
                end = min(close, start+length+slack)
                if start+length > end:
                    continue
                rows.append(dict(cycle_id=f"{app.deferrable_id}-R1-{len(rows):05d}", profile_id=p.profile_id,
                    earliest_start_time_step=start, latest_start_time_step=end-length if app.controlled else start,
                    deadline_time_step=end-1 if app.controlled else start+length-1,
                    priority=float(np.round(.35+.65*rng.beta(2, 2), 3)), must_run=True,
                    day_kind=str(calendar.day_kind.iloc[start]), year=year))
                if not app.controlled:
                    fixed[start:start+length] += values
                last = end
    return pd.DataFrame(rows), fixed


def catering_support(root,app,operations,daily,calendar):
    """Unshiftable warm-up/holding energy, separate from flexible wash batches.

    Numeric equipment priors, not fitted FIKElectricity machine parameters.
    Reclassified from the site's background budget to avoid double counting.
    """
    result=np.zeros(N);clock=pd.DatetimeIndex(calendar.timestamp_utc)
    opened=operations.is_open.to_numpy()
    cfg=CONFIG["appliances"]
    for year,group in daily.groupby("year"):
        rng=rng_for(root,app.deferrable_id,year,"catering_support")
        for day in group.itertuples():
            d=pd.Timestamp(day.date,tz="Europe/Lisbon");a,b=at(clock,d,0),at(clock,d,24)
            slots=np.flatnonzero(opened[a:b])+a
            if not len(slots):continue
            standby=rng.uniform(*cfg["catering_holding_kw"])
            result[slots]=standby*DT
            result[slots[0]]+=rng.uniform(.5,1.)
    assert result.max()/DT<=app.nominal_power_kw
    return result


def revise_physical(root, calendar):
    members = source_frame(root/"catalogs/members.parquet")
    daily = build_calendar(root, members, calendar)
    write(root/"catalogs/household_activity.parquet", daily)
    clock = pd.DatetimeIndex(calendar.timestamp_utc)
    apps = source_frame(root/"catalogs/deferrables.parquet")
    schema = json.loads((root/"schema.json").read_text())
    totals = [];commercial_days=[]
    for m in members.itertuples():
        days = daily.loc[daily.member_id.eq(m.member_id)]
        occupancy = presence(clock, days, m.occupants) if m.category == "residential" else np.ones(N)
        ledger = source_frame(root/f"end_uses/{m.member_id}.parquet").astype(float)
        frame = source_frame(root/f"timeseries/{m.member_id}.parquet")
        original = ledger.copy()
        if m.category == "residential":
            ledger["base_excluding_cycles_kwh"] = residential_shape(root, m,
                original.base_excluding_cycles_kwh, calendar, days, occupancy)
            ledger.heating_proxy_kwh *= .35+.65*occupancy
            ledger.cooling_proxy_kwh *= .2+.8*occupancy
            fraction = CONFIG["activity"]["dhw_standing_loss_fraction"]
            for lo, hi in ((0, N23), (N23, N)):
                draw = original.dhw_proxy_kwh.iloc[lo:hi].to_numpy()
                ledger.loc[lo:hi-1, "dhw_proxy_kwh"] = fraction*draw.mean()+(1-fraction)*draw*occupancy[lo:hi]
            frame["occupant_count"] = (occupancy*m.occupants).astype("float32")
        else:
            ledger,occupancy,days,operations=commercial_end_uses(root,m,calendar,original)
            frame["occupant_count"]=occupancy.astype("float32")
            commercial_days.append(days)
            write(root/f"operations/{m.member_id}.parquet",operations)
        for year, lo, hi in ((2023, 0, N23), (2024, N23, N)):
            rng = rng_for(root, m.member_id, year, "annual_activity")
            sigma = CONFIG["activity"]["annual_load_log_sigma"]
            annual_factor = float(rng.lognormal(-.5*sigma*sigma, sigma)) if year == 2024 else 1.
            for col in ("base_excluding_cycles_kwh", "heating_proxy_kwh", "cooling_proxy_kwh"):
                goal = (original if m.category=="residential" else ledger)[col].iloc[lo:hi].sum()*annual_factor
                ledger.loc[lo:hi-1, col] *= goal/max(1e-12, ledger[col].iloc[lo:hi].sum())
            # DHW absences genuinely reduce draws; do not restore lost draws by
            # annual normalization. Standing losses are retained explicitly.
            ledger.loc[lo:hi-1, "dhw_proxy_kwh"] *= annual_factor
        ledger["fixed_appliances_kwh"] = 0.
        appliance_requested=np.zeros(2)
        building_budget=[float(ledger.iloc[lo:hi].drop(columns="fixed_appliances_kwh").to_numpy().sum()) for lo,hi in ((0,N23),(N23,N))]
        if m.category!="residential":ledger["professional_service_kwh"]=0.
        for ix, app in apps.loc[apps.member_id.eq(m.member_id)].iterrows():
            app = app.to_frame().T.itertuples(index=False).__next__()
            pro = profiles_for(root, app)
            schedule, fixed = appliance_schedule(root, m, app, pro, calendar, days, occupancy)
            write(root/f"deferrables/{app.deferrable_id}_profiles.parquet", pro)
            write(root/f"deferrables/{app.deferrable_id}_schedule.parquet", schedule)
            ledger.fixed_appliances_kwh += fixed
            for yi,year in enumerate((2023,2024)):
                appliance_requested[yi]+=schedule.loc[schedule.year.eq(year),"profile_id"].map(pro.set_index("profile_id").total_energy_kwh).sum()
            if app.equipment_use=="catering_dishwashing_batch":
                ledger.professional_service_kwh+=catering_support(root,app,operations,days,calendar)
            apps.loc[ix, "annual_cycle_count"] = int(schedule.year.eq(2023).sum())
            apps.loc[ix, "annual_requested_energy_kwh"] = schedule.loc[schedule.year.eq(2023), "profile_id"].map(pro.set_index("profile_id").total_energy_kwh).sum()
            apps.loc[ix, "minimum_profile_energy_kwh"] = pro.total_energy_kwh.min()
            apps.loc[ix, "maximum_profile_energy_kwh"] = pro.total_energy_kwh.max()
            apps.loc[ix, "maximum_profile_power_kw"] = max(max(json.loads(p))/DT for p in pro.load_profile)
            apps.loc[ix, "profile_method"] = ("REFIT development houses 3/5: house-balanced measured candidates resampled at 15 min; energy jitter and use/deadlines are priors" if "reference_house" in pro else "Miele Portuguese cold-water 12 kg 60C specification; load/temperature/duration transfer and throughput priors, installed power retained" if "load_kg" in pro else "Professional catering aggregate prior, not measured cycles; warm-up/holding is non-shiftable and excluded from flexible batches")
        if "professional_service_kwh" in ledger:
            for yi,(lo,hi) in enumerate(((0,N23),(N23,N))):
                base=ledger.base_excluding_cycles_kwh.iloc[lo:hi].sum()
                support=ledger.professional_service_kwh.iloc[lo:hi].sum()
                # Annual references describe the complete building. Reclassify
                # all programmes out of that budget, including controlled ones
                # added by the simulator later, so none is counted twice.
                reserved=support+appliance_requested[yi]
                assert reserved<base
                ledger.loc[lo:hi-1,"base_excluding_cycles_kwh"]*=1-reserved/base
        frame["non_shiftable_load"] = ledger.sum(axis=1).astype("float32")
        # Coincident aggregate episodes must respect the installed service.
        # Redistribute base energy inside each year; never discard energy.
        limit = schema["buildings"][m.member_id]["electrical_service"]["limits"]["total"]["import_kw"]*DT
        other = ledger.drop(columns="base_excluding_cycles_kwh").sum(axis=1).to_numpy()
        assert (other <= limit+1e-7).all(), (m.member_id, "non-base demand exceeds service")
        for lo, hi in ((0,N23),(N23,N)):
            base = ledger.base_excluding_cycles_kwh.iloc[lo:hi].to_numpy()
            ledger.loc[lo:hi-1,"base_excluding_cycles_kwh"] = bounded_energy_allocation(base, limit-other[lo:hi], base.sum())
        frame["non_shiftable_load"] = ledger.sum(axis=1).astype("float32")
        write(root/f"end_uses/{m.member_id}.parquet", ledger.astype("float32"))
        write(root/f"timeseries/{m.member_id}.parquet", frame)
        for year, lo, hi in ((2023, 0, N23), (2024, N23, N)):
            totals.append(dict(member_id=m.member_id, year=year,
                non_shiftable_load_kwh=float(frame.non_shiftable_load.iloc[lo:hi].sum()),
                building_budget_excluding_ev_kwh=building_budget[year-2023] if m.category!="residential" else None,
                pv_generation_kwh=float(frame.solar_generation.iloc[lo:hi].sum()),
                heating_proxy_kwh=float(ledger.heating_proxy_kwh.iloc[lo:hi].sum()),
                cooling_proxy_kwh=float(ledger.cooling_proxy_kwh.iloc[lo:hi].sum())))
    write(root/"catalogs/deferrables.parquet", apps)
    if commercial_days:
        write(root/"catalogs/commercial_activity.parquet",pd.concat(commercial_days,ignore_index=True))
    write(root/"catalogs/member_year_totals.parquet", pd.DataFrame(totals))
    return daily


def random_window(root, key, year, duration, weights=None):
    rng = rng_for(root, key, year, "event_time")
    lo, hi = (0, N23) if year == 2023 else (N23, N)
    # Reserve both year edges; preparatory views never include a future event.
    slots = np.arange(lo+96, hi-duration-96)
    probability = None if weights is None else weights[slots]/weights[slots].sum()
    start = int(rng.choice(slots, p=probability))
    return start, start+duration-1


def revise_events(root):
    """Compile latent events once per cause, then expand equivalent channels."""
    cfg = CONFIG["events"]
    for path in sorted((root/"evaluation").rglob("*.parquet")):
        if "preparation_2023" in str(path) or path.name not in ("health.parquet", "observations_and_health.parquet", "measurement_noise.parquet", "robustness.parquet", "confirmation_robustness.parquet", "requests.parquet"):
            continue
        old = source_frame(path)
        if old.empty:
            continue
        if "request_id" in old:
            rows = []
            prices = pd.read_parquet(root/"shared/pricing_omie.parquet").electricity_pricing.to_numpy()
            weather = pd.read_parquet(root/"shared/weather.parquet").diffuse_solar_irradiance.to_numpy()
            for row in old.to_dict("records"):
                year = 2023 if row["start_time_step"] < N23 else 2024
                rng = rng_for(root, row["request_id"], year, "dr")
                stress = row["stratum"] == "impossible_stress"
                duration = 16 if stress else int(rng.integers(2, 9))
                weights = .2+prices/max(prices.mean(), 1e-9) if row["direction"] == "down" else .2+weather/max(weather.mean(), 1e-9)
                row["start_time_step"], row["end_time_step"] = random_window(root, row["request_id"], year, duration, weights)
                if not stress:
                    row["target_power_kw"] = round(row["target_power_kw"]*float(np.clip(rng.lognormal(0, cfg["dr_target_log_sigma"]), .6, 1.4)), 2)
                    row["tolerance_power_kw"] = .05*row["target_power_kw"]
                row["generation_regime"] = "controlled_stress" if stress else "synthetic_service_request"
                rows.append(row)
            result = pd.DataFrame(rows).sort_values("start_time_step")
            assert (result.start_time_step.to_numpy()[1:] > result.end_time_step.to_numpy()[:-1]).all()
            write(path, result)
            continue
        old["cause"] = old.event_id.str.replace(r"_channel_\d+", "", regex=True).str.replace(r"_2024$", "", regex=True)
        rows = []
        for (cause, start), group in old.groupby(["cause", "start_time_step"], sort=True):
            year = 2023 if start < N23 else 2024
            continuous = start == 0 and group.end_time_step.max() == N-1
            noise = "measurement_noise" in cause
            pv = "pv_missing" in cause or (group.target_type.eq("pv").any() and group["mode"].eq("missing").any())
            groups = []
            if pv and group.target_type.eq("pv").any():
                members = sorted({str(v).split("/")[0] for v in group.loc[group.target_type.eq("pv"), "target_id"]})
                for mid in members:
                    # Community PV is computed from the same missing sensor
                    # feeds, not an independent perfect measurement. Hide its
                    # two aggregate aliases during each local dropout as well.
                    local = group.loc[group.target_id.isin([mid, mid+"/pv"]) | group.target_type.eq("district")].copy()
                    rng = rng_for(root, cause+mid, year, "pv_fault_count")
                    for k in range(int(rng.poisson(cfg["local_pv_faults_per_member_year"]))):
                        groups.append((cause+"_local_"+mid+f"_{k}", local, "local_fault"))
                rng = rng_for(root, cause, year, "common_pv")
                if rng.random() < cfg["common_pv_telemetry_probability_per_year"]:
                    groups.append((cause+"_common", group, "common_cause_fault"))
            else:
                groups.append((cause, group, "controlled_noise_stress" if noise else "controlled_stress"))
            for key, local, regime in groups:
                if noise and ("confirmation" in path.parts or path.name.startswith("confirmation_")):
                    # A reserved confirmation programme needs an independent
                    # realization, including continuous-noise event IDs (used
                    # by the runtime to derive deterministic random errors).
                    key = "confirmation_"+key
                timing_key = key.replace("_domain", "") if noise else key
                rng = rng_for(root, timing_key, year, "event_duration")
                duration = int(group.end_time_step.iloc[0]-start+1)
                if not continuous:
                    duration = max(1, int(round(duration*np.clip(rng.lognormal(0, cfg["event_duration_log_sigma"]), .4, 2.))))
                    a, b = random_window(root, timing_key, year, duration)
                else:
                    a, b = 0, N-1
                for row in local.drop(columns="cause").to_dict("records"):
                    row.update(start_time_step=a, end_time_step=b, cause_id=f"{key}_{year}", generation_regime=regime)
                    # One native event ID per feature/target; true dependencies
                    # retain identical times, severities and replacement values.
                    row["event_id"] = f"{key}_{year}_{row['target_type']}_{row['target_id']}_{row['target_feature']}_{row['mode']}"
                    rows.append(row)
        result = pd.DataFrame(rows).sort_values(["start_time_step", "event_id"]).reset_index(drop=True)
        assert result.event_id.is_unique, path
        write(path, result)


def outage_programme(root, mid):
    cfg = CONFIG["events"]
    rows = []
    for year in (2023, 2024):
        rng = rng_for(root, mid, year, "grid_local")
        for k in range(int(rng.poisson(cfg["local_outages_per_member_year"]))):
            duration = int(np.clip(round(rng.lognormal(np.log(cfg["outage_median_minutes"]), cfg["outage_log_sigma"])/15), 1, 12))
            a, b = random_window(root, f"grid_{mid}_{k}", year, duration)
            rows.append(dict(member_id=mid, start_time_step=a, end_time_step=b, generation_regime="local_fault", cause_id=f"grid_{mid}_{year}_{k}"))
        common = rng_for(root, "grid", year, "common_occurrence").random() < cfg["common_outage_probability_per_year"]
        if common:
            common_rng = rng_for(root, "grid", year, "common_duration")
            duration = int(np.clip(round(common_rng.lognormal(np.log(cfg["common_outage_median_minutes"]),cfg["outage_log_sigma"])/15),1,12))
            a, b = random_window(root, "grid_common", year, duration)
            rows.append(dict(member_id=mid, start_time_step=a, end_time_step=b, generation_regime="common_cause_fault", cause_id=f"grid_common_{year}"))
        # Explicit common-mode benchmark stress remains available independently
        # of whether a rare naturalistic common event happened in this draw.
        a, b = random_window(root, "grid_controlled_stress", year, int(cfg["controlled_grid_stress_hours_per_year"]/DT))
        rows.append(dict(member_id=mid, start_time_step=a, end_time_step=b, generation_regime="controlled_stress", cause_id=f"grid_stress_{year}"))
    return rows


def schemas_in(root):
    result = {}
    for p in [root/"schema.json", *sorted((root/"evaluation").rglob("*.json"))]:
        s = json.loads(p.read_text())
        if isinstance(s, dict) and "buildings" in s:
            result[p] = s
    return result


def describe_current_inputs(root, path, schema):
    """Keep explanatory metadata synchronized without changing execution."""
    design = schema["experimental_design"]
    noise = design.get("measurement_noise", {})
    if noise:
        noise["temporal_exposure"] = "Seeded channel-specific stress exposures; see actual event files. Noise/clip pairs share one window; not observed sensor reliability."
        for phase in list(noise.get("schedules", {})):
            source = root/f"evaluation/events/{phase}/measurement_noise.parquet"
            if source.exists():
                f = pd.read_parquet(source)
                if path.parent.name == "preparation_2023":
                    f = f.loc[f.start_time_step.lt(N23)].copy()
                    f["end_time_step"] = f.end_time_step.clip(upper=N23-1)
                noise["schedules"][phase] = f[["start_time_step", "end_time_step"]].drop_duplicates().sort_values("start_time_step").to_dict("records")
    return schema


def propagate(root):
    schemas = schemas_in(root)
    changed = set()
    outage_rows = []
    members = pd.read_parquet(root/"catalogs/members.parquet").set_index("member_id")
    supplemental_path=root/"evaluation/campaign_2024/catalogue.json"
    templates={m["member_id"]:m["profile_template_member"] for m in json.loads(supplemental_path.read_text()).get("supplemental_members",[])} if supplemental_path.exists() else {}
    for path, schema in schemas.items():
        for mid, b in schema["buildings"].items():
            energy_path = root/b["energy_simulation"]
            donor=mid if mid in members.index else templates.get(mid)
            if donor is None or energy_path in changed:
                continue
            original = source_frame(energy_path)
            if len(original) != N:
                continue
            physical = pd.read_parquet(root/f"timeseries/{donor}.parquet")
            old_physical = source_frame(root/f"timeseries/{donor}.parquet")
            actual = original.copy()
            live, pv, battery = active_masks(schema, mid)
            # Keep source historical overlays (including absent asset standby)
            # and replace only their physical demand realization.
            actual["non_shiftable_load"] = np.where(live, np.maximum(0., original.non_shiftable_load.to_numpy()+physical.non_shiftable_load.to_numpy()-old_physical.non_shiftable_load.to_numpy()), 0.).astype("float32")
            actual["occupant_count"] = np.where(live, physical.occupant_count, 0).astype("float32")
            if original.power_outage.any():
                actual["power_outage"] = np.zeros(N, dtype="float32")
                records = outage_programme(root, mid)
                for event in records:
                    a, end = event["start_time_step"], event["end_time_step"]+1
                    actual.loc[a:end-1, "power_outage"] = live[a:end].astype("float32")
                    outage_rows.append(dict(input_file=str(energy_path.relative_to(root)), **event,
                                            active_slots=int(live[a:end].sum())))
                actual["solar_generation"] = np.where(live & pv & (~actual.power_outage.gt(0).to_numpy() | battery), physical.solar_generation, 0).astype("float32")
            write(energy_path, actual)
            changed.add(energy_path)
    # The same programme is filtered by member/asset availability, never newly
    # sampled for a scenario. Preserve complete cycles at lifecycle boundaries.
    for path, schema in schemas.items():
        for mid, b in schema["buildings"].items():
            for aid, app in b.get("deferrable_appliances", {}).items():
                target = root/app["flexibility_schedule_file"]
                if target in changed:
                    continue
                base = root/f"deferrables/{aid}_schedule.parquet"
                if not base.exists():
                    continue
                frame = pd.read_parquet(base)
                live, _, _ = active_masks(schema, mid)
                for e in sorted(schema.get("topology_events", []), key=lambda x:x["time_step"]):
                    if e.get("target_asset_id") == aid:
                        live[e["time_step"]:] &= e["operation"] != "remove_asset"
                valid = [bool(live[int(s.earliest_start_time_step):int(s.deadline_time_step)+1].all()) for s in frame.itertuples()]
                write(target, frame.loc[valid].reset_index(drop=True))
                changed.add(target)
    write(root/"catalogs/grid_event_programme.parquet", pd.DataFrame(outage_rows,
        columns=["input_file", "member_id", "start_time_step", "end_time_step", "generation_regime", "cause_id", "active_slots"]))
    for path, schema in schemas.items():
        # Known faults is retained as an explicit benchmark assumption. A hidden
        # fault experiment must specify its feature allowlist, not silently mask
        # essential control inputs with a guessed MARL observation design.
        schema.setdefault("robustness", {})["agent_observation_contract"] = {"mode": "known_faults"}
        schema["experimental_design"].update(realism_revision=CONFIG["revision"],
            activity_calendar="catalogs/household_activity.parquet",
            commercial_activity="catalogs/commercial_activity.parquet" if (root/"catalogs/commercial_activity.parquet").exists() else None,
            occupancy_semantics="occupant_count is sampled integer presence at commercial sites, separately tracking staff, clients and care residents; operations.expected_people is its compatibility alias; catalogue occupants counts household residents. Public opening hours are posted service constraints; preparation/cleanup may occur outside them.",
            agent_fault_labels="known_faults explicitly enabled; not a fault-diagnosis benchmark",
            realism_parameter_status=CONFIG["parameter_status"],
            fixed_thermal_placeholders="Indoor temperature 21 C and humidity 50% are inactive nonthermal placeholders, not comfort ground truth",
            disturbance_regimes="generation_regime distinguishes synthetic local/common faults and retained controlled stress; continuous noise remains an explicit sensor stress control")
        dump(path, describe_current_inputs(root, path, schema))
    # Recompile preparation files after event revision; never expose 2024 events.
    for path, schema in schemas.items():
        if path.parent.name == "preparation_2023":
            source = json.loads((root/"evaluation/campaign_2024/source_schemas"/path.name).read_text())
            dump(path, preparation_view(root, source, path.stem))


def revise_weather(root):
    source = ROOT/"data/portugal_reference_2023/era5_open_meteo_2023.json.gz"
    if not source.exists():
        raise FileNotFoundError("Acquire and retain the ERA5 humidity source before generation: "+str(source))
    raw = json.loads(gzip.decompress(source.read_bytes()))["data"]["hourly"]
    clock = pd.date_range("2023-01-01", periods=N, freq="15min", tz="UTC")
    f = source_frame(root/"shared/weather.parquet")
    previous_forecasts = f.copy()
    humidity = np.interp((clock[:N23]+pd.Timedelta(minutes=7.5)).asi8.astype(float), pd.to_datetime(raw["time"], utc=True).asi8.astype(float), raw["relative_humidity_2m"])
    f.loc[:N23-1, "outdoor_relative_humidity"] = humidity.astype("float32")
    # Transparent causal benchmark: combine daily persistence with the latest
    # anomaly decaying with horizon. No realized future or added oracle noise.
    for col in [x for x in f if "predicted" not in x]:
        values = f[col].to_numpy()
        issue = np.arange(N)
        anomaly = values-values[np.maximum(0, issue-96)]
        for k, horizon in enumerate((4, 24, 96), 1):
            pred = values[np.clip(issue+horizon-96, 0, N-1)]+np.exp(-horizon/24)*anomaly
            if "humidity" in col:
                pred = np.clip(pred, 0, 100)
            if "solar" in col:
                # Preserve zero forecasts from the causal source persistence
                # and its astronomical night mask, without future weather.
                mask = previous_forecasts[f"{col}_predicted_{k}"].to_numpy() == 0
                pred = np.maximum(pred, 0.)
                pred[mask] = 0.
            f[f"{col}_predicted_{k}"] = pred.astype("float32")
    write(root/"shared/weather.parquet", f)


def finish(root):
    report = dict(revision=CONFIG["revision"], immutable_source_commit=SOURCE, configuration=CONFIG,
                  empirical_validation="Residential aggregate developed on seven raw SHEERM houses, six held out. Commercial annual budgets incorporate Portuguese office/small-commerce references with explicit boundary/transfer uncertainty; discrete occupancy remains a prior. Commercial hourly anchors use BDG2 development buildings, domestic programmes use REFIT houses 3/5; separate identities held out. FIKElectricity is a professional-kitchen structural diagnostic. See empirical_validation.json and commercial_appliance_validation.json for explicit cohort mismatches and unvalidated priors. EV Norwegian measured sessions do not validate arrival SOC. Outage exposed-cohort moments anchored to E-REDES BT 2024; stress excluded.",
                  forecast_method="Causal daily persistence + latest anomaly with 6h exponential decay; benchmark, not a fitted operational forecast")
    # Only explicitly named, current generated files authorize deviation from
    # historical source hashes in the audit; never replace source checksums.
    catalogues = {"charging_sessions.parquet", "electric_vehicles.parquet", "chargers.parquet",
        "mobility_episodes.parquet", "ev_annual_energy.parquet", "household_activity.parquet",
        "grid_event_programme.parquet", "deferrables.parquet", "member_year_totals.parquet", "commercial_activity.parquet"}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix in (".parquet", ".json") and (
        p.relative_to(root).parts[0] in ("timeseries", "end_uses", "deferrables", "mobility", "chargers", "evaluation", "operations")
        or (p.parent.name == "catalogs" and p.name in catalogues)
        or p == root/"schema.json" or p == root/"shared/weather.parquet")]
    report["output_checksums"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    source = ROOT/"data/portugal_reference_2023/era5_open_meteo_2023.json.gz"
    report["additional_source_checksums"] = {str(source.relative_to(ROOT)): hashlib.sha256(source.read_bytes()).hexdigest()}
    for source in sorted((ROOT/"data/portugal_validation_reference").glob("refit_house_[35]_cycles.parquet")):
        report["additional_source_checksums"][str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    old = source_frame(root/"catalogs/member_year_totals.parquet")
    current = pd.read_parquet(root/"catalogs/member_year_totals.parquet")
    report["annual_physical_nonshiftable_kwh"] = [dict(year=year,
        original=float(old.loc[old.year.eq(year), "non_shiftable_load_kwh"].sum()),
        revised=float(current.loc[current.year.eq(year), "non_shiftable_load_kwh"].sum()),
        scope="physical inventory, not topology-filtered executed annual controller demand") for year in (2023, 2024)]
    dump(root/"realism_revision.json", report)
    refresh_checksums(root)


def finish_manifest():
    path = ROOT/"results/portugal_rec_native/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["realism_revision"] = CONFIG["revision"]
    for group in ("variants", "references", "preparation_schemas"):
        for e in manifest[group]:
            e["sha256"] = hashlib.sha256((ROOT/e["schema"]).read_bytes()).hexdigest()
    dump(path, manifest)
    for root in sorted((ROOT/"data/datasets").glob("rec_pt_2023_*_v2")):
        readme = root/"README.md"
        text = readme.read_text().split("\n## Revisão de realismo corrente")[0]
        text = text.replace("mantêm-se definitivas em 2024; não se repetem os eventos de topologia.",
            "mantêm-se definitivas; as histórias de topologia incluem a campanha reservada de 2024.")
        readme.write_text(text.rstrip()+"\n\n## Revisão de realismo corrente\n\n"
            f"`{CONFIG['revision']}` / `portugal-ev-mobility-1`: atividade partilhada, ciclos variáveis, mobilidade contínua de referência e programas de falha distintos por ano. "
            "Ver `realism_revision.json` e `ev_mobility_revision.json` para configuração, distribuições e hashes; `catalogs/household_activity.parquet` para o calendário. "
            "As narrativas e valores de dimensionamento antigos são proveniência, não a descrição operacional desta revisão. "
            "Os parâmetros de variabilidade do consumo foram ajustados em sete casas SHEERM; seis outras casas são reservadas para avaliação independente. Os restantes pressupostos e desvios estão identificados em `results/portugal_rec_native/empirical_validation.json`. Método e limites em `docs/portugal_rec_native/Fontes_e_metodo.md`.\n")
        for name in ("CURRENT_EVALUATION.json", "dataset_manifest.json", "extension_2024.json"):
            p = root/name
            value = json.loads(p.read_text())
            value.update(realism_revision=CONFIG["revision"], realism_evidence="realism_revision.json")
            for group in ("variants", "references", "preparation_schemas"):
                if group in value:
                    for e in value[group]:
                        if "schema" in e and (ROOT/e["schema"]).exists():
                            e["sha256"] = hashlib.sha256((ROOT/e["schema"]).read_bytes()).hexdigest()
            dump(p, value)
        finish(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--family", default="*")
    parser.add_argument("--config", type=str, help="Alternative explicit JSON prior configuration")
    parser.add_argument("--seed", type=int, help="Override master seed for sensitivity ensembles")
    parser.add_argument("--skip-mobility", action="store_true", help="Development stage only; final release requires mobility regeneration")
    args = parser.parse_args()
    if args.config:
        from pathlib import Path
        supplied = json.loads(Path(args.config).read_text())
        assert set(CONFIG) <= set(supplied), "Alternative config must define every documented section"
        CONFIG.clear()
        CONFIG.update(supplied)
    if args.seed is not None:
        CONFIG["seed"] = args.seed
    if not args.apply:
        print(json.dumps(CONFIG, indent=2))
        return
    for root in sorted((ROOT/"data/datasets").glob(f"rec_pt_2023_{args.family}_v2")):
        print(root.name, "activity, appliances and physical end uses", flush=True)
        revise_physical(root, pd.read_parquet(root/"shared/calendar.parquet"))
        revise_weather(root)
        revise_events(root)
        propagate(root)
        if not args.skip_mobility:
            from revise_portugal_ev_mobility import revise_family, finish_revision
            report = revise_family(root, apply=True)
            print(json.dumps(report["distributions"]), flush=True)
            finish_revision()
        finish(root)
    finish_manifest()


if __name__ == "__main__":
    main()

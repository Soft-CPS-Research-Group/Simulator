"""Reproducible, offline EV revision of the two-year Portugal datasets.

Read the immutable v3.3 inputs directly from Git, without extracting a second
dataset tree. Only mobility, charging inputs and their current metadata change.
The continuous reference ledger is a data-generation witness, not a claim that
the simulator applies trip energy between its prescribed session arrivals.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from apply_portugal_omie_pricing import refresh_checksums

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "c54b13b62823fb1256397ea98cb37a830a24124d"
REVISION = "portugal-ev-mobility-1"
N = 70176
N23 = 35040
DT = .25


def source_frame(path, repo=ROOT):
    raw = subprocess.check_output(["git", "show", f"{SOURCE}:{path.relative_to(repo)}"], cwd=repo)
    return pd.read_parquet(io.BytesIO(raw))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def write(path, frame):
    frame.to_parquet(path, index=False, compression="zstd")


def rng_for(root, eid):
    # Nested core communities retain the same random stream for a shared car.
    from portugal_activity import rng_for as process_rng
    return process_rng(root, eid, 0, "charging_decisions")


def profiles(vehicles):
    residents = vehicles.loc[vehicles.vehicle_pool.eq("resident")].sort_values("ev_id")
    counts = {2: (1, 1, 0), 6: (1, 3, 2), 9: (2, 4, 2), 37: (7, 17, 9)}[len(residents)]
    available = ["frequent"]*counts[0] + ["periodic"]*counts[1] + ["low_soc"]*counts[2]
    np.random.default_rng(851).shuffle(available)
    return {v.ev_id: ("visitor" if v.vehicle_pool != "resident" else
            "work" if v.work_member_id else available.pop()) for v in vehicles.itertuples()}


def local_step(clock, day, hour):
    naive = day.tz_localize(None).normalize() + pd.Timedelta(hours=hour)
    local = naive.tz_localize("Europe/Lisbon", ambiguous=True, nonexistent="shift_forward")
    return int(np.clip(clock.searchsorted(local.tz_convert("UTC")), 0, len(clock)))


def resident_mobility(v, member, old, old_sessions, calendar, rng):
    """Keep weekday routines; resample leisure days and linked multi-day trips.

    Each annual driving-energy budget is preserved, including the two legs of
    an absence. Holidays come from the existing year-specific local calendar.
    """
    clock = pd.DatetimeIndex(calendar.timestamp_utc)
    home = old_sessions.loc[old_sessions.purpose.eq("home")].sort_values("arrival_time_step")
    away = []
    for left, right in zip(home.itertuples(), list(home.itertuples())[1:]):
        a, b = int(left.observed_end_time_step), int(right.arrival_time_step)
        if b > a:
            away.append(dict(start=a, end=b, kind="routine", energy=float(old.trip_energy_kwh.iloc[a:b+1].sum())))
    # Preserve a possible initial/final away interval as well.
    if len(home) and home.iloc[0].arrival_time_step > 0:
        b = int(home.iloc[0].arrival_time_step)
        away.insert(0, dict(start=0, end=b, kind="routine", energy=float(old.trip_energy_kwh.iloc[:b+1].sum())))
    if len(home) and home.iloc[-1].observed_end_time_step < N:
        a = int(home.iloc[-1].observed_end_time_step)
        away.append(dict(start=a, end=N, kind="routine", energy=float(old.trip_energy_kwh.iloc[a:].sum())))
    days = pd.date_range("2023-01-01", "2024-12-31", tz="Europe/Lisbon")
    episodes = []
    for day in days:
        lo, hi = local_step(clock, day, 0), local_step(clock, day, 24)
        rest = bool(calendar.is_rest_day.iloc[lo])
        vacation_start = pd.Timestamp(year=2023, month=1, day=1) + pd.Timedelta(days=int(member.vacation_start_day)-1)
        vacation = 0 <= (day.tz_localize(None) - vacation_start.replace(year=day.year)).days < member.vacation_days
        # Night shifts retain their actual multi-day work/rest pattern.
        if member.routine_archetype == "night_shift" and not vacation:
            continue
        if not (rest or vacation):
            continue
        overlapping = [x for x in away if x["start"] < hi and x["end"] > lo]
        if any(x["start"] < lo or x["end"] >= hi for x in overlapping):
            continue
        away = [x for x in away if x not in overlapping]
        kind = rng.choice(["none", "local", "regional", "long"], p=[.30, .50, .18, .02])
        if kind == "none":
            continue
        km = float(rng.uniform(*{"local": (5, 50), "regional": (50, 150), "long": (150, 350)}[kind]))
        depart = float(rng.uniform(8, 11) if kind != "local" else rng.uniform(9, 14))
        duration = float(rng.uniform(7, 11) if kind != "local" else rng.uniform(1.5, 5))
        away.append(dict(start=local_step(clock, day, depart), end=local_step(clock, day, min(23, depart+duration)),
                         kind=kind, energy=km*v.consumption_kwh_per_km))
    # A few linked absences per year, including holiday bridges and annual leave.
    for year in (2023, 2024):
        year_days = [d for d in days if d.year == year]
        fridays = [d for d in year_days if d.dayofweek == 4 and d.month in (3, 4, 5, 6, 7, 9, 10)]
        chosen = list(rng.choice(len(fridays), size=2, replace=False))
        spans = [(fridays[int(i)], 2, "weekend_away") for i in chosen]
        holidays = [d for d in year_days if bool(calendar.is_national_holiday.iloc[local_step(clock, d, 0)]) and d.dayofweek in (1, 3)]
        if holidays:
            holiday = holidays[int(rng.integers(len(holidays)))]
            begin = holiday-pd.DateOffset(days=3) if holiday.dayofweek == 1 else holiday
            spans.append((begin, 3, "holiday_bridge"))
        vacation_day = pd.Timestamp(year=2023, month=1, day=1)+pd.Timedelta(days=int(member.vacation_start_day)-1)
        spans.append((vacation_day.replace(year=year).tz_localize("Europe/Lisbon"), int(rng.integers(3, 6)), "vacation_away"))
        for day, duration, kind in spans:
            a, b = local_step(clock, day, 18 if kind == "weekend_away" else 8), local_step(clock, day+pd.DateOffset(days=duration), 20)
            if not 0 < a < b < N:
                continue
            # Expand to physical departure/return boundaries of overlapping trips.
            overlaps = [x for x in away if x["start"] < b and x["end"] > a]
            a = min([a]+[x["start"] for x in overlaps]); b = max([b]+[x["end"] for x in overlaps])
            away = [x for x in away if x not in overlaps]
            energy = float(rng.uniform(180, 350))*v.consumption_kwh_per_km
            away.append(dict(start=a, end=b, kind=kind, energy=energy))
    away.sort(key=lambda x: x["start"])
    assert all(x["end"] <= y["start"] for x, y in zip(away, away[1:]))
    travel = np.zeros(N)
    external = np.zeros(N)
    for x in away:
        a, b = x["start"], x["end"]
        if x["kind"] == "routine":
            travel[a:min(b+1, N)] += old.trip_energy_kwh.iloc[a:min(b+1, N)].to_numpy()
        else:
            travel[a] += x["energy"]*.5
            if b < N:
                travel[b] += x["energy"]*.5
        episodes.append(dict(ev_id=v.ev_id, departure_time_step=a, return_time_step=b, trip_kind=x["kind"]))
    for lo, hi in ((0, N23), (N23, N)):
        travel[lo:hi] *= old.trip_energy_kwh.iloc[lo:hi].sum()/travel[lo:hi].sum()
    # Explicit finite en-route supply on long absences. No free SOC reset.
    for x, record in zip(away, episodes):
        a, b = x["start"], x["end"]
        energy = float(travel[a:min(b+1, N)].sum())
        record["trip_energy_kwh"] = energy
        record["energy_equivalent_distance_km"] = energy/v.consumption_kwh_per_km
        if energy > v.battery_capacity_kwh*.65:
            offered = max(energy-v.battery_capacity_kwh*.65, 0.)
            t = min(a+max(2, (b-a)//2), N-5)
            external[t:t+4] += offered/4
    candidates = []
    cursor = 0
    for x in away:
        a, b = x["start"], x["end"]
        if a-cursor >= 2:
            candidates.append(dict(ev_id=v.ev_id, charger_id=v.home_charger_id, member_id=v.home_member_id,
                                   arrival_time_step=cursor, observed_end_time_step=a, purpose="home"))
        cursor = b
    if N-cursor >= 2:
        candidates.append(dict(ev_id=v.ev_id, charger_id=v.home_charger_id, member_id=v.home_member_id,
                               arrival_time_step=cursor, observed_end_time_step=N, purpose="home"))
    # Work parking is retained only during its original work journey.
    for s in old_sessions.loc[~old_sessions.purpose.eq("home")].to_dict("records"):
        if any(x["kind"] == "routine" and x["start"] <= s["arrival_time_step"] and x["end"] >= s["observed_end_time_step"] for x in away):
            candidates.append(s)
    return travel, external, candidates, episodes


def phase_vector(phase):
    return np.ones(3)/3 if phase == "all_phases" else np.array([float(phase == p) for p in ("L1", "L2", "L3")])


def cycle_power_envelope(profiles, schedule, steps=N):
    """Upper bound at each step for any legal appliance cycle start."""
    result = np.zeros(steps)
    for cycle in schedule.itertuples():
        profile = np.asarray(profiles[cycle.profile_id])/DT
        lo, hi = int(cycle.earliest_start_time_step), int(cycle.latest_start_time_step)
        band = np.zeros(hi-lo+len(profile))
        for offset, power in enumerate(profile):
            band[offset:offset+hi-lo+1] = np.maximum(band[offset:offset+hi-lo+1], power)
        result[lo:min(lo+len(band), steps)] += band[:steps-lo]
    return result


def power_budgets(root, schema):
    """Conservative shared-site witness, reserving controllable appliance peaks.

    Share remaining site/phase power among installed chargers. Stationary
    storage is not needed to make these reference requests feasible.
    """
    result = {}
    for member, b in schema["buildings"].items():
        chargers = b.get("chargers", {})
        if not chargers:
            continue
        f = pd.read_parquet(root/b["energy_simulation"])
        net = (f.non_shiftable_load.to_numpy()-f.solar_generation.to_numpy())/DT
        service = b["electrical_service"]
        split = phase_vector(service["default_split"])
        appliance_peak = np.zeros(N)
        for app in b.get("deferrable_appliances", {}).values():
            profiles = pd.read_parquet(root/app["cycle_profiles_file"])
            profiles = {r.profile_id: json.loads(r.load_profile) for r in profiles.itertuples()}
            schedule = pd.read_parquet(root/app["flexibility_schedule_file"])
            appliance_peak += cycle_power_envelope(profiles, schedule)
        total = np.maximum(0., service["limits"]["total"]["import_kw"]-net-appliance_peak)/len(chargers)
        splits = {cid: phase_vector(ch["attributes"]["phase_connection"]) for cid, ch in chargers.items()}
        for cid, ch in chargers.items():
            attrs = ch["attributes"]
            power = np.minimum(total, attrs["max_charging_power"])
            for j, phase in enumerate(("L1", "L2", "L3")):
                if splits[cid][j]:
                    count = sum(x[j] > 0 for x in splits.values())
                    # Reserve the entire appliance envelope on each used phase;
                    # this is safe even when appliance phase declarations differ.
                    # Keep the phase share weak so the float32 input keeps the
                    # rounding used to generate the published ledgers under NumPy 1.
                    phase_net = net*float(split[j])
                    headroom = np.maximum(0., service["limits"]["per_phase"][phase]["import_kw"]-phase_net-appliance_peak)
                    power = np.minimum(power, headroom/count/splits[cid][j])
            power[power < attrs["min_charging_power"]] = 0.
            result[cid] = power*attrs["efficiency"]
    return result


def charge_trace(energy, target, powers, battery):
    cap = battery["capacity"]
    curve = np.asarray(battery["capacity_power_curve"])
    eta = np.sqrt(battery["efficiency"])
    gains = np.zeros(len(powers))
    for t, p in enumerate(powers):
        need = max(0., target*cap-energy)
        if need < 1e-10:
            break
        power = min(p, battery["nominal_power"]*np.interp(energy/cap, curve[:, 0], curve[:, 1]))
        gains[t] = min(need, power*DT*eta)
        energy += gains[t]
    return energy, gains


def latest_charge_start(energy, target, powers, battery):
    """Latest feasible start offset, retaining time-varying headroom and taper."""
    lo, hi = 0, len(powers)
    while lo < hi:
        mid = (lo+hi+1)//2
        end, _ = charge_trace(energy, target, powers[mid:], battery)
        if end >= target*battery["capacity"]-1e-7:
            lo = mid
        else:
            hi = mid-1
    return lo


def coast(energy, travel, external, cap):
    """Pre-charge battery energy; upper clipping records offered vs used supply."""
    delta = external-travel
    energy_path = energy+np.cumsum(delta)
    spill = np.maximum.accumulate(np.maximum(energy_path-cap, 0.))
    path = energy_path-spill
    # Travel occurs before an external charge at the same step.
    before_external = np.r_[energy, path[:-1]]-travel
    return path, before_external


def required_energy(travel, external, candidates, powers, battery):
    """Backward reachability envelope, including short/slow future stops."""
    available = np.zeros(len(travel))
    for s in candidates:
        if s["eligible"]:
            a, b = int(s["arrival_time_step"]), int(s["observed_end_time_step"])
            available[a:b] = powers[s["charger_id"]][a:b]
    cap, nominal = battery["capacity"], battery["nominal_power"]
    curve = np.asarray(battery["capacity_power_curve"])
    k = DT*np.sqrt(battery["efficiency"])
    required = np.zeros(len(travel)+1)
    for t in range(len(travel)-1, -1, -1):
        need = required[t+1]
        p = available[t]
        if p > 0 and need > 0:
            # Invert the monotone piecewise-linear one-step charge function.
            knots = list(curve[:, 0]*cap)
            for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
                if y0 != y1 and min(y0, y1) < p/nominal < max(y0, y1):
                    knots.append(cap*(x0+(p/nominal-y0)*(x1-x0)/(y1-y0)))
            x = np.array(sorted(knots))
            y = x+np.minimum(p, nominal*np.interp(x/cap, curve[:, 0], curve[:, 1]))*k
            need = float(np.interp(need, y, x, left=0.))
        required[t] = travel[t]+max(0., need-external[t])
        if required[t] > cap+1e-6:
            raise ValueError(f"Journey cannot be supplied even with all parking: step {t}")
    return required


def plan_charging(v, kind, candidates, travel, external, battery, powers, calendar, rng, forbidden, vary_connections=True):
    from portugal_activity import CONFIG
    candidates = sorted(candidates, key=lambda s: s["arrival_time_step"])
    cap = battery["capacity"]
    threshold = {"frequent": .80, "periodic": float(rng.uniform(.25, .40)),
                 "low_soc": float(rng.uniform(.12, .25)), "work": .5, "visitor": .55}[kind]
    for s in candidates:
        a, b = int(s["arrival_time_step"]), int(s["observed_end_time_step"])
        # Being parked does not imply immediate connection. Preserve physical
        # travel/opening envelopes; the offline witness still proves feasibility.
        if vary_connections and a > 0 and rng.random() < CONFIG["charging"]["delayed_connection_probability"]:
            delay = min(int(rng.geometric(.4)), max(0,(b-a-4)//5), 12)
            a += delay
            s["arrival_time_step"] = a
        s["eligible"] = not any(a < t < b for t in forbidden)
        s["base_target"] = float(rng.uniform(*{"frequent": (.75, .90), "periodic": (.80, .90),
                                               "low_soc": (.85, .97), "work": (.88, .95), "visitor": (.75, .95)}[kind]))
        s["full"] = rng.random() < .10
        s["willing"] = rng.random() < .8 if kind == "frequent" else True
        s["trigger_threshold"] = float(np.clip(rng.normal(threshold, CONFIG["charging"]["threshold_std"]),.05,.9))
    required = required_energy(travel, external, candidates, powers, battery)
    records = []
    gains = np.zeros(N)
    cursor = 0
    energy = cap*v.initial_soc
    first = 0
    while first < len(candidates):
        path, before = coast(energy, travel[cursor:], external[cursor:], cap)
        bad = np.flatnonzero(before < -1e-8)
        horizon = cursor+int(bad[0]) if len(bad) else N
        feasible = []
        emergency = []
        chosen = None
        # Occasional delayed full recharge; this preference cannot override
        # travel feasibility or a short stay. No artificial SOC is assigned.
        delayed_full = kind != "visitor" and rng.random() < .015
        for k in range(first, len(candidates)):
            s = candidates[k]
            a, b = int(s["arrival_time_step"]), int(s["observed_end_time_step"])
            if a >= horizon:
                break
            if not s["eligible"] or a < cursor:
                continue
            soc = path[a-cursor]/cap
            if soc >= 1.-1e-10:
                continue
            target = 1. if s["full"] or delayed_full or soc <= .05 else s["base_target"]
            reason = "near_empty_full_charge" if soc <= .05 else "occasional_full_charge" if target == 1 else "habit_and_next_journeys"
            hard_minimum = min(1., required[b]/cap)
            minimum = min(1., hard_minimum+.03)
            if minimum > target:
                target = 1. if minimum >= .95 else minimum
                reason = "advance_reserve_for_long_journey"
            if target <= soc+1e-8:
                continue
            final, gain = charge_trace(path[a-cursor], target, powers[s["charger_id"]][a:b], battery)
            if final < target*cap-1e-7:
                # This is a separate opportunity/advance-reserve request, not
                # an infeasible full request whose target is silently clipped.
                reachable = final/cap
                if reachable < hard_minimum-1e-9:
                    continue
                desired = min(.9, soc+float(rng.uniform(.10,.25))) if kind == "visitor" else .9
                # A discretionary reserve must not rule out the small charge
                # that actually makes the next journeys feasible. Reachability
                # uses the complete taper/loss/headroom envelope.
                target = max(hard_minimum, min(desired, reachable-min(.001,max(0.,reachable-hard_minimum)/2)))
                if target <= soc+1e-8:
                    continue
                final, gain = charge_trace(path[a-cursor], target, powers[s["charger_id"]][a:b], battery)
                reason = "short_visit_energy_request" if kind == "visitor" else "advance_reserve_short_stop"
                if final < target*cap-1e-7:
                    continue
                if kind != "visitor":
                    emergency.append((k, s, soc, target, reason, final, gain))
                    continue
            item = (k, s, soc, target, reason, final, gain)
            feasible.append(item)
            preferred_site = kind != "work" or s["purpose"] != "home"
            if soc <= (.05 if delayed_full else s["trigger_threshold"]) and s["willing"] and preferred_site:
                chosen = item
                break
        if chosen is None:
            if horizon == N:
                break
            if not feasible:
                feasible = emergency
            if not feasible:
                raise ValueError(f"No feasible charge before travel: {v.ev_id} at {horizon}, {energy/cap:.3f}")
            chosen = feasible[-1]
            chosen = (*chosen[:4], "advance_charge_for_next_journeys", *chosen[5:])
        k, s, soc, target, reason, final, gain = chosen
        a, b = int(s["arrival_time_step"]), int(s["observed_end_time_step"])
        assert not travel[a+1:b].any() and not external[a:b].any(), (v.ev_id, a, b)
        gains[a:b] = gain
        record = {key: s[key] for key in ("ev_id", "charger_id", "member_id", "purpose")}
        record.update(arrival_time_step=a, departure_time_step=np.nan if b == N and s["purpose"] == "home" else b,
                      observed_end_time_step=b, right_censored=b == N and s["purpose"] == "home",
                      session_id=f"{v.ev_id}-M1-{len(records):04d}", target_soc=target, target_reason=reason,
                      arrival_soc_reference=soc, requested_energy_reference_kwh=(target-soc)*cap,
                      day_kind=str(calendar.day_kind.iloc[a]), year=2023 if a < N23 else 2024,
                      charging_profile=kind, reference_charge_steps=int(np.count_nonzero(gain)),
                      reference_slack_hours=np.nan if b == N and s["purpose"] == "home" else
                      latest_charge_start(soc*cap, target, powers[s["charger_id"]][a:b], battery)*DT)
        records.append(record)
        energy = final
        cursor = b
        first = k+1
    # Independent continuous replay; never conceal a driving deficit by clipping.
    soc = np.zeros(N); end_soc = np.zeros(N); used = np.zeros(N)
    energy = cap*v.initial_soc
    for t in range(N):
        energy -= travel[t]
        if energy < -1e-7:
            raise ValueError(f"Unserved journey {v.ev_id}: {t}, {energy}")
        used[t] = min(external[t], max(0., cap-energy))
        energy += used[t]
        soc[t] = energy/cap
        energy += gains[t]
        end_soc[t] = energy/cap
    for k, s in enumerate(records):
        a, b = s["arrival_time_step"], s["observed_end_time_step"]
        assert abs(soc[a]-s["arrival_soc_reference"]) < 1e-8
        assert end_soc[b-1] >= s["target_soc"]-1e-8
        nxt = records[k+1]["arrival_time_step"]+1 if k+1 < len(records) else N
        s["next_trip_energy_kwh"] = float(travel[b:nxt].sum())
    return records, pd.DataFrame(dict(trip_energy_kwh=travel, external_charge_kwh=external,
        external_charge_used_kwh=used, reference_soc=soc, reference_charge_kwh=gains, reference_end_soc=end_soc))


def schedules(sessions, ledgers, charger_ids):
    result = {}
    for cid in charger_ids:
        frame = pd.DataFrame(dict(electric_vehicle_charger_state=np.full(N, 3, dtype="int8"),
            electric_vehicle_id=np.full(N, "", dtype=object), electric_vehicle_session_id=np.full(N, "", dtype=object)))
        for name in ("departure_time", "required_soc_departure", "estimated_arrival_time", "estimated_soc_arrival", "current_soc"):
            frame["electric_vehicle_"+name] = np.nan
        previous = 0
        for s in sessions.loc[sessions.charger_id.eq(cid)].sort_values("arrival_time_step").itertuples():
            a, b = int(s.arrival_time_step), int(s.observed_end_time_step)
            assert a >= previous
            incoming = max(previous, a-4)
            frame.loc[a:b-1, "electric_vehicle_charger_state"] = 1
            if incoming < a:
                frame.loc[incoming:a-1, "electric_vehicle_charger_state"] = 2
                frame.loc[incoming:a-1, "electric_vehicle_estimated_arrival_time"] = np.arange(a-incoming, 0, -1)
            frame.loc[incoming:b-1, "electric_vehicle_id"] = s.ev_id
            frame.loc[incoming:b-1, "electric_vehicle_session_id"] = s.session_id
            if not s.right_censored:
                frame.loc[a:b-1, "electric_vehicle_departure_time"] = np.arange(b-a, 0, -1)
            frame.loc[a:b-1, "electric_vehicle_required_soc_departure"] = s.target_soc
            frame.loc[incoming:b-1, "electric_vehicle_estimated_soc_arrival"] = s.arrival_soc_reference
            frame.loc[a:b-1, "electric_vehicle_current_soc"] = ledgers[s.ev_id].reference_soc.iloc[a:b].to_numpy()
            previous = b
        result[cid] = frame
    return result


def summary(sessions, vehicles):
    rows = []
    for year, days in ((2023, 365), (2024, 366)):
        for pool in ("resident", "external_visitors"):
            ids = set(vehicles.loc[vehicles.vehicle_pool.eq(pool), "ev_id"])
            f = sessions.loc[sessions.ev_id.isin(ids) & sessions.year.eq(year)]
            if f.empty:
                continue
            gap = (f.target_soc-f.arrival_soc_reference)*100
            rows.append(dict(year=year, pool=pool, vehicles=len(ids), sessions=len(f),
                sessions_per_vehicle_week=len(f)/len(ids)/(days/7), arrival_mean_pct=float(f.arrival_soc_reference.mean()*100),
                target_mean_pct=float(f.target_soc.mean()*100), request_mean_pp=float(gap.mean()),
                full_target_pct=float(f.target_soc.ge(.999999).mean()*100),
                near_empty_to_full_pct=float((f.arrival_soc_reference.le(.05)&f.target_soc.ge(.999999)).mean()*100),
                gap_bins_pct=(pd.cut(gap, [0, 10, 30, 50, 70, 101], right=False).value_counts(normalize=True, sort=False)*100).tolist(),
                arrival_bins_pct=(pd.cut(f.arrival_soc_reference*100, [-1e-8, 5, 10, 20, 40, 60, 101], right=False).value_counts(normalize=True, sort=False)*100).tolist(),
                median_slack_hours=float(f.reference_slack_hours.median())))
    return rows


def revise_family(root, apply=False):
    calendar = pd.read_parquet(root/"shared/calendar.parquet")
    vehicles = source_frame(root/"catalogs/electric_vehicles.parquet")
    old_sessions = source_frame(root/"catalogs/charging_sessions.parquet")
    members = pd.read_parquet(root/"catalogs/members.parquet").set_index("member_id")
    schema = json.loads((root/"schema.json").read_text())
    power = power_budgets(root, schema)
    assignment = profiles(vehicles)
    ledgers = {}; all_sessions = []; all_trips = []; annual = []; connection_fallbacks = []
    # A common service-free boundary keeps all matched topology variants valid.
    boundaries = {}
    for path in (root/"evaluation").rglob("*.json"):
        sc = json.loads(path.read_text())
        if not isinstance(sc, dict):
            continue
        for e in sc.get("topology_events", []):
            if e["operation"] == "remove_member":
                boundaries.setdefault(e["target_member_id"], set()).add(int(e["time_step"]))
    for v in vehicles.itertuples():
        old = source_frame(root/f"mobility/{v.ev_id}.parquet")
        original_sessions = old_sessions.loc[old_sessions.ev_id.eq(v.ev_id)]
        rng = rng_for(root, v.ev_id)
        if v.vehicle_pool == "resident":
            activity_path = root/"catalogs/household_activity.parquet"
            if activity_path.exists():
                from portugal_activity import mobility
                daily = pd.read_parquet(activity_path)
                daily = daily.loc[daily.member_id.eq(v.home_member_id)]
                travel, external, candidates, trips = mobility(v, members.loc[v.home_member_id], old, original_sessions, calendar, daily, root)
            else:
                travel, external, candidates, trips = resident_mobility(v, members.loc[v.home_member_id], old, original_sessions, calendar, rng)
            all_trips.extend(trips)
        else:
            travel, external = old.trip_energy_kwh.to_numpy().copy(), old.external_charge_kwh.to_numpy().copy()
            candidates = original_sessions.to_dict("records")
        full_candidates = deepcopy(candidates)
        if v.vehicle_pool != "resident":
            # Appointment-like source windows remain the allocation envelope,
            # but actual visitor departure varies within the reserved slot.
            for s in candidates:
                a,b=int(s["arrival_time_step"]),int(s["observed_end_time_step"])
                s["observed_end_time_step"] = b-int(rng.integers(0,min(5,max(1,(b-a)//5))))
        battery = schema["electric_vehicles_def"][v.ev_id]["battery"]["attributes"]
        window_sampling = "variable_connection"
        try:
            records, ledger = plan_charging(v, assignment[v.ev_id], candidates, travel, external,
                battery, power, calendar, rng, boundaries.get(v.home_member_id, set()))
        except ValueError as error:
            if not str(error).startswith("No feasible charge before travel"):
                raise
            # Condition optional connection jitter on service feasibility. A
            # rejected draw restores the physical parking/opening envelope,
            # never energy, SOC, driving budgets or charger ratings.
            records, ledger = plan_charging(v, assignment[v.ev_id], full_candidates, travel, external,
                battery, power, calendar, rng_for(root,v.ev_id), boundaries.get(v.home_member_id,set()),
                vary_connections=False)
            connection_fallbacks.append(v.ev_id)
            window_sampling = "full_envelope_after_infeasible_jitter"
        for record in records:
            record["connection_window_sampling"] = window_sampling
        all_sessions.extend(records); ledgers[v.ev_id] = ledger
        for year, lo, hi in ((2023, 0, N23), (2024, N23, N)):
            annual.append(dict(ev_id=v.ev_id, year=year, original_trip_energy_kwh=float(old.trip_energy_kwh.iloc[lo:hi].sum()),
                trip_energy_kwh=float(travel[lo:hi].sum()), external_charge_used_kwh=float(ledger.external_charge_used_kwh.iloc[lo:hi].sum()),
                reference_charge_kwh=float(ledger.reference_charge_kwh.iloc[lo:hi].sum())))
    sessions = pd.DataFrame(all_sessions).sort_values(["charger_id", "arrival_time_step"]).reset_index(drop=True)
    report = dict(revision=REVISION, source_commit=SOURCE, family=root.name,
                  profiles=pd.Series(assignment).value_counts().to_dict(), distributions=summary(sessions, vehicles),
                  mobility_episodes=len(all_trips), annual_energy=annual,
                  connection_jitter_rejected_ev_ids=connection_fallbacks,
                  reference_scope="Continuous offline ledger, conservative normal-site headroom, no stationary battery support; runtime arrivals remain prescribed. Disturbance scenarios may intentionally miss requests.")
    if not apply:
        return report
    charging = schedules(sessions, ledgers, power)
    partial_ids = set(sessions.loc[sessions.right_censored, "session_id"])
    partial_files = 0
    # Keep each existing variant's immutable unavailable intervals. The remaining
    # paths all receive the same revised charging realization.
    for path in sorted(root.rglob("CHG-*.parquet")):
        if path.stem not in charging:
            continue
        original = source_frame(path)
        if len(original) != N:
            continue  # Historical one-year construction evidence, not live inputs.
        frame = charging[path.stem].copy()
        unavailable = original.electric_vehicle_charger_state.eq(0)
        for col in frame:
            frame.loc[unavailable, col] = 0 if col == "electric_vehicle_charger_state" else "" if col in ("electric_vehicle_id", "electric_vehicle_session_id") else np.nan
        partial_files += int((frame.electric_vehicle_session_id.isin(partial_ids) & frame.electric_vehicle_charger_state.eq(1)).any())
        write(path, frame[original.columns])
    write(root/"catalogs/charging_sessions.parquet", sessions)
    write(root/"catalogs/mobility_episodes.parquet", pd.DataFrame(all_trips))
    write(root/"catalogs/ev_annual_energy.parquet", pd.DataFrame(annual))
    vehicles["charging_profile"] = vehicles.ev_id.map(assignment)
    for i, v in vehicles.iterrows():
        ledger = ledgers[v.ev_id]
        vehicles.loc[i, "annual_trip_energy_kwh"] = ledger.trip_energy_kwh.iloc[:N23].sum()
        vehicles.loc[i, "annual_external_charge_offered_kwh"] = ledger.external_charge_kwh.iloc[:N23].sum()
        write(root/f"mobility/{v.ev_id}.parquet", ledger)
    write(root/"catalogs/electric_vehicles.parquet", vehicles)
    chargers = source_frame(root/"catalogs/chargers.parquet")
    for i, ch in chargers.iterrows():
        f = sessions.loc[sessions.charger_id.eq(ch.charger_id) & sessions.year.eq(2023)]
        chargers.loc[i, "annual_session_count"] = len(f)
        chargers.loc[i, "vehicle_pool_size"] = sessions.loc[sessions.charger_id.eq(ch.charger_id), "ev_id"].nunique()
    write(root/"catalogs/chargers.parquet", chargers)
    mutable = [root/"catalogs"/name for name in ("charging_sessions.parquet", "electric_vehicles.parquet",
        "chargers.parquet", "mobility_episodes.parquet", "ev_annual_energy.parquet")]
    mutable += list((root/"mobility").glob("*.parquet"))
    mutable += [p for p in root.rglob("CHG-*.parquet") if len(pd.read_parquet(p)) == N]
    report["output_checksums"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(mutable)}
    # Record current evidence without replacing historical evaluation claims.
    dump(root/"ev_mobility_revision.json", report)
    for name in ("dataset_manifest.json", "CURRENT_EVALUATION.json", "extension_2024.json"):
        path = root/name
        metadata = json.loads(path.read_text())
        metadata["mobility_revision"] = REVISION
        metadata["mobility_evidence"] = "ev_mobility_revision.json"
        metadata["mobility_catalogue_scope"] = "charging_profile, charging_sessions, mobility_episodes and ev_annual_energy describe current mobility. soc_preference, reserve_soc, member prose and planning energy fields retain historical sizing provenance; reserve_soc is not a minimum allowed SOC in this revision. Prior controller evaluations do not evaluate this mobility revision."
        if name == "extension_2024.json":
            metadata["year_end_session_bridges"] = [dict(charger_id=s.charger_id, session_id=s.session_id,
                continued_slots_2024=int(s.observed_end_time_step-N23), scope="Continuous revised mobility ledger across New Year")
                for s in sessions.loc[sessions.arrival_time_step.lt(N23) & sessions.observed_end_time_step.gt(N23)].itertuples()]
        dump(path, metadata)
    path = root/"boundary_integrity.json"
    boundary = json.loads(path.read_text())
    boundary["right_censored_sessions"] = sessions.loc[sessions.right_censored, "session_id"].tolist()
    boundary["schedule_files_with_partial_sessions"] = partial_files
    boundary["mobility_revision"] = REVISION
    boundary["session_contract"] = "observed_end_time_step is exclusive; departure_time_step is null for terminal home charging; current sessions and SOC come from the mobility revision"
    dump(path, boundary)
    refresh_checksums(root)
    return report


def finish_revision(repo=ROOT):
    path = repo/"results/portugal_rec_native/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["mobility_revision"] = REVISION
    manifest["mobility_evidence"] = [str(p.relative_to(repo)) for p in sorted((repo/"data/datasets").glob("rec_pt_2023_*_v2/ev_mobility_revision.json"))]
    dump(path, manifest)
    path = repo/"results/portugal_rec_native/boundary_integrity.json"
    report = json.loads(path.read_text())
    report["mobility_revision"] = REVISION
    for entry in report["families"]:
        root = repo/"data/datasets"/entry["family"]
        sessions = pd.read_parquet(root/"catalogs/charging_sessions.parquet")
        entry.update(sessions=len(sessions), complete_sessions=int((~sessions.right_censored).sum()),
                     right_censored_sessions=int(sessions.right_censored.sum()),
                     schedule_files_with_partial_sessions=json.loads((root/"boundary_integrity.json").read_text())["schedule_files_with_partial_sessions"])
    report["preserved"] = "Boundary censoring and inactive-outage corrections remain valid. Current EV arrivals, targets and sessions use portugal-ev-mobility-1; realism_revision.json additionally describes the shared activity calendar and physical/event revisions where present."
    dump(path, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--family", default="*")
    args = parser.parse_args()
    for root in sorted((ROOT/"data/datasets").glob(f"rec_pt_2023_{args.family}_v2")):
        result = revise_family(root, args.apply)
        print(json.dumps({k: v for k, v in result.items() if k not in ("annual_energy", "output_checksums")}, indent=2), flush=True)
    if args.apply:
        finish_revision()

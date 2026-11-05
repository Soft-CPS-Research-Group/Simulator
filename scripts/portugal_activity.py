"""Shared civil-day household state for the Portugal synthetic data generator.

Structural inspiration: CREST and ResStock. Numeric priors are explicit scenario
assumptions in portugal_realism_config.json, NOT fitted population parameters.
Random streams are keyed by family/member/year/process; scenario names never
enter a physical stream. Multi-day absences are shared by all household uses.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = json.loads(Path(__file__).with_name("portugal_realism_config.json").read_text())
N = 70176
N23 = 35040
DT = .25


def rng_for(root, identity, year, process):
    # Checkout location must not influence the scientific random stream.
    family = "premium" if "premium" in Path(root).name else "core"
    key = f"{CONFIG['seed']}|{family}|{identity}|{year}|{process}"
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))


def at(clock, day, hour):
    local = (day.tz_localize(None).normalize()+pd.Timedelta(hours=float(hour))).tz_localize(
        "Europe/Lisbon", ambiguous=True, nonexistent="shift_forward")
    return int(np.clip(clock.searchsorted(local.tz_convert("UTC")), 0, len(clock)))


def build_calendar(root, members, calendar):
    clock = pd.DatetimeIndex(calendar.timestamp_utc)
    days = pd.date_range("2023-01-01", "2024-12-31", tz="Europe/Lisbon")
    cfg = CONFIG["activity"]
    rows = []
    for m in members.itertuples():
        if m.category != "residential":
            continue
        member_rows = []
        for year in (2023, 2024):
            rng = rng_for(root, m.member_id, year, "activity")
            individual = rng_for(root, m.member_id, 0, "travel_propensity")
            rate = individual.gamma(cfg["weekend_trip_rate_gamma_shape"], cfg["weekend_trip_rate_gamma_scale"])
            year_days = [d for d in days if d.year == year]
            from portugal_stochastic_demand import activity_intensity
            intensities = activity_intensity(root, m.member_id, year, len(year_days))
            leave_start = pd.Timestamp("2023-01-01")+pd.Timedelta(days=int(m.vacation_start_day)-1)
            leave_start = leave_start.replace(year=year).tz_localize("Europe/Lisbon")
            leave_rng = rng_for(root, m.member_id, year, "leave_dates")
            leave_start += pd.DateOffset(days=int(np.clip(round(leave_rng.normal(0, cfg["leave_start_jitter_days"])), -10, 10)))
            leave_days = max(3, int(m.vacation_days)+int(leave_rng.integers(-2, 3)))
            leave_start = max(pd.Timestamp(f"{year}-01-01",tz="Europe/Lisbon"), min(leave_start,
                pd.Timestamp(f"{year}-12-31",tz="Europe/Lisbon")-pd.DateOffset(days=leave_days)))
            work_rng = rng_for(root, m.member_id, 0, "personal_work_hours")
            personal_shift = float(work_rng.normal(0, .65))
            spans = []
            fridays = [d for d in year_days if d.month in (3, 4, 5, 6, 7, 9, 10) and d.dayofweek == 4]
            count = min(len(fridays), int(rng.poisson(rate)))
            for ix in rng.choice(len(fridays), count, replace=False):
                day = fridays[int(ix)]
                spans.append((day, int(rng.choice([2, 3], p=[.8, .2])), "weekend_away"))
            for day in year_days:
                if bool(calendar.is_national_holiday.iloc[at(clock, day, 0)]) and day.dayofweek in (1, 3):
                    if rng.random() < cfg["holiday_bridge_probability"]:
                        spans.append((day-pd.DateOffset(days=3) if day.dayofweek == 1 else day, 3, "holiday_bridge"))
            if rng.random() < cfg["vacation_travel_probability"]:
                duration = int(rng.integers(3, max(4, leave_days+1)))
                offset = int(rng.integers(0, max(1, leave_days-duration+1)))
                spans.append((leave_start+pd.DateOffset(days=offset), duration, "vacation_away"))
            # Merge overlapping absences instead of creating two simultaneous trips.
            intervals = []
            for day, duration, kind in sorted(spans):
                a = at(clock, day, rng.uniform(16, 20) if kind == "weekend_away" else rng.uniform(7, 11))
                b = at(clock, day+pd.DateOffset(days=duration), rng.uniform(17, 22))
                if intervals and a <= intervals[-1][1]:
                    intervals[-1] = (intervals[-1][0], max(b, intervals[-1][1]), intervals[-1][2])
                else:
                    intervals.append((a, b, kind))
            for day_number, day in enumerate(year_days):
                lo, hi = at(clock, day, 0), at(clock, day, 24)
                rest = bool(calendar.is_rest_day.iloc[lo])
                leave = leave_start <= day < leave_start+pd.DateOffset(days=leave_days)
                routine = m.routine_archetype
                telework = routine == "hybrid_worker" and not rest and not leave and day.dayofweek in (1, 3)
                if routine == "hybrid_worker" and not rest and not leave and rng.random() < cfg["hybrid_exception_probability"]:
                    telework = not telework
                work = not rest and not leave and routine not in ("retired", "school_runs") and not telework
                if routine == "night_shift":
                    work = (day.dayofyear+int(m.member_index)) % 8 < 4 and not leave
                if work and rng.random() < cfg["work_exception_probability"]:
                    work = False
                dep, ret = (np.nan, np.nan)
                second_dep, second_ret = (np.nan, np.nan)
                school_run = routine == "school_runs" and not rest and not leave and not bool(calendar.is_school_break.iloc[lo])
                if work:
                    center, back = (14., 22.) if routine == "late_worker" else (20., 32.) if routine == "night_shift" else (7.75, 18.)
                    dep, ret = center+personal_shift+rng.normal(0, .45), back+personal_shift+rng.normal(0, .6)
                    if routine != "night_shift" and rng.random() < .12:
                        ret = min(23.75, ret+rng.uniform(.5, 2.))
                elif school_run:
                    dep, ret = 8.+rng.normal(0,.25), 9.25+rng.normal(0,.25)
                    second_dep, second_ret = 15.5+rng.normal(0,.35), 17.+rng.normal(0,.35)
                elif rng.random() < (.65 if rest or leave else .45):
                    dep = rng.uniform(9, 15)
                    ret = min(23., dep+rng.uniform(1., 5.))
                overlaps = [(a, b, k) for a, b, k in intervals if a < hi and b > lo]
                a, b, kind = overlaps[0] if overlaps else (-1, -1, "")
                member_rows.append(dict(member_id=m.member_id, date=day.strftime("%Y-%m-%d"), year=year,
                    start_time_step=lo, end_time_step=hi, is_rest_day=rest, on_leave=leave,
                    work_from_home=telework, work_outside=work, departure_hour=dep, return_hour=ret,
                    secondary_departure_hour=second_dep, secondary_return_hour=second_ret,
                    activity_intensity=float(intensities[day_number]),
                    absence_start=a, absence_end=b, absence_kind=kind,
                    wake_hour=float(np.clip(rng.normal(8.5 if rest or leave else 7.2, .5), 5.5, 11)),
                    sleep_hour=float(np.clip(rng.normal(23., .5), 21.5, 24.5))))
        rows.extend(member_rows)
    return pd.DataFrame(rows)


def presence(clock, daily, occupants):
    """Fraction at home; long trips remove the whole household, work the driver.

    Other occupants retain a conservative 50% expected daytime presence. This
    is an expected occupancy proxy, not a simulated census of each person.
    """
    result = np.ones(len(clock))
    for d in daily.itertuples():
        if np.isfinite(d.departure_hour):
            day = pd.Timestamp(d.date, tz="Europe/Lisbon")
            a, b = at(clock, day, d.departure_hour), at(clock, day, d.return_hour)
            result[a:b] = min(result[a:b].min(initial=1.), max(0., (occupants-1)*.5/max(occupants, 1)))
        if np.isfinite(d.secondary_departure_hour):
            day = pd.Timestamp(d.date, tz="Europe/Lisbon")
            a, b = at(clock, day, d.secondary_departure_hour), at(clock, day, d.secondary_return_hour)
            result[a:b] = np.minimum(result[a:b], max(0., (occupants-1)*.5/max(occupants, 1)))
        if d.absence_start >= 0:
            result[int(d.absence_start):int(d.absence_end)] = 0.
    return result


def mobility(v, member, old, original_sessions, calendar, daily, root):
    """Generate travel on the same activity calendar used by domestic end uses."""
    clock = pd.DatetimeIndex(calendar.timestamp_utc)
    away = []
    for year, group in daily.groupby("year"):
        rng = rng_for(root, v.ev_id, year, "journeys")
        for d in group.itertuples():
            if not np.isfinite(d.departure_hour):
                continue
            day = pd.Timestamp(d.date, tz="Europe/Lisbon")
            a, b = at(clock, day, d.departure_hour), at(clock, day, d.return_hour)
            km = rng.lognormal(np.log(35 if d.work_outside else 20), .45)
            away.append(dict(start=a, end=b, kind="routine" if d.work_outside else "local", energy=km*v.consumption_kwh_per_km))
            if np.isfinite(d.secondary_departure_hour):
                a,b=at(clock,day,d.secondary_departure_hour),at(clock,day,d.secondary_return_hour)
                away.append(dict(start=a,end=b,kind="local",energy=float(rng.lognormal(np.log(12),.35))*v.consumption_kwh_per_km))
        for d in group.loc[group.absence_start.ge(0)].drop_duplicates(["absence_start", "absence_end"]).itertuples():
            a, b = int(d.absence_start), int(d.absence_end)
            away = [x for x in away if not (x["start"] < b and x["end"] > a)]
            away.append(dict(start=a, end=b, kind=d.absence_kind, energy=float(rng.uniform(180, 550))*v.consumption_kwh_per_km))
    away.sort(key=lambda x: x["start"])
    # Overnight shifts can intersect a next-day errand: merge physical absences.
    merged = []
    for item in away:
        if merged and item["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], item["end"])
            merged[-1]["energy"] += item["energy"]
        else:
            merged.append(item.copy())
    travel, external = np.zeros(N), np.zeros(N)
    episodes = []
    for item in merged:
        a, b = item["start"], item["end"]
        if a >= N or b <= a:
            continue
        travel[a] += .5*item["energy"]
        if b < N:
            travel[b] += .5*item["energy"]
        interior = []
        if item["kind"] in ("vacation_away", "weekend_away", "holiday_bridge"):
            rng = rng_for(root, v.ev_id, 2023 if a < N23 else 2024, f"destination-{a}")
            for t in range(a+96, b-48, 96):
                if rng.random() < CONFIG["activity"]["destination_trip_daily_probability"]:
                    interior.append(t)
                    travel[t] += float(rng.uniform(5, 55))*v.consumption_kwh_per_km
        episodes.append(dict(ev_id=v.ev_id, departure_time_step=a, return_time_step=b,
                             trip_kind=item["kind"], destination_trip_count=len(interior)))
    for lo, hi in ((0, N23), (N23, N)):
        travel[lo:hi] *= float(old.trip_energy_kwh.iloc[lo:hi].sum())/travel[lo:hi].sum()
    # Finite, explicit destination/public charging before long return legs. A
    # trip cannot consume >65% capacity between available public stops.
    for item, record in zip(merged, episodes):
        a, b = item["start"], item["end"]
        record["trip_energy_kwh"] = float(travel[a:min(b+1, N)].sum())
        record["energy_equivalent_distance_km"] = record["trip_energy_kwh"]/v.consumption_kwh_per_km
        for t in np.flatnonzero(travel[a:min(b+1, N)])+a:
            if travel[t] > v.battery_capacity_kwh*.65:
                # Split an exceptionally long leg into driving/public-charge
                # segments inside the away interval; never charge while home.
                energy = travel[t]
                travel[t] = 0.
                count = int(np.ceil(energy/(v.battery_capacity_kwh*.55)))
                start = a if t == a else max(a, t-count*5)
                for k in range(count):
                    slot = min(start+5*k, b-1)
                    travel[slot] += energy/count
        cumulative = 0.
        previous = a
        for t in np.flatnonzero(travel[a:min(b+1, N)])+a:
            if cumulative+travel[t] > v.battery_capacity_kwh*.65:
                slot = max(previous+1, t-4)
                assert slot < t, (v.ev_id, a, b, t)
                external[slot:t] += cumulative/(t-slot)
                cumulative = 0.
            cumulative += travel[t]
            previous = t
    candidates = []
    cursor = 0
    for x in merged:
        a, b = x["start"], x["end"]
        if a-cursor >= 2:
            candidates.append(dict(ev_id=v.ev_id, charger_id=v.home_charger_id, member_id=v.home_member_id,
                                   arrival_time_step=cursor, observed_end_time_step=a, purpose="home"))
        cursor = b
    if N-cursor >= 2:
        candidates.append(dict(ev_id=v.ev_id, charger_id=v.home_charger_id, member_id=v.home_member_id,
                               arrival_time_step=cursor, observed_end_time_step=N, purpose="home"))
    work = original_sessions.loc[original_sessions.purpose.ne("home")]
    if len(work):
        example = work.iloc[0]
        opened = pd.read_parquet(root/f"operations/{example.member_id}.parquet").is_open.to_numpy()
        for x in merged:
            if x["kind"] != "routine":
                continue
            available = np.flatnonzero(opened[x["start"]+2:x["end"]-2])+x["start"]+2
            if len(available) < 4:
                continue
            chunks = np.split(available, np.flatnonzero(np.diff(available)>1)+1)
            block = max(chunks, key=len)
            a, b = int(block[0]), int(block[-1]+1)
            if travel[a+1:b].any() or external[a:b].any():
                continue
            candidates.append(dict(ev_id=v.ev_id, charger_id=example.charger_id, member_id=example.member_id,
                                   arrival_time_step=a, observed_end_time_step=b, purpose=example.purpose))
    return travel, external, candidates, episodes

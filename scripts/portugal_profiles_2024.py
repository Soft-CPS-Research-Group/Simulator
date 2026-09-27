"""2024 temporal profiles for the fixed Portugal catalogue, data construction only.

Physical load/appliance/mobility models extracted from the archived v2.3 data
builder. No simulator imports. Identity and ratings come from frozen catalogues;
2024 operational randomness is separate from immutable equipment draws.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pvlib
import portugal_rec_common as common
from portugal_site_calendar import service_hours, opening_series
YEAR=2024
N=35136
DT=.25
HOLIDAYS={'01-01','03-29','03-31','04-25','05-01','05-30','06-10','08-15','10-05','11-01','12-01','12-08','12-25'}

def school_break(day):
    """Primary-school 2023/24 and 2024/25 calendars; local September start 16 Sep."""
    md = day.strftime("%m-%d")
    return (
        md <= "01-02"
        or "02-12" <= md <= "02-14"
        or "03-25" <= md <= "04-05"
        or "06-29" <= md <= "09-15"
        or md >= "12-18"
    )


def rng_for(family, i, offset=0):
    # Core-15 / Core-30 intentionally nested. Premium is an independent draw.
    return np.random.default_rng(
        20230925 + (100000 if family == "premium" else 0) + i * 101 + offset
    )


def pv_generation(meta, w):
    if not meta["has_pv"]:
        return np.zeros(N)
    pos = w["position"]
    zen = pos.zenith.to_numpy()
    azi = pos.azimuth.to_numpy()
    ghi = w["dhi"] + w["dni"] * np.maximum(np.cos(np.radians(zen)), 0)
    poa = pvlib.irradiance.get_total_irradiance(
        meta["pv_tilt_deg"],
        meta["pv_azimuth_deg"],
        zen,
        azi,
        w["dni"],
        ghi,
        w["dhi"],
        albedo=0.2,
    )["poa_global"]
    cell = pvlib.temperature.faiman(poa, w["temperature"], w["wind"])
    # Generic crystalline-silicon reference, -0.35%/K, 10% non-inverter losses,
    # 96% inverter, explicit DC nameplate and AC clipping, no battery included.
    dc = (
        meta["pv_capacity_kw"]
        * poa
        / 1000
        * np.maximum(1 - 0.0035 * (cell - 25), 0)
        * 0.90
        * meta["pv_shading_factor"]
    )
    ac = np.minimum(np.maximum(dc * 0.96, 0), meta["pv_inverter_ac_kw"])
    ac = np.where(pos.elevation.to_numpy() > 0, ac, 0)
    return ac * DT


def member_load(m, c, w, profiles, cfg):
    i = m["member_index"]
    r = rng_for(cfg.key, i, 17 + 1000000)
    local = pd.DatetimeIndex(c.timestamp_utc).tz_convert(common.TIMEZONE)
    hour = local.hour.to_numpy() + local.minute.to_numpy() / 60
    day = local.dayofyear.to_numpy()
    rest = c.is_rest_day.to_numpy()
    routine = m["routine_archetype"]
    vacation = (day >= m["vacation_start_day"]) & (
        day < m["vacation_start_day"] + m["vacation_days"]
    )
    shift = r.normal(0, 0.4, 367)[day]
    daily = r.lognormal(-0.5 * 0.14**2, 0.14, 367)[day]
    gauss = lambda center, width: common._periodic_gaussian(hour, center, width)
    if m["category"] == "residential":
        if routine == "school_runs":
            rest = rest | c.is_school_break.to_numpy()
        if routine == "night_shift":
            rest = day % 8 >= 4
        morning = {
            "commuter": 7,
            "hybrid_worker": 8,
            "night_shift": 9,
            "school_runs": 7.5,
            "retired": 9,
            "late_worker": 10,
        }[routine]
        evening = {
            "commuter": 19.5,
            "hybrid_worker": 19,
            "night_shift": 18,
            "school_runs": 19,
            "retired": 18.5,
            "late_worker": 22,
        }[routine]
        home_day = (
            ((day + i) % 5 < 2)
            if routine == "hybrid_worker"
            else np.full(N, routine in ("school_runs", "retired"))
        )
        activity = (
            0.22
            + gauss(morning + shift + rest * 0.7, 0.75) * 0.65
            + gauss(evening + shift * 0.4, 1.0)
            + (0.15 + 0.4 * (home_day | rest)) * gauss(13, 1.8)
        )
        activity *= np.where(vacation, 0.35, 1)
        annual = 650 + 520 * m["occupants"] + 3.5 * m["floor_area_m2"]
        annual *= r.uniform(0.85, 1.15)
        dhw = 420 * m["occupants"] if m["electric_dhw"] else 0
        heat = 4.5 * m["floor_area_m2"] if m["electric_space_heating"] else 0
        cool = 1.7 * m["floor_area_m2"] if m["cooling_present"] else 0
        prior = profiles.BTN_C.to_numpy()
        prior = prior / prior.mean()
        shape = (
            (0.4 * prior + 0.6 * activity / activity.mean())
            * daily
            * r.lognormal(-0.5 * 0.12**2, 0.12, N)
        )
        shape *= np.where(vacation, 0.4, 1)
    else:
        on = opening_series(m, c)
        floor = 0.35 if routine in ("grocery", "care_home") else 0.10
        shape = (
            (floor + on * (1 + 0.2 * gauss(13 + shift, 2)))
            * daily
            * r.lognormal(-0.5 * 0.07**2, 0.07, N)
        )
        annual = {
            "office": m["floor_area_m2"] * 35,
            "retail": m["floor_area_m2"] * 40,
            "clinic": m["floor_area_m2"] * 45,
            "cafe": m["floor_area_m2"] * 70,
            "workshop": m["floor_area_m2"] * 45,
            "sports_centre": 32000,
            "grocery": 48000,
            "care_home": 45000,
        }[routine]
        dhw = annual * 0.08 if routine in ("cafe", "care_home", "sports_centre") else 0
        heat = annual * 0.12
        cool = annual * 0.08

    def scale(shape, energy):
        return np.asarray(shape) * energy / max(float(np.sum(shape)), 1e-12)

    base = scale(shape, annual)
    hotwater = scale(0.1 + gauss(8 + shift, 1) + gauss(20, 1.4), dhw)
    thermal_activity = (
        (0.3 + shape / shape.mean())
        if m["category"] == "residential"
        else np.where(on, 1.0, 0.08)
    )
    heating = scale(np.maximum(18 - w["temperature"], 0) * thermal_activity, heat)
    cooling = scale(np.maximum(w["temperature"] - 24, 0) * thermal_activity, cool)
    standby = np.full(N, m["battery_standby_w"] / 1000 * DT)
    ledger = pd.DataFrame(
        dict(
            base_excluding_cycles_kwh=base,
            dhw_proxy_kwh=hotwater,
            heating_proxy_kwh=heating,
            cooling_proxy_kwh=cooling,
            battery_standby_kwh=standby,
            fixed_appliances_kwh=np.zeros(N),
        )
    )
    # Thermal energy fields remain zero: these are uncontrolled electrical end
    # uses, not validated thermodynamic comfort models.
    frame = pd.DataFrame(
        dict(
            month=local.month,
            hour=local.hour + 1,
            minutes=local.minute,
            day_type=np.where(rest, 1, local.dayofweek + 2),
            daylight_savings_status=c.daylight_savings_status,
            indoor_dry_bulb_temperature=np.full(N, 21.0),
            average_unmet_cooling_setpoint_difference=np.zeros(N),
            indoor_relative_humidity=np.full(N, 50.0),
            occupant_count=np.full(N, m["occupants"]),
            non_shiftable_load=ledger.sum(axis=1),
            dhw_demand=np.zeros(N),
            cooling_demand=np.zeros(N),
            heating_demand=np.zeros(N),
            solar_generation=pv_generation(m, w),
            power_outage=np.zeros(N),
        )
    )
    return frame, ledger


def appliance(m, c, kind, controlled, cfg):
    """Noninterruptible electrical cycles; synthetic shapes, label-scale budgets.

    Fixed machines are embedded once in the end-use ledger. Controllable ones
    are absent from native load and added only when the simulator runs them.
    """
    r = rng_for(cfg.key, m["member_index"], 41 + (kind == "dishwasher"))
    i = m["member_index"]
    res = m["category"] == "residential"
    routine = m["routine_archetype"]
    ident = f"DEF-M{i:03d}-" + ("01" if kind == "washing_machine" else "02")
    professional = not res and (kind == "washing_machine" or routine == "cafe")
    if professional:
        parameters = (
            [
                ("linen_eco", int(r.integers(6, 9)), float(r.uniform(1.3, 1.7))),
                ("linen_hot", int(r.integers(5, 8)), float(r.uniform(1.7, 2.2))),
            ]
            if kind == "washing_machine"
            else [
                ("catering_batch", 2, float(r.uniform(0.9, 1.2))),
                ("catering_heavy_batch", 3, float(r.uniform(1.2, 1.5))),
            ]
        )
        nominal_kw = 4.6 if kind == "washing_machine" else 6.0
        use = (
            "linen_laundry"
            if kind == "washing_machine"
            else "catering_dishwashing_batch"
        )
    elif kind == "washing_machine":
        parameters = [
            ("eco_40_60", int(r.integers(13, 17)), float(r.uniform(0.48, 0.75))),
            ("mixed_40", int(r.integers(6, 9)), float(r.uniform(0.65, 1.05))),
        ]
        nominal_kw = 2.3
        use = "household_laundry"
    else:
        parameters = [
            ("eco_50", int(r.integers(14, 18)), float(r.uniform(0.72, 1.02))),
            ("daily_60", int(r.integers(7, 10)), float(r.uniform(1.00, 1.28))),
        ]
        nominal_kw = 2.3
        use = "household_dishes" if res else "staff_kitchen_dishes"
    profiles = []
    for k, (program, duration, energy) in enumerate(parameters):
        weights = np.full(duration, 0.05 / duration)
        if professional and kind == "dishwasher":
            weights += 0.95 / duration
        elif kind == "washing_machine":
            weights[1] += 0.45
            weights[2] += 0.30
            weights[duration // 2] += 0.10
            weights[-1] += 0.10
        else:
            weights[1] += 0.40
            weights[max(2, duration - 4)] += 0.40
            weights[duration // 2] += 0.10
            weights[-1] += 0.05
        values = weights / weights.sum() * energy
        # Overlapping heater stages in short programs spill into the adjacent
        # sampled stages. Keep cycle energy while respecting electrical rating.
        while values.max() > nominal_kw * DT + 1e-10:
            excess = np.maximum(values - nominal_kw * DT, 0).sum()
            values = np.minimum(values, nominal_kw * DT)
            free = values < nominal_kw * DT - 1e-10
            values[free] += excess / free.sum()
        assert values.max() / DT <= nominal_kw + 1e-8
        profiles.append(
            dict(
                profile_id=f"{ident}-P{k}",
                program=program,
                duration_steps=duration,
                total_energy_kwh=float(values.sum()),
                load_profile=json.dumps(values.tolist()),
            )
        )
    pro = pd.DataFrame(profiles)
    r = rng_for(cfg.key, i, 1000041 + (kind == "dishwasher"))
    utc = pd.DatetimeIndex(c.timestamp_utc)
    at = lambda day, hour: int(
        utc.searchsorted(common._local_at(day, int(round(hour * 4) * 15)))
    )
    rows = []
    fixed = np.zeros(N)
    last = 0
    for day in pd.date_range("2024-01-01", "2024-12-31", tz=common.TIMEZONE):
        holiday = day.strftime("%m-%d") in HOLIDAYS | {"06-13"}
        off = day.dayofweek >= 5 or holiday
        vacation = (
            m["vacation_start_day"]
            <= day.dayofyear
            < m["vacation_start_day"] + m["vacation_days"]
        )
        if vacation:
            continue
        hours = None if res else service_hours(m, day)
        if not res and hours is None:
            continue
        prob = (
            min(
                0.92,
                (
                    (0.17 + 0.10 * m["occupants"])
                    if kind == "washing_machine"
                    else (0.25 + 0.12 * m["occupants"])
                ),
            )
            if res
            else (0.93 if professional else 0.75)
        )
        if r.random() > min(0.98, prob * (1.25 if off and res else 1)):
            continue
        count = 2 if professional and r.random() < 0.6 else 1
        for batch in range(count):
            daytime = off or routine in (
                "retired",
                "school_runs",
                "hybrid_worker",
                "night_shift",
            )
            h = r.uniform(9, 14) if daytime else r.uniform(18, 20.5)
            # Some households load a delayed-start machine before work. Windows
            # vary per day; do not move the physical job/energy to manufacture PV.
            before_work = res and controlled and not daytime and r.random() < 0.45
            if before_work:
                h = r.uniform(6.5, 7.5)
            if not res:
                lo, hi = hours
                h = (
                    min(lo + (4.5 if hi - lo >= 8 else 1), hi - 2.5)
                    + r.uniform(0, 0.5)
                    + batch * 4
                )
            start = max(at(day, h), last)
            close = min(N, at(day, 23 if res else hours[1]))
            p = pro.iloc[int(r.choice([0, 1], p=[0.7, 0.3]))]
            if start + int(p.duration_steps) > close:
                p = pro.loc[pro.duration_steps.idxmin()]
            v = np.array(json.loads(p.load_profile))
            if start + len(v) > close:
                continue
            slack = {"narrow": 4, "moderate": 12, "wide": 24}[
                m["flexibility_preference"]
            ]
            if before_work:
                slack = max(slack, 28)  # seven hours: timer start before returning
            if off and res:
                slack += 8
            if professional:
                slack = 2 if kind == "dishwasher" else 8
            end = min(start + len(v) + slack, close)
            rows.append(
                dict(
                    cycle_id=f"{ident}-2024-C{len(rows):04d}",
                    profile_id=p.profile_id,
                    earliest_start_time_step=start,
                    latest_start_time_step=end - len(v),
                    deadline_time_step=end - 1,
                    priority=float(r.choice([0.5, 0.75, 1.0])),
                    must_run=True,
                    day_kind=(
                        "holiday" if holiday else ("weekend" if off else "weekday")
                    ),
                )
            )
            if not controlled:
                fixed[start : start + len(v)] += v
            last = end
    schedule = pd.DataFrame(rows)
    meta = dict(
        member_id=m["member_id"],
        deferrable_id=ident,
        appliance_type=kind,
        equipment_use=use,
        controlled=controlled,
        nominal_power_kw=nominal_kw,
        rated_capacity=(
            12
            if professional and kind == "washing_machine"
            else (
                int(r.choice([7, 8, 9]))
                if kind == "washing_machine"
                else int(r.choice([12, 13, 14]))
            )
        ),
        capacity_unit=(
            "kg"
            if kind == "washing_machine"
            else ("place_settings_per_batch" if professional else "place_settings")
        ),
        profile_method="synthetic integrated 15-minute heater/motor stages; label-scale energy/duration, not a measured waveform",
        annual_cycle_count=len(rows),
        annual_requested_energy_kwh=sum(
            float(
                pro.loc[pro.profile_id == x["profile_id"], "total_energy_kwh"].iloc[0]
            )
            for x in rows
        ),
        minimum_profile_energy_kwh=float(pro.total_energy_kwh.min()),
        maximum_profile_energy_kwh=float(pro.total_energy_kwh.max()),
        maximum_profile_power_kw=max(max(json.loads(v)) / DT for v in pro.load_profile),
        phase_connection="L1",
    )
    return pro, schedule, meta, fixed


def mobility(cfg, members, contracts, c):
    """Create shared vehicle identities, nonoverlapping physical visits and trips.

    Generate descriptive trips to derive prescribed arrival SOC. The original
    runtime uses independent arrival SOC, not this synthetic ledger as a
    longitudinal battery trajectory. All timestamps use local civil time.
    """
    utc = pd.DatetimeIndex(c.timestamp_utc)
    days = pd.date_range("2024-01-01", "2024-12-31", tz=common.TIMEZONE)
    step = lambda d, h: int(
        np.clip(utc.searchsorted(common._local_at(d, int(round(h * 4) * 15))), 0, N)
    )
    vehicles = {}
    sessions = []
    chargers = []
    trips = {}
    external = {}
    bymember = {m["member_index"]: m for m in members}
    office = 81 if cfg.key == "premium" else (25 if cfg.key == "core30" else None)
    linked = [
        i for i in (1, 26, 38, 50) if i in cfg.charger_counts and office is not None
    ]

    def new_ev(eid, i, j, outside=False):
        r = rng_for(cfg.key, i, 100 + j)
        cap = float(
            r.choice(
                [40, 50, 60, 64, 77, 82, 95, 105],
                p=[0.10, 0.16, 0.20, 0.20, 0.18, 0.08, 0.05, 0.03],
            )
        )
        habits = ("needs_50", "balanced_70", "reserve_80", "reserve_80", "high_90")[
            (i + j) % 5
        ]
        vehicles[eid] = dict(
            ev_id=eid,
            home_member_id="" if outside else common._member_id(i),
            home_charger_id="" if outside else common._charger_id(i, j),
            work_member_id=common._member_id(office) if i in linked else "",
            vehicle_pool="external_visitors" if outside else "resident",
            battery_capacity_kwh=cap,
            nominal_power_kw=22.08,
            max_ac_charging_power=22.08 if i % 7 == 0 else 11.04,
            depth_of_discharge=1.0,
            initial_soc=0.80 if outside else 0.70,
            reserve_soc=0.18 + float(r.uniform(0, 0.08)),
            soc_preference=habits,
            consumption_kwh_per_km=float(
                r.uniform(0.18, 0.25) if cap >= 95 else r.uniform(0.14, 0.21)
            ),
            annual_energy_equivalent_distance_km=0.0,
            annual_trip_energy_kwh=0.0,
        )
        trips[eid] = np.zeros(N)
        external[eid] = np.zeros(N)
        return vehicles[eid], rng_for(cfg.key, i, 1000100 + j)

    def trip(eid, t, kwh):
        if 0 < t < N:
            trips[eid][t] += kwh

    def visit(eid, cid, i, start, stop, purpose):
        start = max(0, start)
        stop = min(N, stop)
        if stop - start < 2:
            return
        sessions.append(
            dict(
                ev_id=eid,
                charger_id=cid,
                member_id=common._member_id(i),
                arrival_time_step=start,
                departure_time_step=stop,
                purpose=purpose,
            )
        )

    for i, count in cfg.charger_counts.items():
        for j, spec in enumerate(contracts[common._member_id(i)]["charger_specs"], 1):
            chargers.append(
                dict(
                    charger_id=common._charger_id(i, j),
                    member_id=common._member_id(i),
                    physical_charger_number_at_member=j,
                    charger_class=(
                        "residential"
                        if bymember[i]["category"] == "residential"
                        else "shared_service"
                    ),
                    min_charging_power_kw=(
                        4.14 if spec["phase_connection"] == "all_phases" else 1.38
                    ),
                    **spec,
                )
            )
    for i, count in cfg.charger_counts.items():
        m = bymember[i]
        if m["category"] != "residential":
            continue
        for j in range(1, count + 1):
            eid = f"EV-M{i:03d}-{j:02d}"
            v, r = new_ev(eid, i, j)
            away = []
            cid = common._charger_id(i, j)
            routine = m["routine_archetype"] if j == 1 else "commuter"
            for day in days:
                rest = day.dayofweek >= 5 or day.strftime("%m-%d") in HOLIDAYS | {
                    "06-13"
                }
                vacation = (
                    m["vacation_start_day"]
                    <= day.dayofyear
                    < m["vacation_start_day"] + m["vacation_days"]
                )
                if routine == "school_runs":
                    rest = rest or school_break(day)
                if routine == "night_shift":
                    rest = day.dayofyear % 8 >= 4
                if vacation:
                    # Vehicle stays at home for part of annual leave, with local
                    # leisure trips; holidays are not forced daily commuting.
                    rest = True
                if rest:
                    if r.random() < 0.38:
                        continue
                    leave = float(r.uniform(9, 14))
                    back = leave + float(r.uniform(1.5, 5))
                    km = float(r.uniform(6, 65))
                    long = r.random() < 0.035
                    if long:
                        leave = 7.0
                        back = 20.0
                        km = float(r.uniform(160, 230))
                elif routine == "night_shift":
                    if day.dayofyear % 8 >= 4:
                        continue
                    leave = 21.0 + float(r.normal(0, 0.25))
                    back = 31.0 + float(r.normal(0, 0.35))
                    km = float(r.uniform(24, 60))
                    long = False
                elif routine == "school_runs":
                    # Two short returns home, not one fabricated overnight absence.
                    for leave, back in ((8.0, 9.25), (15.75, 17.0)):
                        a, b = step(day, leave + float(r.normal(0, 0.15))), step(
                            day, back + float(r.normal(0, 0.2))
                        )
                        away.append((a, b))
                        trip(eid, a, float(r.uniform(0.6, 1.3)))
                        trip(eid, b, float(r.uniform(0.6, 1.3)))
                    continue
                elif routine == "retired":
                    if r.random() < 0.45:
                        continue
                    leave = float(r.uniform(9, 14))
                    back = leave + float(r.uniform(1, 3))
                    km = float(r.uniform(5, 30))
                    long = False
                elif routine == "hybrid_worker" and day.dayofweek in (1, 3):
                    if r.random() < 0.6:
                        continue
                    leave = 12.0
                    back = 14.0
                    km = float(r.uniform(4, 20))
                    long = False
                else:
                    late = routine == "late_worker"
                    leave = (12.0 if late else 8.0) + float(r.normal(0, 0.45))
                    back = (22.0 if late else 18.0) + float(r.normal(0, 0.65))
                    km = float(r.uniform(15, 65))
                    long = False
                a, b = step(day, leave), step(day, back)
                if b <= a:
                    continue
                away.append((a, b))
                energy = (
                    km
                    * v["consumption_kwh_per_km"]
                    * (1.10 if day.month in (1, 2, 12) else 1.0)
                )
                trip(eid, a, energy * 0.5)
                trip(eid, b, energy * 0.5)
                if long:
                    t = min(a + 8, N - 1)
                    external[eid][t : min(t + 4, N)] += (
                        max(energy * 0.25, energy - v["battery_capacity_kwh"] * 0.60)
                        / 4
                    )
                if (
                    i in linked
                    and j == 1
                    and not rest
                    and not vacation
                    and service_hours(bymember[office], day) is not None
                    and back - leave > 6
                    and leave < 10
                ):
                    wcid = common._charger_id(office, linked.index(i) + 1)
                    hours = service_hours(bymember[office], day)
                    start = max(a + 3, step(day, hours[0]))
                    stop = min(b - 3, step(day, hours[1]))
                    if day.dayofyear % 3 == 0:
                        lunch = step(day, 12.5)
                        ret = step(day, 13.5)
                        visit(eid, wcid, office, start, lunch, "work_morning")
                        visit(eid, wcid, office, ret, stop, "work_afternoon")
                        trip(eid, lunch, 1.4)
                    else:
                        visit(eid, wcid, office, start, stop, "work_day")
            # Union away intervals; complement is all parking at home, including
            # long stays and start/end-of-year partial sessions.
            merged = []
            for a, b in sorted(away):
                if merged and a <= merged[-1][1]:
                    merged[-1][1] = max(b, merged[-1][1])
                else:
                    merged.append([a, b])
            cursor = 0
            for a, b in merged:
                visit(eid, cid, i, cursor, a, "home")
                cursor = b
            visit(eid, cid, i, cursor, N, "home")
    for i, count in cfg.charger_counts.items():
        m = bymember[i]
        if m["category"] == "residential":
            continue
        for j in range(1, count + 1):
            if i == office and j <= len(linked):
                continue
            cid = common._charger_id(i, j)
            for slot in range(2):
                eid = f"EV-V{i:03d}-{j:02d}-{slot}"
                v, r = new_ev(eid, i, j * 10 + slot, True)
                for day in days:
                    hours = service_hours(m, day)
                    if hours is None:
                        continue
                    # This parking bay's visitor cohort starts in April. The
                    # dynamic scenario commissions its physical charger then.
                    if (
                        cfg.key == "premium"
                        and i == 86
                        and j == 4
                        and day < pd.Timestamp("2023-04-05", tz=common.TIMEZONE)
                    ):
                        continue
                    # Named repeat visitors attend intermittently, never on a
                    # closed date. Retail stops differ from office visits.
                    if r.random() > (
                        0.60 if m["routine_archetype"] == "office" else 0.48
                    ):
                        continue
                    lo = max(hours[0], 8.25 if slot == 0 else 13.5)
                    hi = min(hours[1], 12.75 if slot == 0 else 19.0)
                    if hi - lo < 1.5:
                        continue
                    arrival = lo + float(r.uniform(0, min(0.5, hi - lo - 1.5)))
                    duration = (
                        float(r.uniform(1.5, 2.5))
                        if m["routine_archetype"] == "retail"
                        else hi - arrival - float(r.uniform(0, 0.25))
                    )
                    a = step(day, arrival)
                    b = min(step(day, hi), max(a + 6, step(day, min(hi, arrival + duration))))
                    energy = float(r.uniform(5, 13))
                    trip(eid, max(a - 2, 1), energy * 0.5)
                    trip(eid, min(b + 2, N - 1), energy * 0.5)
                    visit(
                        eid,
                        cid,
                        i,
                        a,
                        b,
                        "visitor_morning" if slot == 0 else "visitor_afternoon",
                    )
                    # Explicit, finite outside-community home supply, 3.68 kW AC
                    # at 90% wall-to-battery efficiency for 15–45 minutes. These
                    # destination-dependent visitors do not fully recharge at home.
                    t = step(day, 20.0)
                    external[eid][t : min(t + int(r.integers(1, 4)), N)] += (
                        3.68 * 0.90 * DT
                    )
    # Target considers every trip before the next charging opportunity plus the
    # driver's reserve and usual target preference. Long-trip target rises.
    charger_byid = {x["charger_id"]: x for x in chargers}
    for eid, v in vehicles.items():
        vs = sorted(
            [s for s in sessions if s["ev_id"] == eid],
            key=lambda x: x["arrival_time_step"],
        )
        if any(
            a["departure_time_step"] > b["arrival_time_step"]
            for a, b in zip(vs, vs[1:])
        ):
            raise ValueError(f"Double booking {eid}")
        for k, s in enumerate(vs):
            next_start = vs[k + 1]["arrival_time_step"] if k + 1 < len(vs) else N
            need = max(
                0.0,
                float(
                    trips[eid][s["departure_time_step"] : min(next_start + 1, N)].sum()
                    - external[eid][
                        s["departure_time_step"] : min(next_start + 1, N)
                    ].sum()
                ),
            )
            base = {
                "needs_50": 0.55,
                "balanced_70": 0.70,
                "reserve_80": 0.80,
                "high_90": 0.90,
            }[v["soc_preference"]]
            variation = float(
                rng_for(cfg.key, abs(sum(map(ord, eid))), k + 1000000).uniform(-0.04, 0.04)
            )
            reason = "next_trips_plus_reserve_and_preference"
            target = min(
                1.0,
                max(
                    0.50,
                    base + variation,
                    v["reserve_soc"] + need / v["battery_capacity_kwh"],
                ),
            )
            # 100% is an explicit occasional home request before a long journey
            # or a driver's periodic full charge, not a daily fleet-wide default.
            if s["purpose"] == "home" and (
                need >= 28.0 or (v["soc_preference"] == "high_90" and k % 17 == 0)
            ):
                target = 1.0
                reason = (
                    "long_journey_full_charge"
                    if need >= 28.0
                    else "periodic_full_charge_preference"
                )
            s.update(
                session_id=f"{eid}-2024-S{k:04d}",
                target_soc=target,
                next_trip_energy_kwh=need,
                target_reason=reason,
                day_kind=str(c.day_kind.iloc[s["arrival_time_step"]]),
            )
        # Plan ahead across parking sessions: a slow home socket may need
        # more than one night before a full-charge request. Propagate required
        # departure reserves backwards, preserving the requested journey and
        # charger rating. This changes synthetic input targets, not the engine.
        def minimum_arrival_soc(session):
            ch=charger_byid[session['charger_id']]
            ac=ch['max_charging_power_kw']*.95
            nominal=min(v['nominal_power_kw'],v['max_ac_charging_power']*.95)
            def attainable(soc):
                for _ in range(session['departure_time_step']-session['arrival_time_step']):
                    power=min(ac,nominal*float(np.interp(soc,[0,.9,1],[1,1,.25])))
                    soc=min(1.,soc+power*DT*np.sqrt(.96)/v['battery_capacity_kwh'])
                    if soc>=session['target_soc']-1e-10:return True
                return False
            lo,hi=0.,session['target_soc']
            if attainable(0.):return 0.
            for _ in range(32):
                mid=(lo+hi)/2
                if attainable(mid):hi=mid
                else:lo=mid
            return hi
        for current,previous in zip(reversed(vs[1:]),reversed(vs[:-1])):
            if current['target_soc']<.999:continue
            required=minimum_arrival_soc(current)
            travel=float(trips[eid][previous['departure_time_step']:current['arrival_time_step']+1].sum())
            outside=float(external[eid][previous['departure_time_step']:current['arrival_time_step']+1].sum())
            target=min(1.,required+(max(0.,travel-outside)+2.)/v['battery_capacity_kwh'])
            if target>previous['target_soc']:
                previous['target_soc']=target
                previous['target_reason']='advance_reserve_for_next_full_charge_request'
        # Reference state is a transparent greedy feasible-power trajectory.
        # It supplies per-session arrival SOC inputs; it is not a runtime trajectory.
        target = np.zeros(N)
        power = np.zeros(N)
        for s in vs:
            a, b = s["arrival_time_step"], s["departure_time_step"]
            target[a:b] = s["target_soc"]
            power[a:b] = (
                min(
                    charger_byid[s["charger_id"]]["max_charging_power_kw"],
                    v["max_ac_charging_power"],
                )
                * 0.95
                * np.sqrt(0.96)
                * DT
            )
        soc = np.zeros(N)
        energy = v["initial_soc"] * v["battery_capacity_kwh"]
        for t in range(N):
            energy = max(0.0, energy - trips[eid][t])
            energy = min(v["battery_capacity_kwh"], energy + external[eid][t])
            soc[t] = energy / v["battery_capacity_kwh"]
            energy += min(
                power[t], max(target[t] * v["battery_capacity_kwh"] - energy, 0)
            )
        v["annual_trip_energy_kwh"] = float(trips[eid].sum())
        v["annual_energy_equivalent_distance_km"] = (
            v["annual_trip_energy_kwh"] / v["consumption_kwh_per_km"]
        )
        v["annual_external_charge_offered_kwh"] = float(external[eid].sum())
        for s in vs:
            s["arrival_soc_reference"] = float(soc[s["arrival_time_step"]])
            s["requested_energy_reference_kwh"] = max(
                0,
                (s["target_soc"] - s["arrival_soc_reference"])
                * v["battery_capacity_kwh"],
            )
        v["_ledger"] = pd.DataFrame(
            dict(
                trip_energy_kwh=trips[eid],
                external_charge_kwh=external[eid],
                reference_soc=soc,
            )
        )
    schedules = {}
    for ch in chargers:
        cid = ch["charger_id"]
        cs = sorted(
            [s for s in sessions if s["charger_id"] == cid],
            key=lambda x: x["arrival_time_step"],
        )
        if any(
            a["departure_time_step"] > b["arrival_time_step"]
            for a, b in zip(cs, cs[1:])
        ):
            raise ValueError(f"Charger overlap {cid}")
        f = pd.DataFrame(
            dict(
                electric_vehicle_charger_state=np.full(N, 3, dtype="int8"),
                electric_vehicle_id=np.full(N, "", dtype=object),
                electric_vehicle_session_id=np.full(N, "", dtype=object),
                electric_vehicle_departure_time=np.full(N, np.nan),
                electric_vehicle_required_soc_departure=np.full(N, np.nan),
                electric_vehicle_estimated_arrival_time=np.full(N, np.nan),
                electric_vehicle_estimated_soc_arrival=np.full(N, np.nan),
                electric_vehicle_current_soc=np.full(N, np.nan),
            )
        )
        previous = 0
        for s in cs:
            a, b = s["arrival_time_step"], s["departure_time_step"]
            eid = s["ev_id"]
            incoming = max(previous, a - 4, 0)
            f.loc[a : b - 1, "electric_vehicle_charger_state"] = 1
            f.loc[incoming : a - 1, "electric_vehicle_charger_state"] = 2
            f.loc[incoming : b - 1, "electric_vehicle_id"] = eid
            f.loc[incoming : b - 1, "electric_vehicle_session_id"] = s["session_id"]
            f.loc[a : b - 1, "electric_vehicle_departure_time"] = np.arange(
                b - a, 0, -1
            )
            f.loc[a : b - 1, "electric_vehicle_required_soc_departure"] = s[
                "target_soc"
            ]
            f.loc[incoming : a - 1, "electric_vehicle_estimated_arrival_time"] = (
                np.arange(a - incoming, 0, -1)
            )
            f.loc[incoming : b - 1, "electric_vehicle_estimated_soc_arrival"] = s[
                "arrival_soc_reference"
            ]
            f.loc[a : b - 1, "electric_vehicle_current_soc"] = (
                vehicles[eid]["_ledger"].reference_soc.iloc[a:b].to_numpy()
            )
            previous = b
        ch["annual_session_count"] = len(cs)
        ch["vehicle_pool_size"] = len({s["ev_id"] for s in cs})
        schedules[cid] = f
    return chargers, sessions, list(vehicles.values()), schedules


def calendar():
    utc=pd.date_range('2024-01-01','2025-01-01',freq='15min',inclusive='left',tz='UTC');loc=utc.tz_convert(common.TIMEZONE)
    offsets=np.array([int(x.utcoffset().total_seconds()//60) for x in loc],dtype='int16')
    c=pd.DataFrame(dict(time_step=np.arange(N,dtype='int32'),timestamp_utc=utc,timestamp_local=loc.astype(str),timezone=common.TIMEZONE,utc_offset_minutes=offsets,month_local=loc.month.astype('int8'),day_local=loc.day.astype('int8'),day_of_week_local=(loc.dayofweek+1).astype('int8'),hour_local=loc.hour.astype('int8'),minute_local=loc.minute.astype('int8'),daylight_savings_status=(offsets!=0).astype('int8')))
    c['is_national_holiday']=np.isin(loc.strftime('%m-%d'),list(HOLIDAYS));c['is_municipal_holiday']=loc.strftime('%m-%d')=='06-13';c['is_weekend']=loc.dayofweek>=5
    c['is_rest_day']=c.is_weekend|c.is_national_holiday|c.is_municipal_holiday;c['is_school_break']=[school_break(x) for x in loc]
    c['day_kind']=np.where(c.is_national_holiday|c.is_municipal_holiday,'holiday',np.where(c.is_weekend,'weekend','weekday'));return c

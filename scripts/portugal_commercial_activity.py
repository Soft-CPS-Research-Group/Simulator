"""Sector-specific service activity; empirical anchors and explicit transfer priors.

BDG2 development buildings supply hourly statistical anchors, not Portuguese
small-business population estimates. Public hours stay contractual. Preparation,
staff attendance, demand intensity and process timing vary around those hours.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from portugal_activity import CONFIG, N, N23, DT, at, rng_for
from portugal_site_calendar import service_hours


def civil_service_hours(member,day):
    """Inventory leave ordinal is based on 2023; retain its month/day in 2024."""
    operational=dict(member)
    if day.is_leap_year and operational["vacation_start_day"]>59:
        operational["vacation_start_day"]+=1
    return service_hours(operational,day)


def presence_day(rng, hours, posted, kind, capacity, modulation, intensity, preparation, cleanup):
    """Sample people independently of electrical noise; rates remain scenario priors.

    Staff attend bounded shifts, clients arrive and leave, and care residents
    persist overnight. Area density is a capacity assumption, not a measurement
    or a statutory limit. Customers cannot enter outside public opening hours.
    """
    cfg=CONFIG["commercial"]["presence"]
    staff=np.zeros(len(hours),dtype=int);clients=staff.copy();residents=staff.copy()
    if posted is None:return staff,clients,residents
    start,end=posted
    if kind=="care_home":residents[:]=max(1,int(round(capacity*cfg["care_resident_fraction"])))
    slots=max(1,int(round(capacity*cfg["staff_fraction"][kind])))
    day_staff=kind in ("office","workshop")
    shifts=1 if day_staff else max(1,int(np.ceil((end-start)/9)))
    attendance=rng.beta(cfg["attendance_mean"]*25,(1-cfg["attendance_mean"])*25)
    for shift in range(shifts):
        lo=start+(end-start)*shift/shifts
        hi=start+(end-start)*(shift+1)/shifts
        for _ in range(max(1,int(rng.binomial(slots,attendance)))):
            arrival=lo+rng.uniform(-preparation,.25)
            departure=hi+rng.uniform(-.25,cleanup)
            if day_staff:
                # Public opening can exceed a person's working day. Avoid
                # replacing the whole office workforce at a fixed midday slot.
                arrival=start+rng.uniform(-.25,max(.25,end-start-8.))
                departure=min(end+cleanup,arrival+rng.uniform(7.5,9.))
            present=(hours>=arrival)&(hours<departure)
            # Independent lunch absence for office/workshop personnel.
            if kind in ("office","workshop") and rng.random()<.8:
                lunch=rng.normal(13,.4)
                present&=~((hours>=lunch)&(hours<lunch+rng.uniform(.5,1.)))
            staff+=present
    staff=np.minimum(staff,capacity-residents)
    live=0;leave_probability=1-np.exp(-DT/cfg["dwell_hours"][kind])
    target=capacity*cfg["client_fraction"][kind]*modulation*intensity
    for t,hour in enumerate(hours):
        visits_open=start<=hour<end
        if kind=="care_home":
            visits_open&=cfg["care_visiting_hours"][0]<=hour<cfg["care_visiting_hours"][1]
        if not visits_open:live=0
        else:
            live-=int(rng.binomial(live,leave_probability))
            live+=int(rng.poisson(max(0,target[t])*leave_probability))
            live=min(live,capacity-int(staff[t]+residents[t]))
        clients[t]=live
    return staff,clients,residents


def commercial_activity(root, member, calendar):
    cfg=CONFIG["commercial"]
    kind=member.routine_archetype;anchor=cfg["development_anchors"][kind]
    clock=pd.DatetimeIndex(calendar.timestamp_utc)
    local=clock.tz_convert("Europe/Lisbon")
    shape=np.zeros(N);activity=np.zeros(N);people=np.zeros(N);opened=np.zeros(N,dtype=bool);rows=[]
    staff_counts=np.zeros(N,dtype="int16");client_counts=staff_counts.copy();resident_counts=staff_counts.copy()
    individual=rng_for(root,member.member_id,0,"commercial_building")
    # Between-building variation is stable across years; daily variation is not.
    background=float(np.clip((1-cfg["shape_transfer_weight"])*(.35 if kind in ("grocery","care_home") else .1)+
        cfg["shape_transfer_weight"]*anchor["night_day_ratio"],.08,.85))
    background*=float(np.clip(individual.lognormal(0,.16),.7,1.3))
    area_per_person={"office":18,"retail":14,"clinic":18,"cafe":5,"workshop":28,
                     "sports_centre":12,"grocery":14,"care_home":22}[kind]
    capacity=max(2,int(round(member.floor_area_m2/area_per_person)))
    peaks={"office":((10,.25,2.2),(15,.25,2.2)),"retail":((12,.2,2),(17,.55,2)),
           "clinic":((10,.35,1.5),(16,.3,1.7)),"cafe":((8,.55,1),(13,1.1,1.3),(19,.8,1.5)),
           "workshop":((10,.4,1.7),(15,.4,1.6)),"sports_centre":((8,.3,1.3),(19,1.,1.8)),
           "grocery":((11,.3,1.7),(18,.5,1.8)),"care_home":((8,.3,1),(13,.4,1.3),(19,.35,1.3))}[kind]
    for year in (2023,2024):
        rng=rng_for(root,member.member_id,year,"commercial_activity")
        presence_rng=rng_for(root,member.member_id,year,"commercial_people")
        dates=pd.date_range(f"{year}-01-01",f"{year}-12-31",tz="Europe/Lisbon")
        sigma=anchor["daily_log_sigma"]*cfg["daily_sigma_multiplier"]
        rho=anchor["daily_persistence"]
        shocks=rng.normal(0,sigma*np.sqrt(1-rho*rho),len(dates));shocks[0]=rng.normal(0,sigma)
        state=lfilter([1],[1,-rho],shocks)
        for i,day in enumerate(dates):
            a,b=at(clock,day,0),at(clock,day,24)
            hours=local[a:b].hour.to_numpy()+local[a:b].minute.to_numpy()/60
            posted=civil_service_hours(member._asdict(),day)
            intensity=float(np.exp(state[i]-.5*sigma*sigma))
            envelope=np.zeros(b-a)
            preparation=cleanup=0.
            if posted:
                start,end=posted
                opened[a:b]=(hours>=start)&(hours<end)
                preparation=float(rng.uniform(.25,1.25));cleanup=float(rng.uniform(.25,1.))
                if kind=="care_home":envelope[:]=1.
                else:envelope=np.clip((hours-start+preparation)/preparation,0,1)*np.clip((end+cleanup-hours)/cleanup,0,1)
            timing=float(rng.normal(0,cfg["timing_jitter_hours"]))
            modulation=np.full(b-a,.65)
            for center,height,width in peaks:
                modulation+=height*np.exp(-.5*((hours-center-timing)/width)**2)
            if kind in ("office","workshop"):
                modulation*=1-.25*np.exp(-.5*((hours-13-timing)/.55)**2)
            # Correlated process variation, not independent white noise per bin.
            noise=lfilter([np.sqrt(1-.75**2)],[1,-.75],rng.normal(0,cfg["intraday_log_sigma"],b-a))
            staff,clients,residents=presence_day(presence_rng,hours,posted,kind,capacity,
                modulation,intensity,preparation,cleanup)
            staff_counts[a:b]=staff;client_counts[a:b]=clients;resident_counts[a:b]=residents
            people[a:b]=staff+clients+residents
            occupancy_signal=people[a:b]/max(1.,capacity*.5)
            weight=cfg["presence"]["load_coupling_weight"]
            service=((1-weight)*envelope*modulation*intensity+weight*occupancy_signal)*np.exp(noise)
            # Background equipment remains energized on closed dates. A smooth
            # seasonal component represents aggregate equipment use, not HVAC.
            seasonal=1+.06*np.cos(2*np.pi*(day.dayofyear-210)/365.25)
            # Background IT/refrigeration/standby is persistent but not perfectly
            # flat on closed days. Otherwise mild-weather Sundays become exact
            # normalized copies even though the daily totals differ.
            shape[a:b]=background*seasonal*np.exp(.35*state[i]+.45*noise)+(1-background)*service
            activity[a:b]=service
            rows.append(dict(member_id=member.member_id,date=day.strftime("%Y-%m-%d"),year=year,
                is_service_day=posted is not None,activity_intensity=intensity,
                public_open_hour=posted[0] if posted else np.nan,public_close_hour=posted[1] if posted else np.nan,
                design_people_capacity=capacity,staff_person_hours=float(staff.sum()*DT),
                client_person_hours=float(clients.sum()*DT),resident_person_hours=float(residents.sum()*DT),
                people_status="Synthetic discrete staff/client/resident presence; capacity, attendance and dwell priors, not measured Portuguese occupancy"))
    operations=pd.DataFrame(dict(time_step=np.arange(N),is_open=opened,
        activity_intensity=activity.astype("float32"),expected_people=people.astype("float32"),
        staff_count=staff_counts,client_count=client_counts,resident_count=resident_counts))
    return shape,activity,people,pd.DataFrame(rows),operations


def commercial_end_uses(root,member,calendar,original):
    shape,activity,people,daily,operations=commercial_activity(root,member,calendar)
    ledger=original.copy();cfg=CONFIG["commercial"]
    kind=member.routine_archetype
    anchor=cfg.get("portuguese_annual_anchors",{}).get(kind)
    target=(anchor or cfg["development_anchors"][kind])["electricity_kwh_m2_year"]
    weight=anchor["transfer_weight"] if anchor else cfg["energy_transfer_weight"]
    sigma=cfg["office_between_building_log_sigma"] if kind=="office" else cfg["between_building_log_sigma"][kind]
    multiplier=rng_for(root,member.member_id,0,"commercial_annual_intensity").lognormal(-sigma*sigma/2,sigma)
    # Persistent differences between buildings, not a new annual redraw. Small
    # Portuguese audit dispersion is descriptive; unsupported sectors use an
    # explicit prior. Means/dispersion never come from the heldout BDG2 sample.
    target*=multiplier
    for lo,hi in ((0,N23),(N23,N)):
        old=original.iloc[lo:hi]
        # Log-space partial transfer prevents treating large overseas buildings
        # as direct observations of small Portuguese businesses. Explicit OAT
        # sensitivity includes no transfer and full transfer.
        old_total=old.drop(columns="fixed_appliances_kwh").to_numpy().sum()
        old_eui=old_total/member.floor_area_m2
        goal=old_total*(target/old_eui)**weight
        service=activity[lo:hi]
        for col in ("heating_proxy_kwh","cooling_proxy_kwh"):
            values=old[col].to_numpy()*(.25+.75*service)
            ledger.loc[lo:hi-1,col]=values*old[col].sum()/max(values.sum(),1e-12)
        dhw=.12+.88*service
        ledger.loc[lo:hi-1,"dhw_proxy_kwh"]=dhw*old.dhw_proxy_kwh.sum()/max(dhw.sum(),1e-12)
        thermal=ledger.loc[lo:hi-1,["dhw_proxy_kwh","heating_proxy_kwh","cooling_proxy_kwh"]].to_numpy().sum()
        assert goal>thermal
        ledger.loc[lo:hi-1,"base_excluding_cycles_kwh"]=shape[lo:hi]*(goal-thermal)/shape[lo:hi].sum()
    return ledger,people,daily,operations

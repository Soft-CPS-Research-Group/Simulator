"""Cross-domain regressions for commercial activity and measured programme use."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"));sys.path.insert(0,str(ROOT/"scripts/audit"))
from portugal_activity import CONFIG,N23
from portugal_commercial_activity import commercial_activity
from revise_portugal_ev_mobility import source_frame
from prepare_portugal_heldout_2024 import active_masks
from portugal_commercial_appliance_validation import extract_cycles,hourly_days,REFERENCE

COMMERCIAL=[ROOT/f"data/datasets/rec_pt_2023_{family}_v2" for family in ("core_30","premium_100")]


def test_measured_reference_integrity_and_disjoint_buildings():
    meta=json.loads((REFERENCE/"commercial_appliance_sources.json").read_text())
    for name,digest in meta["derived_sha256"].items():
        assert hashlib.sha256((REFERENCE/name).read_bytes()).hexdigest()==digest
    f=pd.read_parquet(REFERENCE/"bdg2_commercial_metadata.parquet")
    assert f.building_id.is_unique
    assert set(f.split)=={"development","heldout"}
    assert meta["appliance_protocol"]["house_split"]=={"2":"heldout","3":"development","5":"development","11":"heldout"}


@pytest.mark.parametrize("root",COMMERCIAL,ids=lambda p:p.name)
def test_commercial_public_hours_presence_and_charging_remain_consistent(root):
    members=pd.read_parquet(root/"catalogs/members.parquet")
    calendar=pd.read_parquet(root/"shared/calendar.parquet")
    sessions=pd.read_parquet(root/"catalogs/charging_sessions.parquet")
    daily=pd.read_parquet(root/"catalogs/commercial_activity.parquet")
    for m in members.loc[members.category.ne("residential")].itertuples():
        operations=pd.read_parquet(root/f"operations/{m.member_id}.parquet")
        frame=pd.read_parquet(root/f"timeseries/{m.member_id}.parquet")
        np.testing.assert_array_equal(operations.is_open,source_frame(root/f"operations/{m.member_id}.parquet").is_open)
        np.testing.assert_allclose(operations.expected_people,frame.occupant_count)
        assert frame.occupant_count.max()>0
        days=daily.loc[daily.member_id.eq(m.member_id)]
        assert len(days)==731 and days.date.is_unique and days.activity_intensity.nunique()>700
        assert frame.occupant_count.max()<=days.design_people_capacity.max()+1e-6
        if m.routine_archetype=="care_home":
            assert operations.is_open.all() and frame.occupant_count.min()>0
            local=pd.DatetimeIndex(calendar.timestamp_utc).tz_convert("Europe/Lisbon")
            start,end=CONFIG["commercial"]["presence"]["care_visiting_hours"]
            outside=(local.hour<start)|(local.hour>=end)
            assert operations.client_count.loc[outside].eq(0).all()
        for s in sessions.loc[sessions.member_id.eq(m.member_id)].itertuples():
            assert operations.is_open.iloc[s.arrival_time_step:s.observed_end_time_step].all()
        assert m.physical_charger_count<=m.parking_spaces


def test_commercial_realizations_reproduce_but_do_not_replay_years_or_seeds():
    root=COMMERCIAL[-1];members=pd.read_parquet(root/"catalogs/members.parquet")
    m=members.loc[members.routine_archetype.eq("office")].iloc[0]
    m=next(m.to_frame().T.itertuples(index=False))
    calendar=pd.read_parquet(root/"shared/calendar.parquet")
    old=deepcopy(CONFIG)
    try:
        a= commercial_activity(root,m,calendar)
        b= commercial_activity(root,m,calendar)
        np.testing.assert_array_equal(a[0],b[0])
        assert not np.array_equal(a[0][:N23],a[0][N23:N23*2])
        closed=a[3].loc[~a[3].is_service_day].iloc[0]
        local=pd.DatetimeIndex(calendar.timestamp_utc).tz_convert("Europe/Lisbon")
        slots=local.strftime("%Y-%m-%d")==closed.date
        assert np.std(a[0][slots])>0,"Closed premises still have varying background equipment"
        CONFIG["seed"]+=1
        c=commercial_activity(root,m,calendar)
        assert not np.array_equal(a[0],c[0])
        np.testing.assert_array_equal(a[-1].is_open,c[-1].is_open)
    finally:CONFIG.clear();CONFIG.update(old)


def test_people_are_discrete_and_independent_of_electrical_noise():
    root=COMMERCIAL[-1];members=pd.read_parquet(root/"catalogs/members.parquet")
    calendar=pd.read_parquet(root/"shared/calendar.parquet");old=deepcopy(CONFIG)
    m=next(members.loc[members.routine_archetype.eq("cafe")].itertuples())
    try:
        a=commercial_activity(root,m,calendar)
        CONFIG["commercial"]["intraday_log_sigma"]*=2
        b=commercial_activity(root,m,calendar)
        np.testing.assert_array_equal(a[2],b[2])
        assert not np.array_equal(a[0],b[0])
        assert np.array_equal(a[2],np.round(a[2]))
        assert (a[-1].client_count.loc[~a[-1].is_open]==0).all()
        np.testing.assert_array_equal(a[2],a[-1][["staff_count","client_count","resident_count"]].sum(axis=1))
        assert a[-1].client_count.max()>0 and a[-1].staff_count.max()>0
    finally:CONFIG.clear();CONFIG.update(old)


def test_office_public_hours_do_not_become_an_eleven_hour_workday():
    from portugal_commercial_activity import presence_day
    hours=np.arange(96)/4
    capacity=36;slots=round(capacity*CONFIG["commercial"]["presence"]["staff_fraction"]["office"])
    for seed in range(20):
        staff,clients,residents=presence_day(np.random.default_rng(seed),hours,(8,19),"office",capacity,np.ones(96),1.,.5,.5)
        assert staff.sum()*.25<=slots*9.25
        assert (clients[(hours<8)|(hours>=19)]==0).all()
        assert not residents.any()


def test_care_residents_and_staff_are_distinct_from_daytime_visitors():
    from portugal_commercial_activity import presence_day
    hours=np.arange(96)/4
    start,end=CONFIG["commercial"]["presence"]["care_visiting_hours"]
    for seed in range(20):
        staff,visitors,residents=presence_day(np.random.default_rng(seed),hours,(0,24),"care_home",41,np.ones(96),1.,.5,.5)
        assert (residents>0).all()
        assert (visitors[(hours<start)|(hours>=end)]==0).all()
        assert staff[(hours<start)|(hours>=end)].sum()>0
        assert visitors.sum()>0


@pytest.mark.parametrize("root",COMMERCIAL,ids=lambda p:p.name)
def test_commercial_annual_budget_includes_every_appliance_once(root):
    members=pd.read_parquet(root/"catalogs/members.parquet")
    totals=pd.read_parquet(root/"catalogs/member_year_totals.parquet")
    apps=pd.read_parquet(root/"catalogs/deferrables.parquet")
    for m in members.loc[members.category.ne("residential")].itertuples():
        frame=pd.read_parquet(root/f"timeseries/{m.member_id}.parquet")
        for year,lo,hi in ((2023,0,35040),(2024,35040,70176)):
            energy=float(frame.non_shiftable_load.iloc[lo:hi].astype(float).sum())
            for app in apps.loc[apps.member_id.eq(m.member_id)&apps.controlled].itertuples():
                profiles=pd.read_parquet(root/f"deferrables/{app.deferrable_id}_profiles.parquet").set_index("profile_id")
                schedule=pd.read_parquet(root/f"deferrables/{app.deferrable_id}_schedule.parquet")
                energy+=schedule.loc[schedule.year.eq(year),"profile_id"].map(profiles.total_energy_kwh).sum()
            target=totals.loc[totals.member_id.eq(m.member_id)&totals.year.eq(year),"building_budget_excluding_ev_kwh"].item()
            assert abs(energy-target)<.01


def test_professional_linen_uses_cold_water_and_retains_hardware_power():
    from revise_portugal_realism import profiles_for
    root=COMMERCIAL[-1];apps=pd.read_parquet(root/"catalogs/deferrables.parquet")
    for app in apps.loc[apps.equipment_use.eq("linen_laundry")].itertuples():
        p=profiles_for(root,app)
        assert p.external_hot_water_kwh.eq(0).all()
        assert p.load_kg.le(app.rated_capacity).all()
        for row in p.itertuples():
            values=np.array(json.loads(row.load_profile))
            assert values.max()*4<=app.nominal_power_kw+1e-8
            assert abs(values.sum()-row.total_energy_kwh)<1e-8
        full=p.loc[p.load_kg.eq(12)&p.wash_temperature_c.eq(60)]
        assert len(full)>0
        np.testing.assert_allclose(full.total_energy_kwh,1.68)
        assert full.duration_steps.ge(4).all()


def test_catering_warmup_and_all_feasible_batches_share_the_machine_rating():
    root=COMMERCIAL[-1];apps=pd.read_parquet(root/"catalogs/deferrables.parquet")
    app=next(apps.loc[apps.equipment_use.eq("catering_dishwashing_batch")].itertuples())
    support=pd.read_parquet(root/f"end_uses/{app.member_id}.parquet").professional_service_kwh.to_numpy()
    profiles=pd.read_parquet(root/f"deferrables/{app.deferrable_id}_profiles.parquet").set_index("profile_id")
    schedule=pd.read_parquet(root/f"deferrables/{app.deferrable_id}_schedule.parquet")
    for job in schedule.itertuples():
        values=np.array(json.loads(profiles.loc[job.profile_id,"load_profile"]))
        for start in range(job.earliest_start_time_step,job.latest_start_time_step+1):
            assert np.max(values+support[start:start+len(values)])*4<=app.nominal_power_kw+1e-7


def test_domestic_programmes_never_draw_from_heldout_houses():
    for root in sorted((ROOT/"data/datasets").glob("rec_pt_2023_*_v2")):
        apps=pd.read_parquet(root/"catalogs/deferrables.parquet")
        for a in apps.loc[apps.equipment_use.isin(["household_laundry","household_dishes","staff_kitchen_dishes"])].itertuples():
            p=pd.read_parquet(root/f"deferrables/{a.deferrable_id}_profiles.parquet")
            assert len(p)==48 and set(p.reference_house)=={3,5}
            assert p.duration_steps.nunique()>2


def test_professional_holding_load_is_not_a_shiftable_programme():
    root=COMMERCIAL[-1]
    apps=pd.read_parquet(root/"catalogs/deferrables.parquet")
    a=next(apps.loc[apps.equipment_use.eq("catering_dishwashing_batch")].itertuples())
    ledger=pd.read_parquet(root/f"end_uses/{a.member_id}.parquet")
    operations=pd.read_parquet(root/f"operations/{a.member_id}.parquet")
    pro=pd.read_parquet(root/f"deferrables/{a.deferrable_id}_profiles.parquet")
    assert ledger.professional_service_kwh.sum()>0
    assert ledger.professional_service_kwh.loc[~operations.is_open].eq(0).all()
    assert pro.duration_steps.between(1,3).all()
    assert pro.program.eq("professional_batch_15min_aggregate_prior").all()
    assert "reference_house" not in pro


@pytest.mark.parametrize("root",COMMERCIAL,ids=lambda p:p.name)
def test_explicit_supplemental_clones_use_current_donor_realization(root):
    catalogue=json.loads((root/"evaluation/campaign_2024/catalogue.json").read_text())
    for m in catalogue["supplemental_members"]:
        donor=pd.read_parquet(root/f"timeseries/{m['profile_template_member']}.parquet")
        for path in (root/"evaluation").glob("*.json"):
            schema=json.loads(path.read_text())
            if m["member_id"] not in schema["buildings"]:continue
            live,_,_=active_masks(schema,m["member_id"])
            actual=pd.read_parquet(root/schema["buildings"][m["member_id"]]["energy_simulation"])
            for col in ("non_shiftable_load","occupant_count"):
                np.testing.assert_allclose(actual[col],np.where(live,donor[col],0),rtol=1e-6,atol=1e-6)


def test_candidate_extraction_preserves_energy_and_does_not_misread_issues_flag():
    times=np.arange(0,4800,8,dtype=np.int64)
    power=np.zeros(len(times));power[(times>=1600)&(times<2800)]=1500.
    rows,_=extract_cycles(times,power,np.ones(len(times)),10,900)
    assert len(rows)==1
    assert rows[0]["energy_kwh"]==pytest.approx(.5)
    assert sum(json.loads(rows[0]["profile_15min_kwh"]))==pytest.approx(.5)
    assert rows[0]["issues_flagged_fraction"]==1.
    times[260:]+=300
    rows,_=extract_cycles(times,power,np.zeros(len(times)),10,900)
    assert rows==[],"Unobserved gaps inside a candidate must not become free energy"


def test_commercial_hourly_reference_does_not_fill_gaps_or_duplicate_hours():
    times=pd.date_range("2024-01-01",periods=72,freq="h")
    power=np.ones(72);power[30]=0
    assert len(hourly_days(times,power,measured=True))==2
    assert len(hourly_days(times.append(pd.DatetimeIndex([times[0]])),np.r_[power,1],measured=True))==1

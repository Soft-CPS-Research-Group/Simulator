"""Regression tests for failures exposed by measured-data/seed validation."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
sys.path.insert(0,str(ROOT/"scripts/audit"))
import revise_portugal_ev_mobility as ev
from portugal_activity import CONFIG
from portugal_stochastic_demand import bounded_energy_allocation
from portugal_empirical_validation import complete_days


@pytest.mark.parametrize("initial,cap,drive,power,kind",[(.985,10,9.9,1.,"frequent"),(.2,1,.21,.1,"visitor")])
def test_short_essential_charge_is_not_rejected_for_small_soc_gap(monkeypatch,initial,cap,drive,power,kind):
    monkeypatch.setattr(ev,"N",8)
    monkeypatch.setattr(ev,"N23",4)
    vehicle=SimpleNamespace(ev_id="probe",initial_soc=initial)
    battery=dict(capacity=cap,nominal_power=1.,efficiency=1.,capacity_power_curve=[[0,1],[1,1]])
    travel=np.zeros(8);travel[4]=drive
    candidate=dict(arrival_time_step=0,observed_end_time_step=1,charger_id="C",ev_id="probe",member_id="M",purpose="home")
    records,ledger=ev.plan_charging(vehicle,kind,[candidate],travel,np.zeros(8),battery,
        {"C":np.full(8,power)},pd.DataFrame({"day_kind":["weekday"]*8}),np.random.default_rng(5),set(),vary_connections=False)
    assert len(records)==1
    assert ledger.reference_soc.min()>=0
    assert records[0]["requested_energy_reference_kwh"]>0
    assert ledger.reference_charge_kwh.sum()<=power*.25+1e-10


def test_ev_random_stream_follows_master_seed_and_checkout_location():
    original=deepcopy(CONFIG)
    try:
        a=ev.rng_for(Path("/a/rec_pt_2023_core_15_v2"),"EV-A").random(6)
        b=ev.rng_for(Path("/other-premium/path/rec_pt_2023_core_15_v2"),"EV-A").random(6)
        np.testing.assert_array_equal(a,b)
        CONFIG["seed"]+=1
        assert not np.array_equal(a,ev.rng_for(Path("/a/rec_pt_2023_core_15_v2"),"EV-A").random(6))
    finally:
        CONFIG.clear();CONFIG.update(original)


def test_physical_envelope_preserves_energy_instead_of_clipping():
    values=np.array([9.,1.,1.,1.]);cap=np.array([3.,5.,5.,5.])
    actual=bounded_energy_allocation(values,cap,12.)
    assert actual.sum()==pytest.approx(12.)
    assert (actual<=cap).all()
    assert actual[0]==pytest.approx(3.)
    with pytest.raises(AssertionError):bounded_energy_allocation(values,cap,19.)


def test_measured_gaps_duplicates_and_dst_are_not_filled():
    times=pd.date_range("2024-01-01",periods=96*3,freq="15min")
    power=np.ones(len(times));power[100]=np.nan
    result=complete_days(times,power)
    assert list(result.index)==[pd.Timestamp("2024-01-01"),pd.Timestamp("2024-01-03")]
    result=complete_days(times.append(pd.DatetimeIndex([times[0]])),np.r_[power,1.])
    assert list(result.index)==[pd.Timestamp("2024-01-03")]
    spring=pd.date_range("2024-03-30",periods=96*3,freq="15min")
    assert pd.Timestamp("2024-03-31") not in complete_days(spring,np.ones(len(spring))).index


def test_nonstress_grid_moments_match_published_bt_totals_after_quantization():
    from scipy.stats import lognorm
    cfg=CONFIG["events"]
    target=cfg["empirical_moment_target"]
    # Probability mass of rounded/clamped 15-minute bins, including both tails.
    def minutes(median):
        cumulative=lognorm.cdf(np.arange(1.5,12,1)*15,s=cfg["outage_log_sigma"],scale=median)
        mass=np.diff(np.r_[0.,cumulative,1.])
        return np.dot(np.arange(1,13)*15,mass)
    local=cfg["local_outages_per_member_year"]
    common=cfg["common_outage_probability_per_year"]
    assert local+common==pytest.approx(target["saifi_per_customer"])
    duration=local*minutes(cfg["outage_median_minutes"])+common*minutes(cfg["common_outage_median_minutes"])
    assert duration==pytest.approx(target["saidi_minutes_per_customer"],abs=1e-8)
    assert cfg["controlled_grid_stress_hours_per_year"]>0

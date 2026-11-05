"""Cross-domain invariants and statistical-regression checks, not fit claims."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from portugal_activity import N, N23, rng_for, presence

FAMILIES = sorted((ROOT/"data/datasets").glob("rec_pt_2023_*_v2"))


def test_random_streams_preserve_nested_members_and_separate_years_and_processes():
    a = rng_for("micro_4", "Member_001", 2023, "activity").random(100)
    np.testing.assert_array_equal(a, rng_for("core_30", "Member_001", 2023, "activity").random(100))
    np.testing.assert_array_equal(a, rng_for("/tmp/premium-parent/core_30", "Member_001", 2023, "activity").random(100))
    for root, year, process in (("premium", 2023, "activity"), ("core", 2024, "activity"), ("core", 2023, "faults")):
        assert not np.array_equal(a, rng_for(root, "Member_001", year, process).random(100))


@pytest.mark.parametrize("root", FAMILIES, ids=lambda p:p.name)
def test_household_calendar_controls_presence_and_long_trip_energy(root):
    calendar = pd.read_parquet(root/"shared/calendar.parquet")
    daily = pd.read_parquet(root/"catalogs/household_activity.parquet")
    members = pd.read_parquet(root/"catalogs/members.parquet")
    clock = pd.DatetimeIndex(calendar.timestamp_utc)
    for m in members.loc[members.category.eq("residential")].itertuples():
        days = daily.loc[daily.member_id.eq(m.member_id)]
        assert len(days) == 731
        assert days.date.is_unique
        assert not (days.work_from_home & days.work_outside).any()
        expected = presence(clock, days, m.occupants)*m.occupants
        physical = pd.read_parquet(root/f"timeseries/{m.member_id}.parquet")
        np.testing.assert_allclose(physical.occupant_count, expected, atol=1e-6)
        for d in days.loc[days.absence_start.ge(0)].drop_duplicates("absence_start").itertuples():
            assert physical.occupant_count.iloc[d.absence_start:d.absence_end].eq(0).all()
    trips = pd.read_parquet(root/"catalogs/mobility_episodes.parquet")
    long = trips.loc[trips.trip_kind.isin(["weekend_away", "vacation_away", "holiday_bridge"])]
    assert long.destination_trip_count.gt(0).any()
    counts = long.assign(year=np.where(long.departure_time_step < N23, 2023, 2024)).groupby(["ev_id", "year"]).size()
    assert counts.nunique() > 1, "Long trips must not be fixed two weekends + one holiday + one vacation per EV/year"


@pytest.mark.parametrize("root", FAMILIES, ids=lambda p:p.name)
def test_cycles_respect_rating_windows_and_fixed_load_accounting(root):
    apps = pd.read_parquet(root/"catalogs/deferrables.parquet")
    fixed = {}
    for a in apps.itertuples():
        pro = pd.read_parquet(root/f"deferrables/{a.deferrable_id}_profiles.parquet").set_index("profile_id")
        schedule = pd.read_parquet(root/f"deferrables/{a.deferrable_id}_schedule.parquet")
        assert pro.total_energy_kwh.nunique() > 2
        assert set(schedule.profile_id) <= set(pro.index)
        assert (schedule.earliest_start_time_step <= schedule.latest_start_time_step).all()
        assert schedule.earliest_start_time_step.ge(0).all() and schedule.deadline_time_step.lt(N).all()
        for p in pro.itertuples():
            v = np.array(json.loads(p.load_profile))
            assert len(v) == p.duration_steps
            assert v.sum() == pytest.approx(p.total_energy_kwh)
            assert (v >= 0).all() and v.max()*4 <= a.nominal_power_kw+1e-8
        target = fixed.setdefault(a.member_id, np.zeros(N))
        for s in schedule.itertuples():
            p = pro.loc[s.profile_id]
            assert s.latest_start_time_step+p.duration_steps == s.deadline_time_step+1
            if not a.controlled:
                v = json.loads(p.load_profile)
                target[s.earliest_start_time_step:s.earliest_start_time_step+len(v)] += v
    for mid, actual in fixed.items():
        ledger = pd.read_parquet(root/f"end_uses/{mid}.parquet")
        np.testing.assert_allclose(actual, ledger.fixed_appliances_kwh, atol=1e-7)
        physical = pd.read_parquet(root/f"timeseries/{mid}.parquet")
        np.testing.assert_allclose(ledger.sum(axis=1), physical.non_shiftable_load, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("family,variant", [("core_30", "core_30_combined"), ("premium_100", "premium_100_allin")])
def test_event_years_are_not_replayed_and_true_sensor_dependencies_stay_joint(family, variant):
    root = ROOT/f"data/datasets/rec_pt_2023_{family}_v2"
    s = json.loads((root/f"evaluation/{variant}.json").read_text())
    f = pd.read_parquet(root/s["robustness"]["events_file"])
    temporal = f.loc[(f.end_time_step-f.start_time_step) < N23]
    old = set(temporal.loc[temporal.start_time_step.lt(N23), "start_time_step"]+34944)
    new = set(temporal.loc[temporal.start_time_step.ge(N23), "start_time_step"])
    assert len(old & new) < max(1, len(new)//10)
    assert {"local_fault", "controlled_stress"} <= set(f.generation_regime)
    for _, group in f.groupby("cause_id"):
        assert group.start_time_step.nunique() == 1 and group.end_time_step.nunique() == 1
    local = f.loc[f.generation_regime.eq("local_fault")]
    assert local.cause_id.nunique() > 2
    assert local.start_time_step.nunique() > 2
    for _, group in local.groupby("cause_id"):
        assert {"community_pv_power_kw", "community_pv_energy_kwh_step"} <= set(group.loc[group.target_type.eq("district"), "target_feature"])
    assert s["robustness"]["agent_observation_contract"]["mode"] == "known_faults"


def test_dr_impossible_control_remains_labelled_and_service_requests_vary():
    root = ROOT/"data/datasets/rec_pt_2023_premium_100_v2"
    f = pd.read_parquet(root/"evaluation/events/development/requests.parquet")
    stress = f.loc[f.stratum.eq("impossible_stress")]
    assert len(stress) == 2 and stress.target_power_kw.eq(1500).all()
    assert stress.generation_regime.eq("controlled_stress").all()
    routine = f.loc[f.stratum.eq("service_candidate")]
    assert routine.target_power_kw.nunique() > 3
    assert (routine.end_time_step-routine.start_time_step).nunique() > 1


def test_confirmation_noise_has_independent_exposure_and_error_stream_ids():
    root = ROOT/"data/datasets/rec_pt_2023_premium_100_v2"
    a = pd.read_parquet(root/"evaluation/events/development/measurement_noise.parquet")
    b = pd.read_parquet(root/"evaluation/events/confirmation/measurement_noise.parquet")
    assert not set(a.event_id) & set(b.event_id)
    assert len(set(a.start_time_step) & set(b.start_time_step)) < len(set(a.start_time_step))/4
    for family, variant in (("core_30", "core_30_robustness"), ("premium_100", "premium_100_allin")):
        root = ROOT/f"data/datasets/rec_pt_2023_{family}_v2"
        a = pd.read_parquet(root/f"evaluation/native_inputs/{variant}/robustness.parquet")
        b = pd.read_parquet(root/f"evaluation/native_inputs/{variant}/confirmation_robustness.parquet")
        noise_a = set(a.loc[a.generation_regime.eq("controlled_noise_stress"), "event_id"])
        noise_b = set(b.loc[b.generation_regime.eq("controlled_noise_stress"), "event_id"])
        assert noise_a and noise_b and not noise_a & noise_b


def test_shared_weather_uses_real_humidity_in_both_years_and_has_distinct_forecast_horizons():
    first = pd.read_parquet(FAMILIES[0]/"shared/weather.parquet")
    for root in FAMILIES[1:]:
        pd.testing.assert_frame_equal(first, pd.read_parquet(root/"shared/weather.parquet"))
    synthetic = np.clip(78-1.7*(first.outdoor_dry_bulb_temperature.iloc[:N23]-14), 30, 95)
    assert np.abs(first.outdoor_relative_humidity.iloc[:N23]-synthetic).mean() > 1.
    assert first.outdoor_relative_humidity.between(0, 100).all()
    errors = []
    for i, h in enumerate((4, 24, 96), 1):
        actual = first.outdoor_dry_bulb_temperature.to_numpy()[h:]
        predicted = first[f"outdoor_dry_bulb_temperature_predicted_{i}"].to_numpy()[:-h]
        errors.append(np.abs(actual-predicted).mean())
    assert errors[0] < errors[-1]


@pytest.mark.parametrize("family,variant", [("core_30", "core_30_combined"), ("premium_100", "premium_100_allin")])
def test_revised_fault_window_activates_exact_declared_channels(family, variant):
    from citylearn.citylearn import CityLearnEnv
    root = ROOT/f"data/datasets/rec_pt_2023_{family}_v2"
    schema = json.loads((root/f"evaluation/{variant}.json").read_text())
    events = pd.read_parquet(root/schema["robustness"]["events_file"])
    selected = events.loc[events.generation_regime.eq("local_fault")].iloc[0]
    t = int(selected.start_time_step)
    schema.update(root_directory=str(root), simulation_start_time_step=t,
                  simulation_end_time_step=t+1, episode_time_steps=2,
                  random_episode_split=False, rolling_episode_split=False)
    env = CityLearnEnv(schema, render_mode="none", offline=True, export_kpis_on_episode_end=False)
    try:
        obs, _ = env.reset(seed=0)
        expected = set(events.loc[(events.start_time_step <= t) & (events.end_time_step >= t), "event_id"])
        assert set(obs["meta"]["robustness"]["active_event_ids"]) == expected
        assert selected.event_id in expected
        assert obs["meta"]["robustness"]["enabled"]
    finally:
        env.close()

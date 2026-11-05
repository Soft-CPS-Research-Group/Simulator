"""Independent conservation, calendar, exposure and physics checks of EV inputs."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from revise_portugal_ev_mobility import N, N23, charge_trace, coast, local_step, required_energy, power_budgets, cycle_power_envelope, latest_charge_start

FAMILIES = sorted((ROOT/"data/datasets").glob("rec_pt_2023_*_v2"))


def test_appliance_envelope_bounds_every_allowed_start_and_is_zero_outside_window():
    schedule = pd.DataFrame([dict(profile_id="P", earliest_start_time_step=2, latest_start_time_step=5)])
    profile = np.array([.1, .5, .2])
    envelope = cycle_power_envelope({"P": profile}, schedule, steps=12)
    for start in range(2, 6):
        actual = np.zeros(12)
        actual[start:start+3] = profile*4
        assert (envelope >= actual).all()
    assert not envelope[:2].any() and not envelope[8:].any()
    assert envelope.max() == 2.


def test_coasting_does_not_hide_deficit_with_a_later_external_charge():
    path, before = coast(2., np.array([3., 0., 2.]), np.array([4., 20., 0.]), 10.)
    np.testing.assert_allclose(path, [3., 10., 8.])
    assert before[0] == -1.


def test_slack_is_latest_feasible_start_not_idle_time_after_early_charge():
    battery = dict(capacity=10., nominal_power=4., efficiency=1., capacity_power_curve=[[0., 1.], [1., 1.]])
    # Three idle hours after an early charge cannot be used to postpone it:
    # all available site headroom disappears after the first hour.
    assert latest_charge_start(0., .4, np.r_[np.full(4, 4.), np.zeros(12)], battery) == 0
    assert latest_charge_start(0., .4, np.full(16, 4.), battery) == 12


def test_charge_envelope_and_taper_require_more_than_constant_power_at_full_soc():
    battery = dict(capacity=60., nominal_power=7., efficiency=.96,
                   capacity_power_curve=[[0., 1.], [.9, 1.], [1., .25]])
    end, gains = charge_trace(0., 1., np.full(50, 7.), battery)
    assert end == pytest.approx(60.)
    assert np.count_nonzero(gains) > np.ceil(60/(7*.25*np.sqrt(.96)))
    required = required_energy(np.array([0., 0., 3.]), np.zeros(3),
        [dict(arrival_time_step=0, observed_end_time_step=2, eligible=True, charger_id="C")],
        {"C": np.full(3, 7.)}, battery)
    assert required[0] == 0.
    assert required[2] == 3.


def test_civil_time_respects_dst_and_does_not_assume_96_slots_per_day():
    clock = pd.date_range("2023-01-01", "2025-01-01", freq="15min", inclusive="left", tz="UTC")
    for date, expected in (("2023-03-26", 92), ("2023-10-29", 100), ("2024-03-31", 92), ("2024-10-27", 100)):
        day = pd.Timestamp(date, tz="Europe/Lisbon")
        assert local_step(clock, day, 24)-local_step(clock, day, 0) == expected


@pytest.mark.parametrize("root", FAMILIES, ids=lambda p: p.name)
def test_continuous_energy_budget_and_session_witness(root):
    vehicles = pd.read_parquet(root/"catalogs/electric_vehicles.parquet")
    sessions = pd.read_parquet(root/"catalogs/charging_sessions.parquet")
    annual = pd.read_parquet(root/"catalogs/ev_annual_energy.parquet")
    schema = json.loads((root/"schema.json").read_text())
    powers = power_budgets(root, schema)
    assert sessions.session_id.is_unique
    np.testing.assert_allclose(annual.original_trip_energy_kwh, annual.trip_energy_kwh, rtol=1e-12)
    for v in vehicles.itertuples():
        ledger = pd.read_parquet(root/f"mobility/{v.ev_id}.parquet")
        assert len(ledger) == N
        assert np.isfinite(ledger.to_numpy()).all()
        assert ledger.reference_soc.between(-1e-10, 1+1e-10).all()
        assert ledger.reference_end_soc.between(-1e-10, 1+1e-10).all()
        cap = v.battery_capacity_kwh
        previous = np.r_[v.initial_soc, ledger.reference_end_soc.to_numpy()[:-1]]*cap
        assert (previous-ledger.trip_energy_kwh >= -1e-7).all()
        np.testing.assert_allclose(ledger.reference_soc*cap,
            previous-ledger.trip_energy_kwh+ledger.external_charge_used_kwh, atol=1e-8)
        np.testing.assert_allclose(ledger.reference_end_soc*cap,
            ledger.reference_soc*cap+ledger.reference_charge_kwh, atol=1e-8)
        assert (ledger.external_charge_used_kwh <= ledger.external_charge_kwh+1e-9).all()
        for year, lo, hi in ((2023, 0, N23), (2024, N23, N)):
            baseline = annual.loc[annual.ev_id.eq(v.ev_id) & annual.year.eq(year), "original_trip_energy_kwh"].item()
            assert ledger.trip_energy_kwh.iloc[lo:hi].sum() == pytest.approx(baseline, rel=1e-12)
        connected = np.zeros(N, dtype=bool)
        battery = schema["electric_vehicles_def"][v.ev_id]["battery"]["attributes"]
        for s in sessions.loc[sessions.ev_id.eq(v.ev_id)].itertuples():
            a, b = s.arrival_time_step, s.observed_end_time_step
            assert not connected[a:b].any()
            connected[a:b] = True
            assert not ledger.trip_energy_kwh.iloc[a+1:b].any()
            assert not ledger.external_charge_kwh.iloc[a:b].any()
            assert ledger.reference_soc.iloc[a] == pytest.approx(s.arrival_soc_reference)
            assert ledger.reference_end_soc.iloc[b-1] == pytest.approx(s.target_soc)
            end, gains = charge_trace(s.arrival_soc_reference*cap, s.target_soc, powers[s.charger_id][a:b], battery)
            assert end >= s.target_soc*cap-1e-7
            np.testing.assert_allclose(gains, ledger.reference_charge_kwh.iloc[a:b], atol=1e-8)
        assert not ledger.reference_charge_kwh.loc[~connected].any()
    resident = sessions.loc[sessions.ev_id.isin(vehicles.loc[vehicles.vehicle_pool.eq("resident"), "ev_id"])]
    assert ((resident.arrival_soc_reference <= .05) & resident.target_soc.eq(1.)).any()
    assert resident.target_soc.eq(1.).any()
    assert ((resident.target_soc-resident.arrival_soc_reference) >= .9).any()


@pytest.mark.parametrize("root", FAMILIES, ids=lambda p: p.name)
def test_multiday_travel_and_charging_calendars(root):
    episodes = pd.read_parquet(root/"catalogs/mobility_episodes.parquet")
    calendar = pd.read_parquet(root/"shared/calendar.parquet")
    assert {"weekend_away", "holiday_bridge", "vacation_away"} <= set(episodes.trip_kind)
    assert (episodes.return_time_step-episodes.departure_time_step > 96).any()
    for eid, group in episodes.groupby("ev_id"):
        group = group.sort_values("departure_time_step")
        assert (group.departure_time_step.to_numpy()[1:] >= group.return_time_step.to_numpy()[:-1]).all(), eid
    sessions = pd.read_parquet(root/"catalogs/charging_sessions.parquet")
    assert sessions.day_kind.tolist() == calendar.day_kind.iloc[sessions.arrival_time_step].tolist()
    assert sessions.departure_time_step.isna().equals(sessions.right_censored)
    for path in (root/"operations").glob("*.parquet"):
        opened = pd.read_parquet(path).is_open.to_numpy()
        for s in sessions.loc[sessions.member_id.eq(path.stem)].itertuples():
            assert opened[s.arrival_time_step:s.observed_end_time_step].all(), s.session_id


@pytest.mark.parametrize("root", FAMILIES, ids=lambda p: p.name)
def test_native_charging_files_match_catalogue_and_preserve_unavailability(root):
    sessions = pd.read_parquet(root/"catalogs/charging_sessions.parquet").set_index("session_id")
    for path in root.rglob("CHG-*.parquet"):
        frame = pd.read_parquet(path)
        if len(frame) != N:
            continue
        absent = frame.electric_vehicle_charger_state.eq(0)
        assert frame.loc[absent, "electric_vehicle_id"].eq("").all()
        assert frame.loc[absent, "electric_vehicle_session_id"].eq("").all()
        connected = frame.loc[frame.electric_vehicle_charger_state.eq(1)]
        for sid, group in connected.groupby("electric_vehicle_session_id", sort=False):
            s = sessions.loc[sid]
            assert group.electric_vehicle_id.eq(s.ev_id).all()
            assert group.electric_vehicle_required_soc_departure.eq(s.target_soc).all()
            assert group.index.min() >= s.arrival_time_step
            assert group.index.max() < s.observed_end_time_step
            # Initial admission may expose an ongoing physical parking session.
            if group.index.min() == s.arrival_time_step:
                assert group.electric_vehicle_current_soc.iloc[0] == pytest.approx(s.arrival_soc_reference)


@pytest.mark.parametrize("root", FAMILIES, ids=lambda p: p.name)
def test_runtime_can_complete_a_near_empty_to_full_request(root):
    from citylearn.citylearn import CityLearnEnv
    sessions = pd.read_parquet(root/"catalogs/charging_sessions.parquet")
    eligible = sessions.loc[sessions.arrival_soc_reference.le(.05) & sessions.target_soc.eq(1.) & ~sessions.right_censored]
    s = eligible.sort_values("reference_charge_steps").iloc[0]
    schema = json.loads((root/"schema.json").read_text())
    for mid, b in schema["buildings"].items():
        b["include"] = mid == s.member_id
    a, b = int(s.arrival_time_step), int(s.observed_end_time_step)
    schema.update(root_directory=str(root), simulation_start_time_step=a,
                  simulation_end_time_step=b, episode_time_steps=b-a+1,
                  random_episode_split=False, rolling_episode_split=False,
                  physics_invariant_checks=True)
    env = CityLearnEnv(schema, render_mode="none", offline=True, export_kpis_on_episode_end=False)
    try:
        env.reset(seed=0)
        building = next(x for x in env.buildings if x.name == s.member_id)
        charger = next(x for x in building.electric_vehicle_chargers if x.charger_id == s.charger_id)
        ev = charger.connected_electric_vehicle
        assert ev is not None
        for _ in range(b-a):
            action = [dict(electrical_storage_action=0.,
                electric_vehicle_storage_actions={ch.charger_id: 1. for ch in member.electric_vehicle_chargers or []},
                deferrable_appliance_actions={}) for member in env.buildings]
            env._runtime_service._step_once(parsed_actions=action, collect_observations=False, collect_info=False)
        assert float(ev.battery.soc[b-a-1]) >= 1.-1e-5
    finally:
        env.close()

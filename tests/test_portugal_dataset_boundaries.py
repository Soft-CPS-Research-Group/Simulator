"""Integrity and executable end-of-observation contracts for Portugal data."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from repair_portugal_dataset_boundaries import censor_sessions, member_presence, TOTAL

FAMILIES = sorted((ROOT/'data/datasets').glob('rec_pt_2023_*_v2'))
SCHEMAS = [p for root in FAMILIES for p in sorted((root/'evaluation').glob('*.json'))]


def test_censoring_preserves_real_terminal_departures_and_is_idempotent():
    original = pd.DataFrame(dict(purpose=['home', 'visitor_afternoon', 'home'],
                                 departure_time_step=[20, 20, 19],
                                 target_soc=[.8, .9, .7]))
    actual = censor_sessions(original, end=20)
    assert actual.right_censored.tolist() == [True, False, False]
    assert actual.departure_time_step.isna().tolist() == [True, False, False]
    assert actual.observed_end_time_step.tolist() == [20, 20, 19]
    pd.testing.assert_series_equal(actual.target_soc, original.target_soc)
    pd.testing.assert_frame_equal(censor_sessions(actual, end=20), actual)


def test_presence_respects_entry_exit_and_readmission():
    schema = dict(buildings={'M': {'include': False}}, topology_events=[
        dict(target_member_id='M', time_step=t, operation=op)
        for t, op in [(2, 'add_member'), (4, 'remove_member'), (6, 'add_member')]])
    assert member_presence(schema, 'M', 8).tolist() == [False, False, True, True, False, False, True, True]


@pytest.mark.parametrize('root', FAMILIES, ids=lambda p: p.name)
def test_catalogues_and_all_complete_sessions_are_individually_feasible(root):
    frame = pd.read_parquet(root/'catalogs/charging_sessions.parquet')
    assert frame.session_id.is_unique
    assert frame.departure_time_step.isna().equals(frame.right_censored)
    assert frame.loc[frame.right_censored, 'observed_end_time_step'].eq(TOTAL).all()
    assert frame.loc[frame.right_censored, 'purpose'].eq('home').all()
    assert (frame.arrival_time_step < frame.observed_end_time_step).all()
    for field in ('ev_id', 'charger_id'):
        for _, group in frame.sort_values('arrival_time_step').groupby(field):
            assert (group.arrival_time_step.to_numpy()[1:] >= group.observed_end_time_step.to_numpy()[:-1]).all()
    schema = json.loads((root/'schema.json').read_text())
    chargers = {cid: c['attributes'] for b in schema['buildings'].values() for cid, c in b.get('chargers', {}).items()}
    failures = []
    for row in frame.loc[~frame.right_censored].itertuples():
        battery = schema['electric_vehicles_def'][row.ev_id]['battery']['attributes']
        charger = chargers[row.charger_id]
        capacity, soc = battery['capacity'], row.arrival_soc_reference
        curve = np.asarray(battery['capacity_power_curve'])
        for _ in range(int(row.departure_time_step) - int(row.arrival_time_step)):
            if soc >= row.target_soc - 1e-9:
                break
            power = min(charger['max_charging_power'] * charger['efficiency'],
                        battery['nominal_power'] * np.interp(soc, curve[:, 0], curve[:, 1]))
            soc = min(1., soc + power * .25 * np.sqrt(battery['efficiency']) / capacity)
        if (row.target_soc - soc) * capacity > 1e-5:
            failures.append(row.session_id)
    assert not failures, failures
    for line in (root/'file_checksums.sha256').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == expected, relative


@pytest.mark.parametrize('path', SCHEMAS, ids=lambda p: p.stem)
def test_native_outages_and_unknown_deadlines(path):
    root = path.parent.parent
    schema = json.loads(path.read_text())
    sessions = pd.read_parquet(root/'catalogs/charging_sessions.parquet')
    partial = set(sessions.loc[sessions.right_censored, 'session_id'])
    for member, building in schema['buildings'].items():
        frame = pd.read_parquet(root/building['energy_simulation'])
        assert not (frame.power_outage.gt(0) & ~member_presence(schema, member, len(frame))).any()
        for charger in building.get('chargers', {}).values():
            frame = pd.read_parquet(root/charger['charger_simulation'])
            connected = frame.electric_vehicle_charger_state.eq(1)
            mask = frame.electric_vehicle_session_id.isin(partial) & connected
            assert frame.loc[mask, 'electric_vehicle_departure_time'].isna().all()
            assert frame.loc[connected & ~mask, 'electric_vehicle_departure_time'].ge(1).all()


@pytest.mark.parametrize('path', SCHEMAS, ids=lambda p: p.stem)
def test_year_end_runs_without_false_ev_departures(path):
    from citylearn.citylearn import CityLearnEnv
    from citylearn.agents.baseline import NormalPolicy
    from citylearn.internal.kpi import CityLearnKPIService
    root = path.parent.parent
    schema = json.loads(path.read_text())
    schema.update(root_directory=str(root), simulation_start_time_step=TOTAL-12,
                  simulation_end_time_step=TOTAL-1, episode_time_steps=12,
                  random_episode_split=False, rolling_episode_split=False,
                  physics_invariant_checks=True)
    env = CityLearnEnv(schema, render_mode='none', offline=True, random_seed=0,
                       export_kpis_on_episode_end=False)
    try:
        obs, _ = env.reset(seed=0)
        policy = NormalPolicy(env)
        while not env.terminated:
            obs, reward, terminated, truncated, _ = env.step(policy.predict(obs, deterministic=True))
            assert np.isfinite(reward).all()
            if terminated or truncated:
                break
        service = CityLearnKPIService(env)
        for building in env.buildings:
            # No physical departures in the last three hours of these fixtures;
            # parking remains connected at the observation boundary.
            for charger in building.electric_vehicle_chargers or []:
                sim = charger.charger_simulation
                if sim.electric_vehicle_charger_state[-1] == 1:
                    assert sim.electric_vehicle_departure_time[-1] == -1
            if building.name in ('Member_006', 'Member_024') and 'premium' in path.stem:
                assert service._compute_ev_metrics(building)['departures_total'] == 0
                reward_obs = building.observations(include_all=True)
                # The public reward view must not turn unknown time into zero.
                evs = reward_obs.get('electric_vehicles_chargers_dict', {})
                assert evs
                for data in evs.values():
                    if data['connected']:
                        assert data['hours_until_departure'] == -1
    finally:
        env.close()


def test_unknown_ev_departure_has_no_deadline_reward_or_penalty():
    from citylearn.reward_function import Electric_Vehicles_Reward_Function
    reward = Electric_Vehicles_Reward_Function({'central_agent': False, 'buildings': []})
    charger = dict(connected=True, previous_battery_soc=.2, battery_soc=.2,
                   battery_capacity=60., min_capacity=0., last_charged_kwh=0.,
                   required_soc=.8, hours_until_departure=-1.,
                   max_charging_power=3.68, max_discharging_power=0.)
    obs = dict(net_electricity_consumption=0., electric_vehicles_chargers_dict={'C': charger})
    assert reward.calculate_ev_penalty(obs, 0.) == 0.
    charger['hours_until_departure'] = 1.
    assert reward.calculate_ev_penalty(obs, 0.) != 0.

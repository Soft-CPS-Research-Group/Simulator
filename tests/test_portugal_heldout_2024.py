"""Reserved 2024 changes must execute and matched references must stay matched."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'scripts/audit'))
from prepare_portugal_heldout_2024 import N23, read_json, resource_state
from audit_portugal_heldout_2024 import static_audit, verify_pair, runtime_window
import audit_portugal_heldout_2024 as audit

CAMPAIGN = read_json(ROOT/'results/portugal_rec_native/heldout_2024/campaign.json')
PAIRS = CAMPAIGN['pairs']
EVENTS = CAMPAIGN['events']


def test_heldout_2024_static_inputs_and_service_cohorts():
    report = static_audit(ROOT)
    assert report['declared_2024_events'] == 18
    assert len(report['pairs']) == 6
    assert len(report['preparation']) == 9
    assert all(c['affected_inflight_sessions'] == 0 for c in report['cohorts'])
    assert all(c['excluded_by_2024_departure'] > 0 for c in report['cohorts'])


@pytest.mark.parametrize('pair', PAIRS, ids=lambda p:p['reference'])
def test_only_declared_factor_changes(pair):
    assert verify_pair(ROOT, pair)['status'] == 'pass'


@pytest.mark.parametrize('pair', PAIRS, ids=lambda p:p['reference'])
def test_pair_verifier_rejects_an_unmatched_random_seed(pair, monkeypatch):
    read = audit.read_json
    reference = ROOT/pair['reference_schema']
    def changed(path):
        result = read(path)
        if Path(path) == reference:
            result['random_seed'] += 1
        return result
    monkeypatch.setattr(audit, 'read_json', changed)
    with pytest.raises(AssertionError):
        verify_pair(ROOT, pair)


@pytest.mark.parametrize('event', EVENTS, ids=lambda e:e['variant']+'-'+e['id'])
def test_new_2024_event_changes_running_resources_and_interfaces(event):
    family = 'core_30' if event['variant'].startswith('core') else 'premium_100'
    path = ROOT/f'data/datasets/rec_pt_2023_{family}_v2/evaluation/{event["variant"]}.json'
    t = event['time_step']
    result = runtime_window(path, t-4, t+4, event['id'])
    assert len(result['executed_events']) == 1
    assert result['executed_events'][0]['applied']


@pytest.mark.parametrize('pair', PAIRS, ids=lambda p:p['reference'])
def test_references_execute_at_first_heldout_change(pair):
    t = next(e['time_step'] for e in EVENTS if e['variant']==pair['scenario'])
    result = runtime_window(ROOT/pair['reference_schema'], t-2, t+2, 'first_heldout_change')
    assert len(result['executed_events']) == (0 if pair['factor']=='topology_2024' else 1)


def test_2024_is_not_a_reset_or_replay_of_2023():
    path = ROOT/'data/datasets/rec_pt_2023_core_30_v2/evaluation/core_30_topology.json'
    schema = read_json(path)
    previous = deepcopy(schema)
    previous['topology_events'] = [e for e in previous['topology_events'] if e['time_step']<N23]
    assert resource_state(schema, N23) == resource_state(previous, N23)
    assert set(resource_state(schema, N23)['ids']['building']).isdisjoint({'Member_001', 'Member_008', 'Member_011', 'Member_031'})
    assert schema['simulation_start_time_step'] == 0
    assert schema['simulation_end_time_step'] == 70175


@pytest.mark.parametrize('pair', [p for p in PAIRS if p['factor']=='topology_2024'], ids=lambda p:p['scenario'])
def test_frozen_counterfactual_has_identical_observations_and_actions_before_first_change(pair):
    from citylearn.citylearn import CityLearnEnv
    from citylearn.agents.baseline import NormalPolicy
    t = next(e['time_step'] for e in EVENTS if e['variant']==pair['scenario'])
    envs = []
    try:
        observations = []
        for key in ('scenario_schema', 'reference_schema'):
            path = ROOT/pair[key]
            schema = read_json(path)
            schema.update(root_directory=str((path.parent/schema['root_directory']).resolve()),
                simulation_start_time_step=t-4, simulation_end_time_step=t+4, episode_time_steps=9)
            env = CityLearnEnv(schema, render_mode='none', offline=True, export_kpis_on_episode_end=False)
            envs.append(env)
            observations.append(env.reset(seed=schema['random_seed'])[0])
        policies = [NormalPolicy(env) for env in envs]
        for tick in range(4):
            for key in ('tables', 'actions'):
                assert envs[0].entity_specs[key] == envs[1].entity_specs[key]
            assert envs[0].entity_specs['topology']['active_ids'] == envs[1].entity_specs['topology']['active_ids']
            for kind in ('tables', 'edges'):
                for name in observations[0][kind]:
                    np.testing.assert_array_equal(observations[0][kind][name], observations[1][kind][name])
            for name, space in envs[0].action_space['tables'].items():
                np.testing.assert_array_equal(space.low, envs[1].action_space['tables'][name].low)
                np.testing.assert_array_equal(space.high, envs[1].action_space['tables'][name].high)
            if tick < 3:
                observations = [env.step(policy.predict(obs, deterministic=True))[0]
                                for env, policy, obs in zip(envs, policies, observations)]
    finally:
        for env in envs:
            env.close()


@pytest.mark.parametrize('family,variant', [('core_30', 'core_30_topology'), ('core_30', 'core_30_combined'), ('premium_100', 'premium_100_allin')])
def test_preparation_view_runs_without_heldout_member_metadata(family, variant):
    from citylearn.citylearn import CityLearnEnv
    from citylearn.agents.baseline import NormalPolicy
    path = ROOT/f'data/datasets/rec_pt_2023_{family}_v2/evaluation/preparation_2023/{variant}.json'
    schema = read_json(path)
    assert schema['simulation_end_time_step'] == N23-1
    schema.update(root_directory=str((path.parent/schema['root_directory']).resolve()),
        simulation_start_time_step=N23-8, episode_time_steps=8)
    env = CityLearnEnv(schema, render_mode='none', offline=True, export_kpis_on_episode_end=False)
    try:
        obs, _ = env.reset(seed=schema['random_seed'])
        newcomer = 'Member_031' if family=='core_30' else 'Member_101'
        assert newcomer not in env.entity_specs['topology']['member_lifecycle']
        assert newcomer not in env.schema['buildings']
        assert all(e['time_step'] < N23 for e in env.schema['topology_events'])
        assert env.time_steps == 8
        policy = NormalPolicy(env)
        while not env.terminated:
            obs, reward, _, _, _ = env.step(policy.predict(obs, deterministic=True))
            assert np.isfinite(reward).all()
        assert env.episode_tracker.episode_end_time_step == N23-1
    finally:
        env.close()

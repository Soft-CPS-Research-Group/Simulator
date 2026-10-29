"""Economic and forecast contracts for the Portuguese OMIE-only datasets."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from apply_portugal_omie_pricing import build_prices, load_market, MARKET, PRICING, UTC


def test_wholesale_units_and_negative_floor():
    clock = pd.date_range('2024-01-01', periods=400, freq='15min', tz='UTC')
    market = pd.Series(100.0, index=clock)
    market.iloc[:3] = [-20.0, 0.0, 100.0]
    frame, _ = build_prices(market, clock[:200])
    assert frame.electricity_pricing.iloc[:3].tolist() == pytest.approx([0, 0, .1])
    assert (frame.to_numpy() >= 0).all()


@pytest.mark.parametrize('day,publication_hour', [('2024-01-15', 13), ('2024-07-15', 12)])
def test_unknown_day_ahead_prices_do_not_leak(day, publication_hour):
    clock = pd.date_range(day, periods=288, freq='15min', tz='UTC')
    market = pd.Series(100., index=clock)
    baseline, _ = build_prices(market, clock[:96])
    market.iloc[96:] = -20.
    changed, _ = build_prices(market, clock[:96])
    cut = publication_hour*4
    # +24h is still prior-day persistence before publication, including negatives.
    np.testing.assert_array_equal(changed.electricity_pricing_predicted_3[:cut],
                                  baseline.electricity_pricing_predicted_3[:cut])
    market_midnight = (publication_hour + 10)*4
    assert changed.electricity_pricing_predicted_3[cut:market_midnight].eq(0).all()
    assert baseline.electricity_pricing_predicted_3[cut:market_midnight].gt(0).all()
    # Once Madrid enters the next day, the +24h target needs a new auction.
    np.testing.assert_array_equal(changed.electricity_pricing_predicted_3[market_midnight:],
                                  baseline.electricity_pricing_predicted_3[market_midnight:])


def test_current_prices_reconcile_to_independent_wholesale_components():
    raw = pd.concat([pd.read_parquet(ROOT/f'data/portugal_reference_{y}/price_components_btn_bi.parquet')
                     for y in (2023, 2024)], ignore_index=True)
    expected = raw.omie_eur_kwh.clip(lower=0).astype('float32')
    for root in sorted((ROOT/'data/datasets').glob('rec_pt_2023_*_v2')):
        actual = pd.read_parquet(root/PRICING)
        contract = json.loads((root/'settlement_contract.json').read_text())
        assert contract['dataset_market_enabled'] is True
        assert contract['active_pricing_file'] == PRICING
        assert contract['local_price_ratio_to_grid_import'] == .8
        np.testing.assert_array_equal(actual.electricity_pricing, expected)
        assert actual.shape == (70176, 4)
        assert np.isfinite(actual.to_numpy()).all() and (actual.to_numpy() >= 0).all()
        for path in [root/'schema.json', *sorted((root/'evaluation').glob('*.json'))]:
            schema = json.loads(path.read_text())
            assert schema['community_market'] == MARKET
            assert all(b['pricing'] == PRICING for b in schema['buildings'].values())


def test_forecasts_across_dst_new_year_and_last_horizon():
    raw, _ = load_market()
    frame = pd.read_parquet(ROOT/'data/datasets/rec_pt_2023_micro_4_v2'/PRICING)
    issues = [pd.Timestamp(s, tz='UTC') for s in
              ['2023-01-01 00:00', '2023-03-26 00:45', '2023-10-29 00:45',
               '2023-12-31 12:45', '2023-12-31 13:00', '2024-03-31 00:45',
               '2024-10-27 00:45', '2024-12-31 12:45', '2024-12-31 13:00', '2024-12-31 23:45']]
    for issue in issues:
        for k, hours in enumerate((1, 6, 24), 1):
            target = issue + pd.Timedelta(hours=hours)
            day = target.tz_convert('Europe/Madrid').date()
            publication = (pd.Timestamp(day)-pd.Timedelta(days=1)+pd.Timedelta(hours=14)).tz_localize('Europe/Madrid')
            source = target if issue >= publication else max(target-pd.Timedelta(days=1), raw.index[0])
            expected = max(float(raw.loc[source])/1000, 0)
            assert frame.iloc[UTC.get_loc(issue)][f'electricity_pricing_predicted_{k}'] == pytest.approx(expected, abs=1e-8)
            assert source <= issue or issue >= publication


def test_missing_forecast_support_fails_instead_of_truncating_horizon():
    market = pd.Series(100., index=pd.date_range('2024-01-01', periods=96, freq='15min', tz='UTC'))
    with pytest.raises(ValueError, match='Missing OMIE'):
        build_prices(market, market.index)


def test_real_negative_omie_window_settles_at_zero_without_nan_rewards():
    from citylearn.citylearn import CityLearnEnv
    from citylearn.agents.baseline import NormalPolicy
    root = ROOT/'data/datasets/rec_pt_2023_core_15_v2'
    schema = json.loads((root/'schema.json').read_text())
    # 16 June 2024, 09:00 UTC: archived OMIE is -2 EUR/MWh.
    schema.update(root_directory=str(root), simulation_start_time_step=51108,
                  simulation_end_time_step=51111, episode_time_steps=4,
                  random_episode_split=False, rolling_episode_split=False,
                  physics_invariant_checks=True)
    env = CityLearnEnv(schema, render_mode='none', offline=True, random_seed=0,
                       export_kpis_on_episode_end=False)
    try:
        obs, _ = env.reset(seed=0)
        policy = NormalPolicy(env)
        local_energy = 0.
        while not env.terminated:
            obs, reward, terminated, truncated, _ = env.step(policy.predict(obs, deterministic=True))
            assert np.isfinite(reward).all()
            rows = env._last_community_market_settlement
            assert rows
            local_energy += sum(r['local_import_kwh'] for r in rows)
            for row in rows:
                assert row['grid_import_price'] == 0
                assert row['local_buy_price'] == row['local_sell_price'] == 0
                assert row['settled_cost_eur'] == 0
            if terminated or truncated:
                break
        assert local_energy > 0, 'Exercise real local energy transfers at the zero price'
    finally:
        env.close()

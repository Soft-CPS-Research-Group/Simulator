"""Independent boundary and information-availability checks for the 2023 inputs."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_portugal_reference import (
    civil_day,
    tariff_components,
    price_forecasts,
    ROOT,
)
from portugal_site_calendar import service_hours


def test_both_source_timezones_preserve_dst_hours():
    for zone in ("Europe/Madrid", "Europe/Lisbon"):
        assert len(civil_day("2023-03-26", zone, "15min")) == 92
        assert len(civil_day("2023-10-29", zone, "15min")) == 100
    h = pd.read_parquet(ROOT / "omie_2023_hourly.parquet")
    assert h.iloc[0].market_period == 2
    assert h.iloc[-1].market_date == "2024-01-01"
    assert h.timestamp_utc.is_unique


def test_access_change_at_lisbon_midnight_and_bte_peak_integral():
    idx = pd.date_range("2023-06-30T22:45Z", periods=2, freq="15min")
    market = pd.DataFrame({"omie_eur_kwh": 0.1, "mibel_eur_kwh": 0}, index=idx)
    c = tariff_components(market, "btn_simple")
    assert c.access_energy_eur_kwh.tolist() == [-0.0958, -0.0121]
    for day in ("2023-01-02", "2023-07-03", "2023-03-26", "2023-10-29"):
        idx = civil_day(day, "Europe/Lisbon", "15min").tz_convert("UTC")
        c = tariff_components(
            pd.DataFrame({"omie_eur_kwh": 0.0, "mibel_eur_kwh": 0.0}, index=idx),
            "bte_daily",
        )
        assert c.access_peak_power_eur_kwh.sum() * 0.25 == pytest.approx(0.4854)


def test_unpublished_omie_and_definitive_mibel_do_not_leak():
    idx = pd.date_range("2023-01-01", periods=288, freq="15min", tz="UTC")
    m = pd.DataFrame({"omie_eur_kwh": 0.1, "mibel_eur_kwh": 0.02}, index=idx)
    a = price_forecasts(tariff_components(m, "btn_simple"))
    m.iloc[96:, m.columns.get_loc("omie_eur_kwh")] = 9
    m.iloc[1:, m.columns.get_loc("mibel_eur_kwh")] = 8
    b = price_forecasts(tariff_components(m, "btn_simple"))
    assert (
        a.iloc[0].electricity_pricing_predicted_3
        == b.iloc[0].electricity_pricing_predicted_3
    )
    # At 13:00 Lisbon, tomorrow's OMIE is allowed, definitive future MI is not.
    m = pd.DataFrame({"omie_eur_kwh": 0.1, "mibel_eur_kwh": 0.02}, index=idx)
    m.iloc[96:, 0] = 9
    d = price_forecasts(tariff_components(m, "btn_simple"))
    assert (
        d.iloc[52].electricity_pricing_predicted_3
        > a.iloc[52].electricity_pricing_predicted_3
    )


def test_carbon_calibration_and_mix_balance():
    c = pd.read_parquet(ROOT / "carbon_components_2023.parquet")
    assert len(c) == 35040 and c.timestamp_utc.is_unique
    assert np.average(
        c.pt_production_kg_co2e_kwh, weights=c.pt_generation_mw
    ) == pytest.approx(0.092)
    assert c.carbon_intensity.between(0, 1).all()
    assert c.balance_residual_mw.abs().quantile(0.99) < 50

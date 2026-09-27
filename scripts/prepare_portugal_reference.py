"""Rebuild aligned 2023 prices/carbon offline from archived public inputs.

All energy prices are variable, pre-VAT EUR/kWh. This is a reproducible
benchmark tariff, not a reconstruction of a named customer's invoice.
"""

from pathlib import Path
import gzip
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data/portugal_reference_2023"
UTC = pd.date_range(
    "2023-01-01", "2024-01-01", freq="15min", inclusive="left", tz="UTC"
)
TARIFFS = ("btn_simple", "btn_bi", "btn_tri", "btn_large_tri", "bte_daily")


def civil_day(day, zone, freq):
    start = pd.Timestamp(day).tz_localize(zone)
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(zone)
    return pd.date_range(start, end, freq=freq, inclusive="left")


def align_market(root=ROOT):
    # Preserve full raw market-day records, including the hour outside UTC 2023.
    raw = root / "omie_2023_market_periods.parquet"
    if not raw.exists():
        pd.read_parquet(root / "omie_2023_hourly.parquet").drop(
            columns="timestamp_utc"
        ).to_parquet(raw, index=False)
    records = pd.read_parquet(raw)
    blocks = []
    for day, g in records.groupby("market_date", sort=True):
        g = g.sort_values("market_period").copy()
        idx = civil_day(day, "Europe/Madrid", "h")
        assert list(g.market_period) == list(range(1, len(idx) + 1))
        g["timestamp_utc"] = idx.tz_convert("UTC")
        blocks.append(g)
    fields = (root / "omie_market_20240101.1").read_text().splitlines()[1].split(";")
    blocks.append(
        pd.DataFrame(
            [
                dict(
                    timestamp_utc=pd.Timestamp("2023-12-31T23:00Z"),
                    market_date="2024-01-01",
                    market_period=1,
                    price_spain_eur_per_mwh=float(fields[4]),
                    price_portugal_eur_per_mwh=float(fields[5]),
                )
            ]
        )
    )
    hourly = (
        pd.concat(blocks)
        .set_index("timestamp_utc")
        .sort_index()
        .loc["2023-01-01":"2023-12-31"]
    )
    assert hourly.index.equals(UTC[::4])
    hourly.reset_index().to_parquet(root / "omie_2023_hourly.parquet", index=False)
    adjustments = []
    for p in sorted((root / "raw_2023").glob("omie_adjustment_*.json.gz")):
        day = p.name.removeprefix("omie_adjustment_").removesuffix(".json.gz")
        j = json.loads(gzip.decompress(p.read_bytes()))["data"]
        idx = civil_day(day, "Europe/Madrid", "h").tz_convert("UTC")
        values = j["series"][0]["data"]
        assert len(idx) == len(values), day
        adjustments.append(pd.Series(values, index=idx, dtype=float))
    # Mechanism expired at the end of the 2023 Spanish market calendar.
    adjustments.append(pd.Series([0.0], index=[pd.Timestamp("2023-12-31T23:00Z")]))
    adjustment = pd.concat(adjustments).reindex(UTC[::4])
    assert adjustment.notna().all()
    out = pd.DataFrame(index=UTC)
    out["omie_eur_kwh"] = (
        hourly.price_portugal_eur_per_mwh.reindex(UTC, method="ffill") / 1000
    )
    out["mibel_eur_kwh"] = adjustment.reindex(UTC, method="ffill") / 1000
    return out


def tariff_components(market, tariff):
    local = market.index.tz_convert("Europe/Lisbon")
    h = local.hour + local.minute / 60
    summer = np.array([x.dst().total_seconds() > 0 for x in local])
    peak = np.where(
        summer,
        ((h >= 10.5) & (h < 13)) | ((h >= 19.5) & (h < 21)),
        ((h >= 9) & (h < 10.5)) | ((h >= 18) & (h < 20.5)),
    )
    off = (h < 8) | (h >= 22)
    superoff = (h >= 2) & (h < 6)
    second = local >= pd.Timestamp("2023-07-01", tz="Europe/Lisbon")
    if tariff == "btn_simple":
        first, last = np.full(len(h), -0.0958), np.full(len(h), -0.0121)
    elif tariff == "btn_bi":
        first, last = np.where(off, -0.1185, -0.0842), np.where(off, -0.0349, -0.0005)
    elif tariff == "btn_tri":
        first = np.select([off, peak], [-0.1185, -0.0018], default=-0.1069)
        last = np.select([off, peak], [-0.0349, 0.0818], default=-0.0232)
    elif tariff == "btn_large_tri":
        first = np.select([off, peak], [-0.0631, 0.0601], default=-0.0449)
        last = np.select([off, peak], [-0.0224, 0.1008], default=-0.0042)
    elif tariff == "bte_daily":
        first = np.select(
            [superoff, off, peak], [-0.0538, -0.0517, -0.0479], default=-0.0492
        )
        last = np.select(
            [superoff, off, peak], [-0.0195, -0.0174, -0.0136], default=-0.0149
        )
    else:
        raise ValueError(tariff)
    out = market.copy()
    out["access_energy_eur_kwh"] = np.where(second, last, first)
    # Daily cycle has exactly 4 peak hours/day. Pp=Ep/Hp, so EUR/kW/day / 4
    # is an exact marginal energy equivalent, also for monthly billing.
    out["access_peak_power_eur_kwh"] = peak.astype(float) * (
        0.4854 / 4 if tariff == "bte_daily" else 0
    )
    out["cgs_eur_kwh"] = 0.004
    out["supplier_margin_eur_kwh"] = 0.010
    out["loss_multiplier"] = 1.15
    out["iec_eur_kwh"] = 0.001
    out["electricity_pricing"] = (
        (out.omie_eur_kwh + out.mibel_eur_kwh + 0.004 + 0.010) * 1.15
        + out.access_energy_eur_kwh
        + out.access_peak_power_eur_kwh
        + 0.001
    )
    out["period"] = np.select(
        [superoff & (tariff == "bte_daily"), off, peak],
        ["super_vazio", "vazio", "ponta"],
        default="cheias",
    )
    return out


def price_forecasts(components):
    idx = components.index
    madrid = idx.tz_convert("Europe/Madrid")
    n = len(idx)
    output = pd.DataFrame(
        {"electricity_pricing": components.electricity_pricing.to_numpy()}
    )
    for k, horizon in enumerate((4, 24, 96), 1):
        target = np.minimum(np.arange(n) + horizon, n - 1)
        day = madrid[target].tz_localize(None).normalize()
        # Conservative day-ahead availability at 14:00 Madrid (13:00 Lisbon).
        publication = (
            (day - pd.Timedelta(days=1) + pd.Timedelta(hours=14))
            .tz_localize("Europe/Madrid")
            .tz_convert("UTC")
        )
        past = np.maximum(target - 96, 0)
        past = np.minimum(past, np.arange(n))
        omie = np.where(
            idx >= publication,
            components.omie_eur_kwh.to_numpy()[target],
            components.omie_eur_kwh.to_numpy()[past],
        )
        # Definitive adjustment is not known day-ahead: deliberately causal proxy.
        mi = components.mibel_eur_kwh.to_numpy()[np.maximum(np.arange(n) - 96, 0)]
        output[f"electricity_pricing_predicted_{k}"] = (
            (omie + mi + 0.014) * 1.15
            + components.access_energy_eur_kwh.to_numpy()[target]
            + components.access_peak_power_eur_kwh.to_numpy()[target]
            + 0.001
        )
    return output.astype("float32")


def carbon_reference(root=ROOT):
    essential = [
        "Consumption",
        "Pumping",
        "Hydro",
        "Solar",
        "Import",
        "Export",
        "Wind",
        "Natural Gas",
        "Other Thermal",
        "Biomass",
        "Coal",
    ]
    blocks = []
    for p in sorted((root / "raw_2023").glob("ren_*.json.gz")):
        day = p.name[4:-8]
        j = json.loads(gzip.decompress(p.read_bytes()))["data"]
        idx = civil_day(day, "Europe/Lisbon", "15min")
        assert list(idx.strftime("%H:%M")) == j["xAxis"]["categories"], day
        d = {x["name"]: x["data"] for x in j["series"]}
        b = pd.DataFrame(
            {key: d[key] for key in essential}, index=idx.tz_convert("UTC"), dtype=float
        )
        assert b.notna().all().all(), day
        b["Wave"] = pd.Series(
            d.get("Wave", [0] * len(idx)), index=b.index, dtype=float
        ).fillna(0)
        blocks.append(b)
    out = pd.concat(blocks)
    assert out.index.equals(UTC)
    assert (out[essential] >= 0).all().all()
    es = []
    for year in (2023, 2024):
        j = json.loads((root / f"ree_emissions_{year}_daily.json").read_text())["data"]
        values = next(x for x in j["included"] if x["type"] == "tCO2 eq./MWh")[
            "attributes"
        ]["values"]
        es.extend(values)
    es = pd.Series(
        [v["value"] for v in es],
        index=pd.to_datetime([v["datetime"] for v in es], utc=True),
    )
    out["es_daily_kg_co2e_kwh"] = es.reindex(UTC, method="ffill")
    generation = out[
        [
            "Hydro",
            "Solar",
            "Wind",
            "Natural Gas",
            "Other Thermal",
            "Biomass",
            "Coal",
            "Wave",
        ]
    ].sum(axis=1)
    fossil = out[["Natural Gas", "Other Thermal", "Coal"]].sum(axis=1)
    factor = 0.092 * generation.sum() / fossil.sum()
    out["pt_generation_mw"] = generation
    out["pt_production_kg_co2e_kwh"] = factor * fossil / generation
    out["pt_import_blend_kg_co2e_kwh"] = (
        factor * fossil + out.Import * out.es_daily_kg_co2e_kwh
    ) / (generation + out.Import)
    # Match the reference delivery-loss convention; never describe this as metered carbon.
    out["carbon_intensity"] = out.pt_import_blend_kg_co2e_kwh * 1.15
    out["balance_residual_mw"] = (
        generation + out.Import - out.Export - out.Consumption - out.Pumping
    )
    out.reset_index(names="timestamp_utc").to_parquet(
        root / "carbon_components_2023.parquet", index=False
    )
    out[["carbon_intensity"]].astype("float32").reset_index(drop=True).to_parquet(
        root / "carbon_intensity.parquet", index=False
    )
    return {
        "effective_thermal_factor_kg_kwh": factor,
        "apa_mainland_generation_weighted_kg_kwh": float(
            np.average(out.pt_production_kg_co2e_kwh, weights=generation)
        ),
        "delivered_mean_kg_kwh": float(out.carbon_intensity.mean()),
        "balance_residual_abs_p99_mw": float(
            out.balance_residual_mw.abs().quantile(0.99)
        ),
        "method": "REN 15-minute production shares; uniform fossil coefficient calibrated to APA mainland 2023 0.092 kg/kWh; imports use REE peninsular daily operational intensity; 1.15 delivery factor",
        "limitations": [
            "derived operational average, not measured 15-minute emissions or marginal emissions",
            "no lifecycle emissions or attribution of stored hydro carbon",
            "imports use Spanish production average, not full cross-border flow tracing",
            "REN telemetering/estimation and APA inventory have different accounting boundaries; calibration is a scenario reconciliation, not independent validation",
        ],
    }


def tariff_for_member(member, contract):
    power = contract["contracted_power_kva"]
    if power > 41.4:
        return "bte_daily"
    if power > 20.7:
        return "btn_large_tri"
    if member["category"] != "residential":
        return "btn_tri"
    return (
        "btn_bi"
        if member["has_charging"] or member["member_index"] % 3 == 0
        else "btn_simple"
    )


def main():
    market = align_market()
    for tariff in TARIFFS:
        c = tariff_components(market, tariff)
        c.reset_index(names="timestamp_utc").to_parquet(
            ROOT / f"price_components_{tariff}.parquet", index=False
        )
        price_forecasts(c).to_parquet(ROOT / f"pricing_{tariff}.parquet", index=False)
    # Retain wholesale as a clearly named diagnostic, not simulator billing input.
    market.reset_index(names="timestamp_utc").to_parquet(
        ROOT / "wholesale_components_2023.parquet", index=False
    )
    metadata = {
        "year": 2023,
        "timezone_axis": "UTC; source OMIE Europe/Madrid; REN and tariffs Europe/Lisbon; DST resolved",
        "tariffs": TARIFFS,
        "formula": "(OMIE + MIBEL + 0.004 + 0.010)*1.15 + TAR_energy + BTE_peak_equivalent + IEC_0.001; EUR/kWh pre-VAT",
        "commercial_scope": "Coopernico BASE Jan2023-inspired benchmark; same margin/loss convention extended to BTE, not an actual BTE offer",
        "exclusions": [
            "VAT, audiovisual fee, DGEG fee",
            "fixed contracted-power charges: reported separately, excluded from marginal control KPI",
            "reactive energy: unity power factor in simulator",
            "no guarantees-of-origin surcharge; no 2024 supplier TSE charge",
        ],
        "negative_prices": "preserved; multiplying a negative price by 0.8 is not a 20 percent saving",
        "forecasts": "OMIE after conservative 14:00 Madrid publication; definitive MIBEL estimated by last-day same-issue-hour persistence, never future realized values",
        "carbon": carbon_reference(),
    }
    (ROOT / "aligned_reference_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

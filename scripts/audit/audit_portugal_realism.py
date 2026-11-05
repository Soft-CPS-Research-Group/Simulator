"""Measure current generated inputs independently; optional runtime fault probes.

These are internal coherence and diversity tests, not empirical validation.
No fitted parameters, dataset copies, training runs or annual KPI claims.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts"))
from portugal_activity import CONFIG, N, N23
from revise_portugal_ev_mobility import dump
from audit_portugal_heldout_2024 import runtime_fingerprint, runtime_window


def measures(root):
    members = pd.read_parquet(root/"catalogs/members.parquet")
    apps = pd.read_parquet(root/"catalogs/deferrables.parquet")
    daily = pd.read_parquet(root/"catalogs/household_activity.parquet")
    trips = pd.read_parquet(root/"catalogs/mobility_episodes.parquet")
    long = trips.loc[trips.trip_kind.isin(["weekend_away", "vacation_away", "holiday_bridge"])].copy()
    long["year"] = np.where(long.departure_time_step < N23, 2023, 2024)
    vehicles = pd.read_parquet(root/"catalogs/electric_vehicles.parquet")
    index = pd.MultiIndex.from_product([vehicles.loc[vehicles.vehicle_pool.eq("resident"), "ev_id"], [2023, 2024]])
    counts = long.groupby(["ev_id", "year"]).size().reindex(index, fill_value=0)
    statistics = dict(family=root.name, members=len(members),
        ev_distributions=json.loads((root/"ev_mobility_revision.json").read_text())["distributions"],
        multiday_trips=len(long), multiday_with_destination_travel=int(long.destination_trip_count.gt(0).sum()),
        multiday_count_distribution=counts.value_counts().sort_index().to_dict(),
        cycle_profiles=int(sum(len(pd.read_parquet(root/f"deferrables/{a.deferrable_id}_profiles.parquet")) for a in apps.itertuples())))
    statistics["annual_physical_nonshiftable_kwh"] = json.loads((root/"realism_revision.json").read_text()).get("annual_physical_nonshiftable_kwh", [])
    conservation, away_dhw, home_dhw, native_violations = [], [], [], []
    loads = []
    schema = json.loads((root/"schema.json").read_text())
    for m in members.itertuples():
        f = pd.read_parquet(root/f"timeseries/{m.member_id}.parquet")
        ledger = pd.read_parquet(root/f"end_uses/{m.member_id}.parquet")
        conservation.append(float(np.max(np.abs(ledger.sum(axis=1)-f.non_shiftable_load))))
        assert (f.non_shiftable_load >= 0).all() and (f.solar_generation >= 0).all()
        loads.append(f.non_shiftable_load.iloc[N23:].to_numpy())
        if m.category == "residential" and m.electric_dhw:
            d = daily.loc[daily.member_id.eq(m.member_id) & daily.absence_start.ge(0)].drop_duplicates("absence_start")
            mask = np.zeros(N, dtype=bool)
            for row in d.itertuples():
                mask[row.absence_start:row.absence_end] = True
            if mask.any():
                away_dhw.extend(ledger.dhw_proxy_kwh[mask].tolist())
                home_dhw.extend(ledger.dhw_proxy_kwh[~mask].tolist())
        b = schema["buildings"][m.member_id]
        native = pd.read_parquet(root/b["energy_simulation"])
        count = int(((native.non_shiftable_load-native.solar_generation)*4 > b["electrical_service"]["limits"]["total"]["import_kw"]+1e-5).sum())
        if count:
            native_violations.append(dict(member_id=m.member_id, steps=count))
    statistics.update(max_end_use_balance_error_kwh=max(conservation),
        absence_to_home_dhw_mean_ratio=float(np.mean(away_dhw)/np.mean(home_dhw)) if away_dhw else None,
        native_import_over_limit=native_violations)
    correlations = np.corrcoef(loads)
    statistics["median_member_load_correlation_2024"] = float(np.median(correlations[np.triu_indices(len(loads), 1)]))
    statistics["exact_duplicate_member_loads_2024"] = sum(np.array_equal(a, b) for i,a in enumerate(loads) for b in loads[i+1:])
    assert statistics["exact_duplicate_member_loads_2024"] == 0
    assert max(conservation) < 1e-5
    event_reports = []
    for path in sorted((root/"evaluation").glob("*.json")):
        s = json.loads(path.read_text())
        if not s.get("robustness", {}).get("enabled"):
            continue
        f = pd.read_parquet(root/s["robustness"]["events_file"])
        # Noise and domain-clipping are two native operators for one sensor
        # exposure. Count that paired exposure only once in the scientific view.
        cause = f.cause_id.where(~f.generation_regime.eq("controlled_noise_stress"),
            f.cause_id.str.replace("_domain", "", regex=False))
        latent = f.loc[~cause.duplicated()]
        temporal = latent.loc[latent.end_time_step-latent.start_time_step < N23]
        starts23 = set(temporal.loc[temporal.start_time_step < N23, "start_time_step"]+34944)
        starts24 = set(temporal.loc[temporal.start_time_step >= N23, "start_time_step"])
        event_reports.append(dict(variant=path.stem, channel_rows=len(f), latent_causes=len(latent),
            latent_causes_by_regime=latent.generation_regime.value_counts().to_dict(),
            exact_52_week_replayed_starts=len(starts23 & starts24), distinct_2024_starts=len(starts24),
            largest_simultaneous_channel_group=int(f.groupby("start_time_step").size().max())))
    statistics["events"] = event_reports
    return statistics


def runtime_probes(report):
    # Exercise every injected fault mode and both DR strata at their NEW dates.
    for family, variant in (("core_30", "core_30_combined"), ("premium_100", "premium_100_allin")):
        root = ROOT/f"data/datasets/rec_pt_2023_{family}_v2"
        path = root/f"evaluation/{variant}.json"
        schema = json.loads(path.read_text())
        f = pd.read_parquet(root/schema["robustness"]["events_file"])
        f = f.loc[f.start_time_step.gt(0)].drop_duplicates(["module", "mode", "generation_regime", "target_type"])
        probes = [(str(r.cause_id), int(r.start_time_step)) for r in f.itertuples()]
        if schema.get("demand_response", {}).get("enabled"):
            dr = pd.read_parquet(root/schema["demand_response"]["requests_file"]).drop_duplicates("stratum")
            probes.extend((r.request_id, int(r.start_time_step)) for r in dr.itertuples())
        grid = pd.read_parquet(root/"catalogs/grid_event_programme.parquet")
        grid = grid.loc[grid.input_file.isin([b["energy_simulation"] for b in schema["buildings"].values()]) & grid.active_slots.gt(0)].drop_duplicates("generation_regime")
        probes.extend((r.cause_id, int(r.start_time_step)) for r in grid.itertuples())
        completed = {(x["schema"], x["label"], x["start"]) for x in report["runtime"] if x["status"] == "pass"}
        for label, t in dict.fromkeys(probes):
            if (path.name, label, t-1) in completed:
                continue
            result = runtime_window(path, t-1, t+2, label)
            report["runtime"].append(result)
            dump(ROOT/"results/portugal_rec_native/realism_validation.json", report)
            print(variant, label, result["status"], flush=True)


def figure(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    families = sorted(report["families"], key=lambda x:x["members"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    labels = [str(x["members"]) for x in families]
    resident = [next(d for d in x["ev_distributions"] if d["year"] == 2024 and d["pool"] == "resident") for x in families]
    axes[0, 0].bar(labels, [d["request_mean_pp"] for d in resident], color="#287d8e")
    axes[0, 0].set(title="Pedido médio EV · residentes, 2024", xlabel="Membros", ylabel="Pontos percentuais de SOC")
    axes[0, 1].bar(labels, [d["sessions_per_vehicle_week"] for d in resident], color="#287d8e")
    axes[0, 1].set(title="Frequência de carregamento", xlabel="Membros", ylabel="Sessões por EV e semana")
    for x in families:
        counts = x["multiday_count_distribution"]
        axes[1, 0].plot([int(k) for k in counts], list(counts.values()), marker="o", label=str(x["members"]))
    axes[1, 0].set(title="Viagens prolongadas variam por veículo e ano", xlabel="Viagens por EV/ano", ylabel="Veículos-ano")
    axes[1, 0].legend(title="Membros", frameon=False)
    premium = next(x for x in families if x["members"] == 100)
    event = next(x for x in premium["events"] if x["variant"].endswith("allin"))
    regimes = event["latent_causes_by_regime"]
    names = {"local_fault":"Falhas locais", "common_cause_fault":"Causa comum", "controlled_stress":"Stress controlado", "controlled_noise_stress":"Ruído de stress"}
    axes[1, 1].barh([names[k] for k in regimes], list(regimes.values()), color="#bd7d3e")
    axes[1, 1].set(title="Premium · causas de telemetria/controlo, 2 anos", xlabel="Causas (não linhas de canais)")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("Portugal · revisão de realismo sintético\nCoerência e diversidade internas; validação empírica apresentada separadamente", fontsize=12)
    fig.savefig(ROOT/"docs/portugal_rec_native/figures/realism_validation.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--reuse-runtime", action="store_true", help="Reuse identical-input probes and run any newly added cases")
    args = parser.parse_args()
    fingerprint = runtime_fingerprint()
    report = dict(revision=CONFIG["revision"], runtime_input_fingerprint=fingerprint,
        validation_scope="Internal coherence, physical feasibility, diversity and matched comparisons; no empirical fit claim",
        families=[measures(root) for root in sorted((ROOT/"data/datasets").glob("rec_pt_2023_*_v2"))], runtime=[])
    reproducibility = json.loads((ROOT/"results/portugal_rec_native/manifest.json").read_text()).get("realism_reproducibility")
    if reproducibility and reproducibility["runtime_input_fingerprint_unchanged"] == fingerprint:
        report["reproducibility"] = reproducibility
    if args.reuse_runtime:
        previous = json.loads((ROOT/"results/portugal_rec_native/realism_validation.json").read_text())
        assert previous["runtime_input_fingerprint"] == fingerprint
        assert all(r["status"] == "pass" for r in previous["runtime"])
        report["runtime"] = previous["runtime"]
        report["runtime_evidence_reused_with_identical_executable_inputs"] = True
    if args.runtime or args.reuse_runtime:
        runtime_probes(report)
    assert runtime_fingerprint() == fingerprint
    report["status"] = "pass"
    report["generator_sha256"] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [ROOT/"scripts"/n for n in ("portugal_activity.py", "portugal_stochastic_demand.py","portugal_commercial_activity.py", "portugal_realism_config.json", "revise_portugal_realism.py", "revise_portugal_ev_mobility.py")]}
    dump(ROOT/"results/portugal_rec_native/realism_validation.json", report)
    figure(report)
    print(json.dumps({"status":report["status"], "runtime_probes":len(report["runtime"])}))


if __name__ == "__main__":
    main()

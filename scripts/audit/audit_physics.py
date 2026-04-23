#!/usr/bin/env python3
"""Deterministic physics audit runner for CityLearn.

Writes a machine-readable report to ``outputs/audit/physics_audit.json``.
Exit codes:
- 0: all scenarios passed
- 1: one or more hard failures
- 2: skipped because required dependencies are missing
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    import numpy as np
    _NUMPY_IMPORT_ERROR: Optional[Exception] = None
except ModuleNotFoundError as exc:
    np = None  # type: ignore[assignment]
    _NUMPY_IMPORT_ERROR = exc

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class Scenario:
    """Single deterministic scenario definition."""

    scenario_id: str
    schema: str
    interface: str
    topology_mode: str
    seconds_per_time_step: int
    seed: int
    simulation_start_time_step: int
    simulation_end_time_step: int
    central_agent: bool = True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/audit/physics_audit.json",
        help="Path to output JSON report.",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=0,
        help="Optional cap for quick local smoke runs. 0 means run all.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately on first hard failure.",
    )
    return parser.parse_args()


def _scenario_matrix() -> List[Scenario]:
    """Build deterministic matrix covering interface/topology/step durations."""

    scenarios: List[Scenario] = []
    seconds_values = [60, 300, 900]
    seed_bases = {60: 100, 300: 200, 900: 300}

    for seconds in seconds_values:
        base = seed_bases[seconds]

        scenarios.append(
            Scenario(
                scenario_id=f"flat_static_{seconds}s",
                schema=str(ROOT / "data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"),
                interface="flat",
                topology_mode="static",
                seconds_per_time_step=seconds,
                seed=base + 1,
                simulation_start_time_step=0,
                simulation_end_time_step=160,
            )
        )

        scenarios.append(
            Scenario(
                scenario_id=f"entity_static_{seconds}s",
                schema=str(ROOT / "data/datasets/citylearn_three_phase_electrical_service_demo/schema.json"),
                interface="entity",
                topology_mode="static",
                seconds_per_time_step=seconds,
                seed=base + 2,
                simulation_start_time_step=0,
                simulation_end_time_step=160,
            )
        )

        scenarios.append(
            Scenario(
                scenario_id=f"entity_dynamic_member_boundary_{seconds}s",
                schema=str(ROOT / "data/datasets/citylearn_three_phase_dynamic_topology_demo/schema.json"),
                interface="entity",
                topology_mode="dynamic",
                seconds_per_time_step=seconds,
                seed=base + 3,
                simulation_start_time_step=0,
                simulation_end_time_step=1060,
            )
        )

        scenarios.append(
            Scenario(
                scenario_id=f"entity_dynamic_asset_boundary_{seconds}s",
                schema=str(ROOT / "data/datasets/citylearn_three_phase_dynamic_topology_demo/schema.json"),
                interface="entity",
                topology_mode="dynamic",
                seconds_per_time_step=seconds,
                seed=base + 4,
                simulation_start_time_step=0,
                simulation_end_time_step=1620,
            )
        )

    return scenarios


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _safe_series_value(series: Any, idx: int) -> Optional[float]:
    if series is None:
        return None
    try:
        n = len(series)  # type: ignore[arg-type]
    except Exception:
        return None
    if n <= 0:
        return None
    clamped = min(max(int(idx), 0), n - 1)
    try:
        value = float(series[clamped])  # type: ignore[index]
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return value


def _collect_non_finite(name: str, array_like: Any) -> Optional[str]:
    arr = np.asarray(array_like)
    if arr.size == 0:
        return None
    if not np.isfinite(arr).all():
        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())
        return f"{name}: non-finite values detected (nan={nan_count}, inf={inf_count})"
    return None


def _check_observation_finite(env, observation: Any) -> Tuple[bool, List[str], int]:
    errors: List[str] = []
    checked = 0

    if str(getattr(env, "interface", "flat")).lower() == "entity":
        if not isinstance(observation, Mapping):
            return False, ["Entity observation is not a mapping payload."], checked
        tables = observation.get("tables", {})
        if not isinstance(tables, Mapping):
            return False, ["Entity observation missing 'tables' mapping."], checked
        for table_name, values in tables.items():
            checked += 1
            err = _collect_non_finite(f"tables.{table_name}", values)
            if err is not None:
                errors.append(err)
    else:
        if not isinstance(observation, (list, tuple, np.ndarray)):
            return False, ["Flat observation is not list/tuple/array."], checked
        for i, row in enumerate(observation):
            checked += 1
            err = _collect_non_finite(f"obs[{i}]", row)
            if err is not None:
                errors.append(err)

    return len(errors) == 0, errors, checked


def _build_zero_action(env) -> Tuple[Any, bool, str]:
    interface = str(getattr(env, "interface", "flat")).lower()
    central_agent = bool(getattr(env, "central_agent", False))

    if interface == "entity":
        space = env.action_space
        try:
            building_space = space["tables"]["building"]
            charger_space = space["tables"]["charger"]
        except Exception as exc:
            return None, False, f"Entity action space missing tables/building/charger: {exc}"

        building = np.zeros(building_space.shape, dtype=np.float32)
        charger = np.zeros(charger_space.shape, dtype=np.float32)
        action = {"tables": {"building": building, "charger": charger}}
        if hasattr(space, "contains") and not bool(space.contains(action)):
            return action, False, "Entity action does not satisfy env.action_space.contains(action)."
        return action, True, ""

    if central_agent:
        try:
            flat_space = env.action_space[0]
        except Exception as exc:
            return None, False, f"Flat central action space missing index 0: {exc}"
        vector = np.zeros(flat_space.shape, dtype=np.float32)
        if hasattr(flat_space, "contains") and not bool(flat_space.contains(vector)):
            return [vector], False, "Flat central zero action outside action space bounds."
        return [vector], True, ""

    vectors = []
    for i, building_space in enumerate(env.action_space):
        vector = np.zeros(building_space.shape, dtype=np.float32)
        if hasattr(building_space, "contains") and not bool(building_space.contains(vector)):
            return None, False, f"Flat decentralized zero action outside bounds for building index {i}."
        vectors.append(vector)
    return vectors, True, ""


def _call_runtime_invariants(env, settled_time_step: int) -> Tuple[bool, str]:
    service = getattr(env, "_physics_invariant_service", None)
    if service is None or not hasattr(service, "assert_step_invariants"):
        return True, ""
    try:
        service.assert_step_invariants(int(max(settled_time_step, 0)))
    except Exception as exc:
        return False, f"Runtime invariant service failure at t={settled_time_step}: {exc}"
    return True, ""


def _iter_storage_soc_series(building) -> Iterable[Tuple[str, Any]]:
    for attr in ("cooling_storage", "heating_storage", "dhw_storage", "electrical_storage"):
        storage = getattr(building, attr, None)
        if storage is None:
            continue
        yield attr, getattr(storage, "soc", None)


def _check_soc_bounds(env, settled_time_step: int, eps: float = 1.0e-4) -> Tuple[bool, List[str], int]:
    errors: List[str] = []
    checked = 0
    t = int(max(settled_time_step, 0))

    for building in getattr(env, "buildings", []) or []:
        building_name = str(getattr(building, "name", "<unknown_building>"))
        for storage_name, soc_series in _iter_storage_soc_series(building):
            value = _safe_series_value(soc_series, t)
            if value is None:
                continue
            checked += 1
            if value < -eps or value > 1.0 + eps:
                errors.append(
                    f"{building_name}.{storage_name} SOC out of bounds at t={t}: {value:.6f} not in [0,1]"
                )

    for ev in getattr(env, "electric_vehicles", []) or []:
        ev_name = str(getattr(ev, "name", "<unknown_ev>"))
        battery = getattr(ev, "battery", None)
        if battery is None:
            continue
        value = _safe_series_value(getattr(battery, "soc", None), t)
        if value is None:
            continue
        checked += 1
        if value < -eps or value > 1.0 + eps:
            errors.append(f"{ev_name}.battery SOC out of bounds at t={t}: {value:.6f} not in [0,1]")

    return len(errors) == 0, errors, checked


def _pair_features(features: Sequence[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    feature_set = set(features)
    for name in features:
        if not str(name).endswith("_power_kw"):
            continue
        prefix = str(name)[: -len("_power_kw")]
        candidate = f"{prefix}_energy_kwh_step"
        if candidate in feature_set:
            pairs.append((name, candidate))
    return pairs


def _check_kw_kwh_step_consistency(env, observation: Any) -> Tuple[bool, List[str], int]:
    if str(getattr(env, "interface", "flat")).lower() != "entity":
        return True, [], 0
    if not isinstance(observation, Mapping):
        return False, ["Entity observation missing mapping payload for kW<->kWh checks."], 0

    tables = observation.get("tables", {})
    specs = getattr(env, "entity_specs", {})
    if not isinstance(tables, Mapping) or not isinstance(specs, Mapping):
        return False, ["Entity observation/specs unavailable for kW<->kWh checks."], 0

    step_hours = max(_safe_float(getattr(env, "seconds_per_time_step", 3600.0), 3600.0) / 3600.0, 1.0e-9)
    errors: List[str] = []
    checked = 0

    tables_spec = specs.get("tables", {}) if isinstance(specs.get("tables", {}), Mapping) else {}

    for table_name, arr_like in tables.items():
        table_spec = tables_spec.get(table_name, {})
        if not isinstance(table_spec, Mapping):
            continue
        features = table_spec.get("features", [])
        if not isinstance(features, Sequence):
            continue
        pairs = _pair_features([str(x) for x in features])
        if len(pairs) == 0:
            continue

        array = np.asarray(arr_like, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] == 0:
            continue
        feature_index = {str(name): idx for idx, name in enumerate(features)}

        for power_name, energy_name in pairs:
            p_idx = feature_index.get(power_name)
            e_idx = feature_index.get(energy_name)
            if p_idx is None or e_idx is None:
                continue

            power_col = array[:, p_idx]
            energy_col = array[:, e_idx]
            expected_energy = power_col * step_hours

            diff = np.abs(energy_col - expected_energy)
            tolerance = np.maximum(5.0e-3, np.abs(expected_energy) * 1.0e-3)
            failing_rows = np.where(diff > tolerance)[0]
            checked += int(array.shape[0])

            if failing_rows.size > 0:
                first = int(failing_rows[0])
                errors.append(
                    f"{table_name}.{power_name}->{energy_name} mismatch row={first}: "
                    f"energy={energy_col[first]:.6f}, expected={expected_energy[first]:.6f}, "
                    f"abs_diff={diff[first]:.6f}, tol={tolerance[first]:.6f}"
                )

    return len(errors) == 0, errors, checked


def _new_checks_state() -> Dict[str, MutableMapping[str, Any]]:
    return {
        "finite_observations": {"calls": 0, "checked": 0, "failures": 0},
        "action_space_compatibility": {"calls": 0, "failures": 0},
        "runtime_invariants": {"calls": 0, "failures": 0},
        "soc_bounds": {"calls": 0, "checked": 0, "failures": 0},
        "kw_kwh_step_consistency": {"calls": 0, "checked": 0, "failures": 0},
    }


def _run_scenario(CityLearnEnv, scenario: Scenario) -> Dict[str, Any]:
    env = None
    failures: List[str] = []
    checks = _new_checks_state()
    steps_executed = 0
    start = time.perf_counter()
    final_topology_version: Optional[int] = None

    try:
        env_kwargs = {
            "interface": scenario.interface,
            "topology_mode": scenario.topology_mode,
            "seconds_per_time_step": scenario.seconds_per_time_step,
            "random_seed": scenario.seed,
            "central_agent": scenario.central_agent,
            "simulation_start_time_step": scenario.simulation_start_time_step,
            "simulation_end_time_step": scenario.simulation_end_time_step,
            "physics_invariant_checks": True,
            "render_mode": "none",
        }
        env = CityLearnEnv(scenario.schema, **env_kwargs)

        observation, _info = env.reset(seed=scenario.seed)

        checks["runtime_invariants"]["calls"] += 1
        ok, message = _call_runtime_invariants(env, int(getattr(env, "time_step", 0)))
        if not ok:
            checks["runtime_invariants"]["failures"] += 1
            failures.append(message)

        while True:
            checks["finite_observations"]["calls"] += 1
            ok, errors, checked = _check_observation_finite(env, observation)
            checks["finite_observations"]["checked"] += checked
            if not ok:
                checks["finite_observations"]["failures"] += len(errors)
                failures.extend(errors)

            checks["kw_kwh_step_consistency"]["calls"] += 1
            ok, errors, checked = _check_kw_kwh_step_consistency(env, observation)
            checks["kw_kwh_step_consistency"]["checked"] += checked
            if not ok:
                checks["kw_kwh_step_consistency"]["failures"] += len(errors)
                failures.extend(errors)

            action, action_ok, action_message = _build_zero_action(env)
            checks["action_space_compatibility"]["calls"] += 1
            if not action_ok:
                checks["action_space_compatibility"]["failures"] += 1
                failures.append(action_message)
                break

            observation, _reward, terminated, truncated, _step_info = env.step(action)
            steps_executed += 1
            settled_step = int(max(getattr(env, "time_step", 0) - 1, 0))

            checks["runtime_invariants"]["calls"] += 1
            ok, message = _call_runtime_invariants(env, settled_step)
            if not ok:
                checks["runtime_invariants"]["failures"] += 1
                failures.append(message)

            checks["soc_bounds"]["calls"] += 1
            ok, errors, checked = _check_soc_bounds(env, settled_step)
            checks["soc_bounds"]["checked"] += checked
            if not ok:
                checks["soc_bounds"]["failures"] += len(errors)
                failures.extend(errors)

            if terminated or truncated:
                break

        final_topology_version = int(getattr(env, "topology_version", 0))

    except Exception as exc:
        failures.append(f"Unhandled scenario error: {exc}")
        failures.append(traceback.format_exc(limit=6))
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    elapsed = time.perf_counter() - start
    status = "passed" if len(failures) == 0 else "failed"
    avg_step_ms = (elapsed / max(steps_executed, 1)) * 1000.0

    return {
        "scenario": asdict(scenario),
        "status": status,
        "steps_executed": int(steps_executed),
        "duration_s": round(float(elapsed), 4),
        "avg_step_ms": round(float(avg_step_ms), 4),
        "final_topology_version": final_topology_version,
        "checks": checks,
        "failures": failures,
    }


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _load_citylearn_env():
    missing: List[str] = []
    if _NUMPY_IMPORT_ERROR is not None:
        missing.append(f"numpy: {_NUMPY_IMPORT_ERROR}")
        return None, missing

    try:
        from citylearn.citylearn import CityLearnEnv  # noqa: WPS433
    except ModuleNotFoundError as exc:
        missing.append(f"{exc.name}: {exc}")
        CityLearnEnv = None
    except Exception as exc:  # pragma: no cover - defensive import diagnostics
        missing.append(f"citylearn import error: {exc}")
        CityLearnEnv = None
    return CityLearnEnv, missing


def main() -> int:
    args = _parse_args()

    CityLearnEnv, missing = _load_citylearn_env()
    if CityLearnEnv is None:
        report = {
            "metadata": {
                "generated_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                "report_version": "physics_audit_v1",
                "root": str(ROOT),
                "python": sys.version,
                "platform": platform.platform(),
            },
            "status": "skipped",
            "reason": "missing_dependencies",
            "missing": missing,
            "scenarios": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 1,
            },
        }
        _write_report(args.output, report)
        print("SKIP physics audit: missing dependencies.")
        for item in missing:
            print(f"- {item}")
        print(f"Report: {args.output}")
        return 2

    scenarios = _scenario_matrix()
    if args.max_scenarios > 0:
        scenarios = scenarios[: args.max_scenarios]

    results: List[Dict[str, Any]] = []
    hard_failures = 0

    for scenario in scenarios:
        result = _run_scenario(CityLearnEnv, scenario)
        results.append(result)
        if result["status"] != "passed":
            hard_failures += 1
            if args.strict:
                break

    passed = sum(1 for r in results if r["status"] == "passed")
    failed = sum(1 for r in results if r["status"] == "failed")
    skipped = len(scenarios) - len(results)

    report = {
        "metadata": {
            "generated_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "report_version": "physics_audit_v1",
            "root": str(ROOT),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "status": "failed" if hard_failures > 0 else "passed",
        "scenario_matrix": {
            "seconds_per_time_step": [60, 300, 900],
            "covers_interface": ["flat", "entity"],
            "covers_topology_mode": ["static", "dynamic (entity only)"],
        },
        "scenarios": results,
        "summary": {
            "total": len(scenarios),
            "executed": len(results),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "hard_failures": hard_failures,
        },
    }
    _write_report(args.output, report)

    print(
        "Physics audit summary: "
        f"total={len(scenarios)} executed={len(results)} passed={passed} failed={failed} skipped={skipped}"
    )
    print(f"Report: {args.output}")

    if failed > 0:
        print("First failures:")
        shown = 0
        for result in results:
            if result["status"] == "passed":
                continue
            print(f"- {result['scenario']['scenario_id']}")
            for err in result.get("failures", [])[:3]:
                one_line = str(err).strip().splitlines()[0]
                print(f"  * {one_line}")
            shown += 1
            if shown >= 3:
                break
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

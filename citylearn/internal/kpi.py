from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from citylearn.cost_function import CostFunction
from citylearn.data import EnergySimulation, ZERO_DIVISION_PLACEHOLDER

if TYPE_CHECKING:
    from citylearn.citylearn import CityLearnEnv


class CityLearnKPIService:
    """Internal KPI/evaluation service for `CityLearnEnv`."""

    EV_DEPARTURE_WITHIN_TOLERANCE_DEFAULT = 0.05
    LEGACY_COST_FUNCTIONS = {
        'all_time_peak_average',
        'annual_normalized_unserved_energy_total',
        'carbon_emissions_total',
        'cost_total',
        'daily_one_minus_load_factor_average',
        'daily_peak_average',
        'discomfort_cold_delta_average',
        'discomfort_cold_delta_maximum',
        'discomfort_cold_delta_minimum',
        'discomfort_cold_proportion',
        'discomfort_hot_delta_average',
        'discomfort_hot_delta_maximum',
        'discomfort_hot_delta_minimum',
        'discomfort_hot_proportion',
        'discomfort_proportion',
        'electricity_consumption_total',
        'monthly_one_minus_load_factor_average',
        'one_minus_thermal_resilience_proportion',
        'power_outage_normalized_unserved_energy_total',
        'ramping_average',
        'zero_net_energy',
    }

    def __init__(self, env: "CityLearnEnv"):
        self.env = env

    @staticmethod
    def _to_scalar(value, default: float = 0.0) -> float:
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return float(default)

        if not np.isfinite(scalar):
            return float(default)

        return scalar

    @staticmethod
    def _safe_div(control_value: float, baseline_value: float):
        c = CityLearnKPIService._to_scalar(control_value, 0.0)
        b = CityLearnKPIService._to_scalar(baseline_value, 0.0)
        eps = float(ZERO_DIVISION_PLACEHOLDER)

        if abs(b) <= eps:
            return 1.0 if abs(c) <= eps else None

        return c / b

    @staticmethod
    def _window_steps(window_seconds: float, seconds_per_time_step: float) -> int:
        step_seconds = max(float(seconds_per_time_step), 1.0)
        return max(1, int(round(float(window_seconds) / step_seconds)))

    @staticmethod
    def _simulated_days(env: "CityLearnEnv") -> float:
        steps = max(int(getattr(env, 'time_step', 0)), 1)
        step_seconds = max(float(getattr(env, 'seconds_per_time_step', 0) or 0), 1.0)
        return (steps * step_seconds) / (24.0 * 3600.0)

    @staticmethod
    def _daily_average(total_value: float, simulated_days: float) -> Optional[float]:
        value = CityLearnKPIService._to_scalar(total_value, np.nan)

        if not np.isfinite(value):
            return None

        if simulated_days <= float(ZERO_DIVISION_PLACEHOLDER):
            return None

        return float(value / simulated_days)

    @staticmethod
    def _normalize_soc_target(value) -> Optional[float]:
        try:
            target = float(value)
        except (TypeError, ValueError):
            return None

        if not np.isfinite(target):
            return None

        if target > 1.0 and target <= 100.0:
            target = target / 100.0

        if target < 0.0 or target > 1.0:
            return None

        return float(target)

    @staticmethod
    def _metric(cost_function: str, value, name: str, level: str) -> Dict[str, object]:
        return {
            'cost_function': cost_function,
            'value': value,
            'name': name,
            'level': level,
        }

    @staticmethod
    def _sum_finite(values) -> float:
        try:
            series = np.array(values, dtype='float64').flatten()
        except (TypeError, ValueError):
            return 0.0

        if series.size == 0:
            return 0.0

        finite = series[np.isfinite(series)]
        if finite.size == 0:
            return 0.0

        return float(finite.sum())

    @staticmethod
    def _equity_relative_benefit_percent(cost_scenario: float, cost_baseline: float) -> Optional[float]:
        scenario = CityLearnKPIService._to_scalar(cost_scenario, np.nan)
        baseline = CityLearnKPIService._to_scalar(cost_baseline, np.nan)

        if not np.isfinite(scenario) or not np.isfinite(baseline) or baseline <= 0.0:
            return None

        return float(100.0 * (baseline - scenario) / baseline)

    @staticmethod
    def _equity_distribution_metrics(relative_benefits: np.ndarray) -> Dict[str, Optional[float]]:
        benefits = np.array(relative_benefits, dtype='float64')
        benefits = benefits[np.isfinite(benefits)]

        if benefits.size == 0:
            return {
                'equity_gini_benefit': None,
                'equity_cr20_benefit': None,
                'equity_losers_percent': None,
            }

        losers_percent = float(100.0 * np.count_nonzero(benefits < 0.0) / benefits.size)
        benefits_plus = np.clip(benefits, 0.0, None)
        total_plus = float(benefits_plus.sum())

        if total_plus <= 0.0:
            return {
                'equity_gini_benefit': None,
                'equity_cr20_benefit': None,
                'equity_losers_percent': losers_percent,
            }

        n = benefits_plus.size
        diff_sum = float(np.abs(benefits_plus[:, None] - benefits_plus[None, :]).sum())
        gini = float(diff_sum / (2.0 * n * total_plus))

        k = max(1, int(np.ceil(0.2 * n)))
        top_sum = float(np.sort(benefits_plus)[::-1][:k].sum())
        cr20 = float(top_sum / total_plus)

        return {
            'equity_gini_benefit': gini,
            'equity_cr20_benefit': cr20,
            'equity_losers_percent': losers_percent,
        }

    @staticmethod
    def _equity_bpr(
        non_negative_relative_benefits: Mapping[str, float],
        groups: Mapping[str, Optional[str]],
    ) -> Optional[float]:
        if len(non_negative_relative_benefits) == 0:
            return None

        asset_poor_values = []
        asset_rich_values = []

        for building_name, value in non_negative_relative_benefits.items():
            group = groups.get(building_name)

            if group == 'asset_poor':
                asset_poor_values.append(float(value))
            elif group == 'asset_rich':
                asset_rich_values.append(float(value))
            else:
                return None

        if len(asset_poor_values) == 0 or len(asset_rich_values) == 0:
            return None

        rich_mean = float(np.mean(asset_rich_values))
        poor_mean = float(np.mean(asset_poor_values))

        if rich_mean <= 0.0:
            return None

        return float(poor_mean / rich_mean)

    def _compute_ev_metrics(self, building) -> Dict[str, float]:
        t_final = int(max(building.time_step, 0))
        departures_total = 0
        departures_met = 0
        departures_within_tolerance = 0
        departure_deficit_sum = 0.0
        charge_total_kwh = 0.0
        v2g_export_total_kwh = 0.0
        tolerance = max(
            self._to_scalar(
                getattr(self.env, 'ev_departure_within_tolerance', self.EV_DEPARTURE_WITHIN_TOLERANCE_DEFAULT),
                self.EV_DEPARTURE_WITHIN_TOLERANCE_DEFAULT,
            ),
            0.0,
        )

        for charger in building.electric_vehicle_chargers or []:
            consumption = np.array(charger.electricity_consumption[0:t_final + 1], dtype='float64')
            charge_total_kwh += float(np.clip(consumption, 0.0, None).sum())
            v2g_export_total_kwh += float(np.clip(-consumption, 0.0, None).sum())

            sim = charger.charger_simulation
            states = np.array(sim.electric_vehicle_charger_state, dtype='float64')
            required_soc = np.array(sim.electric_vehicle_required_soc_departure, dtype='float64')
            history_limit = min(t_final, len(states) - 2, len(required_soc) - 1, len(charger.past_connected_evs) - 1)

            if history_limit < 0:
                continue

            for t in range(history_limit + 1):
                current_state = states[t]
                next_state = states[t + 1]

                if current_state != 1 or next_state == 1:
                    continue

                ev = charger.past_connected_evs[t]
                if ev is None:
                    continue

                target_soc = self._normalize_soc_target(required_soc[t])
                if target_soc is None:
                    continue

                if t >= len(ev.battery.soc):
                    continue

                actual_soc = self._to_scalar(ev.battery.soc[t], np.nan)
                if not np.isfinite(actual_soc):
                    continue

                departures_total += 1
                if abs(actual_soc - target_soc) <= tolerance + 1e-6:
                    departures_within_tolerance += 1
                deficit = max(target_soc - actual_soc, 0.0)
                departure_deficit_sum += deficit
                if deficit <= 1e-6:
                    departures_met += 1

        success_rate = None if departures_total == 0 else departures_met / departures_total
        within_tolerance_rate = None if departures_total == 0 else departures_within_tolerance / departures_total
        deficit_mean = None if departures_total == 0 else departure_deficit_sum / departures_total

        return {
            'departures_total': float(departures_total),
            'departures_met': float(departures_met),
            'departures_within_tolerance': float(departures_within_tolerance),
            'departure_deficit_sum': float(departure_deficit_sum),
            'ev_departure_success_rate': success_rate,
            'ev_departure_within_tolerance_rate': within_tolerance_rate,
            'ev_departure_soc_deficit_mean': deficit_mean,
            'ev_charge_total_kwh': float(charge_total_kwh),
            'ev_v2g_export_total_kwh': float(v2g_export_total_kwh),
        }

    def _compute_bess_metrics(self, building) -> Dict[str, float]:
        t_final = int(max(building.time_step, 0))
        storage = building.electrical_storage
        storage_series = np.array(building.electrical_storage_electricity_consumption[0:t_final + 1], dtype='float64')
        charge_total = float(np.clip(storage_series, 0.0, None).sum())
        discharge_total = float(np.clip(-storage_series, 0.0, None).sum())
        throughput_total = charge_total + discharge_total

        capacity = self._to_scalar(getattr(storage, 'capacity', 0.0), 0.0)
        degraded_capacity = self._to_scalar(getattr(storage, 'degraded_capacity', capacity), capacity)
        equivalent_cycles = None if capacity <= 0.0 else throughput_total / (2.0 * capacity)
        fade_ratio = None if capacity <= 0.0 else (capacity - degraded_capacity) / capacity

        if fade_ratio is not None:
            fade_ratio = float(np.clip(fade_ratio, 0.0, 1.0))

        return {
            'bess_charge_total_kwh': charge_total,
            'bess_discharge_total_kwh': discharge_total,
            'bess_throughput_total_kwh': throughput_total,
            'bess_equivalent_full_cycles': equivalent_cycles,
            'bess_capacity_fade_ratio': fade_ratio,
            '_bess_capacity_kwh': capacity,
            '_bess_degraded_capacity_kwh': degraded_capacity,
        }

    def _compute_pv_metrics(self, building) -> Dict[str, float]:
        t_final = int(max(building.time_step, 0))
        solar = np.array(building.solar_generation[0:t_final + 1], dtype='float64')
        net = np.array(building.net_electricity_consumption[0:t_final + 1], dtype='float64')

        generation = np.clip(-solar, 0.0, None)
        export = np.clip(-net, 0.0, None)
        pv_generation_total = float(generation.sum())
        pv_export_total = float(np.minimum(generation, export).sum())
        self_consumption_ratio = None if pv_generation_total <= 0.0 else (pv_generation_total - pv_export_total) / pv_generation_total

        return {
            'pv_generation_total_kwh': pv_generation_total,
            'pv_export_total_kwh': pv_export_total,
            'pv_self_consumption_ratio': self_consumption_ratio,
        }

    def _compute_phase_metrics(self, building) -> Dict[str, object]:
        if not getattr(building, '_electrical_service_enabled', False):
            return {
                'electrical_service_violation_total_kwh': 0.0,
                'electrical_service_violation_time_step_count': 0.0,
                'phase_imbalance_ratio_average': None,
                'phase_import_peak_kw': {},
                'phase_export_peak_kw': {},
                '_imbalance_sum': 0.0,
                '_imbalance_count': 0.0,
            }

        t_final = int(max(building.time_step, 0))
        violation_history = np.array(getattr(building, '_charging_constraint_violation_history', [0.0]), dtype='float64')[0:t_final + 1]
        violation_total = float(np.clip(violation_history, 0.0, None).sum())
        violation_count = float(np.count_nonzero(violation_history > 1e-9))

        phase_history = getattr(building, '_charging_phase_power_history_kw', {}) or {}
        phase_import_peak = {}
        phase_export_peak = {}

        for phase_name, values in phase_history.items():
            series = np.array(values[0:t_final + 1], dtype='float64')
            phase_import_peak[phase_name] = float(np.clip(series, 0.0, None).max(initial=0.0))
            phase_export_peak[phase_name] = float(np.clip(-series, 0.0, None).max(initial=0.0))

        imbalance_sum = 0.0
        imbalance_count = 0.0
        imbalance_average = None

        if getattr(building, '_electrical_service_mode', 'single_phase') == 'three_phase':
            names = [n for n in ['L1', 'L2', 'L3'] if n in phase_history]
            if len(names) == 3:
                stacked = np.stack([np.array(phase_history[n][0:t_final + 1], dtype='float64') for n in names], axis=1)
                for row in stacked:
                    abs_row = np.abs(row)
                    mean_abs = float(abs_row.mean())
                    if mean_abs <= 1e-9:
                        ratio = 0.0
                    else:
                        ratio = float((abs_row.max() - abs_row.min()) / mean_abs)
                    imbalance_sum += ratio
                    imbalance_count += 1.0

                if imbalance_count > 0:
                    imbalance_average = imbalance_sum / imbalance_count

        return {
            'electrical_service_violation_total_kwh': violation_total,
            'electrical_service_violation_time_step_count': violation_count,
            'phase_imbalance_ratio_average': imbalance_average,
            'phase_import_peak_kw': phase_import_peak,
            'phase_export_peak_kw': phase_export_peak,
            '_imbalance_sum': imbalance_sum,
            '_imbalance_count': imbalance_count,
        }

    def _collect_market_totals(self, building_names: List[str]) -> Tuple[Mapping[str, Mapping[str, float]], Mapping[str, float]]:
        history = getattr(self.env, '_community_market_settlement_history', []) or []
        by_building = {
            name: {
                'community_local_import_total_kwh': 0.0,
                'community_local_export_total_kwh': 0.0,
                'community_grid_import_after_local_total_kwh': 0.0,
                'community_grid_export_after_local_total_kwh': 0.0,
                'community_settled_cost_total_eur': 0.0,
                'community_counterfactual_cost_total_eur': 0.0,
                'community_market_savings_total_eur': 0.0,
            }
            for name in building_names
        }

        for rows in history:
            for row in rows:
                name = row.get('building')
                if name not in by_building:
                    continue

                target = by_building[name]
                target['community_local_import_total_kwh'] += self._to_scalar(row.get('local_import_kwh'), 0.0)
                target['community_local_export_total_kwh'] += self._to_scalar(row.get('local_export_kwh'), 0.0)
                target['community_grid_import_after_local_total_kwh'] += self._to_scalar(row.get('grid_import_kwh'), 0.0)
                target['community_grid_export_after_local_total_kwh'] += self._to_scalar(row.get('grid_export_kwh'), 0.0)
                target['community_settled_cost_total_eur'] += self._to_scalar(row.get('settled_cost_eur', row.get('settled_cost')), 0.0)
                target['community_counterfactual_cost_total_eur'] += self._to_scalar(row.get('counterfactual_cost_eur'), 0.0)
                target['community_market_savings_total_eur'] += self._to_scalar(row.get('market_savings_eur'), 0.0)

        district = {
            key: float(sum(values[key] for values in by_building.values()))
            for key in [
                'community_local_import_total_kwh',
                'community_local_export_total_kwh',
                'community_grid_import_after_local_total_kwh',
                'community_grid_export_after_local_total_kwh',
                'community_settled_cost_total_eur',
                'community_counterfactual_cost_total_eur',
                'community_market_savings_total_eur',
            ]
        }

        return by_building, district

    @staticmethod
    def _resolve_step_value(value, time_step: int, default: float = 0.0) -> float:
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                return float(default)
            index = min(max(time_step, 0), len(value) - 1)
            return CityLearnKPIService._to_scalar(value[index], default)

        return CityLearnKPIService._to_scalar(value, default)

    @staticmethod
    def _allocate_weighted_share_import(imports: np.ndarray, traded_kwh: float, weights: np.ndarray) -> np.ndarray:
        allocations = np.zeros_like(imports, dtype='float64')
        remaining = max(float(traded_kwh), 0.0)
        eps = 1e-9
        weights = np.array(weights, dtype='float64')
        weights = np.clip(weights, 0.0, None)

        while remaining > eps:
            needs = imports - allocations
            active = needs > eps
            if not np.any(active):
                break

            active_weights = weights[active]
            sum_weights = float(active_weights.sum())

            if sum_weights <= eps:
                # Fallback to equal-share for active importers when weights are invalid.
                share = remaining / float(np.count_nonzero(active))
                granted = np.minimum(share, needs[active])
            else:
                granted = np.minimum(remaining * (active_weights / sum_weights), needs[active])

            granted_total = float(granted.sum())
            if granted_total <= eps:
                break

            allocations[active] += granted
            remaining -= granted_total

        return allocations

    def _default_building_conditions(self, building, control_condition, baseline_condition, *, evaluation_condition_cls, dynamics_building_cls):
        if isinstance(building, dynamics_building_cls):
            building_control_condition = (
                evaluation_condition_cls.WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV
                if control_condition is None else control_condition
            )
            building_baseline_condition = (
                evaluation_condition_cls.WITHOUT_STORAGE_AND_PARTIAL_LOAD_BUT_WITH_PV
                if baseline_condition is None else baseline_condition
            )
        else:
            building_control_condition = (
                evaluation_condition_cls.WITH_STORAGE_AND_PV
                if control_condition is None else control_condition
            )
            building_baseline_condition = (
                evaluation_condition_cls.WITHOUT_STORAGE_BUT_WITH_PV
                if baseline_condition is None else baseline_condition
            )

        return building_control_condition, building_baseline_condition

    def _condition_settled_cost_totals(
        self,
        *,
        condition_by_building: Mapping[str, object],
    ) -> Tuple[Mapping[str, float], float]:
        env = self.env
        building_names = [b.name for b in env.buildings]
        totals = {name: 0.0 for name in building_names}
        if len(building_names) == 0:
            return totals, 0.0

        final_t = int(max(getattr(env, 'time_step', 0), 0))
        ratio = self._to_scalar(getattr(env, 'community_market_sell_ratio', 0.8), 0.8)
        ratio = min(max(ratio, 0.0), 1.0)
        weights_config = getattr(env, 'community_market_import_member_weights', {}) or {}
        weights = np.array(
            [
                self._to_scalar(weights_config.get(name, 1.0), 1.0)
                for name in building_names
            ],
            dtype='float64',
        )

        for t in range(final_t + 1):
            net_values = []
            for building in env.buildings:
                condition = condition_by_building.get(building.name)
                net_series = np.array(getattr(building, f'net_electricity_consumption{condition.value}'), dtype='float64')
                net_values.append(self._to_scalar(net_series[t] if t < len(net_series) else np.nan, 0.0))

            net_values = np.array(net_values, dtype='float64')
            imports = np.clip(net_values, 0.0, None)
            exports = np.clip(-net_values, 0.0, None)
            total_import = float(imports.sum())
            total_export = float(exports.sum())
            traded_kwh = min(total_import, total_export)

            if total_import > 0.0 and traded_kwh > 0.0:
                local_import = self._allocate_weighted_share_import(imports, traded_kwh, weights)
            else:
                local_import = np.zeros_like(imports, dtype='float64')

            if total_export > 0.0 and traded_kwh > 0.0:
                local_export = exports * (traded_kwh / total_export)
            else:
                local_export = np.zeros_like(exports, dtype='float64')

            grid_export_price_cfg = getattr(env, 'community_market_grid_export_price', 0.0)

            for idx, building in enumerate(env.buildings):
                grid_import_price = self._to_scalar(building.pricing.electricity_pricing[t], 0.0)
                local_price = ratio * grid_import_price
                grid_export_price = self._resolve_step_value(grid_export_price_cfg, t, 0.0)
                grid_import_remaining = max(imports[idx] - local_import[idx], 0.0)
                grid_export_remaining = max(exports[idx] - local_export[idx], 0.0)

                cost = (
                    grid_import_remaining * grid_import_price
                    + local_import[idx] * local_price
                    - local_export[idx] * local_price
                    - grid_export_remaining * grid_export_price
                )
                totals[building.name] += float(cost)

        district_total = float(sum(totals.values()))
        return totals, district_total

    def evaluate(
        self,
        control_condition=None,
        baseline_condition=None,
        comfort_band: float = None,
        *,
        evaluation_condition_cls,
        dynamics_building_cls,
    ) -> pd.DataFrame:
        """Evaluate cost functions at current time step."""

        env = self.env

        get_net_electricity_consumption = lambda x, c: getattr(x, f'net_electricity_consumption{c.value}')
        get_net_electricity_consumption_cost = lambda x, c: getattr(x, f'net_electricity_consumption_cost{c.value}')
        get_net_electricity_consumption_emission = lambda x, c: getattr(x, f'net_electricity_consumption_emission{c.value}')

        comfort_band = EnergySimulation.DEFUALT_COMFORT_BAND if comfort_band is None else comfort_band
        daily_steps = self._window_steps(24.0 * 3600.0, env.seconds_per_time_step)
        monthly_steps = self._window_steps(730.0 * 3600.0, env.seconds_per_time_step)
        simulated_days = self._simulated_days(env)

        legacy_building_frames: List[pd.DataFrame] = []
        extended_building_rows: List[Dict[str, object]] = []

        ev_departures_total = 0.0
        ev_departures_met = 0.0
        ev_departures_within_tolerance = 0.0
        ev_deficit_sum = 0.0
        ev_charge_total = 0.0
        ev_v2g_total = 0.0

        bess_charge_total = 0.0
        bess_discharge_total = 0.0
        bess_throughput_total = 0.0
        bess_capacity_total = 0.0
        bess_capacity_loss_total = 0.0

        pv_generation_total = 0.0
        pv_export_total = 0.0

        phase_violation_total = 0.0
        phase_violation_count = 0.0
        phase_imbalance_sum = 0.0
        phase_imbalance_count = 0.0

        building_names = [building.name for building in env.buildings]
        equity_group_by_building = {building.name: getattr(building, 'equity_group', None) for building in env.buildings}
        equity_relative_benefit_by_building: Dict[str, Optional[float]] = {}
        equity_valid_benefits: Dict[str, float] = {}

        for building in env.buildings:
            if isinstance(building, dynamics_building_cls):
                building_control_condition = (
                    evaluation_condition_cls.WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV
                    if control_condition is None else control_condition
                )
                building_baseline_condition = (
                    evaluation_condition_cls.WITHOUT_STORAGE_AND_PARTIAL_LOAD_BUT_WITH_PV
                    if baseline_condition is None else baseline_condition
                )
            else:
                building_control_condition = (
                    evaluation_condition_cls.WITH_STORAGE_AND_PV
                    if control_condition is None else control_condition
                )
                building_baseline_condition = (
                    evaluation_condition_cls.WITHOUT_STORAGE_BUT_WITH_PV
                    if baseline_condition is None else baseline_condition
                )

            discomfort_kwargs = {
                'indoor_dry_bulb_temperature': building.indoor_dry_bulb_temperature,
                'dry_bulb_temperature_cooling_set_point': building.indoor_dry_bulb_temperature_cooling_set_point,
                'dry_bulb_temperature_heating_set_point': building.indoor_dry_bulb_temperature_heating_set_point,
                'band': building.comfort_band if comfort_band is None else comfort_band,
                'occupant_count': building.occupant_count,
            }
            unmet, cold, hot, \
                cold_minimum_delta, cold_maximum_delta, cold_average_delta, \
                hot_minimum_delta, hot_maximum_delta, hot_average_delta = CostFunction.discomfort(**discomfort_kwargs)
            expected_energy = building.cooling_demand + building.heating_demand + building.dhw_demand + building.non_shiftable_load
            served_energy = building.energy_from_cooling_device + building.energy_from_cooling_storage \
                + building.energy_from_heating_device + building.energy_from_heating_storage \
                + building.energy_from_dhw_device + building.energy_from_dhw_storage \
                + building.energy_to_non_shiftable_load
            ec_c = CostFunction.electricity_consumption(get_net_electricity_consumption(building, building_control_condition))[-1]
            ec_b = CostFunction.electricity_consumption(get_net_electricity_consumption(building, building_baseline_condition))[-1]
            net_c_series = np.array(get_net_electricity_consumption(building, building_control_condition), dtype='float64')
            net_b_series = np.array(get_net_electricity_consumption(building, building_baseline_condition), dtype='float64')
            export_c = self._sum_finite(np.clip(-net_c_series, 0.0, None))
            export_b = self._sum_finite(np.clip(-net_b_series, 0.0, None))
            zne_c = CostFunction.zero_net_energy(get_net_electricity_consumption(building, building_control_condition))[-1]
            zne_b = CostFunction.zero_net_energy(get_net_electricity_consumption(building, building_baseline_condition))[-1]
            ce_c = CostFunction.carbon_emissions(get_net_electricity_consumption_emission(building, building_control_condition))[-1]
            ce_b = CostFunction.carbon_emissions(get_net_electricity_consumption_emission(building, building_baseline_condition))[-1] if sum(building.carbon_intensity.carbon_intensity) != 0 else 0
            control_cost_series = get_net_electricity_consumption_cost(building, building_control_condition)
            baseline_cost_series = get_net_electricity_consumption_cost(building, building_baseline_condition)
            cost_c_legacy = CostFunction.cost(control_cost_series)[-1]
            cost_b_legacy = CostFunction.cost(baseline_cost_series)[-1]
            cost_c_raw = self._sum_finite(control_cost_series)
            cost_b_raw = self._sum_finite(baseline_cost_series)
            equity_benefit = self._equity_relative_benefit_percent(cost_c_raw, cost_b_raw)
            equity_relative_benefit_by_building[building.name] = equity_benefit

            if equity_benefit is not None:
                equity_valid_benefits[building.name] = float(equity_benefit)

            legacy_building_frame = pd.DataFrame([{
                'cost_function': 'electricity_consumption_total',
                'value': self._safe_div(ec_c, ec_b),
            }, {
                'cost_function': 'zero_net_energy',
                'value': self._safe_div(zne_c, zne_b),
            }, {
                'cost_function': 'carbon_emissions_total',
                'value': self._safe_div(ce_c, ce_b),
            }, {
                'cost_function': 'cost_total',
                'value': self._safe_div(cost_c_legacy, cost_b_legacy),
            }, {
                'cost_function': 'discomfort_proportion',
                'value': unmet[-1],
            }, {
                'cost_function': 'discomfort_cold_proportion',
                'value': cold[-1],
            }, {
                'cost_function': 'discomfort_hot_proportion',
                'value': hot[-1],
            }, {
                'cost_function': 'discomfort_cold_delta_minimum',
                'value': cold_minimum_delta[-1],
            }, {
                'cost_function': 'discomfort_cold_delta_maximum',
                'value': cold_maximum_delta[-1],
            }, {
                'cost_function': 'discomfort_cold_delta_average',
                'value': cold_average_delta[-1],
            }, {
                'cost_function': 'discomfort_hot_delta_minimum',
                'value': hot_minimum_delta[-1],
            }, {
                'cost_function': 'discomfort_hot_delta_maximum',
                'value': hot_maximum_delta[-1],
            }, {
                'cost_function': 'discomfort_hot_delta_average',
                'value': hot_average_delta[-1],
            }, {
                'cost_function': 'one_minus_thermal_resilience_proportion',
                'value': CostFunction.one_minus_thermal_resilience(power_outage=building.power_outage_signal, **discomfort_kwargs)[-1],
            }, {
                'cost_function': 'power_outage_normalized_unserved_energy_total',
                'value': CostFunction.normalized_unserved_energy(expected_energy, served_energy, power_outage=building.power_outage_signal)[-1],
            }, {
                'cost_function': 'annual_normalized_unserved_energy_total',
                'value': CostFunction.normalized_unserved_energy(expected_energy, served_energy)[-1],
            }])
            legacy_building_frame['name'] = building.name
            legacy_building_frames.append(legacy_building_frame)

            extended_building_rows.extend([
                self._metric('electricity_consumption_control_total_kwh', ec_c, building.name, 'building'),
                self._metric('electricity_consumption_baseline_total_kwh', ec_b, building.name, 'building'),
                self._metric('electricity_consumption_delta_total_kwh', ec_c - ec_b, building.name, 'building'),
                self._metric('electricity_consumption_control_daily_average_kwh', self._daily_average(ec_c, simulated_days), building.name, 'building'),
                self._metric('electricity_consumption_baseline_daily_average_kwh', self._daily_average(ec_b, simulated_days), building.name, 'building'),
                self._metric('electricity_consumption_delta_daily_average_kwh', self._daily_average(ec_c - ec_b, simulated_days), building.name, 'building'),
                self._metric('electricity_export_control_total_kwh', export_c, building.name, 'building'),
                self._metric('electricity_export_baseline_total_kwh', export_b, building.name, 'building'),
                self._metric('electricity_export_delta_total_kwh', export_c - export_b, building.name, 'building'),
                self._metric('electricity_export_control_daily_average_kwh', self._daily_average(export_c, simulated_days), building.name, 'building'),
                self._metric('electricity_export_baseline_daily_average_kwh', self._daily_average(export_b, simulated_days), building.name, 'building'),
                self._metric('electricity_export_delta_daily_average_kwh', self._daily_average(export_c - export_b, simulated_days), building.name, 'building'),
                self._metric('zero_net_energy_control_total_kwh', zne_c, building.name, 'building'),
                self._metric('zero_net_energy_baseline_total_kwh', zne_b, building.name, 'building'),
                self._metric('zero_net_energy_delta_total_kwh', zne_c - zne_b, building.name, 'building'),
                self._metric('zero_net_energy_control_daily_average_kwh', self._daily_average(zne_c, simulated_days), building.name, 'building'),
                self._metric('zero_net_energy_baseline_daily_average_kwh', self._daily_average(zne_b, simulated_days), building.name, 'building'),
                self._metric('zero_net_energy_delta_daily_average_kwh', self._daily_average(zne_c - zne_b, simulated_days), building.name, 'building'),
                self._metric('carbon_emissions_control_total_kgco2', ce_c, building.name, 'building'),
                self._metric('carbon_emissions_baseline_total_kgco2', ce_b, building.name, 'building'),
                self._metric('carbon_emissions_delta_total_kgco2', ce_c - ce_b, building.name, 'building'),
                self._metric('carbon_emissions_control_daily_average_kgco2', self._daily_average(ce_c, simulated_days), building.name, 'building'),
                self._metric('carbon_emissions_baseline_daily_average_kgco2', self._daily_average(ce_b, simulated_days), building.name, 'building'),
                self._metric('carbon_emissions_delta_daily_average_kgco2', self._daily_average(ce_c - ce_b, simulated_days), building.name, 'building'),
                self._metric('cost_control_total_eur', cost_c_raw, building.name, 'building'),
                self._metric('cost_baseline_total_eur', cost_b_raw, building.name, 'building'),
                self._metric('cost_delta_total_eur', cost_c_raw - cost_b_raw, building.name, 'building'),
                self._metric('cost_control_daily_average_eur', self._daily_average(cost_c_raw, simulated_days), building.name, 'building'),
                self._metric('cost_baseline_daily_average_eur', self._daily_average(cost_b_raw, simulated_days), building.name, 'building'),
                self._metric('cost_delta_daily_average_eur', self._daily_average(cost_c_raw - cost_b_raw, simulated_days), building.name, 'building'),
                self._metric('equity_relative_benefit_percent', equity_benefit, building.name, 'building'),
            ])

            ev_metrics = self._compute_ev_metrics(building)
            extended_building_rows.extend([
                self._metric('ev_departure_events_count', ev_metrics['departures_total'], building.name, 'building'),
                self._metric('ev_departure_met_events_count', ev_metrics['departures_met'], building.name, 'building'),
                self._metric('ev_departure_within_tolerance_events_count', ev_metrics['departures_within_tolerance'], building.name, 'building'),
                self._metric('ev_departure_success_rate', ev_metrics['ev_departure_success_rate'], building.name, 'building'),
                self._metric('ev_departure_within_tolerance_rate', ev_metrics['ev_departure_within_tolerance_rate'], building.name, 'building'),
                self._metric('ev_departure_soc_deficit_mean', ev_metrics['ev_departure_soc_deficit_mean'], building.name, 'building'),
                self._metric('ev_charge_total_kwh', ev_metrics['ev_charge_total_kwh'], building.name, 'building'),
                self._metric('ev_v2g_export_total_kwh', ev_metrics['ev_v2g_export_total_kwh'], building.name, 'building'),
            ])
            ev_departures_total += ev_metrics['departures_total']
            ev_departures_met += ev_metrics['departures_met']
            ev_departures_within_tolerance += ev_metrics['departures_within_tolerance']
            ev_deficit_sum += ev_metrics['departure_deficit_sum']
            ev_charge_total += ev_metrics['ev_charge_total_kwh']
            ev_v2g_total += ev_metrics['ev_v2g_export_total_kwh']

            bess_metrics = self._compute_bess_metrics(building)
            extended_building_rows.extend([
                self._metric('bess_charge_total_kwh', bess_metrics['bess_charge_total_kwh'], building.name, 'building'),
                self._metric('bess_discharge_total_kwh', bess_metrics['bess_discharge_total_kwh'], building.name, 'building'),
                self._metric('bess_throughput_total_kwh', bess_metrics['bess_throughput_total_kwh'], building.name, 'building'),
                self._metric('bess_equivalent_full_cycles', bess_metrics['bess_equivalent_full_cycles'], building.name, 'building'),
                self._metric('bess_capacity_fade_ratio', bess_metrics['bess_capacity_fade_ratio'], building.name, 'building'),
            ])
            bess_charge_total += bess_metrics['bess_charge_total_kwh']
            bess_discharge_total += bess_metrics['bess_discharge_total_kwh']
            bess_throughput_total += bess_metrics['bess_throughput_total_kwh']
            bess_capacity_total += bess_metrics['_bess_capacity_kwh']
            bess_capacity_loss_total += max(bess_metrics['_bess_capacity_kwh'] - bess_metrics['_bess_degraded_capacity_kwh'], 0.0)

            pv_metrics = self._compute_pv_metrics(building)
            extended_building_rows.extend([
                self._metric('pv_generation_total_kwh', pv_metrics['pv_generation_total_kwh'], building.name, 'building'),
                self._metric('pv_export_total_kwh', pv_metrics['pv_export_total_kwh'], building.name, 'building'),
                self._metric('pv_generation_daily_average_kwh', self._daily_average(pv_metrics['pv_generation_total_kwh'], simulated_days), building.name, 'building'),
                self._metric('pv_export_daily_average_kwh', self._daily_average(pv_metrics['pv_export_total_kwh'], simulated_days), building.name, 'building'),
                self._metric('pv_self_consumption_ratio', pv_metrics['pv_self_consumption_ratio'], building.name, 'building'),
            ])
            pv_generation_total += pv_metrics['pv_generation_total_kwh']
            pv_export_total += pv_metrics['pv_export_total_kwh']

            phase_metrics = self._compute_phase_metrics(building)
            extended_building_rows.extend([
                self._metric('electrical_service_violation_total_kwh', phase_metrics['electrical_service_violation_total_kwh'], building.name, 'building'),
                self._metric('electrical_service_violation_time_step_count', phase_metrics['electrical_service_violation_time_step_count'], building.name, 'building'),
                self._metric('phase_imbalance_ratio_average', phase_metrics['phase_imbalance_ratio_average'], building.name, 'building'),
            ])
            for phase_name, value in phase_metrics['phase_import_peak_kw'].items():
                extended_building_rows.append(self._metric(f'phase_import_peak_kw_{phase_name}', value, building.name, 'building'))
            for phase_name, value in phase_metrics['phase_export_peak_kw'].items():
                extended_building_rows.append(self._metric(f'phase_export_peak_kw_{phase_name}', value, building.name, 'building'))

            phase_violation_total += phase_metrics['electrical_service_violation_total_kwh']
            phase_violation_count += phase_metrics['electrical_service_violation_time_step_count']
            phase_imbalance_sum += phase_metrics['_imbalance_sum']
            phase_imbalance_count += phase_metrics['_imbalance_count']

        legacy_building = pd.concat(legacy_building_frames, ignore_index=True) if legacy_building_frames else pd.DataFrame(columns=['cost_function', 'value', 'name'])
        legacy_building['level'] = 'building'

        env_control_condition = (
            evaluation_condition_cls.WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV
            if control_condition is None else control_condition
        )
        env_baseline_condition = (
            evaluation_condition_cls.WITHOUT_STORAGE_AND_PARTIAL_LOAD_BUT_WITH_PV
            if baseline_condition is None else baseline_condition
        )

        ramp_c = CostFunction.ramping(get_net_electricity_consumption(env, env_control_condition))[-1]
        ramp_b = CostFunction.ramping(get_net_electricity_consumption(env, env_baseline_condition))[-1]
        dlf_daily_c = CostFunction.one_minus_load_factor(get_net_electricity_consumption(env, env_control_condition), window=daily_steps)[-1]
        dlf_daily_b = CostFunction.one_minus_load_factor(get_net_electricity_consumption(env, env_baseline_condition), window=daily_steps)[-1]
        dlf_monthly_c = CostFunction.one_minus_load_factor(get_net_electricity_consumption(env, env_control_condition), window=monthly_steps)[-1]
        dlf_monthly_b = CostFunction.one_minus_load_factor(get_net_electricity_consumption(env, env_baseline_condition), window=monthly_steps)[-1]
        peak_daily_c = CostFunction.peak(get_net_electricity_consumption(env, env_control_condition), window=daily_steps)[-1]
        peak_daily_b = CostFunction.peak(get_net_electricity_consumption(env, env_baseline_condition), window=daily_steps)[-1]
        peak_all_c = CostFunction.peak(get_net_electricity_consumption(env, env_control_condition), window=env.time_steps)[-1]
        peak_all_b = CostFunction.peak(get_net_electricity_consumption(env, env_baseline_condition), window=env.time_steps)[-1]

        legacy_district_base = pd.DataFrame([{
            'cost_function': 'ramping_average',
            'value': self._safe_div(ramp_c, ramp_b),
        }, {
            'cost_function': 'daily_one_minus_load_factor_average',
            'value': self._safe_div(dlf_daily_c, dlf_daily_b),
        }, {
            'cost_function': 'monthly_one_minus_load_factor_average',
            'value': self._safe_div(dlf_monthly_c, dlf_monthly_b),
        }, {
            'cost_function': 'daily_peak_average',
            'value': self._safe_div(peak_daily_c, peak_daily_b),
        }, {
            'cost_function': 'all_time_peak_average',
            'value': self._safe_div(peak_all_c, peak_all_b),
        }])

        legacy_district = pd.concat([legacy_district_base, legacy_building], ignore_index=True, sort=False)
        legacy_district = legacy_district.groupby(['cost_function'])[['value']].mean().reset_index()
        legacy_district['name'] = 'District'
        legacy_district['level'] = 'district'

        legacy_cost_functions = pd.concat([legacy_district, legacy_building], ignore_index=True, sort=False)

        # Extended KPI district-level
        ec_c_env = CostFunction.electricity_consumption(get_net_electricity_consumption(env, env_control_condition))[-1]
        ec_b_env = CostFunction.electricity_consumption(get_net_electricity_consumption(env, env_baseline_condition))[-1]
        net_c_env_series = np.array(get_net_electricity_consumption(env, env_control_condition), dtype='float64')
        net_b_env_series = np.array(get_net_electricity_consumption(env, env_baseline_condition), dtype='float64')
        export_c_env = self._sum_finite(np.clip(-net_c_env_series, 0.0, None))
        export_b_env = self._sum_finite(np.clip(-net_b_env_series, 0.0, None))
        zne_c_env = CostFunction.zero_net_energy(get_net_electricity_consumption(env, env_control_condition))[-1]
        zne_b_env = CostFunction.zero_net_energy(get_net_electricity_consumption(env, env_baseline_condition))[-1]
        ce_c_env = CostFunction.carbon_emissions(get_net_electricity_consumption_emission(env, env_control_condition))[-1]
        ce_b_env = CostFunction.carbon_emissions(get_net_electricity_consumption_emission(env, env_baseline_condition))[-1]
        env_control_cost_series = get_net_electricity_consumption_cost(env, env_control_condition)
        env_baseline_cost_series = get_net_electricity_consumption_cost(env, env_baseline_condition)
        cost_c_env_raw = self._sum_finite(env_control_cost_series)
        cost_b_env_raw = self._sum_finite(env_baseline_cost_series)

        extended_district_rows = [
            self._metric('electricity_consumption_control_total_kwh', ec_c_env, 'District', 'district'),
            self._metric('electricity_consumption_baseline_total_kwh', ec_b_env, 'District', 'district'),
            self._metric('electricity_consumption_delta_total_kwh', ec_c_env - ec_b_env, 'District', 'district'),
            self._metric('electricity_consumption_control_daily_average_kwh', self._daily_average(ec_c_env, simulated_days), 'District', 'district'),
            self._metric('electricity_consumption_baseline_daily_average_kwh', self._daily_average(ec_b_env, simulated_days), 'District', 'district'),
            self._metric('electricity_consumption_delta_daily_average_kwh', self._daily_average(ec_c_env - ec_b_env, simulated_days), 'District', 'district'),
            self._metric('electricity_export_control_total_kwh', export_c_env, 'District', 'district'),
            self._metric('electricity_export_baseline_total_kwh', export_b_env, 'District', 'district'),
            self._metric('electricity_export_delta_total_kwh', export_c_env - export_b_env, 'District', 'district'),
            self._metric('electricity_export_control_daily_average_kwh', self._daily_average(export_c_env, simulated_days), 'District', 'district'),
            self._metric('electricity_export_baseline_daily_average_kwh', self._daily_average(export_b_env, simulated_days), 'District', 'district'),
            self._metric('electricity_export_delta_daily_average_kwh', self._daily_average(export_c_env - export_b_env, simulated_days), 'District', 'district'),
            self._metric('zero_net_energy_control_total_kwh', zne_c_env, 'District', 'district'),
            self._metric('zero_net_energy_baseline_total_kwh', zne_b_env, 'District', 'district'),
            self._metric('zero_net_energy_delta_total_kwh', zne_c_env - zne_b_env, 'District', 'district'),
            self._metric('zero_net_energy_control_daily_average_kwh', self._daily_average(zne_c_env, simulated_days), 'District', 'district'),
            self._metric('zero_net_energy_baseline_daily_average_kwh', self._daily_average(zne_b_env, simulated_days), 'District', 'district'),
            self._metric('zero_net_energy_delta_daily_average_kwh', self._daily_average(zne_c_env - zne_b_env, simulated_days), 'District', 'district'),
            self._metric('carbon_emissions_control_total_kgco2', ce_c_env, 'District', 'district'),
            self._metric('carbon_emissions_baseline_total_kgco2', ce_b_env, 'District', 'district'),
            self._metric('carbon_emissions_delta_total_kgco2', ce_c_env - ce_b_env, 'District', 'district'),
            self._metric('carbon_emissions_control_daily_average_kgco2', self._daily_average(ce_c_env, simulated_days), 'District', 'district'),
            self._metric('carbon_emissions_baseline_daily_average_kgco2', self._daily_average(ce_b_env, simulated_days), 'District', 'district'),
            self._metric('carbon_emissions_delta_daily_average_kgco2', self._daily_average(ce_c_env - ce_b_env, simulated_days), 'District', 'district'),
            self._metric('cost_control_total_eur', cost_c_env_raw, 'District', 'district'),
            self._metric('cost_baseline_total_eur', cost_b_env_raw, 'District', 'district'),
            self._metric('cost_delta_total_eur', cost_c_env_raw - cost_b_env_raw, 'District', 'district'),
            self._metric('cost_control_daily_average_eur', self._daily_average(cost_c_env_raw, simulated_days), 'District', 'district'),
            self._metric('cost_baseline_daily_average_eur', self._daily_average(cost_b_env_raw, simulated_days), 'District', 'district'),
            self._metric('cost_delta_daily_average_eur', self._daily_average(cost_c_env_raw - cost_b_env_raw, simulated_days), 'District', 'district'),
        ]

        ev_success_rate = None if ev_departures_total <= 0.0 else ev_departures_met / ev_departures_total
        ev_within_tolerance_rate = None if ev_departures_total <= 0.0 else ev_departures_within_tolerance / ev_departures_total
        ev_deficit_mean = None if ev_departures_total <= 0.0 else ev_deficit_sum / ev_departures_total
        extended_district_rows.extend([
            self._metric('ev_departure_events_count', ev_departures_total, 'District', 'district'),
            self._metric('ev_departure_met_events_count', ev_departures_met, 'District', 'district'),
            self._metric('ev_departure_within_tolerance_events_count', ev_departures_within_tolerance, 'District', 'district'),
            self._metric('ev_departure_success_rate', ev_success_rate, 'District', 'district'),
            self._metric('ev_departure_within_tolerance_rate', ev_within_tolerance_rate, 'District', 'district'),
            self._metric('ev_departure_soc_deficit_mean', ev_deficit_mean, 'District', 'district'),
            self._metric('ev_charge_total_kwh', ev_charge_total, 'District', 'district'),
            self._metric('ev_v2g_export_total_kwh', ev_v2g_total, 'District', 'district'),
        ])

        district_bess_cycles = None if bess_capacity_total <= 0.0 else bess_throughput_total / (2.0 * bess_capacity_total)
        district_bess_fade = None if bess_capacity_total <= 0.0 else bess_capacity_loss_total / bess_capacity_total
        extended_district_rows.extend([
            self._metric('bess_charge_total_kwh', bess_charge_total, 'District', 'district'),
            self._metric('bess_discharge_total_kwh', bess_discharge_total, 'District', 'district'),
            self._metric('bess_throughput_total_kwh', bess_throughput_total, 'District', 'district'),
            self._metric('bess_equivalent_full_cycles', district_bess_cycles, 'District', 'district'),
            self._metric('bess_capacity_fade_ratio', district_bess_fade, 'District', 'district'),
        ])

        district_pv_ratio = None if pv_generation_total <= 0.0 else (pv_generation_total - pv_export_total) / pv_generation_total
        extended_district_rows.extend([
            self._metric('pv_generation_total_kwh', pv_generation_total, 'District', 'district'),
            self._metric('pv_export_total_kwh', pv_export_total, 'District', 'district'),
            self._metric('pv_generation_daily_average_kwh', self._daily_average(pv_generation_total, simulated_days), 'District', 'district'),
            self._metric('pv_export_daily_average_kwh', self._daily_average(pv_export_total, simulated_days), 'District', 'district'),
            self._metric('pv_self_consumption_ratio', district_pv_ratio, 'District', 'district'),
        ])

        district_phase_imbalance = None if phase_imbalance_count <= 0.0 else phase_imbalance_sum / phase_imbalance_count
        extended_district_rows.extend([
            self._metric('electrical_service_violation_total_kwh', phase_violation_total, 'District', 'district'),
            self._metric('electrical_service_violation_time_step_count', phase_violation_count, 'District', 'district'),
            self._metric('phase_imbalance_ratio_average', district_phase_imbalance, 'District', 'district'),
        ])

        phase_union = ['L1', 'L2', 'L3']
        for phase_name in phase_union:
            phase_series = None
            for building in env.buildings:
                history_map = getattr(building, '_charging_phase_power_history_kw', {}) or {}
                if phase_name not in history_map:
                    continue
                t_final = int(max(building.time_step, 0))
                values = np.array(history_map[phase_name][0:t_final + 1], dtype='float64')
                if phase_series is None:
                    phase_series = np.zeros_like(values)
                size = min(len(phase_series), len(values))
                phase_series[:size] += values[:size]

            if phase_series is None:
                continue

            extended_district_rows.append(
                self._metric(f'phase_import_peak_kw_{phase_name}', float(np.clip(phase_series, 0.0, None).max(initial=0.0)), 'District', 'district')
            )
            extended_district_rows.append(
                self._metric(f'phase_export_peak_kw_{phase_name}', float(np.clip(-phase_series, 0.0, None).max(initial=0.0)), 'District', 'district')
            )

        market_by_building, market_district = self._collect_market_totals(building_names)

        for building_name in building_names:
            totals = market_by_building.get(building_name, {})
            local_import = self._to_scalar(totals.get('community_local_import_total_kwh'), 0.0)
            local_export = self._to_scalar(totals.get('community_local_export_total_kwh'), 0.0)
            grid_import = self._to_scalar(totals.get('community_grid_import_after_local_total_kwh'), 0.0)
            grid_export = self._to_scalar(totals.get('community_grid_export_after_local_total_kwh'), 0.0)
            import_share = None if (local_import + grid_import) <= 0.0 else local_import / (local_import + grid_import)
            export_share = None if (local_export + grid_export) <= 0.0 else local_export / (local_export + grid_export)

            extended_building_rows.extend([
                self._metric('community_local_import_total_kwh', local_import, building_name, 'building'),
                self._metric('community_local_export_total_kwh', local_export, building_name, 'building'),
                self._metric('community_grid_import_after_local_total_kwh', grid_import, building_name, 'building'),
                self._metric('community_grid_export_after_local_total_kwh', grid_export, building_name, 'building'),
                self._metric('community_local_import_daily_average_kwh', self._daily_average(local_import, simulated_days), building_name, 'building'),
                self._metric('community_local_export_daily_average_kwh', self._daily_average(local_export, simulated_days), building_name, 'building'),
                self._metric('community_grid_import_after_local_daily_average_kwh', self._daily_average(grid_import, simulated_days), building_name, 'building'),
                self._metric('community_grid_export_after_local_daily_average_kwh', self._daily_average(grid_export, simulated_days), building_name, 'building'),
                self._metric('community_settled_cost_total_eur', totals.get('community_settled_cost_total_eur', 0.0), building_name, 'building'),
                self._metric('community_counterfactual_cost_total_eur', totals.get('community_counterfactual_cost_total_eur', 0.0), building_name, 'building'),
                self._metric('community_market_savings_total_eur', totals.get('community_market_savings_total_eur', 0.0), building_name, 'building'),
                self._metric('community_settled_cost_daily_average_eur', self._daily_average(totals.get('community_settled_cost_total_eur', 0.0), simulated_days), building_name, 'building'),
                self._metric('community_counterfactual_cost_daily_average_eur', self._daily_average(totals.get('community_counterfactual_cost_total_eur', 0.0), simulated_days), building_name, 'building'),
                self._metric('community_market_savings_daily_average_eur', self._daily_average(totals.get('community_market_savings_total_eur', 0.0), simulated_days), building_name, 'building'),
                self._metric('community_local_share_of_demand', import_share, building_name, 'building'),
                self._metric('community_local_share_of_export', export_share, building_name, 'building'),
            ])

        district_local_import = market_district['community_local_import_total_kwh']
        district_local_export = market_district['community_local_export_total_kwh']
        district_grid_import = market_district['community_grid_import_after_local_total_kwh']
        district_grid_export = market_district['community_grid_export_after_local_total_kwh']

        extended_district_rows.extend([
            self._metric('community_local_import_total_kwh', district_local_import, 'District', 'district'),
            self._metric('community_local_export_total_kwh', district_local_export, 'District', 'district'),
            self._metric('community_grid_import_after_local_total_kwh', district_grid_import, 'District', 'district'),
            self._metric('community_grid_export_after_local_total_kwh', district_grid_export, 'District', 'district'),
            self._metric('community_local_import_daily_average_kwh', self._daily_average(district_local_import, simulated_days), 'District', 'district'),
            self._metric('community_local_export_daily_average_kwh', self._daily_average(district_local_export, simulated_days), 'District', 'district'),
            self._metric('community_grid_import_after_local_daily_average_kwh', self._daily_average(district_grid_import, simulated_days), 'District', 'district'),
            self._metric('community_grid_export_after_local_daily_average_kwh', self._daily_average(district_grid_export, simulated_days), 'District', 'district'),
            self._metric('community_settled_cost_total_eur', market_district['community_settled_cost_total_eur'], 'District', 'district'),
            self._metric('community_counterfactual_cost_total_eur', market_district['community_counterfactual_cost_total_eur'], 'District', 'district'),
            self._metric('community_market_savings_total_eur', market_district['community_market_savings_total_eur'], 'District', 'district'),
            self._metric('community_settled_cost_daily_average_eur', self._daily_average(market_district['community_settled_cost_total_eur'], simulated_days), 'District', 'district'),
            self._metric('community_counterfactual_cost_daily_average_eur', self._daily_average(market_district['community_counterfactual_cost_total_eur'], simulated_days), 'District', 'district'),
            self._metric('community_market_savings_daily_average_eur', self._daily_average(market_district['community_market_savings_total_eur'], simulated_days), 'District', 'district'),
            self._metric(
                'community_local_share_of_demand',
                None if (district_local_import + district_grid_import) <= 0.0 else district_local_import / (district_local_import + district_grid_import),
                'District',
                'district',
            ),
            self._metric(
                'community_local_share_of_export',
                None if (district_local_export + district_grid_export) <= 0.0 else district_local_export / (district_local_export + district_grid_export),
                'District',
                'district',
            ),
        ])

        equity_distribution = self._equity_distribution_metrics(np.array(list(equity_valid_benefits.values()), dtype='float64'))
        non_negative_benefits = {name: max(value, 0.0) for name, value in equity_valid_benefits.items()}
        has_complete_manual_groups = all(
            equity_group_by_building.get(name) in {'asset_rich', 'asset_poor'}
            for name in building_names
        )
        equity_bpr = self._equity_bpr(non_negative_benefits, equity_group_by_building) if has_complete_manual_groups else None

        extended_district_rows.extend([
            self._metric('equity_gini_benefit', equity_distribution['equity_gini_benefit'], 'District', 'district'),
            self._metric('equity_cr20_benefit', equity_distribution['equity_cr20_benefit'], 'District', 'district'),
            self._metric('equity_losers_percent', equity_distribution['equity_losers_percent'], 'District', 'district'),
            self._metric('equity_bpr_asset_poor_over_rich', equity_bpr, 'District', 'district'),
        ])

        extended_building = pd.DataFrame(extended_building_rows)
        extended_district = pd.DataFrame(extended_district_rows)

        cost_functions = pd.concat([legacy_cost_functions, extended_district, extended_building], ignore_index=True, sort=False)

        return cost_functions

    @staticmethod
    def _v2_name(
        level: str,
        family: str,
        subfamily: str,
        metric: str,
        variant: Optional[str] = None,
        unit: Optional[str] = None,
    ) -> str:
        tokens = [level, family, subfamily, metric]

        if variant not in (None, ''):
            tokens.append(str(variant))

        if unit not in (None, ''):
            tokens.append(str(unit))

        return '_'.join(tokens)

    def evaluate_legacy(
        self,
        control_condition=None,
        baseline_condition=None,
        comfort_band: float = None,
        *,
        evaluation_condition_cls,
        dynamics_building_cls,
    ) -> pd.DataFrame:
        all_metrics = self.evaluate(
            control_condition=control_condition,
            baseline_condition=baseline_condition,
            comfort_band=comfort_band,
            evaluation_condition_cls=evaluation_condition_cls,
            dynamics_building_cls=dynamics_building_cls,
        )

        legacy = all_metrics[all_metrics['cost_function'].isin(self.LEGACY_COST_FUNCTIONS)].copy()
        return legacy.reset_index(drop=True)

    def evaluate_v2(
        self,
        control_condition=None,
        baseline_condition=None,
        comfort_band: float = None,
        *,
        evaluation_condition_cls,
        dynamics_building_cls,
    ) -> pd.DataFrame:
        env = self.env
        all_metrics = self.evaluate(
            control_condition=control_condition,
            baseline_condition=baseline_condition,
            comfort_band=comfort_band,
            evaluation_condition_cls=evaluation_condition_cls,
            dynamics_building_cls=dynamics_building_cls,
        )
        legacy_df = all_metrics[all_metrics['cost_function'].isin(self.LEGACY_COST_FUNCTIONS)].copy()
        extended_df = all_metrics[~all_metrics['cost_function'].isin(self.LEGACY_COST_FUNCTIONS)].copy()
        simulated_days = self._simulated_days(env)

        records: Dict[Tuple[str, str, str], Dict[str, object]] = {}

        def put(level: str, name: str, cost_function: str, value):
            records[(level, name, cost_function)] = {
                'cost_function': cost_function,
                'value': value,
                'name': name,
                'level': level,
            }

        def v2(
            level: str,
            family: str,
            subfamily: str,
            metric: str,
            variant: Optional[str] = None,
            unit: Optional[str] = None,
        ) -> str:
            return self._v2_name(
                level,
                family,
                subfamily,
                metric,
                variant=variant,
                unit=unit,
            )

        def map_from(
            source_df: pd.DataFrame,
            old_name: str,
            family: str,
            subfamily: str,
            metric: str,
            variant: Optional[str] = None,
            unit: Optional[str] = None,
        ):
            subset = source_df[source_df['cost_function'] == old_name]
            for _, row in subset.iterrows():
                level = str(row['level'])
                put(
                    level,
                    str(row['name']),
                    v2(level, family, subfamily, metric, variant, unit),
                    row['value'],
                )

        # Ratios to baseline and comfort/resilience from legacy.
        legacy_map = [
            ('electricity_consumption_total', 'energy_grid', 'ratio_to_baseline', 'import_total', None, 'ratio'),
            ('zero_net_energy', 'energy_grid', 'ratio_to_baseline', 'net_exchange_total', None, 'ratio'),
            ('carbon_emissions_total', 'emissions', 'ratio_to_baseline', 'total', None, 'ratio'),
            ('cost_total', 'cost', 'ratio_to_baseline', 'total', None, 'ratio'),
            ('ramping_average', 'energy_grid', 'shape_quality', 'ramping_average_to_baseline', None, 'ratio'),
            ('daily_one_minus_load_factor_average', 'energy_grid', 'shape_quality', 'load_factor_penalty_daily_average_to_baseline', None, 'ratio'),
            ('monthly_one_minus_load_factor_average', 'energy_grid', 'shape_quality', 'load_factor_penalty_monthly_average_to_baseline', None, 'ratio'),
            ('daily_peak_average', 'energy_grid', 'shape_quality', 'peak_daily_average_to_baseline', None, 'ratio'),
            ('all_time_peak_average', 'energy_grid', 'shape_quality', 'peak_all_time_average_to_baseline', None, 'ratio'),
            ('discomfort_proportion', 'comfort_resilience', 'discomfort', 'overall', None, 'ratio'),
            ('discomfort_cold_proportion', 'comfort_resilience', 'discomfort', 'cold', None, 'ratio'),
            ('discomfort_hot_proportion', 'comfort_resilience', 'discomfort', 'hot', None, 'ratio'),
            ('discomfort_cold_delta_minimum', 'comfort_resilience', 'discomfort', 'cold_delta', 'min', 'c'),
            ('discomfort_cold_delta_maximum', 'comfort_resilience', 'discomfort', 'cold_delta', 'max', 'c'),
            ('discomfort_cold_delta_average', 'comfort_resilience', 'discomfort', 'cold_delta', 'average', 'c'),
            ('discomfort_hot_delta_minimum', 'comfort_resilience', 'discomfort', 'hot_delta', 'min', 'c'),
            ('discomfort_hot_delta_maximum', 'comfort_resilience', 'discomfort', 'hot_delta', 'max', 'c'),
            ('discomfort_hot_delta_average', 'comfort_resilience', 'discomfort', 'hot_delta', 'average', 'c'),
            ('one_minus_thermal_resilience_proportion', 'comfort_resilience', 'resilience', 'one_minus_thermal', None, 'ratio'),
            ('power_outage_normalized_unserved_energy_total', 'comfort_resilience', 'resilience', 'unserved_energy_outage_normalized', None, 'ratio'),
            ('annual_normalized_unserved_energy_total', 'comfort_resilience', 'resilience', 'unserved_energy_annual_normalized', None, 'ratio'),
        ]
        for old_name, family, subfamily, metric, variant, unit in legacy_map:
            map_from(legacy_df, old_name, family, subfamily, metric, variant, unit)

        extended_map = [
            ('cost_control_total_eur', 'cost', 'total', 'control', None, 'eur'),
            ('cost_baseline_total_eur', 'cost', 'total', 'baseline', None, 'eur'),
            ('cost_delta_total_eur', 'cost', 'total', 'delta', None, 'eur'),
            ('cost_control_daily_average_eur', 'cost', 'daily_average', 'control', None, 'eur'),
            ('cost_baseline_daily_average_eur', 'cost', 'daily_average', 'baseline', None, 'eur'),
            ('cost_delta_daily_average_eur', 'cost', 'daily_average', 'delta', None, 'eur'),
            ('electricity_consumption_control_total_kwh', 'energy_grid', 'total', 'import', 'control', 'kwh'),
            ('electricity_consumption_baseline_total_kwh', 'energy_grid', 'total', 'import', 'baseline', 'kwh'),
            ('electricity_consumption_delta_total_kwh', 'energy_grid', 'total', 'import', 'delta', 'kwh'),
            ('electricity_consumption_control_daily_average_kwh', 'energy_grid', 'daily_average', 'import', 'control', 'kwh'),
            ('electricity_consumption_baseline_daily_average_kwh', 'energy_grid', 'daily_average', 'import', 'baseline', 'kwh'),
            ('electricity_consumption_delta_daily_average_kwh', 'energy_grid', 'daily_average', 'import', 'delta', 'kwh'),
            ('electricity_export_control_total_kwh', 'energy_grid', 'total', 'export', 'control', 'kwh'),
            ('electricity_export_baseline_total_kwh', 'energy_grid', 'total', 'export', 'baseline', 'kwh'),
            ('electricity_export_delta_total_kwh', 'energy_grid', 'total', 'export', 'delta', 'kwh'),
            ('electricity_export_control_daily_average_kwh', 'energy_grid', 'daily_average', 'export', 'control', 'kwh'),
            ('electricity_export_baseline_daily_average_kwh', 'energy_grid', 'daily_average', 'export', 'baseline', 'kwh'),
            ('electricity_export_delta_daily_average_kwh', 'energy_grid', 'daily_average', 'export', 'delta', 'kwh'),
            ('zero_net_energy_control_total_kwh', 'energy_grid', 'total', 'net_exchange', 'control', 'kwh'),
            ('zero_net_energy_baseline_total_kwh', 'energy_grid', 'total', 'net_exchange', 'baseline', 'kwh'),
            ('zero_net_energy_delta_total_kwh', 'energy_grid', 'total', 'net_exchange', 'delta', 'kwh'),
            ('zero_net_energy_control_daily_average_kwh', 'energy_grid', 'daily_average', 'net_exchange', 'control', 'kwh'),
            ('zero_net_energy_baseline_daily_average_kwh', 'energy_grid', 'daily_average', 'net_exchange', 'baseline', 'kwh'),
            ('zero_net_energy_delta_daily_average_kwh', 'energy_grid', 'daily_average', 'net_exchange', 'delta', 'kwh'),
            ('carbon_emissions_control_total_kgco2', 'emissions', 'total', 'control', None, 'kgco2'),
            ('carbon_emissions_baseline_total_kgco2', 'emissions', 'total', 'baseline', None, 'kgco2'),
            ('carbon_emissions_delta_total_kgco2', 'emissions', 'total', 'delta', None, 'kgco2'),
            ('carbon_emissions_control_daily_average_kgco2', 'emissions', 'daily_average', 'control', None, 'kgco2'),
            ('carbon_emissions_baseline_daily_average_kgco2', 'emissions', 'daily_average', 'baseline', None, 'kgco2'),
            ('carbon_emissions_delta_daily_average_kgco2', 'emissions', 'daily_average', 'delta', None, 'kgco2'),
            ('pv_generation_total_kwh', 'solar_self_consumption', 'total', 'generation', None, 'kwh'),
            ('pv_export_total_kwh', 'solar_self_consumption', 'total', 'export', None, 'kwh'),
            ('pv_generation_daily_average_kwh', 'solar_self_consumption', 'daily_average', 'generation', None, 'kwh'),
            ('pv_export_daily_average_kwh', 'solar_self_consumption', 'daily_average', 'export', None, 'kwh'),
            ('pv_self_consumption_ratio', 'solar_self_consumption', 'ratio', 'self_consumption', None, 'ratio'),
            ('ev_departure_events_count', 'ev', 'events', 'departure', None, 'count'),
            ('ev_departure_met_events_count', 'ev', 'events', 'departure_met', None, 'count'),
            ('ev_departure_within_tolerance_events_count', 'ev', 'events', 'departure_within_tolerance', None, 'count'),
            ('ev_departure_success_rate', 'ev', 'performance', 'departure_success', None, 'ratio'),
            ('ev_departure_within_tolerance_rate', 'ev', 'performance', 'departure_within_tolerance', None, 'ratio'),
            ('ev_departure_soc_deficit_mean', 'ev', 'performance', 'departure_soc_deficit_mean', None, 'ratio'),
            ('ev_charge_total_kwh', 'ev', 'total', 'charge', None, 'kwh'),
            ('ev_v2g_export_total_kwh', 'ev', 'total', 'v2g_export', None, 'kwh'),
            ('bess_charge_total_kwh', 'battery', 'total', 'charge', None, 'kwh'),
            ('bess_discharge_total_kwh', 'battery', 'total', 'discharge', None, 'kwh'),
            ('bess_throughput_total_kwh', 'battery', 'total', 'throughput', None, 'kwh'),
            ('bess_equivalent_full_cycles', 'battery', 'health', 'equivalent_full_cycles', None, 'count'),
            ('bess_capacity_fade_ratio', 'battery', 'health', 'capacity_fade', None, 'ratio'),
            ('electrical_service_violation_total_kwh', 'electrical_service_phase', 'violations', 'energy_total', None, 'kwh'),
            ('electrical_service_violation_time_step_count', 'electrical_service_phase', 'violations', 'event', None, 'count'),
            ('phase_imbalance_ratio_average', 'electrical_service_phase', 'imbalance', 'phase_average', None, 'ratio'),
            ('equity_relative_benefit_percent', 'equity', 'benefit', 'relative', None, 'percent'),
            ('equity_gini_benefit', 'equity', 'distribution', 'gini_benefit', None, 'ratio'),
            ('equity_cr20_benefit', 'equity', 'distribution', 'top20_benefit', None, 'ratio'),
            ('equity_losers_percent', 'equity', 'distribution', 'losers', None, 'percent'),
            ('equity_bpr_asset_poor_over_rich', 'equity', 'distribution', 'bpr_asset_poor_over_rich', None, 'ratio'),
        ]
        for old_name, family, subfamily, metric, variant, unit in extended_map:
            map_from(extended_df, old_name, family, subfamily, metric, variant, unit)

        # Phase peaks have dynamic suffixes (L1/L2/L3) and are conditionally present.
        for _, row in extended_df.iterrows():
            old_name = str(row['cost_function'])
            level = str(row['level'])
            name = str(row['name'])

            if old_name.startswith('phase_import_peak_kw_'):
                phase = old_name.split('phase_import_peak_kw_', 1)[1].lower()
                put(
                    level,
                    name,
                    v2(level, 'electrical_service_phase', 'phase_peaks', f'import_peak_{phase}', None, 'kw'),
                    row['value'],
                )
            elif old_name.startswith('phase_export_peak_kw_'):
                phase = old_name.split('phase_export_peak_kw_', 1)[1].lower()
                put(
                    level,
                    name,
                    v2(level, 'electrical_service_phase', 'phase_peaks', f'export_peak_{phase}', None, 'kw'),
                    row['value'],
                )

        building_names = [building.name for building in env.buildings]

        # Optional community KPIs are district-only.
        if getattr(env, 'community_market_enabled', False):
            district_rows = extended_df[(extended_df['level'] == 'district') & (extended_df['name'] == 'District')]
            if getattr(env, 'community_market_kpi_local_traded_enabled', True):
                local_total = district_rows[district_rows['cost_function'] == 'community_local_import_total_kwh']['value']
                local_daily = district_rows[district_rows['cost_function'] == 'community_local_import_daily_average_kwh']['value']
                if len(local_total) > 0:
                    put(
                        'district',
                        'District',
                        v2('district', 'energy_grid', 'community_market', 'local_traded', 'total', 'kwh'),
                        local_total.iloc[0],
                    )
                if len(local_daily) > 0:
                    put(
                        'district',
                        'District',
                        v2('district', 'energy_grid', 'community_market', 'local_traded', 'daily_average', 'kwh'),
                        local_daily.iloc[0],
                    )

            if getattr(env, 'community_market_kpi_self_consumption_enabled', True):
                local_total = district_rows[district_rows['cost_function'] == 'community_local_import_total_kwh']['value']
                if len(local_total) > 0:
                    local_total_value = self._to_scalar(local_total.iloc[0], 0.0)
                    district_import_control = self._to_scalar(
                        records.get((
                            'district',
                            'District',
                            v2('district', 'energy_grid', 'total', 'import', 'control', 'kwh'),
                        ), {}).get('value'),
                        0.0,
                    )
                    share = None if district_import_control <= float(ZERO_DIVISION_PLACEHOLDER) else local_total_value / district_import_control
                    put(
                        'district',
                        'District',
                        v2('district', 'solar_self_consumption', 'community_market', 'import_share', None, 'ratio'),
                        share,
                    )

        # Export ratio_to_baseline is derived from totals with safe division.
        for building in env.buildings:
            control_cond, baseline_cond = self._default_building_conditions(
                building,
                control_condition,
                baseline_condition,
                evaluation_condition_cls=evaluation_condition_cls,
                dynamics_building_cls=dynamics_building_cls,
            )
            net_c = np.array(getattr(building, f'net_electricity_consumption{control_cond.value}'), dtype='float64')
            net_b = np.array(getattr(building, f'net_electricity_consumption{baseline_cond.value}'), dtype='float64')
            export_c = self._sum_finite(np.clip(-net_c, 0.0, None))
            export_b = self._sum_finite(np.clip(-net_b, 0.0, None))
            put(
                'building',
                building.name,
                v2('building', 'energy_grid', 'ratio_to_baseline', 'export_total', None, 'ratio'),
                self._safe_div(export_c, export_b),
            )

        env_control_condition = (
            evaluation_condition_cls.WITH_STORAGE_AND_PARTIAL_LOAD_AND_PV
            if control_condition is None else control_condition
        )
        env_baseline_condition = (
            evaluation_condition_cls.WITHOUT_STORAGE_AND_PARTIAL_LOAD_BUT_WITH_PV
            if baseline_condition is None else baseline_condition
        )
        env_net_c = np.array(getattr(env, f'net_electricity_consumption{env_control_condition.value}'), dtype='float64')
        env_net_b = np.array(getattr(env, f'net_electricity_consumption{env_baseline_condition.value}'), dtype='float64')
        env_export_c = self._sum_finite(np.clip(-env_net_c, 0.0, None))
        env_export_b = self._sum_finite(np.clip(-env_net_b, 0.0, None))
        put(
            'district',
            'District',
            v2('district', 'energy_grid', 'ratio_to_baseline', 'export_total', None, 'ratio'),
            self._safe_div(env_export_c, env_export_b),
        )

        # Cost metrics: market-enabled scenarios settle both control and baseline with same rules.
        control_condition_by_building = {}
        baseline_condition_by_building = {}
        for building in env.buildings:
            c_cond, b_cond = self._default_building_conditions(
                building,
                control_condition,
                baseline_condition,
                evaluation_condition_cls=evaluation_condition_cls,
                dynamics_building_cls=dynamics_building_cls,
            )
            control_condition_by_building[building.name] = c_cond
            baseline_condition_by_building[building.name] = b_cond

        control_cost_totals: Dict[str, float] = {}
        baseline_cost_totals: Dict[str, float] = {}
        district_control_total = 0.0
        district_baseline_total = 0.0

        if getattr(env, 'community_market_enabled', False):
            control_cost_totals, district_control_total = self._condition_settled_cost_totals(
                condition_by_building=control_condition_by_building,
            )
            baseline_cost_totals, district_baseline_total = self._condition_settled_cost_totals(
                condition_by_building=baseline_condition_by_building,
            )
        else:
            # Read mapped values from records when market is disabled.
            for building_name in building_names:
                key_control = ('building', building_name, v2('building', 'cost', 'total', 'control', None, 'eur'))
                key_baseline = ('building', building_name, v2('building', 'cost', 'total', 'baseline', None, 'eur'))
                control_cost_totals[building_name] = self._to_scalar(records.get(key_control, {}).get('value'), 0.0)
                baseline_cost_totals[building_name] = self._to_scalar(records.get(key_baseline, {}).get('value'), 0.0)

            district_control_total = self._to_scalar(
                records.get(('district', 'District', v2('district', 'cost', 'total', 'control', None, 'eur')), {}).get('value'),
                0.0,
            )
            district_baseline_total = self._to_scalar(
                records.get(('district', 'District', v2('district', 'cost', 'total', 'baseline', None, 'eur')), {}).get('value'),
                0.0,
            )

        for building_name in building_names:
            control_total = self._to_scalar(control_cost_totals.get(building_name), 0.0)
            baseline_total = self._to_scalar(baseline_cost_totals.get(building_name), 0.0)
            delta_total = control_total - baseline_total
            put('building', building_name, v2('building', 'cost', 'total', 'control', None, 'eur'), control_total)
            put('building', building_name, v2('building', 'cost', 'total', 'baseline', None, 'eur'), baseline_total)
            put('building', building_name, v2('building', 'cost', 'total', 'delta', None, 'eur'), delta_total)
            put('building', building_name, v2('building', 'cost', 'daily_average', 'control', None, 'eur'), self._daily_average(control_total, simulated_days))
            put('building', building_name, v2('building', 'cost', 'daily_average', 'baseline', None, 'eur'), self._daily_average(baseline_total, simulated_days))
            put('building', building_name, v2('building', 'cost', 'daily_average', 'delta', None, 'eur'), self._daily_average(delta_total, simulated_days))
            put('building', building_name, v2('building', 'cost', 'ratio_to_baseline', 'total', None, 'ratio'), self._safe_div(control_total, baseline_total))

        district_delta_total = district_control_total - district_baseline_total
        put('district', 'District', v2('district', 'cost', 'total', 'control', None, 'eur'), district_control_total)
        put('district', 'District', v2('district', 'cost', 'total', 'baseline', None, 'eur'), district_baseline_total)
        put('district', 'District', v2('district', 'cost', 'total', 'delta', None, 'eur'), district_delta_total)
        put('district', 'District', v2('district', 'cost', 'daily_average', 'control', None, 'eur'), self._daily_average(district_control_total, simulated_days))
        put('district', 'District', v2('district', 'cost', 'daily_average', 'baseline', None, 'eur'), self._daily_average(district_baseline_total, simulated_days))
        put('district', 'District', v2('district', 'cost', 'daily_average', 'delta', None, 'eur'), self._daily_average(district_delta_total, simulated_days))
        put('district', 'District', v2('district', 'cost', 'ratio_to_baseline', 'total', None, 'ratio'), self._safe_div(district_control_total, district_baseline_total))

        # Equity metrics are recomputed from cost totals for consistency.
        equity_valid_benefits: Dict[str, float] = {}
        groups = {building.name: getattr(building, 'equity_group', None) for building in env.buildings}
        for building_name in building_names:
            benefit = self._equity_relative_benefit_percent(
                control_cost_totals.get(building_name, 0.0),
                baseline_cost_totals.get(building_name, 0.0),
            )
            put(
                'building',
                building_name,
                v2('building', 'equity', 'benefit', 'relative', None, 'percent'),
                benefit,
            )
            if benefit is not None:
                equity_valid_benefits[building_name] = float(benefit)

        equity_distribution = self._equity_distribution_metrics(np.array(list(equity_valid_benefits.values()), dtype='float64'))
        non_negative_benefits = {name: max(value, 0.0) for name, value in equity_valid_benefits.items()}
        has_complete_manual_groups = all(groups.get(name) in {'asset_rich', 'asset_poor'} for name in building_names)
        equity_bpr = self._equity_bpr(non_negative_benefits, groups) if has_complete_manual_groups else None
        put('district', 'District', v2('district', 'equity', 'distribution', 'gini_benefit', None, 'ratio'), equity_distribution['equity_gini_benefit'])
        put('district', 'District', v2('district', 'equity', 'distribution', 'top20_benefit', None, 'ratio'), equity_distribution['equity_cr20_benefit'])
        put('district', 'District', v2('district', 'equity', 'distribution', 'losers', None, 'percent'), equity_distribution['equity_losers_percent'])
        put('district', 'District', v2('district', 'equity', 'distribution', 'bpr_asset_poor_over_rich', None, 'ratio'), equity_bpr)

        output = pd.DataFrame(list(records.values()))
        if output.empty:
            return pd.DataFrame(columns=['cost_function', 'value', 'name', 'level'])

        if getattr(env, 'topology_mode', 'static') == 'dynamic':
            lifecycle = getattr(env, 'topology_member_lifecycle', {}) or {}
            existing_buildings = set(output[output['level'] == 'building']['name'].astype(str).tolist())
            lifecycle_rows = []

            for member_id in lifecycle.keys():
                if member_id not in existing_buildings:
                    lifecycle_rows.append(
                        {
                            'cost_function': 'topology_lifecycle_presence',
                            'value': np.nan,
                            'name': member_id,
                            'level': 'building',
                        }
                    )

            if lifecycle_rows:
                output = pd.concat([output, pd.DataFrame(lifecycle_rows)], ignore_index=True)

            output['topology_born_at'] = np.nan
            output['topology_removed_at'] = np.nan
            output['topology_active'] = np.nan
            for member_id, state in lifecycle.items():
                mask = (output['level'] == 'building') & (output['name'] == member_id)
                output.loc[mask, 'topology_born_at'] = state.get('born_at')
                output.loc[mask, 'topology_removed_at'] = state.get('removed_at')
                output.loc[mask, 'topology_active'] = 1.0 if state.get('active', False) else 0.0

        output = output.sort_values(['level', 'name', 'cost_function']).reset_index(drop=True)
        return output

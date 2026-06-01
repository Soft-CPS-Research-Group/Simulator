from __future__ import annotations

from copy import deepcopy
import time
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Union

import numpy as np

from citylearn.base import Environment
from citylearn.data import ChargerSimulation

if TYPE_CHECKING:
    from citylearn.citylearn import CityLearnEnv


class CityLearnRuntimeService:
    """Internal runtime orchestration for `CityLearnEnv`."""

    def __init__(self, env: "CityLearnEnv"):
        self.env = env

    @staticmethod
    def _ev_unconnected_drift_std(seconds_per_time_step: float) -> float:
        """Return per-step drift std scaled by physical step duration."""

        seconds = max(float(seconds_per_time_step), 1.0)
        step_hours = seconds / 3600.0
        return 0.2 * np.sqrt(step_hours)

    def step(self, actions):
        """Apply actions, update env variables/reward, then advance time."""

        return self._step_once(actions=actions)

    def step_many(
        self,
        action,
        repeat_steps: int = 1,
        stop_on_done: bool = True,
        return_substeps: bool = False,
    ):
        """Apply one action over multiple internal simulator steps."""

        env = self.env

        try:
            repeat_steps = int(repeat_steps)
        except (TypeError, ValueError) as exc:
            raise ValueError('repeat_steps must be an integer >= 1.') from exc

        if repeat_steps < 1:
            raise ValueError('repeat_steps must be >= 1.')

        if env.terminated or env.truncated:
            raise RuntimeError('Episode has already terminated/truncated. Call reset() before calling step_many() again.')

        if not stop_on_done:
            # CityLearn cannot advance after a terminal transition; keep the
            # argument for API compatibility but still stop on done.
            stop_on_done = True

        debug_timing = bool(getattr(env, 'debug_timing', False))
        macro_start = time.perf_counter() if debug_timing else 0.0
        parse_time = 0.0
        static_action_layout = getattr(env, 'topology_mode', 'static') != 'dynamic'
        parsed_action = None

        if static_action_layout:
            parse_start = time.perf_counter() if debug_timing else 0.0
            parsed_action = self.parse_actions(action)
            if debug_timing:
                parse_time = time.perf_counter() - parse_start

        executed_steps = 0
        reward_total: Optional[np.ndarray] = None
        substep_rewards = [] if return_substeps else None
        substep_infos = [] if return_substeps else None
        substep_actions_applied = [] if return_substeps else None
        final_observations = None
        final_info = {}

        for _ in range(repeat_steps):
            step_parsed_action = parsed_action
            if step_parsed_action is None:
                parse_start = time.perf_counter() if debug_timing else 0.0
                step_parsed_action = self.parse_actions(action)
                if debug_timing:
                    parse_time += time.perf_counter() - parse_start

            collect_observations = bool(return_substeps or getattr(env, 'check_observation_limits', False))
            observations, reward, terminated, truncated, info = self._step_once(
                parsed_actions=step_parsed_action,
                collect_observations=collect_observations,
                collect_info=return_substeps,
            )

            executed_steps += 1
            reward_total = self._accumulate_reward(reward_total, reward)

            if return_substeps:
                substep_rewards.append(reward)
                substep_infos.append(info)
                substep_actions_applied.append(deepcopy(step_parsed_action))
                final_info = info

            if observations is not None:
                final_observations = observations

            if (terminated or truncated) and stop_on_done:
                break

        if final_observations is None:
            final_observations = env.observations

        info = dict(final_info) if return_substeps else dict(env.get_info())
        seconds_per_time_step = float(env.seconds_per_time_step)
        macro_seconds = executed_steps * seconds_per_time_step
        info.update({
            'executed_steps': executed_steps,
            'seconds_per_time_step': env.seconds_per_time_step,
            'macro_seconds': macro_seconds,
        })

        if return_substeps:
            info['substep_rewards'] = substep_rewards
            info['substep_infos'] = substep_infos
            info['substep_actions_applied'] = substep_actions_applied

        if debug_timing:
            info['step_many_parse_actions_time'] = parse_time
            info['step_many_total_time'] = time.perf_counter() - macro_start

        return final_observations, self._reward_total_to_list(reward_total), env.terminated, env.truncated, info

    @staticmethod
    def _reward_to_array(reward) -> np.ndarray:
        reward_array = np.asarray(reward, dtype='float64')
        if reward_array.ndim == 0:
            return reward_array.reshape(1)

        return reward_array.reshape(-1)

    def _accumulate_reward(self, total: Optional[np.ndarray], reward) -> np.ndarray:
        reward_array = self._reward_to_array(reward)
        if total is None:
            return reward_array.copy()

        if total.shape != reward_array.shape:
            raise RuntimeError(
                'Reward vector shape changed during step_many; cannot aggregate macro-step rewards safely.'
            )

        return total + reward_array

    @staticmethod
    def _reward_total_to_list(total: Optional[np.ndarray]) -> List[float]:
        if total is None:
            return []

        return [float(value) for value in total.tolist()]

    def _step_once(
        self,
        actions=None,
        *,
        parsed_actions: Optional[List[Mapping[str, float]]] = None,
        collect_observations: bool = True,
        collect_info: bool = True,
    ):
        """Run one simulator transition with optional final observation/info assembly."""

        env = self.env
        debug_timing = bool(getattr(env, 'debug_timing', False)) and bool(collect_info)
        step_start = time.perf_counter() if debug_timing else 0.0
        timings = {} if debug_timing else None

        if env.terminated or env.truncated:
            raise RuntimeError('Episode has already terminated/truncated. Call reset() before calling step() again.')

        env._observations_cache = None
        env._observations_cache_time_step = -1
        timer_start = time.perf_counter() if debug_timing else 0.0
        actions = self.parse_actions(actions) if parsed_actions is None else parsed_actions
        if debug_timing:
            timings['parse_actions_time'] = time.perf_counter() - timer_start

        timer_start = time.perf_counter() if debug_timing else 0.0
        if getattr(getattr(env, '_robustness_service', None), 'enabled', False):
            actions = env._robustness_service.apply_actions(actions)
        if debug_timing:
            timings['robustness_action_time'] = time.perf_counter() - timer_start

        timer_start = time.perf_counter() if debug_timing else 0.0
        clear_step_electric_loads_time = 0.0
        building_apply_action_times = []
        for building, building_actions in zip(env.buildings, actions):
            if int(env.time_step) == 0:
                clear_start = time.perf_counter() if debug_timing else 0.0
                building.clear_step_electric_loads_for_action()
                if debug_timing:
                    clear_step_electric_loads_time += time.perf_counter() - clear_start
            building_start = time.perf_counter() if debug_timing else 0.0
            building.apply_actions(**building_actions)
            if debug_timing:
                building_apply_action_times.append(time.perf_counter() - building_start)
        if debug_timing:
            timings['apply_actions_time'] = time.perf_counter() - timer_start
            timings['clear_step_electric_loads_time'] = clear_step_electric_loads_time
            timings['building_apply_actions_time'] = float(sum(building_apply_action_times))
            timings['building_apply_actions_mean_time'] = (
                float(np.mean(building_apply_action_times)) if building_apply_action_times else 0.0
            )
            timings['building_apply_actions_max_time'] = (
                float(np.max(building_apply_action_times)) if building_apply_action_times else 0.0
            )

        timer_start = time.perf_counter() if debug_timing else 0.0
        self.update_variables()
        if debug_timing:
            timings['update_variables_time'] = time.perf_counter() - timer_start
            timings.update(getattr(env, '_last_update_variables_debug_timing', {}))

        timer_start = time.perf_counter() if debug_timing else 0.0
        if getattr(getattr(env, '_demand_response_service', None), 'enabled', False):
            env._demand_response_service.settle_current_time_step()
        if debug_timing:
            timings['demand_response_settlement_time'] = time.perf_counter() - timer_start

        if bool(getattr(env, 'physics_invariant_checks', False)):
            timer_start = time.perf_counter() if debug_timing else 0.0
            env._physics_invariant_service.assert_step_invariants(int(env.time_step))
            if debug_timing:
                timings['physics_invariants_time'] = time.perf_counter() - timer_start
        elif debug_timing:
            timings['physics_invariants_time'] = 0.0

        timer_start = time.perf_counter() if debug_timing else 0.0
        reward_observation_names = self._reward_observation_names()
        if reward_observation_names is None:
            reward_observations = [
                b.observations(include_all=True, normalize=False, periodic_normalization=False) for b in env.buildings
            ]
        else:
            reward_observations = [
                b.observations(
                    include_all=True,
                    normalize=False,
                    periodic_normalization=False,
                    observation_names=reward_observation_names,
                )
                for b in env.buildings
            ]
        if debug_timing:
            reward_observations_time = time.perf_counter() - timer_start
            timings['reward_observations_time'] = reward_observations_time
            timings['building_observations_retrieval_time'] = reward_observations_time

        timer_start = time.perf_counter() if debug_timing else 0.0
        reward = env.reward_function.calculate(observations=reward_observations)
        env.rewards.append(reward)
        if debug_timing:
            timings['reward_calculation_time'] = time.perf_counter() - timer_start

        timer_start = time.perf_counter() if debug_timing else 0.0
        partial_render_time = self.next_time_step()
        if debug_timing:
            timings['next_time_step_time'] = time.perf_counter() - timer_start
            timings.update(getattr(env, '_last_next_time_step_debug_timing', {}))

        topology_changed = False
        timer_start = time.perf_counter() if debug_timing else 0.0
        if getattr(env, 'topology_mode', 'static') == 'dynamic':
            topology_changed = env._topology_service.apply_events_for_time_step(int(env.time_step))
            if topology_changed:
                # Re-associate only when topology actually changed (e.g., added member/charger).
                # Otherwise this would duplicate the association already done in `next_time_step()`.
                self.associate_chargers_to_electric_vehicles()
                env._refresh_action_cache()
                env._entity_service.invalidate()
                env.reward_function.env_metadata = env.get_metadata()
        if debug_timing:
            timings['topology_time'] = time.perf_counter() - timer_start

        end_export_time = 0.0
        final_kpi_export_time = 0.0
        timer_start = time.perf_counter() if debug_timing else 0.0
        env._maybe_log_periodic_metrics()
        if debug_timing:
            timings['periodic_metrics_time'] = time.perf_counter() - timer_start

        if env.terminated:
            timer_start = time.perf_counter() if debug_timing else 0.0
            reward_history = env.rewards[1:]
            reward_vectors: List[np.ndarray] = []
            max_len = 1

            for reward_item in reward_history:
                arr = np.asarray(reward_item, dtype='float32')
                vec = arr.reshape(1) if arr.ndim == 0 else arr.reshape(-1)
                reward_vectors.append(vec)
                max_len = max(max_len, int(vec.size))

            rewards = np.full((len(reward_vectors), max_len), np.nan, dtype='float32')
            for i, vec in enumerate(reward_vectors):
                rewards[i, :vec.size] = vec

            valid = ~np.isnan(rewards)
            valid_count = valid.sum(axis=0).astype(np.float32)
            safe_min = np.where(np.any(valid, axis=0), np.nanmin(rewards, axis=0), np.nan)
            safe_max = np.where(np.any(valid, axis=0), np.nanmax(rewards, axis=0), np.nan)
            safe_sum = np.nansum(rewards, axis=0)
            safe_mean = np.divide(
                safe_sum,
                valid_count,
                out=np.full_like(safe_sum, np.nan, dtype='float32'),
                where=valid_count > 0,
            )

            def _to_python(value: np.ndarray):
                scalar = value.reshape(-1)
                if scalar.size == 1:
                    return float(scalar[0])
                return scalar.tolist()

            env.episode_rewards.append({
                'min': _to_python(safe_min),
                'max': _to_python(safe_max),
                'sum': _to_python(safe_sum),
                'mean': _to_python(safe_mean),
            })
            if debug_timing:
                timings['terminal_reward_summary_time'] = time.perf_counter() - timer_start

            if env.render_mode == 'end' and env.render_enabled:
                final_index = min(env.time_steps - 1, env.time_step - 1) if env.time_step > 0 else 0
                if debug_timing:
                    export_start = time.perf_counter()
                env._export_episode_render_data(final_index)
                if debug_timing:
                    end_export_time = time.perf_counter() - export_start
            elif env.render_mode == 'during' and env.render_enabled:
                if debug_timing:
                    export_start = time.perf_counter()
                env._flush_render_buffer()
                if debug_timing:
                    end_export_time = time.perf_counter() - export_start

            if (
                env.export_kpis_on_episode_end
                and env._should_export_current_episode()
                and not env._final_kpis_exported
            ):
                if debug_timing:
                    final_kpi_export_start = time.perf_counter()
                env.export_final_kpis()
                if debug_timing:
                    final_kpi_export_time = time.perf_counter() - final_kpi_export_start
        elif debug_timing:
            timings['terminal_reward_summary_time'] = 0.0

        timer_start = time.perf_counter() if debug_timing else 0.0
        next_observations = env.observations if collect_observations else None
        if debug_timing:
            timings['next_observations_time'] = time.perf_counter() - timer_start
            if collect_observations and getattr(env, 'interface', 'flat') == 'entity':
                timings.update(getattr(env, '_last_entity_observation_debug_timing', {}))

        timer_start = time.perf_counter() if debug_timing else 0.0
        info = dict(env.get_info()) if collect_info else {}
        if debug_timing:
            timings['get_info_time'] = time.perf_counter() - timer_start
            info['partial_render_time'] = partial_render_time
            info['end_export_time'] = end_export_time
            info['final_kpi_export_time'] = final_kpi_export_time
            info['terminal_export_time'] = end_export_time + final_kpi_export_time
            info.update(timings)
            info['step_total_time'] = time.perf_counter() - step_start

        return next_observations, reward, env.terminated, env.truncated, info

    def _reward_observation_names(self):
        for attribute_name in ('required_observation_names', 'required_observations'):
            names = getattr(self.env.reward_function, attribute_name, None)
            if callable(names):
                names = names()
            names = self._normalize_reward_observation_names(names)
            if names is not None:
                return names

        provider = getattr(self.env.reward_function, 'get_required_observation_names', None)
        if not callable(provider):
            return None

        return self._normalize_reward_observation_names(provider())

    @staticmethod
    def _normalize_reward_observation_names(names):
        if names is None:
            return None

        if isinstance(names, str):
            return (names,)

        if isinstance(names, Mapping):
            merged = []
            for value in names.values():
                if value is None:
                    return None
                if isinstance(value, str):
                    merged.append(value)
                else:
                    merged.extend(value)
            names = merged

        return tuple(dict.fromkeys(names))

    def step_without_feedback(self, actions):
        """Apply actions and advance state without reward, observation or info output.

        This is used by internal sidecar rollouts whose policy reads state directly
        from the environment and whose outputs are only evaluated after rollout.
        """

        env = self.env

        if env.terminated or env.truncated:
            raise RuntimeError('Episode has already terminated/truncated. Call reset() before calling step() again.')

        env._observations_cache = None
        env._observations_cache_time_step = -1
        actions = self.parse_actions(actions)
        if getattr(getattr(env, '_robustness_service', None), 'enabled', False):
            actions = env._robustness_service.apply_actions(actions)

        for building, building_actions in zip(env.buildings, actions):
            if int(env.time_step) == 0:
                building.clear_step_electric_loads_for_action()
            building.apply_actions(**building_actions)

        self.update_variables()
        if getattr(getattr(env, '_demand_response_service', None), 'enabled', False):
            env._demand_response_service.settle_current_time_step()
        if bool(getattr(env, 'physics_invariant_checks', False)):
            env._physics_invariant_service.assert_step_invariants(int(env.time_step))

        self.next_time_step()

        if getattr(env, 'topology_mode', 'static') == 'dynamic':
            topology_changed = env._topology_service.apply_events_for_time_step(int(env.time_step))
            if topology_changed:
                self.associate_chargers_to_electric_vehicles()
                env._refresh_action_cache()
                env._entity_service.invalidate()
                env.reward_function.env_metadata = env.get_metadata()

        return env.terminated, env.truncated

    def parse_actions(self, actions) -> List[Mapping[str, float]]:
        """Return mapping of action name to action value for each building."""

        env = self.env

        if getattr(env, 'interface', 'flat') == 'entity':
            return env._entity_service.parse_actions(actions)

        building_actions = []
        cache = getattr(env, '_active_actions_cache', None)
        cached_expected = getattr(env, '_expected_central_action_count', None)
        current_actions = [list(b.active_actions) for b in env.buildings]
        current_expected = sum(len(v) for v in current_actions)

        if cache is None or cached_expected != current_expected or cache != current_actions:
            env._refresh_action_cache()

        def _is_scalar(value: Any) -> bool:
            return bool(np.isscalar(value))

        def _to_vector(value: Any, *, context: str) -> List[float]:
            if isinstance(value, np.ndarray):
                array = np.asarray(value)
                if array.ndim == 1:
                    return array.tolist()
                if array.ndim == 2 and array.shape[0] == 1:
                    return array[0].tolist()
                raise AssertionError(f'{context} must be a 1D action vector.')

            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    return []
                if all(_is_scalar(v) for v in value):
                    return list(value)
                if len(value) == 1:
                    inner = value[0]
                    if isinstance(inner, (list, tuple, np.ndarray)):
                        return _to_vector(inner, context=context)
                raise AssertionError(f'{context} must be a 1D action vector.')

            raise AssertionError(f'{context} must be a 1D action vector.')

        if env.central_agent:
            actions = _to_vector(actions, context='central_agent actions')
            number_of_actions = len(actions)
            expected_number_of_actions = env._expected_central_action_count
            assert number_of_actions == expected_number_of_actions, \
                f'Expected {expected_number_of_actions} actions but {number_of_actions} were parsed to env.step.'

            for building in env.buildings:
                size = building.action_space.shape[0]
                building_actions.append(actions[0:size])
                actions = actions[size:]

        else:
            if isinstance(actions, np.ndarray):
                array = np.asarray(actions)
                if array.ndim == 2:
                    building_actions = [row.tolist() for row in array]
                else:
                    raise AssertionError(
                        'Expected one action vector per building when central_agent=False.'
                    )
            elif isinstance(actions, (list, tuple)):
                building_actions = []
                for idx, action_vector in enumerate(actions):
                    if isinstance(action_vector, (list, tuple, np.ndarray)):
                        building_actions.append(_to_vector(action_vector, context=f'building action vector at index {idx}'))
                    else:
                        raise AssertionError(
                            'Expected one action vector per building when central_agent=False.'
                        )
            else:
                raise AssertionError('Expected one action vector per building when central_agent=False.')

            number_of_building_actions = len(building_actions)
            expected_building_actions = len(env.buildings)
            assert number_of_building_actions == expected_building_actions, \
                f'Expected {expected_building_actions} building action vectors but {number_of_building_actions} were provided.'

        for building, building_action in zip(env.buildings, building_actions):
            number_of_actions = len(building_action)
            expected_number_of_actions = building.action_space.shape[0]
            assert number_of_actions == expected_number_of_actions, \
                f'Expected {expected_number_of_actions} for {building.name} but {number_of_actions} actions were provided.'

        active_actions = env._active_actions_cache
        parsed_actions = []

        for i, _building in enumerate(env.buildings):
            action_dict = {}
            electric_vehicle_actions = {}
            deferrable_appliance_actions = {}

            for action_name, action in zip(active_actions[i], building_actions[i]):
                if 'electric_vehicle_storage' in action_name:
                    charger_id = action_name.replace('electric_vehicle_storage_', '')
                    electric_vehicle_actions[charger_id] = action
                elif action_name.startswith('deferrable_appliance_'):
                    deferrable_appliance_actions[action_name] = action
                else:
                    action_dict[f'{action_name}_action'] = action

            if electric_vehicle_actions:
                action_dict['electric_vehicle_storage_actions'] = electric_vehicle_actions

            if deferrable_appliance_actions:
                action_dict['deferrable_appliance_actions'] = deferrable_appliance_actions

            parsed_actions.append(action_dict)

        return parsed_actions

    def next_time_step(self):
        r"""Advance all buildings to next `time_step`."""

        env = self.env
        debug_timing = bool(getattr(env, 'debug_timing', False))
        timings = {} if debug_timing else None
        current_step = int(env.time_step)
        last_action_step = max(env.time_steps - 2, 0)
        reached_terminal_transition = current_step >= last_action_step

        partial_render_time = 0.0
        if getattr(env, 'render_enabled', False):
            if env.render_mode == 'during':
                if debug_timing:
                    render_start = time.perf_counter()
                    env.render()
                    partial_render_time = time.perf_counter() - render_start
                else:
                    env.render()
        if debug_timing:
            timings['next_time_step_render_time'] = partial_render_time

        if not reached_terminal_transition:
            timer_start = time.perf_counter() if debug_timing else 0.0
            for building in env.buildings:
                building.next_time_step()
            if debug_timing:
                timings['next_time_step_buildings_time'] = time.perf_counter() - timer_start

            timer_start = time.perf_counter() if debug_timing else 0.0
            for electric_vehicle in env.electric_vehicles:
                electric_vehicle.next_time_step()
            if debug_timing:
                timings['next_time_step_electric_vehicles_time'] = time.perf_counter() - timer_start
        elif debug_timing:
            timings['next_time_step_buildings_time'] = 0.0
            timings['next_time_step_electric_vehicles_time'] = 0.0

        timer_start = time.perf_counter() if debug_timing else 0.0
        Environment.next_time_step(env)
        if debug_timing:
            timings['next_time_step_environment_time'] = time.perf_counter() - timer_start

        if not reached_terminal_transition:
            timer_start = time.perf_counter() if debug_timing else 0.0
            self.simulate_unconnected_ev_soc()
            if debug_timing:
                timings['next_time_step_unconnected_ev_soc_time'] = time.perf_counter() - timer_start

            timer_start = time.perf_counter() if debug_timing else 0.0
            self.associate_chargers_to_electric_vehicles()
            if debug_timing:
                timings['next_time_step_associate_chargers_time'] = time.perf_counter() - timer_start
        elif debug_timing:
            timings['next_time_step_unconnected_ev_soc_time'] = 0.0
            timings['next_time_step_associate_chargers_time'] = 0.0

        if debug_timing:
            env._last_next_time_step_debug_timing = timings

        return partial_render_time

    def associate_chargers_to_electric_vehicles(self):
        r"""Associate charger to its corresponding EV based on charger simulation state."""

        env = self.env
        ev_by_name = {ev.name: ev for ev in env.electric_vehicles}

        def _resolve_arrival_soc(
            simulation: ChargerSimulation,
            step: int,
            prev_state: float,
            prev_id: Union[str, None],
            ev_identifier: str,
        ) -> Union[float, None]:
            current_soc = getattr(simulation, 'electric_vehicle_current_soc', None)
            if current_soc is not None and 0 <= step < len(current_soc):
                current_value = current_soc[step]
                if isinstance(current_value, (float, np.floating)) and not np.isnan(current_value) and 0.0 <= current_value <= 1.0:
                    return float(current_value)

            candidate_index = None

            if prev_state in (2, 3) and step > 0:
                if isinstance(prev_id, str) and prev_id.strip() not in {'', 'nan'} and prev_id != ev_identifier:
                    raise ValueError(
                        f"Charger dataset EV mismatch: expected '{ev_identifier}' but found '{prev_id}' at time step {step - 1}."
                    )
                candidate_index = step - 1

            elif 0 <= step < len(simulation.electric_vehicle_estimated_soc_arrival):
                candidate_index = step

            soc_value = None

            if candidate_index is not None and 0 <= candidate_index < len(simulation.electric_vehicle_estimated_soc_arrival):
                candidate = simulation.electric_vehicle_estimated_soc_arrival[candidate_index]
                if isinstance(candidate, (float, np.floating)) and not np.isnan(candidate) and 0.0 <= candidate <= 1.0:
                    soc_value = float(candidate)

            return soc_value

        for building in env.buildings:
            if building.electric_vehicle_chargers is None:
                continue

            for charger in building.electric_vehicle_chargers:
                sim = charger.charger_simulation
                state = sim.electric_vehicle_charger_state[env.time_step]

                if np.isnan(state) or state not in [1, 2]:
                    continue

                ev_id = sim.electric_vehicle_id[env.time_step]
                prev_state = np.nan
                prev_ev_id = None
                if env.time_step > 0:
                    idx = env.time_step - 1
                    if idx < len(sim.electric_vehicle_charger_state):
                        prev_state = sim.electric_vehicle_charger_state[idx]
                    if idx < len(sim.electric_vehicle_id):
                        prev_ev_id = sim.electric_vehicle_id[idx]

                if isinstance(ev_id, str) and ev_id.strip() not in ['', 'nan']:
                    ev = ev_by_name.get(ev_id)
                    if ev is None:
                        continue

                    if state == 1:
                        # Idempotent behavior: the method may be called again in the same
                        # time step when topology changed and action/observation layouts are refreshed.
                        just_connected = False
                        if charger.connected_electric_vehicle is None:
                            charger.plug_car(ev)
                            just_connected = True
                        elif charger.connected_electric_vehicle is not ev:
                            raise ValueError(
                                f"Charger '{charger.charger_id}' already connected to "
                                f"'{charger.connected_electric_vehicle.name}' but dataset requires '{ev.name}' "
                                f"at time step {env.time_step}."
                            )
                        is_new_connection = (
                            just_connected and (
                                prev_state != 1
                                or not isinstance(prev_ev_id, str)
                                or prev_ev_id != ev_id
                            )
                        )
                        if is_new_connection:
                            soc_value = _resolve_arrival_soc(sim, env.time_step, prev_state, prev_ev_id, ev_id)
                            if soc_value is not None:
                                ev.battery.force_set_soc(soc_value)
                    elif state == 2:
                        charger.associate_incoming_car(ev)

    def simulate_unconnected_ev_soc(self):
        """Simulate SOC changes for EVs that are not under charger control at t+1."""

        env = self.env
        random_state = getattr(env, '_ev_drift_random_state', None)

        if random_state is None:
            episode_index = int(getattr(getattr(env, 'episode_tracker', None), 'episode', 0))
            random_state = np.random.RandomState(int(env.random_seed) + episode_index)
            env._ev_drift_random_state = random_state

        t = env.time_step
        if t + 1 >= env.episode_tracker.episode_time_steps:
            return

        charger_event_by_ev_id = {}
        for building in env.buildings:
            for charger in building.electric_vehicle_chargers or []:
                sim: ChargerSimulation = charger.charger_simulation

                curr_id = sim.electric_vehicle_id[t] if t < len(sim.electric_vehicle_id) else ''
                next_id = sim.electric_vehicle_id[t + 1] if t + 1 < len(sim.electric_vehicle_id) else ''
                curr_state = sim.electric_vehicle_charger_state[t] if t < len(sim.electric_vehicle_charger_state) else np.nan
                next_state = sim.electric_vehicle_charger_state[t + 1] if t + 1 < len(sim.electric_vehicle_charger_state) else np.nan

                if isinstance(curr_id, str) and curr_id == next_id and curr_state == 1:
                    charger_event_by_ev_id.setdefault(curr_id, ('connected', None))
                    continue

                if isinstance(curr_id, str) and curr_state == 1:
                    charger_event_by_ev_id.setdefault(curr_id, ('connected', None))

                is_connecting = (
                    isinstance(next_id, str)
                    and next_id.strip() not in {'', 'nan'}
                    and next_state == 1
                    and curr_state != 1
                )
                if not is_connecting:
                    continue

                is_incoming = isinstance(curr_id, str) and curr_id == next_id and curr_state == 2
                soc_index = t if is_incoming else t + 1
                soc = (
                    sim.electric_vehicle_estimated_soc_arrival[soc_index]
                    if soc_index < len(sim.electric_vehicle_estimated_soc_arrival)
                    else np.nan
                )
                charger_event_by_ev_id.setdefault(next_id, ('connecting', soc))

        for ev in env.electric_vehicles:
            ev_id = ev.name
            event = charger_event_by_ev_id.get(ev_id)
            if event is not None:
                event_type, soc = event
                try:
                    soc_value = float(soc)
                except (TypeError, ValueError):
                    soc_value = np.nan
                if event_type == 'connecting' and 0 <= soc_value <= 1:
                    ev.battery.force_set_soc(soc_value)
                continue

            if t > 0:
                last_soc = ev.battery.soc[t - 1]
                drift_std = self._ev_unconnected_drift_std(env.seconds_per_time_step)
                variability = np.clip(random_state.normal(1.0, drift_std), 0.6, 1.4)
                # Exogenous away-from-home movement: driving can reduce SOC, offsite charging can raise it.
                ev.battery.force_set_soc(float(np.clip(last_soc * variability, 0.0, 1.0)))

    def update_variables(self):
        """Update district aggregate series from current building states."""

        env = self.env
        debug_timing = bool(getattr(env, 'debug_timing', False))
        timings = {} if debug_timing else None

        timer_start = time.perf_counter() if debug_timing else 0.0
        building_update_times = []
        for building in env.buildings:
            building_start = time.perf_counter() if debug_timing else 0.0
            building.update_variables()
            if debug_timing:
                building_update_times.append(time.perf_counter() - building_start)
        if debug_timing:
            timings['update_variables_buildings_time'] = time.perf_counter() - timer_start
            timings['building_update_variables_mean_time'] = (
                float(np.mean(building_update_times)) if building_update_times else 0.0
            )
            timings['building_update_variables_max_time'] = (
                float(np.max(building_update_times)) if building_update_times else 0.0
            )

        timer_start = time.perf_counter() if debug_timing else 0.0
        if getattr(env, 'community_market_enabled', False):
            self._apply_community_market_settlement()
        if debug_timing:
            timings['community_market_settlement_time'] = time.perf_counter() - timer_start

        def _set_or_append(lst, value):
            if len(lst) == env.time_step:
                lst.append(value)
            elif len(lst) == env.time_step + 1:
                lst[env.time_step] = value
            else:
                del lst[env.time_step + 1:]
                if len(lst) < env.time_step:
                    lst.extend([0.0] * (env.time_step - len(lst)))
                lst.append(value)

        timer_start = time.perf_counter() if debug_timing else 0.0
        total = sum(building.net_electricity_consumption[env.time_step] for building in env.buildings)
        _set_or_append(env.net_electricity_consumption, total)

        total_cost = sum(building.net_electricity_consumption_cost[env.time_step] for building in env.buildings)
        _set_or_append(env.net_electricity_consumption_cost, total_cost)

        total_emission = sum(building.net_electricity_consumption_emission[env.time_step] for building in env.buildings)
        _set_or_append(env.net_electricity_consumption_emission, total_emission)
        if debug_timing:
            timings['district_aggregate_update_time'] = time.perf_counter() - timer_start
            env._last_update_variables_debug_timing = timings

    @staticmethod
    def _to_scalar(value, default: float = 0.0) -> float:
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return float(default)

        if not np.isfinite(scalar):
            return float(default)

        return scalar

    def _resolve_step_value(self, value, time_step: int, default: float = 0.0) -> float:
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                return float(default)
            index = min(max(time_step, 0), len(value) - 1)
            return self._to_scalar(value[index], default)

        return self._to_scalar(value, default)

    @staticmethod
    def _allocate_equal_share_import(imports: np.ndarray, traded_kwh: float) -> np.ndarray:
        """Allocate local traded energy equally among importers with demand caps."""

        allocations = np.zeros_like(imports, dtype='float64')
        remaining = max(float(traded_kwh), 0.0)
        eps = 1e-9

        while remaining > eps:
            needs = imports - allocations
            active = needs > eps
            active_count = int(np.count_nonzero(active))

            if active_count == 0:
                break

            share = remaining / active_count
            granted = np.minimum(share, needs[active])
            granted_total = float(granted.sum())

            if granted_total <= eps:
                break

            allocations[active] += granted
            remaining -= granted_total

        return allocations

    @staticmethod
    def _allocate_weighted_share_import(imports: np.ndarray, traded_kwh: float, weights: np.ndarray) -> np.ndarray:
        """Allocate local traded energy among importers using member weights with demand caps."""

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
            weight_sum = float(active_weights.sum())
            if weight_sum <= eps:
                granted = np.minimum(remaining / float(np.count_nonzero(active)), needs[active])
            else:
                granted = np.minimum(remaining * (active_weights / weight_sum), needs[active])

            granted_total = float(granted.sum())
            if granted_total <= eps:
                break

            allocations[active] += granted
            remaining -= granted_total

        return allocations

    def _apply_community_market_settlement(self):
        """Apply optional intracommunity settlement and override building costs for current step."""

        env = self.env
        t = env.time_step
        if len(env.buildings) == 0:
            return

        ratio = self._to_scalar(getattr(env, 'community_market_sell_ratio', 0.8), 0.8)
        ratio = min(max(ratio, 0.0), 1.0)

        net_values = np.array([self._to_scalar(building.net_electricity_consumption[t], 0.0) for building in env.buildings], dtype='float64')
        imports = np.clip(net_values, 0.0, None)
        exports = np.clip(-net_values, 0.0, None)

        total_import = float(imports.sum())
        total_export = float(exports.sum())
        traded_kwh = min(total_import, total_export)
        weights_cfg = getattr(env, 'community_market_import_member_weights', {}) or {}
        weights = np.array(
            [self._to_scalar(weights_cfg.get(building.name, 1.0), 1.0) for building in env.buildings],
            dtype='float64',
        )

        if total_import > 0.0 and traded_kwh > 0.0:
            local_import = self._allocate_weighted_share_import(imports, traded_kwh, weights)
        else:
            local_import = np.zeros_like(imports, dtype='float64')

        if total_export > 0.0:
            local_export = exports * (traded_kwh / total_export)
        else:
            local_export = np.zeros_like(exports, dtype='float64')

        grid_export_price_cfg = getattr(env, 'community_market_grid_export_price', 0.0)
        market_settlement = []

        for idx, building in enumerate(env.buildings):
            grid_import_price = self._to_scalar(building.pricing.electricity_pricing[t], 0.0)
            local_price = ratio * grid_import_price
            grid_export_price = self._resolve_step_value(grid_export_price_cfg, t, 0.0)
            counterfactual_legacy_cost = self._to_scalar(building.net_electricity_consumption_cost[t], 0.0)

            grid_import_remaining = max(imports[idx] - local_import[idx], 0.0)
            grid_export_remaining = max(exports[idx] - local_export[idx], 0.0)

            cost = (
                grid_import_remaining * grid_import_price
                + local_import[idx] * local_price
                - local_export[idx] * local_price
                - grid_export_remaining * grid_export_price
            )
            savings = counterfactual_legacy_cost - cost

            building.set_net_electricity_consumption_cost(cost, time_step=t)
            market_settlement.append(
                {
                    'building': building.name,
                    'local_import_kwh': float(local_import[idx]),
                    'local_export_kwh': float(local_export[idx]),
                    'grid_import_kwh': float(grid_import_remaining),
                    'grid_export_kwh': float(grid_export_remaining),
                    'local_price': float(local_price),
                    'grid_import_price': float(grid_import_price),
                    'grid_export_price': float(grid_export_price),
                    'counterfactual_cost_eur': float(counterfactual_legacy_cost),
                    'settled_cost_eur': float(cost),
                    'market_savings_eur': float(savings),
                }
            )

        env._last_community_market_settlement = market_settlement

        history = getattr(env, '_community_market_settlement_history', None)
        if history is not None:
            if len(history) == t:
                history.append(market_settlement)
            elif len(history) == t + 1:
                history[t] = market_settlement
            else:
                del history[t + 1:]
                if len(history) < t:
                    history.extend([[] for _ in range(t - len(history))])
                history.append(market_settlement)

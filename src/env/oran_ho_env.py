"""Gymnasium-compatible environment for handover-aware O-RAN orchestration."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.model.constraint import OrchestrationConstraintEvaluator
from src.model.causal_observation import (
    CAUSAL_RESOURCE_V2_OBSERVATION_FEATURE_MODE,
    LEGACY_OBSERVATION_FEATURE_MODE,
    RESOURCE_OBSERVATION_FEATURE_MODES,
    causal_resource_features_for_mode,
    causal_resource_features_numba,
    validate_observation_feature_mode,
)
from src.model.serial_numba import NUMBA_AVAILABLE, build_causal_static
from src.model.inference.post_process import ActionSpaceSpec, decode_flat_actions
from src.pipeline.schema import MockEpisode, PlacementAction

class OranHandoverEnv(gym.Env):
    """One episode exposes each time slot as a placement decision step."""

    metadata = {"render_modes": []}

    def __init__(self, episode: MockEpisode, project_config: dict[str, Any] | None = None,
                 observation_feature_mode: str = LEGACY_OBSERVATION_FEATURE_MODE) -> None:
        super().__init__()
        episode.validate()
        self.episode = episode
        self.evaluator = OrchestrationConstraintEvaluator(episode, project_config=project_config)
        self.observation_feature_mode = validate_observation_feature_mode(observation_feature_mode)
        self.time_index = 0
        self.previous_action: PlacementAction | None = None
        self.previous_resource_loads = None
        self._causal_numba_static = None
        spec = ActionSpaceSpec.from_topology(episode.config)
        self.action_space = spaces.MultiDiscrete([spec.num_actions_per_rh] * episode.config.num_rhs)
        obs_size = self._observation_at(0).shape[0]
        self.observation_space = spaces.Box(
            low=-np.finfo(np.float32).max,
            high=np.finfo(np.float32).max,
            shape=(obs_size,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        self.time_index = 0
        self.previous_action = None
        self.previous_resource_loads = None
        return self._observation_at(self.time_index), {"time_index": self.time_index}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        try:
            placement = decode_flat_actions(np.asarray(action, dtype=np.int64), self.episode.config)
            prior_action = self.previous_action
            report = self.evaluator.evaluate(self.time_index, placement, self.previous_action)
            reward = self.evaluator.reward_from_cost(report.total_cost)
            info = {
                "time_index": self.time_index,
                "feasible": report.feasible,
                "total_cost": report.total_cost,
                "reward_input_cost": self.evaluator.reward_input_from_cost(report.total_cost),
                "placement_cost": report.placement_cost,
                "compute_overload_cost": report.compute_overload_cost,
                "link_violation_cost": report.link_violation_cost,
                "handover_state_transfer_cost": report.handover_state_transfer_cost,
                "handover_sla_cost": report.handover_sla_cost,
                "handover_control_cost": report.handover_control_cost,
                "reconfiguration_cost": report.reconfiguration_cost,
                "bandwidth_violation_cost": report.bandwidth_violation_cost,
                "latency_violation_cost": report.latency_violation_cost,
                "violations": report.violations,
            }
            # ``evaluate`` performs the one shared joint-action accounting
            # calculation; causal observation keeps that exact result.
            self.previous_resource_loads = self.evaluator.last_resource_loads
            self.previous_action = placement
            info.update(self.resource_observation_diagnostics(prior_action, placement))
        except (TypeError, ValueError) as exc:
            reward = self.evaluator.reward_config.invalid_action_reward
            info = {
                "time_index": self.time_index,
                "feasible": False,
                "total_cost": float(self.episode.cost_weights.invalid_action),
                "reward_input_cost": float(self.episode.cost_weights.invalid_action),
                "placement_cost": 0.0,
                "compute_overload_cost": 0.0,
                "link_violation_cost": 0.0,
                "handover_state_transfer_cost": 0.0,
                "handover_sla_cost": 0.0,
                "handover_control_cost": 0.0,
                "reconfiguration_cost": 0.0,
                "bandwidth_violation_cost": 0.0,
                "latency_violation_cost": 0.0,
                "violations": {"invalid_action": 1.0},
                "error": str(exc),
            }
        self.time_index += 1
        terminated = self.time_index >= self.episode.config.horizon
        truncated = False
        obs_index = min(self.time_index, self.episode.config.horizon - 1)
        return self._observation_at(obs_index), float(reward), terminated, truncated, info

    def causal_rh_features(self) -> np.ndarray:
        """Features visible now, derived only from the previous successful step."""

        if self._causal_numba_static is not None:
            try:
                return causal_resource_features_numba(
                    self.episode,
                    self.previous_action,
                    self.previous_resource_loads,
                    self._causal_numba_static,
                    self.observation_feature_mode,
                )
            except Exception:
                if self.observation_feature_mode == CAUSAL_RESOURCE_V2_OBSERVATION_FEATURE_MODE:
                    raise
                # Compilation/runtime incompatibility disables only the optional
                # kernel; the scientific reference implementation remains live.
                self._causal_numba_static = None
        return causal_resource_features_for_mode(
            self.episode,
            self.previous_action,
            self.previous_resource_loads,
            self.observation_feature_mode,
        )

    def enable_serial_numba(self) -> bool:
        """Enable the selected backend only for causal-resource training."""

        if (
            self.observation_feature_mode not in RESOURCE_OBSERVATION_FEATURE_MODES
            or not NUMBA_AVAILABLE
        ):
            return False
        try:
            self._causal_numba_static = build_causal_static(self.episode)
        except Exception:
            self._causal_numba_static = None
            return False
        if not self.evaluator.enable_serial_numba():
            self._causal_numba_static = None
            return False
        return True

    def resource_observation_diagnostics(
        self, prior_action: PlacementAction | None, action: PlacementAction
    ) -> dict[str, float]:
        loads = self.previous_resource_loads
        if loads is None:
            return {}
        es_u = (loads.es_utilization if loads.es_utilization is not None
                else loads.es / self.episode.es_compute_capacity)
        rc_u = (loads.rc_utilization if loads.rc_utilization is not None
                else loads.rc / self.episode.rc_compute_capacity)
        mh_u = np.divide(
            loads.midhaul, self.episode.midhaul_bandwidth_mbps,
            out=np.zeros_like(loads.midhaul), where=self.episode.midhaul_bandwidth_mbps > 0,
        )
        changed = 0.0 if prior_action is None else float(np.mean(
            (prior_action.split_id != action.split_id)
            | (prior_action.es_id != action.es_id)
            | (prior_action.rc_id != action.rc_id)
        ))
        return {
            "previous_es_utilization_mean": float(es_u.mean()),
            "previous_es_utilization_max": float(es_u.max()),
            "previous_rc_utilization_mean": float(rc_u.mean()),
            "previous_rc_utilization_max": float(rc_u.max()),
            "previous_midhaul_utilization_mean": float(mh_u[self.episode.midhaul_mask].mean()),
            "previous_midhaul_utilization_max": float(mh_u[self.episode.midhaul_mask].max()),
            "overloaded_es_fraction": float(np.mean(es_u > 1.0)),
            "overloaded_rc_fraction": float(np.mean(rc_u > 1.0)),
            "overloaded_midhaul_fraction": float(np.mean(mh_u[self.episode.midhaul_mask] > 1.0)),
            "previous_action_change_rate": changed,
            "actor_action_change_rate": changed,
        }

    def _observation_at(self, time_index: int) -> np.ndarray:
        ep = self.episode
        rho = ep.rho_rt[time_index] if ep.rho_rt is not None else np.zeros(ep.config.num_rhs)
        features = [
            ep.num_ues_rt[time_index],
            ep.lambda_rt[time_index],
            ep.s_pdcp_rt[time_index],
            ep.s_rlc_rt[time_index],
            rho,
            ep.handover_matrix_t[time_index].reshape(-1),
            ep.handover_count_t[time_index].reshape(-1),
            ep.fronthaul_mask.astype(np.float64).reshape(-1),
            ep.fronthaul_latency_ms.reshape(-1),
            ep.fronthaul_bandwidth_mbps.reshape(-1),
            ep.fronthaul_cost.reshape(-1),
            ep.midhaul_mask.astype(np.float64).reshape(-1),
            ep.midhaul_latency_ms.reshape(-1),
            ep.midhaul_bandwidth_mbps.reshape(-1),
            ep.midhaul_cost.reshape(-1),
            ep.direct_rh_rc_mask.astype(np.float64).reshape(-1),
            ep.direct_rh_rc_latency_ms.reshape(-1),
            ep.direct_rh_rc_bandwidth_mbps.reshape(-1),
            ep.direct_rh_rc_cost.reshape(-1),
            ep.es_compute_capacity,
            ep.rc_compute_capacity,
        ]
        return np.concatenate(features).astype(np.float32)

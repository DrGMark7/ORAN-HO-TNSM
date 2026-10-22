"""Environment adapter for training on prebuilt flat mock-data datasets."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.env.oran_ho_env import OranHandoverEnv
from src.model.inference.post_process import decode_flat_actions
from src.pipeline.schema import MockEpisode


class PrebuiltEpisodeWindowEnv(OranHandoverEnv):
    """Expose a flat dataset horizon as fixed-length concatenated episodes."""

    def __init__(
        self,
        episode: MockEpisode,
        *,
        episode_length: int,
        num_episodes: int | None = None,
        project_config: dict[str, Any] | None = None,
        episode_order_mode: str = "sequential",
        episode_order_seed: int = 0,
        observation_feature_mode: str = "legacy_rh_only",
    ) -> None:
        if episode_length <= 0:
            raise ValueError("episode_length must be positive")
        super().__init__(episode, project_config=project_config,
                         observation_feature_mode=observation_feature_mode)
        self.flat_horizon = episode.config.horizon
        inferred_num_episodes, remainder = divmod(self.flat_horizon, episode_length)
        if num_episodes is None:
            if remainder != 0:
                raise ValueError(
                    "flat horizon is not divisible by episode_length; "
                    f"flat_horizon={self.flat_horizon} episode_length={episode_length}"
                )
            num_episodes = inferred_num_episodes
        if num_episodes <= 0:
            raise ValueError("num_episodes must be positive")
        if num_episodes * episode_length != self.flat_horizon:
            raise ValueError(
                "prebuilt dataset episode layout mismatch: "
                f"flat_horizon={self.flat_horizon} episode_length={episode_length} "
                f"num_episodes={num_episodes}"
            )
        self.episode_length = episode_length
        self.num_episodes = num_episodes
        self.current_episode_id = 0
        self.current_local_t = 0
        self.previous_action = None
        self.previous_resource_loads = None
        self._next_episode_id = 0
        if episode_order_mode not in {"sequential", "shuffle_each_cycle"}:
            raise ValueError("episode_order_mode must be sequential or shuffle_each_cycle")
        self.episode_order_mode = episode_order_mode
        self._episode_order_rng = np.random.default_rng(episode_order_seed)
        self._episode_order: list[int] = []

    def _next_training_episode(self) -> int:
        if self.episode_order_mode == "sequential":
            episode_id = self._next_episode_id
            self._next_episode_id = (episode_id + 1) % self.num_episodes
            return episode_id
        if not self._episode_order:
            self._episode_order = self._episode_order_rng.permutation(self.num_episodes).tolist()
        return self._episode_order.pop(0)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        episode_id = self._next_training_episode() if options is None else int(options.get("episode_id", self._next_episode_id))
        if episode_id < 0 or episode_id >= self.num_episodes:
            raise ValueError(f"episode_id must be in [0, {self.num_episodes - 1}]")
        self.current_episode_id = episode_id
        self.current_local_t = 0
        self.time_index = episode_id * self.episode_length
        profile_id = self.episode.topology_profile_id_for_episode(episode_id)
        self.episode.activate_topology_profile(profile_id)
        if options is not None:
            self._next_episode_id = (episode_id + 1) % self.num_episodes
        info = self._time_info(self.time_index, self.current_local_t)
        if profile_id is not None:
            info["topology_profile_id"] = profile_id
        return self._observation_at(self.time_index), info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        absolute_t = self.time_index
        local_t = self.current_local_t
        try:
            placement = decode_flat_actions(np.asarray(action, dtype=np.int64), self.episode.config)
            prior_action = self.previous_action
            report = self.evaluator.evaluate(absolute_t, placement, self.previous_action)
            reward = self.evaluator.reward_from_cost(report.total_cost)
            info = {
                **self._time_info(absolute_t, local_t),
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
            # Reuse the one joint-action load calculation already performed
            # by ``evaluate`` for causal-resource observation.
            self.previous_resource_loads = self.evaluator.last_resource_loads
            self.previous_action = placement
            info.update(self.resource_observation_diagnostics(prior_action, placement))
        except (TypeError, ValueError) as exc:
            reward = self.evaluator.reward_config.invalid_action_reward
            info = {
                **self._time_info(absolute_t, local_t),
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
        self.time_index = absolute_t + 1
        self.current_local_t = local_t + 1
        terminated = self.current_local_t >= self.episode_length or self.time_index >= self.flat_horizon
        truncated = False
        obs_index = min(self.time_index, self.flat_horizon - 1)
        return self._observation_at(obs_index), float(reward), terminated, truncated, info

    def episode_id_for_time_index(self, time_index: int) -> int:
        if time_index < 0 or time_index >= self.flat_horizon:
            raise IndexError("time_index is outside the flat dataset horizon")
        return time_index // self.episode_length

    def local_time_for_time_index(self, time_index: int) -> int:
        if time_index < 0 or time_index >= self.flat_horizon:
            raise IndexError("time_index is outside the flat dataset horizon")
        return time_index % self.episode_length

    def episode_end_indices(self) -> tuple[int, ...]:
        return tuple(((episode_id + 1) * self.episode_length) - 1 for episode_id in range(self.num_episodes))

    def _time_info(self, absolute_t: int, local_t: int) -> dict[str, int | str]:
        return {
            "time_index": absolute_t,
            "episode_id": self.episode_id_for_time_index(absolute_t),
            "local_t": local_t,
        }

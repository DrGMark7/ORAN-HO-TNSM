"""Full-episode recurrent PPO training utilities for the ST-GNN policy.

Rollout carries per-RH GRU state through each environment episode. PPO updates
recompute every complete episode chronologically from its initial hidden state,
so the current policy parameters produce the current causal hidden trajectory.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch_geometric.data import HeteroData

from src.env.oran_ho_env import OranHandoverEnv
from src.model.inference.post_process import ActionSpaceSpec, build_link_feasibility_action_mask
from src.model.graph_size_normalization import entropy_scaling_factor, scale_entropy
from src.model.training.ppo import (
    PPOConfig,
    _module_gradient_norm,
    _finite_correlation,
    _mean_episode_sum,
    compute_ppo_ratio,
    resolve_device,
)
from src.model.training.scenario import TrainingScenario, validate_same_topology
from src.pipeline.data_pipeline import HandoverGraphPipeline
from src.utils.config import load_project_config
from src.utils.timing import synchronized_time
from src.model.causal_observation import (
    CAUSAL_RESOURCE_OBSERVATION_FEATURE_MODE,
    CAUSAL_RESOURCE_V2_OBSERVATION_FEATURE_MODE,
    append_causal_features,
)
from src.model.serial_numba import NUMBA_AVAILABLE


DistributionImplementation = Literal["vectorized", "historical_non_vectorized"]

# The loop used by ``historical_non_vectorized`` is copied directly from this
# source-preserving Git command:
# git show 8f3cc691c8eca51f7836a800a9fb6116172f9f0b:src/model/training/recurrent_ppo.py
# RecurrentPPOTrainer._update, lines 377-396 (the distribution/loss boundary).
HISTORICAL_DISTRIBUTION_IMPLEMENTATION_SOURCE = (
    "8f3cc691c8eca51f7836a800a9fb6116172f9f0b:"
    "src/model/training/recurrent_ppo.py:"
    "RecurrentPPOTrainer._update:377-396"
)


@dataclass(frozen=True)
class RecurrentPPOConfig(PPOConfig):
    """PPO config for the minimal Approach-A recurrent trainer."""

    enable_temporal_logit_delta_diagnostic: bool = field(default=False, kw_only=True)
    distribution_implementation: DistributionImplementation = field(
        default="vectorized", kw_only=True
    )

    def __post_init__(self) -> None:
        if self.distribution_implementation not in {
            "vectorized",
            "historical_non_vectorized",
        }:
            raise ValueError(
                "distribution_implementation must be 'vectorized' or "
                "'historical_non_vectorized'"
            )

    @classmethod
    def from_project_config(cls, project_config: dict | None = None) -> "RecurrentPPOConfig":
        resolved_config = project_config or load_project_config()
        base = PPOConfig.from_project_config(resolved_config)
        return cls(
            **vars(base),
            enable_temporal_logit_delta_diagnostic=bool(
                resolved_config["ppo"].get("enable_temporal_logit_delta_diagnostic", False)
            ),
        )


@dataclass
class RecurrentPPOBatch:
    time_indices: list[int]
    scenario_ids: list[int]
    scenario_names: list[str]
    params_paths: list[str]
    actions: list[np.ndarray]
    rewards: list[float]
    log_probs: list[Tensor]
    values: list[Tensor]
    hidden_states: list[Tensor | None]
    rh_features: list[Tensor] = field(default_factory=list)
    episode_ends: list[bool] = field(default_factory=list)
    episode_ids: list[int] = field(default_factory=list)
    local_timesteps: list[int] = field(default_factory=list)
    episode_start_indices: list[int] = field(default_factory=list)
    episode_initial_hidden_states: list[Tensor | None] = field(default_factory=list)
    total_costs: list[float] = field(default_factory=list)
    reward_input_costs: list[float] = field(default_factory=list)
    overall_feasible: list[bool] = field(default_factory=list)
    bandwidth_feasible: list[bool] = field(default_factory=list)
    compute_feasible: list[bool] = field(default_factory=list)


@dataclass(frozen=True)
class RecurrentSnapshotCache:
    """Compact dynamic snapshots with one shared static handover edge index."""

    rh_features: Tensor
    handover_edge_index: Tensor
    handover_edge_attrs: Tensor


class RecurrentPPOTrainer:
    """Small PPO trainer that carries per-RH GRU state through an episode."""

    def __init__(
        self,
        env: OranHandoverEnv,
        model: nn.Module,
        config: RecurrentPPOConfig | None = None,
        device: str | torch.device = "auto",
        scenarios: list[TrainingScenario] | None = None,
        scenario_seed: int = 0,
        snapshot_transform: Callable[[HeteroData], HeteroData] | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.config = config or RecurrentPPOConfig.from_project_config()
        self.snapshot_transform = snapshot_transform
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.scenarios = scenarios or [
            TrainingScenario.from_env(
                scenario_id=0,
                scenario_name="single",
                params_path="",
                env=env,
            )
        ]
        validate_same_topology(self.scenarios)
        self.causal_numba_enabled_by_scenario: dict[int, bool] = {}
        for scenario in self.scenarios:
            enabled = scenario.env.enable_serial_numba()
            self.causal_numba_enabled_by_scenario[scenario.scenario_id] = enabled
            if (
                NUMBA_AVAILABLE
                and scenario.env.observation_feature_mode
                == CAUSAL_RESOURCE_V2_OBSERVATION_FEATURE_MODE
                and not enabled
            ):
                raise RuntimeError("serial Numba initialization failed for causal_resource_v2")
        self.current_scenario = self.scenarios[0]
        self.env = self.current_scenario.env
        self.graph_pipeline = self.current_scenario.graph_pipeline
        spec = ActionSpaceSpec.from_topology(self.env.episode.config)
        self.num_actions_per_rh = spec.num_actions_per_rh
        self.link_action_mask = self.current_scenario.link_action_mask
        self.valid_action_count_per_rh = self.current_scenario.valid_action_count_per_rh
        self._scenario_rng = np.random.default_rng(scenario_seed)
        self.last_hidden_state_norm_mean = 0.0
        self.last_hidden_state_norm_max = 0.0
        self.last_hidden_state_delta_mean = 0.0
        self.last_hidden_state_delta_max = 0.0
        self.last_temporal_logit_delta_mean = 0.0
        self.last_temporal_logit_delta_max = 0.0
        self.last_sequence_recompute_logprob_max_abs_error = 0.0
        self.last_sequence_recompute_value_max_abs_error = 0.0
        self.last_hidden_state_reset_count = 0
        self.last_unexpected_hidden_reset_count = 0
        self.last_scenario_id = self.current_scenario.scenario_id
        self.last_scenario_name = self.current_scenario.scenario_name
        self.last_params_path = self.current_scenario.params_path
        self.last_rollout_seconds = 0.0
        self.last_update_seconds = 0.0
        self.last_update_profile: dict[str, float] = {}
        self._snapshot_cache_by_scenario = {
            scenario.scenario_id: self._build_snapshot_cache(scenario)
            for scenario in self.scenarios
        }

    def train(self, iterations: int | None = None) -> list[dict[str, float]]:
        project_config = load_project_config()
        iteration_count = int(project_config["ppo"]["default_iterations"]) if iterations is None else iterations
        history: list[dict[str, float]] = []
        for _ in range(iteration_count):
            rollout_start = self._timestamp()
            batch = self._collect_rollouts()
            self.last_rollout_seconds = self._timestamp() - rollout_start
            update_start = self._timestamp()
            metrics = self._update(batch)
            self.last_update_seconds = self._timestamp() - update_start
            metrics["rollout_seconds"] = self.last_rollout_seconds
            metrics["update_seconds"] = self.last_update_seconds
            metrics["train_internal_seconds"] = self.last_rollout_seconds + self.last_update_seconds
            history.append(metrics)
        return history

    def _timestamp(self) -> float:
        return synchronized_time(self.device)

    def _build_snapshot_cache(self, scenario: TrainingScenario) -> RecurrentSnapshotCache:
        horizon = scenario.env.episode.config.horizon
        first_graph = scenario.graph_pipeline.build_snapshot(0)
        if self.snapshot_transform is not None:
            first_graph = self.snapshot_transform(first_graph)
        first_handover = first_graph[("rh", "handover", "rh")]
        rh_features = torch.empty(
            (horizon, *first_graph["rh"].x.shape), dtype=first_graph["rh"].x.dtype
        )
        handover_edge_attrs = torch.empty(
            (horizon, *first_handover.edge_attr.shape), dtype=first_handover.edge_attr.dtype
        )
        shared_edge_index = first_handover.edge_index.clone()
        rh_features[0].copy_(first_graph["rh"].x)
        handover_edge_attrs[0].copy_(first_handover.edge_attr)

        for time_index in range(1, horizon):
            graph = scenario.graph_pipeline.build_snapshot(time_index)
            if self.snapshot_transform is not None:
                graph = self.snapshot_transform(graph)
            handover = graph[("rh", "handover", "rh")]
            if not torch.equal(handover.edge_index, shared_edge_index):
                raise ValueError("handover edge_index must remain static across cached snapshots")
            rh_features[time_index].copy_(graph["rh"].x)
            handover_edge_attrs[time_index].copy_(handover.edge_attr)
        return RecurrentSnapshotCache(
            rh_features=rh_features.to(self.device),
            handover_edge_index=shared_edge_index.to(self.device),
            handover_edge_attrs=handover_edge_attrs.to(self.device),
        )

    def _collect_rollouts(self) -> RecurrentPPOBatch:
        time_indices: list[int] = []
        scenario_ids: list[int] = []
        scenario_names: list[str] = []
        params_paths: list[str] = []
        actions: list[np.ndarray] = []
        rewards: list[float] = []
        log_probs: list[Tensor] = []
        values: list[Tensor] = []
        hidden_states: list[Tensor | None] = []
        rollout_rh_features: list[Tensor] = []
        episode_ends: list[bool] = []
        episode_ids: list[int] = []
        local_timesteps: list[int] = []
        episode_start_indices: list[int] = []
        episode_initial_hidden_states: list[Tensor | None] = []
        hidden_norms: list[Tensor] = []
        hidden_deltas: list[Tensor] = []
        temporal_logit_deltas: list[Tensor] = []
        total_costs: list[float] = []
        reward_input_costs: list[float] = []
        overall_feasible: list[bool] = []
        bandwidth_feasible: list[bool] = []
        compute_feasible: list[bool] = []
        reset_count = 0
        unexpected_reset_count = 0

        for rollout_episode_id in range(self.config.rollout_episodes):
            scenario = self._sample_scenario()
            self._activate_scenario(scenario)
            _, reset_info = self.env.reset()
            hidden_state: Tensor | None = None
            episode_start_indices.append(len(time_indices))
            episode_initial_hidden_states.append(None)
            reset_count += 1
            done = False
            local_t = 0
            while not done:
                time_index = self.env.time_index
                hidden_before = hidden_state.detach() if hidden_state is not None else None
                current_features = self._features_at(time_index, scenario.scenario_id)
                with torch.no_grad():
                    logits, value, next_hidden_state = self._model_at(
                        time_index,
                        hidden_before,
                        scenario.scenario_id,
                        current_features,
                    )
                    dist = Categorical(
                        logits=self._masked_logits(logits.squeeze(0), scenario.scenario_id, time_index)
                    )
                    action = dist.sample()
                    if (
                        self.config.enable_temporal_logit_delta_diagnostic
                        and hasattr(self.model, "temporal_fusion_mode")
                    ):
                        reset_logits, _, _ = self._model_at(
                            time_index, None, scenario.scenario_id, current_features
                        )
                        temporal_logit_deltas.append(
                            torch.linalg.vector_norm(logits - reset_logits).detach()
                        )
                _, reward, terminated, truncated, info = self.env.step(
                    action.detach().cpu().numpy()
                )

                time_indices.append(time_index)
                scenario_ids.append(scenario.scenario_id)
                scenario_names.append(scenario.scenario_name)
                params_paths.append(scenario.params_path)
                actions.append(action.detach().cpu().numpy())
                rewards.append(float(reward))
                log_probs.append(dist.log_prob(action).sum().detach())
                values.append(value.squeeze(0).detach())
                hidden_states.append(hidden_before)
                rollout_rh_features.append(current_features.detach().cpu())
                episode_ends.append(bool(terminated or truncated))
                episode_ids.append(int(reset_info.get("episode_id", rollout_episode_id)))
                local_timesteps.append(local_t)
                total_costs.append(float(info["total_cost"]))
                reward_input_costs.append(float(info.get("reward_input_cost", info["total_cost"])))
                overall_feasible.append(bool(info.get("feasible", False)))
                bandwidth_feasible.append(float(info.get("bandwidth_violation_cost", 0.0)) <= 0.0)
                compute_feasible.append(float(info.get("compute_overload_cost", 0.0)) <= 0.0)
                detached_next = next_hidden_state.detach()
                hidden_norms.append(torch.linalg.vector_norm(detached_next))
                hidden_reference = (
                    torch.zeros_like(detached_next) if hidden_before is None else hidden_before
                )
                hidden_deltas.append(torch.linalg.vector_norm(detached_next - hidden_reference))

                done = terminated or truncated
                hidden_state = None if done else detached_next
                local_t += 1

        self.last_hidden_state_norm_mean, self.last_hidden_state_norm_max = self._mean_and_max(hidden_norms)
        self.last_hidden_state_delta_mean, self.last_hidden_state_delta_max = self._mean_and_max(hidden_deltas)
        if self.config.enable_temporal_logit_delta_diagnostic:
            self.last_temporal_logit_delta_mean, self.last_temporal_logit_delta_max = self._mean_and_max(
                temporal_logit_deltas
            )
        else:
            self.last_temporal_logit_delta_mean = float("nan")
            self.last_temporal_logit_delta_max = float("nan")
        self.last_hidden_state_reset_count = reset_count
        self.last_unexpected_hidden_reset_count = unexpected_reset_count
        return RecurrentPPOBatch(
            time_indices=time_indices,
            scenario_ids=scenario_ids,
            scenario_names=scenario_names,
            params_paths=params_paths,
            actions=actions,
            rewards=rewards,
            log_probs=log_probs,
            values=values,
            hidden_states=hidden_states,
            rh_features=rollout_rh_features,
            episode_ends=episode_ends,
            episode_ids=episode_ids,
            local_timesteps=local_timesteps,
            episode_start_indices=episode_start_indices,
            episode_initial_hidden_states=episode_initial_hidden_states,
            total_costs=total_costs,
            reward_input_costs=reward_input_costs,
            overall_feasible=overall_feasible,
            bandwidth_feasible=bandwidth_feasible,
            compute_feasible=compute_feasible,
        )

    @staticmethod
    def _mean_and_max(values: list[Tensor]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        stacked = torch.stack(values)
        return float(stacked.mean().cpu()), float(stacked.max().cpu())

    def _discounted_returns(
        self,
        rewards: list[float],
        episode_ends: list[bool] | None = None,
    ) -> Tensor:
        if episode_ends is None:
            episode_ends = [False] * len(rewards)
            if episode_ends:
                episode_ends[-1] = True
        if len(episode_ends) != len(rewards):
            raise ValueError("episode_ends must have the same length as rewards")

        returns: list[float] = []
        running = 0.0
        for reward, episode_end in zip(reversed(rewards), reversed(episode_ends), strict=True):
            if episode_end:
                running = 0.0
            running = reward + self.config.gamma * running
            returns.append(running)
        returns.reverse()
        return torch.as_tensor(returns, dtype=torch.float32, device=self.device)

    def _update(self, batch: RecurrentPPOBatch) -> dict[str, float]:
        profile = {
            "update_prepare_seconds": 0.0,
            "update_forward_seconds": 0.0,
            "update_distribution_seconds": 0.0,
            "update_loss_seconds": 0.0,
            "update_backward_seconds": 0.0,
            "update_optimizer_seconds": 0.0,
        }
        prepare_start = self._timestamp()
        returns = self._discounted_returns(batch.rewards, batch.episode_ends)
        old_log_probs = torch.stack(batch.log_probs)
        old_values = torch.stack(batch.values).float()
        advantages_before_normalization = returns - old_values
        advantages = (
            advantages_before_normalization - advantages_before_normalization.mean()
        ) / (advantages_before_normalization.std() + 1e-8)
        profile["update_prepare_seconds"] += self._timestamp() - prepare_start

        last_loss = torch.tensor(0.0, device=self.device)
        last_policy_loss = torch.tensor(0.0, device=self.device)
        last_value_loss = torch.tensor(0.0, device=self.device)
        last_entropy = torch.tensor(0.0, device=self.device)
        last_scaled_entropy = torch.tensor(0.0, device=self.device)
        last_approx_kl = torch.tensor(0.0, device=self.device)
        last_ratio = torch.ones_like(old_log_probs)
        last_actor_gradient_norm = 0.0
        last_critic_gradient_norm = 0.0
        last_total_gradient_norm = 0.0
        self.last_sequence_recompute_logprob_max_abs_error = 0.0
        self.last_sequence_recompute_value_max_abs_error = 0.0
        for update_epoch in range(self.config.update_epochs):
            forward_start = self._timestamp()
            logits_by_step, values_t = self._recompute_full_episode_sequences(batch)
            profile["update_forward_seconds"] += self._timestamp() - forward_start

            distribution_start = self._timestamp()
            if self.config.distribution_implementation == "vectorized":
                _, _, new_log_probs_t, entropy_terms_t = self._distribution_terms_vectorized(
                    logits_by_step, batch
                )
            else:
                new_log_probs, entropy_terms = self._distribution_terms_historical_from_8f3cc(
                    logits_by_step, batch
                )
            profile["update_distribution_seconds"] += self._timestamp() - distribution_start

            loss_start = self._timestamp()
            if self.config.distribution_implementation == "vectorized":
                last_entropy = entropy_terms_t.mean()
            else:
                new_log_probs_t = torch.stack(new_log_probs)
                last_entropy = torch.stack(entropy_terms).mean()
            last_scaled_entropy = scale_entropy(
                last_entropy,
                self.config.entropy_scale_mode,
                self.env.episode.config.num_rhs,
            )
            if update_epoch == 0:
                self.last_sequence_recompute_logprob_max_abs_error = float(
                    torch.max(torch.abs(new_log_probs_t.detach() - old_log_probs)).cpu()
                )
                self.last_sequence_recompute_value_max_abs_error = float(
                    torch.max(torch.abs(values_t.detach() - old_values)).cpu()
                )
            ratio = compute_ppo_ratio(new_log_probs_t, old_log_probs)
            last_ratio = ratio.detach()
            clipped = torch.clamp(
                ratio,
                1.0 - self.config.clip_epsilon,
                1.0 + self.config.clip_epsilon,
            )
            last_policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
            last_value_loss = nn.functional.mse_loss(values_t, returns)
            last_approx_kl = (old_log_probs - new_log_probs_t).mean()
            last_loss = (
                last_policy_loss
                + self.config.value_coef * last_value_loss
                - self.config.entropy_coef * last_scaled_entropy
            )
            profile["update_loss_seconds"] += self._timestamp() - loss_start
            backward_start = self._timestamp()
            self.optimizer.zero_grad()
            last_loss.backward()
            last_actor_gradient_norm = _module_gradient_norm(self.model.actor)
            last_critic_gradient_norm = _module_gradient_norm(self.model.critic)
            last_total_gradient_norm = _module_gradient_norm(self.model)
            profile["update_backward_seconds"] += self._timestamp() - backward_start
            optimizer_start = self._timestamp()
            self.optimizer.step()
            profile["update_optimizer_seconds"] += self._timestamp() - optimizer_start

        self.last_update_profile = profile
        scenario_summary = self._scenario_metric_summary(batch)
        ratio_quantiles = torch.quantile(
            last_ratio.float(),
            torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], device=last_ratio.device),
        )
        clip_low = 1.0 - self.config.clip_epsilon
        clip_high = 1.0 + self.config.clip_epsilon
        num_rh = self.env.episode.config.num_rhs
        return {
            "loss": float(last_loss.detach().cpu()),
            "policy_loss": float(last_policy_loss.detach().cpu()),
            "value_loss": float(last_value_loss.detach().cpu()),
            "weighted_value_term": float(
                (self.config.value_coef * last_value_loss).detach().cpu()
            ),
            "entropy": float(last_entropy.detach().cpu()),
            "raw_entropy_sum": float(last_entropy.detach().cpu()),
            "entropy_per_rh": float(last_entropy.detach().cpu()) / num_rh,
            "scaled_entropy": float(last_scaled_entropy.detach().cpu()),
            "weighted_entropy_contribution": float(
                (-self.config.entropy_coef * last_scaled_entropy).detach().cpu()
            ),
            "approx_kl": float(last_approx_kl.detach().cpu()),
            "ppo_ratio_p1": float(ratio_quantiles[0].cpu()),
            "ppo_ratio_p5": float(ratio_quantiles[1].cpu()),
            "ppo_ratio_p50": float(ratio_quantiles[2].cpu()),
            "ppo_ratio_p95": float(ratio_quantiles[3].cpu()),
            "ppo_ratio_p99": float(ratio_quantiles[4].cpu()),
            "clip_fraction": float(
                ((last_ratio < clip_low) | (last_ratio > clip_high)).float().mean().cpu()
            ),
            "actor_gradient_norm": last_actor_gradient_norm,
            "critic_gradient_norm": last_critic_gradient_norm,
            "total_gradient_norm": last_total_gradient_norm,
            "advantage_mean_before_normalization": float(
                advantages_before_normalization.mean().cpu()
            ),
            "advantage_std_before_normalization": float(
                advantages_before_normalization.std().cpu()
            ),
            "return_mean": float(returns.mean().cpu()),
            "return_std": float(returns.std().cpu()),
            "mean_reward": float(np.mean(batch.rewards)),
            "reward_std": float(np.std(batch.rewards)),
            "rollout_total_cost_mean": float(np.mean(batch.total_costs)),
            "rollout_total_j_mean": _mean_episode_sum(batch.total_costs, batch.episode_ends),
            "reward_input_cost_mean": float(np.mean(batch.reward_input_costs)),
            "rollout_reward_input_j_mean": _mean_episode_sum(
                batch.reward_input_costs, batch.episode_ends
            ),
            "rollout_overall_feasible_rate": float(np.mean(batch.overall_feasible)),
            "rollout_bandwidth_feasible_rate": float(np.mean(batch.bandwidth_feasible)),
            "rollout_compute_feasible_rate": float(np.mean(batch.compute_feasible)),
            "advantage_raw_cost_correlation": _finite_correlation(
                advantages_before_normalization.detach().cpu().numpy(), batch.total_costs
            ),
            "num_rh": float(num_rh),
            "entropy_scaling_factor": entropy_scaling_factor(
                self.config.entropy_scale_mode, num_rh
            ),
            "reward_scaling_factor": float(
                self.env.evaluator.formulation_metadata()["reward_scaling_factor"]
            ),
            "entropy_scale_mode": self.config.entropy_scale_mode,
            "reward_graph_scale_mode": self.env.evaluator.reward_config.reward_graph_scale_mode,
            "hidden_state_norm_mean": self.last_hidden_state_norm_mean,
            "hidden_state_norm_max": self.last_hidden_state_norm_max,
            "hidden_state_delta_mean": self.last_hidden_state_delta_mean,
            "hidden_state_delta_max": self.last_hidden_state_delta_max,
            "temporal_logit_delta_mean": self.last_temporal_logit_delta_mean,
            "temporal_logit_delta_max": self.last_temporal_logit_delta_max,
            "temporal_logit_delta_diagnostic_enabled": float(
                self.config.enable_temporal_logit_delta_diagnostic
            ),
            "sequence_recompute_logprob_max_abs_error": self.last_sequence_recompute_logprob_max_abs_error,
            "sequence_recompute_value_max_abs_error": self.last_sequence_recompute_value_max_abs_error,
            "episode_hidden_reset_count": float(self.last_hidden_state_reset_count),
            "unexpected_hidden_reset_count": float(self.last_unexpected_hidden_reset_count),
            "hidden_state_reset_count": float(self.last_hidden_state_reset_count),
            **scenario_summary,
            **profile,
        }

    def _distribution_terms_historical_from_8f3cc(
        self,
        logits_by_step: list[Tensor],
        batch: RecurrentPPOBatch,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Return the literal pre-vectorization loop from ``8f3cc691``.

        The surrounding update path retains the historical tensor stacking and
        entropy reduction order immediately after this loop.
        """

        new_log_probs: list[Tensor] = []
        entropy_terms: list[Tensor] = []
        for batch_index, (logits, action_np) in enumerate(
            zip(logits_by_step, batch.actions, strict=True)
        ):
            scenario_id = batch.scenario_ids[batch_index]
            dist = Categorical(
                logits=self._masked_logits(
                    logits.squeeze(0), scenario_id, batch.time_indices[batch_index]
                )
            )
            action = torch.as_tensor(action_np, dtype=torch.long, device=self.device)
            new_log_probs.append(dist.log_prob(action).sum())
            entropy_terms.append(dist.entropy().sum())
        return new_log_probs, entropy_terms

    def _distribution_terms_stepwise(
        self,
        logits_by_step: list[Tensor],
        batch: RecurrentPPOBatch,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Legacy per-timestep masked categorical calculation for exact audits."""

        masked_logits: list[Tensor] = []
        probabilities: list[Tensor] = []
        log_probs: list[Tensor] = []
        entropy: list[Tensor] = []
        for batch_index, (logits, action_np) in enumerate(
            zip(logits_by_step, batch.actions, strict=True)
        ):
            scenario_id = batch.scenario_ids[batch_index]
            masked = self._masked_logits(
                logits.squeeze(0), scenario_id, batch.time_indices[batch_index]
            )
            dist = Categorical(logits=masked)
            action = torch.as_tensor(action_np, dtype=torch.long, device=self.device)
            masked_logits.append(masked)
            probabilities.append(dist.probs)
            log_probs.append(dist.log_prob(action).sum())
            entropy.append(dist.entropy().sum())
        return (
            torch.stack(masked_logits),
            torch.stack(probabilities),
            torch.stack(log_probs),
            torch.stack(entropy),
        )

    def _distribution_terms_vectorized(
        self,
        logits_by_step: list[Tensor],
        batch: RecurrentPPOBatch,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Apply timestep-specific masks and categorical operations in one batch.

        Each row remains the same ``[RH, action]`` categorical calculation as
        the legacy path. The strict reference check in ``scripts/optimize``
        guards this implementation against any numerical or semantic drift.
        """

        raw_logits = torch.stack([logits.squeeze(0) for logits in logits_by_step])
        masks = torch.stack([
            torch.as_tensor(
                self._scenario_by_id(batch.scenario_ids[batch_index]).action_mask_for_time_index(
                    batch.time_indices[batch_index]
                ),
                dtype=torch.bool,
                device=self.device,
            )
            for batch_index in range(len(batch.actions))
        ])
        masked_logits = raw_logits.masked_fill(~masks, torch.finfo(raw_logits.dtype).min)
        actions = torch.as_tensor(np.stack(batch.actions), dtype=torch.long, device=self.device)
        dist = Categorical(logits=masked_logits)
        return (
            masked_logits,
            dist.probs,
            dist.log_prob(actions).sum(dim=-1),
            dist.entropy().sum(dim=-1),
        )

    def _recompute_full_episode_sequences(
        self,
        batch: RecurrentPPOBatch,
    ) -> tuple[list[Tensor], Tensor]:
        """Replay complete episodes in order using the current recurrent parameters."""

        episode_ranges = self._episode_ranges(batch)
        if len(episode_ranges) != len(batch.episode_initial_hidden_states):
            raise ValueError("one episode_initial_hidden_state is required per recurrent sequence")

        logits_by_step: list[Tensor] = []
        values_by_step: list[Tensor] = []
        for sequence_index, (start, stop) in enumerate(episode_ranges):
            initial_hidden = batch.episode_initial_hidden_states[sequence_index]
            hidden_state = None if initial_hidden is None else initial_hidden.to(self.device)
            for batch_index in range(start, stop):
                logits, value, next_hidden_state = self._model_at(
                    batch.time_indices[batch_index],
                    hidden_state,
                    batch.scenario_ids[batch_index],
                    batch.rh_features[batch_index].to(self.device),
                )
                logits_by_step.append(logits)
                values_by_step.append(value.squeeze(0))
                hidden_state = next_hidden_state
        return logits_by_step, torch.stack(values_by_step).float()

    @staticmethod
    def _episode_ranges(batch: RecurrentPPOBatch) -> list[tuple[int, int]]:
        if not batch.episode_ends or len(batch.episode_ends) != len(batch.time_indices):
            raise ValueError("episode_ends must identify every recurrent rollout timestep")
        stops = [index + 1 for index, is_end in enumerate(batch.episode_ends) if is_end]
        if not stops or stops[-1] != len(batch.time_indices):
            raise ValueError("the recurrent rollout must end at a real episode boundary")
        starts = [0, *stops[:-1]]
        ranges = list(zip(starts, stops, strict=True))
        if batch.episode_start_indices and batch.episode_start_indices != starts:
            raise ValueError("episode_start_indices do not match episode boundaries")
        if batch.local_timesteps:
            for start, stop in ranges:
                if batch.local_timesteps[start:stop] != list(range(stop - start)):
                    raise ValueError("local timesteps must remain ordered within each episode")
                if len(set(batch.episode_ids[start:stop])) != 1:
                    raise ValueError("a recurrent sequence cannot mix episode identities")
                if len(set(batch.scenario_ids[start:stop])) != 1:
                    raise ValueError("a recurrent sequence cannot mix scenarios")
        return ranges

    def _model_at(
        self,
        time_index: int,
        hidden_state: Tensor | None = None,
        scenario_id: int | None = None,
        rh_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        selected_id = self.current_scenario.scenario_id if scenario_id is None else scenario_id
        cache = self._snapshot_cache_by_scenario[selected_id]
        if rh_features is None:
            selected_features = self._features_at(time_index, selected_id)
        else:
            selected_features = rh_features
        features = selected_features.unsqueeze(0).unsqueeze(0)
        edge_index = cache.handover_edge_index
        edge_attr = cache.handover_edge_attrs[time_index]
        return self.model.forward_recurrent(
            features,
            [edge_index],
            [edge_attr],
            hidden_state=hidden_state,
        )

    def _features_at(self, time_index: int, scenario_id: int) -> Tensor:
        scenario = self._scenario_by_id(scenario_id)
        cache = self._snapshot_cache_by_scenario[scenario_id]
        if scenario.env.observation_feature_mode == CAUSAL_RESOURCE_V2_OBSERVATION_FEATURE_MODE:
            causal = torch.as_tensor(
                scenario.env.causal_rh_features(),
                dtype=cache.rh_features.dtype,
                device=self.device,
            )
            return torch.cat((cache.rh_features[time_index], causal), dim=1)
        if scenario.env.observation_feature_mode != CAUSAL_RESOURCE_OBSERVATION_FEATURE_MODE:
            return cache.rh_features[time_index]
        graph = scenario.graph_pipeline.build_snapshot(time_index)
        graph = append_causal_features(
            graph, scenario.env.causal_rh_features(), scenario.env.episode
        )
        if self.snapshot_transform is not None:
            graph = self.snapshot_transform(graph)
        return graph["rh"].x.to(self.device)

    def _masked_logits(
        self,
        logits: Tensor,
        scenario_id: int | None = None,
        time_index: int | None = None,
    ) -> Tensor:
        scenario = self._scenario_by_id(self.current_scenario.scenario_id if scenario_id is None else scenario_id)
        if logits.shape != (scenario.env.episode.config.num_rhs, self.num_actions_per_rh):
            raise ValueError("logits must have shape (num_rhs, num_actions_per_rh)")
        resolved_time = scenario.env.time_index if time_index is None else time_index
        selected_mask = scenario.action_mask_for_time_index(resolved_time)
        mask = torch.as_tensor(selected_mask, dtype=torch.bool, device=logits.device)
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

    def _sample_scenario(self) -> TrainingScenario:
        if len(self.scenarios) == 1:
            return self.scenarios[0]
        index = int(self._scenario_rng.integers(0, len(self.scenarios)))
        return self.scenarios[index]

    def _activate_scenario(self, scenario: TrainingScenario) -> None:
        self.current_scenario = scenario
        self.env = scenario.env
        self.graph_pipeline = scenario.graph_pipeline
        self.link_action_mask = scenario.link_action_mask
        self.valid_action_count_per_rh = scenario.valid_action_count_per_rh
        self.last_scenario_id = scenario.scenario_id
        self.last_scenario_name = scenario.scenario_name
        self.last_params_path = scenario.params_path

    def _scenario_by_id(self, scenario_id: int) -> TrainingScenario:
        for scenario in self.scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario
        raise KeyError(f"unknown scenario_id {scenario_id}")

    def _scenario_metric_summary(self, batch: RecurrentPPOBatch) -> dict[str, float | str]:
        unique_ids = sorted(set(batch.scenario_ids))
        if len(unique_ids) == 1:
            scenario_id = unique_ids[0]
            first_index = batch.scenario_ids.index(scenario_id)
            return {
                "scenario_id": float(scenario_id),
                "scenario_name": batch.scenario_names[first_index],
                "params_path": batch.params_paths[first_index],
            }
        return {
            "scenario_id": -1.0,
            "scenario_name": "mixed",
            "params_path": ";".join(sorted(set(batch.params_paths))),
        }

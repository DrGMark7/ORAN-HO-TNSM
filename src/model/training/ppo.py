"""Minimal PPO training utilities for the ST-GNN policy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch_geometric.data import HeteroData

from src.env.oran_ho_env import OranHandoverEnv
from src.model.inference.post_process import (
    ActionSpaceSpec,
    action_mask_kwargs_from_episode,
    build_link_feasibility_action_mask,
    build_topology_profile_action_masks,
)
from src.model.graph_size_normalization import (
    ENTROPY_SCALE_MODES,
    entropy_scaling_factor,
    scale_entropy,
)
from src.pipeline.data_pipeline import HandoverGraphPipeline
from src.utils.config import load_project_config
from src.model.causal_observation import (
    RESOURCE_OBSERVATION_FEATURE_MODES,
    rh_feature_names,
)


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float
    gamma: float
    clip_epsilon: float
    entropy_coef: float
    value_coef: float
    update_epochs: int
    rollout_episodes: int
    entropy_scale_mode: str = field(default="legacy_sum", kw_only=True)

    @classmethod
    def from_project_config(cls, project_config: dict | None = None) -> "PPOConfig":
        config = project_config or load_project_config()
        ppo = config["ppo"]
        resolved = cls(
            learning_rate=float(ppo["learning_rate"]),
            gamma=float(ppo["gamma"]),
            clip_epsilon=float(ppo["clip_epsilon"]),
            entropy_coef=float(ppo["entropy_coef"]),
            value_coef=float(ppo["value_coef"]),
            update_epochs=int(ppo["update_epochs"]),
            rollout_episodes=int(ppo["rollout_episodes"]),
            entropy_scale_mode=str(ppo.get("entropy_scale_mode", "legacy_sum")),
        )
        if resolved.entropy_scale_mode not in ENTROPY_SCALE_MODES:
            raise ValueError(f"ppo.entropy_scale_mode must be one of {ENTROPY_SCALE_MODES}")
        return resolved


@dataclass
class PPOBatch:
    time_indices: list[int]
    actions: list[np.ndarray]
    rewards: list[float]
    log_probs: list[Tensor]
    values: list[Tensor]
    rh_features: list[Tensor] = field(default_factory=list)
    episode_ends: list[bool] = field(default_factory=list)
    total_costs: list[float] = field(default_factory=list)
    reward_input_costs: list[float] = field(default_factory=list)
    overall_feasible: list[bool] = field(default_factory=list)
    bandwidth_feasible: list[bool] = field(default_factory=list)
    compute_feasible: list[bool] = field(default_factory=list)


@dataclass(frozen=True)
class GPPOSnapshotCache:
    """Time-major model tensors with one shared handover edge index."""

    rh_features: Tensor
    handover_edge_index: Tensor
    handover_edge_attrs: Tensor
    rh_feature_names: tuple[str, ...]
    handover_feature_names: tuple[str, ...]


class PPOTrainer:
    """Small PPO loop for smoke-training the actor-critic model."""

    def __init__(
        self,
        env: OranHandoverEnv,
        model: nn.Module,
        config: PPOConfig | None = None,
        device: str | torch.device = "auto",
        snapshot_transform: Callable[[HeteroData], HeteroData] | None = None,
    ) -> None:
        self.env = env
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.config = config or PPOConfig.from_project_config()
        self.snapshot_transform = snapshot_transform
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.graph_pipeline = HandoverGraphPipeline(env.episode)
        self._snapshot_cache = self._build_snapshot_cache()
        spec = ActionSpaceSpec.from_topology(env.episode.config)
        self.num_actions_per_rh = spec.num_actions_per_rh
        self.link_action_mask = build_link_feasibility_action_mask(
            env.episode.config,
            env.episode.fronthaul_mask,
            env.episode.midhaul_mask,
            env.episode.direct_rh_rc_mask,
            env.episode.split_demand.split_names,
            **action_mask_kwargs_from_episode(env.episode),
        )
        self.topology_profile_action_masks = build_topology_profile_action_masks(env.episode)
        self.valid_action_count_per_rh = self.link_action_mask.sum(axis=1).astype(np.int64)

    def _build_snapshot_cache(self) -> GPPOSnapshotCache:
        horizon = self.env.episode.config.horizon
        first_graph = self.graph_pipeline.build_snapshot(0)
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
            graph = self.graph_pipeline.build_snapshot(time_index)
            if self.snapshot_transform is not None:
                graph = self.snapshot_transform(graph)
            handover = graph[("rh", "handover", "rh")]
            if not torch.equal(handover.edge_index, shared_edge_index):
                raise ValueError("handover edge_index must remain static across cached snapshots")
            rh_features[time_index].copy_(graph["rh"].x)
            handover_edge_attrs[time_index].copy_(handover.edge_attr)
        return GPPOSnapshotCache(
            rh_features=rh_features.to(self.device),
            handover_edge_index=shared_edge_index.to(self.device),
            handover_edge_attrs=handover_edge_attrs.to(self.device),
            rh_feature_names=tuple(getattr(first_graph["rh"], "feature_names", ())),
            handover_feature_names=tuple(getattr(first_handover, "feature_names", ())),
        )

    def train(self, iterations: int | None = None) -> list[dict[str, float]]:
        project_config = load_project_config()
        iteration_count = int(project_config["ppo"]["default_iterations"]) if iterations is None else iterations
        history: list[dict[str, float]] = []
        for _ in range(iteration_count):
            batch = self._collect_rollouts()
            history.append(self._update(batch))
        return history

    def _collect_rollouts(self) -> PPOBatch:
        time_indices: list[int] = []
        actions: list[np.ndarray] = []
        rewards: list[float] = []
        log_probs: list[Tensor] = []
        values: list[Tensor] = []
        rh_features: list[Tensor] = []
        episode_ends: list[bool] = []
        total_costs: list[float] = []
        reward_input_costs: list[float] = []
        overall_feasible: list[bool] = []
        bandwidth_feasible: list[bool] = []
        compute_feasible: list[bool] = []

        for _ in range(self.config.rollout_episodes):
            self.env.reset()
            done = False
            while not done:
                time_index = self.env.time_index
                graph, current_features = self._graph_and_features_at(time_index)
                logits, value = self._model_at(time_index, current_features, graph=graph)
                dist = Categorical(logits=self._masked_logits(logits.squeeze(0), time_index))
                action = dist.sample()
                _, reward, terminated, truncated, info = self.env.step(
                    action.detach().cpu().numpy()
                )

                time_indices.append(time_index)
                actions.append(action.detach().cpu().numpy())
                rewards.append(float(reward))
                log_probs.append(dist.log_prob(action).sum().detach())
                values.append(value.squeeze(0).detach())
                rh_features.append(current_features.detach().cpu())
                episode_ends.append(bool(terminated or truncated))
                total_costs.append(float(info["total_cost"]))
                reward_input_costs.append(float(info.get("reward_input_cost", info["total_cost"])))
                overall_feasible.append(bool(info.get("feasible", False)))
                bandwidth_feasible.append(float(info.get("bandwidth_violation_cost", 0.0)) <= 0.0)
                compute_feasible.append(float(info.get("compute_overload_cost", 0.0)) <= 0.0)

                done = terminated or truncated
        return PPOBatch(
            time_indices,
            actions,
            rewards,
            log_probs,
            values,
            rh_features,
            episode_ends,
            total_costs,
            reward_input_costs,
            overall_feasible,
            bandwidth_feasible,
            compute_feasible,
        )

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

    def _update(self, batch: PPOBatch) -> dict[str, float]:
        returns = self._discounted_returns(batch.rewards, batch.episode_ends)
        old_log_probs = torch.stack(batch.log_probs)
        old_values = torch.stack(batch.values).float()
        advantages_before_normalization = returns - old_values
        advantages = (
            advantages_before_normalization - advantages_before_normalization.mean()
        ) / (advantages_before_normalization.std() + 1e-8)

        last_loss = torch.tensor(0.0)
        last_policy_loss = torch.tensor(0.0)
        last_value_loss = torch.tensor(0.0)
        last_entropy_sum = torch.tensor(0.0)
        last_scaled_entropy = torch.tensor(0.0)
        last_approx_kl = torch.tensor(0.0)
        last_ratio = torch.ones_like(old_log_probs)
        last_actor_gradient_norm = 0.0
        last_critic_gradient_norm = 0.0
        last_total_gradient_norm = 0.0
        for _ in range(self.config.update_epochs):
            new_log_probs: list[Tensor] = []
            entropy_terms: list[Tensor] = []
            values: list[Tensor] = []
            for time_index, action_np, features in zip(
                batch.time_indices, batch.actions, batch.rh_features, strict=True
            ):
                logits, value = self._model_at(time_index, features.to(self.device))
                dist = Categorical(logits=self._masked_logits(logits.squeeze(0), time_index))
                action = torch.as_tensor(action_np, dtype=torch.long, device=self.device)
                new_log_probs.append(dist.log_prob(action).sum())
                entropy_terms.append(dist.entropy().sum())
                values.append(value.squeeze(0))

            new_log_probs_t = torch.stack(new_log_probs)
            values_t = torch.stack(values).float()
            last_entropy_sum = torch.stack(entropy_terms).mean()
            last_scaled_entropy = scale_entropy(
                last_entropy_sum,
                self.config.entropy_scale_mode,
                self.env.episode.config.num_rhs,
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
            self.optimizer.zero_grad()
            last_loss.backward()
            last_actor_gradient_norm = _module_gradient_norm(self.model.actor)
            last_critic_gradient_norm = _module_gradient_norm(self.model.critic)
            last_total_gradient_norm = _module_gradient_norm(self.model)
            self.optimizer.step()

        ratio_quantiles = torch.quantile(
            last_ratio.float(),
            torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], device=last_ratio.device),
        )
        clip_low = 1.0 - self.config.clip_epsilon
        clip_high = 1.0 + self.config.clip_epsilon
        num_rh = self.env.episode.config.num_rhs
        entropy_factor = entropy_scaling_factor(self.config.entropy_scale_mode, num_rh)
        return {
            "loss": float(last_loss.detach().cpu()),
            "policy_loss": float(last_policy_loss.detach().cpu()),
            "value_loss": float(last_value_loss.detach().cpu()),
            "weighted_value_term": float(
                (self.config.value_coef * last_value_loss).detach().cpu()
            ),
            "entropy": float(last_entropy_sum.detach().cpu()),
            "raw_entropy_sum": float(last_entropy_sum.detach().cpu()),
            "entropy_per_rh": float(last_entropy_sum.detach().cpu()) / num_rh,
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
            "clip_fraction": float(((last_ratio < clip_low) | (last_ratio > clip_high)).float().mean().cpu()),
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
            "entropy_scaling_factor": entropy_factor,
            "reward_scaling_factor": float(
                self.env.evaluator.formulation_metadata()["reward_scaling_factor"]
            ),
            "entropy_scale_mode": self.config.entropy_scale_mode,
            "reward_graph_scale_mode": self.env.evaluator.reward_config.reward_graph_scale_mode,
        }

    def _features_at(self, time_index: int) -> Tensor:
        """Return current RH features without retaining the graph snapshot."""

        _, features = self._graph_and_features_at(time_index)
        return features

    def _graph_and_features_at(self, time_index: int) -> tuple[HeteroData, Tensor]:
        """Create the minimal model view from compact time-major tensors."""

        base_features = self._snapshot_cache.rh_features[time_index]
        if self.env.observation_feature_mode in RESOURCE_OBSERVATION_FEATURE_MODES:
            causal = torch.as_tensor(
                self.env.causal_rh_features(), dtype=base_features.dtype, device=self.device
            )
            features = torch.cat((base_features, causal), dim=1)
        else:
            features = base_features
        graph = HeteroData()
        graph["rh"].x = features
        graph["rh"].feature_names = rh_feature_names(
            self.env.episode, self.env.observation_feature_mode
        )
        handover = graph[("rh", "handover", "rh")]
        handover.edge_index = self._snapshot_cache.handover_edge_index
        handover.edge_attr = self._snapshot_cache.handover_edge_attrs[time_index]
        handover.feature_names = self._snapshot_cache.handover_feature_names
        return graph, features

    def _model_at(
        self,
        time_index: int,
        rh_features: Tensor | None = None,
        graph: HeteroData | None = None,
    ) -> tuple[Tensor, Tensor]:
        features_tensor = self._features_at(time_index) if rh_features is None else rh_features
        features = features_tensor.unsqueeze(0).unsqueeze(0)
        if graph is None:
            edge_index = self._snapshot_cache.handover_edge_index
            edge_attr = self._snapshot_cache.handover_edge_attrs[time_index]
        else:
            handover = graph[("rh", "handover", "rh")]
            edge_index = handover.edge_index
            edge_attr = handover.edge_attr
        return self.model(
            features,
            [edge_index],
            [edge_attr],
        )

    def _masked_logits(self, logits: Tensor, time_index: int | None = None) -> Tensor:
        if logits.shape != (self.env.episode.config.num_rhs, self.num_actions_per_rh):
            raise ValueError("logits must have shape (num_rhs, num_actions_per_rh)")
        selected_mask = self.link_action_mask
        if self.topology_profile_action_masks:
            resolved_time = self.env.time_index if time_index is None else time_index
            episode_id_for_time = getattr(self.env, "episode_id_for_time_index", None)
            profile_id = (
                self.env.episode.metadata.get("active_topology_profile_id")
                if episode_id_for_time is None
                else self.env.episode.topology_profile_id_for_episode(
                    int(episode_id_for_time(resolved_time))
                )
            )
            if profile_id is not None:
                selected_mask = self.topology_profile_action_masks[str(profile_id)]
        mask = torch.as_tensor(selected_mask, dtype=torch.bool, device=logits.device)
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve a training device without changing model semantics."""

    requested = str(device)
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return torch.device(device)


def _module_gradient_norm(module: nn.Module) -> float:
    squared_norm = sum(
        float(torch.sum(parameter.grad.detach() ** 2).cpu())
        for parameter in module.parameters()
        if parameter.grad is not None
    )
    return float(np.sqrt(squared_norm))


def compute_ppo_ratio(new_log_probs: Tensor, old_log_probs: Tensor) -> Tensor:
    """Compute the unchanged joint PPO probability ratio."""

    return torch.exp(new_log_probs - old_log_probs)


def _mean_episode_sum(values: list[float], episode_ends: list[bool]) -> float:
    sums: list[float] = []
    running = 0.0
    for value, episode_end in zip(values, episode_ends, strict=True):
        running += float(value)
        if episode_end:
            sums.append(running)
            running = 0.0
    if running or not sums:
        sums.append(running)
    return float(np.mean(sums))


def _finite_correlation(left: np.ndarray, right: list[float]) -> float:
    right_array = np.asarray(right, dtype=np.float64)
    if left.size != right_array.size or left.std() == 0.0 or right_array.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right_array)[0, 1])

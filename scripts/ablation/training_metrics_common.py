"""Shared helpers for Method A/B smoke metric scripts."""

from __future__ import annotations

from pathlib import Path
import math
import sys
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env.prebuilt_oran_ho_env import PrebuiltEpisodeWindowEnv
from src.env.oran_ho_env import OranHandoverEnv
from src.model.inference.post_process import ActionSpaceSpec
from src.model.gppo import GPPOActorCritic
from src.model.model import STGNNActorCritic
from src.model.causal_observation import LEGACY_OBSERVATION_FEATURE_MODE, rh_feature_dim
from src.model.runtime_modes import RewardMode, episode_for_reward_mode, validate_project_5split_episode
from src.model.training.scenario import (
    TrainingScenario,
    scenario_name_from_params,
    validate_same_topology,
)
from src.pipeline.data_pipeline import MockCsvDatasetLoader
from src.utils.config import DEFAULT_CONFIG_PATH, load_project_config


COMMON_FIELDNAMES = [
    "iteration",
    "observation_feature_mode",
    "observation_schema_version",
    "scenario_id",
    "scenario_name",
    "params_path",
    "difficulty",
    "loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "mean_reward",
    "mean_total_cost",
    "mean_handover_state_cost",
    "mean_link_violation_cost",
    "mean_compute_overload_cost",
    "infeasible_direct_rh_rc_paths",
    "direct_rh_rc_bw_shortfall_mbps",
    "direct_rh_rc_latency_excess_ms",
    "direct_rh_rc_feasible_rate",
    "feasible_rate",
    "connectivity_feasible_rate",
    "bandwidth_feasible_rate",
    "latency_feasible_rate",
    "compute_feasible_rate",
    "overall_feasible_rate",
    "avg_valid_action_count_per_rh",
    "reward_std",
    "rollout_total_cost_mean",
    "rollout_total_j_mean",
    "reward_input_cost_mean",
    "rollout_reward_input_j_mean",
    "weighted_value_term",
    "raw_entropy_sum",
    "entropy_per_rh",
    "scaled_entropy",
    "weighted_entropy_contribution",
    "ppo_ratio_p1",
    "ppo_ratio_p5",
    "ppo_ratio_p50",
    "ppo_ratio_p95",
    "ppo_ratio_p99",
    "clip_fraction",
    "actor_gradient_norm",
    "critic_gradient_norm",
    "total_gradient_norm",
    "advantage_mean_before_normalization",
    "advantage_std_before_normalization",
    "advantage_raw_cost_correlation",
    "return_mean",
    "return_std",
    "rollout_overall_feasible_rate",
    "rollout_bandwidth_feasible_rate",
    "rollout_compute_feasible_rate",
    "num_rh",
    "entropy_scaling_factor",
    "reward_scaling_factor",
    "entropy_scale_mode",
    "reward_graph_scale_mode",
]

TRAINING_DIAGNOSTIC_FIELDS = COMMON_FIELDNAMES[COMMON_FIELDNAMES.index("reward_std") :]

EVALUATION_DIAGNOSTIC_FIELDS = [
    "iteration",
    "eval_iteration",
    "observation_feature_mode",
    "observation_schema_version",
    "eval_total_j",
    "eval_j_per_rh",
    "eval_mean_reward",
    "eval_reward_std",
    "overall_feasible_rate",
    "bandwidth_feasible_rate",
    "compute_feasible_rate",
    "eval_compute_violation",
    "eval_bandwidth_violation",
    "eval_latency_violation",
    "action_disagreement_from_initial",
    "eval_entropy_per_rh",
    "approx_kl",
    "clip_fraction",
]
COMMON_FIELDNAMES.extend(
    field for field in EVALUATION_DIAGNOSTIC_FIELDS if field not in COMMON_FIELDNAMES
)


def build_prebuilt_scenario_env_and_model(
    dataset_dir: str | Path,
    *,
    params_path: str | Path,
    seed: int | None,
    episode_length: int,
    num_episodes: int,
    project_config_path: str | Path = DEFAULT_CONFIG_PATH,
    reward_mode: RewardMode | None = None,
    observation_feature_mode: str = LEGACY_OBSERVATION_FEATURE_MODE,
) -> tuple[dict[str, Any], list[TrainingScenario], STGNNActorCritic]:
    """Load an existing dataset directory without regenerating any CSVs."""

    project_config, scenario = build_prebuilt_scenario(
        dataset_dir, params_path=params_path, episode_length=episode_length,
        num_episodes=num_episodes, project_config_path=project_config_path,
        reward_mode=reward_mode, seed=seed, observation_feature_mode=observation_feature_mode,
    )
    episode = scenario.env.episode
    spec = ActionSpaceSpec.from_topology(episode.config)
    model = STGNNActorCritic(
        rh_feature_dim=rh_feature_dim(episode, observation_feature_mode),
        hidden_dim=int(project_config["model"]["hidden_dim"]),
        num_actions_per_rh=spec.num_actions_per_rh,
    )
    return project_config, [scenario], model


def build_prebuilt_scenario(
    dataset_dir: str | Path,
    *,
    params_path: str | Path,
    episode_length: int,
    num_episodes: int,
    project_config_path: str | Path = DEFAULT_CONFIG_PATH,
    reward_mode: RewardMode | None = None,
    seed: int | None = None,
    observation_feature_mode: str = LEGACY_OBSERVATION_FEATURE_MODE,
) -> tuple[dict[str, Any], TrainingScenario]:
    """Build the shared prebuilt environment used by recurrent and GPPO runs."""

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset_dir does not exist: {dataset_path}")
    loader = MockCsvDatasetLoader(dataset_path, project_config_path)
    # The loader supplies either the unchanged legacy compatibility mapping or
    # the resolved v3 Patch-1 runtime view.  Do not replace a v3 formulation
    # with the RL-only JSON's legacy compatibility defaults here.
    project_config = loader.project_config
    episode = loader.load_episode()
    if reward_mode is not None:
        validate_project_5split_episode(episode)
        episode = episode_for_reward_mode(episode, reward_mode)
    env = PrebuiltEpisodeWindowEnv(
        episode, episode_length=episode_length, num_episodes=num_episodes,
        project_config=project_config,
        episode_order_mode=(
            "shuffle_each_cycle"
            if episode.metadata.get("mobility_observation_schema_version") in {
                "lagged_observation_v2", "daily_288_lagged_observation_v3"
            }
            else "sequential"
        ),
        episode_order_seed=0 if seed is None else seed,
        observation_feature_mode=observation_feature_mode,
    )
    return project_config, TrainingScenario.from_env(
        scenario_id=0, scenario_name=scenario_name_from_params(params_path),
        params_path=str(params_path), env=env,
    )


def build_prebuilt_scenario_env_and_gppo_model(
    dataset_dir: str | Path,
    *,
    params_path: str | Path,
    seed: int | None,
    episode_length: int,
    num_episodes: int,
    reward_mode: RewardMode,
    project_config_path: str | Path = DEFAULT_CONFIG_PATH,
    observation_feature_mode: str = LEGACY_OBSERVATION_FEATURE_MODE,
) -> tuple[dict[str, Any], list[TrainingScenario], GPPOActorCritic]:
    """Load existing CSVs and build the project-compatible five-split GPPO baseline."""

    project_config, scenario = build_prebuilt_scenario(
        dataset_dir, params_path=params_path, episode_length=episode_length,
        num_episodes=num_episodes, project_config_path=project_config_path,
        reward_mode=reward_mode, seed=seed, observation_feature_mode=observation_feature_mode,
    )
    episode = scenario.env.episode
    spec = ActionSpaceSpec.from_topology(episode.config)
    model = GPPOActorCritic(
        rh_feature_dim=rh_feature_dim(episode, observation_feature_mode),
        hidden_dim=int(project_config["model"]["hidden_dim"]),
        num_actions_per_rh=spec.num_actions_per_rh,
    )
    return project_config, [scenario], model


def empty_eval_metrics() -> dict[str, float]:
    """Return placeholder evaluation metrics when greedy evaluation is skipped."""

    return {
        "eval_iteration": math.nan,
        "eval_total_j": math.nan,
        "eval_j_per_rh": math.nan,
        "eval_mean_reward": math.nan,
        "eval_reward_std": math.nan,
        "eval_compute_violation": math.nan,
        "eval_bandwidth_violation": math.nan,
        "eval_latency_violation": math.nan,
        "mean_total_cost": math.nan,
        "mean_handover_state_cost": math.nan,
        "mean_link_violation_cost": math.nan,
        "mean_compute_overload_cost": math.nan,
        "infeasible_direct_rh_rc_paths": math.nan,
        "direct_rh_rc_bw_shortfall_mbps": math.nan,
        "direct_rh_rc_latency_excess_ms": math.nan,
        "direct_rh_rc_feasible_rate": math.nan,
        "feasible_rate": math.nan,
        "connectivity_feasible_rate": math.nan,
        "bandwidth_feasible_rate": math.nan,
        "latency_feasible_rate": math.nan,
        "compute_feasible_rate": math.nan,
        "overall_feasible_rate": math.nan,
    }


def evaluate_greedy_policy(
    trainer: Any,
    seed: int,
    *,
    recurrent: bool = False,
    scenario_scope: str = "all",
) -> dict[str, float]:
    """Run deterministic greedy evaluation.

    scenario_scope controls diagnostic cost only:
    - all: evaluate every scenario and average metrics.
    - current: evaluate only the trainer's active scenario.
    - none: skip evaluation and return NaN placeholders.
    """

    if scenario_scope == "none":
        return empty_eval_metrics()
    if scenario_scope not in {"all", "current", "none"}:
        raise ValueError(f"unsupported scenario_scope={scenario_scope!r}")

    scenarios = getattr(trainer, "scenarios", None)
    if scenario_scope == "all" and scenarios and len(scenarios) > 1:
        previous_scenario = getattr(trainer, "current_scenario", None)
        scenario_metrics = []
        for scenario in scenarios:
            if hasattr(trainer, "_activate_scenario"):
                trainer._activate_scenario(scenario)
            scenario_metrics.append(_evaluate_greedy_current_scenario(trainer, seed, recurrent=recurrent))
        if previous_scenario is not None and hasattr(trainer, "_activate_scenario"):
            trainer._activate_scenario(previous_scenario)
        return _mean_metric_dicts(scenario_metrics)

    return _evaluate_greedy_current_scenario(trainer, seed, recurrent=recurrent)


def evaluate_greedy_policy_diagnostics(
    trainer: Any,
    seed: int,
    *,
    recurrent: bool = False,
    initial_actions: np.ndarray | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    """Evaluate one fixed scenario and return scientific metrics plus actions."""

    env = trainer.env
    was_training = trainer.model.training
    trainer.model.eval()
    infos: list[dict[str, Any]] = []
    rewards: list[float] = []
    actions: list[np.ndarray] = []
    entropy_per_rh: list[float] = []
    hidden_state = None
    with torch.no_grad():
        env.reset(seed=seed, options={"episode_id": 0})
        done = False
        while not done:
            time_index = env.time_index
            if recurrent:
                logits, _, next_hidden_state = trainer._model_at(time_index, hidden_state)
            else:
                logits, _ = trainer._model_at(time_index)
                next_hidden_state = None
            masked_logits = trainer._masked_logits(logits.squeeze(0))
            distribution = Categorical(logits=masked_logits)
            action = masked_logits.argmax(dim=-1).detach().cpu().numpy()
            _, reward, terminated, truncated, info = env.step(action)
            infos.append(info)
            rewards.append(float(reward))
            actions.append(action.copy())
            entropy_per_rh.append(float(distribution.entropy().mean().cpu()))
            done = terminated or truncated
            hidden_state = None if done else (
                next_hidden_state.detach() if next_hidden_state is not None else None
            )
    if was_training:
        trainer.model.train()

    action_array = np.stack(actions)
    disagreement = (
        0.0
        if initial_actions is None
        else float(np.mean(action_array != initial_actions))
    )
    metrics = {
        "mean_total_cost": _mean_info(infos, "total_cost"),
        "mean_handover_state_cost": _mean_info(infos, "handover_state_transfer_cost"),
        "mean_link_violation_cost": _mean_info(infos, "link_violation_cost"),
        "mean_compute_overload_cost": _mean_info(infos, "compute_overload_cost"),
        **direct_rh_rc_violation_metrics(infos),
        "feasible_rate": float(np.mean([bool(info.get("feasible", False)) for info in infos])),
        **constraint_feasible_rates(infos),
        "eval_total_j": float(sum(float(info["total_cost"]) for info in infos)),
        "eval_j_per_rh": float(
            sum(float(info["total_cost"]) for info in infos) / env.episode.config.num_rhs
        ),
        "eval_mean_reward": float(np.mean(rewards)),
        "eval_reward_std": float(np.std(rewards)),
        **evaluation_violation_means(infos),
        "action_disagreement_from_initial": disagreement,
        "eval_entropy_per_rh": float(np.mean(entropy_per_rh)),
    }
    return metrics, action_array


def _evaluate_greedy_current_scenario(trainer: Any, seed: int, *, recurrent: bool = False) -> dict[str, float]:
    env = trainer.env
    was_training = trainer.model.training
    trainer.model.eval()
    infos: list[dict[str, Any]] = []
    rewards: list[float] = []
    hidden_state = None
    with torch.no_grad():
        env.reset(seed=seed)
        done = False
        while not done:
            time_index = env.time_index
            if recurrent:
                logits, _, next_hidden_state = trainer._model_at(time_index, hidden_state)
            else:
                logits, _ = trainer._model_at(time_index)
                next_hidden_state = None
            action = trainer._masked_logits(logits.squeeze(0)).argmax(dim=-1).detach().cpu().numpy()
            _, reward, terminated, truncated, info = env.step(action)
            infos.append(info)
            rewards.append(float(reward))
            done = terminated or truncated
            hidden_state = None if done else (
                next_hidden_state.detach() if next_hidden_state is not None else None
            )
    if was_training:
        trainer.model.train()

    total_j = float(sum(float(info["total_cost"]) for info in infos))
    return {
        "mean_total_cost": _mean_info(infos, "total_cost"),
        "mean_handover_state_cost": _mean_info(infos, "handover_state_transfer_cost"),
        "mean_link_violation_cost": _mean_info(infos, "link_violation_cost"),
        "mean_compute_overload_cost": _mean_info(infos, "compute_overload_cost"),
        **direct_rh_rc_violation_metrics(infos),
        "feasible_rate": float(np.mean([1.0 if info.get("feasible", False) else 0.0 for info in infos]))
        if infos
        else math.nan,
        **constraint_feasible_rates(infos),
        "eval_total_j": total_j,
        "eval_j_per_rh": total_j / env.episode.config.num_rhs,
        "eval_mean_reward": float(np.mean(rewards)) if rewards else math.nan,
        "eval_reward_std": float(np.std(rewards)) if rewards else math.nan,
        **evaluation_violation_means(infos),
    }


def infer_difficulty(params_path: str | Path) -> str:
    name = Path(params_path).stem.lower()
    for difficulty in ("easy", "medium", "hard"):
        if difficulty in name:
            return difficulty
    return "unknown"


def scenario_metric_fields(metrics: dict[str, Any], fallback_params: str | Path) -> dict[str, Any]:
    params_path = str(metrics.get("params_path", fallback_params))
    return {
        "scenario_id": metrics.get("scenario_id", 0.0),
        "scenario_name": metrics.get("scenario_name", scenario_name_from_params(params_path)),
        "params_path": params_path,
        "difficulty": infer_difficulty(params_path),
    }


def constraint_feasible_rates(infos: list[dict[str, Any]]) -> dict[str, float]:
    if not infos:
        return {
            "connectivity_feasible_rate": math.nan,
            "bandwidth_feasible_rate": math.nan,
            "latency_feasible_rate": math.nan,
            "compute_feasible_rate": math.nan,
            "overall_feasible_rate": math.nan,
        }

    connectivity_ok = []
    bandwidth_ok = []
    latency_ok = []
    compute_ok = []
    overall_ok = []
    for info in infos:
        violations = info.get("violations", {})
        connectivity_ok.append(
            _violation_sum(
                violations,
                (
                    "invalid_action",
                    "infeasible_fronthaul_paths",
                    "infeasible_midhaul_paths",
                    "infeasible_direct_rh_rc_paths",
                ),
            )
            <= 0.0
        )
        bandwidth_ok.append(
            _violation_sum(
                violations,
                ("fronthaul_bw_shortfall_mbps", "midhaul_bw_shortfall_mbps", "direct_rh_rc_bw_shortfall_mbps"),
            )
            <= 0.0
        )
        latency_ok.append(
            _violation_sum(
                violations,
                ("end_to_end_latency_excess_ms",),
            )
            <= 0.0
        )
        compute_ok.append(
            _violation_sum(violations, ("es_compute_overload", "rc_compute_overload")) <= 0.0
        )
        overall_ok.append(bool(info.get("feasible", False)))

    return {
        "connectivity_feasible_rate": float(np.mean(connectivity_ok)),
        "bandwidth_feasible_rate": float(np.mean(bandwidth_ok)),
        "latency_feasible_rate": float(np.mean(latency_ok)),
        "compute_feasible_rate": float(np.mean(compute_ok)),
        "overall_feasible_rate": float(np.mean(overall_ok)),
    }


def direct_rh_rc_violation_metrics(infos: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate direct RH->RC violation diagnostics used by S4D actions."""

    if not infos:
        return {
            "infeasible_direct_rh_rc_paths": math.nan,
            "direct_rh_rc_bw_shortfall_mbps": math.nan,
            "direct_rh_rc_latency_excess_ms": math.nan,
            "direct_rh_rc_feasible_rate": math.nan,
        }
    path_values = []
    bw_values = []
    latency_values = []
    direct_ok = []
    for info in infos:
        violations = info.get("violations", {})
        path = float(violations.get("infeasible_direct_rh_rc_paths", 0.0))
        bw = float(violations.get("direct_rh_rc_bw_shortfall_mbps", 0.0))
        latency = float(violations.get("direct_rh_rc_latency_excess_ms", 0.0))
        path_values.append(path)
        bw_values.append(bw)
        latency_values.append(latency)
        direct_ok.append(path + bw + latency <= 0.0)
    return {
        "infeasible_direct_rh_rc_paths": float(np.mean(path_values)),
        "direct_rh_rc_bw_shortfall_mbps": float(np.mean(bw_values)),
        "direct_rh_rc_latency_excess_ms": float(np.mean(latency_values)),
        "direct_rh_rc_feasible_rate": float(np.mean(direct_ok)),
    }


def evaluation_violation_means(infos: list[dict[str, Any]]) -> dict[str, float]:
    """Return mean raw evaluator violations for one greedy evaluation."""

    if not infos:
        return {
            "eval_compute_violation": math.nan,
            "eval_bandwidth_violation": math.nan,
            "eval_latency_violation": math.nan,
        }
    return {
        "eval_compute_violation": float(np.mean([
            _violation_sum(info.get("violations", {}), ("es_compute_overload", "rc_compute_overload"))
            for info in infos
        ])),
        "eval_bandwidth_violation": float(np.mean([
            _violation_sum(
                info.get("violations", {}),
                ("fronthaul_bw_shortfall_mbps", "midhaul_bw_shortfall_mbps", "direct_rh_rc_bw_shortfall_mbps"),
            )
            for info in infos
        ])),
        "eval_latency_violation": float(np.mean([
            _violation_sum(info.get("violations", {}), ("end_to_end_latency_excess_ms",))
            for info in infos
        ])),
    }


def _mean_info(infos: list[dict[str, Any]], key: str) -> float:
    values = [float(info[key]) for info in infos if key in info]
    return float(np.mean(values)) if values else math.nan


def _violation_sum(violations: dict[str, Any], keys: tuple[str, ...]) -> float:
    return float(sum(float(violations.get(key, 0.0)) for key in keys))


def _mean_metric_dicts(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = sorted(set().union(*(metric.keys() for metric in metrics)))
    averaged: dict[str, float] = {}
    for key in keys:
        values = [float(metric[key]) for metric in metrics if key in metric and not math.isnan(float(metric[key]))]
        averaged[key] = float(np.mean(values)) if values else math.nan
    return averaged


def format_metric(value: float) -> str:
    if math.isnan(float(value)):
        return "nan"
    return f"{float(value):.4f}"

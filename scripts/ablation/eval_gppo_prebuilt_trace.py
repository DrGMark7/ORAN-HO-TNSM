"""Evaluate a project-compatible five-split GPPO checkpoint on prebuilt data."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .eval_recurrent_prebuilt_trace import (
        PrebuiltTraceResult,
        RESOURCE_EXTRA_COLUMNS,
        REPAIR_ACTION_COLUMNS,
        TRACE_STEP_COLUMNS,
        _augment_resource_rows,
        _episode_summary_columns,
        _episode_summary_row,
        _eval_summary,
        _main_violation_reason,
        _metadata_warning,
        _resolve_episode_ids,
        _sanity_check_exported_files_prebuilt,
        _sanity_check_selected_trace_rows,
        _sanity_check_trace_steps,
)
from .infer_episode_trace import (
        DIRECT_RH_RC_LINK_LOAD_COLUMNS,
        ES_LOAD_COLUMNS,
        MIDHAUL_LINK_LOAD_COLUMNS,
        RC_LOAD_COLUMNS,
        _assert_actions_match_mask,
        _direct_rh_rc_link_load_rows,
        _es_load_rows,
        _masked_logits,
        _midhaul_link_load_rows,
        _rc_load_rows,
        _rh_decision_rows,
        _step_feasibility,
        _write_csv,
        load_model_checkpoint,
        load_runtime_episode,
        validate_constraint_formulation_metadata,
        validate_normalization_metadata,
)

from src.env.prebuilt_oran_ho_env import PrebuiltEpisodeWindowEnv
from src.model.gppo import GPPOActorCritic
from src.model.inference.post_process import (
    ActionSpaceSpec,
    action_mask_kwargs_from_episode,
    build_link_feasibility_action_mask,
    build_topology_profile_action_masks,
    decode_flat_actions,
    topology_action_mask_for_time_index,
)
from src.model.inference.infeasibility_repair import repair_topk_action, repair_topk_action_deadline
from src.model.runtime_modes import (
    FEATURE_MODES,
    REWARD_MODES,
    SUPPORTED_SPLIT_MODE,
    episode_for_reward_mode,
    snapshot_for_feature_mode,
    validate_gppo_modes,
    validate_project_5split_episode,
)
from src.model.training.ppo import resolve_device
from src.model.causal_observation import (
    OBSERVATION_FEATURE_MODES,
    LEGACY_OBSERVATION_FEATURE_MODE,
    RESOURCE_OBSERVATION_FEATURE_MODES,
    append_causal_features,
    observation_schema_for_mode,
    rh_feature_dim,
    rh_feature_names,
)
from src.pipeline.data_pipeline import HandoverGraphPipeline
from src.utils.config import DEFAULT_CONFIG_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a project-compatible five-split GPPO checkpoint")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episode_length", type=int, required=True)
    parser.add_argument("--num_episodes", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--episodes", default="all")
    parser.add_argument("--policy", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--infeasibility-repair", choices=("none", "topk"), default="none")
    parser.add_argument("--repair-top-k", type=int, default=5)
    parser.add_argument("--repair-max-iterations", type=int, default=5)
    parser.add_argument("--repair-beam-width", type=int, default=4)
    parser.add_argument("--repair-mode", choices=("bounded", "deadline"), default="bounded")
    parser.add_argument("--repair-decision-budget-seconds", type=float)
    parser.add_argument(
        "--allow-schema-mismatch",
        action="store_true",
        help="record an observation-schema mismatch as a warning instead of failing after evaluation",
    )
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--feature_mode", choices=FEATURE_MODES, default=None)
    parser.add_argument("--reward_mode", choices=REWARD_MODES, default=None)
    parser.add_argument("--split_mode", default=SUPPORTED_SPLIT_MODE)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--observation_feature_mode", choices=OBSERVATION_FEATURE_MODES,
                        default=LEGACY_OBSERVATION_FEATURE_MODE)
    parser.add_argument("--trust-checkpoint", action="store_true",
                        help="allow unsafe pickle loading for checkpoints from a trusted source")
    return parser.parse_args()


def run_prebuilt_gppo_trace(
    *,
    dataset_dir: str | Path,
    params_path: str | Path,
    checkpoint_path: str | Path,
    episode_length: int,
    num_episodes: int,
    output_dir: str | Path,
    device: str | torch.device = "auto",
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    episodes: str = "all",
    policy: str = "greedy",
    max_episodes: int | None = None,
    seed: int = 13,
    feature_mode: str | None = None,
    reward_mode: str | None = None,
    split_mode: str = SUPPORTED_SPLIT_MODE,
    verbose: bool = False,
    allow_schema_mismatch: bool = False,
    observation_feature_mode: str = LEGACY_OBSERVATION_FEATURE_MODE,
    trust_checkpoint: bool = False,
    infeasibility_repair: str = "none",
    repair_top_k: int = 5,
    repair_max_iterations: int = 5,
    repair_beam_width: int = 4,
    repair_mode: str = "bounded",
    repair_decision_budget_seconds: float | None = None,
    runtime_context: dict[str, Any] | None = None,
) -> PrebuiltTraceResult:
    dataset_path = Path(dataset_dir)
    checkpoint_path = Path(checkpoint_path)
    output_root = Path(output_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset_dir does not exist: {dataset_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    if infeasibility_repair not in {"none", "topk"}:
        raise ValueError("infeasibility_repair must be 'none' or 'topk'")
    if repair_mode not in {"bounded", "deadline"}:
        raise ValueError("repair_mode must be 'bounded' or 'deadline'")
    if repair_mode == "deadline" and (
        infeasibility_repair != "topk"
        or repair_decision_budget_seconds is None
        or not np.isfinite(repair_decision_budget_seconds)
        or repair_decision_budget_seconds <= 0
    ):
        raise ValueError("deadline repair requires topk repair and a positive finite decision budget")

    # A persistent benchmark worker can supply these already-initialised
    # objects.  The evaluation/export loop below is deliberately unchanged so
    # its artifacts remain byte-schema compatible with the normal CLI path.
    if runtime_context is not None:
        torch_device = runtime_context["torch_device"]
        loaded_episode = runtime_context["loaded_episode"]
        project_config = runtime_context["project_config"]
        spec = runtime_context["spec"]
        model = runtime_context["model"]
        checkpoint_metadata = runtime_context["checkpoint_metadata"]
        selected_feature_mode = runtime_context["feature_mode"]
        selected_reward_mode = runtime_context["reward_mode"]
        selected_split_mode = runtime_context["split_mode"]
        episode = runtime_context["episode"]
        env = runtime_context["env"]
        graph_pipeline = runtime_context["graph_pipeline"]
        transform = runtime_context["transform"]
        topology_profile_action_masks = runtime_context["topology_profile_action_masks"]
    else:
        torch_device = resolve_device(device)
        np.random.seed(seed)
        torch.manual_seed(seed)
        loaded_episode, project_config = load_runtime_episode(dataset_path, config_path)
        validate_project_5split_episode(loaded_episode)
        spec = ActionSpaceSpec.from_topology(loaded_episode.config)
        model = GPPOActorCritic(
        rh_feature_dim=rh_feature_dim(loaded_episode, observation_feature_mode),
        hidden_dim=int(project_config["model"]["hidden_dim"]),
        num_actions_per_rh=spec.num_actions_per_rh,
        ).to(torch_device)
        checkpoint_metadata = load_model_checkpoint(
        checkpoint_path, model, torch_device,
        trust_checkpoint=trust_checkpoint,
        expected_observation_feature_mode=observation_feature_mode,
        expected_observation_schema_version=observation_schema_for_mode(
            loaded_episode, observation_feature_mode
        ),
        expected_observation_feature_names=rh_feature_names(
            loaded_episode, observation_feature_mode
        ),
        )
        validate_constraint_formulation_metadata(checkpoint_metadata, project_config)
        validate_normalization_metadata(checkpoint_metadata, project_config)
        saved_metadata = checkpoint_metadata.get("metadata", checkpoint_metadata)
        selected_feature_mode = feature_mode or saved_metadata.get("feature_mode", "handover")
        selected_reward_mode = reward_mode or saved_metadata.get("reward_mode", "handover_aware")
        selected_split_mode = saved_metadata.get("split_mode", split_mode)
        validate_gppo_modes(selected_feature_mode, selected_reward_mode, selected_split_mode)

        episode = episode_for_reward_mode(loaded_episode, selected_reward_mode)
        env = PrebuiltEpisodeWindowEnv(
        episode, episode_length=episode_length, num_episodes=num_episodes,
        project_config=project_config,
        observation_feature_mode=observation_feature_mode,
        )
        graph_pipeline = HandoverGraphPipeline(episode)
        transform = partial(snapshot_for_feature_mode, feature_mode=selected_feature_mode)
        topology_profile_action_masks = build_topology_profile_action_masks(episode)
    link_action_mask = build_link_feasibility_action_mask(
        episode.config,
        episode.fronthaul_mask,
        episode.midhaul_mask,
        episode.direct_rh_rc_mask,
        episode.split_demand.split_names,
        **action_mask_kwargs_from_episode(episode),
    )
    selected_episode_ids = _resolve_episode_ids(env, episodes=episodes, max_episodes=max_episodes)
    model.eval()

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "rh").mkdir(parents=True, exist_ok=True)
    rh_rows: list[dict[str, Any]] = []
    es_rows: list[dict[str, Any]] = []
    rc_rows: list[dict[str, Any]] = []
    midhaul_rows: list[dict[str, Any]] = []
    direct_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    repair_action_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for episode_id in selected_episode_ids:
            env.reset(seed=seed, options={"episode_id": episode_id})
            link_action_mask = topology_action_mask_for_time_index(
                episode,
                episode_id * episode_length,
                episode_length=episode_length,
                profile_masks=topology_profile_action_masks,
            )
            step_rows: list[dict[str, Any]] = []
            episode_rh_rows: list[dict[str, Any]] = []
            done = False
            while not done:
                absolute_t = env.time_index
                local_t = env.current_local_t
                graph = graph_pipeline.build_snapshot(absolute_t).to(torch_device)
                if observation_feature_mode in RESOURCE_OBSERVATION_FEATURE_MODES:
                    graph = append_causal_features(
                        graph, env.causal_rh_features(), episode, observation_feature_mode
                    ).to(torch_device)
                graph = transform(graph)
                handover = graph[("rh", "handover", "rh")]
                # This covers only policy decision work: neural inference, action
                # selection, and any Top-K repair.  In particular, it deliberately
                # ends before ``env.step`` so environment/evaluator advancement is
                # not charged to decision latency.
                decision_start_ns = time.perf_counter_ns()
                logits, _ = model(
                    graph["rh"].x.unsqueeze(0).unsqueeze(0),
                    [handover.edge_index],
                    [handover.edge_attr],
                )
                masked_logits = _masked_logits(
                    logits.squeeze(0), link_action_mask, episode.config.num_rhs, spec.num_actions_per_rh
                )
                action = masked_logits.argmax(dim=-1) if policy == "greedy" else Categorical(logits=masked_logits).sample()
                action_np = action.detach().cpu().numpy()
                repair = None
                if policy == "greedy" and infeasibility_repair == "topk":
                    repair_kwargs = dict(
                        masked_logits=masked_logits.detach().cpu().numpy(), connectivity_mask=link_action_mask,
                        evaluator=env.evaluator, time_index=absolute_t, previous_action=env.previous_action,
                        top_k=repair_top_k, max_iterations=repair_max_iterations, beam_width=repair_beam_width,
                    )
                    if repair_mode == "deadline":
                        assert decision_start_ns is not None and repair_decision_budget_seconds is not None
                        repair = repair_topk_action_deadline(
                            **repair_kwargs,
                            decision_start_ns=decision_start_ns,
                            deadline_ns=decision_start_ns + int(repair_decision_budget_seconds * 1_000_000_000),
                        )
                    else:
                        repair = repair_topk_action(**repair_kwargs)
                    action_np = repair.actions
                _assert_actions_match_mask(action_np, link_action_mask)
                placement = decode_flat_actions(action_np, episode.config)
                decision_ms = (time.perf_counter_ns() - decision_start_ns) / 1_000_000.0
                _, reward, terminated, truncated, info = env.step(action_np)
                feasibility = _step_feasibility(info)
                step_rows.append({"reward": reward, "decision_ms": decision_ms,
                                  **info, **feasibility, **(repair.diagnostics if repair else {})})

                for row in _rh_decision_rows(absolute_t, episode, placement, reward, info, feasibility):
                    rh_id = int(row["rh_id"])
                    augmented = {
                        **row,
                        "episode_id": episode_id,
                        "local_t": local_t,
                        "absolute_t": absolute_t,
                        "episode_length": episode_length,
                        "num_episodes": num_episodes,
                        "reset_boundary": local_t == 0,
                        "action_id": int(action_np[rh_id]),
                        "decision_ms": decision_ms,
                        "feasible": bool(info.get("feasible", False)),
                        "connectivity_feasible": bool(feasibility["connectivity_ok"]),
                        "latency_feasible": bool(feasibility["latency_ok"]),
                        "bandwidth_feasible": bool(feasibility["bandwidth_ok"]),
                        "compute_feasible": bool(feasibility["compute_ok"]),
                        "main_violation_reason": _main_violation_reason(info),
                    }
                    if repair is not None:
                        augmented.update(repair.diagnostics)
                    episode_rh_rows.append(augmented)
                    rh_rows.append(augmented)
                if repair is not None:
                    repair_action_rows.extend(
                        {"episode_id": episode_id, "timestep": local_t, "absolute_t": absolute_t, **change}
                        for change in repair.changes
                    )

                extra = dict(
                    episode_id=episode_id,
                    local_t=local_t,
                    absolute_t=absolute_t,
                    episode_length=episode_length,
                    num_episodes=num_episodes,
                )
                es_rows.extend(_augment_resource_rows(_es_load_rows(absolute_t, episode, placement), **extra))
                rc_rows.extend(_augment_resource_rows(_rc_load_rows(absolute_t, episode, placement), **extra))
                midhaul_rows.extend(
                    _augment_resource_rows(_midhaul_link_load_rows(absolute_t, episode, placement), **extra)
                )
                direct_rows.extend(
                    _augment_resource_rows(_direct_rh_rc_link_load_rows(absolute_t, episode, placement), **extra)
                )
                done = terminated or truncated

            summary_rows.append(
                _episode_summary_row(
                    episode=episode, episode_id=episode_id, step_rows=step_rows, rh_rows=episode_rh_rows
                )
            )

    _sanity_check_selected_trace_rows(
        episode,
        selected_episode_ids=selected_episode_ids,
        rh_rows=rh_rows,
        es_rows=es_rows,
        rc_rows=rc_rows,
        midhaul_link_rows=midhaul_rows,
        direct_link_rows=direct_rows,
        episode_length=episode_length,
    )
    _sanity_check_trace_steps(rh_rows, selected_episode_ids, episode.config.num_rhs, episode_length)

    paths = PrebuiltTraceResult(
        trace_steps=output_root / "trace_steps.csv",
        rh_decisions=output_root / "rh_decisions.csv",
        es_loads=output_root / "es_loads.csv",
        rc_loads=output_root / "rc_loads.csv",
        midhaul_link_loads=output_root / "midhaul_link_loads.csv",
        direct_rh_rc_link_loads=output_root / "direct_rh_rc_link_loads.csv",
        episode_summary=output_root / "episode_summary.csv",
        eval_summary=output_root / "eval_summary.json",
    )
    _write_csv(paths.trace_steps, rh_rows, TRACE_STEP_COLUMNS)
    _write_csv(paths.rh_decisions, rh_rows, TRACE_STEP_COLUMNS)
    for rh_id in range(episode.config.num_rhs):
        _write_csv(
            output_root / "rh" / f"RH_{rh_id}.csv",
            [row for row in rh_rows if int(row["rh_id"]) == rh_id],
            TRACE_STEP_COLUMNS,
        )
    _write_csv(paths.es_loads, es_rows, ES_LOAD_COLUMNS + RESOURCE_EXTRA_COLUMNS)
    _write_csv(paths.rc_loads, rc_rows, RC_LOAD_COLUMNS + RESOURCE_EXTRA_COLUMNS)
    _write_csv(paths.midhaul_link_loads, midhaul_rows, MIDHAUL_LINK_LOAD_COLUMNS + RESOURCE_EXTRA_COLUMNS)
    _write_csv(
        paths.direct_rh_rc_link_loads,
        direct_rows,
        DIRECT_RH_RC_LINK_LOAD_COLUMNS + RESOURCE_EXTRA_COLUMNS,
    )
    _write_csv(paths.episode_summary, summary_rows, _episode_summary_columns(episode))
    _write_csv(output_root / "repair_actions.csv", repair_action_rows, REPAIR_ACTION_COLUMNS)
    warnings = []
    warning = _metadata_warning(
        checkpoint_metadata, episode_length, num_episodes,
        dataset_schema=str(episode.metadata.get("mobility_observation_schema_version", "legacy_same_step_v1")),
        allow_schema_mismatch=allow_schema_mismatch,
    )
    if warning:
        warnings.append(warning)
    eval_summary = _eval_summary(
        dataset_dir=dataset_path,
        checkpoint_path=checkpoint_path,
        params_path=Path(params_path),
        env=env,
        selected_episode_ids=selected_episode_ids,
        episode_summary_rows=summary_rows,
        checkpoint_metadata=checkpoint_metadata,
        warning_messages=warnings,
        hidden_state_reset_count=0,
    )
    eval_summary.update(
        {
            "model_type": "gppo",
            "split_mode": selected_split_mode,
            "feature_mode": selected_feature_mode,
            "reward_mode": selected_reward_mode,
            "observation_feature_mode": observation_feature_mode,
            "observation_schema_version": observation_schema_for_mode(
                episode, observation_feature_mode
            ),
            "infeasibility_repair": infeasibility_repair,
            "repair_top_k": repair_top_k,
            "repair_max_iterations": repair_max_iterations,
            "repair_beam_width": repair_beam_width,
            "repair_mode": repair_mode,
            "repair_decision_budget_seconds": repair_decision_budget_seconds,
        }
    )
    paths.eval_summary.write_text(json.dumps(eval_summary, indent=2, sort_keys=True), encoding="utf-8")
    _sanity_check_exported_files_prebuilt(
        output_root,
        episode=episode,
        selected_episode_count=len(selected_episode_ids),
        episode_length=episode_length,
    )
    if verbose:
        print(f"evaluated episodes={selected_episode_ids} feature_mode={selected_feature_mode} reward_mode={selected_reward_mode}")
    return paths


def main() -> None:
    args = parse_args()
    result = run_prebuilt_gppo_trace(
        dataset_dir=args.dataset_dir,
        params_path=args.params,
        checkpoint_path=args.checkpoint,
        episode_length=args.episode_length,
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        device=args.device,
        config_path=args.config,
        episodes=args.episodes,
        policy=args.policy,
        max_episodes=args.max_episodes,
        seed=args.seed,
        feature_mode=args.feature_mode,
        reward_mode=args.reward_mode,
        split_mode=args.split_mode,
        verbose=args.verbose,
        observation_feature_mode=args.observation_feature_mode,
        trust_checkpoint=args.trust_checkpoint,
        infeasibility_repair=args.infeasibility_repair,
        repair_top_k=args.repair_top_k,
        repair_max_iterations=args.repair_max_iterations,
        repair_beam_width=args.repair_beam_width,
        repair_mode=args.repair_mode,
        repair_decision_budget_seconds=args.repair_decision_budget_seconds,
        allow_schema_mismatch=args.allow_schema_mismatch,
    )
    print(f"wrote trace steps to {result.trace_steps}")
    print(f"wrote evaluation summary to {result.eval_summary}")


if __name__ == "__main__":
    main()

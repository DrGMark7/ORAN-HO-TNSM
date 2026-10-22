"""Evaluate a recurrent checkpoint on a prebuilt flat mock-data dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any
from functools import partial

import numpy as np
import torch
from torch.distributions import Categorical


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .infer_episode_trace import (
    DIRECT_RH_RC_LINK_LOAD_COLUMNS,
    ES_LOAD_COLUMNS,
    MIDHAUL_LINK_LOAD_COLUMNS,
    RC_LOAD_COLUMNS,
    RH_DECISION_COLUMNS,
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
from src.model.inference.post_process import (
    ActionSpaceSpec,
    action_mask_kwargs_from_episode,
    build_link_feasibility_action_mask,
    build_topology_profile_action_masks,
    decode_flat_actions,
    topology_action_mask_for_time_index,
)
from src.model.inference.infeasibility_repair import repair_topk_action, repair_topk_action_deadline
from src.model.model import (
    STGNNActorCritic,
    STGNN_ARCHITECTURE_VERSION,
    STGNN_HIDDEN_RESET_MODE,
    STGNN_PPO_REPLAY_MODE,
    STGNN_TEMPORAL_FUSION_MODE,
)
from src.model.runtime_modes import snapshot_for_feature_mode
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
from src.pipeline.schema import MockEpisode, PlacementAction
from src.utils.config import DEFAULT_CONFIG_PATH


LEGACY_COMPATIBLE_FILES = [
    "rh_decisions.csv",
    "es_loads.csv",
    "rc_loads.csv",
    "midhaul_link_loads.csv",
    "direct_rh_rc_link_loads.csv",
    "episode_summary.csv",
    "rh/",
]
PREBUILT_SPECIFIC_FILES = [
    "trace_steps.csv",
    "eval_summary.json",
]


@dataclass(frozen=True)
class PrebuiltTraceResult:
    trace_steps: Path
    rh_decisions: Path
    es_loads: Path
    rc_loads: Path
    midhaul_link_loads: Path
    direct_rh_rc_link_loads: Path
    episode_summary: Path
    eval_summary: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a recurrent checkpoint on a prebuilt mock-data dataset")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episode_length", type=int, required=True)
    parser.add_argument("--num_episodes", type=int, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda device such as cuda:0")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--episodes", default="all", help="'all' or a comma-separated list such as 0,1,2")
    parser.add_argument("--policy", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--infeasibility-repair", choices=("none", "topk"), default="none")
    parser.add_argument("--repair-top-k", type=int, default=5)
    parser.add_argument("--repair-max-iterations", type=int, default=5)
    parser.add_argument("--repair-beam-width", type=int, default=4)
    parser.add_argument("--repair-mode", choices=("bounded", "deadline"), default="bounded")
    parser.add_argument("--repair-decision-budget-seconds", type=float)
    parser.add_argument(
        "--allow-schema-mismatch", action="store_true",
        help="record a checkpoint/dataset schema mismatch as a warning instead of failing",
    )
    parser.add_argument(
        "--temporal_mode",
        choices=("recurrent", "reset_each_step"),
        default="recurrent",
    )
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--observation_feature_mode", choices=OBSERVATION_FEATURE_MODES,
                        default=LEGACY_OBSERVATION_FEATURE_MODE)
    parser.add_argument("--trust-checkpoint", action="store_true",
                        help="allow unsafe pickle loading for checkpoints from a trusted source")
    return parser.parse_args()


def run_prebuilt_recurrent_trace(
    *,
    dataset_dir: str | Path,
    params_path: str | Path,
    checkpoint_path: str | Path,
    episode_length: int,
    num_episodes: int,
    device: str | torch.device = "auto",
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path,
    episodes: str = "all",
    policy: str = "greedy",
    max_episodes: int | None = None,
    seed: int = 13,
    verbose: bool = False,
    allow_schema_mismatch: bool = False,
    temporal_mode: str = "recurrent",
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
    checkpoint = Path(checkpoint_path)
    output_root = Path(output_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"--dataset_dir does not exist: {dataset_path}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"--checkpoint does not exist: {checkpoint}")
    if policy not in {"greedy", "sample"}:
        raise ValueError("--policy must be one of: greedy, sample")
    if temporal_mode not in {"recurrent", "reset_each_step"}:
        raise ValueError("temporal_mode must be recurrent or reset_each_step")
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

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "rh").mkdir(parents=True, exist_ok=True)

    # Persistent workers inject an already-loaded model and dataset runtime.
    # The standard evaluator/export loop below stays authoritative.
    if runtime_context is not None:
        torch_device = runtime_context["torch_device"]
        episode = runtime_context["episode"]
        project_config = runtime_context["project_config"]
        env = runtime_context["env"]
        spec = runtime_context["spec"]
        model = runtime_context["model"]
        checkpoint_metadata = runtime_context["checkpoint_metadata"]
        transform = runtime_context["transform"]
        graph_pipeline = runtime_context["graph_pipeline"]
        topology_profile_action_masks = runtime_context["topology_profile_action_masks"]
    else:
        torch_device = resolve_device(device)
        np.random.seed(seed)
        torch.manual_seed(seed)

        episode, project_config = load_runtime_episode(dataset_path, config_path)
        env = PrebuiltEpisodeWindowEnv(
        episode,
        episode_length=episode_length,
        num_episodes=num_episodes,
        project_config=project_config,
        observation_feature_mode=observation_feature_mode,
        )
        spec = ActionSpaceSpec.from_topology(episode.config)
        model = STGNNActorCritic(
        rh_feature_dim=rh_feature_dim(episode, observation_feature_mode),
        hidden_dim=int(project_config["model"]["hidden_dim"]),
        num_actions_per_rh=spec.num_actions_per_rh,
        ).to(torch_device)
        checkpoint_metadata = load_model_checkpoint(
        checkpoint, model, torch_device,
        trust_checkpoint=trust_checkpoint,
        expected_observation_feature_mode=observation_feature_mode,
        expected_observation_schema_version=observation_schema_for_mode(
            episode, observation_feature_mode
        ),
        expected_observation_feature_names=rh_feature_names(
            episode, observation_feature_mode
        ),
        )
        validate_constraint_formulation_metadata(checkpoint_metadata, project_config)
        validate_normalization_metadata(checkpoint_metadata, project_config)
        saved_metadata = checkpoint_metadata.get("metadata", checkpoint_metadata)
        validate_stgnn_recurrent_metadata(
        saved_metadata,
        episode_length=episode_length,
        hidden_dim=int(project_config["model"]["hidden_dim"]),
        observation_schema_version=observation_schema_for_mode(
            episode, observation_feature_mode
        ),
        )
        feature_mode = saved_metadata.get("feature_mode", "handover")
        transform = partial(snapshot_for_feature_mode, feature_mode=feature_mode)
        graph_pipeline = HandoverGraphPipeline(episode)
        topology_profile_action_masks = build_topology_profile_action_masks(episode)
    model.eval()

    selected_episode_ids = _resolve_episode_ids(env, episodes=episodes, max_episodes=max_episodes)
    link_action_mask = build_link_feasibility_action_mask(
        episode.config,
        episode.fronthaul_mask,
        episode.midhaul_mask,
        episode.direct_rh_rc_mask,
        episode.split_demand.split_names,
        **action_mask_kwargs_from_episode(episode),
    )

    if verbose:
        print(
            f"mode=recurrent-prebuilt-eval dataset_dir={dataset_path} checkpoint={checkpoint} "
            f"params={params_path} T={episode.config.horizon} episode_length={episode_length} "
            f"num_episodes={num_episodes} evaluated_episodes={selected_episode_ids} policy={policy} "
            f"temporal_mode={temporal_mode} device={torch_device}"
        )

    warning_messages: list[str] = []
    metadata_warning = _metadata_warning(
        checkpoint_metadata, episode_length, num_episodes,
        dataset_schema=str(episode.metadata.get("mobility_observation_schema_version", "legacy_same_step_v1")),
        allow_schema_mismatch=allow_schema_mismatch,
    )
    if metadata_warning is not None:
        warning_messages.append(metadata_warning)
        if verbose:
            print(f"warning: {metadata_warning}")

    rh_rows: list[dict[str, Any]] = []
    es_rows: list[dict[str, Any]] = []
    rc_rows: list[dict[str, Any]] = []
    midhaul_link_rows: list[dict[str, Any]] = []
    direct_link_rows: list[dict[str, Any]] = []
    episode_summary_rows: list[dict[str, Any]] = []
    hidden_state_reset_count = 0
    unexpected_hidden_reset_count = 0
    hidden_norm_tensors: list[torch.Tensor] = []
    hidden_delta_tensors: list[torch.Tensor] = []
    temporal_logit_delta_tensors: list[torch.Tensor] = []
    repair_action_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for episode_id in selected_episode_ids:
            _, _ = env.reset(seed=seed, options={"episode_id": episode_id})
            link_action_mask = topology_action_mask_for_time_index(
                episode,
                episode_id * episode_length,
                episode_length=episode_length,
                profile_masks=topology_profile_action_masks,
            )
            recurrent_hidden_state = None
            hidden_state_reset_count += 1
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
                recurrent_inputs = (
                    graph["rh"].x.unsqueeze(0).unsqueeze(0),
                    [handover.edge_index],
                    [handover.edge_attr],
                )
                # Time the complete policy decision only (both recurrent forward
                # passes, masking/selection, and repair), excluding ``env.step``.
                decision_start_ns = time.perf_counter_ns()
                recurrent_logits, _, next_recurrent_hidden = model.forward_recurrent(
                    *recurrent_inputs, hidden_state=recurrent_hidden_state
                )
                reset_logits, _, _ = model.forward_recurrent(*recurrent_inputs, hidden_state=None)
                logits = recurrent_logits if temporal_mode == "recurrent" else reset_logits
                temporal_logit_delta_tensors.append(
                    torch.linalg.vector_norm(recurrent_logits - reset_logits)
                )
                detached_next_hidden = next_recurrent_hidden.detach()
                hidden_reference = (
                    torch.zeros_like(detached_next_hidden)
                    if recurrent_hidden_state is None
                    else recurrent_hidden_state
                )
                hidden_norm_tensors.append(torch.linalg.vector_norm(detached_next_hidden))
                hidden_delta_tensors.append(
                    torch.linalg.vector_norm(detached_next_hidden - hidden_reference)
                )
                masked_logits = _masked_logits(
                    logits.squeeze(0),
                    link_action_mask,
                    episode.config.num_rhs,
                    spec.num_actions_per_rh,
                )
                if policy == "greedy":
                    action_tensor = masked_logits.argmax(dim=-1)
                else:
                    action_tensor = Categorical(logits=masked_logits).sample()
                action_np = action_tensor.detach().cpu().numpy()
                repair = None
                if policy == "greedy" and infeasibility_repair == "topk":
                    repair_kwargs = dict(
                        masked_logits=masked_logits.detach().cpu().numpy(),
                        connectivity_mask=link_action_mask,
                        evaluator=env.evaluator,
                        time_index=absolute_t,
                        previous_action=env.previous_action,
                        top_k=repair_top_k,
                        max_iterations=repair_max_iterations,
                        beam_width=repair_beam_width,
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
                violation_reason = _main_violation_reason(info)
                step_rows.append({"reward": reward, "decision_ms": decision_ms,
                                  **info, **feasibility, **(repair.diagnostics if repair else {})})

                base_rh_rows = _rh_decision_rows(absolute_t, episode, placement, reward, info, feasibility)
                for row in base_rh_rows:
                    rh_id = int(row["rh_id"])
                    augmented = dict(row)
                    augmented.update(
                        {
                            "episode_id": episode_id,
                            "local_t": local_t,
                            "absolute_t": absolute_t,
                            "action_id": int(action_np[rh_id]),
                            "decision_ms": decision_ms,
                            "feasible": bool(info.get("feasible", False)),
                            "connectivity_feasible": bool(feasibility["connectivity_ok"]),
                            "latency_feasible": bool(feasibility["latency_ok"]),
                            "bandwidth_feasible": bool(feasibility["bandwidth_ok"]),
                            "compute_feasible": bool(feasibility["compute_ok"]),
                            "main_violation_reason": violation_reason,
                            "episode_length": episode_length,
                            "num_episodes": num_episodes,
                            "reset_boundary": local_t == 0,
                        }
                    )
                    if repair is not None:
                        augmented.update(repair.diagnostics)
                    episode_rh_rows.append(augmented)
                    rh_rows.append(augmented)
                if repair is not None:
                    repair_action_rows.extend(
                        {"episode_id": episode_id, "timestep": local_t, "absolute_t": absolute_t, **change}
                        for change in repair.changes
                    )

                es_rows.extend(
                    _augment_resource_rows(
                        _es_load_rows(absolute_t, episode, placement),
                        episode_id=episode_id,
                        local_t=local_t,
                        absolute_t=absolute_t,
                        episode_length=episode_length,
                        num_episodes=num_episodes,
                    )
                )
                rc_rows.extend(
                    _augment_resource_rows(
                        _rc_load_rows(absolute_t, episode, placement),
                        episode_id=episode_id,
                        local_t=local_t,
                        absolute_t=absolute_t,
                        episode_length=episode_length,
                        num_episodes=num_episodes,
                    )
                )
                midhaul_link_rows.extend(
                    _augment_resource_rows(
                        _midhaul_link_load_rows(absolute_t, episode, placement),
                        episode_id=episode_id,
                        local_t=local_t,
                        absolute_t=absolute_t,
                        episode_length=episode_length,
                        num_episodes=num_episodes,
                    )
                )
                direct_link_rows.extend(
                    _augment_resource_rows(
                        _direct_rh_rc_link_load_rows(absolute_t, episode, placement),
                        episode_id=episode_id,
                        local_t=local_t,
                        absolute_t=absolute_t,
                        episode_length=episode_length,
                        num_episodes=num_episodes,
                    )
                )

                done = terminated or truncated
                recurrent_hidden_state = None if done else detached_next_hidden

            episode_summary_rows.append(
                _episode_summary_row(
                    episode=episode,
                    episode_id=episode_id,
                    step_rows=step_rows,
                    rh_rows=episode_rh_rows,
                )
            )

    _sanity_check_selected_trace_rows(
        episode,
        selected_episode_ids=selected_episode_ids,
        rh_rows=rh_rows,
        es_rows=es_rows,
        rc_rows=rc_rows,
        midhaul_link_rows=midhaul_link_rows,
        direct_link_rows=direct_link_rows,
        episode_length=episode_length,
    )
    _sanity_check_trace_steps(rh_rows, selected_episode_ids, episode.config.num_rhs, episode_length)

    trace_steps_path = output_root / "trace_steps.csv"
    rh_decisions_path = output_root / "rh_decisions.csv"
    es_loads_path = output_root / "es_loads.csv"
    rc_loads_path = output_root / "rc_loads.csv"
    midhaul_link_loads_path = output_root / "midhaul_link_loads.csv"
    direct_rh_rc_link_loads_path = output_root / "direct_rh_rc_link_loads.csv"
    episode_summary_path = output_root / "episode_summary.csv"
    eval_summary_path = output_root / "eval_summary.json"

    _write_csv(trace_steps_path, rh_rows, TRACE_STEP_COLUMNS)
    _write_csv(rh_decisions_path, rh_rows, TRACE_STEP_COLUMNS)
    for rh_id in range(episode.config.num_rhs):
        _write_csv(
            output_root / "rh" / f"RH_{rh_id}.csv",
            [row for row in rh_rows if int(row["rh_id"]) == rh_id],
            TRACE_STEP_COLUMNS,
        )
    _write_csv(es_loads_path, es_rows, ES_LOAD_COLUMNS + RESOURCE_EXTRA_COLUMNS)
    _write_csv(rc_loads_path, rc_rows, RC_LOAD_COLUMNS + RESOURCE_EXTRA_COLUMNS)
    _write_csv(midhaul_link_loads_path, midhaul_link_rows, MIDHAUL_LINK_LOAD_COLUMNS + RESOURCE_EXTRA_COLUMNS)
    _write_csv(
        direct_rh_rc_link_loads_path,
        direct_link_rows,
        DIRECT_RH_RC_LINK_LOAD_COLUMNS + RESOURCE_EXTRA_COLUMNS,
    )
    _write_csv(episode_summary_path, episode_summary_rows, _episode_summary_columns(episode))
    _write_csv(output_root / "repair_actions.csv", repair_action_rows, REPAIR_ACTION_COLUMNS)

    eval_summary = _eval_summary(
        dataset_dir=dataset_path,
        checkpoint_path=checkpoint,
        params_path=Path(params_path),
        env=env,
        selected_episode_ids=selected_episode_ids,
        episode_summary_rows=episode_summary_rows,
        checkpoint_metadata=checkpoint_metadata,
        warning_messages=warning_messages,
        hidden_state_reset_count=hidden_state_reset_count,
        unexpected_hidden_reset_count=unexpected_hidden_reset_count,
        temporal_mode=temporal_mode,
        hidden_norms=_tensor_scalars(hidden_norm_tensors),
        hidden_deltas=_tensor_scalars(hidden_delta_tensors),
        temporal_logit_deltas=_tensor_scalars(temporal_logit_delta_tensors),
    )
    eval_summary["observation_feature_mode"] = observation_feature_mode
    eval_summary["infeasibility_repair"] = infeasibility_repair
    eval_summary["repair_top_k"] = repair_top_k
    eval_summary["repair_max_iterations"] = repair_max_iterations
    eval_summary["repair_beam_width"] = repair_beam_width
    eval_summary["repair_mode"] = repair_mode
    eval_summary["repair_decision_budget_seconds"] = repair_decision_budget_seconds
    eval_summary["observation_schema_version"] = observation_schema_for_mode(
        episode, observation_feature_mode
    )
    eval_summary_path.write_text(json.dumps(eval_summary, indent=2, sort_keys=True), encoding="utf-8")

    _sanity_check_exported_files_prebuilt(
        output_root,
        episode=episode,
        selected_episode_count=len(selected_episode_ids),
        episode_length=episode_length,
    )

    return PrebuiltTraceResult(
        trace_steps=trace_steps_path,
        rh_decisions=rh_decisions_path,
        es_loads=es_loads_path,
        rc_loads=rc_loads_path,
        midhaul_link_loads=midhaul_link_loads_path,
        direct_rh_rc_link_loads=direct_rh_rc_link_loads_path,
        episode_summary=episode_summary_path,
        eval_summary=eval_summary_path,
    )


def _resolve_episode_ids(
    env: PrebuiltEpisodeWindowEnv,
    *,
    episodes: str,
    max_episodes: int | None,
) -> list[int]:
    if episodes == "all":
        selected = list(range(env.num_episodes))
    else:
        selected = []
        for part in episodes.split(","):
            value = part.strip()
            if not value:
                continue
            episode_id = int(value)
            if episode_id < 0 or episode_id >= env.num_episodes:
                raise ValueError(f"episode_id must be in [0, {env.num_episodes - 1}]")
            selected.append(episode_id)
        if not selected:
            raise ValueError("--episodes must contain at least one episode id")
    selected = sorted(dict.fromkeys(selected))
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("--max_episodes must be positive when provided")
        selected = selected[:max_episodes]
    return selected


def validate_stgnn_recurrent_metadata(
    metadata: dict[str, Any],
    *,
    episode_length: int,
    hidden_dim: int,
    observation_schema_version: str,
) -> None:
    """Reject checkpoints that cannot reproduce full-episode recurrent semantics."""

    expected = {
        "architecture_version": STGNN_ARCHITECTURE_VERSION,
        "episode_length": episode_length,
        "recurrent_sequence_length": episode_length,
        "hidden_dim": hidden_dim,
        "num_gru_layers": 1,
        "temporal_fusion_mode": STGNN_TEMPORAL_FUSION_MODE,
        "hidden_reset_mode": STGNN_HIDDEN_RESET_MODE,
        "ppo_replay_mode": STGNN_PPO_REPLAY_MODE,
        "observation_schema_version": observation_schema_version,
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"incompatible ST-GNN checkpoint metadata: {key}={metadata.get(key)!r}, "
                f"required={expected_value!r}"
            )
    required_present = (
        "dataset_dynamic_hash",
        "dataset_topology_hash",
        "dataset_config_hash",
        "constraint_formulation_version",
        "reward_transform",
        "feature_mode",
        "reward_mode",
    )
    missing = [key for key in required_present if key not in metadata]
    if missing:
        raise ValueError(f"incompatible ST-GNN checkpoint metadata: missing {missing}")


def _tensor_scalars(values: list[torch.Tensor]) -> list[float]:
    """Transfer accumulated scalar diagnostics once, outside the timestep loop."""

    if not values:
        return []
    return torch.stack(values).detach().cpu().tolist()


def _metadata_warning(
    metadata: dict[str, Any], episode_length: int, num_episodes: int, *,
    dataset_schema: str = "legacy_same_step_v1", allow_schema_mismatch: bool = False,
) -> str | None:
    checkpoint_metadata = metadata.get("metadata", metadata)
    saved_episode_length = checkpoint_metadata.get("episode_length")
    saved_num_episodes = checkpoint_metadata.get("num_episodes")
    mismatches: list[str] = []
    saved_schema = checkpoint_metadata.get("mobility_observation_schema_version")
    if saved_schema is not None and str(saved_schema) != dataset_schema:
        schema_mismatch = (
            "checkpoint/dataset mobility observation schema mismatch: "
            f"checkpoint={saved_schema!r} dataset={dataset_schema!r}"
        )
        if not allow_schema_mismatch:
            raise ValueError(schema_mismatch)
        mismatches.append(schema_mismatch)
    if saved_episode_length is not None and int(saved_episode_length) != int(episode_length):
        mismatches.append(
            f"checkpoint episode_length={saved_episode_length} differs from CLI episode_length={episode_length}"
        )
    if saved_num_episodes is not None and int(saved_num_episodes) != int(num_episodes):
        mismatches.append(
            f"checkpoint num_episodes={saved_num_episodes} differs from CLI num_episodes={num_episodes}"
        )
    if not mismatches:
        return None
    return "; ".join(mismatches)


def _main_violation_reason(info: dict[str, Any]) -> str:
    violations = info.get("violations", {})
    true_violation_fields = {
        "es_compute_overload", "rc_compute_overload",
        "fronthaul_bw_shortfall_mbps", "midhaul_bw_shortfall_mbps", "direct_rh_rc_bw_shortfall_mbps",
        "end_to_end_latency_excess_ms",
    }
    positive = [
        (name, float(value)) for name, value in violations.items()
        if name in true_violation_fields and float(value) > 0.0
    ]
    if not positive:
        return ""
    positive.sort(key=lambda item: (-item[1], item[0]))
    return positive[0][0]


def _augment_resource_rows(
    rows: list[dict[str, Any]],
    *,
    episode_id: int,
    local_t: int,
    absolute_t: int,
    episode_length: int,
    num_episodes: int,
) -> list[dict[str, Any]]:
    augmented_rows: list[dict[str, Any]] = []
    for row in rows:
        augmented = dict(row)
        augmented.update(
            {
                "episode_id": episode_id,
                "local_t": local_t,
                "absolute_t": absolute_t,
                "episode_length": episode_length,
                "num_episodes": num_episodes,
            }
        )
        augmented_rows.append(augmented)
    return augmented_rows


def _episode_summary_row(
    *,
    episode: MockEpisode,
    episode_id: int,
    step_rows: list[dict[str, Any]],
    rh_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rewards = [float(row["reward"]) for row in step_rows]
    total_costs = [float(row["total_cost"]) for row in step_rows]
    summary: dict[str, Any] = {
        "episode_id": episode_id,
        "total_reward": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "total_cost": float(np.sum(total_costs)),
        "mean_total_cost": float(np.mean(total_costs)) if total_costs else float("nan"),
        "overall_feasible_rate": _bool_mean(step_rows, "overall_ok"),
        "connectivity_feasible_rate": _bool_mean(step_rows, "connectivity_ok"),
        "latency_feasible_rate": _bool_mean(step_rows, "latency_ok"),
        "bandwidth_feasible_rate": _bool_mean(step_rows, "bandwidth_ok"),
        "compute_feasible_rate": _bool_mean(step_rows, "compute_ok"),
    }
    split_ids = [int(row["split_id"]) for row in rh_rows]
    es_ids = [int(row["es_id"]) for row in rh_rows if not bool(row["uses_direct_rh_rc"])]
    rc_ids = [int(row["rc_id"]) for row in rh_rows]
    split_total = len(split_ids)
    es_total = len(es_ids)
    rc_total = len(rc_ids)
    for split_id, split_name in enumerate(episode.split_demand.split_names):
        count = int(sum(1 for value in split_ids if value == split_id))
        summary[f"split_count_{split_name}"] = count
        summary[f"split_frac_{split_name}"] = float(count / split_total) if split_total > 0 else float("nan")
    for es_id in range(episode.config.num_es):
        count = int(sum(1 for value in es_ids if value == es_id))
        summary[f"es_usage_count_{es_id}"] = count
        summary[f"es_usage_frac_{es_id}"] = float(count / es_total) if es_total > 0 else 0.0
    for rc_id in range(episode.config.num_rc):
        count = int(sum(1 for value in rc_ids if value == rc_id))
        summary[f"rc_usage_count_{rc_id}"] = count
        summary[f"rc_usage_frac_{rc_id}"] = float(count / rc_total) if rc_total > 0 else 0.0
    return summary


def _bool_mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows])) if rows else float("nan")


def _episode_summary_columns(episode: MockEpisode) -> list[str]:
    columns = [
        "episode_id",
        "total_reward",
        "mean_reward",
        "total_cost",
        "mean_total_cost",
        "overall_feasible_rate",
        "connectivity_feasible_rate",
        "latency_feasible_rate",
        "bandwidth_feasible_rate",
        "compute_feasible_rate",
    ]
    for split_name in episode.split_demand.split_names:
        columns.append(f"split_count_{split_name}")
        columns.append(f"split_frac_{split_name}")
    for es_id in range(episode.config.num_es):
        columns.append(f"es_usage_count_{es_id}")
        columns.append(f"es_usage_frac_{es_id}")
    for rc_id in range(episode.config.num_rc):
        columns.append(f"rc_usage_count_{rc_id}")
        columns.append(f"rc_usage_frac_{rc_id}")
    return columns


def _eval_summary(
    *,
    dataset_dir: Path,
    checkpoint_path: Path,
    params_path: Path,
    env: PrebuiltEpisodeWindowEnv,
    selected_episode_ids: list[int],
    episode_summary_rows: list[dict[str, Any]],
    checkpoint_metadata: dict[str, Any],
    warning_messages: list[str],
    hidden_state_reset_count: int,
    unexpected_hidden_reset_count: int = 0,
    temporal_mode: str = "snapshot",
    hidden_norms: list[float] | None = None,
    hidden_deltas: list[float] | None = None,
    temporal_logit_deltas: list[float] | None = None,
) -> dict[str, Any]:
    def _mean(key: str) -> float:
        return float(np.mean([float(row[key]) for row in episode_summary_rows])) if episode_summary_rows else float("nan")

    checkpoint_identity = checkpoint_metadata.get("metadata", checkpoint_metadata)
    dataset_metadata_path = dataset_dir / "dataset_metadata.json"
    dataset_identity = (
        json.loads(dataset_metadata_path.read_text(encoding="utf-8"))
        if dataset_metadata_path.exists()
        else {}
    )
    training_profile_set_hash = checkpoint_identity.get("topology_profile_set_hash", "unknown")
    training_dataset_topology_hash = checkpoint_identity.get("dataset_topology_hash", "unknown")
    evaluation_profile_set_hash = dataset_identity.get("topology_profile_set_hash", "unknown")
    evaluation_dataset_topology_hash = dataset_identity.get("topology_hash", "unknown")
    return {
        "dataset_dir": str(dataset_dir),
        "checkpoint": str(checkpoint_path),
        "params": str(params_path),
        "episode_length": env.episode_length,
        "num_episodes": env.num_episodes,
        "flat_horizon": env.flat_horizon,
        "evaluated_episodes": selected_episode_ids,
        "mean_reward": _mean("mean_reward"),
        "mean_total_cost": _mean("mean_total_cost"),
        "overall_feasible_rate": _mean("overall_feasible_rate"),
        "connectivity_feasible_rate": _mean("connectivity_feasible_rate"),
        "latency_feasible_rate": _mean("latency_feasible_rate"),
        "bandwidth_feasible_rate": _mean("bandwidth_feasible_rate"),
        "compute_feasible_rate": _mean("compute_feasible_rate"),
        "checkpoint_metadata": checkpoint_metadata,
        "checkpoint_training_topology_profile_set_hash": training_profile_set_hash,
        "checkpoint_training_dataset_topology_hash": training_dataset_topology_hash,
        "evaluation_dataset_topology_profile_set_hash": evaluation_profile_set_hash,
        "evaluation_dataset_topology_hash": evaluation_dataset_topology_hash,
        "cross_topology_evaluation": (
            training_profile_set_hash != evaluation_profile_set_hash
            if training_profile_set_hash != "unknown"
            else training_dataset_topology_hash != evaluation_dataset_topology_hash
        ),
        "warning": warning_messages[0] if warning_messages else "",
        "warnings": warning_messages,
        "legacy_compatible_output_files": LEGACY_COMPATIBLE_FILES,
        "prebuilt_specific_output_files": PREBUILT_SPECIFIC_FILES,
        "hidden_state_reset_count": hidden_state_reset_count,
        "episode_hidden_reset_count": hidden_state_reset_count,
        "unexpected_hidden_reset_count": unexpected_hidden_reset_count,
        "temporal_mode": temporal_mode,
        "hidden_state_norm_mean": float(np.mean(hidden_norms)) if hidden_norms else 0.0,
        "hidden_state_norm_max": float(np.max(hidden_norms)) if hidden_norms else 0.0,
        "hidden_state_delta_mean": float(np.mean(hidden_deltas)) if hidden_deltas else 0.0,
        "hidden_state_delta_max": float(np.max(hidden_deltas)) if hidden_deltas else 0.0,
        "temporal_logit_delta_mean": (
            float(np.mean(temporal_logit_deltas)) if temporal_logit_deltas else 0.0
        ),
        "temporal_logit_delta_max": (
            float(np.max(temporal_logit_deltas)) if temporal_logit_deltas else 0.0
        ),
        "episode_end_indices": list(env.episode_end_indices()),
    }


def _sanity_check_selected_trace_rows(
    episode: MockEpisode,
    *,
    selected_episode_ids: list[int],
    rh_rows: list[dict[str, Any]],
    es_rows: list[dict[str, Any]],
    rc_rows: list[dict[str, Any]],
    midhaul_link_rows: list[dict[str, Any]],
    direct_link_rows: list[dict[str, Any]],
    episode_length: int,
) -> None:
    expected_ts = [
        episode_id * episode_length + local_t
        for episode_id in selected_episode_ids
        for local_t in range(episode_length)
    ]
    if len(rh_rows) != len(expected_ts) * episode.config.num_rhs:
        raise ValueError("rh_decisions rows must equal selected_steps * num_rhs")
    if len(es_rows) != len(expected_ts) * episode.config.num_es:
        raise ValueError("es_loads rows must equal selected_steps * num_es")
    if len(rc_rows) != len(expected_ts) * episode.config.num_rc:
        raise ValueError("rc_loads rows must equal selected_steps * num_rc")
    if len(midhaul_link_rows) != len(expected_ts) * episode.config.num_es * episode.config.num_rc:
        raise ValueError("midhaul_link_loads rows must equal selected_steps * num_es * num_rc")

    for absolute_t in expected_ts:
        rh_at_t = [row for row in rh_rows if int(row["absolute_t"]) == absolute_t]
        if len(rh_at_t) != episode.config.num_rhs:
            raise ValueError(f"absolute_t={absolute_t} must have one RH decision per RH")
        rc_count = sum(int(row["num_assigned_rhs"]) for row in rc_rows if int(row["absolute_t"]) == absolute_t)
        if rc_count != episode.config.num_rhs:
            raise ValueError(f"RC assigned RH count at absolute_t={absolute_t} must equal num_rhs")
        midhaul_pair_count = sum(
            int(row["num_assigned_rhs"]) for row in midhaul_link_rows if int(row["absolute_t"]) == absolute_t
        )
        direct_count = sum(1 for row in direct_link_rows if int(row["absolute_t"]) == absolute_t)
        if midhaul_pair_count + direct_count != episode.config.num_rhs:
            raise ValueError(
                f"gateway plus direct assigned RH count at absolute_t={absolute_t} must equal num_rhs"
            )


def _sanity_check_trace_steps(
    rh_rows: list[dict[str, Any]],
    selected_episode_ids: list[int],
    num_rhs: int,
    episode_length: int,
) -> None:
    expected_rows = len(selected_episode_ids) * episode_length * num_rhs
    if len(rh_rows) != expected_rows:
        raise ValueError("trace_steps rows must equal evaluated_episodes * episode_length * num_rhs")
    for row in rh_rows:
        absolute_t = int(row["absolute_t"])
        if int(row["t"]) != absolute_t:
            raise ValueError("t must equal absolute_t for legacy compatibility")
        if int(row["episode_id"]) != absolute_t // episode_length:
            raise ValueError("episode_id must equal absolute_t // episode_length")
        if int(row["local_t"]) != absolute_t % episode_length:
            raise ValueError("local_t must equal absolute_t % episode_length")


def _sanity_check_exported_files_prebuilt(
    output_root: Path,
    *,
    episode: MockEpisode,
    selected_episode_count: int,
    episode_length: int,
) -> None:
    import polars as pl

    expected_rows = selected_episode_count * episode_length * episode.config.num_rhs
    rh_decisions = pl.read_csv(output_root / "rh_decisions.csv")
    trace_steps = pl.read_csv(output_root / "trace_steps.csv")
    es_loads = pl.read_csv(output_root / "es_loads.csv")
    rc_loads = pl.read_csv(output_root / "rc_loads.csv")
    midhaul_link_loads = pl.read_csv(output_root / "midhaul_link_loads.csv")
    episode_summary = pl.read_csv(output_root / "episode_summary.csv")
    if rh_decisions.height != expected_rows:
        raise ValueError("rh_decisions.csv row count mismatch")
    if trace_steps.height != expected_rows:
        raise ValueError("trace_steps.csv row count mismatch")
    if es_loads.height != selected_episode_count * episode_length * episode.config.num_es:
        raise ValueError("es_loads.csv row count mismatch")
    if rc_loads.height != selected_episode_count * episode_length * episode.config.num_rc:
        raise ValueError("rc_loads.csv row count mismatch")
    if midhaul_link_loads.height != selected_episode_count * episode_length * episode.config.num_es * episode.config.num_rc:
        raise ValueError("midhaul_link_loads.csv row count mismatch")
    if episode_summary.height != selected_episode_count:
        raise ValueError("episode_summary.csv row count mismatch")
    for rh_id in range(episode.config.num_rhs):
        rh_file = pl.read_csv(output_root / "rh" / f"RH_{rh_id}.csv")
        if rh_file.height != selected_episode_count * episode_length:
            raise ValueError(f"rh/RH_{rh_id}.csv row count mismatch")


TRACE_STEP_COLUMNS = [
    "t",
    "episode_id",
    "local_t",
    "absolute_t",
    "episode_length",
    "num_episodes",
    "reset_boundary",
    "rh_id",
    "split_id",
    "split_name",
    "path_type",
    "uses_direct_rh_rc",
    "es_id",
    "effective_es_id",
    "effective_rc_id",
    "rc_id",
    "action_id",
    "traffic_load_mbps",
    "n_ue",
    "s_pdcp_mb",
    "s_rlc_mb",
    "rho",
    "pdcp_anchor_type",
    "pdcp_anchor_id",
    "rlc_anchor_type",
    "rlc_anchor_id",
    "reward",
    "total_cost",
    "placement_cost",
    "handover_state_transfer_cost",
    "handover_sla_cost",
    "link_violation_cost",
    "compute_overload_cost",
    "feasible",
    "connectivity_feasible",
    "latency_feasible",
    "bandwidth_feasible",
    "compute_feasible",
    "main_violation_reason",
    "connectivity_ok",
    "bandwidth_ok",
    "latency_ok",
    "compute_ok",
    "overall_ok",
    "decision_ms",
    "repair_triggered", "repair_succeeded", "repair_iterations", "num_actions_changed",
    "original_service_feasible", "final_service_feasible", "original_V_CMP", "original_V_BW",
    "original_V_LAT", "final_V_CMP", "final_V_BW", "final_V_LAT", "original_J", "final_J", "delta_J",
    "deadline_enabled", "deadline_hit", "decision_budget_ms", "repair_elapsed_ms",
    "deadline_overshoot_ms", "search_depth_reached", "max_beam_size_reached", "candidate_evaluations",
    # Bounded Top-k diagnostics.  These are also emitted for ordinary (non-deadline)
    # repair and must remain representable in the legacy-compatible action trace.
    "runtime_aborted", "original_implicated_rhs_count", "searched_implicated_rhs_count",
    "max_beam_states", "max_children_before_prune", "search_stop_reason", "previous_action_fallback_used",
]

REPAIR_ACTION_COLUMNS = [
    "episode_id", "timestep", "absolute_t", "rh_id", "original_rank", "original_split", "original_es",
    "original_rc", "replacement_rank", "replacement_split", "replacement_es", "replacement_rc",
    "original_logit", "replacement_logit", "logit_loss",
    "runtime_aborted", "original_implicated_rhs_count", "searched_implicated_rhs_count",
    "max_beam_states", "max_children_before_prune", "search_stop_reason",
]

RESOURCE_EXTRA_COLUMNS = [
    "episode_id",
    "local_t",
    "absolute_t",
    "episode_length",
    "num_episodes",
]


def main() -> None:
    args = parse_args()
    result = run_prebuilt_recurrent_trace(
        dataset_dir=args.dataset_dir,
        params_path=args.params,
        checkpoint_path=args.checkpoint,
        episode_length=args.episode_length,
        num_episodes=args.num_episodes,
        device=args.device,
        config_path=args.config,
        output_dir=args.output_dir,
        episodes=args.episodes,
        policy=args.policy,
        max_episodes=args.max_episodes,
        seed=args.seed,
        verbose=args.verbose,
        allow_schema_mismatch=args.allow_schema_mismatch,
        temporal_mode=args.temporal_mode,
        observation_feature_mode=args.observation_feature_mode,
        trust_checkpoint=args.trust_checkpoint,
        infeasibility_repair=args.infeasibility_repair,
        repair_top_k=args.repair_top_k,
        repair_max_iterations=args.repair_max_iterations,
        repair_beam_width=args.repair_beam_width,
        repair_mode=args.repair_mode,
        repair_decision_budget_seconds=args.repair_decision_budget_seconds,
    )
    print(f"wrote trace steps to {result.trace_steps}")
    print(f"wrote legacy-compatible RH decisions to {result.rh_decisions}")
    print(f"wrote ES loads to {result.es_loads}")
    print(f"wrote RC loads to {result.rc_loads}")
    print(f"wrote midhaul link loads to {result.midhaul_link_loads}")
    print(f"wrote direct RH-RC link loads to {result.direct_rh_rc_link_loads}")
    print(f"wrote episode summary to {result.episode_summary}")
    print(f"wrote evaluation summary to {result.eval_summary}")


if __name__ == "__main__":
    main()

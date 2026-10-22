"""Run trained-model inference over one mock episode and export trace CSVs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any
from collections.abc import Callable

import numpy as np
import torch
from torch.distributions import Categorical


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env.oran_ho_env import OranHandoverEnv
from src.model.inference.post_process import (
    ActionSpaceSpec,
    action_mask_kwargs_from_episode,
    build_link_feasibility_action_mask,
    decode_flat_actions,
)
from src.model.model import STGNNActorCritic
from src.model.model import STGNN_ARCHITECTURE_VERSION
from src.model.causal_observation import (
    LEGACY_OBSERVATION_SCHEMA,
    RESOURCE_OBSERVATION_FEATURE_MODES,
)
from src.model.split_semantics import (
    pdcp_anchor_type,
    rlc_anchor_type,
    split_path_type,
    uses_direct_rh_rc,
    uses_es_compute,
    uses_midhaul,
)
from src.model.training.ppo import resolve_device
from src.pipeline.data_pipeline import HandoverGraphPipeline, MockCsvDatasetLoader
from src.pipeline.schema import MockEpisode, PlacementAction
from torch_geometric.data import HeteroData
from src.utils.config import DEFAULT_CONFIG_PATH


@dataclass(frozen=True)
class TraceResult:
    """Paths produced by one inference export run."""

    rh_decisions: Path
    es_loads: Path
    rc_loads: Path
    midhaul_link_loads: Path
    direct_rh_rc_link_loads: Path
    episode_summary: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a full-episode inference trace")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method", choices=("ppo", "recurrent", "windowed"), required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--dataset_dir", default="mock_data", help="Directory containing generated mock CSV files for inference.",)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda device such as cuda:0")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--temporal_window_size", type=int, default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--episode_index", type=int, default=0)
    return parser.parse_args()


def save_model_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    method: str | None = None,
    metadata: dict[str, Any] | None = None,
    training_state: dict[str, Any] | None = None,
) -> None:
    """Save the minimal PyTorch state_dict checkpoint used by inference."""

    checkpoint: dict[str, Any] = {"model_state_dict": model.state_dict()}
    if method is not None:
        checkpoint["method"] = method
    if metadata is not None:
        checkpoint["metadata"] = metadata
    if training_state is not None:
        checkpoint["training_state"] = training_state
    torch.save(checkpoint, Path(path))


def load_model_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    device: torch.device,
    *,
    expected_normalization_modes: dict[str, str] | None = None,
    expected_observation_feature_mode: str | None = None,
    expected_observation_schema_version: str | None = None,
    expected_observation_feature_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Load either a raw state_dict or a dict containing ``model_state_dict``."""

    # These are repository-produced full checkpoints containing trusted NumPy
    # metadata/training state in addition to tensors. PyTorch 2.6 defaults to
    # weights_only=True, which cannot deserialize that established schema.
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = {
            key: value
            for key, value in checkpoint.items()
            if key not in {"model_state_dict", "training_state"}
        }
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        metadata = {
            key: value
            for key, value in checkpoint.items()
            if key not in {"state_dict", "training_state"}
        }
    else:
        state_dict = checkpoint
        metadata = {}
    if isinstance(model, STGNNActorCritic):
        checkpoint_metadata = metadata.get("metadata", metadata)
        architecture_version = checkpoint_metadata.get("architecture_version")
        if architecture_version != STGNN_ARCHITECTURE_VERSION:
            raise ValueError(
                "incompatible ST-GNN checkpoint architecture: "
                f"checkpoint={architecture_version!r}, required={STGNN_ARCHITECTURE_VERSION!r}; "
                "retrain with the full-episode recurrent patch"
            )
    if isinstance(metadata.get("metadata"), dict):
        # Normalize compatibility metadata only for this in-memory load. The
        # serialized checkpoint remains immutable.
        metadata["metadata"] = dict(metadata["metadata"])
    checkpoint_metadata = metadata.get("metadata", metadata)
    actual_observation_mode = str(
        checkpoint_metadata.get("observation_feature_mode", "legacy_rh_only")
    )
    if (expected_observation_feature_mode is not None
            and actual_observation_mode != expected_observation_feature_mode):
        raise ValueError(
            "checkpoint observation feature mode mismatch: "
            f"checkpoint={actual_observation_mode!r}, runtime={expected_observation_feature_mode!r}"
        )
    actual_observation_schema = checkpoint_metadata.get("observation_schema_version")
    if (
        expected_observation_schema_version is not None
        and actual_observation_schema != expected_observation_schema_version
    ):
        stored_feature_names = tuple(checkpoint_metadata.get("observation_schema", ()))
        compatible_historical_resource_checkpoint = (
            expected_observation_feature_mode in RESOURCE_OBSERVATION_FEATURE_MODES
            and actual_observation_schema == LEGACY_OBSERVATION_SCHEMA
            and expected_observation_feature_names is not None
            and stored_feature_names == expected_observation_feature_names
        )
        if not compatible_historical_resource_checkpoint:
            raise ValueError(
                "checkpoint observation schema mismatch: "
                f"checkpoint={actual_observation_schema!r}, "
                f"runtime={expected_observation_schema_version!r}"
            )
        checkpoint_metadata["observation_schema_version"] = (
            expected_observation_schema_version
        )
        checkpoint_metadata["observation_schema_compatibility"] = (
            "historical_resource_checkpoint_metadata_normalized"
        )
    checkpoint_dim = int(checkpoint_metadata.get("rh_feature_dim", 5))
    model_dim = int(getattr(model, "rh_feature_dim", 5))
    if checkpoint_dim != model_dim:
        raise ValueError(
            "checkpoint RH observation dimension mismatch: "
            f"checkpoint={checkpoint_dim}, runtime={model_dim}"
        )
    model.load_state_dict(state_dict)
    if expected_normalization_modes is not None:
        validate_normalization_metadata(metadata, expected_normalization_modes)
    return metadata


def load_runtime_episode(
    dataset_dir: str | Path,
    config_path: str | Path,
) -> tuple[MockEpisode, dict[str, Any]]:
    """Load one episode and the exact resolved configuration that owns it."""

    loader = MockCsvDatasetLoader(dataset_dir, config_path)
    return loader.load_episode(), loader.project_config


def normalization_modes_from_project_config(project_config: dict[str, Any]) -> dict[str, str]:
    """Resolve normalization modes, including legacy defaults."""

    return {
        "entropy_scale_mode": str(
            project_config.get("ppo", {}).get("entropy_scale_mode", "legacy_sum")
        ),
        "reward_graph_scale_mode": str(
            project_config.get("reward", {}).get(
                "reward_graph_scale_mode", "legacy_total_cost"
            )
        ),
    }


def validate_normalization_metadata(
    checkpoint_metadata: dict[str, Any],
    expected: dict[str, str] | dict[str, Any],
) -> None:
    """Reject explicit mode mismatch while accepting old checkpoints as legacy."""

    metadata = checkpoint_metadata.get("metadata", checkpoint_metadata)
    expected_modes = (
        normalization_modes_from_project_config(expected)
        if "ppo" in expected or "reward" in expected
        else expected
    )
    actual_modes = {
        "entropy_scale_mode": str(metadata.get("entropy_scale_mode", "legacy_sum")),
        "reward_graph_scale_mode": str(
            metadata.get("reward_graph_scale_mode", "legacy_total_cost")
        ),
    }
    if actual_modes != expected_modes:
        raise ValueError(
            "checkpoint normalization mode mismatch: "
            f"checkpoint={actual_modes}, runtime={expected_modes}"
        )


def validate_constraint_formulation_metadata(
    checkpoint_metadata: dict[str, Any], project_config: dict[str, Any]
) -> None:
    metadata = checkpoint_metadata.get("metadata", checkpoint_metadata)
    expected = project_config.get("constraint_formulation_version")
    actual = metadata.get("constraint_formulation_version")
    if actual != expected:
        raise ValueError(
            "checkpoint constraint formulation mismatch: "
            f"checkpoint={actual!r}, runtime={expected!r}; retrain under the current formulation"
        )


def run_episode_trace(
    *,
    checkpoint_path: str | Path,
    method: str,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    dataset_dir: str | Path = "mock_data",
    output_dir: str | Path,
    seed: int = 13,
    device: str | torch.device = "auto",
    deterministic: bool = True,
    temporal_window_size: int | None = None,
    episode_index: int = 0,
) -> TraceResult:
    """Load a checkpoint, run one full episode, and export trace CSVs."""

    method = method.lower()
    if method not in {"ppo", "recurrent", "windowed"}:
        raise ValueError("method must be one of: ppo, recurrent, windowed")

    torch_device = resolve_device(device)
    np.random.seed(seed)
    torch.manual_seed(seed)

    episode, project_config = load_runtime_episode(dataset_dir, config_path)
    env = OranHandoverEnv(episode, project_config=project_config)
    spec = ActionSpaceSpec.from_topology(episode.config)
    model = STGNNActorCritic(
        rh_feature_dim=5,
        hidden_dim=int(project_config["model"]["hidden_dim"]),
        num_actions_per_rh=spec.num_actions_per_rh,
    ).to(torch_device)
    checkpoint_metadata = load_model_checkpoint(checkpoint_path, model, torch_device)
    validate_constraint_formulation_metadata(checkpoint_metadata, project_config)
    validate_normalization_metadata(checkpoint_metadata, project_config)
    model.eval()

    graph_pipeline = HandoverGraphPipeline(episode)
    link_action_mask = build_link_feasibility_action_mask(
        episode.config,
        episode.fronthaul_mask,
        episode.midhaul_mask,
        episode.direct_rh_rc_mask,
        episode.split_demand.split_names,
        **action_mask_kwargs_from_episode(episode),
    )
    window_size = (
        int(temporal_window_size)
        if temporal_window_size is not None
        else int(project_config["ppo"].get("temporal_window_size", 4))
    )
    if window_size <= 0:
        raise ValueError("temporal_window_size must be positive")

    rh_rows: list[dict[str, Any]] = []
    es_rows: list[dict[str, Any]] = []
    rc_rows: list[dict[str, Any]] = []
    midhaul_link_rows: list[dict[str, Any]] = []
    direct_link_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "rh").mkdir(parents=True, exist_ok=True)

    hidden_state = None
    with torch.no_grad():
        env.reset(seed=seed)
        done = False
        while not done:
            t = env.time_index
            if method == "recurrent":
                logits, _, next_hidden_state = _model_at_recurrent(
                    model,
                    graph_pipeline,
                    t,
                    torch_device,
                    hidden_state,
                )
            elif method == "windowed":
                logits, _, next_hidden_state = (
                    *_model_at_windowed(model, graph_pipeline, t, window_size, torch_device),
                    None,
                )
            else:
                logits, _, next_hidden_state = (
                    *_model_at_single(model, graph_pipeline, t, torch_device),
                    None,
                )

            masked_logits = _masked_logits(
                logits.squeeze(0),
                link_action_mask,
                episode.config.num_rhs,
                spec.num_actions_per_rh,
            )
            if deterministic:
                action_tensor = masked_logits.argmax(dim=-1)
            else:
                action_tensor = Categorical(logits=masked_logits).sample()
            action_np = action_tensor.detach().cpu().numpy()
            _assert_actions_match_mask(action_np, link_action_mask)
            placement = decode_flat_actions(action_np, episode.config)
            _, reward, terminated, truncated, info = env.step(action_np)

            feasibility = _step_feasibility(info)
            rh_rows.extend(_rh_decision_rows(t, episode, placement, reward, info, feasibility))
            es_rows.extend(_es_load_rows(t, episode, placement))
            rc_rows.extend(_rc_load_rows(t, episode, placement))
            midhaul_link_rows.extend(_midhaul_link_load_rows(t, episode, placement))
            direct_link_rows.extend(_direct_rh_rc_link_load_rows(t, episode, placement))
            step_rows.append({"reward": reward, **info, **feasibility})

            done = terminated or truncated
            hidden_state = None if done else (
                next_hidden_state.detach() if next_hidden_state is not None else None
            )

    _sanity_check_trace_rows(episode, rh_rows, es_rows, rc_rows, midhaul_link_rows, direct_link_rows)

    rh_decisions_path = output_root / "rh_decisions.csv"
    es_loads_path = output_root / "es_loads.csv"
    rc_loads_path = output_root / "rc_loads.csv"
    midhaul_link_loads_path = output_root / "midhaul_link_loads.csv"
    direct_rh_rc_link_loads_path = output_root / "direct_rh_rc_link_loads.csv"
    episode_summary_path = output_root / "episode_summary.csv"

    _write_csv(rh_decisions_path, rh_rows, RH_DECISION_COLUMNS)
    for rh_id in range(episode.config.num_rhs):
        _write_csv(
            output_root / "rh" / f"RH_{rh_id}.csv",
            [row for row in rh_rows if int(row["rh_id"]) == rh_id],
            RH_DECISION_COLUMNS,
        )
    _write_csv(es_loads_path, es_rows, ES_LOAD_COLUMNS)
    _write_csv(rc_loads_path, rc_rows, RC_LOAD_COLUMNS)
    _write_csv(midhaul_link_loads_path, midhaul_link_rows, MIDHAUL_LINK_LOAD_COLUMNS)
    _write_csv(direct_rh_rc_link_loads_path, direct_link_rows, DIRECT_RH_RC_LINK_LOAD_COLUMNS)
    _write_csv(
        episode_summary_path,
        [
            _episode_summary_row(
                episode_index=episode_index,
                method=method,
                checkpoint_path=str(checkpoint_path),
                episode=episode,
                step_rows=step_rows,
            )
        ],
        EPISODE_SUMMARY_COLUMNS,
    )

    _sanity_check_exported_files(output_root, episode)
    return TraceResult(
        rh_decisions_path,
        es_loads_path,
        rc_loads_path,
        midhaul_link_loads_path,
        direct_rh_rc_link_loads_path,
        episode_summary_path,
    )


def _model_at_single(
    model: STGNNActorCritic,
    graph_pipeline: HandoverGraphPipeline,
    time_index: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    graph = graph_pipeline.build_snapshot(time_index).to(device)
    handover = graph[("rh", "handover", "rh")]
    return model(
        graph["rh"].x.unsqueeze(0).unsqueeze(0),
        [handover.edge_index],
        [handover.edge_attr],
    )


def _model_at_recurrent(
    model: STGNNActorCritic,
    graph_pipeline: HandoverGraphPipeline,
    time_index: int,
    device: torch.device,
    hidden_state: torch.Tensor | None,
    snapshot_transform: Callable[[HeteroData], HeteroData] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    graph = graph_pipeline.build_snapshot(time_index).to(device)
    if snapshot_transform is not None:
        graph = snapshot_transform(graph)
    handover = graph[("rh", "handover", "rh")]
    return model.forward_recurrent(
        graph["rh"].x.unsqueeze(0).unsqueeze(0),
        [handover.edge_index],
        [handover.edge_attr],
        hidden_state=hidden_state,
    )


def _model_at_windowed(
    model: STGNNActorCritic,
    graph_pipeline: HandoverGraphPipeline,
    time_index: int,
    window_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = [max(0, t) for t in range(time_index - window_size + 1, time_index + 1)]
    graphs = [graph_pipeline.build_snapshot(t).to(device) for t in indices]
    features = torch.stack([graph["rh"].x for graph in graphs], dim=0).unsqueeze(0)
    handover_edges = [graph[("rh", "handover", "rh")] for graph in graphs]
    return model(
        features,
        [edge.edge_index for edge in handover_edges],
        [edge.edge_attr for edge in handover_edges],
    )


def _masked_logits(
    logits: torch.Tensor,
    link_action_mask: np.ndarray,
    num_rhs: int,
    num_actions_per_rh: int,
) -> torch.Tensor:
    if logits.shape != (num_rhs, num_actions_per_rh):
        raise ValueError("logits must have shape (num_rhs, num_actions_per_rh)")
    mask = torch.as_tensor(link_action_mask, dtype=torch.bool, device=logits.device)
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def _assert_actions_match_mask(action_np: np.ndarray, link_action_mask: np.ndarray) -> None:
    for rh_id, action_id in enumerate(action_np):
        if not bool(link_action_mask[rh_id, int(action_id)]):
            raise ValueError(f"decoded action violates connectivity mask: rh={rh_id}, action={action_id}")


def _step_feasibility(info: dict[str, Any]) -> dict[str, bool]:
    violations = info.get("violations", {})
    connectivity_ok = (
        _violation_sum(
            violations,
            ("invalid_action", "infeasible_fronthaul_paths", "infeasible_midhaul_paths", "infeasible_direct_rh_rc_paths"),
        )
        <= 0.0
    )
    bandwidth_ok = (
        _violation_sum(
            violations,
            ("fronthaul_bw_shortfall_mbps", "midhaul_bw_shortfall_mbps", "direct_rh_rc_bw_shortfall_mbps"),
        )
        <= 0.0
    )
    latency_ok = (
        _violation_sum(
            violations,
            ("end_to_end_latency_excess_ms",),
        )
        <= 0.0
    )
    compute_ok = _violation_sum(violations, ("es_compute_overload", "rc_compute_overload")) <= 0.0
    return {
        "connectivity_ok": connectivity_ok,
        "bandwidth_ok": bandwidth_ok,
        "latency_ok": latency_ok,
        "compute_ok": compute_ok,
        "overall_ok": bool(info.get("feasible", False)),
    }


def _violation_sum(violations: dict[str, Any], keys: tuple[str, ...]) -> float:
    return float(sum(float(violations.get(key, 0.0)) for key in keys))


def _rh_decision_rows(
    t: int,
    episode: MockEpisode,
    placement: PlacementAction,
    reward: float,
    info: dict[str, Any],
    feasibility: dict[str, bool],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rho = episode.rho_rt[t] if episode.rho_rt is not None else np.zeros(episode.config.num_rhs)
    for rh_id in range(episode.config.num_rhs):
        split_id = int(placement.split_id[rh_id])
        path_type = split_path_type(split_id, episode.split_demand.split_names)
        direct_path = uses_direct_rh_rc(split_id, episode.split_demand.split_names)
        effective_es_id = "" if direct_path else int(placement.es_id[rh_id])
        pdcp_anchor_type, pdcp_anchor_id = _pdcp_anchor(placement, rh_id, episode)
        rlc_anchor_type, rlc_anchor_id = _rlc_anchor_with_episode(placement, rh_id, episode)
        rows.append(
            {
                "t": t,
                "rh_id": rh_id,
                "split_id": split_id,
                "split_name": episode.split_demand.split_names[split_id],
                "path_type": path_type,
                "uses_direct_rh_rc": direct_path,
                "es_id": int(placement.es_id[rh_id]),
                "effective_es_id": effective_es_id,
                "effective_rc_id": int(placement.rc_id[rh_id]),
                "rc_id": int(placement.rc_id[rh_id]),
                "traffic_load_mbps": float(episode.lambda_rt[t, rh_id]),
                "n_ue": int(episode.num_ues_rt[t, rh_id]),
                "s_pdcp_mb": float(episode.s_pdcp_rt[t, rh_id]),
                "s_rlc_mb": float(episode.s_rlc_rt[t, rh_id]),
                "rho": float(rho[rh_id]),
                "pdcp_anchor_type": pdcp_anchor_type,
                "pdcp_anchor_id": pdcp_anchor_id,
                "rlc_anchor_type": rlc_anchor_type,
                "rlc_anchor_id": rlc_anchor_id,
                "reward": float(reward),
                "total_cost": float(info["total_cost"]),
                "placement_cost": float(info["placement_cost"]),
                "handover_state_transfer_cost": float(info["handover_state_transfer_cost"]),
                "handover_sla_cost": float(info["handover_sla_cost"]),
                "link_violation_cost": float(info["link_violation_cost"]),
                "compute_overload_cost": float(info["compute_overload_cost"]),
                **feasibility,
            }
        )
    return rows


def _rlc_anchor_with_episode(placement: PlacementAction, rh_id: int, episode: MockEpisode) -> tuple[str, int]:
    anchor = rlc_anchor_type(int(placement.split_id[rh_id]), episode.split_demand.split_names)
    if anchor == "es":
        return "es", int(placement.es_id[rh_id])
    return "rc", int(placement.rc_id[rh_id])


def _pdcp_anchor(placement: PlacementAction, rh_id: int, episode: MockEpisode) -> tuple[str, int]:
    anchor = pdcp_anchor_type(int(placement.split_id[rh_id]), episode.split_demand.split_names)
    if anchor == "es":
        return "es", int(placement.es_id[rh_id])
    return "rc", int(placement.rc_id[rh_id])


def _es_load_rows(t: int, episode: MockEpisode, placement: PlacementAction) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for es_id in range(episode.config.num_es):
        assigned = [
            rh_id
            for rh_id in range(episode.config.num_rhs)
            if int(placement.es_id[rh_id]) == es_id
            and not uses_direct_rh_rc(int(placement.split_id[rh_id]), episode.split_demand.split_names)
        ]
        traffic = float(sum(episode.lambda_rt[t, rh_id] for rh_id in assigned))
        compute_load = float(
            sum(
                episode.lambda_rt[t, rh_id]
                * episode.split_demand.es_compute_per_mbps[int(placement.split_id[rh_id])]
                * float(uses_es_compute(int(placement.split_id[rh_id]), episode.split_demand.split_names))
                for rh_id in assigned
            )
        )
        bw_used = float(
            sum(
                episode.lambda_rt[t, rh_id]
                * episode.split_demand.fronthaul_bw_per_mbps[int(placement.split_id[rh_id])]
                for rh_id in assigned
            )
        )
        bw_capacity = float(sum(episode.fronthaul_bandwidth_mbps[rh_id, es_id] for rh_id in assigned))
        capacity = float(episode.es_compute_capacity[es_id])
        rows.append(
            {
                "t": t,
                "es_id": es_id,
                "assigned_rh_ids": _format_ids(assigned),
                "num_assigned_rhs": len(assigned),
                "traffic_load_mbps": traffic,
                "compute_load": compute_load,
                "compute_capacity": capacity,
                "compute_utilization": compute_load / capacity if capacity > 0.0 else float("inf"),
                "is_overloaded": compute_load > capacity,
                "fronthaul_bw_used_mbps": bw_used,
                "fronthaul_bw_capacity_mbps": bw_capacity,
            }
        )
    return rows


def _rc_load_rows(t: int, episode: MockEpisode, placement: PlacementAction) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rc_id in range(episode.config.num_rc):
        assigned = [rh_id for rh_id in range(episode.config.num_rhs) if int(placement.rc_id[rh_id]) == rc_id]
        traffic = float(sum(episode.lambda_rt[t, rh_id] for rh_id in assigned))
        compute_load = float(
            sum(
                episode.lambda_rt[t, rh_id]
                * episode.split_demand.rc_compute_per_mbps[int(placement.split_id[rh_id])]
                for rh_id in assigned
            )
        )
        bw_used = float(
            sum(
                episode.lambda_rt[t, rh_id]
                * episode.split_demand.midhaul_bw_per_mbps[int(placement.split_id[rh_id])]
                * float(uses_midhaul(int(placement.split_id[rh_id]), episode.split_demand.split_names))
                for rh_id in assigned
            )
        )
        selected_es_ids = sorted({int(placement.es_id[rh_id]) for rh_id in assigned})
        bw_capacity = float(sum(episode.midhaul_bandwidth_mbps[es_id, rc_id] for es_id in selected_es_ids))
        capacity = float(episode.rc_compute_capacity[rc_id])
        rows.append(
            {
                "t": t,
                "rc_id": rc_id,
                "assigned_rh_ids": _format_ids(assigned),
                "num_assigned_rhs": len(assigned),
                "traffic_load_mbps": traffic,
                "compute_load": compute_load,
                "compute_capacity": capacity,
                "compute_utilization": compute_load / capacity if capacity > 0.0 else float("inf"),
                "is_overloaded": compute_load > capacity,
                "midhaul_bw_used_mbps": bw_used,
                "midhaul_bw_capacity_mbps": bw_capacity,
            }
        )
    return rows


def _midhaul_link_load_rows(t: int, episode: MockEpisode, placement: PlacementAction) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for es_id in range(episode.config.num_es):
        for rc_id in range(episode.config.num_rc):
            assigned = [
                rh_id
                for rh_id in range(episode.config.num_rhs)
                if int(placement.es_id[rh_id]) == es_id and int(placement.rc_id[rh_id]) == rc_id
                and not uses_direct_rh_rc(int(placement.split_id[rh_id]), episode.split_demand.split_names)
            ]
            bw_used = float(
                sum(
                    episode.lambda_rt[t, rh_id]
                    * episode.split_demand.midhaul_bw_per_mbps[int(placement.split_id[rh_id])]
                    * float(uses_midhaul(int(placement.split_id[rh_id]), episode.split_demand.split_names))
                    for rh_id in assigned
                )
            )
            bw_capacity = float(episode.midhaul_bandwidth_mbps[es_id, rc_id])
            rows.append(
                {
                    "t": t,
                    "es_id": es_id,
                    "rc_id": rc_id,
                    "assigned_rh_ids": _format_ids(assigned),
                    "num_assigned_rhs": len(assigned),
                    "midhaul_bw_used_mbps": bw_used,
                    "midhaul_bw_capacity_mbps": bw_capacity,
                    "midhaul_bw_shortfall_mbps": max(bw_used - bw_capacity, 0.0),
                }
            )
    return rows


def _direct_rh_rc_link_load_rows(t: int, episode: MockEpisode, placement: PlacementAction) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rh_id in range(episode.config.num_rhs):
        split_id = int(placement.split_id[rh_id])
        if not uses_direct_rh_rc(split_id, episode.split_demand.split_names):
            continue
        rc_id = int(placement.rc_id[rh_id])
        demand = float(episode.lambda_rt[t, rh_id] * episode.split_demand.fronthaul_bw_per_mbps[split_id])
        capacity = float(episode.direct_rh_rc_bandwidth_mbps[rh_id, rc_id])
        latency = float(episode.direct_rh_rc_latency_ms[rh_id, rc_id])
        max_latency = float(episode.split_demand.max_fronthaul_latency_ms[split_id])
        rows.append(
            {
                "t": t,
                "rh_id": rh_id,
                "rc_id": rc_id,
                "demand_mbps": demand,
                "capacity_mbps": capacity,
                "latency_ms": latency,
                "shortfall_mbps": max(demand - capacity, 0.0),
                "latency_excess_ms": max(latency - max_latency, 0.0),
            }
        )
    return rows


def _format_ids(ids: list[int]) -> str:
    return ";".join(str(value) for value in ids)


def _episode_summary_row(
    *,
    episode_index: int,
    method: str,
    checkpoint_path: str,
    episode: MockEpisode,
    step_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rewards = [float(row["reward"]) for row in step_rows]
    total_costs = [float(row["total_cost"]) for row in step_rows]
    return {
        "episode_id": episode_index,
        "method": method,
        "checkpoint_path": checkpoint_path,
        "T": episode.config.horizon,
        "mean_reward": float(np.mean(rewards)) if rewards else float("nan"),
        "total_episode_reward": float(np.sum(rewards)),
        "mean_total_cost": float(np.mean(total_costs)) if total_costs else float("nan"),
        "total_episode_cost": float(np.sum(total_costs)),
        "connectivity_feasible_rate": _bool_mean(step_rows, "connectivity_ok"),
        "bandwidth_feasible_rate": _bool_mean(step_rows, "bandwidth_ok"),
        "latency_feasible_rate": _bool_mean(step_rows, "latency_ok"),
        "compute_feasible_rate": _bool_mean(step_rows, "compute_ok"),
        "overall_feasible_rate": _bool_mean(step_rows, "overall_ok"),
        "mean_handover_state_transfer_cost": _mean_info(step_rows, "handover_state_transfer_cost"),
        "mean_handover_sla_cost": _mean_info(step_rows, "handover_sla_cost"),
        "mean_link_violation_cost": _mean_info(step_rows, "link_violation_cost"),
        "mean_compute_overload_cost": _mean_info(step_rows, "compute_overload_cost"),
        "infeasible_direct_rh_rc_paths": _mean_violation(step_rows, "infeasible_direct_rh_rc_paths"),
        "direct_rh_rc_bw_shortfall_mbps": _mean_violation(step_rows, "direct_rh_rc_bw_shortfall_mbps"),
        "direct_rh_rc_latency_excess_ms": _mean_violation(step_rows, "direct_rh_rc_latency_excess_ms"),
        "direct_rh_rc_feasible_rate": _direct_rh_rc_feasible_rate(step_rows),
    }


def _bool_mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows])) if rows else float("nan")


def _mean_info(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else float("nan")


def _mean_violation(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([float(row.get("violations", {}).get(key, 0.0)) for row in rows]))


def _direct_rh_rc_feasible_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return float("nan")
    values = []
    for row in rows:
        violations = row.get("violations", {})
        direct_total = (
            float(violations.get("infeasible_direct_rh_rc_paths", 0.0))
            + float(violations.get("direct_rh_rc_bw_shortfall_mbps", 0.0))
            + float(violations.get("direct_rh_rc_latency_excess_ms", 0.0))
        )
        values.append(direct_total <= 0.0)
    return float(np.mean(values))


def _sanity_check_trace_rows(
    episode: MockEpisode,
    rh_rows: list[dict[str, Any]],
    es_rows: list[dict[str, Any]],
    rc_rows: list[dict[str, Any]],
    midhaul_link_rows: list[dict[str, Any]],
    direct_link_rows: list[dict[str, Any]],
) -> None:
    config = episode.config
    if len(rh_rows) != config.horizon * config.num_rhs:
        raise ValueError("rh_decisions rows must equal T * num_rhs")
    if len(es_rows) != config.horizon * config.num_es:
        raise ValueError("es_loads rows must equal T * num_es")
    if len(rc_rows) != config.horizon * config.num_rc:
        raise ValueError("rc_loads rows must equal T * num_rc")
    if len(midhaul_link_rows) != config.horizon * config.num_es * config.num_rc:
        raise ValueError("midhaul_link_loads rows must equal T * num_es * num_rc")

    for t in range(config.horizon):
        rh_at_t = [row for row in rh_rows if int(row["t"]) == t]
        if len(rh_at_t) != config.num_rhs:
            raise ValueError(f"t={t} must have one RH decision per RH")
        rc_count = sum(int(row["num_assigned_rhs"]) for row in rc_rows if int(row["t"]) == t)
        if rc_count != config.num_rhs:
            raise ValueError(f"RC assigned RH count at t={t} must equal num_rhs")
        midhaul_pair_count = sum(
            int(row["num_assigned_rhs"]) for row in midhaul_link_rows if int(row["t"]) == t
        )
        direct_count = sum(1 for row in direct_link_rows if int(row["t"]) == t)
        if midhaul_pair_count + direct_count != config.num_rhs:
            raise ValueError(f"gateway plus direct assigned RH count at t={t} must equal num_rhs")
        for es_row in [row for row in es_rows if int(row["t"]) == t]:
            es_id = int(es_row["es_id"])
            expected = sorted(
                int(row["rh_id"])
                for row in rh_at_t
                if int(row["es_id"]) == es_id and not bool(row.get("uses_direct_rh_rc", False))
            )
            if _parse_ids(str(es_row["assigned_rh_ids"])) != expected:
                raise ValueError(f"ES assigned_rh_ids mismatch at t={t}, es={es_id}")
        for rc_row in [row for row in rc_rows if int(row["t"]) == t]:
            rc_id = int(rc_row["rc_id"])
            expected = sorted(int(row["rh_id"]) for row in rh_at_t if int(row["rc_id"]) == rc_id)
            if _parse_ids(str(rc_row["assigned_rh_ids"])) != expected:
                raise ValueError(f"RC assigned_rh_ids mismatch at t={t}, rc={rc_id}")


def _sanity_check_exported_files(output_root: Path, episode: MockEpisode) -> None:
    import polars as pl

    config = episode.config
    rh_decisions = pl.read_csv(output_root / "rh_decisions.csv")
    es_loads = pl.read_csv(output_root / "es_loads.csv")
    rc_loads = pl.read_csv(output_root / "rc_loads.csv")
    midhaul_link_loads = pl.read_csv(output_root / "midhaul_link_loads.csv")
    direct_link_loads = pl.read_csv(output_root / "direct_rh_rc_link_loads.csv")
    if rh_decisions.height != config.horizon * config.num_rhs:
        raise ValueError("rh_decisions.csv row count mismatch")
    if es_loads.height != config.horizon * config.num_es:
        raise ValueError("es_loads.csv row count mismatch")
    if rc_loads.height != config.horizon * config.num_rc:
        raise ValueError("rc_loads.csv row count mismatch")
    if midhaul_link_loads.height != config.horizon * config.num_es * config.num_rc:
        raise ValueError("midhaul_link_loads.csv row count mismatch")
    if direct_link_loads.height > config.horizon * config.num_rhs:
        raise ValueError("direct_rh_rc_link_loads.csv row count mismatch")
    for rh_id in range(config.num_rhs):
        rh_file = pl.read_csv(output_root / "rh" / f"RH_{rh_id}.csv")
        if rh_file.height != config.horizon:
            raise ValueError(f"rh/RH_{rh_id}.csv row count mismatch")


def _parse_ids(value: str) -> list[int]:
    if value == "":
        return []
    return sorted(int(part) for part in value.split(";"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


RH_DECISION_COLUMNS = [
    "t",
    "rh_id",
    "split_id",
    "split_name",
    "path_type",
    "uses_direct_rh_rc",
    "es_id",
    "effective_es_id",
    "effective_rc_id",
    "rc_id",
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
    "connectivity_ok",
    "bandwidth_ok",
    "latency_ok",
    "compute_ok",
    "overall_ok",
]

ES_LOAD_COLUMNS = [
    "t",
    "es_id",
    "assigned_rh_ids",
    "num_assigned_rhs",
    "traffic_load_mbps",
    "compute_load",
    "compute_capacity",
    "compute_utilization",
    "is_overloaded",
    "fronthaul_bw_used_mbps",
    "fronthaul_bw_capacity_mbps",
]

RC_LOAD_COLUMNS = [
    "t",
    "rc_id",
    "assigned_rh_ids",
    "num_assigned_rhs",
    "traffic_load_mbps",
    "compute_load",
    "compute_capacity",
    "compute_utilization",
    "is_overloaded",
    "midhaul_bw_used_mbps",
    "midhaul_bw_capacity_mbps",
]

MIDHAUL_LINK_LOAD_COLUMNS = [
    "t",
    "es_id",
    "rc_id",
    "assigned_rh_ids",
    "num_assigned_rhs",
    "midhaul_bw_used_mbps",
    "midhaul_bw_capacity_mbps",
    "midhaul_bw_shortfall_mbps",
]

DIRECT_RH_RC_LINK_LOAD_COLUMNS = [
    "t",
    "rh_id",
    "rc_id",
    "demand_mbps",
    "capacity_mbps",
    "latency_ms",
    "shortfall_mbps",
    "latency_excess_ms",
]

EPISODE_SUMMARY_COLUMNS = [
    "episode_id",
    "method",
    "checkpoint_path",
    "T",
    "mean_reward",
    "total_episode_reward",
    "mean_total_cost",
    "total_episode_cost",
    "connectivity_feasible_rate",
    "bandwidth_feasible_rate",
    "latency_feasible_rate",
    "compute_feasible_rate",
    "overall_feasible_rate",
    "mean_handover_state_transfer_cost",
    "mean_handover_sla_cost",
    "mean_link_violation_cost",
    "mean_compute_overload_cost",
    "infeasible_direct_rh_rc_paths",
    "direct_rh_rc_bw_shortfall_mbps",
    "direct_rh_rc_latency_excess_ms",
    "direct_rh_rc_feasible_rate",
]


def main() -> None:
    args = parse_args()
    result = run_episode_trace(
        checkpoint_path=args.checkpoint,
        method=args.method,
        config_path=args.config,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        deterministic=args.deterministic,
        temporal_window_size=args.temporal_window_size,
        episode_index=args.episode_index,
    )
    print(f"wrote RH decisions to {result.rh_decisions}")
    print(f"wrote ES loads to {result.es_loads}")
    print(f"wrote RC loads to {result.rc_loads}")
    print(f"wrote direct RH-RC link loads to {result.direct_rh_rc_link_loads}")
    print(f"wrote episode summary to {result.episode_summary}")


if __name__ == "__main__":
    main()

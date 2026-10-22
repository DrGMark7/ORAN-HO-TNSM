"""In-memory experiment modes for project-compatible GPPO ablations."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from torch_geometric.data import HeteroData

from src.pipeline.schema import DEFAULT_GRAPH_FEATURE_SCHEMA, MockEpisode


FeatureMode = Literal["no_handover", "handover"]
RewardMode = Literal["baseline_no_ho", "handover_aware"]

FEATURE_MODES = ("no_handover", "handover")
REWARD_MODES = ("baseline_no_ho", "handover_aware")
SUPPORTED_SPLIT_MODE = "5split_project"
PROJECT_5SPLIT_NAMES = ("S1", "S2", "S3", "S4G", "S4D")
HO_REWARD_WEIGHT_NAMES = (
    "pdcp_state_transfer",
    "rlc_state_transfer",
    "handover_sla",
    "rrc_anchor_crossing",
)


def validate_gppo_modes(feature_mode: str, reward_mode: str, split_mode: str) -> None:
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"feature_mode must be one of {FEATURE_MODES}")
    if reward_mode not in REWARD_MODES:
        raise ValueError(f"reward_mode must be one of {REWARD_MODES}")
    if split_mode != SUPPORTED_SPLIT_MODE:
        raise NotImplementedError(
            "strict Ngo-compatible four-split semantics are not implemented; "
            f"split_mode must be {SUPPORTED_SPLIT_MODE!r}"
        )


def validate_project_5split_episode(episode: MockEpisode) -> None:
    """Reject legacy or ambiguous split semantics in the first GPPO baseline."""

    if episode.config.num_splits != 5 or tuple(episode.split_demand.split_names) != PROJECT_5SPLIT_NAMES:
        raise ValueError(
            "project_gppo_style_5split requires split_names "
            f"{PROJECT_5SPLIT_NAMES}, got {tuple(episode.split_demand.split_names)}"
        )


def episode_for_reward_mode(episode: MockEpisode, reward_mode: RewardMode) -> MockEpisode:
    """Return an episode with runtime-only reward weights for the selected mode."""

    if reward_mode == "handover_aware":
        return episode
    if reward_mode != "baseline_no_ho":
        raise ValueError(f"reward_mode must be one of {REWARD_MODES}")
    return replace(
        episode,
        cost_weights=replace(
            episode.cost_weights,
            **{name: 0.0 for name in HO_REWARD_WEIGHT_NAMES},
        ),
    )


def snapshot_for_feature_mode(graph: HeteroData, feature_mode: FeatureMode) -> HeteroData:
    """Return a graph snapshot with handover inputs masked in memory when requested."""

    if feature_mode == "handover":
        return graph
    if feature_mode != "no_handover":
        raise ValueError(f"feature_mode must be one of {FEATURE_MODES}")

    masked = graph.clone()
    rh_feature_names = tuple(
        getattr(masked["rh"], "feature_names", DEFAULT_GRAPH_FEATURE_SCHEMA.rh_node_features)
    )
    for feature_name in ("s_pdcp_mb", "s_rlc_mb", "rho"):
        masked["rh"].x[:, rh_feature_names.index(feature_name)] = 0.0

    handover = masked[("rh", "handover", "rh")]
    edge_feature_names = tuple(
        getattr(handover, "feature_names", DEFAULT_GRAPH_FEATURE_SCHEMA.handover_edge_features)
    )
    for feature_name in ("observed_handover_prob", "observed_handover_count"):
        handover.edge_attr[:, edge_feature_names.index(feature_name)] = 0.0

    # The pipeline uses the complete directed non-self RH graph at every timestep,
    # so retaining edge_index cannot leak dynamic handover topology.
    return masked


def reward_mode_weights(episode: MockEpisode, reward_mode: RewardMode) -> dict[str, float]:
    """Return the effective reward weights for experiment metadata."""

    resolved = episode_for_reward_mode(episode, reward_mode).cost_weights
    return {name: float(value) for name, value in vars(resolved).items()}

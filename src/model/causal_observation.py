"""Versioned, leak-free RH resource/action observation features."""

from __future__ import annotations

import numpy as np

from src.model.constraint import ResourceLoads
from src.model.serial_numba import (
    CausalKernelStatic,
    causal_args,
    causal_kernel,
    causal_v2_kernel,
)
from src.model.split_semantics import uses_direct_rh_rc, uses_es_compute, uses_midhaul
from src.pipeline.schema import MockEpisode, PlacementAction

LEGACY_OBSERVATION_FEATURE_MODE = "legacy_rh_only"
CAUSAL_RESOURCE_OBSERVATION_FEATURE_MODE = "causal_resource_v1"
CAUSAL_RESOURCE_V2_OBSERVATION_FEATURE_MODE = "causal_resource_v2"
OBSERVATION_FEATURE_MODES = (
    LEGACY_OBSERVATION_FEATURE_MODE,
    CAUSAL_RESOURCE_OBSERVATION_FEATURE_MODE,
    CAUSAL_RESOURCE_V2_OBSERVATION_FEATURE_MODE,
)
RESOURCE_OBSERVATION_FEATURE_MODES = (
    CAUSAL_RESOURCE_OBSERVATION_FEATURE_MODE,
    CAUSAL_RESOURCE_V2_OBSERVATION_FEATURE_MODE,
)
LEGACY_OBSERVATION_SCHEMA = "daily_288_lagged_observation_v3"
CAUSAL_RESOURCE_OBSERVATION_SCHEMA = "daily_288_lagged_observation_resource_v1"
CAUSAL_RESOURCE_V2_OBSERVATION_SCHEMA = "daily_288_lagged_observation_resource_v2"


def validate_observation_feature_mode(mode: str) -> str:
    if mode not in OBSERVATION_FEATURE_MODES:
        raise ValueError(f"observation_feature_mode must be one of {OBSERVATION_FEATURE_MODES}")
    return mode


def rh_feature_names(episode: MockEpisode, mode: str) -> tuple[str, ...]:
    validate_observation_feature_mode(mode)
    base = ("n_ue", "traffic_load_mbps", "s_pdcp_mb", "s_rlc_mb", "rho")
    if mode == LEGACY_OBSERVATION_FEATURE_MODE:
        return base
    names = list(base)
    names += [f"previous_split_{i}" for i in range(episode.config.num_splits)]
    names += [f"previous_es_{i}" for i in range(episode.config.num_es)]
    names += [f"previous_rc_{i}" for i in range(episode.config.num_rc)]
    names += ["previous_path_gateway", "previous_path_direct", "previous_action_valid"]
    if mode == CAUSAL_RESOURCE_V2_OBSERVATION_FEATURE_MODE:
        names += [
            "previous_selected_es_remaining",
            "previous_selected_rc_remaining",
            "previous_selected_transport_link_remaining",
            "previous_selected_midhaul_remaining",
        ]
        for resource in ("feasible_es", "feasible_rc", "feasible_midhaul"):
            names += [f"previous_{resource}_remaining_{stat}" for stat in ("min", "mean", "max")]
        return tuple(names)
    for resource in ("selected_link", "selected_es", "selected_midhaul", "selected_rc"):
        names += [f"previous_{resource}_utilization", f"previous_{resource}_remaining_ratio", f"previous_{resource}_available"]
    for resource in ("feasible_es", "feasible_rc", "feasible_midhaul"):
        names += [f"previous_{resource}_{stat}_utilization" for stat in ("min", "mean", "max")]
    return tuple(names)


def rh_feature_dim(episode: MockEpisode, mode: str) -> int:
    return len(rh_feature_names(episode, mode))


def observation_schema_for_mode(episode: MockEpisode, mode: str) -> str:
    """Return the checkpoint-visible schema identity for an observation mode."""

    validate_observation_feature_mode(mode)
    if mode == CAUSAL_RESOURCE_OBSERVATION_FEATURE_MODE:
        return CAUSAL_RESOURCE_OBSERVATION_SCHEMA
    if mode == CAUSAL_RESOURCE_V2_OBSERVATION_FEATURE_MODE:
        return CAUSAL_RESOURCE_V2_OBSERVATION_SCHEMA
    return str(episode.metadata.get("mobility_observation_schema_version", "legacy_same_step_v1"))


def causal_resource_features(
    episode: MockEpisode,
    previous_action: PlacementAction | None,
    previous_loads: ResourceLoads | None,
) -> np.ndarray:
    """Return per-RH context generated solely by the previous successful step."""

    cfg = episode.config
    width = rh_feature_dim(episode, CAUSAL_RESOURCE_OBSERVATION_FEATURE_MODE) - 5
    result = np.zeros((cfg.num_rhs, width), dtype=np.float32)
    if previous_action is None or previous_loads is None:
        return result

    def ratio(load: float, capacity: float) -> tuple[float, float]:
        utilization = float(load) / float(capacity)
        return utilization, 1.0 - utilization

    es_u = (previous_loads.es_utilization if previous_loads.es_utilization is not None
            else previous_loads.es / episode.es_compute_capacity)
    rc_u = (previous_loads.rc_utilization if previous_loads.rc_utilization is not None
            else previous_loads.rc / episode.rc_compute_capacity)
    mh_u = np.divide(
        previous_loads.midhaul, episode.midhaul_bandwidth_mbps,
        out=np.zeros_like(previous_loads.midhaul), where=episode.midhaul_bandwidth_mbps > 0,
    )
    for r in range(cfg.num_rhs):
        split, es, rc = (int(previous_action.split_id[r]), int(previous_action.es_id[r]), int(previous_action.rc_id[r]))
        values: list[float] = []
        values += np.eye(cfg.num_splits, dtype=np.float32)[split].tolist()
        values += np.eye(cfg.num_es, dtype=np.float32)[es].tolist()
        values += np.eye(cfg.num_rc, dtype=np.float32)[rc].tolist()
        direct = uses_direct_rh_rc(split, episode.split_demand.split_names)
        values += [float(not direct), float(direct), 1.0]
        if direct:
            values += [*ratio(previous_loads.direct[r, rc], episode.direct_rh_rc_bandwidth_mbps[r, rc]), 1.0]
            values += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            values += [*ratio(previous_loads.fronthaul[r, es], episode.fronthaul_bandwidth_mbps[r, es]), 1.0]
            if uses_es_compute(split, episode.split_demand.split_names):
                values += [*ratio(previous_loads.es[es], episode.es_compute_capacity[es]), 1.0]
            else:
                values += [0.0, 0.0, 0.0]
            if uses_midhaul(split, episode.split_demand.split_names):
                values += [*ratio(previous_loads.midhaul[es, rc], episode.midhaul_bandwidth_mbps[es, rc]), 1.0]
            else:
                values += [0.0, 0.0, 0.0]
        values += [*ratio(previous_loads.rc[rc], episode.rc_compute_capacity[rc]), 1.0]

        feasible_es = np.flatnonzero(episode.fronthaul_mask[r])
        feasible_rc = set(np.flatnonzero(episode.direct_rh_rc_mask[r]).tolist())
        feasible_mh: list[float] = []
        for es_id in feasible_es:
            for rc_id in np.flatnonzero(episode.midhaul_mask[es_id]):
                feasible_rc.add(int(rc_id)); feasible_mh.append(float(mh_u[es_id, rc_id]))
        for candidates in (es_u[feasible_es], rc_u[sorted(feasible_rc)], np.asarray(feasible_mh)):
            values += ([float(np.min(candidates)), float(np.mean(candidates)), float(np.max(candidates))] if len(candidates) else [0.0, 0.0, 0.0])
        result[r] = np.asarray(values, dtype=np.float32)
    return result


def causal_resource_v2_features(
    episode: MockEpisode,
    previous_action: PlacementAction | None,
    previous_loads: ResourceLoads | None,
) -> np.ndarray:
    """Return the compact identity-preserving v2 context."""

    cfg = episode.config
    width = rh_feature_dim(episode, CAUSAL_RESOURCE_V2_OBSERVATION_FEATURE_MODE) - 5
    result = np.zeros((cfg.num_rhs, width), dtype=np.float32)
    if previous_action is None or previous_loads is None:
        return result

    es_u = (previous_loads.es_utilization if previous_loads.es_utilization is not None
            else previous_loads.es / episode.es_compute_capacity)
    rc_u = (previous_loads.rc_utilization if previous_loads.rc_utilization is not None
            else previous_loads.rc / episode.rc_compute_capacity)
    mh_u = np.divide(
        previous_loads.midhaul, episode.midhaul_bandwidth_mbps,
        out=np.zeros_like(previous_loads.midhaul), where=episode.midhaul_bandwidth_mbps > 0,
    )
    identity_width = cfg.num_splits + cfg.num_es + cfg.num_rc
    for r in range(cfg.num_rhs):
        split = int(previous_action.split_id[r])
        es = int(previous_action.es_id[r])
        rc = int(previous_action.rc_id[r])
        result[r, split] = 1.0
        result[r, cfg.num_splits + es] = 1.0
        result[r, cfg.num_splits + cfg.num_es + rc] = 1.0
        direct = uses_direct_rh_rc(split, episode.split_demand.split_names)
        result[r, identity_width : identity_width + 3] = (not direct, direct, 1.0)

        offset = identity_width + 3
        if uses_es_compute(split, episode.split_demand.split_names):
            result[r, offset] = np.float32(
                1.0 - float(previous_loads.es[es]) / float(episode.es_compute_capacity[es])
            )
        result[r, offset + 1] = np.float32(
            1.0 - float(previous_loads.rc[rc]) / float(episode.rc_compute_capacity[rc])
        )
        if direct:
            result[r, offset + 2] = np.float32(
                1.0 - float(previous_loads.direct[r, rc])
                / float(episode.direct_rh_rc_bandwidth_mbps[r, rc])
            )
        else:
            result[r, offset + 2] = np.float32(
                1.0 - float(previous_loads.fronthaul[r, es])
                / float(episode.fronthaul_bandwidth_mbps[r, es])
            )
            if uses_midhaul(split, episode.split_demand.split_names):
                result[r, offset + 3] = np.float32(
                    1.0 - float(previous_loads.midhaul[es, rc])
                    / float(episode.midhaul_bandwidth_mbps[es, rc])
                )

        feasible_es = np.flatnonzero(episode.fronthaul_mask[r])
        feasible_rc = set(np.flatnonzero(episode.direct_rh_rc_mask[r]).tolist())
        feasible_mh: list[float] = []
        for es_id in feasible_es:
            for rc_id in np.flatnonzero(episode.midhaul_mask[es_id]):
                feasible_rc.add(int(rc_id))
                feasible_mh.append(float(mh_u[es_id, rc_id]))
        offset += 4
        for candidates in (es_u[feasible_es], rc_u[sorted(feasible_rc)], np.asarray(feasible_mh)):
            if len(candidates):
                result[r, offset : offset + 3] = np.asarray(
                    (1.0 - np.max(candidates), 1.0 - np.mean(candidates), 1.0 - np.min(candidates)),
                    dtype=np.float32,
                )
            offset += 3
    return result


def causal_resource_features_for_mode(
    episode: MockEpisode,
    previous_action: PlacementAction | None,
    previous_loads: ResourceLoads | None,
    mode: str,
) -> np.ndarray:
    """Dispatch an explicitly supported resource observation mode."""

    if mode == CAUSAL_RESOURCE_OBSERVATION_FEATURE_MODE:
        return causal_resource_features(episode, previous_action, previous_loads)
    if mode == CAUSAL_RESOURCE_V2_OBSERVATION_FEATURE_MODE:
        return causal_resource_v2_features(episode, previous_action, previous_loads)
    raise ValueError(f"mode is not a resource observation mode: {mode!r}")


def causal_resource_features_numba(
    episode: MockEpisode,
    previous_action: PlacementAction | None,
    previous_loads: ResourceLoads | None,
    static: CausalKernelStatic,
    mode: str = CAUSAL_RESOURCE_OBSERVATION_FEATURE_MODE,
) -> np.ndarray:
    """Run the optional exact kernel; callers own fallback after compilation failure."""

    if previous_action is None or previous_loads is None:
        return np.zeros(
            (
                episode.config.num_rhs,
                rh_feature_dim(episode, mode) - 5,
            ),
            dtype=np.float32,
        )
    kernel = (
        causal_kernel
        if mode == CAUSAL_RESOURCE_OBSERVATION_FEATURE_MODE
        else causal_v2_kernel if mode == CAUSAL_RESOURCE_V2_OBSERVATION_FEATURE_MODE else None
    )
    if kernel is None:
        raise RuntimeError("serial Numba causal kernel is unavailable")
    return kernel(*causal_args(episode, previous_action, previous_loads, static))


def append_causal_features(
    graph: object,
    features: np.ndarray,
    episode: MockEpisode,
    mode: str = CAUSAL_RESOURCE_OBSERVATION_FEATURE_MODE,
) -> object:
    """Clone a PyG graph and append causal features without altering legacy graphs."""
    import torch
    enriched = graph.clone()
    enriched["rh"].x = torch.cat((enriched["rh"].x, torch.as_tensor(features, dtype=enriched["rh"].x.dtype)), dim=1)
    enriched["rh"].feature_names = rh_feature_names(episode, mode)
    return enriched

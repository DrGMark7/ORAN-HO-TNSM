"""Centralized action encoding and decoding utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.pipeline.schema import IntArray, MockEpisode, PlacementAction, TopologyConfig
from src.model.split_semantics import canonical_es_id_for_split, infer_split_names, uses_direct_rh_rc


@dataclass(frozen=True)
class ActionSpaceSpec:
    """Flattened categorical action space for one RH."""

    num_splits: int
    num_es: int
    num_rc: int

    @classmethod
    def from_topology(cls, config: TopologyConfig) -> "ActionSpaceSpec":
        return cls(config.num_splits, config.num_es, config.num_rc)

    @property
    def num_actions_per_rh(self) -> int:
        return self.num_splits * self.num_es * self.num_rc

    def validate_flat_action(self, action_id: int) -> None:
        if action_id < 0 or action_id >= self.num_actions_per_rh:
            raise ValueError(
                f"action_id must be in [0, {self.num_actions_per_rh}); got {action_id}"
            )


def encode_tuple_action(split_id: int, es_id: int, rc_id: int, spec: ActionSpaceSpec) -> int:
    """Encode ``(split_id, es_id, rc_id)`` as one categorical action."""

    if split_id < 0 or split_id >= spec.num_splits:
        raise ValueError("split_id is out of range")
    if es_id < 0 or es_id >= spec.num_es:
        raise ValueError("es_id is out of range")
    if rc_id < 0 or rc_id >= spec.num_rc:
        raise ValueError("rc_id is out of range")
    return (split_id * spec.num_es * spec.num_rc) + (es_id * spec.num_rc) + rc_id


def decode_tuple_action(action_id: int, spec: ActionSpaceSpec) -> tuple[int, int, int]:
    """Decode one categorical action into ``(split_id, es_id, rc_id)``."""

    spec.validate_flat_action(int(action_id))
    split_id = int(action_id) // (spec.num_es * spec.num_rc)
    remainder = int(action_id) % (spec.num_es * spec.num_rc)
    es_id = remainder // spec.num_rc
    rc_id = remainder % spec.num_rc
    return split_id, es_id, rc_id


def decode_flat_actions(actions: IntArray | list[int], config: TopologyConfig) -> PlacementAction:
    """Decode one flattened categorical action for each RH."""

    action_array = np.asarray(actions, dtype=np.int64)
    if action_array.shape != (config.num_rhs,):
        raise ValueError(f"actions must have shape ({config.num_rhs},)")
    spec = ActionSpaceSpec.from_topology(config)
    decoded = np.array([decode_tuple_action(int(a), spec) for a in action_array], dtype=np.int64)
    placement = PlacementAction(
        split_id=decoded[:, 0],
        es_id=decoded[:, 1],
        rc_id=decoded[:, 2],
    )
    placement.validate(config)
    return placement


def encode_flat_actions(placement: PlacementAction, config: TopologyConfig) -> IntArray:
    """Encode decoded per-RH tuples into flattened categorical actions."""

    placement.validate(config)
    spec = ActionSpaceSpec.from_topology(config)
    return np.array(
        [
            encode_tuple_action(int(s), int(d), int(c), spec)
            for s, d, c in zip(placement.split_id, placement.es_id, placement.rc_id, strict=True)
        ],
        dtype=np.int64,
    )


def decode_multidiscrete_actions(actions: IntArray | list[list[int]], config: TopologyConfig) -> PlacementAction:
    """Decode explicit ``[split_id, es_id, rc_id]`` rows for each RH."""

    action_array = np.asarray(actions, dtype=np.int64)
    if action_array.shape != (config.num_rhs, 3):
        raise ValueError(f"actions must have shape ({config.num_rhs}, 3)")
    placement = PlacementAction(
        split_id=action_array[:, 0],
        es_id=action_array[:, 1],
        rc_id=action_array[:, 2],
    )
    placement.validate(config)
    return placement


def build_link_feasibility_action_mask(
    config: TopologyConfig,
    fronthaul_mask: np.ndarray,
    midhaul_mask: np.ndarray,
    direct_rh_rc_mask: np.ndarray | None = None,
    split_names: tuple[str, ...] | list[str] | None = None,
    allowed_split_names: tuple[str, ...] | list[str] | None = None,
    topology_profile: str = "legacy",
    ensure_each_rh_has_valid_action: bool = True,
) -> NDArray[np.bool_]:
    """Build per-RH action masks for static fronthaul/midhaul feasibility.

    The returned mask has shape ``[num_rhs, num_actions_per_rh]``. A flattened
    gateway action is valid when selected ``rh->es`` fronthaul and ``es->rc``
    midhaul paths are feasible. Direct S4D actions are valid only when selected
    ``rh->rc`` direct transport exists and ``es_id`` equals the canonical
    placeholder ES. Compute capacity is intentionally not masked here.
    """

    fronthaul = np.asarray(fronthaul_mask, dtype=np.bool_)
    midhaul = np.asarray(midhaul_mask, dtype=np.bool_)
    if fronthaul.shape != (config.num_rhs, config.num_es):
        raise ValueError("fronthaul_mask shape must be (num_rhs, num_es)")
    if midhaul.shape != (config.num_es, config.num_rc):
        raise ValueError("midhaul_mask shape must be (num_es, num_rc)")
    direct = (
        np.zeros((config.num_rhs, config.num_rc), dtype=np.bool_)
        if direct_rh_rc_mask is None
        else np.asarray(direct_rh_rc_mask, dtype=np.bool_)
    )
    if direct.shape != (config.num_rhs, config.num_rc):
        raise ValueError("direct_rh_rc_mask shape must be (num_rhs, num_rc)")
    resolved_split_names = infer_split_names(config.num_splits, split_names)
    resolved_allowed_split_names = _resolve_allowed_split_names(
        resolved_split_names,
        allowed_split_names,
    )
    allowed_split_ids = {
        split_id
        for split_id, split_name in enumerate(resolved_split_names)
        if split_name in resolved_allowed_split_names
    }

    summary = summarize_link_feasibility_action_mask(
        config,
        fronthaul,
        midhaul,
        direct,
        resolved_split_names,
        resolved_allowed_split_names,
    )
    mask = summary["mask"]
    gateway_valid_counts = summary["gateway_valid_counts"]
    direct_valid_counts = summary["direct_valid_counts"]
    valid_counts = summary["valid_counts"]
    invalid_rhs = np.flatnonzero(valid_counts == 0)
    if ensure_each_rh_has_valid_action and invalid_rhs.size > 0:
        detail = [
            "rh={rh_id} gateway_valid_count={gateway} s4d_valid_count={direct}".format(
                rh_id=int(rh_id),
                gateway=int(gateway_valid_counts[rh_id]),
                direct=int(direct_valid_counts[rh_id]),
            )
            for rh_id in invalid_rhs
        ]
        raise ValueError(
            "link-feasibility action mask has no valid action for RH ids after split policy and link feasibility; "
            f"rh_ids={invalid_rhs.tolist()} topology_profile={topology_profile!r} "
            f"allowed_split_names={resolved_allowed_split_names!r} details={detail}"
        )
    return mask


def build_topology_profile_action_masks(episode: MockEpisode) -> dict[str, NDArray[np.bool_]]:
    """Build one reusable action mask for each cached episode topology profile."""

    kwargs = action_mask_kwargs_from_episode(episode)
    return {
        profile_id: build_link_feasibility_action_mask(
            episode.config,
            profile.fronthaul_mask,
            profile.midhaul_mask,
            profile.direct_rh_rc_mask,
            episode.split_demand.split_names,
            **kwargs,
        )
        for profile_id, profile in episode.topology_profiles.items()
    }


def topology_action_mask_for_time_index(
    episode: MockEpisode,
    time_index: int,
    *,
    episode_length: int,
    profile_masks: dict[str, NDArray[np.bool_]] | None = None,
) -> NDArray[np.bool_]:
    """Select only the action mask assigned to an absolute dataset timestep."""

    if episode.topology_profiles:
        if episode_length <= 0:
            raise ValueError("episode_length must be positive")
        profile_id = episode.topology_profile_id_for_episode(time_index // episode_length)
        masks = profile_masks if profile_masks is not None else build_topology_profile_action_masks(episode)
        if profile_id is None:
            raise ValueError("profile-aware episode is missing an episode assignment")
        return masks[profile_id]
    return build_link_feasibility_action_mask(
        episode.config,
        episode.fronthaul_mask,
        episode.midhaul_mask,
        episode.direct_rh_rc_mask,
        episode.split_demand.split_names,
        **action_mask_kwargs_from_episode(episode),
    )


def summarize_link_feasibility_action_mask(
    config: TopologyConfig,
    fronthaul_mask: np.ndarray,
    midhaul_mask: np.ndarray,
    direct_rh_rc_mask: np.ndarray | None = None,
    split_names: tuple[str, ...] | list[str] | None = None,
    allowed_split_names: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Return the link-feasibility mask plus per-RH gateway/direct valid counts."""

    fronthaul = np.asarray(fronthaul_mask, dtype=np.bool_)
    midhaul = np.asarray(midhaul_mask, dtype=np.bool_)
    if fronthaul.shape != (config.num_rhs, config.num_es):
        raise ValueError("fronthaul_mask shape must be (num_rhs, num_es)")
    if midhaul.shape != (config.num_es, config.num_rc):
        raise ValueError("midhaul_mask shape must be (num_es, num_rc)")
    direct = (
        np.zeros((config.num_rhs, config.num_rc), dtype=np.bool_)
        if direct_rh_rc_mask is None
        else np.asarray(direct_rh_rc_mask, dtype=np.bool_)
    )
    if direct.shape != (config.num_rhs, config.num_rc):
        raise ValueError("direct_rh_rc_mask shape must be (num_rhs, num_rc)")
    resolved_split_names = infer_split_names(config.num_splits, split_names)
    resolved_allowed_split_names = _resolve_allowed_split_names(
        resolved_split_names,
        allowed_split_names,
    )
    allowed_split_ids = {
        split_id
        for split_id, split_name in enumerate(resolved_split_names)
        if split_name in resolved_allowed_split_names
    }
    spec = ActionSpaceSpec.from_topology(config)
    mask = np.zeros((config.num_rhs, spec.num_actions_per_rh), dtype=np.bool_)
    gateway_valid_counts = np.zeros(config.num_rhs, dtype=np.int64)
    direct_valid_counts = np.zeros(config.num_rhs, dtype=np.int64)
    for rh_id in range(config.num_rhs):
        for action_id in range(spec.num_actions_per_rh):
            split_id, es_id, rc_id = decode_tuple_action(action_id, spec)
            if split_id not in allowed_split_ids:
                continue
            if uses_direct_rh_rc(split_id, resolved_split_names):
                canonical_es = canonical_es_id_for_split(split_id, resolved_split_names)
                valid = bool(es_id == canonical_es and direct[rh_id, rc_id])
                mask[rh_id, action_id] = valid
                if valid:
                    direct_valid_counts[rh_id] += 1
            else:
                valid = bool(fronthaul[rh_id, es_id] and midhaul[es_id, rc_id])
                mask[rh_id, action_id] = valid
                if valid:
                    gateway_valid_counts[rh_id] += 1
    valid_counts = mask.sum(axis=1).astype(np.int64)
    return {
        "mask": mask,
        "gateway_valid_counts": gateway_valid_counts,
        "direct_valid_counts": direct_valid_counts,
        "valid_counts": valid_counts,
        "split_names": resolved_split_names,
        "allowed_split_names": resolved_allowed_split_names,
    }


def action_mask_kwargs_from_episode(episode: Any) -> dict[str, Any]:
    """Return policy/topology metadata kwargs for action-mask construction."""

    metadata = getattr(episode, "metadata", {}) or {}
    split_names = getattr(getattr(episode, "split_demand", None), "split_names", None)
    return {
        "allowed_split_names": tuple(metadata.get("allowed_split_names", split_names or ())),
        "topology_profile": str(metadata.get("topology_profile", "legacy")),
        "ensure_each_rh_has_valid_action": bool(metadata.get("ensure_each_rh_has_valid_action", True)),
    }


def _resolve_allowed_split_names(
    split_names: tuple[str, ...],
    allowed_split_names: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if allowed_split_names is None:
        return split_names
    resolved = tuple(str(name) for name in allowed_split_names)
    invalid = [name for name in resolved if name not in split_names]
    if invalid:
        raise ValueError(
            "allowed_split_names contains unknown split names "
            f"{invalid}; known split names are {list(split_names)}"
        )
    return resolved

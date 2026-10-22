"""Shared schemas for handover-aware O-RAN orchestration.

Mathematical variable mapping follows ``markdown/HandoverConstraint_revised.md``:

* ``lambda_rt``: aggregate RH traffic load, shape ``[T, R]``.
* ``num_ues_rt``: number of UEs served by each RH, shape ``[T, R]``.
* ``handover_count_t``: sampled integer UE handover counts ``C[t, r, r']``,
  shape ``[T, R, R]`` with diagonal entries fixed to zero.
* ``handover_matrix_t``: handover fraction ``H[t, r, r'] = C[t, r, r'] / N[t, r]``,
  shape ``[T, R, R]`` with diagonal entries fixed to zero.
* ``s_pdcp_rt`` and ``s_rlc_rt``: expected layer state volume in MB/UE,
  each with shape ``[T, R]``.
* ``observed_handover_count_t`` and ``observed_handover_matrix_t``: positive-lag
  mobility observations available before action selection, shape ``[T, R, R]``.
* ``rho_rt``: lagged EWMA outgoing mobility pressure, shape ``[T, R]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.model.split_semantics import contains_direct_rc_split


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class MockDatasetPaths:
    """CSV files that define one exported mock dataset."""

    root_dir: str
    rh_timeseries: str
    handover_edges: str
    rh_nodes: str
    es_nodes: str
    rc_nodes: str
    fronthaul_edges: str
    midhaul_edges: str
    direct_rh_rc_edges: str | None = None
    effective_generation_config: str | None = None
    episode_topology_profiles: str | None = None


@dataclass(frozen=True)
class GraphFeatureSchema:
    """Feature ordering used when converting CSV data into PyG graphs."""

    rh_node_features: tuple[str, ...]
    es_node_features: tuple[str, ...]
    rc_node_features: tuple[str, ...]
    handover_edge_features: tuple[str, ...]
    fronthaul_edge_features: tuple[str, ...]
    midhaul_edge_features: tuple[str, ...]
    direct_rh_rc_edge_features: tuple[str, ...] = ("connected", "latency_ms", "bandwidth_mbps", "cost")


DEFAULT_GRAPH_FEATURE_SCHEMA = GraphFeatureSchema(
    rh_node_features=("n_ue", "traffic_load_mbps", "s_pdcp_mb", "s_rlc_mb", "rho"),
    es_node_features=("cpu_capacity",),
    rc_node_features=("cpu_capacity",),
    handover_edge_features=("observed_handover_prob", "observed_handover_count"),
    fronthaul_edge_features=("feasible", "latency_ms", "bandwidth_mbps", "cost"),
    midhaul_edge_features=("feasible", "latency_ms", "bandwidth_mbps", "cost"),
)


@dataclass(frozen=True)
class TopologyConfig:
    """Problem dimensions for one mock orchestration instance."""

    num_rhs: int
    num_es: int
    num_rc: int
    horizon: int
    num_splits: int

    def validate(self) -> None:
        if self.num_rhs <= 1:
            raise ValueError("num_rhs must be greater than 1 to model handovers")
        if self.num_es <= 0 or self.num_rc <= 0:
            raise ValueError("num_es and num_rc must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.num_splits not in (4, 5):
            raise ValueError("phase 1 supports four legacy splits or five S4G/S4D splits")


@dataclass(frozen=True)
class SplitResourceDemand:
    """Split-dependent resource and link demand coefficients.

    Coefficients are multiplied by ``lambda_rt`` during constraint checks.
    Split IDs use zero-based implementation IDs. Four-split legacy configs use
    ``0=S1``, ``1=S2``, ``2=S3``, ``3=S4`` where ``S4`` is gateway-style
    centralized mode. Five-split configs use ``S4G`` and ``S4D`` explicitly.
    """

    es_compute_per_mbps: FloatArray
    rc_compute_per_mbps: FloatArray
    fronthaul_bw_per_mbps: FloatArray
    midhaul_bw_per_mbps: FloatArray
    max_fronthaul_latency_ms: FloatArray
    max_midhaul_latency_ms: FloatArray
    es_processing_latency_ms: FloatArray
    rc_processing_latency_ms: FloatArray
    max_end_to_end_latency_ms: FloatArray
    split_names: tuple[str, ...]

    def validate(self, num_splits: int) -> None:
        for name in (
            "es_compute_per_mbps",
            "rc_compute_per_mbps",
            "fronthaul_bw_per_mbps",
            "midhaul_bw_per_mbps",
            "max_fronthaul_latency_ms",
            "max_midhaul_latency_ms",
            "es_processing_latency_ms",
            "rc_processing_latency_ms",
            "max_end_to_end_latency_ms",
        ):
            value = getattr(self, name)
            if value.shape != (num_splits,):
                raise ValueError(f"{name} must have shape ({num_splits},)")
            if np.any(~np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f"{name} must contain finite nonnegative values")
        if np.any(self.max_end_to_end_latency_ms <= 0):
            raise ValueError("max_end_to_end_latency_ms must contain positive values")
        for split_id, split_name in enumerate(self.split_names):
            if split_name.upper() in {"S4G", "S4D"} and self.es_processing_latency_ms[split_id] != 0.0:
                raise ValueError(f"{split_name} must have zero ES processing latency")


@dataclass(frozen=True)
class CostWeights:
    """Weights used to convert constraint violations into scalar cost."""

    placement: float
    compute_overload: float
    link_violation: float
    pdcp_state_transfer: float
    rlc_state_transfer: float
    handover_sla: float
    rrc_anchor_crossing: float
    invalid_action: float


@dataclass(frozen=True)
class TopologyProfile:
    """One cached connectivity/link realization selectable at episode reset."""

    profile_id: str
    fronthaul_mask: BoolArray
    midhaul_mask: BoolArray
    fronthaul_latency_ms: FloatArray
    midhaul_latency_ms: FloatArray
    fronthaul_bandwidth_mbps: FloatArray
    midhaul_bandwidth_mbps: FloatArray
    fronthaul_cost: FloatArray
    midhaul_cost: FloatArray
    direct_rh_rc_mask: BoolArray
    direct_rh_rc_latency_ms: FloatArray
    direct_rh_rc_bandwidth_mbps: FloatArray
    direct_rh_rc_cost: FloatArray

    def validate(
        self,
        config: TopologyConfig,
        *,
        require_single_midhaul_route: bool = True,
        allow_large_transport_degree: bool = False,
        max_total_degree_per_rh: int = 2,
    ) -> None:
        shapes = {
            "fronthaul_mask": (config.num_rhs, config.num_es),
            "midhaul_mask": (config.num_es, config.num_rc),
            "fronthaul_latency_ms": (config.num_rhs, config.num_es),
            "midhaul_latency_ms": (config.num_es, config.num_rc),
            "fronthaul_bandwidth_mbps": (config.num_rhs, config.num_es),
            "midhaul_bandwidth_mbps": (config.num_es, config.num_rc),
            "fronthaul_cost": (config.num_rhs, config.num_es),
            "midhaul_cost": (config.num_es, config.num_rc),
            "direct_rh_rc_mask": (config.num_rhs, config.num_rc),
            "direct_rh_rc_latency_ms": (config.num_rhs, config.num_rc),
            "direct_rh_rc_bandwidth_mbps": (config.num_rhs, config.num_rc),
            "direct_rh_rc_cost": (config.num_rhs, config.num_rc),
        }
        for name, shape in shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"topology profile {self.profile_id}: {name} must have shape {shape}")
        if not np.all(
            self.fronthaul_mask.any(axis=1) | self.direct_rh_rc_mask.any(axis=1)
        ):
            raise ValueError(
                f"topology profile {self.profile_id}: every RH needs an outbound transport link"
            )
        midhaul_degree = self.midhaul_mask.sum(axis=1)
        if require_single_midhaul_route and not np.all(midhaul_degree == 1):
            raise ValueError(f"topology profile {self.profile_id}: every ES needs exactly one RC link")
        if not require_single_midhaul_route and not np.all(midhaul_degree >= 1):
            raise ValueError(f"topology profile {self.profile_id}: every ES needs at least one RC link")
        if max_total_degree_per_rh < 1:
            raise ValueError("max_total_degree_per_rh must be positive")
        if not allow_large_transport_degree and np.any(
            self.fronthaul_mask.sum(axis=1) + self.direct_rh_rc_mask.sum(axis=1)
            > max_total_degree_per_rh
        ):
            raise ValueError(
                f"topology profile {self.profile_id}: RH transport degree exceeds "
                f"{max_total_degree_per_rh}"
            )


@dataclass(frozen=True)
class MockEpisode:
    """Aggregated RH-level episode used by the environment and pipeline."""

    config: TopologyConfig
    lambda_rt: FloatArray
    num_ues_rt: FloatArray
    handover_count_t: FloatArray
    handover_matrix_t: FloatArray
    s_pdcp_rt: FloatArray
    s_rlc_rt: FloatArray
    rho_rt: FloatArray | None
    rh_x: FloatArray
    rh_y: FloatArray
    rh_region_id: IntArray
    fronthaul_mask: BoolArray
    midhaul_mask: BoolArray
    fronthaul_latency_ms: FloatArray
    midhaul_latency_ms: FloatArray
    fronthaul_bandwidth_mbps: FloatArray
    midhaul_bandwidth_mbps: FloatArray
    fronthaul_cost: FloatArray
    midhaul_cost: FloatArray
    direct_rh_rc_mask: BoolArray
    direct_rh_rc_latency_ms: FloatArray
    direct_rh_rc_bandwidth_mbps: FloatArray
    direct_rh_rc_cost: FloatArray
    es_compute_capacity: FloatArray
    rc_compute_capacity: FloatArray
    es_placement_cost: FloatArray
    rc_placement_cost: FloatArray
    split_demand: SplitResourceDemand
    cost_weights: CostWeights
    metadata: dict[str, Any] = field(default_factory=dict)
    observed_handover_count_t: FloatArray | None = None
    observed_handover_matrix_t: FloatArray | None = None
    realized_outgoing_ratio_rt: FloatArray | None = None
    topology_profiles: dict[str, TopologyProfile] = field(default_factory=dict)
    episode_topology_profile_ids: tuple[str, ...] = ()
    rh_service_deadline_ms: FloatArray | None = None

    def topology_profile_id_for_episode(self, episode_id: int) -> str | None:
        if not self.episode_topology_profile_ids:
            return None
        if episode_id < 0 or episode_id >= len(self.episode_topology_profile_ids):
            raise IndexError("episode_id is outside topology profile assignment")
        return self.episode_topology_profile_ids[episode_id]

    def activate_topology_profile(self, profile_id: str | None) -> None:
        """Copy one cached profile into the legacy active topology arrays."""

        if profile_id is None:
            return
        if self.metadata.get("active_topology_profile_id") == profile_id:
            return
        try:
            profile = self.topology_profiles[profile_id]
        except KeyError as exc:
            raise ValueError(f"unknown topology profile {profile_id!r}") from exc
        for name in (
            "fronthaul_mask",
            "midhaul_mask",
            "fronthaul_latency_ms",
            "midhaul_latency_ms",
            "fronthaul_bandwidth_mbps",
            "midhaul_bandwidth_mbps",
            "fronthaul_cost",
            "midhaul_cost",
            "direct_rh_rc_mask",
            "direct_rh_rc_latency_ms",
            "direct_rh_rc_bandwidth_mbps",
            "direct_rh_rc_cost",
        ):
            np.copyto(getattr(self, name), getattr(profile, name))
        self.metadata["active_topology_profile_id"] = profile_id

    def validate(self, *, allow_large_transport_degree: bool = False) -> None:
        c = self.config
        c.validate()
        r, d, rc, t, s = c.num_rhs, c.num_es, c.num_rc, c.horizon, c.num_splits
        expected_shapes = {
            "lambda_rt": (t, r),
            "num_ues_rt": (t, r),
            "handover_count_t": (t, r, r),
            "handover_matrix_t": (t, r, r),
            "s_pdcp_rt": (t, r),
            "s_rlc_rt": (t, r),
            "rh_x": (r,),
            "rh_y": (r,),
            "rh_region_id": (r,),
            "fronthaul_mask": (r, d),
            "midhaul_mask": (d, rc),
            "fronthaul_latency_ms": (r, d),
            "midhaul_latency_ms": (d, rc),
            "fronthaul_bandwidth_mbps": (r, d),
            "midhaul_bandwidth_mbps": (d, rc),
            "fronthaul_cost": (r, d),
            "midhaul_cost": (d, rc),
            "direct_rh_rc_mask": (r, rc),
            "direct_rh_rc_latency_ms": (r, rc),
            "direct_rh_rc_bandwidth_mbps": (r, rc),
            "direct_rh_rc_cost": (r, rc),
            "es_compute_capacity": (d,),
            "rc_compute_capacity": (rc,),
            "es_placement_cost": (d,),
            "rc_placement_cost": (rc,),
        }
        for name, shape in expected_shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if self.rho_rt is not None and self.rho_rt.shape != (t, r):
            raise ValueError(f"rho_rt must have shape {(t, r)}")
        if self.rh_service_deadline_ms is not None:
            if self.rh_service_deadline_ms.shape != (r,):
                raise ValueError(f"rh_service_deadline_ms must have shape {(r,)}")
            if np.any(~np.isfinite(self.rh_service_deadline_ms)) or np.any(self.rh_service_deadline_ms <= 0.0):
                raise ValueError("rh_service_deadline_ms must contain finite positive values")
        for name in ("observed_handover_count_t", "observed_handover_matrix_t"):
            value = getattr(self, name)
            if value is not None and value.shape != (t, r, r):
                raise ValueError(f"{name} must have shape {(t, r, r)}")
            if value is not None and not np.allclose(np.diagonal(value, axis1=1, axis2=2), 0.0):
                raise ValueError(f"{name} diagonal must be zero")
        if self.realized_outgoing_ratio_rt is not None and self.realized_outgoing_ratio_rt.shape != (t, r):
            raise ValueError(f"realized_outgoing_ratio_rt must have shape {(t, r)}")
        if self.rho_rt is not None and (np.any(self.rho_rt < 0) or np.any(self.rho_rt > 1)):
            raise ValueError("rho_rt must remain in [0, 1]")
        if not np.allclose(np.diagonal(self.handover_matrix_t, axis1=1, axis2=2), 0.0):
            raise ValueError("handover_matrix_t diagonal must be zero")
        if not np.allclose(np.diagonal(self.handover_count_t, axis1=1, axis2=2), 0.0):
            raise ValueError("handover_count_t diagonal must be zero")
        if np.any(self.handover_matrix_t < 0):
            raise ValueError("handover_matrix_t cannot contain negative values")
        if np.any(self.handover_count_t < 0):
            raise ValueError("handover_count_t cannot contain negative values")
        if not np.allclose(self.handover_count_t, np.round(self.handover_count_t), atol=1e-8):
            raise ValueError("handover_count_t must be integer-valued")
        if np.any(self.handover_matrix_t > 1.0 + 1e-8):
            raise ValueError("handover_matrix_t cannot contain values greater than 1")
        if np.any(self.handover_matrix_t.sum(axis=2) > 1.0 + 1e-8):
            raise ValueError("outgoing handover probabilities must sum to <= 1")
        if np.any(self.handover_count_t.sum(axis=2) > self.num_ues_rt + 1e-8):
            raise ValueError("outgoing handover counts must sum to <= source UE count")
        expected_prob = np.zeros_like(self.handover_matrix_t)
        np.divide(
            self.handover_count_t,
            self.num_ues_rt[:, :, None],
            out=expected_prob,
            where=self.num_ues_rt[:, :, None] > 0,
        )
        if not np.allclose(self.handover_matrix_t, expected_prob, atol=1e-8):
            raise ValueError("handover_matrix_t must equal handover_count_t / source UE count")
        self.split_demand.validate(s)
        if len(self.split_demand.split_names) != s:
            raise ValueError("split_names length must equal num_splits")
        if not allow_large_transport_degree:
            self._validate_rh_transport_budget()
        allowed_split_names = tuple(self.metadata.get("allowed_split_names", self.split_demand.split_names))
        if contains_direct_rc_split(allowed_split_names) and self.metadata.get("topology_profile") == "direct_only":
            if not np.all(self.direct_rh_rc_mask.any(axis=1)):
                raise ValueError("direct_only topology requires each RH to have at least one direct RH-to-RC link")
        self._validate_topology_intent()
        require_single_midhaul_route = not bool(
            self.metadata.get("allow_multiple_midhaul_routes_per_es", False)
        )
        max_total_degree_per_rh = int(
            self.metadata.get("rh_transport_budget_max_total_degree_per_rh", 2)
        )
        for profile in self.topology_profiles.values():
            profile.validate(
                c,
                require_single_midhaul_route=require_single_midhaul_route,
                allow_large_transport_degree=allow_large_transport_degree,
                max_total_degree_per_rh=max_total_degree_per_rh,
            )
        if self.episode_topology_profile_ids:
            unknown = set(self.episode_topology_profile_ids).difference(self.topology_profiles)
            if unknown:
                raise ValueError(f"episode topology assignment references unknown profiles: {sorted(unknown)}")

    def _validate_topology_intent(self) -> None:
        topology_profile = str(self.metadata.get("topology_profile", "legacy"))
        ensure_valid = bool(self.metadata.get("ensure_each_rh_has_valid_action", False))
        if topology_profile == "direct_only":
            if not np.all(self.direct_rh_rc_mask.any(axis=1)):
                raise ValueError("direct_only topology requires each RH to have at least one direct RH-to-RC link")
            return
        if not ensure_valid:
            return

        gateway_possible = self._gateway_route_possible_per_rh()
        direct_possible = self.direct_rh_rc_mask.any(axis=1)
        if topology_profile == "gateway_only":
            if not np.all(gateway_possible):
                missing = np.flatnonzero(~gateway_possible).tolist()
                raise ValueError(f"gateway_only topology has RHs without a gateway route: {missing}")
        elif topology_profile == "mixed":
            possible = gateway_possible | direct_possible
            if not np.all(possible):
                missing = np.flatnonzero(~possible).tolist()
                raise ValueError(f"mixed topology has RHs without gateway or direct reachability: {missing}")

    def _gateway_route_possible_per_rh(self) -> BoolArray:
        reachable_es = self.midhaul_mask.any(axis=1)
        return (self.fronthaul_mask & reachable_es[None, :]).any(axis=1)

    def _validate_rh_transport_budget(self) -> None:
        if not bool(self.metadata.get("rh_transport_budget_enabled", False)):
            return
        max_total = int(self.metadata.get("rh_transport_budget_max_total_degree_per_rh", 2))
        min_total = int(self.metadata.get("rh_transport_budget_min_total_degree_per_rh", 1))
        enforce_min = bool(self.metadata.get("ensure_each_rh_has_valid_action", True))
        gateway_degree = self.fronthaul_mask.sum(axis=1)
        direct_degree = self.direct_rh_rc_mask.sum(axis=1)
        total = gateway_degree + direct_degree
        max_violations = np.flatnonzero(total > max_total)
        min_violations = np.flatnonzero(total < min_total) if enforce_min else np.array([], dtype=np.int64)
        if max_violations.size > 0 or min_violations.size > 0:
            rh_id = int(max_violations[0] if max_violations.size > 0 else min_violations[0])
            raise ValueError(
                "RH transport budget violation: "
                f"rh_id={rh_id} rh_es_degree={int(gateway_degree[rh_id])} "
                f"direct_rh_rc_degree={int(direct_degree[rh_id])} "
                f"total_transport_degree={int(total[rh_id])} "
                f"min_total_degree_per_rh={min_total} max_total_degree_per_rh={max_total} "
                f"topology_profile={self.metadata.get('topology_profile', 'legacy')!r}"
            )


@dataclass(frozen=True)
class PlacementAction:
    """Decoded orchestration action for every RH."""

    split_id: IntArray
    es_id: IntArray
    rc_id: IntArray

    def validate(self, config: TopologyConfig) -> None:
        if self.split_id.shape != (config.num_rhs,):
            raise ValueError("split_id must have one entry per RH")
        if self.es_id.shape != (config.num_rhs,):
            raise ValueError("es_id must have one entry per RH")
        if self.rc_id.shape != (config.num_rhs,):
            raise ValueError("rc_id must have one entry per RH")
        if np.any((self.split_id < 0) | (self.split_id >= config.num_splits)):
            raise ValueError("split_id contains an out-of-range split")
        if np.any((self.es_id < 0) | (self.es_id >= config.num_es)):
            raise ValueError("es_id contains an out-of-range ES")
        if np.any((self.rc_id < 0) | (self.rc_id >= config.num_rc)):
            raise ValueError("rc_id contains an out-of-range RC")


@dataclass(frozen=True)
class ConstraintReport:
    """Cost and violation details for a decoded placement action."""

    feasible: bool
    total_cost: float
    placement_cost: float
    compute_overload_cost: float
    link_violation_cost: float
    handover_state_transfer_cost: float
    handover_sla_cost: float
    handover_control_cost: float
    reconfiguration_cost: float
    bandwidth_violation_cost: float
    latency_violation_cost: float
    violations: dict[str, float]

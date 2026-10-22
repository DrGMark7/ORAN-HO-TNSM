"""Feasibility checks and orchestration cost functions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from src.pipeline.schema import ConstraintReport, MockEpisode, PlacementAction
from src.model.split_semantics import (
    canonical_es_id_for_split,
    pdcp_anchor_type,
    rlc_anchor_type,
)
from src.model.graph_size_normalization import (
    REWARD_GRAPH_SCALE_MODES,
    reward_input_cost,
    reward_scaling_factor,
)
from src.model.serial_numba import (
    NUMBA_AVAILABLE,
    build_constraint_static,
    handover_args,
    handover_kernel,
    resource_args,
    resource_loads_kernel,
)
from src.utils.config import load_project_config


@dataclass(frozen=True)
class HandoverSLAConfig:
    """Simple latency parameters for layer-wise handover cost."""

    base_latency_ms: float
    budget_ms: float
    mbps_to_mb_per_ms: float

    @classmethod
    def from_project_config(cls, project_config: dict | None = None) -> "HandoverSLAConfig":
        config = project_config or load_project_config()
        sla = config["handover_sla"]
        return cls(
            base_latency_ms=float(sla["base_latency_ms"]),
            budget_ms=float(sla["budget_ms"]),
            mbps_to_mb_per_ms=float(sla["mbps_to_mb_per_ms"]),
        )


@dataclass(frozen=True)
class ConstraintPenaltyConfig:
    """Penalty multipliers for feasibility violation cost."""

    infeasible_path_penalty: float
    latency_excess_penalty: float
    min_effective_bandwidth_mbps: float
    latency_objective_mode: str

    @classmethod
    def from_project_config(cls, project_config: dict | None = None) -> "ConstraintPenaltyConfig":
        config = project_config or load_project_config()
        constraint = config["constraint"]
        resolved = cls(
            infeasible_path_penalty=float(constraint["infeasible_path_penalty"]),
            latency_excess_penalty=float(constraint["latency_excess_penalty"]),
            min_effective_bandwidth_mbps=float(constraint["min_effective_bandwidth_mbps"]),
            latency_objective_mode=str(constraint.get("latency_objective_mode", "end_to_end")),
        )
        if resolved.latency_objective_mode != "end_to_end":
            raise ValueError("constraint.latency_objective_mode must be 'end_to_end'")
        return resolved


@dataclass(frozen=True)
class RewardConfig:
    transform: str
    invalid_action_reward: float
    reward_graph_scale_mode: str = "legacy_total_cost"

    @classmethod
    def from_project_config(cls, project_config: dict | None = None) -> "RewardConfig":
        reward = (project_config or load_project_config()).get("reward", {})
        resolved = cls(
            transform=str(reward.get("transform", "inverse_log")),
            invalid_action_reward=float(reward.get("invalid_action_reward", -1.0)),
            reward_graph_scale_mode=str(
                reward.get("reward_graph_scale_mode", "legacy_total_cost")
            ),
        )
        if resolved.transform not in {
            "inverse_log",
            "negative_cost",
            "negative_cost_per_10_rh",
        }:
            raise ValueError(
                "reward.transform must be 'inverse_log', 'negative_cost', or "
                "'negative_cost_per_10_rh'"
            )
        if not math.isfinite(resolved.invalid_action_reward):
            raise ValueError("reward.invalid_action_reward must be finite")
        if resolved.reward_graph_scale_mode not in REWARD_GRAPH_SCALE_MODES:
            raise ValueError(
                "reward.reward_graph_scale_mode must be one of "
                f"{REWARD_GRAPH_SCALE_MODES}"
            )
        return resolved


@dataclass(frozen=True)
class HandoverControlConfig:
    enabled: bool

    @classmethod
    def from_project_config(cls, project_config: dict | None = None) -> "HandoverControlConfig":
        raw = (project_config or load_project_config()).get("handover_control", {})
        if not isinstance(raw.get("enabled", False), bool):
            raise ValueError("handover_control.enabled must be boolean")
        weight = float(raw.get("rrc_anchor_crossing_weight", 0.0))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("handover_control.rrc_anchor_crossing_weight must be finite and nonnegative")
        return cls(enabled=bool(raw.get("enabled", False)))


@dataclass(frozen=True)
class ReconfigurationConfig:
    enabled: bool
    split_change_weight: float
    es_change_weight: float
    rc_change_weight: float

    @classmethod
    def from_project_config(cls, project_config: dict | None = None) -> "ReconfigurationConfig":
        raw = (project_config or load_project_config()).get("reconfiguration_cost", {})
        if not isinstance(raw.get("enabled", False), bool):
            raise ValueError("reconfiguration_cost.enabled must be boolean")
        resolved = cls(
            enabled=bool(raw.get("enabled", False)),
            split_change_weight=float(raw.get("split_change_weight", 0.0)),
            es_change_weight=float(raw.get("es_change_weight", 0.0)),
            rc_change_weight=float(raw.get("rc_change_weight", 0.0)),
        )
        weights = (resolved.split_change_weight, resolved.es_change_weight, resolved.rc_change_weight)
        if not all(math.isfinite(weight) and weight >= 0.0 for weight in weights):
            raise ValueError("reconfiguration cost weights must be finite and nonnegative")
        return resolved


@dataclass(frozen=True)
class ResourceLoads:
    """Joint-action resource loads used by both constraints and observations."""

    es: np.ndarray
    rc: np.ndarray
    fronthaul: np.ndarray
    midhaul: np.ndarray
    direct: np.ndarray
    es_utilization: np.ndarray | None = None
    rc_utilization: np.ndarray | None = None


def transform_reward(
    total_cost: float,
    config: RewardConfig,
    num_rhs: int | None = None,
) -> float:
    """Apply the configured reward transform without changing raw cost."""

    if total_cost < -1e-9:
        raise ValueError("total_cost must be nonnegative")
    cost = max(float(total_cost), 0.0)
    if config.transform == "negative_cost":
        return -cost
    if config.transform == "negative_cost_per_10_rh":
        if num_rhs is None or num_rhs <= 0:
            raise ValueError("num_rhs must be positive for negative_cost_per_10_rh")
        return -cost / (10.0 * float(num_rhs))
    return 1.0 / (1.0 + math.log1p(cost))


class OrchestrationConstraintEvaluator:
    """Evaluate connectivity validity and orchestration cost components."""

    def __init__(
        self,
        episode: MockEpisode,
        sla: HandoverSLAConfig | None = None,
        penalties: ConstraintPenaltyConfig | None = None,
        project_config: dict[str, Any] | None = None,
    ) -> None:
        episode.validate()
        self.episode = episode
        self.project_config = project_config or load_project_config()
        self.sla = sla or HandoverSLAConfig.from_project_config(self.project_config)
        self.penalties = penalties or ConstraintPenaltyConfig.from_project_config(self.project_config)
        self.reward_config = RewardConfig.from_project_config(self.project_config)
        self.handover_control = HandoverControlConfig.from_project_config(self.project_config)
        self.reconfiguration = ReconfigurationConfig.from_project_config(self.project_config)
        try:
            self.formulation = dict(self.project_config["formulation"])
        except KeyError as exc:
            raise ValueError("active evaluator requires a resolved problem formulation") from exc
        self.formulation_version = str(self.formulation["version"])
        self._last_resource_loads: ResourceLoads | None = None
        self._numba_static = None

    def enable_serial_numba(self) -> bool:
        """Enable optional kernels for a trainer that selected this backend."""

        if not NUMBA_AVAILABLE:
            return False
        try:
            self._numba_static = build_constraint_static(self)
        except Exception:
            self._numba_static = None
        return self._numba_static is not None

    @property
    def last_resource_loads(self) -> ResourceLoads | None:
        """Loads calculated for the most recent successful evaluator call."""

        return self._last_resource_loads

    def evaluate(
        self,
        time_index: int,
        action: PlacementAction,
        previous_action: PlacementAction | None = None,
    ) -> ConstraintReport:
        if time_index < 0 or time_index >= self.episode.config.horizon:
            raise IndexError("time_index is outside the episode horizon")
        violations: dict[str, float] = {}
        loads = self.resource_loads(time_index, action)
        self._last_resource_loads = loads
        placement_cost = self._placement_cost(action)
        compute_overload_cost = self._compute_overload_cost(loads, violations)
        bandwidth_violation_cost = self._bandwidth_violation_cost(loads, violations)
        latency_violation_cost = self._e2e_latency_violation_cost(action, loads, violations)
        link_violation_cost = bandwidth_violation_cost + latency_violation_cost
        handover_cost, handover_sla_cost = self._handover_cost(time_index, action, violations)
        handover_control_cost = self._handover_control_cost(time_index, action)
        reconfiguration_cost = self._reconfiguration_cost(action, previous_action)

        total_cost = (
            placement_cost
            + compute_overload_cost
            + link_violation_cost
            + handover_cost
            + handover_sla_cost
            + handover_control_cost
            + reconfiguration_cost
        )
        feasible_keys = (
            "fronthaul_bw_shortfall_mbps",
            "midhaul_bw_shortfall_mbps",
            "direct_rh_rc_bw_shortfall_mbps",
            "end_to_end_latency_excess_ms",
            "es_compute_overload",
            "rc_compute_overload",
            "diagonal_handover_prob",
            "diagonal_handover_count",
        )
        feasible = all(float(violations.get(key, 0.0)) <= 0.0 for key in feasible_keys)
        return ConstraintReport(
            feasible=feasible,
            total_cost=float(total_cost),
            placement_cost=float(placement_cost),
            compute_overload_cost=float(compute_overload_cost),
            link_violation_cost=float(link_violation_cost),
            handover_state_transfer_cost=float(handover_cost),
            handover_sla_cost=float(handover_sla_cost),
            handover_control_cost=float(handover_control_cost),
            reconfiguration_cost=float(reconfiguration_cost),
            bandwidth_violation_cost=float(bandwidth_violation_cost),
            latency_violation_cost=float(latency_violation_cost),
            violations=violations,
        )

    def resource_loads(self, time_index: int, action: PlacementAction) -> ResourceLoads:
        """Use exact compiled loops when available, with a literal fallback."""

        if self._numba_static is None or resource_loads_kernel is None:
            return self._resource_loads_python(time_index, action)
        if time_index < 0 or time_index >= self.episode.config.horizon:
            raise IndexError("time_index is outside the episode horizon")
        action.validate(self.episode.config)
        self._validate_connectivity(action)
        try:
            es, rc, fronthaul, midhaul, direct = resource_loads_kernel(
                *resource_args(self, time_index, action, self._numba_static)
            )
        except Exception:
            self._numba_static = None
            return self._resource_loads_python(time_index, action)
        return ResourceLoads(
            es=es,
            rc=rc,
            fronthaul=fronthaul,
            midhaul=midhaul,
            direct=direct,
            es_utilization=es / self.episode.es_compute_capacity,
            rc_utilization=rc / self.episode.rc_compute_capacity,
        )

    def _resource_loads_python(
        self, time_index: int, action: PlacementAction
    ) -> ResourceLoads:
        """Compute evaluator-consistent loads for a complete applied joint action."""

        if time_index < 0 or time_index >= self.episode.config.horizon:
            raise IndexError("time_index is outside the episode horizon")
        action.validate(self.episode.config)
        self._validate_connectivity(action)
        ep = self.episode
        traffic = ep.lambda_rt[time_index]
        intensity = float(self.formulation["compute"]["total_intensity_per_mbps"])
        demand = self.formulation["transport"]["demand"]
        splits = self.formulation["splits"]
        es = np.zeros(ep.config.num_es, dtype=np.float64)
        rc = np.zeros(ep.config.num_rc, dtype=np.float64)
        fronthaul = np.zeros((ep.config.num_rhs, ep.config.num_es), dtype=np.float64)
        midhaul = np.zeros((ep.config.num_es, ep.config.num_rc), dtype=np.float64)
        direct = np.zeros((ep.config.num_rhs, ep.config.num_rc), dtype=np.float64)
        for r in range(ep.config.num_rhs):
            split = int(action.split_id[r])
            es_id = int(action.es_id[r])
            rc_id = int(action.rc_id[r])
            record = splits[ep.split_demand.split_names[split]]
            compute = record["compute"]
            workload = float(traffic[r]) * intensity
            es[es_id] += workload * float(compute["es_fraction"])
            rc[rc_id] += workload * float(compute["rc_fraction"])
            transport = record["transport"]
            if transport["direct"] is not None:
                direct[r, rc_id] += self._transport_demand_mbps(str(transport["direct"]), traffic[r], demand)
                continue
            fronthaul[r, es_id] += self._transport_demand_mbps(str(transport["fronthaul"]), traffic[r], demand)
            midhaul[es_id, rc_id] += self._transport_demand_mbps(str(transport["midhaul"]), traffic[r], demand)
        return ResourceLoads(
            es=es, rc=rc, fronthaul=fronthaul, midhaul=midhaul, direct=direct,
            es_utilization=es / ep.es_compute_capacity,
            rc_utilization=rc / ep.rc_compute_capacity,
        )

    @staticmethod
    def _transport_demand_mbps(kind: str, traffic_mbps: float, demand: dict[str, Any]) -> float:
        if kind in {"o7_2x", "o8"}:
            return float(demand[f"{kind}_mbps"])
        rule = demand[kind]
        return float(traffic_mbps) * float(rule["multiplier"]) + float(rule["offset_mbps"])

    def reward(
        self,
        time_index: int,
        action: PlacementAction,
        previous_action: PlacementAction | None = None,
    ) -> float:
        return self.reward_from_cost(self.evaluate(time_index, action, previous_action).total_cost)

    def reward_from_cost(self, total_cost: float) -> float:
        return transform_reward(
            self.reward_input_from_cost(total_cost),
            self.reward_config,
            self.episode.config.num_rhs,
        )

    def reward_input_from_cost(self, total_cost: float) -> float:
        """Return the transform input while retaining raw ``J_t`` in diagnostics."""

        return reward_input_cost(
            total_cost,
            self.reward_config.reward_graph_scale_mode,
            self.episode.config.num_rhs,
        )

    def formulation_metadata(self) -> dict[str, Any]:
        metadata = {
            "constraint_formulation_version": self.formulation_version,
            "reward_transform": self.reward_config.transform,
            "invalid_action_reward": self.reward_config.invalid_action_reward,
            "reward_graph_scale_mode": self.reward_config.reward_graph_scale_mode,
            "reward_scaling_factor": reward_scaling_factor(
                self.reward_config.reward_graph_scale_mode,
                self.episode.config.num_rhs,
            ),
            "latency_objective_mode": self.penalties.latency_objective_mode,
            "latency_config_source": self.episode.metadata.get("latency_config_source", "unknown"),
            "handover_control_configured_enabled": self.handover_control.enabled,
            "handover_control_enabled": (
                self.handover_control.enabled and self.episode.cost_weights.rrc_anchor_crossing > 0.0
            ),
            "rrc_anchor_crossing_weight": self.episode.cost_weights.rrc_anchor_crossing,
            "reconfiguration_cost": vars(self.reconfiguration),
            "es_processing_latency_ms": self.episode.split_demand.es_processing_latency_ms.tolist(),
            "rc_processing_latency_ms": self.episode.split_demand.rc_processing_latency_ms.tolist(),
            "max_end_to_end_latency_ms": self.episode.split_demand.max_end_to_end_latency_ms.tolist(),
        }
        metadata["rh_service_deadline_ms"] = (
            self.episode.rh_service_deadline_ms.tolist()
            if self.episode.rh_service_deadline_ms is not None else None
        )
        metadata["resource_accounting"] = "joint_action_once_per_evaluate"
        return metadata

    def _validate_connectivity(self, action: PlacementAction) -> None:
        ep = self.episode
        for r in range(ep.config.num_rhs):
            split, es, rc = int(action.split_id[r]), int(action.es_id[r]), int(action.rc_id[r])
            if self._uses_direct_transport(split):
                canonical_es = canonical_es_id_for_split(split, ep.split_demand.split_names)
                if es != canonical_es or not bool(ep.direct_rh_rc_mask[r, rc]):
                    raise ValueError(f"connectivity-invalid S4D action for RH {r}")
            elif not bool(ep.fronthaul_mask[r, es]) or not bool(ep.midhaul_mask[es, rc]):
                raise ValueError(f"connectivity-invalid gateway action for RH {r}")

    def _placement_cost(self, action: PlacementAction) -> float:
        ep = self.episode
        weights = ep.cost_weights
        link_cost = 0.0
        es_cost = 0.0
        for rh_id in range(ep.config.num_rhs):
            split = int(action.split_id[rh_id])
            es_id = action.es_id[rh_id]
            rc_id = action.rc_id[rh_id]
            if self._uses_direct_transport(split):
                link_cost += ep.direct_rh_rc_cost[rh_id, rc_id]
            else:
                es_cost += ep.es_placement_cost[es_id]
                link_cost += ep.fronthaul_cost[rh_id, es_id] + ep.midhaul_cost[es_id, rc_id]
        return weights.placement * float(es_cost + ep.rc_placement_cost[action.rc_id].sum() + link_cost)

    # J_compute: ES/RC compute overload penalty.
    def _compute_overload_cost(self, loads: ResourceLoads, violations: dict[str, float]) -> float:
        ep = self.episode
        es_load, rc_load = loads.es, loads.rc

        es_overload = np.maximum(es_load - ep.es_compute_capacity, 0.0)
        rc_overload = np.maximum(rc_load - ep.rc_compute_capacity, 0.0)
        violations["es_compute_overload"] = float(es_overload.sum())
        violations["rc_compute_overload"] = float(rc_overload.sum())
        violations["es_compute_load"] = float(es_load.sum())
        violations["rc_compute_load"] = float(rc_load.sum())
        normalized_es_overload = np.divide(
            es_overload, ep.es_compute_capacity, out=np.zeros_like(es_overload),
            where=ep.es_compute_capacity > 0.0,
        )
        normalized_rc_overload = np.divide(
            rc_overload, ep.rc_compute_capacity, out=np.zeros_like(rc_overload),
            where=ep.rc_compute_capacity > 0.0,
        )
        violations["es_compute_overload_normalized"] = float(normalized_es_overload.sum())
        violations["rc_compute_overload_normalized"] = float(normalized_rc_overload.sum())
        self._append_utilization_diagnostics("es", loads.es_utilization, violations)
        self._append_utilization_diagnostics("rc", loads.rc_utilization, violations)
        return float(self.formulation["objective"]["weights"]["compute_violation"]) * float(
            normalized_es_overload.sum() + normalized_rc_overload.sum()
        )

    # J_BW: positive bandwidth capacity excess; connectivity is validated separately.
    def _bandwidth_violation_cost(self, loads: ResourceLoads, violations: dict[str, float]) -> float:
        ep = self.episode
        fronthaul_bw_shortfall = float(
            np.maximum(loads.fronthaul - ep.fronthaul_bandwidth_mbps, 0.0).sum()
        )
        direct_bw_shortfall = float(
            np.maximum(loads.direct - ep.direct_rh_rc_bandwidth_mbps, 0.0).sum()
        )

        # Fronthaul is modeled as one selected RH->ES path per RH. Midhaul is
        # shared by all RHs selecting the same ES->RC pair, so aggregate first.
        midhaul_bw_shortfall = float(
            np.maximum(loads.midhaul - ep.midhaul_bandwidth_mbps, 0.0).sum()
        )
        violations["infeasible_fronthaul_paths"] = 0.0
        violations["infeasible_midhaul_paths"] = 0.0
        violations["infeasible_direct_rh_rc_paths"] = 0.0
        violations["fronthaul_bw_shortfall_mbps"] = float(fronthaul_bw_shortfall)
        violations["midhaul_bw_shortfall_mbps"] = midhaul_bw_shortfall
        violations["direct_rh_rc_bw_shortfall_mbps"] = float(direct_bw_shortfall)
        normalized_fronthaul = np.divide(
            np.maximum(loads.fronthaul - ep.fronthaul_bandwidth_mbps, 0.0),
            ep.fronthaul_bandwidth_mbps, out=np.zeros_like(loads.fronthaul),
            where=ep.fronthaul_bandwidth_mbps > 0.0,
        ).sum()
        normalized_midhaul = np.divide(
            np.maximum(loads.midhaul - ep.midhaul_bandwidth_mbps, 0.0),
            ep.midhaul_bandwidth_mbps, out=np.zeros_like(loads.midhaul),
            where=ep.midhaul_bandwidth_mbps > 0.0,
        ).sum()
        normalized_direct = np.divide(
            np.maximum(loads.direct - ep.direct_rh_rc_bandwidth_mbps, 0.0),
            ep.direct_rh_rc_bandwidth_mbps, out=np.zeros_like(loads.direct),
            where=ep.direct_rh_rc_bandwidth_mbps > 0.0,
        ).sum()
        violations["fronthaul_bw_shortfall_normalized"] = float(normalized_fronthaul)
        violations["midhaul_bw_shortfall_normalized"] = float(normalized_midhaul)
        violations["direct_rh_rc_bw_shortfall_normalized"] = float(normalized_direct)
        return float(self.formulation["objective"]["weights"]["bandwidth_violation"]) * float(
            normalized_fronthaul + normalized_midhaul + normalized_direct
        )

    # J_latency: split-specific end-to-end transport plus explicit processing delay.
    def _e2e_latency_violation_cost(
        self, action: PlacementAction, loads: ResourceLoads, violations: dict[str, float]
    ) -> float:
        ep = self.episode
        if ep.rh_service_deadline_ms is None:
            raise ValueError("active E2E latency requires rh_service_deadline_ms from rh_nodes.csv")
        tau = float(self.formulation["compute"]["base_service_time_ms"])
        cap = float(self.formulation["compute"]["utilization_cap"])
        es_delay = self.processing_delay_ms(loads.es_utilization, tau, cap)
        rc_delay = self.processing_delay_ms(loads.rc_utilization, tau, cap)
        transport_total = 0.0
        processing_total = 0.0
        e2e_total = 0.0
        excess_total = 0.0
        normalized_excess_total = 0.0
        direct_excess = 0.0
        deadline_total = 0.0
        selected_es_delays: list[float] = []
        selected_rc_delays: list[float] = []
        for r in range(ep.config.num_rhs):
            split, es, rc = int(action.split_id[r]), int(action.es_id[r]), int(action.rc_id[r])
            record = self.formulation["splits"][ep.split_demand.split_names[split]]
            fractions = record["compute"]
            if record["transport"]["direct"] is not None:
                transport = float(ep.direct_rh_rc_latency_ms[r, rc])
                processing = float(fractions["rc_fraction"]) * float(rc_delay[rc])
            else:
                transport = float(ep.fronthaul_latency_ms[r, es] + ep.midhaul_latency_ms[es, rc])
                processing = float(
                    float(fractions["es_fraction"]) * float(es_delay[es])
                    + float(fractions["rc_fraction"]) * float(rc_delay[rc])
                )
            selected_es_delays.append(float(fractions["es_fraction"]) * float(es_delay[es]))
            selected_rc_delays.append(float(fractions["rc_fraction"]) * float(rc_delay[rc]))
            e2e = transport + processing
            excess = max(e2e - float(ep.rh_service_deadline_ms[r]), 0.0)
            transport_total += transport
            processing_total += processing
            e2e_total += e2e
            excess_total += excess
            normalized_excess_total += excess / float(ep.rh_service_deadline_ms[r])
            deadline_total += float(ep.rh_service_deadline_ms[r])
            if record["transport"]["direct"] is not None:
                direct_excess += excess
        violations["transport_latency_ms"] = transport_total
        violations["processing_latency_ms"] = processing_total
        violations["es_processing_delay_ms"] = float(np.mean(selected_es_delays))
        violations["es_processing_delay_ms_min"] = float(np.min(selected_es_delays))
        violations["es_processing_delay_ms_max"] = float(np.max(selected_es_delays))
        violations["rc_processing_delay_ms"] = float(np.mean(selected_rc_delays))
        violations["rc_processing_delay_ms_min"] = float(np.min(selected_rc_delays))
        violations["rc_processing_delay_ms_max"] = float(np.max(selected_rc_delays))
        violations["end_to_end_latency_ms"] = e2e_total
        violations["end_to_end_latency_excess_ms"] = excess_total
        violations["end_to_end_latency_excess_normalized"] = float(normalized_excess_total)
        violations["latency_deadline_ms"] = deadline_total
        violations["fronthaul_latency_excess_ms"] = 0.0
        violations["midhaul_latency_excess_ms"] = 0.0
        violations["direct_rh_rc_latency_excess_ms"] = direct_excess
        return float(self.formulation["objective"]["weights"]["latency_violation"]) * normalized_excess_total

    @staticmethod
    def _append_utilization_diagnostics(
        prefix: str, utilization: np.ndarray | None, violations: dict[str, float]
    ) -> None:
        if utilization is None:
            raise ValueError("resource loads must include utilization")
        violations[f"{prefix}_utilization"] = float(utilization.mean())
        violations[f"{prefix}_utilization_min"] = float(utilization.min())
        violations[f"{prefix}_utilization_mean"] = float(utilization.mean())
        violations[f"{prefix}_utilization_max"] = float(utilization.max())

    @staticmethod
    def processing_delay_ms(
        utilization: np.ndarray | float, tau_ms: float, utilization_cap: float
    ) -> np.ndarray:
        """M/M/1 delay, capping only the denominator utilization."""

        return float(tau_ms) / (1.0 - np.minimum(np.asarray(utilization, dtype=np.float64), utilization_cap))

    # J_HO_state and J_HO_SLA: layer-anchor state transfer and mobility SLA costs.
    def _handover_cost(
        self, time_index: int, action: PlacementAction, violations: dict[str, float]
    ) -> tuple[float, float]:
        if self._numba_static is None or handover_kernel is None:
            return self._handover_cost_python(time_index, action, violations)
        try:
            state, sla, diagonal_prob, diagonal_count = handover_kernel(
                *handover_args(self, time_index, action, self._numba_static)
            )
        except Exception:
            self._numba_static = None
            return self._handover_cost_python(time_index, action, violations)
        violations["diagonal_handover_prob"] = float(diagonal_prob)
        violations["diagonal_handover_count"] = float(diagonal_count)
        return float(state), float(sla)

    def _handover_cost_python(
        self, time_index: int, action: PlacementAction, violations: dict[str, float]
    ) -> tuple[float, float]:
        ep = self.episode
        handover_prob = ep.handover_matrix_t[time_index]
        handover_count = ep.handover_count_t[time_index]
        pdcp = ep.s_pdcp_rt[time_index]
        rlc = ep.s_rlc_rt[time_index]

        state_cost = 0.0
        sla_cost = 0.0
        diagonal_handover_prob = 0.0
        diagonal_handover_count = 0.0

        for r in range(ep.config.num_rhs):
            for rp in range(ep.config.num_rhs):
                if r == rp:
                    diagonal_handover_prob += handover_prob[r, rp]
                    diagonal_handover_count += handover_count[r, rp]
                    continue
                expected_ues = handover_count[r, rp]
                if expected_ues <= 0.0:
                    continue

                pdcp_crosses = self._pdcp_anchor(action, r) != self._pdcp_anchor(action, rp)
                rlc_crosses = self._rlc_anchor(action, r) != self._rlc_anchor(action, rp)

                state_cost += expected_ues * (
                    ep.cost_weights.pdcp_state_transfer * pdcp[r] * float(pdcp_crosses)
                    + ep.cost_weights.rlc_state_transfer * rlc[r] * float(rlc_crosses)
                )

                split_delay_ms = self._split_handover_delay_ms(
                    time_index=time_index,
                    action=action,
                    source_rh=r,
                    target_rh=rp,
                    pdcp_crosses=pdcp_crosses,
                    rlc_crosses=rlc_crosses,
                )
                latency_excess = max(self.sla.base_latency_ms + split_delay_ms - self.sla.budget_ms, 0.0)
                sla_cost += ep.cost_weights.handover_sla * expected_ues * latency_excess

        violations["diagonal_handover_prob"] = float(diagonal_handover_prob)
        violations["diagonal_handover_count"] = float(diagonal_handover_count)
        return float(state_cost), float(sla_cost)

    # J_HO_ctrl: optional RC/RRC control-anchor crossing cost.
    def _handover_control_cost(self, time_index: int, action: PlacementAction) -> float:
        if not self.handover_control.enabled or self.episode.cost_weights.rrc_anchor_crossing <= 0.0:
            return 0.0
        counts = self.episode.handover_count_t[time_index]
        crossing_count = 0.0
        for r in range(self.episode.config.num_rhs):
            for rp in range(self.episode.config.num_rhs):
                if r != rp and int(action.rc_id[r]) != int(action.rc_id[rp]):
                    crossing_count += float(counts[r, rp])
        return self.episode.cost_weights.rrc_anchor_crossing * crossing_count

    # J_reconf: split/vDU/vCU changes relative to the previous applied action.
    def _reconfiguration_cost(
        self, action: PlacementAction, previous_action: PlacementAction | None
    ) -> float:
        if not self.reconfiguration.enabled or previous_action is None:
            return 0.0
        cost = 0.0
        for r in range(self.episode.config.num_rhs):
            previous_split = int(previous_action.split_id[r])
            current_split = int(action.split_id[r])
            cost += self.reconfiguration.split_change_weight * float(previous_split != current_split)
            both_gateway = not self._uses_direct_transport(previous_split) and not self._uses_direct_transport(current_split)
            if both_gateway:
                cost += self.reconfiguration.es_change_weight * float(
                    int(previous_action.es_id[r]) != int(action.es_id[r])
                )
            cost += self.reconfiguration.rc_change_weight * float(
                int(previous_action.rc_id[r]) != int(action.rc_id[r])
            )
        return cost

    def _rlc_anchor(self, action: PlacementAction, rh_id: int) -> tuple[str, int]:
        anchor = rlc_anchor_type(int(action.split_id[rh_id]), self.episode.split_demand.split_names)
        if anchor == "es":
            return "es", int(action.es_id[rh_id])
        return "rc", int(action.rc_id[rh_id])

    def _uses_direct_transport(self, split_id: int) -> bool:
        split_name = self.episode.split_demand.split_names[split_id]
        return self.formulation["splits"][split_name]["transport"]["direct"] is not None

    def _pdcp_anchor(self, action: PlacementAction, rh_id: int) -> tuple[str, int]:
        anchor = pdcp_anchor_type(int(action.split_id[rh_id]), self.episode.split_demand.split_names)
        if anchor == "es":
            return "es", int(action.es_id[rh_id])
        return "rc", int(action.rc_id[rh_id])

    def _split_handover_delay_ms(
        self,
        time_index: int,
        action: PlacementAction,
        source_rh: int,
        target_rh: int,
        pdcp_crosses: bool,
        rlc_crosses: bool,
    ) -> float:
        ep = self.episode
        source_path_bw = self._selected_path_bandwidth_mbps(action, source_rh)
        target_path_bw = self._selected_path_bandwidth_mbps(action, target_rh)
        effective_mbps = max(
            min(source_path_bw, target_path_bw),
            self.penalties.min_effective_bandwidth_mbps,
        )
        effective_mb_per_ms = effective_mbps * self.sla.mbps_to_mb_per_ms

        transfer_mb = 0.0
        if pdcp_crosses:
            transfer_mb += ep.s_pdcp_rt[time_index, source_rh]
        if rlc_crosses:
            transfer_mb += ep.s_rlc_rt[time_index, source_rh]
        return transfer_mb / effective_mb_per_ms

    def _selected_path_bandwidth_mbps(self, action: PlacementAction, rh_id: int) -> float:
        ep = self.episode
        split = int(action.split_id[rh_id])
        es = int(action.es_id[rh_id])
        rc = int(action.rc_id[rh_id])
        if self._uses_direct_transport(split):
            return float(ep.direct_rh_rc_bandwidth_mbps[rh_id, rc])
        return float(min(ep.fronthaul_bandwidth_mbps[rh_id, es], ep.midhaul_bandwidth_mbps[es, rc]))

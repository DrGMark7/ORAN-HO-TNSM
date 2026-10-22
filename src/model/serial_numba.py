"""Optional exact serial Numba kernels for CPU-side orchestration arithmetic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised in the normal no-Numba environment
    njit = None

from src.model.split_semantics import (
    pdcp_anchor_type,
    rlc_anchor_type,
    uses_direct_rh_rc,
    uses_es_compute,
    uses_midhaul,
)


NUMBA_AVAILABLE = njit is not None


def _causal_kernel(
    split_id: np.ndarray,
    es_id: np.ndarray,
    rc_id: np.ndarray,
    es_load: np.ndarray,
    rc_load: np.ndarray,
    fronthaul_load: np.ndarray,
    midhaul_load: np.ndarray,
    direct_load: np.ndarray,
    es_capacity: np.ndarray,
    rc_capacity: np.ndarray,
    fronthaul_capacity: np.ndarray,
    midhaul_capacity: np.ndarray,
    direct_capacity: np.ndarray,
    direct_split: np.ndarray,
    es_compute_split: np.ndarray,
    midhaul_split: np.ndarray,
    feasible_es: np.ndarray,
    feasible_rc: np.ndarray,
    feasible_mh_es: np.ndarray,
    feasible_mh_rc: np.ndarray,
) -> np.ndarray:
    num_rh = split_id.shape[0]
    num_splits = direct_split.shape[0]
    num_es = es_capacity.shape[0]
    num_rc = rc_capacity.shape[0]
    result = np.zeros((num_rh, num_splits + num_es + num_rc + 24), dtype=np.float32)
    es_u = es_load / es_capacity
    rc_u = rc_load / rc_capacity
    for r in range(num_rh):
        split = int(split_id[r])
        es = int(es_id[r])
        rc = int(rc_id[r])
        result[r, split] = np.float32(1.0)
        result[r, num_splits + es] = np.float32(1.0)
        result[r, num_splits + num_es + rc] = np.float32(1.0)
        offset = num_splits + num_es + num_rc
        direct = bool(direct_split[split])
        result[r, offset] = np.float32(not direct)
        result[r, offset + 1] = np.float32(direct)
        result[r, offset + 2] = np.float32(1.0)
        offset += 3
        if direct:
            utilization = float(direct_load[r, rc]) / float(direct_capacity[r, rc])
            result[r, offset] = np.float32(utilization)
            result[r, offset + 1] = np.float32(1.0 - utilization)
            result[r, offset + 2] = np.float32(1.0)
            offset += 9
        else:
            utilization = float(fronthaul_load[r, es]) / float(fronthaul_capacity[r, es])
            result[r, offset] = np.float32(utilization)
            result[r, offset + 1] = np.float32(1.0 - utilization)
            result[r, offset + 2] = np.float32(1.0)
            offset += 3
            if es_compute_split[split]:
                utilization = float(es_load[es]) / float(es_capacity[es])
                result[r, offset] = np.float32(utilization)
                result[r, offset + 1] = np.float32(1.0 - utilization)
                result[r, offset + 2] = np.float32(1.0)
            offset += 3
            if midhaul_split[split]:
                utilization = float(midhaul_load[es, rc]) / float(midhaul_capacity[es, rc])
                result[r, offset] = np.float32(utilization)
                result[r, offset + 1] = np.float32(1.0 - utilization)
                result[r, offset + 2] = np.float32(1.0)
            offset += 3
        utilization = float(rc_load[rc]) / float(rc_capacity[rc])
        result[r, offset] = np.float32(utilization)
        result[r, offset + 1] = np.float32(1.0 - utilization)
        result[r, offset + 2] = np.float32(1.0)
        offset += 3
        for group in range(3):
            minimum = np.inf
            maximum = -np.inf
            total = 0.0
            count = 0
            if group == 0:
                for index in range(feasible_es.shape[1]):
                    candidate = int(feasible_es[r, index])
                    if candidate < 0:
                        continue
                    value = float(es_u[candidate])
                    minimum = min(minimum, value)
                    maximum = max(maximum, value)
                    total += value
                    count += 1
            elif group == 1:
                for index in range(feasible_rc.shape[1]):
                    candidate = int(feasible_rc[r, index])
                    if candidate < 0:
                        continue
                    value = float(rc_u[candidate])
                    minimum = min(minimum, value)
                    maximum = max(maximum, value)
                    total += value
                    count += 1
            else:
                for index in range(feasible_mh_es.shape[1]):
                    mh_es = int(feasible_mh_es[r, index])
                    mh_rc = int(feasible_mh_rc[r, index])
                    if mh_es < 0:
                        continue
                    bandwidth = float(midhaul_capacity[mh_es, mh_rc])
                    value = (
                        0.0
                        if bandwidth <= 0.0
                        else float(midhaul_load[mh_es, mh_rc]) / bandwidth
                    )
                    minimum = min(minimum, value)
                    maximum = max(maximum, value)
                    total += value
                    count += 1
            if count:
                result[r, offset] = np.float32(minimum)
                result[r, offset + 1] = np.float32(total / count)
                result[r, offset + 2] = np.float32(maximum)
            offset += 3
    return result


def _causal_v2_kernel(
    split_id: np.ndarray,
    es_id: np.ndarray,
    rc_id: np.ndarray,
    es_load: np.ndarray,
    rc_load: np.ndarray,
    fronthaul_load: np.ndarray,
    midhaul_load: np.ndarray,
    direct_load: np.ndarray,
    es_capacity: np.ndarray,
    rc_capacity: np.ndarray,
    fronthaul_capacity: np.ndarray,
    midhaul_capacity: np.ndarray,
    direct_capacity: np.ndarray,
    direct_split: np.ndarray,
    es_compute_split: np.ndarray,
    midhaul_split: np.ndarray,
    feasible_es: np.ndarray,
    feasible_rc: np.ndarray,
    feasible_mh_es: np.ndarray,
    feasible_mh_rc: np.ndarray,
) -> np.ndarray:
    """Pack the compact identity-preserving v2 context."""

    num_rh = split_id.shape[0]
    num_splits = direct_split.shape[0]
    num_es = es_capacity.shape[0]
    num_rc = rc_capacity.shape[0]
    result = np.zeros((num_rh, num_splits + num_es + num_rc + 16), dtype=np.float32)
    es_u = es_load / es_capacity
    rc_u = rc_load / rc_capacity
    for r in range(num_rh):
        split = int(split_id[r])
        es = int(es_id[r])
        rc = int(rc_id[r])
        result[r, split] = np.float32(1.0)
        result[r, num_splits + es] = np.float32(1.0)
        result[r, num_splits + num_es + rc] = np.float32(1.0)
        offset = num_splits + num_es + num_rc
        direct = bool(direct_split[split])
        result[r, offset] = np.float32(not direct)
        result[r, offset + 1] = np.float32(direct)
        result[r, offset + 2] = np.float32(1.0)

        # Selected remaining: ES, RC, transport (FH/direct), midhaul.
        if es_compute_split[split]:
            result[r, offset + 3] = np.float32(
                1.0 - float(es_load[es]) / float(es_capacity[es])
            )
        result[r, offset + 4] = np.float32(
            1.0 - float(rc_load[rc]) / float(rc_capacity[rc])
        )
        if direct:
            result[r, offset + 5] = np.float32(
                1.0 - float(direct_load[r, rc]) / float(direct_capacity[r, rc])
            )
        else:
            result[r, offset + 5] = np.float32(
                1.0 - float(fronthaul_load[r, es]) / float(fronthaul_capacity[r, es])
            )
            if midhaul_split[split]:
                result[r, offset + 6] = np.float32(
                    1.0 - float(midhaul_load[es, rc]) / float(midhaul_capacity[es, rc])
                )

        offset += 7
        for group in range(3):
            minimum = np.inf
            maximum = -np.inf
            total = 0.0
            count = 0
            if group == 0:
                for index in range(feasible_es.shape[1]):
                    candidate = int(feasible_es[r, index])
                    if candidate < 0:
                        continue
                    value = float(es_u[candidate])
                    minimum = min(minimum, value)
                    maximum = max(maximum, value)
                    total += value
                    count += 1
            elif group == 1:
                for index in range(feasible_rc.shape[1]):
                    candidate = int(feasible_rc[r, index])
                    if candidate < 0:
                        continue
                    value = float(rc_u[candidate])
                    minimum = min(minimum, value)
                    maximum = max(maximum, value)
                    total += value
                    count += 1
            else:
                for index in range(feasible_mh_es.shape[1]):
                    mh_es = int(feasible_mh_es[r, index])
                    mh_rc = int(feasible_mh_rc[r, index])
                    if mh_es < 0:
                        continue
                    bandwidth = float(midhaul_capacity[mh_es, mh_rc])
                    value = (
                        0.0
                        if bandwidth <= 0.0
                        else float(midhaul_load[mh_es, mh_rc]) / bandwidth
                    )
                    minimum = min(minimum, value)
                    maximum = max(maximum, value)
                    total += value
                    count += 1
            if count:
                result[r, offset] = np.float32(1.0 - maximum)
                result[r, offset + 1] = np.float32(1.0 - total / count)
                result[r, offset + 2] = np.float32(1.0 - minimum)
            offset += 3
    return result


def _resource_loads_kernel(
    traffic: np.ndarray,
    split_id: np.ndarray,
    es_id: np.ndarray,
    rc_id: np.ndarray,
    intensity: float,
    es_fraction: np.ndarray,
    rc_fraction: np.ndarray,
    direct_split: np.ndarray,
    fh_multiplier: np.ndarray,
    fh_offset: np.ndarray,
    mh_multiplier: np.ndarray,
    mh_offset: np.ndarray,
    direct_multiplier: np.ndarray,
    direct_offset: np.ndarray,
    num_es: int,
    num_rc: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_rh = traffic.shape[0]
    es = np.zeros(num_es, dtype=np.float64)
    rc = np.zeros(num_rc, dtype=np.float64)
    fronthaul = np.zeros((num_rh, num_es), dtype=np.float64)
    midhaul = np.zeros((num_es, num_rc), dtype=np.float64)
    direct = np.zeros((num_rh, num_rc), dtype=np.float64)
    for r in range(num_rh):
        split = int(split_id[r])
        selected_es = int(es_id[r])
        selected_rc = int(rc_id[r])
        workload = float(traffic[r]) * intensity
        es[selected_es] += workload * float(es_fraction[split])
        rc[selected_rc] += workload * float(rc_fraction[split])
        if direct_split[split]:
            direct[r, selected_rc] += (
                float(traffic[r]) * direct_multiplier[split] + direct_offset[split]
            )
        else:
            fronthaul[r, selected_es] += (
                float(traffic[r]) * fh_multiplier[split] + fh_offset[split]
            )
            midhaul[selected_es, selected_rc] += (
                float(traffic[r]) * mh_multiplier[split] + mh_offset[split]
            )
    return es, rc, fronthaul, midhaul, direct


def _handover_kernel(
    handover_prob: np.ndarray,
    handover_count: np.ndarray,
    pdcp: np.ndarray,
    rlc: np.ndarray,
    split_id: np.ndarray,
    es_id: np.ndarray,
    rc_id: np.ndarray,
    pdcp_anchor_is_es: np.ndarray,
    rlc_anchor_is_es: np.ndarray,
    direct_split: np.ndarray,
    fronthaul_bw: np.ndarray,
    midhaul_bw: np.ndarray,
    direct_bw: np.ndarray,
    pdcp_weight: float,
    rlc_weight: float,
    sla_weight: float,
    base_latency_ms: float,
    budget_ms: float,
    mbps_to_mb_per_ms: float,
    min_effective_bw: float,
) -> tuple[float, float, float, float]:
    num_rh = split_id.shape[0]
    num_es = fronthaul_bw.shape[1]
    pdcp_anchor = np.empty(num_rh, dtype=np.int64)
    rlc_anchor = np.empty(num_rh, dtype=np.int64)
    path_bw = np.empty(num_rh, dtype=np.float64)
    for r in range(num_rh):
        split = int(split_id[r])
        selected_es = int(es_id[r])
        selected_rc = int(rc_id[r])
        pdcp_anchor[r] = selected_es if pdcp_anchor_is_es[split] else num_es + selected_rc
        rlc_anchor[r] = selected_es if rlc_anchor_is_es[split] else num_es + selected_rc
        if direct_split[split]:
            path_bw[r] = direct_bw[r, selected_rc]
        else:
            path_bw[r] = min(
                fronthaul_bw[r, selected_es], midhaul_bw[selected_es, selected_rc]
            )
    state_cost = 0.0
    sla_cost = 0.0
    diagonal_prob = 0.0
    diagonal_count = 0.0
    for r in range(num_rh):
        for rp in range(num_rh):
            if r == rp:
                diagonal_prob += handover_prob[r, rp]
                diagonal_count += handover_count[r, rp]
                continue
            expected_ues = handover_count[r, rp]
            if expected_ues <= 0.0:
                continue
            pdcp_crosses = pdcp_anchor[r] != pdcp_anchor[rp]
            rlc_crosses = rlc_anchor[r] != rlc_anchor[rp]
            state_cost += expected_ues * (
                pdcp_weight * pdcp[r] * (1.0 if pdcp_crosses else 0.0)
                + rlc_weight * rlc[r] * (1.0 if rlc_crosses else 0.0)
            )
            effective_mbps = max(min(path_bw[r], path_bw[rp]), min_effective_bw)
            transfer_mb = 0.0
            if pdcp_crosses:
                transfer_mb += pdcp[r]
            if rlc_crosses:
                transfer_mb += rlc[r]
            split_delay_ms = transfer_mb / (effective_mbps * mbps_to_mb_per_ms)
            latency_excess = max(base_latency_ms + split_delay_ms - budget_ms, 0.0)
            sla_cost += sla_weight * expected_ues * latency_excess
    return state_cost, sla_cost, diagonal_prob, diagonal_count


if NUMBA_AVAILABLE:
    causal_kernel = njit(cache=True, parallel=False, fastmath=False)(_causal_kernel)
    causal_v2_kernel = njit(cache=True, parallel=False, fastmath=False)(_causal_v2_kernel)
    resource_loads_kernel = njit(cache=True, parallel=False, fastmath=False)(
        _resource_loads_kernel
    )
    handover_kernel = njit(cache=True, parallel=False, fastmath=False)(_handover_kernel)
else:
    causal_kernel = None
    causal_v2_kernel = None
    resource_loads_kernel = None
    handover_kernel = None


def _padded(rows: list[np.ndarray]) -> np.ndarray:
    width = max((len(row) for row in rows), default=0)
    result = np.full((len(rows), width), -1, dtype=np.int64)
    for row_id, row in enumerate(rows):
        result[row_id, : len(row)] = row
    return result


def _build_lookup(
    fronthaul_mask: np.ndarray,
    midhaul_mask: np.ndarray,
    direct_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feasible_es: list[np.ndarray] = []
    feasible_rc: list[np.ndarray] = []
    feasible_mh_es: list[np.ndarray] = []
    feasible_mh_rc: list[np.ndarray] = []
    for r in range(fronthaul_mask.shape[0]):
        es_ids = np.flatnonzero(fronthaul_mask[r]).astype(np.int64)
        rc_ids = set(np.flatnonzero(direct_mask[r]).tolist())
        mh_es: list[int] = []
        mh_rc: list[int] = []
        for es in es_ids:
            for rc in np.flatnonzero(midhaul_mask[es]):
                rc_ids.add(int(rc))
                mh_es.append(int(es))
                mh_rc.append(int(rc))
        feasible_es.append(es_ids)
        feasible_rc.append(np.asarray(sorted(rc_ids), dtype=np.int64))
        feasible_mh_es.append(np.asarray(mh_es, dtype=np.int64))
        feasible_mh_rc.append(np.asarray(mh_rc, dtype=np.int64))
    return (
        _padded(feasible_es),
        _padded(feasible_rc),
        _padded(feasible_mh_es),
        _padded(feasible_mh_rc),
    )


@dataclass(frozen=True)
class CausalKernelStatic:
    direct_split: np.ndarray
    es_compute_split: np.ndarray
    midhaul_split: np.ndarray
    lookups: dict[str | None, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


def build_causal_static(episode: Any) -> CausalKernelStatic:
    names = episode.split_demand.split_names
    static = CausalKernelStatic(
        direct_split=np.asarray(
            [uses_direct_rh_rc(i, names) for i in range(len(names))], dtype=np.bool_
        ),
        es_compute_split=np.asarray(
            [uses_es_compute(i, names) for i in range(len(names))], dtype=np.bool_
        ),
        midhaul_split=np.asarray(
            [uses_midhaul(i, names) for i in range(len(names))], dtype=np.bool_
        ),
        lookups={
            None: _build_lookup(
                episode.fronthaul_mask,
                episode.midhaul_mask,
                episode.direct_rh_rc_mask,
            )
        },
    )
    for profile_id, profile in episode.topology_profiles.items():
        static.lookups[str(profile_id)] = _build_lookup(
            profile.fronthaul_mask, profile.midhaul_mask, profile.direct_rh_rc_mask
        )
    return static


def causal_args(
    episode: Any, action: Any, loads: Any, static: CausalKernelStatic
) -> tuple[Any, ...]:
    profile_id = episode.metadata.get("active_topology_profile_id")
    lookup = static.lookups[None if profile_id is None else str(profile_id)]
    return (
        action.split_id,
        action.es_id,
        action.rc_id,
        loads.es,
        loads.rc,
        loads.fronthaul,
        loads.midhaul,
        loads.direct,
        episode.es_compute_capacity,
        episode.rc_compute_capacity,
        episode.fronthaul_bandwidth_mbps,
        episode.midhaul_bandwidth_mbps,
        episode.direct_rh_rc_bandwidth_mbps,
        static.direct_split,
        static.es_compute_split,
        static.midhaul_split,
        *lookup,
    )


@dataclass(frozen=True)
class ConstraintKernelStatic:
    intensity: float
    split_tables: tuple[np.ndarray, ...]
    pdcp_anchor_is_es: np.ndarray
    rlc_anchor_is_es: np.ndarray


def build_constraint_static(evaluator: Any) -> ConstraintKernelStatic:
    episode = evaluator.episode
    names = episode.split_demand.split_names
    count = len(names)
    tables = [
        np.zeros(count, dtype=np.float64),
        np.zeros(count, dtype=np.float64),
        np.zeros(count, dtype=np.bool_),
        np.zeros(count, dtype=np.float64),
        np.zeros(count, dtype=np.float64),
        np.zeros(count, dtype=np.float64),
        np.zeros(count, dtype=np.float64),
        np.zeros(count, dtype=np.float64),
        np.zeros(count, dtype=np.float64),
    ]
    es_fraction, rc_fraction, direct, fh_mult, fh_offset, mh_mult, mh_offset, direct_mult, direct_offset = tables
    demand = evaluator.formulation["transport"]["demand"]

    def coefficients(kind: str) -> tuple[float, float]:
        if kind in {"o7_2x", "o8"}:
            return 0.0, float(demand[f"{kind}_mbps"])
        rule = demand[kind]
        return float(rule["multiplier"]), float(rule["offset_mbps"])

    for split, name in enumerate(names):
        record = evaluator.formulation["splits"][name]
        es_fraction[split] = float(record["compute"]["es_fraction"])
        rc_fraction[split] = float(record["compute"]["rc_fraction"])
        transport = record["transport"]
        direct[split] = transport["direct"] is not None
        if direct[split]:
            direct_mult[split], direct_offset[split] = coefficients(
                str(transport["direct"])
            )
        else:
            fh_mult[split], fh_offset[split] = coefficients(str(transport["fronthaul"]))
            mh_mult[split], mh_offset[split] = coefficients(str(transport["midhaul"]))
    return ConstraintKernelStatic(
        intensity=float(evaluator.formulation["compute"]["total_intensity_per_mbps"]),
        split_tables=tuple(tables),
        pdcp_anchor_is_es=np.asarray(
            [pdcp_anchor_type(i, names) == "es" for i in range(count)], dtype=np.bool_
        ),
        rlc_anchor_is_es=np.asarray(
            [rlc_anchor_type(i, names) == "es" for i in range(count)], dtype=np.bool_
        ),
    )


def resource_args(
    evaluator: Any, time_index: int, action: Any, static: ConstraintKernelStatic
) -> tuple[Any, ...]:
    episode = evaluator.episode
    return (
        episode.lambda_rt[time_index],
        action.split_id,
        action.es_id,
        action.rc_id,
        static.intensity,
        *static.split_tables,
        episode.config.num_es,
        episode.config.num_rc,
    )


def handover_args(
    evaluator: Any, time_index: int, action: Any, static: ConstraintKernelStatic
) -> tuple[Any, ...]:
    episode = evaluator.episode
    return (
        episode.handover_matrix_t[time_index],
        episode.handover_count_t[time_index],
        episode.s_pdcp_rt[time_index],
        episode.s_rlc_rt[time_index],
        action.split_id,
        action.es_id,
        action.rc_id,
        static.pdcp_anchor_is_es,
        static.rlc_anchor_is_es,
        static.split_tables[2],
        episode.fronthaul_bandwidth_mbps,
        episode.midhaul_bandwidth_mbps,
        episode.direct_rh_rc_bandwidth_mbps,
        float(episode.cost_weights.pdcp_state_transfer),
        float(episode.cost_weights.rlc_state_transfer),
        float(episode.cost_weights.handover_sla),
        float(evaluator.sla.base_latency_ms),
        float(evaluator.sla.budget_ms),
        float(evaluator.sla.mbps_to_mb_per_ms),
        float(evaluator.penalties.min_effective_bandwidth_mbps),
    )

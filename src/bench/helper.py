"""Numerical helpers for the CPLEX processing-delay approximation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
DEFAULT_APPROXIMATION_VERSION = (
    "capped_reciprocal_v1_drho_0p01_upper_adjacent_chord_flat_cap_tail_v1"
)


def _f64(values: ArrayLike) -> FloatArray:
    array = np.ascontiguousarray(values, dtype=np.float64)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ProcessingDelayApproximationSpec:
    """Scientific function and approved upper-PWL policy."""

    tau_ms: float
    utilization_cap: float
    utilization_step: float = 0.01
    approximation_version: str = DEFAULT_APPROXIMATION_VERSION

    def validate(self) -> None:
        if not math.isfinite(self.tau_ms) or self.tau_ms <= 0.0:
            raise ValueError("tau_ms must be finite and positive")
        if not math.isfinite(self.utilization_cap) or not 0.0 < self.utilization_cap < 1.0:
            raise ValueError("utilization_cap must be finite and in (0, 1)")
        if not math.isclose(self.utilization_step, 0.01, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("the utilization step must be exactly 0.01")
        quotient = self.utilization_cap / self.utilization_step
        if not math.isclose(quotient, round(quotient), rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("utilization_cap must lie on the utilization grid")
        if not self.approximation_version:
            raise ValueError("approximation_version must be non-empty")


@dataclass(frozen=True)
class UpperProcessingDelayPwl:
    """Common upper chord PWL for ``tau / (1 - min(rho, cap))``."""

    rho_points: FloatArray
    values: FloatArray
    domain_max: float
    max_error_ms: float
    approximation_version: str

    def metadata(self) -> dict[str, Any]:
        return {
            "approximation_version": self.approximation_version,
            "utilization_points": int(self.rho_points.size),
            "domain_max": float(self.domain_max),
            "max_upper_error_ms": float(self.max_error_ms),
        }


@dataclass(frozen=True)
class NodeUtilizationUpperBounds:
    """Time/node utilization bounds for connectivity-valid actions."""

    es: FloatArray
    rc: FloatArray

    @property
    def domain_max(self) -> float:
        return float(max(np.max(self.es, initial=0.0), np.max(self.rc, initial=0.0)))


def exact_processing_delay(
    utilization: ArrayLike,
    tau_ms: float,
    utilization_cap: float,
) -> FloatArray:
    """Evaluate the authoritative capped processing delay."""

    rho = np.asarray(utilization, dtype=np.float64)
    if np.any(~np.isfinite(rho)) or np.any(rho < 0.0):
        raise ValueError("utilization must be finite and nonnegative")
    return np.asarray(tau_ms / (1.0 - np.minimum(rho, utilization_cap)), dtype=np.float64)


def evaluate_upper_processing_delay_pwl(
    utilization: ArrayLike,
    table: UpperProcessingDelayPwl,
) -> FloatArray:
    """Evaluate the adjacent-chord upper PWL."""

    rho = np.asarray(utilization, dtype=np.float64)
    if np.any(~np.isfinite(rho)) or np.any(rho < 0.0) or np.any(rho > table.domain_max + 1e-12):
        raise ValueError("utilization lies outside the validated table domain")
    return np.asarray(np.interp(rho, table.rho_points, table.values), dtype=np.float64)


def _measure_upper_error(
    spec: ProcessingDelayApproximationSpec,
    rho_points: FloatArray,
    values: FloatArray,
    domain_max: float,
) -> float:
    """Compute the exact maximum chord error on every reciprocal segment."""

    reciprocal_domain_max = min(domain_max, spec.utilization_cap)
    maximum = 0.0
    cap_indices = np.flatnonzero(rho_points <= spec.utilization_cap + 1e-12)
    for left_index, right_index in zip(cap_indices[:-1], cap_indices[1:], strict=True):
        left = float(rho_points[left_index])
        right = float(rho_points[right_index])
        if left >= reciprocal_domain_max or right <= left:
            continue
        evaluated_right = min(right, reciprocal_domain_max)
        slope = float((values[right_index] - values[left_index]) / (right - left))
        candidates = [left, evaluated_right]
        if slope > 0.0:
            stationary = 1.0 - math.sqrt(spec.tau_ms / slope)
            if left < stationary < evaluated_right:
                candidates.append(stationary)
        for rho in candidates:
            chord = values[left_index] + slope * (rho - left)
            exact = spec.tau_ms / (1.0 - rho)
            maximum = max(maximum, float(chord - exact))
    return maximum


def generate_upper_processing_delay_pwl(
    spec: ProcessingDelayApproximationSpec,
    domain_max: float,
) -> UpperProcessingDelayPwl:
    """Generate the common 0.01 upper chord PWL and exact flat cap tail."""

    spec.validate()
    if not math.isfinite(domain_max) or domain_max < 0.0:
        raise ValueError("domain_max must be finite and nonnegative")
    cap = spec.utilization_cap
    grid_count = int(round(cap / spec.utilization_step))
    canonical = np.arange(grid_count + 1, dtype=np.float64) * spec.utilization_step
    canonical[-1] = cap
    rho_points = canonical
    if domain_max > cap + 1e-12:
        rho_points = np.concatenate((canonical, np.asarray([domain_max], dtype=np.float64)))
    values = exact_processing_delay(rho_points, spec.tau_ms, cap)
    points_f = _f64(rho_points)
    values_f = _f64(values)
    table = UpperProcessingDelayPwl(
        rho_points=points_f,
        values=values_f,
        domain_max=float(domain_max),
        max_error_ms=_measure_upper_error(spec, points_f, values_f, domain_max),
        approximation_version=spec.approximation_version,
    )
    validate_upper_processing_delay_pwl(spec, table)
    return table


def validate_upper_processing_delay_pwl(
    spec: ProcessingDelayApproximationSpec,
    table: UpperProcessingDelayPwl,
) -> None:
    """Validate the common chord construction and exact constant tail."""

    spec.validate()
    if table.approximation_version != spec.approximation_version:
        raise ValueError("processing-delay approximation version mismatch")
    points = table.rho_points
    values = table.values
    if any(array.ndim != 1 or not array.flags.c_contiguous for array in (points, values)):
        raise ValueError("processing-delay arrays must be one-dimensional and C-contiguous")
    if any(np.any(~np.isfinite(array)) for array in (points, values)):
        raise ValueError("processing-delay arrays must be finite")
    if points.size < 2 or values.shape != points.shape:
        raise ValueError("processing-delay table shapes are inconsistent")
    if np.any(np.diff(points) <= 0.0):
        raise ValueError("rho points must be strictly increasing")
    if not math.isclose(float(points[0]), 0.0, abs_tol=1e-12):
        raise ValueError("rho table must start at zero")
    if points[-1] + 1e-12 < table.domain_max:
        raise ValueError("upper table does not cover the declared domain")
    cap_indices = np.flatnonzero(
        np.isclose(points, spec.utilization_cap, rtol=0.0, atol=1e-12)
    )
    if cap_indices.size != 1:
        raise ValueError("utilization cap must occur exactly once in the rho table")
    cap_index = int(cap_indices[0])
    canonical = np.arange(cap_index + 1, dtype=np.float64) * spec.utilization_step
    canonical[-1] = spec.utilization_cap
    if not np.allclose(points[: cap_index + 1], canonical, rtol=0.0, atol=1e-12):
        raise ValueError("rho points do not use the approved 0.01 canonical grid")
    expected = exact_processing_delay(points, spec.tau_ms, spec.utilization_cap)
    if not np.allclose(values, expected, rtol=1e-13, atol=1e-13):
        raise ValueError("upper table is not exact at the chord endpoints")
    if np.any(np.diff(values[cap_index:]) != 0.0):
        raise ValueError("upper approximation must use an exact flat cap tail")
    measured = _measure_upper_error(spec, points, values, table.domain_max)
    if not math.isclose(measured, table.max_error_ms, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("stored upper approximation error is invalid")
    if measured < -1e-12:
        raise ValueError("upper approximation error has an invalid sign")


def measure_upper_processing_delay_error(
    spec: ProcessingDelayApproximationSpec,
    table: UpperProcessingDelayPwl,
) -> float:
    """Return the analytic maximum one-node upper-PWL error."""

    validate_upper_processing_delay_pwl(spec, table)
    return float(table.max_error_ms)


def compute_node_utilization_upper_bounds(
    traffic: ArrayLike,
    action_owner: ArrayLike,
    action_es: ArrayLike,
    action_rc: ArrayLike,
    action_es_fraction: ArrayLike,
    action_rc_fraction: ArrayLike,
    compute_intensity: float,
    es_capacity: ArrayLike,
    rc_capacity: ArrayLike,
) -> NodeUtilizationUpperBounds:
    """Bound each node by independently maximizing each RH contribution."""

    lam = np.asarray(traffic, dtype=np.float64)
    owner = np.asarray(action_owner, dtype=np.int64)
    es = np.asarray(action_es, dtype=np.int64)
    rc = np.asarray(action_rc, dtype=np.int64)
    es_fraction = np.asarray(action_es_fraction, dtype=np.float64)
    rc_fraction = np.asarray(action_rc_fraction, dtype=np.float64)
    es_cap = np.asarray(es_capacity, dtype=np.float64)
    rc_cap = np.asarray(rc_capacity, dtype=np.float64)
    if lam.ndim != 2:
        raise ValueError("traffic must have shape [T, R]")
    k = owner.size
    if any(array.shape != (k,) for array in (es, rc, es_fraction, rc_fraction)):
        raise ValueError("action arrays must be one-dimensional with equal length")
    if not math.isfinite(compute_intensity) or compute_intensity <= 0.0:
        raise ValueError("compute_intensity must be finite and positive")
    if np.any(~np.isfinite(lam)) or np.any(lam < 0.0):
        raise ValueError("traffic must be finite and nonnegative")
    if np.any(es_cap <= 0.0) or np.any(rc_cap <= 0.0):
        raise ValueError("node capacities must be positive")
    _, rhs = lam.shape
    if np.any(owner < 0) or np.any(owner >= rhs):
        raise ValueError("action owner is outside the traffic RH dimension")
    max_es_fraction = np.zeros((rhs, es_cap.size), dtype=np.float64)
    max_rc_fraction = np.zeros((rhs, rc_cap.size), dtype=np.float64)
    for action in range(k):
        max_es_fraction[owner[action], es[action]] = max(
            max_es_fraction[owner[action], es[action]], es_fraction[action]
        )
        max_rc_fraction[owner[action], rc[action]] = max(
            max_rc_fraction[owner[action], rc[action]], rc_fraction[action]
        )
    es_load_bound = compute_intensity * lam @ max_es_fraction
    rc_load_bound = compute_intensity * lam @ max_rc_fraction
    return NodeUtilizationUpperBounds(
        es=_f64(es_load_bound / es_cap[None, :]),
        rc=_f64(rc_load_bound / rc_cap[None, :]),
    )

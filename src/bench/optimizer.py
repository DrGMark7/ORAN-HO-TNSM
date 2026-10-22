"""CPLEX MILP oracle for the runtime O-RAN orchestration objective.

Python owns project-aware loading and coefficient preparation.  The compiled
Concert backend receives only contiguous numeric arrays through a stable C ABI.
The existing :mod:`src.model.constraint` evaluator remains the numerical
authority for every returned action trace.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from src.bench.helper import (
    ProcessingDelayApproximationSpec,
    compute_node_utilization_upper_bounds,
    evaluate_upper_processing_delay_pwl,
    exact_processing_delay,
    generate_upper_processing_delay_pwl,
    measure_upper_processing_delay_error,
    validate_upper_processing_delay_pwl,
)
from src.bench.native_hard_service import (
    NativeHardServiceAudit,
    audit_native_hard_service_instance,
)
from src.model.constraint import OrchestrationConstraintEvaluator
from src.model.inference.post_process import (
    ActionSpaceSpec,
    action_mask_kwargs_from_episode,
    build_link_feasibility_action_mask,
    decode_flat_actions,
    decode_tuple_action,
    encode_flat_actions,
)
from src.model.split_semantics import (
    pdcp_anchor_type,
    rlc_anchor_type,
    uses_direct_rh_rc,
)
from src.pipeline.schema import MockEpisode, PlacementAction


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]
OptimizationMode = Literal["mirror_soft", "hard_service_feasibility"]
OPTIMIZATION_MODE_VALUES: Mapping[OptimizationMode, int] = MappingProxyType({
    "mirror_soft": 0,
    "hard_service_feasibility": 1,
})

COMPONENT_NAMES = (
    "placement_cost",
    "compute_overload_cost",
    "bandwidth_violation_cost",
    "latency_violation_cost",
    "handover_state_transfer_cost",
    "handover_sla_cost",
    "handover_control_cost",
    "reconfiguration_cost",
)
REQUIRED_FORMULATION = "split_transport_utilization_latency_v1"
MILP_ABI_VERSION = 3


class MilpBackendError(RuntimeError):
    """A typed C ABI/backend failure."""

    def __init__(self, code: int, message: str, episode_id: int, solve_mode: str) -> None:
        super().__init__(
            f"CPLEX backend error {code} for episode={episode_id} mode={solve_mode}: {message}"
        )
        self.code = code
        self.backend_message = message
        self.episode_id = episode_id
        self.solve_mode = solve_mode


class MilpValidationError(RuntimeError):
    """The extracted solver trace does not reproduce the evaluator objective."""


@dataclass(frozen=True)
class MilpSolveOptions:
    solve_mode: Literal["snapshot", "episode"] = "episode"
    optimization_mode: OptimizationMode = "mirror_soft"
    time_limit_seconds: float | None = None
    relative_mip_gap: float = 1e-6
    threads: int | None = 1
    deterministic: bool = True
    random_seed: int = 0
    mip_emphasis: int | None = None
    use_policy_mip_start: bool = False
    hard_service_snapshot_mip_start: bool = False
    hard_service_snapshot_time_limit_seconds: float = 10.0
    native_hard_latency_deadline_ms: float | None = None
    validate_with_evaluator: bool = True
    validation_tolerance: float = 1e-7
    validation_relative_tolerance: float = 1e-8

    def validate(self) -> None:
        if self.solve_mode not in {"snapshot", "episode"}:
            raise ValueError("solve_mode must be 'snapshot' or 'episode'")
        if self.optimization_mode not in OPTIMIZATION_MODE_VALUES:
            raise ValueError(f"unknown optimization_mode: {self.optimization_mode!r}")
        if self.time_limit_seconds is not None and self.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")
        if not math.isfinite(self.relative_mip_gap) or self.relative_mip_gap < 0:
            raise ValueError("relative_mip_gap must be finite and nonnegative")
        if self.threads is not None and self.threads <= 0:
            raise ValueError("threads must be positive")
        if self.hard_service_snapshot_time_limit_seconds <= 0:
            raise ValueError("hard_service_snapshot_time_limit_seconds must be positive")
        if (
            self.native_hard_latency_deadline_ms is not None
            and (
                not math.isfinite(self.native_hard_latency_deadline_ms)
                or self.native_hard_latency_deadline_ms <= 0
            )
        ):
            raise ValueError("native_hard_latency_deadline_ms must be finite and positive")
        if self.validation_tolerance < 0 or self.validation_relative_tolerance < 0:
            raise ValueError("validation tolerances must be nonnegative")


@dataclass(frozen=True)
class MilpSolverStats:
    status_code: int
    status_name: str
    is_proven_optimal: bool
    has_incumbent: bool
    objective_value: float | None
    best_bound: float | None
    relative_gap: float | None
    build_time_seconds: float
    solve_time_seconds: float
    num_binary_variables: int
    num_continuous_variables: int
    num_constraints: int
    cplex_version: str
    mip_start_status: str
    cplex_status_code: int
    cplex_status_name: str
    termination_reason: str
    num_nonzero_coefficients: int
    node_count: int
    remaining_node_count: int
    resolved_time_limit_seconds: float
    resolved_relative_mip_gap: float
    resolved_threads: int
    resolved_random_seed: int
    resolved_parallel_mode: int
    resolved_mip_emphasis: int
    mip_start_objective: float | None = None
    mip_start_accepted: bool = False


@dataclass(frozen=True)
class MilpEpisodeResult:
    episode_id: int
    absolute_start_t: int
    episode_length: int
    actions: tuple[PlacementAction, ...]
    encoded_actions: NDArray[np.int64]
    optimization_mode: OptimizationMode
    solver_total_cost: float
    evaluator_total_cost: float
    solver_component_costs: Mapping[str, float]
    component_costs: Mapping[str, float]
    step_costs: tuple[Mapping[str, Any], ...]
    solver_step_costs: tuple[Mapping[str, Any], ...]
    feasible_rate: float
    solver_stats: MilpSolverStats
    formulation_metadata: Mapping[str, Any]
    preparation_time_seconds: float
    extraction_time_seconds: float
    validation_time_seconds: float


@dataclass(frozen=True)
class MilpInstance:
    """Solver-neutral sparse numeric instance.

    ``action_*`` arrays use global valid-action indices.  A given valid action
    belongs to exactly one RH according to ``action_offsets``.  Time/action
    coefficient matrices are flattened row-major before crossing the ABI.
    """

    episode_id: int
    optimization_mode: OptimizationMode
    absolute_times: IntArray
    num_rhs: int
    num_es: int
    num_rc: int
    action_offsets: IntArray
    action_owner_rh: IntArray
    action_encoded_id: IntArray
    action_split: IntArray
    action_es: IntArray
    action_rc: IntArray
    action_gateway: IntArray
    placement_cost: FloatArray
    action_transport_latency: FloatArray
    action_es_fraction: FloatArray
    action_rc_fraction: FloatArray
    es_load: FloatArray
    rc_load: FloatArray
    fronthaul_load: FloatArray
    midhaul_load: FloatArray
    direct_load: FloatArray
    es_capacity: FloatArray
    rc_capacity: FloatArray
    fronthaul_capacity: FloatArray
    midhaul_capacity: FloatArray
    direct_capacity: FloatArray
    rh_deadline_ms: FloatArray
    upper_rho_points: FloatArray
    upper_values: FloatArray
    es_utilization_upper: FloatArray
    rc_utilization_upper: FloatArray
    utilization_cap: float
    ho_time: IntArray
    ho_action_a: IntArray
    ho_action_b: IntArray
    ho_state_cost: FloatArray
    ho_sla_cost: FloatArray
    ho_control_cost: FloatArray
    compute_weight: float
    bandwidth_weight: float
    latency_weight: float
    reconf_enabled: bool
    split_change_weight: float
    es_change_weight: float
    rc_change_weight: float
    fixed_reconf_cost: FloatArray
    mip_start_action: IntArray
    metadata: Mapping[str, Any]
    initial_previous_action: PlacementAction | None = None

    @property
    def horizon(self) -> int:
        return int(self.absolute_times.size)

    @property
    def num_actions(self) -> int:
        return int(self.action_encoded_id.size)

    def validate(self) -> None:
        t, r, d, c, k = self.horizon, self.num_rhs, self.num_es, self.num_rc, self.num_actions
        vectors = {
            "absolute_times": (self.absolute_times, (t,)),
            "action_offsets": (self.action_offsets, (r + 1,)),
            "action_owner_rh": (self.action_owner_rh, (k,)),
            "action_encoded_id": (self.action_encoded_id, (k,)),
            "action_split": (self.action_split, (k,)),
            "action_es": (self.action_es, (k,)),
            "action_rc": (self.action_rc, (k,)),
            "action_gateway": (self.action_gateway, (k,)),
            "placement_cost": (self.placement_cost, (k,)),
            "action_transport_latency": (self.action_transport_latency, (k,)),
            "action_es_fraction": (self.action_es_fraction, (k,)),
            "action_rc_fraction": (self.action_rc_fraction, (k,)),
            "es_load": (self.es_load, (t, k)),
            "rc_load": (self.rc_load, (t, k)),
            "fronthaul_load": (self.fronthaul_load, (t, k)),
            "midhaul_load": (self.midhaul_load, (t, k)),
            "direct_load": (self.direct_load, (t, k)),
            "es_capacity": (self.es_capacity, (d,)),
            "rc_capacity": (self.rc_capacity, (c,)),
            "fronthaul_capacity": (self.fronthaul_capacity, (r * d,)),
            "midhaul_capacity": (self.midhaul_capacity, (d * c,)),
            "direct_capacity": (self.direct_capacity, (r * c,)),
            "rh_deadline_ms": (self.rh_deadline_ms, (r,)),
            "es_utilization_upper": (self.es_utilization_upper, (t, d)),
            "rc_utilization_upper": (self.rc_utilization_upper, (t, c)),
            "fixed_reconf_cost": (self.fixed_reconf_cost, (k,)),
        }
        for name, (array, shape) in vectors.items():
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}; got {array.shape}")
            if not array.flags.c_contiguous:
                raise ValueError(f"{name} must be C-contiguous")
        if self.action_offsets[0] != 0 or self.action_offsets[-1] != k:
            raise ValueError("action_offsets endpoints are inconsistent")
        if np.any(np.diff(self.action_offsets) <= 0):
            raise ValueError("every RH must have at least one valid action")
        if np.any(self.action_owner_rh < 0) or np.any(self.action_owner_rh >= r):
            raise ValueError("action_owner_rh is out of range")
        for rh in range(r):
            lo, hi = int(self.action_offsets[rh]), int(self.action_offsets[rh + 1])
            if not np.all(self.action_owner_rh[lo:hi] == rh):
                raise ValueError("action owner/offset mismatch")
        n_ho = self.ho_time.size
        for name in (
            "ho_action_a", "ho_action_b", "ho_state_cost", "ho_sla_cost", "ho_control_cost"
        ):
            if getattr(self, name).shape != (n_ho,):
                raise ValueError(f"{name} must have shape ({n_ho},)")
        if self.upper_rho_points.ndim != 1 or self.upper_values.shape != self.upper_rho_points.shape:
            raise ValueError("upper PWL arrays must be one-dimensional and equally sized")
        pwl_arrays = (self.upper_rho_points, self.upper_values)
        if any(not array.flags.c_contiguous for array in pwl_arrays):
            raise ValueError("PWL arrays must be C-contiguous")
        if self.upper_rho_points.size < 2 or np.any(np.diff(self.upper_rho_points) <= 0.0):
            raise ValueError("upper PWL rho points must be strictly increasing")
        numeric = [
            self.placement_cost, self.action_transport_latency,
            self.action_es_fraction, self.action_rc_fraction,
            self.es_load, self.rc_load, self.fronthaul_load, self.midhaul_load,
            self.direct_load, self.es_capacity, self.rc_capacity,
            self.fronthaul_capacity, self.midhaul_capacity, self.direct_capacity,
            self.rh_deadline_ms, self.upper_rho_points, self.upper_values, self.es_utilization_upper,
            self.rc_utilization_upper, self.ho_state_cost, self.ho_sla_cost,
            self.ho_control_cost, self.fixed_reconf_cost,
        ]
        if any(np.any(~np.isfinite(a)) or np.any(a < -1e-12) for a in numeric):
            raise ValueError("MILP numeric arrays must be finite and nonnegative")
        if not math.isfinite(self.utilization_cap):
            raise ValueError("PWL utilization cap must be finite")
        if np.any(self.ho_time < 0) or np.any(self.ho_time >= t):
            raise ValueError("handover time index is out of range")
        if np.any(self.ho_action_a < 0) or np.any(self.ho_action_a >= k):
            raise ValueError("handover source action is out of range")
        if np.any(self.ho_action_b < 0) or np.any(self.ho_action_b >= k):
            raise ValueError("handover target action is out of range")
        if self.mip_start_action.size not in {0, t * r}:
            raise ValueError("MIP start must be empty or contain one global action index per t/RH")


class _CInput(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_int32), ("struct_size", ctypes.c_uint64),
        ("num_times", ctypes.c_int32), ("num_rhs", ctypes.c_int32),
        ("num_es", ctypes.c_int32), ("num_rc", ctypes.c_int32),
        ("num_actions", ctypes.c_int32), ("num_upper_points", ctypes.c_int32),
        ("num_ho_terms", ctypes.c_int64),
        ("action_offsets", ctypes.POINTER(ctypes.c_int32)),
        ("action_owner", ctypes.POINTER(ctypes.c_int32)),
        ("action_split", ctypes.POINTER(ctypes.c_int32)),
        ("action_es", ctypes.POINTER(ctypes.c_int32)),
        ("action_rc", ctypes.POINTER(ctypes.c_int32)),
        ("action_gateway", ctypes.POINTER(ctypes.c_int32)),
        ("placement_cost", ctypes.POINTER(ctypes.c_double)),
        ("action_transport_latency", ctypes.POINTER(ctypes.c_double)),
        ("action_es_fraction", ctypes.POINTER(ctypes.c_double)),
        ("action_rc_fraction", ctypes.POINTER(ctypes.c_double)),
        ("es_load", ctypes.POINTER(ctypes.c_double)),
        ("rc_load", ctypes.POINTER(ctypes.c_double)),
        ("fronthaul_load", ctypes.POINTER(ctypes.c_double)),
        ("midhaul_load", ctypes.POINTER(ctypes.c_double)),
        ("direct_load", ctypes.POINTER(ctypes.c_double)),
        ("es_capacity", ctypes.POINTER(ctypes.c_double)),
        ("rc_capacity", ctypes.POINTER(ctypes.c_double)),
        ("fronthaul_capacity", ctypes.POINTER(ctypes.c_double)),
        ("midhaul_capacity", ctypes.POINTER(ctypes.c_double)),
        ("direct_capacity", ctypes.POINTER(ctypes.c_double)),
        ("rh_deadline_ms", ctypes.POINTER(ctypes.c_double)),
        ("upper_rho_points", ctypes.POINTER(ctypes.c_double)),
        ("upper_values", ctypes.POINTER(ctypes.c_double)),
        ("es_utilization_upper", ctypes.POINTER(ctypes.c_double)),
        ("rc_utilization_upper", ctypes.POINTER(ctypes.c_double)),
        ("utilization_cap", ctypes.c_double),
        ("ho_time", ctypes.POINTER(ctypes.c_int32)),
        ("ho_action_a", ctypes.POINTER(ctypes.c_int32)),
        ("ho_action_b", ctypes.POINTER(ctypes.c_int32)),
        ("ho_state_cost", ctypes.POINTER(ctypes.c_double)),
        ("ho_sla_cost", ctypes.POINTER(ctypes.c_double)),
        ("ho_control_cost", ctypes.POINTER(ctypes.c_double)),
        ("compute_weight", ctypes.c_double),
        ("bandwidth_weight", ctypes.c_double), ("latency_weight", ctypes.c_double),
        ("reconf_enabled", ctypes.c_int32),
        ("split_change_weight", ctypes.c_double),
        ("es_change_weight", ctypes.c_double), ("rc_change_weight", ctypes.c_double),
        ("fixed_reconf_cost", ctypes.POINTER(ctypes.c_double)),
        ("mip_start_action", ctypes.POINTER(ctypes.c_int32)),
        ("mip_start_size", ctypes.c_int64),
    ]


class _COptions(ctypes.Structure):
    _fields_ = [
        ("time_limit_seconds", ctypes.c_double), ("relative_mip_gap", ctypes.c_double),
        ("threads", ctypes.c_int32), ("deterministic", ctypes.c_int32),
        ("random_seed", ctypes.c_int32), ("mip_emphasis", ctypes.c_int32),
        ("optimization_mode", ctypes.c_int32),
    ]


class _COutput(ctypes.Structure):
    _fields_ = [
        ("status_code", ctypes.c_int32), ("is_proven_optimal", ctypes.c_int32),
        ("has_incumbent", ctypes.c_int32), ("objective_value", ctypes.c_double),
        ("best_bound", ctypes.c_double), ("relative_gap", ctypes.c_double),
        ("build_time_seconds", ctypes.c_double), ("solve_time_seconds", ctypes.c_double),
        ("num_binary_variables", ctypes.c_int64),
        ("num_continuous_variables", ctypes.c_int64), ("num_constraints", ctypes.c_int64),
        ("num_nonzero_coefficients", ctypes.c_int64),
        ("node_count", ctypes.c_int64), ("remaining_node_count", ctypes.c_int64),
        ("cplex_status_code", ctypes.c_int32),
        ("resolved_time_limit_seconds", ctypes.c_double),
        ("resolved_relative_mip_gap", ctypes.c_double),
        ("resolved_threads", ctypes.c_int32), ("resolved_random_seed", ctypes.c_int32),
        ("resolved_parallel_mode", ctypes.c_int32), ("resolved_mip_emphasis", ctypes.c_int32),
        ("selected_action", ctypes.POINTER(ctypes.c_int32)),
        ("selected_action_size", ctypes.c_int64),
        ("component_costs", ctypes.c_double * 8),
        ("status_name", ctypes.c_char * 128),
        ("cplex_status_name", ctypes.c_char * 128),
        ("termination_reason", ctypes.c_char * 128),
        ("cplex_version", ctypes.c_char * 64),
        ("mip_start_status", ctypes.c_char * 64),
    ]


def _i32(values: Any) -> IntArray:
    return np.ascontiguousarray(values, dtype=np.int32)


def _f64(values: Any) -> FloatArray:
    return np.ascontiguousarray(values, dtype=np.float64)


def _ptr(array: np.ndarray, ctype: type[ctypes._SimpleCData]) -> Any:
    if array.size == 0:
        return ctypes.POINTER(ctype)()
    return array.ctypes.data_as(ctypes.POINTER(ctype))


def dataset_sha256(dataset_dir: str | Path) -> str:
    """Hash dataset files deterministically without modifying the dataset."""

    root = Path(dataset_dir)
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _anchor(kind: str, split: int, es: int, rc: int, names: tuple[str, ...]) -> tuple[str, int]:
    anchor_type = pdcp_anchor_type(split, names) if kind == "pdcp" else rlc_anchor_type(split, names)
    return ("es", es) if anchor_type == "es" else ("rc", rc)


def _path_bandwidth(ep: MockEpisode, rh: int, split: int, es: int, rc: int) -> float:
    if uses_direct_rh_rc(split, ep.split_demand.split_names):
        return float(ep.direct_rh_rc_bandwidth_mbps[rh, rc])
    return float(min(ep.fronthaul_bandwidth_mbps[rh, es], ep.midhaul_bandwidth_mbps[es, rc]))


class OrchestrationMilpOptimizer:
    """Prepare, solve, and validate CPLEX orchestration MILPs."""

    def __init__(
        self,
        episode: MockEpisode,
        *,
        project_config: dict[str, Any] | None = None,
        library_path: str | Path | None = None,
        dataset_dir: str | Path | None = None,
        episode_length: int = 288,
    ) -> None:
        episode.validate()
        if episode.config.horizon % episode_length:
            raise ValueError("flat dataset horizon must be divisible by episode_length")
        self.episode = episode
        self.episode_length = int(episode_length)
        self.num_episodes = episode.config.horizon // self.episode_length
        self.evaluator = OrchestrationConstraintEvaluator(episode, project_config=project_config)
        if self.evaluator.formulation_version != REQUIRED_FORMULATION:
            raise ValueError(
                f"MILP requires formulation {REQUIRED_FORMULATION!r}; "
                f"got {self.evaluator.formulation_version!r}"
            )
        self.dataset_dir = Path(dataset_dir or episode.metadata.get("dataset_dir", ""))
        self.dataset_hash = dataset_sha256(self.dataset_dir) if self.dataset_dir.is_dir() else "unknown"
        self.library_path = Path(library_path).resolve() if library_path is not None else None
        self._lib: ctypes.CDLL | None = None

    def _load_backend(self) -> ctypes.CDLL:
        if self._lib is not None:
            return self._lib
        if self.library_path is None:
            raise FileNotFoundError(
                "CPLEX optimizer shared library path is required. Build it with "
                "`cmake -S src/bench/cpp -B build/cplex -DCPLEX_ROOT=/path/to/CPLEX_Studio` "
                "and pass --library."
            )
        if not self.library_path.is_file():
            raise FileNotFoundError(
                f"CPLEX optimizer library not found: {self.library_path}. Build it with "
                "`cmake -S src/bench/cpp -B build/cplex -DCPLEX_ROOT=/path/to/CPLEX_Studio` "
                "followed by `cmake --build build/cplex`."
            )
        lib = ctypes.CDLL(str(self.library_path))
        lib.oran_cplex_solve.argtypes = [
            ctypes.POINTER(_CInput), ctypes.POINTER(_COptions), ctypes.POINTER(_COutput),
            ctypes.POINTER(ctypes.c_char), ctypes.c_size_t,
        ]
        lib.oran_cplex_solve.restype = ctypes.c_int
        lib.oran_cplex_free_output.argtypes = [ctypes.POINTER(_COutput)]
        lib.oran_cplex_free_output.restype = None
        lib.oran_sizeof_milp_input.argtypes = []
        lib.oran_sizeof_milp_input.restype = ctypes.c_size_t
        lib.oran_sizeof_milp_options.argtypes = []
        lib.oran_sizeof_milp_options.restype = ctypes.c_size_t
        lib.oran_sizeof_milp_output.argtypes = []
        lib.oran_sizeof_milp_output.restype = ctypes.c_size_t
        if lib.oran_sizeof_milp_input() != ctypes.sizeof(_CInput):
            raise RuntimeError("C ABI mismatch: OranMilpInput size differs")
        if lib.oran_sizeof_milp_options() != ctypes.sizeof(_COptions):
            raise RuntimeError("C ABI mismatch: OranMilpOptions size differs")
        if lib.oran_sizeof_milp_output() != ctypes.sizeof(_COutput):
            raise RuntimeError("C ABI mismatch: OranMilpOutput size differs")
        self._lib = lib
        return lib

    def formulation_metadata(self) -> dict[str, Any]:
        ep = self.episode
        result = dict(self.evaluator.formulation_metadata())
        objective_weights = dict(self.evaluator.formulation["objective"]["weights"])
        result.update(
            dataset_hash=self.dataset_hash,
            dataset_dir=str(self.dataset_dir),
            split_names=list(ep.split_demand.split_names),
            num_rhs=ep.config.num_rhs, num_es=ep.config.num_es, num_rc=ep.config.num_rc,
            flat_horizon=ep.config.horizon, episode_length=self.episode_length,
            num_episodes=self.num_episodes,
            cost_weights=vars(ep.cost_weights),
            objective_weights=objective_weights,
        )
        service_weights = [
            float(objective_weights[name])
            for name in (
                "compute_violation",
                "bandwidth_violation",
                "latency_violation",
            )
        ]
        result["resolved_service_weight"] = (
            service_weights[0]
            if service_weights.count(service_weights[0]) == len(service_weights)
            else None
        )
        return result

    def prepare_instance(
        self,
        *,
        episode_id: int,
        local_times: Sequence[int] | None = None,
        previous_action: PlacementAction | None = None,
        policy_mip_start: Sequence[PlacementAction] | NDArray[np.integer[Any]] | None = None,
        optimization_mode: OptimizationMode = "mirror_soft",
        native_hard_latency_deadline_ms: float | None = None,
    ) -> MilpInstance:
        if episode_id < 0 or episode_id >= self.num_episodes:
            raise ValueError(f"episode_id must be in [0, {self.num_episodes})")
        if optimization_mode not in OPTIMIZATION_MODE_VALUES:
            raise ValueError(f"unknown optimization_mode: {optimization_mode!r}")
        if local_times is None:
            local = np.arange(self.episode_length, dtype=np.int32)
        else:
            local = _i32(local_times)
            if local.ndim != 1 or local.size == 0:
                raise ValueError("local_times must be a non-empty one-dimensional sequence")
            if np.any(local < 0) or np.any(local >= self.episode_length):
                raise ValueError("local time is outside the episode")
            if np.any(np.diff(local) != 1):
                raise ValueError("local_times must be consecutive and increasing")
        absolute = _i32(episode_id * self.episode_length + local)
        ep = self.episode
        profile_id = ep.topology_profile_id_for_episode(episode_id)
        ep.activate_topology_profile(profile_id)
        if ep.rh_service_deadline_ms is None:
            raise ValueError("active MILP formulation requires per-RH service deadlines")
        if (
            native_hard_latency_deadline_ms is not None
            and (
                not math.isfinite(native_hard_latency_deadline_ms)
                or native_hard_latency_deadline_ms <= 0
            )
        ):
            raise ValueError("native hard-latency deadline must be finite and positive")
        names = ep.split_demand.split_names
        spec = ActionSpaceSpec.from_topology(ep.config)
        mask = build_link_feasibility_action_mask(
            ep.config, ep.fronthaul_mask, ep.midhaul_mask, ep.direct_rh_rc_mask,
            names, **action_mask_kwargs_from_episode(ep),
        )
        offsets = [0]
        owner: list[int] = []
        encoded: list[int] = []
        splits: list[int] = []
        es_ids: list[int] = []
        rc_ids: list[int] = []
        gateways: list[int] = []
        invalid_rhs: list[int] = []
        for rh in range(ep.config.num_rhs):
            for action_id in np.flatnonzero(mask[rh]):
                split, es, rc = decode_tuple_action(int(action_id), spec)
                owner.append(rh); encoded.append(int(action_id)); splits.append(split)
                es_ids.append(es); rc_ids.append(rc)
                gateways.append(int(not uses_direct_rh_rc(split, names)))
            if offsets[-1] == len(encoded):
                invalid_rhs.append(rh)
            offsets.append(len(encoded))
        if invalid_rhs:
            raise ValueError(
                f"episode={episode_id} has no connectivity-valid action for RHs {invalid_rhs}"
            )
        owner_a, encoded_a = _i32(owner), _i32(encoded)
        split_a, es_a, rc_a, gateway_a = _i32(splits), _i32(es_ids), _i32(rc_ids), _i32(gateways)
        offsets_a = _i32(offsets)
        k, nt = len(encoded), len(absolute)

        formulation = self.evaluator.formulation
        split_records = [formulation["splits"][name] for name in names]
        es_fraction = np.asarray(
            [float(record["compute"]["es_fraction"]) for record in split_records]
        )
        rc_fraction = np.asarray(
            [float(record["compute"]["rc_fraction"]) for record in split_records]
        )
        weights = formulation["objective"]["weights"]
        compute_cfg = formulation["compute"]
        compute_intensity = float(compute_cfg["total_intensity_per_mbps"])
        delay_spec = ProcessingDelayApproximationSpec(
            tau_ms=float(compute_cfg["base_service_time_ms"]),
            utilization_cap=float(compute_cfg["utilization_cap"]),
        )
        placement = np.zeros(k)
        transport_latency = np.zeros(k)
        action_es_fraction = es_fraction[split_a]
        action_rc_fraction = rc_fraction[split_a]
        for a in range(k):
            rh, split, es, rc = owner[a], splits[a], es_ids[a], rc_ids[a]
            if gateways[a]:
                base = ep.es_placement_cost[es] + ep.rc_placement_cost[rc]
                base += ep.fronthaul_cost[rh, es] + ep.midhaul_cost[es, rc]
                transport_latency[a] = (
                    ep.fronthaul_latency_ms[rh, es] + ep.midhaul_latency_ms[es, rc]
                )
            else:
                base = ep.rc_placement_cost[rc] + ep.direct_rh_rc_cost[rh, rc]
                transport_latency[a] = ep.direct_rh_rc_latency_ms[rh, rc]
            placement[a] = float(weights["placement"]) * base

        es_load = np.zeros((nt, k)); rc_load = np.zeros((nt, k))
        fh_load = np.zeros((nt, k)); mh_load = np.zeros((nt, k))
        direct_load = np.zeros((nt, k))
        demand = formulation["transport"]["demand"]
        for ti, abs_t in enumerate(absolute):
            traffic = ep.lambda_rt[int(abs_t)]
            for a in range(k):
                rh, split, es, rc = owner[a], splits[a], es_ids[a], rc_ids[a]
                rate = float(traffic[rh])
                es_load[ti, a] = rate * compute_intensity * action_es_fraction[a]
                rc_load[ti, a] = rate * compute_intensity * action_rc_fraction[a]
                transport = split_records[split]["transport"]
                if gateways[a]:
                    fh_load[ti, a] = self._transport_demand_mbps(
                        str(transport["fronthaul"]), rate, demand
                    )
                    mh_load[ti, a] = self._transport_demand_mbps(
                        str(transport["midhaul"]), rate, demand
                    )
                else:
                    direct_load[ti, a] = self._transport_demand_mbps(
                        str(transport["direct"]), rate, demand
                    )

        utilization_bounds = compute_node_utilization_upper_bounds(
            ep.lambda_rt[absolute], owner_a, es_a, rc_a,
            action_es_fraction, action_rc_fraction, compute_intensity,
            ep.es_compute_capacity, ep.rc_compute_capacity,
        )
        delay_table = generate_upper_processing_delay_pwl(
            delay_spec, utilization_bounds.domain_max
        )
        validate_upper_processing_delay_pwl(delay_spec, delay_table)

        ho_t: list[int] = []; ho_a: list[int] = []; ho_b: list[int] = []
        ho_state: list[float] = []; ho_sla: list[float] = []; ho_ctrl: list[float] = []
        control_enabled = self.evaluator.handover_control.enabled and ep.cost_weights.rrc_anchor_crossing > 0
        for ti, abs_t in enumerate(absolute):
            counts = ep.handover_count_t[int(abs_t)]
            for rh in range(ep.config.num_rhs):
                for rhp in range(ep.config.num_rhs):
                    count = float(counts[rh, rhp])
                    if rh == rhp or count <= 0:
                        continue
                    for a in range(offsets[rh], offsets[rh + 1]):
                        for b in range(offsets[rhp], offsets[rhp + 1]):
                            pdcp_cross = _anchor("pdcp", splits[a], es_ids[a], rc_ids[a], names) != _anchor("pdcp", splits[b], es_ids[b], rc_ids[b], names)
                            rlc_cross = _anchor("rlc", splits[a], es_ids[a], rc_ids[a], names) != _anchor("rlc", splits[b], es_ids[b], rc_ids[b], names)
                            state = count * (
                                ep.cost_weights.pdcp_state_transfer * ep.s_pdcp_rt[int(abs_t), rh] * pdcp_cross
                                + ep.cost_weights.rlc_state_transfer * ep.s_rlc_rt[int(abs_t), rh] * rlc_cross
                            )
                            effective_mbps = max(
                                min(
                                    _path_bandwidth(ep, rh, splits[a], es_ids[a], rc_ids[a]),
                                    _path_bandwidth(ep, rhp, splits[b], es_ids[b], rc_ids[b]),
                                ), self.evaluator.penalties.min_effective_bandwidth_mbps,
                            )
                            transfer_mb = ep.s_pdcp_rt[int(abs_t), rh] * pdcp_cross + ep.s_rlc_rt[int(abs_t), rh] * rlc_cross
                            delay = transfer_mb / (effective_mbps * self.evaluator.sla.mbps_to_mb_per_ms)
                            sla = ep.cost_weights.handover_sla * count * max(
                                self.evaluator.sla.base_latency_ms + delay - self.evaluator.sla.budget_ms, 0.0
                            )
                            ctrl = ep.cost_weights.rrc_anchor_crossing * count * (rc_ids[a] != rc_ids[b]) if control_enabled else 0.0
                            if state > 0 or sla > 0 or ctrl > 0:
                                ho_t.append(ti); ho_a.append(a); ho_b.append(b)
                                ho_state.append(float(state)); ho_sla.append(float(sla)); ho_ctrl.append(float(ctrl))

        fixed_reconf = np.zeros(k)
        reconf = self.evaluator.reconfiguration
        if previous_action is not None:
            previous_action.validate(ep.config)
            if nt != 1:
                raise ValueError("fixed previous_action is supported only for snapshot solves")
            rh_encoded = encode_flat_actions(previous_action, ep.config)
            for a in range(k):
                rh = owner[a]
                ps, pe, pr = decode_tuple_action(int(rh_encoded[rh]), spec)
                fixed_reconf[a] = self._transition_cost(ps, pe, pr, splits[a], es_ids[a], rc_ids[a])

        mip_start = self._prepare_mip_start(policy_mip_start, offsets, encoded, nt)
        metadata = self.formulation_metadata()
        metadata["optimization_mode"] = optimization_mode
        metadata["optimization_mode_label"] = {
            "mirror_soft": "upper-PWL soft-objective oracle",
            "hard_service_feasibility": "hard-service-feasible oracle",
        }[optimization_mode]
        metadata["processing_delay_approximation"] = {
            **delay_table.metadata(),
            "max_upper_error_ms": measure_upper_processing_delay_error(
                delay_spec, delay_table
            ),
            "utilization_step": delay_spec.utilization_step,
            "utilization_cap": delay_spec.utilization_cap,
            "es_utilization_upper_max": float(np.max(utilization_bounds.es)),
            "rc_utilization_upper_max": float(np.max(utilization_bounds.rc)),
            "construction": "upper adjacent chords with an exact flat cap tail",
        }
        instance = MilpInstance(
            episode_id=episode_id, optimization_mode=optimization_mode,
            absolute_times=absolute, num_rhs=ep.config.num_rhs,
            num_es=ep.config.num_es, num_rc=ep.config.num_rc, action_offsets=offsets_a,
            action_owner_rh=owner_a, action_encoded_id=encoded_a, action_split=split_a,
            action_es=es_a, action_rc=rc_a, action_gateway=gateway_a,
            placement_cost=_f64(placement),
            action_transport_latency=_f64(transport_latency),
            action_es_fraction=_f64(action_es_fraction),
            action_rc_fraction=_f64(action_rc_fraction),
            es_load=_f64(es_load), rc_load=_f64(rc_load),
            fronthaul_load=_f64(fh_load), midhaul_load=_f64(mh_load),
            direct_load=_f64(direct_load), es_capacity=_f64(ep.es_compute_capacity),
            rc_capacity=_f64(ep.rc_compute_capacity),
            fronthaul_capacity=_f64(ep.fronthaul_bandwidth_mbps.reshape(-1)),
            midhaul_capacity=_f64(ep.midhaul_bandwidth_mbps.reshape(-1)),
            direct_capacity=_f64(ep.direct_rh_rc_bandwidth_mbps.reshape(-1)),
            rh_deadline_ms=(
                _f64(np.full(ep.config.num_rhs, native_hard_latency_deadline_ms))
                if native_hard_latency_deadline_ms is not None
                else _f64(ep.rh_service_deadline_ms)
            ),
            upper_rho_points=_f64(delay_table.rho_points),
            upper_values=_f64(delay_table.values),
            es_utilization_upper=_f64(utilization_bounds.es),
            rc_utilization_upper=_f64(utilization_bounds.rc),
            utilization_cap=delay_spec.utilization_cap,
            ho_time=_i32(ho_t), ho_action_a=_i32(ho_a), ho_action_b=_i32(ho_b),
            ho_state_cost=_f64(ho_state), ho_sla_cost=_f64(ho_sla),
            ho_control_cost=_f64(ho_ctrl), compute_weight=float(weights["compute_violation"]),
            bandwidth_weight=float(weights["bandwidth_violation"]),
            latency_weight=float(weights["latency_violation"]),
            reconf_enabled=bool(reconf.enabled and (reconf.split_change_weight or reconf.es_change_weight or reconf.rc_change_weight)),
            split_change_weight=reconf.split_change_weight, es_change_weight=reconf.es_change_weight,
            rc_change_weight=reconf.rc_change_weight, fixed_reconf_cost=_f64(fixed_reconf),
            mip_start_action=mip_start, metadata=MappingProxyType(metadata),
            initial_previous_action=previous_action,
        )
        instance.validate()
        return instance

    @staticmethod
    def _transport_demand_mbps(
        kind: str, traffic_mbps: float, demand: Mapping[str, Any]
    ) -> float:
        fixed_key = f"{kind}_mbps"
        if fixed_key in demand:
            return float(demand[fixed_key])
        record = demand.get(kind)
        if not isinstance(record, Mapping):
            raise ValueError(f"unknown named transport demand: {kind!r}")
        mode = str(record.get("mode"))
        if mode not in {"traffic", "affine_traffic"}:
            raise ValueError(f"unsupported transport demand mode {mode!r} for {kind!r}")
        return (
            float(record.get("multiplier", 1.0)) * float(traffic_mbps)
            + float(record.get("offset_mbps", 0.0))
        )

    def _transition_cost(self, ps: int, pe: int, pr: int, cs: int, ce: int, cr: int) -> float:
        cfg = self.evaluator.reconfiguration
        if not cfg.enabled:
            return 0.0
        direct = self.episode.split_demand.split_names
        cost = cfg.split_change_weight * (ps != cs) + cfg.rc_change_weight * (pr != cr)
        if not uses_direct_rh_rc(ps, direct) and not uses_direct_rh_rc(cs, direct):
            cost += cfg.es_change_weight * (pe != ce)
        return float(cost)

    def _prepare_mip_start(
        self, trace: Sequence[PlacementAction] | NDArray[np.integer[Any]] | None,
        offsets: list[int], encoded: list[int], nt: int,
    ) -> IntArray:
        if trace is None:
            return _i32([])
        if isinstance(trace, np.ndarray) and trace.ndim == 2:
            encoded_trace = np.asarray(trace, dtype=np.int64)
        else:
            placements = tuple(trace)  # type: ignore[arg-type]
            if len(placements) != nt:
                raise ValueError("policy MIP start length does not match solve horizon")
            encoded_trace = np.vstack([encode_flat_actions(a, self.episode.config) for a in placements])
        if encoded_trace.shape != (nt, self.episode.config.num_rhs):
            raise ValueError("encoded policy MIP start must have shape [T, R]")
        result = np.empty_like(encoded_trace, dtype=np.int32)
        for ti in range(nt):
            for rh in range(self.episode.config.num_rhs):
                candidates = encoded[offsets[rh]:offsets[rh + 1]]
                try:
                    result[ti, rh] = offsets[rh] + candidates.index(int(encoded_trace[ti, rh]))
                except ValueError as exc:
                    raise ValueError(f"connectivity-invalid MIP start at t={ti}, rh={rh}") from exc
        return _i32(result.reshape(-1))

    def solve_snapshot(
        self, episode_id: int, local_t: int, *, options: MilpSolveOptions | None = None,
        previous_action: PlacementAction | None = None,
        policy_mip_start: Sequence[PlacementAction] | NDArray[np.integer[Any]] | None = None,
    ) -> MilpEpisodeResult:
        opts = options or MilpSolveOptions(solve_mode="snapshot")
        if opts.solve_mode != "snapshot":
            opts = MilpSolveOptions(**{**vars(opts), "solve_mode": "snapshot"})
        return self._prepare_solve(
            episode_id, [local_t], opts, previous_action=previous_action,
            policy_mip_start=policy_mip_start,
        )

    def solve_episode(
        self, episode_id: int, *, options: MilpSolveOptions | None = None,
        policy_mip_start: Sequence[PlacementAction] | NDArray[np.integer[Any]] | None = None,
    ) -> MilpEpisodeResult:
        opts = options or MilpSolveOptions(solve_mode="episode")
        if opts.solve_mode != "episode":
            opts = MilpSolveOptions(**{**vars(opts), "solve_mode": "episode"})
        opts.validate()
        if (
            opts.optimization_mode == "hard_service_feasibility"
            and opts.hard_service_snapshot_mip_start
            and policy_mip_start is None
        ):
            policy_mip_start = self._hard_service_snapshot_mip_start(episode_id, opts)
            opts = replace(
                opts,
                use_policy_mip_start=True,
            )
        seed_objective: float | None = None
        if policy_mip_start is not None:
            if opts.optimization_mode == "hard_service_feasibility":
                audit = self.audit_native_hard_service_trace(
                    episode_id,
                    policy_mip_start,
                    native_hard_latency_deadline_ms=opts.native_hard_latency_deadline_ms,
                )
                if not audit.feasible:
                    assert audit.first_violation is not None
                    violation = audit.first_violation
                    raise MilpValidationError(
                        "refusing native-infeasible hard-service MIP start: "
                        f"{violation.kind}, margin={violation.margin:.17g}, "
                        f"row={violation.row}"
                    )
            seed_objective = self._exact_trace_objective(
                episode_id, policy_mip_start
            )

        result = self._prepare_solve(
            episode_id, None, opts, policy_mip_start=policy_mip_start
        )
        if seed_objective is None:
            return result

        allowed = opts.validation_tolerance + opts.validation_relative_tolerance * max(
            abs(seed_objective), abs(result.evaluator_total_cost)
        )
        if result.evaluator_total_cost > seed_objective + allowed:
            raise MilpValidationError(
                "CPLEX incumbent does not dominate the submitted native-feasible "
                f"MIP start: incumbent_J={result.evaluator_total_cost:.17g}, "
                f"seed_J={seed_objective:.17g}, tolerance={allowed:.17g}, "
                f"backend_status={result.solver_stats.mip_start_status}"
            )
        if result.solver_stats.mip_start_status != "accepted_by_cplex":
            raise MilpValidationError(
                "CPLEX did not confirm that the submitted MIP start produced "
                "an initial incumbent: "
                f"backend_status={result.solver_stats.mip_start_status}"
            )
        return replace(
            result,
            solver_stats=replace(
                result.solver_stats,
                mip_start_status="accepted_incumbent",
                mip_start_objective=seed_objective,
                mip_start_accepted=True,
            ),
        )

    def audit_native_hard_service_trace(
        self,
        episode_id: int,
        policy_mip_start: Sequence[PlacementAction] | NDArray[np.integer[Any]],
        *,
        native_hard_latency_deadline_ms: float | None = None,
    ) -> NativeHardServiceAudit:
        """Audit a full trace against the native upper-PWL hard rows."""

        return self.audit_native_hard_service_traces(
            episode_id,
            (policy_mip_start,),
            native_hard_latency_deadline_ms=native_hard_latency_deadline_ms,
        )[0]

    def audit_native_hard_service_traces(
        self,
        episode_id: int,
        policy_mip_starts: Sequence[
            Sequence[PlacementAction] | NDArray[np.integer[Any]]
        ],
        *,
        native_hard_latency_deadline_ms: float | None = None,
    ) -> tuple[NativeHardServiceAudit, ...]:
        """Audit multiple traces while preparing episode coefficients once."""

        base = self.prepare_instance(
            episode_id=episode_id,
            optimization_mode="hard_service_feasibility",
            native_hard_latency_deadline_ms=native_hard_latency_deadline_ms,
        )
        offsets = [int(value) for value in base.action_offsets]
        encoded = [int(value) for value in base.action_encoded_id]
        return tuple(
            audit_native_hard_service_instance(
                replace(
                    base,
                    mip_start_action=self._prepare_mip_start(
                        trace, offsets, encoded, base.horizon
                    ),
                )
            )
            for trace in policy_mip_starts
        )

    def _exact_trace_objective(
        self,
        episode_id: int,
        actions: Sequence[PlacementAction] | NDArray[np.integer[Any]],
    ) -> float:
        placements = (
            tuple(
                decode_flat_actions(row, self.episode.config)
                for row in np.asarray(actions, dtype=np.int64)
            )
            if isinstance(actions, np.ndarray)
            else tuple(actions)
        )
        if len(placements) != self.episode_length:
            raise ValueError("policy MIP start must contain one complete episode")
        total = 0.0
        previous: PlacementAction | None = None
        absolute_start = episode_id * self.episode_length
        for local_t, action in enumerate(placements):
            report = self.evaluator.evaluate(absolute_start + local_t, action, previous)
            total += sum(float(getattr(report, name)) for name in COMPONENT_NAMES)
            previous = action
        return total

    def _hard_service_snapshot_mip_start(
        self,
        episode_id: int,
        episode_options: MilpSolveOptions,
    ) -> list[PlacementAction]:
        """Create a full-episode hard-feasible MIP start from snapshot solves.

        Hard service constraints are per-timestep. Handover and reconfiguration
        are objective-only, so independent hard-feasible snapshot actions form
        a valid feasible sequence for the full episode model.
        """

        snapshot_limit = min(
            episode_options.time_limit_seconds or episode_options.hard_service_snapshot_time_limit_seconds,
            episode_options.hard_service_snapshot_time_limit_seconds,
        )
        snapshot_options = replace(
            episode_options,
            solve_mode="snapshot",
            time_limit_seconds=snapshot_limit,
            # Snapshot solves run sequentially, so they can safely use the
            # caller's requested CPLEX thread budget. Forcing one thread makes
            # large-topology snapshots needlessly miss otherwise sufficient
            # per-snapshot time limits.
            threads=episode_options.threads,
            use_policy_mip_start=False,
            hard_service_snapshot_mip_start=False,
            validate_with_evaluator=True,
        )
        actions: list[PlacementAction] = []
        for local_t in range(self.episode_length):
            try:
                snapshot = self.solve_snapshot(
                    episode_id,
                    local_t,
                    options=snapshot_options,
                )
            except MilpBackendError as exc:
                raise RuntimeError(
                    "hard_service_feasibility has no snapshot incumbent at "
                    f"episode={episode_id}, local_t={local_t}; full-episode "
                    "solve is infeasible or needs a larger snapshot time limit"
                ) from exc
            actions.append(snapshot.actions[0])
        previous: PlacementAction | None = None
        absolute_start = episode_id * self.episode_length
        for local_t, action in enumerate(actions):
            report = self.evaluator.evaluate(absolute_start + local_t, action, previous)
            service_costs = (
                report.compute_overload_cost,
                report.bandwidth_violation_cost,
                report.latency_violation_cost,
            )
            if not report.feasible or any(abs(float(value)) > 1e-9 for value in service_costs):
                raise MilpValidationError(
                    "assembled hard-service snapshot seed is not exact-evaluator feasible "
                    f"at episode={episode_id}, local_t={local_t}"
                )
            previous = action
        print(
            "[CPLEX upper-PWL] assembled hard-service snapshot seed passed "
            f"exact evaluator service feasibility {self.episode_length}/{self.episode_length}"
        )
        return actions

    def _prepare_solve(
        self, episode_id: int, local_times: Sequence[int] | None, options: MilpSolveOptions,
        *, previous_action: PlacementAction | None = None,
        policy_mip_start: Sequence[PlacementAction] | NDArray[np.integer[Any]] | None = None,
    ) -> MilpEpisodeResult:
        options.validate()
        before_hash = dataset_sha256(self.dataset_dir) if self.dataset_dir.is_dir() else self.dataset_hash
        started = time.perf_counter()
        instance = self.prepare_instance(
            episode_id=episode_id, local_times=local_times, previous_action=previous_action,
            policy_mip_start=policy_mip_start if options.use_policy_mip_start else None,
            optimization_mode=options.optimization_mode,
            native_hard_latency_deadline_ms=options.native_hard_latency_deadline_ms,
        )
        instance = MilpInstance(**{
            **vars(instance),
            "metadata": MappingProxyType({
                **dict(instance.metadata),
                "solver_options": {
                    "solve_mode": options.solve_mode,
                    "optimization_mode": options.optimization_mode,
                    "time_limit_seconds": options.time_limit_seconds,
                    "relative_mip_gap_target": options.relative_mip_gap,
                    "threads": options.threads,
                    "deterministic": options.deterministic,
                    "random_seed": options.random_seed,
                    "hard_service_snapshot_mip_start": options.hard_service_snapshot_mip_start,
                    "hard_service_snapshot_time_limit_seconds": options.hard_service_snapshot_time_limit_seconds,
                    "native_hard_latency_deadline_ms": options.native_hard_latency_deadline_ms,
                },
            }),
        })
        prep_seconds = time.perf_counter() - started
        native = self._solve_instance(instance, options)
        result = self._exact_incumbent_result(
            native,
            phase_label=(
                "upper-PWL snapshot"
                if options.solve_mode == "snapshot"
                else "upper-PWL full episode"
            ),
        )
        after_hash = dataset_sha256(self.dataset_dir) if self.dataset_dir.is_dir() else self.dataset_hash
        if before_hash != after_hash:
            raise RuntimeError("dataset SHA-256 changed during benchmark")
        return MilpEpisodeResult(**{**vars(result), "preparation_time_seconds": prep_seconds})

    def _exact_incumbent_result(
        self,
        result: MilpEpisodeResult,
        *,
        phase_label: str,
    ) -> MilpEpisodeResult:
        if not result.component_costs or not result.step_costs:
            raise MilpValidationError(f"{phase_label} was not exact-evaluator validated")
        diagnostics = self._approximation_diagnostics(
            result.actions,
            result.absolute_start_t,
            result.formulation_metadata,
        )
        if "snapshot" not in phase_label:
            self._log_approximation_diagnostics(phase_label, diagnostics)
        exact_total = float(result.evaluator_total_cost)
        stats = replace(
            result.solver_stats,
            objective_value=exact_total,
        )
        if "snapshot" not in phase_label:
            print(
                "[CPLEX upper-PWL] "
                f"native_objective={result.solver_stats.objective_value:.17g}, "
                f"exact_evaluator_J={exact_total:.17g}, "
                f"native_best_bound={result.solver_stats.best_bound:.17g}, "
                f"native_mip_gap={result.solver_stats.relative_gap:.17g}"
            )
        return replace(
            result,
            solver_total_cost=exact_total,
            solver_component_costs=result.component_costs,
            solver_step_costs=result.step_costs,
            solver_stats=stats,
        )

    def _approximation_diagnostics(
        self,
        actions: Sequence[PlacementAction],
        absolute_start_t: int,
        metadata: Mapping[str, Any],
    ) -> dict[str, float | int]:
        compute = self.evaluator.formulation["compute"]
        spec = ProcessingDelayApproximationSpec(
            tau_ms=float(compute["base_service_time_ms"]),
            utilization_cap=float(compute["utilization_cap"]),
        )
        approximation_metadata = metadata["processing_delay_approximation"]
        table = generate_upper_processing_delay_pwl(
            spec, float(approximation_metadata["domain_max"])
        )
        node_errors: list[float] = []
        selected_errors: list[float] = []
        classification_mismatches = 0
        names = self.episode.split_demand.split_names
        for offset, action in enumerate(actions):
            absolute_t = absolute_start_t + offset
            loads = self.evaluator.resource_loads(absolute_t, action)
            exact_es = exact_processing_delay(
                loads.es_utilization, spec.tau_ms, spec.utilization_cap
            )
            exact_rc = exact_processing_delay(
                loads.rc_utilization, spec.tau_ms, spec.utilization_cap
            )
            approx_es = evaluate_upper_processing_delay_pwl(loads.es_utilization, table)
            approx_rc = evaluate_upper_processing_delay_pwl(loads.rc_utilization, table)
            node_errors.extend((approx_es - exact_es).tolist())
            node_errors.extend((approx_rc - exact_rc).tolist())
            for rh in range(self.episode.config.num_rhs):
                split = int(action.split_id[rh])
                es = int(action.es_id[rh])
                rc = int(action.rc_id[rh])
                record = self.evaluator.formulation["splits"][names[split]]
                alpha_es = float(record["compute"]["es_fraction"])
                alpha_rc = float(record["compute"]["rc_fraction"])
                if record["transport"]["direct"] is not None:
                    transport = float(self.episode.direct_rh_rc_latency_ms[rh, rc])
                else:
                    transport = float(
                        self.episode.fronthaul_latency_ms[rh, es]
                        + self.episode.midhaul_latency_ms[es, rc]
                    )
                exact_e2e = transport + alpha_es * exact_es[es] + alpha_rc * exact_rc[rc]
                approx_e2e = transport + alpha_es * approx_es[es] + alpha_rc * approx_rc[rc]
                signed_error = approx_e2e - exact_e2e
                if signed_error < -1e-9:
                    raise MilpValidationError(
                        "upper processing-delay approximation fell below exact by "
                        f"{signed_error:.17g} ms at absolute_t={absolute_t}, RH={rh}"
                    )
                selected_errors.append(max(float(signed_error), 0.0))
                deadline = float(self.episode.rh_service_deadline_ms[rh])
                classification_mismatches += int(
                    (exact_e2e > deadline + 1e-9) != (approx_e2e > deadline + 1e-9)
                )
        return {
            "maximum_node_processing_delay_error_ms": max(node_errors, default=0.0),
            "maximum_selected_rh_e2e_error_ms": max(selected_errors, default=0.0),
            "mean_selected_rh_e2e_error_ms": (
                float(np.mean(selected_errors)) if selected_errors else 0.0
            ),
            "deadline_classification_mismatches": classification_mismatches,
        }

    @staticmethod
    def _log_approximation_diagnostics(
        phase_label: str, diagnostics: Mapping[str, float | int]
    ) -> None:
        print(
            f"[CPLEX upper-PWL] phase={phase_label}, "
            f"max_node_delay_error_ms={diagnostics['maximum_node_processing_delay_error_ms']:.9g}, "
            f"max_selected_e2e_error_ms={diagnostics['maximum_selected_rh_e2e_error_ms']:.9g}, "
            f"mean_selected_e2e_error_ms={diagnostics['mean_selected_rh_e2e_error_ms']:.9g}, "
            f"deadline_classification_mismatches={diagnostics['deadline_classification_mismatches']}"
        )

    def _solve_instance(
        self,
        instance: MilpInstance,
        options: MilpSolveOptions,
    ) -> MilpEpisodeResult:
        lib = self._load_backend()
        ci = _CInput(
            MILP_ABI_VERSION, ctypes.sizeof(_CInput),
            instance.horizon, instance.num_rhs, instance.num_es, instance.num_rc,
            instance.num_actions, instance.upper_rho_points.size,
            instance.ho_time.size,
            _ptr(instance.action_offsets, ctypes.c_int32), _ptr(instance.action_owner_rh, ctypes.c_int32),
            _ptr(instance.action_split, ctypes.c_int32), _ptr(instance.action_es, ctypes.c_int32),
            _ptr(instance.action_rc, ctypes.c_int32), _ptr(instance.action_gateway, ctypes.c_int32),
            _ptr(instance.placement_cost, ctypes.c_double),
            _ptr(instance.action_transport_latency, ctypes.c_double),
            _ptr(instance.action_es_fraction, ctypes.c_double),
            _ptr(instance.action_rc_fraction, ctypes.c_double),
            _ptr(instance.es_load, ctypes.c_double), _ptr(instance.rc_load, ctypes.c_double),
            _ptr(instance.fronthaul_load, ctypes.c_double),
            _ptr(instance.midhaul_load, ctypes.c_double),
            _ptr(instance.direct_load, ctypes.c_double),
            _ptr(instance.es_capacity, ctypes.c_double), _ptr(instance.rc_capacity, ctypes.c_double),
            _ptr(instance.fronthaul_capacity, ctypes.c_double),
            _ptr(instance.midhaul_capacity, ctypes.c_double),
            _ptr(instance.direct_capacity, ctypes.c_double),
            _ptr(instance.rh_deadline_ms, ctypes.c_double),
            _ptr(instance.upper_rho_points, ctypes.c_double),
            _ptr(instance.upper_values, ctypes.c_double),
            _ptr(instance.es_utilization_upper, ctypes.c_double),
            _ptr(instance.rc_utilization_upper, ctypes.c_double),
            instance.utilization_cap,
            _ptr(instance.ho_time, ctypes.c_int32),
            _ptr(instance.ho_action_a, ctypes.c_int32), _ptr(instance.ho_action_b, ctypes.c_int32),
            _ptr(instance.ho_state_cost, ctypes.c_double), _ptr(instance.ho_sla_cost, ctypes.c_double),
            _ptr(instance.ho_control_cost, ctypes.c_double), instance.compute_weight,
            instance.bandwidth_weight, instance.latency_weight,
            int(instance.reconf_enabled), instance.split_change_weight, instance.es_change_weight,
            instance.rc_change_weight, _ptr(instance.fixed_reconf_cost, ctypes.c_double),
            _ptr(instance.mip_start_action, ctypes.c_int32), instance.mip_start_action.size,
        )
        co = _COptions(
            options.time_limit_seconds or 0.0, options.relative_mip_gap,
            options.threads or 0, int(options.deterministic), options.random_seed,
            options.mip_emphasis if options.mip_emphasis is not None else -1,
            OPTIMIZATION_MODE_VALUES[options.optimization_mode],
        )
        out = _COutput(); error = ctypes.create_string_buffer(4096)
        code = lib.oran_cplex_solve(ctypes.byref(ci), ctypes.byref(co), ctypes.byref(out), error, len(error))
        if code:
            raise MilpBackendError(code, error.value.decode("utf-8", "replace"), instance.episode_id, options.solve_mode)
        try:
            if not out.has_incumbent:
                generic_status = bytes(out.status_name).split(b"\0", 1)[0].decode()
                cplex_status = bytes(out.cplex_status_name).split(b"\0", 1)[0].decode()
                raise MilpBackendError(
                    out.status_code,
                    f"no incumbent: status={generic_status}, cplex_status={cplex_status}",
                    instance.episode_id, options.solve_mode,
                )
            extract_started = time.perf_counter()
            selected = np.ctypeslib.as_array(out.selected_action, shape=(out.selected_action_size,)).copy()
            selected = selected.reshape(instance.horizon, instance.num_rhs)
            encoded = np.empty_like(selected, dtype=np.int64)
            actions: list[PlacementAction] = []
            for ti in range(instance.horizon):
                s = np.empty(instance.num_rhs, dtype=np.int64); d = s.copy(); c = s.copy()
                for rh in range(instance.num_rhs):
                    a = int(selected[ti, rh])
                    if a < instance.action_offsets[rh] or a >= instance.action_offsets[rh + 1]:
                        raise MilpValidationError(f"solver selected action {a} outside RH {rh} offsets")
                    s[rh], d[rh], c[rh] = instance.action_split[a], instance.action_es[a], instance.action_rc[a]
                    encoded[ti, rh] = instance.action_encoded_id[a]
                actions.append(PlacementAction(s, d, c))
            extraction_seconds = time.perf_counter() - extract_started
            solver_components = {name: float(out.component_costs[i]) for i, name in enumerate(COMPONENT_NAMES)}
            validation_started = time.perf_counter()
            evaluator_components, steps, solver_steps, feasible_rate = self.validate_solution(
                instance, actions, selected, solver_components, float(out.objective_value),
                options,
            ) if options.validate_with_evaluator else ({}, (), (), float("nan"))
            validation_seconds = time.perf_counter() - validation_started
            status_name = bytes(out.status_name).split(b"\0", 1)[0].decode()
            stats = MilpSolverStats(
                status_code=out.status_code, status_name=status_name,
                is_proven_optimal=bool(out.is_proven_optimal), has_incumbent=bool(out.has_incumbent),
                objective_value=float(out.objective_value), best_bound=float(out.best_bound),
                relative_gap=float(out.relative_gap), build_time_seconds=float(out.build_time_seconds),
                solve_time_seconds=float(out.solve_time_seconds),
                num_binary_variables=int(out.num_binary_variables),
                num_continuous_variables=int(out.num_continuous_variables),
                num_constraints=int(out.num_constraints),
                cplex_version=bytes(out.cplex_version).split(b"\0", 1)[0].decode(),
                mip_start_status=bytes(out.mip_start_status).split(b"\0", 1)[0].decode(),
                cplex_status_code=int(out.cplex_status_code),
                cplex_status_name=bytes(out.cplex_status_name).split(b"\0", 1)[0].decode(),
                termination_reason=bytes(out.termination_reason).split(b"\0", 1)[0].decode(),
                num_nonzero_coefficients=int(out.num_nonzero_coefficients),
                node_count=int(out.node_count),
                remaining_node_count=int(out.remaining_node_count),
                resolved_time_limit_seconds=float(out.resolved_time_limit_seconds),
                resolved_relative_mip_gap=float(out.resolved_relative_mip_gap),
                resolved_threads=int(out.resolved_threads),
                resolved_random_seed=int(out.resolved_random_seed),
                resolved_parallel_mode=int(out.resolved_parallel_mode),
                resolved_mip_emphasis=int(out.resolved_mip_emphasis),
                mip_start_objective=None,
                mip_start_accepted=False,
            )
            evaluator_total = sum(evaluator_components.values()) if evaluator_components else float(out.objective_value)
            return MilpEpisodeResult(
                episode_id=instance.episode_id, absolute_start_t=int(instance.absolute_times[0]),
                episode_length=instance.horizon, actions=tuple(actions), encoded_actions=encoded,
                optimization_mode=instance.optimization_mode,
                solver_total_cost=float(out.objective_value), evaluator_total_cost=evaluator_total,
                solver_component_costs=MappingProxyType(solver_components),
                component_costs=MappingProxyType(evaluator_components), step_costs=tuple(steps),
                solver_step_costs=tuple(solver_steps),
                feasible_rate=feasible_rate, solver_stats=stats,
                formulation_metadata=instance.metadata, preparation_time_seconds=0.0,
                extraction_time_seconds=extraction_seconds, validation_time_seconds=validation_seconds,
            )
        finally:
            lib.oran_cplex_free_output(ctypes.byref(out))

    def validate_solution(
        self, instance: MilpInstance, actions: Sequence[PlacementAction],
        selected: NDArray[np.integer[Any]],
        solver_components: Mapping[str, float], solver_total: float, options: MilpSolveOptions,
    ) -> tuple[
        dict[str, float], tuple[Mapping[str, Any], ...],
        tuple[Mapping[str, Any], ...], float,
    ]:
        totals = {name: 0.0 for name in COMPONENT_NAMES}
        steps: list[Mapping[str, Any]] = []
        intended_steps = self._intended_non_latency_steps(instance, selected)
        previous: PlacementAction | None = instance.initial_previous_action
        for ti, (abs_t, action) in enumerate(zip(instance.absolute_times, actions, strict=True)):
            report = self.evaluator.evaluate(int(abs_t), action, previous)
            row = {
                "episode_id": instance.episode_id, "local_t": int(abs_t) % self.episode_length,
                "absolute_t": int(abs_t), "feasible": report.feasible,
                "placement_cost": report.placement_cost,
                "compute_overload_cost": report.compute_overload_cost,
                "bandwidth_violation_cost": report.bandwidth_violation_cost,
                "latency_violation_cost": report.latency_violation_cost,
                "handover_state_transfer_cost": report.handover_state_transfer_cost,
                "handover_sla_cost": report.handover_sla_cost,
                "handover_control_cost": report.handover_control_cost,
                "reconfiguration_cost": report.reconfiguration_cost,
                "total_cost": report.total_cost,
            }
            for name in COMPONENT_NAMES:
                totals[name] += float(row[name])
                if name != "latency_violation_cost":
                    self._assert_close(
                        f"local_t={row['local_t']}:{name}",
                        float(intended_steps[ti][name]), float(row[name]), options,
                    )
            steps.append(MappingProxyType(row))
            previous = action
        evaluator_total = sum(totals.values())
        self._assert_close(
            "native objective/component sum", solver_total,
            sum(float(solver_components[name]) for name in COMPONENT_NAMES), options,
        )
        for name in COMPONENT_NAMES:
            if name == "latency_violation_cost":
                continue
            self._assert_close(name, solver_components[name], totals[name], options)
        if instance.optimization_mode == "mirror_soft":
            allowed = options.validation_tolerance + options.validation_relative_tolerance * max(
                abs(solver_total), abs(evaluator_total)
            )
            if solver_total + allowed < evaluator_total:
                raise MilpValidationError(
                    "upper-PWL incumbent objective is below exact evaluator J: "
                    f"upper={solver_total:.17g}, exact={evaluator_total:.17g}"
                )
        feasible_rate = sum(bool(row["feasible"]) for row in steps) / len(steps)
        if instance.optimization_mode == "hard_service_feasibility":
            service_names = (
                "latency_violation_cost", "bandwidth_violation_cost",
                "compute_overload_cost",
            )
            for row in steps:
                for name in service_names:
                    if abs(float(row[name])) > options.validation_tolerance:
                        raise MilpValidationError(
                            f"hard_service_feasibility returned {name}={row[name]:.17g} "
                            f"at local_t={row['local_t']}"
                        )
                if not bool(row["feasible"]):
                    raise MilpValidationError(
                        "hard_service_feasibility returned an evaluator-infeasible "
                        f"action at local_t={row['local_t']}"
                    )
            if feasible_rate != 1.0:
                raise MilpValidationError(
                    f"hard_service_feasibility feasible rate is {feasible_rate:.17g}, not 1.0"
                )
        exact_steps = tuple(steps)
        return totals, exact_steps, exact_steps, feasible_rate

    def _intended_non_latency_steps(
        self,
        instance: MilpInstance,
        selected: NDArray[np.integer[Any]],
    ) -> tuple[Mapping[str, float], ...]:
        """Reconstruct exact linear/native components for per-step validation."""

        selected_a = np.asarray(selected, dtype=np.int64)
        if selected_a.shape != (instance.horizon, instance.num_rhs):
            raise ValueError("selected actions must have shape [T, R]")
        chosen = np.zeros((instance.horizon, instance.num_actions), dtype=bool)
        for ti in range(instance.horizon):
            chosen[ti, selected_a[ti]] = True
        ho_by_time = [np.flatnonzero(instance.ho_time == ti) for ti in range(instance.horizon)]
        rows: list[Mapping[str, float]] = []
        for ti in range(instance.horizon):
            ids = selected_a[ti]
            values = {name: 0.0 for name in COMPONENT_NAMES}
            values["placement_cost"] = float(instance.placement_cost[ids].sum())
            es_total = np.zeros(instance.num_es)
            rc_total = np.zeros(instance.num_rc)
            fh_total = np.zeros((instance.num_rhs, instance.num_es))
            mh_total = np.zeros((instance.num_es, instance.num_rc))
            direct_total = np.zeros((instance.num_rhs, instance.num_rc))
            for rh, action_id in enumerate(ids):
                action = int(action_id)
                es = int(instance.action_es[action])
                rc = int(instance.action_rc[action])
                es_total[es] += instance.es_load[ti, action]
                rc_total[rc] += instance.rc_load[ti, action]
                if instance.action_gateway[action]:
                    fh_total[rh, es] += instance.fronthaul_load[ti, action]
                    mh_total[es, rc] += instance.midhaul_load[ti, action]
                else:
                    direct_total[rh, rc] += instance.direct_load[ti, action]
            values["compute_overload_cost"] = instance.compute_weight * float(
                (np.maximum(es_total - instance.es_capacity, 0.0) / instance.es_capacity).sum()
                + (np.maximum(rc_total - instance.rc_capacity, 0.0) / instance.rc_capacity).sum()
            )
            values["bandwidth_violation_cost"] = instance.bandwidth_weight * float(
                np.divide(np.maximum(
                    fh_total - instance.fronthaul_capacity.reshape(
                        instance.num_rhs, instance.num_es
                    ),
                    0.0), instance.fronthaul_capacity.reshape(instance.num_rhs, instance.num_es),
                    out=np.zeros_like(fh_total), where=instance.fronthaul_capacity.reshape(instance.num_rhs, instance.num_es) > 0.0).sum()
                + np.divide(np.maximum(
                    mh_total - instance.midhaul_capacity.reshape(
                        instance.num_es, instance.num_rc
                    ),
                    0.0), instance.midhaul_capacity.reshape(instance.num_es, instance.num_rc),
                    out=np.zeros_like(mh_total), where=instance.midhaul_capacity.reshape(instance.num_es, instance.num_rc) > 0.0).sum()
                + np.divide(np.maximum(
                    direct_total - instance.direct_capacity.reshape(
                        instance.num_rhs, instance.num_rc
                    ),
                    0.0), instance.direct_capacity.reshape(instance.num_rhs, instance.num_rc),
                    out=np.zeros_like(direct_total), where=instance.direct_capacity.reshape(instance.num_rhs, instance.num_rc) > 0.0).sum()
            )
            for index in ho_by_time[ti]:
                i = int(index)
                if chosen[ti, int(instance.ho_action_a[i])] and chosen[
                    ti, int(instance.ho_action_b[i])
                ]:
                    values["handover_state_transfer_cost"] += float(instance.ho_state_cost[i])
                    values["handover_sla_cost"] += float(instance.ho_sla_cost[i])
                    values["handover_control_cost"] += float(instance.ho_control_cost[i])
            values["reconfiguration_cost"] = float(instance.fixed_reconf_cost[ids].sum())
            if instance.reconf_enabled and ti > 0:
                for previous_id, current_id in zip(selected_a[ti - 1], ids, strict=True):
                    a, b = int(previous_id), int(current_id)
                    values["reconfiguration_cost"] += self._transition_cost(
                        int(instance.action_split[a]), int(instance.action_es[a]),
                        int(instance.action_rc[a]), int(instance.action_split[b]),
                        int(instance.action_es[b]), int(instance.action_rc[b]),
                    )
            rows.append(MappingProxyType(values))
        return tuple(rows)

    @staticmethod
    def _assert_close(name: str, solver: float, evaluator: float, options: MilpSolveOptions) -> None:
        allowed = options.validation_tolerance + options.validation_relative_tolerance * max(abs(solver), abs(evaluator))
        if abs(solver - evaluator) > allowed:
            raise MilpValidationError(
                f"{name} mismatch: CPLEX={solver:.17g}, evaluator={evaluator:.17g}, "
                f"difference={solver-evaluator:.17g}, tolerance={allowed:.3g}"
            )


def optimality_gap(model_cost: float, oracle_cost: float, epsilon: float = 1e-12) -> float:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return (float(model_cost) - float(oracle_cost)) / (abs(float(oracle_cost)) + epsilon)


def assert_compatible_metadata(model: Mapping[str, Any], oracle: Mapping[str, Any]) -> None:
    required = (
        "constraint_formulation_version", "split_names", "num_rhs", "num_es", "num_rc",
        "episode_length", "dataset_hash", "es_processing_latency_ms",
        "rc_processing_latency_ms", "max_end_to_end_latency_ms", "cost_weights",
        "handover_control_enabled", "rrc_anchor_crossing_weight", "reconfiguration_cost",
    )
    mismatches = [key for key in required if model.get(key) != oracle.get(key)]
    if mismatches:
        raise ValueError(f"incompatible benchmark metadata fields: {mismatches}")


def metadata_json(metadata: Mapping[str, Any]) -> str:
    """Stable JSON representation used by scripts and tests."""

    return json.dumps(dict(metadata), indent=2, sort_keys=True, default=lambda value: vars(value))

"""Project configuration loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
import ast
import math
from typing import Any
import warnings

import numpy as np

from src.utils.data_config import normalize_mobility_profile, normalize_traffic_profile
from src.utils.data_config_validation import ConfigError, normalize_topology_config
from src.pipeline.schema import CostWeights, SplitResourceDemand, TopologyConfig
from src.model.split_semantics import infer_split_names, validate_split_coefficients


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.json"
DEFAULT_DATA_PARAMS_PATH = Path(__file__).resolve().parents[2] / "config" / "mock_data" / "default.yaml"
ALLOWED_TOPOLOGY_PROFILES = {"gateway_only", "direct_only", "mixed", "legacy"}
ALLOWED_TRANSPORT_BUDGET_MODES = {"profile_aware", "weighted", "legacy_independent"}


def load_project_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the legacy merged runtime mapping through the compatibility adapter.

    ``config/default.json`` is now an RL-only v2 configuration.  Existing
    evaluator/trainer call sites still accept the historical merged mapping,
    so this public helper is intentionally a legacy adapter.  New v3 code
    must use :func:`src.utils.config_contract.resolve_run_config` instead.
    """

    from src.utils.config_contract import project_config_compatibility_adapter

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    return project_config_compatibility_adapter(config_path)


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Load the project's simple YAML parameter files without external dependencies.

    Supported syntax is intentionally small: nested mappings using two-space
    indentation, scalar values, booleans, and inline lists such as
    ``[1.0, 2.0, 3.0]``.
    """

    yaml_path = Path(path)
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        line_without_comment = raw_line.split("#", 1)[0].rstrip()
        if not line_without_comment.strip():
            continue
        indent = len(line_without_comment) - len(line_without_comment.lstrip(" "))
        if ":" not in line_without_comment:
            raise ValueError(f"invalid YAML line: {raw_line}")
        key, raw_value = line_without_comment.strip().split(":", 1)
        value_text = raw_value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value_text == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_yaml_scalar(value_text)
    return root


def _parse_yaml_scalar(value_text: str) -> Any:
    if value_text in {"true", "True"}:
        return True
    if value_text in {"false", "False"}:
        return False
    if value_text in {"null", "None"}:
        return None
    try:
        return ast.literal_eval(value_text)
    except (SyntaxError, ValueError):
        return value_text.strip("\"'")


def load_data_generation_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML data-generation parameters and map them to internal config keys."""

    params_path = Path(path) if path is not None else DEFAULT_DATA_PARAMS_PATH
    params = load_simple_yaml(params_path)
    mobility_observation = dict(params.get("mobility_observation", {}))
    if mobility_observation:
        lag_steps = int(mobility_observation.get("lag_steps", 0))
        alpha = float(mobility_observation.get("rho_ewma_alpha", 0.0))
        warmup_steps = int(mobility_observation.get("warmup_steps", 0))
        episode_length = int(params.get("episode_length", params.get("number", {}).get("horizon", 1)))
        if lag_steps < 1:
            raise ValueError("mobility_observation.lag_steps must be >= 1")
        if not 0.0 < alpha < 1.0:
            raise ValueError("mobility_observation.rho_ewma_alpha must be in (0, 1)")
        if warmup_steps < lag_steps:
            raise ValueError("mobility_observation.warmup_steps must be >= lag_steps")
        if warmup_steps >= episode_length:
            raise ValueError("mobility_observation.warmup_steps must be < episode_length")
    normalized_topology = normalize_topology_config(params)
    topology_profile = normalized_topology.topology_profile
    number = params["number"]
    traffic = params["traffic"]
    mobility = params["mobility"]
    state = params["state_transfer"]
    fronthaul = params["fronthaul"]
    backhaul = params["backhaul"]
    direct = params.get("direct_rh_rc", {})
    transport_budget = params.get("rh_transport_budget", {})
    capacity = params["capacity"]
    split = params["split_options"]
    split_names = infer_split_names(int(number["num_splits"]), split.get("names"))
    latency_config_source = "explicit"
    if not all(
        name in split
        for name in ("es_processing_latency_ms", "rc_processing_latency_ms", "max_end_to_end_latency_ms")
    ):
        latency_config_source = "legacy_derived_zero_processing"
        warnings.warn(
            "split_options processing/E2E latency fields are missing; using explicit zero processing "
            "and deriving E2E budgets from existing transport budgets for compatibility",
            UserWarning,
            stacklevel=2,
        )
        split["es_processing_latency_ms"] = [0.0] * len(split_names)
        split["rc_processing_latency_ms"] = [0.0] * len(split_names)
        split["max_end_to_end_latency_ms"] = [
            float(fh) + float(mh)
            for fh, mh in zip(split["max_fronthaul_latency_ms"], split["max_midhaul_latency_ms"], strict=True)
        ]
    latency_fields = (
        "es_processing_latency_ms",
        "rc_processing_latency_ms",
        "max_end_to_end_latency_ms",
    )
    for field in latency_fields:
        values = split[field]
        if len(values) != len(split_names):
            raise ValueError(f"split_options.{field} must contain one value per split")
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
            raise ValueError(f"split_options.{field} must contain finite nonnegative values")
    if any(float(value) <= 0.0 for value in split["max_end_to_end_latency_ms"]):
        raise ValueError("split_options.max_end_to_end_latency_ms must contain positive values")
    for split_id, split_name in enumerate(split_names):
        if split_name.upper() in {"S4G", "S4D"} and float(split["es_processing_latency_ms"][split_id]) != 0.0:
            raise ValueError(f"split_options.es_processing_latency_ms[{split_name}] must be zero")
    allowed_split_names = normalized_topology.allowed_split_names
    ensure_each_rh_has_valid_action = normalized_topology.ensure_each_rh_has_valid_action
    direct_enabled = normalized_topology.direct_rh_rc_enabled
    fronthaul_enabled = normalized_topology.fronthaul_enabled
    transport_budget_mode = str(
        transport_budget.get(
            "allocation_mode",
            "profile_aware"
            if bool(transport_budget.get("enabled", False)) and topology_profile != "legacy"
            else "legacy_independent",
        )
    )
    split_coefficients = {
        "es_compute_per_mbps": split["es_compute_per_mbps"],
        "midhaul_bw_per_mbps": split["midhaul_bw_per_mbps"],
    }
    validate_split_coefficients(split_names, split_coefficients)
    for message in normalized_topology.warnings:
        warnings.warn(message, stacklevel=2)
    return {
        "scope": "handover-aware-orchestration-phase-1",
        "mobility_observation_schema_version": params.get(
            "mobility_observation_schema_version", "legacy_same_step_v1"
        ),
        "mobility_observation": mobility_observation,
        "episode_order": params.get("episode_order", {"training": "sequential", "evaluation": "sequential"}),
        "episode_length": int(params.get("episode_length", number["horizon"])),
        "topology_episode_profiles": dict(params.get("topology_episode_profiles", {})),
        "topology_profile": topology_profile,
        "config_compatibility": {
            "strict": normalized_topology.compatibility.strict,
            "allow_profile_overrides": normalized_topology.compatibility.allow_profile_overrides,
            "warn_deprecated_keys": normalized_topology.compatibility.warn_deprecated_keys,
        },
        "effective_topology": normalized_topology.effective_topology.as_dict(),
        "topology": {
            "num_rhs": int(number["num_rhs"]),
            "num_es": int(number["num_es"]),
            "num_rc": int(number["num_rc"]),
            "horizon": int(number["horizon"]),
            "num_splits": int(number["num_splits"]),
        },
        "mock_data": {
            "seed": int(number["seed"]),
            "traffic_profile": normalize_traffic_profile(
                str(traffic.get("profile", traffic.get("traffic_profile", "smooth")))
            ),
            "base_traffic_mbps": float(traffic["base_traffic_mbps"]),
            "traffic_amplitude_mbps": float(traffic["traffic_amplitude_mbps"]),
            "wave_pi_multiplier": float(traffic["wave_pi_multiplier"]),
            "traffic_phase_rh_step": float(traffic["traffic_phase_rh_step"]),
            "cluster_traffic_bias_mbps": float(traffic["cluster_traffic_bias_mbps"]),
            "traffic_noise_std_mbps": float(traffic["traffic_noise_std_mbps"]),
            "per_rh_baseline_jitter_mbps": float(traffic.get("per_rh_baseline_jitter_mbps", 0.0)),
            "burst_probability": float(traffic.get("burst_probability", 0.0)),
            "burst_magnitude_mbps": float(traffic.get("burst_magnitude_mbps", 0.0)),
            "burst_duration_min": int(traffic.get("burst_duration_min", 1)),
            "burst_duration_max": int(traffic.get("burst_duration_max", 1)),
            "correlated_noise_strength": float(
                traffic.get("correlated_noise_strength", traffic.get("temporal_ar_noise_strength", 0.0))
            ),
            "random_walk_strength_mbps": float(traffic.get("random_walk_strength_mbps", 0.0)),
            "chaotic_periodic_weight": float(traffic.get("chaotic_periodic_weight", 0.35)),
            "regime_switch_probability": float(traffic.get("regime_switch_probability", 0.03)),
            "event_state_multiplier_min": float(traffic.get("event_state_multiplier_min", 1.2)),
            "event_state_multiplier_max": float(traffic.get("event_state_multiplier_max", 2.2)),
            "heavy_tail_burst_probability": float(traffic.get("heavy_tail_burst_probability", 0.02)),
            "heavy_tail_burst_scale_mbps": float(traffic.get("heavy_tail_burst_scale_mbps", 30.0)),
            "cluster_event_probability": float(traffic.get("cluster_event_probability", 0.015)),
            "cluster_event_multiplier_min": float(traffic.get("cluster_event_multiplier_min", 1.1)),
            "cluster_event_multiplier_max": float(traffic.get("cluster_event_multiplier_max", 1.8)),
            "min_traffic_mbps": float(traffic["min_traffic_mbps"]),
            "mean_rate_per_ue_min_mbps": float(traffic["mean_rate_per_ue_min_mbps"]),
            "mean_rate_per_ue_max_mbps": float(traffic["mean_rate_per_ue_max_mbps"]),
            "ue_temporal_noise_std": float(traffic["ue_temporal_noise_std"]),
            "min_ues": int(traffic["min_ues"]),
            "mobility_profile": normalize_mobility_profile(
                str(mobility.get("profile", mobility.get("mobility_profile", traffic.get("profile", "smooth"))))
            ),
            "mobility_scale": float(mobility["mobility_scale"]),
            "min_mobility_clusters": int(mobility["min_mobility_clusters"]),
            "max_mobility_clusters": int(mobility["max_mobility_clusters"]),
            "rh_region_spacing": float(mobility["rh_region_spacing"]),
            "rh_position_jitter": float(mobility["rh_position_jitter"]),
            "base_handover_intensity": float(mobility["base_handover_intensity"]),
            "same_cluster_intensity": float(mobility["same_cluster_intensity"]),
            "circular_neighbor_intensity": float(mobility["circular_neighbor_intensity"]),
            "repeated_pair_intensity": float(mobility["repeated_pair_intensity"]),
            "repeated_pair_pulse_intensity": float(mobility["repeated_pair_pulse_intensity"]),
            "pulse_base": float(mobility["pulse_base"]),
            "pulse_amplitude": float(mobility["pulse_amplitude"]),
            "max_outflow_fraction": float(mobility["max_outflow_fraction"]),
            "small_cell_outflow_allowance": float(mobility["small_cell_outflow_allowance"]),
            "handover_noise_strength": float(mobility.get("handover_noise_strength", 0.0)),
            "handover_burst_probability": float(mobility.get("handover_burst_probability", 0.0)),
            "handover_burst_multiplier": float(mobility.get("handover_burst_multiplier", 1.0)),
            "handover_cluster_jitter": float(mobility.get("handover_cluster_jitter", 0.0)),
            "chaotic_pair_shuffle_probability": float(mobility.get("chaotic_pair_shuffle_probability", 0.05)),
            "chaotic_handover_multiplier_min": float(mobility.get("chaotic_handover_multiplier_min", 0.7)),
            "chaotic_handover_multiplier_max": float(mobility.get("chaotic_handover_multiplier_max", 1.8)),
            "chaotic_cluster_event_probability": float(mobility.get("chaotic_cluster_event_probability", 0.02)),
            "min_ue_denominator": float(state["min_ue_denominator"]),
            "mbit_per_mb": float(state["mbit_per_mb"]),
            "t_fwd_s": float(state["t_fwd_s"]),
            "t_drain_s": float(state["t_drain_s"]),
            "topology_profile": topology_profile,
            "fronthaul_enabled": fronthaul_enabled,
            "rh_es_degree_mode": str(fronthaul.get("rh_es_degree_mode", "bounded")),
            "rh_es_coverage_probability": float(fronthaul.get("rh_es_coverage_probability", 1.0)),
            "rh_es_min_degree": int(fronthaul.get("rh_es_min_degree", 1)),
            "rh_es_max_degree": int(fronthaul.get("rh_es_max_degree", 2)),
            "rh_es_degree_one_probability": float(fronthaul.get("rh_es_degree_one_probability", 0.70)),
            "extra_fronthaul_probability": float(fronthaul.get("extra_link_probability", 0.0)),
            "midhaul_drop_probability": float(backhaul["drop_probability"]),
            "midhaul_es_rc_degree_mode": str(backhaul.get("es_rc_degree_mode", "legacy_drop")),
            "midhaul_es_rc_min_degree": int(backhaul.get("es_rc_min_degree", 1)),
            "midhaul_es_rc_max_degree": int(backhaul.get("es_rc_max_degree", 1)),
            "midhaul_rc_max_degree": None if backhaul.get("rc_max_degree") is None else int(backhaul.get("rc_max_degree")),
            "direct_rh_rc_enabled": direct_enabled,
            "direct_rh_rc_degree_mode": str(direct.get("degree_mode", "pairwise")),
            "direct_rh_rc_min_degree_per_rh": int(direct.get("min_degree_per_rh", 1)),
            "direct_rh_rc_max_degree_per_rh": int(direct.get("max_degree_per_rh", 2)),
            "direct_rh_rc_degree_one_probability": float(direct.get("degree_one_probability", 0.70)),
            "direct_rh_rc_connectivity_probability": float(direct.get("connectivity_probability", 0.0)),
            "direct_rh_rc_min_total_edges": int(direct.get("min_total_edges", 0)),
            "direct_rh_rc_max_total_edges": int(direct.get("max_total_edges", 0)),
            "direct_rh_rc_max_degree_per_rc": None
            if direct.get("max_degree_per_rc") is None
            else int(direct.get("max_degree_per_rc")),
            "direct_rh_rc_selected_rh_strategy": str(direct.get("selected_rh_strategy", "random")),
            "direct_rh_rc_latency_min_ms": float(direct.get("latency_min_ms", fronthaul["latency_min_ms"])),
            "direct_rh_rc_latency_max_ms": float(direct.get("latency_max_ms", fronthaul["latency_max_ms"])),
            "direct_rh_rc_bandwidth_min_mbps": float(
                direct.get("bandwidth_min_mbps", fronthaul["bandwidth_min_mbps"])
            ),
            "direct_rh_rc_bandwidth_max_mbps": float(
                direct.get("bandwidth_max_mbps", fronthaul["bandwidth_max_mbps"])
            ),
            "direct_rh_rc_cost_min": float(direct.get("cost_min", fronthaul["cost_min"])),
            "direct_rh_rc_cost_max": float(direct.get("cost_max", fronthaul["cost_max"])),
            "rh_transport_budget_enabled": bool(transport_budget.get("enabled", False)),
            "rh_transport_budget_max_total_degree_per_rh": int(transport_budget.get("max_total_degree_per_rh", 2)),
            "rh_transport_budget_min_total_degree_per_rh": int(transport_budget.get("min_total_degree_per_rh", 1)),
            "rh_transport_budget_allocation_mode": transport_budget_mode,
            "rh_transport_budget_prefer_direct_fallback": bool(
                transport_budget.get("prefer_direct_fallback", topology_profile == "mixed")
            ),
            "rh_transport_budget_gateway_probability": float(transport_budget.get("gateway_probability", 0.50)),
            "rh_transport_budget_direct_probability": float(transport_budget.get("direct_probability", 1.0)),
            "fronthaul_latency_min_ms": float(fronthaul["latency_min_ms"]),
            "fronthaul_latency_max_ms": float(fronthaul["latency_max_ms"]),
            "midhaul_latency_min_ms": float(backhaul["latency_min_ms"]),
            "midhaul_latency_max_ms": float(backhaul["latency_max_ms"]),
            "infeasible_latency_sentinel_ms": float(params["link"]["infeasible_latency_sentinel_ms"]),
            "fronthaul_bandwidth_min_mbps": float(fronthaul["bandwidth_min_mbps"]),
            "fronthaul_bandwidth_max_mbps": float(fronthaul["bandwidth_max_mbps"]),
            "midhaul_bandwidth_min_mbps": float(backhaul["bandwidth_min_mbps"]),
            "midhaul_bandwidth_max_mbps": float(backhaul["bandwidth_max_mbps"]),
            "infeasible_bandwidth_mbps": float(params["link"]["infeasible_bandwidth_mbps"]),
            "fronthaul_cost_min": float(fronthaul["cost_min"]),
            "fronthaul_cost_max": float(fronthaul["cost_max"]),
            "midhaul_cost_min": float(backhaul["cost_min"]),
            "midhaul_cost_max": float(backhaul["cost_max"]),
            "infeasible_link_cost": float(params["link"]["infeasible_link_cost"]),
            "es_compute_capacity_min": float(capacity["es_cpu_min"]),
            "es_compute_capacity_max": float(capacity["es_cpu_max"]),
            "rc_compute_capacity_min": float(capacity["rc_cpu_min"]),
            "rc_compute_capacity_max": float(capacity["rc_cpu_max"]),
            "es_placement_cost_min": float(capacity["es_base_cost_min"]),
            "es_placement_cost_max": float(capacity["es_base_cost_max"]),
            "rc_placement_cost_min": float(capacity["rc_base_cost_min"]),
            "rc_placement_cost_max": float(capacity["rc_base_cost_max"]),
        },
        "split_demand": {
            "split_names": split_names,
            "es_compute_per_mbps": split["es_compute_per_mbps"],
            "rc_compute_per_mbps": split["rc_compute_per_mbps"],
            "fronthaul_bw_per_mbps": split["fronthaul_bw_per_mbps"],
            "midhaul_bw_per_mbps": split["midhaul_bw_per_mbps"],
            "max_fronthaul_latency_ms": split["max_fronthaul_latency_ms"],
            "max_midhaul_latency_ms": split["max_midhaul_latency_ms"],
            "es_processing_latency_ms": split["es_processing_latency_ms"],
            "rc_processing_latency_ms": split["rc_processing_latency_ms"],
            "max_end_to_end_latency_ms": split["max_end_to_end_latency_ms"],
            "latency_config_source": latency_config_source,
        },
        "split_policy": {
            "allowed_split_names": allowed_split_names,
        },
        "validity": {
            "ensure_each_rh_has_valid_action": ensure_each_rh_has_valid_action,
        },
    }


def topology_from_config(config: dict[str, Any]) -> TopologyConfig:
    topology = config["topology"]
    return TopologyConfig(
        num_rhs=int(topology["num_rhs"]),
        num_es=int(topology["num_es"]),
        num_rc=int(topology["num_rc"]),
        horizon=int(topology["horizon"]),
        num_splits=int(topology["num_splits"]),
    )


def cost_weights_from_config(config: dict[str, Any]) -> CostWeights:
    weights = config["cost_weights"]
    return CostWeights(
        placement=float(weights["placement"]),
        compute_overload=float(weights["compute_overload"]),
        link_violation=float(weights["link_violation"]),
        pdcp_state_transfer=float(weights["pdcp_state_transfer"]),
        rlc_state_transfer=float(weights["rlc_state_transfer"]),
        handover_sla=float(weights["handover_sla"]),
        rrc_anchor_crossing=float(config.get("handover_control", {}).get("rrc_anchor_crossing_weight", 0.0)),
        invalid_action=float(weights["invalid_action"]),
    )


def split_demand_from_config(config: dict[str, Any]) -> SplitResourceDemand:
    split = config["split_demand"]
    return SplitResourceDemand(
        es_compute_per_mbps=np.asarray(split["es_compute_per_mbps"], dtype=np.float64),
        rc_compute_per_mbps=np.asarray(split["rc_compute_per_mbps"], dtype=np.float64),
        fronthaul_bw_per_mbps=np.asarray(split["fronthaul_bw_per_mbps"], dtype=np.float64),
        midhaul_bw_per_mbps=np.asarray(split["midhaul_bw_per_mbps"], dtype=np.float64),
        max_fronthaul_latency_ms=np.asarray(split["max_fronthaul_latency_ms"], dtype=np.float64),
        max_midhaul_latency_ms=np.asarray(split["max_midhaul_latency_ms"], dtype=np.float64),
        es_processing_latency_ms=np.asarray(split["es_processing_latency_ms"], dtype=np.float64),
        rc_processing_latency_ms=np.asarray(split["rc_processing_latency_ms"], dtype=np.float64),
        max_end_to_end_latency_ms=np.asarray(split["max_end_to_end_latency_ms"], dtype=np.float64),
        split_names=tuple(str(name) for name in split["split_names"]),
    )

"""Configuration dataclass for mock dataset generation."""

from __future__ import annotations

from dataclasses import dataclass

from src.pipeline.schema import TopologyConfig


ALLOWED_PROFILES = {"smooth", "realistic", "chaotic"}
TRAFFIC_PROFILE_ALIASES = {
    "background": "smooth",
    "residential_low": "realistic",
    "mixed_urban": "realistic",
    "road_burst": "chaotic",
}
MOBILITY_PROFILE_ALIASES = {
    "openireland_calibrated": "realistic",
}
ALLOWED_TOPOLOGY_PROFILES = {"gateway_only", "direct_only", "mixed", "legacy"}
ALLOWED_TRANSPORT_BUDGET_MODES = {"profile_aware", "weighted", "legacy_independent"}


@dataclass(frozen=True)
class MockDataConfig:
    """Configuration for reproducible RH-level mock episodes."""

    topology: TopologyConfig
    seed: int
    topology_profile: str
    allowed_split_names: tuple[str, ...]
    ensure_each_rh_has_valid_action: bool
    traffic_profile: str
    base_traffic_mbps: float
    traffic_amplitude_mbps: float
    wave_pi_multiplier: float
    traffic_phase_rh_step: float
    cluster_traffic_bias_mbps: float
    traffic_noise_std_mbps: float
    per_rh_baseline_jitter_mbps: float
    burst_probability: float
    burst_magnitude_mbps: float
    burst_duration_min: int
    burst_duration_max: int
    correlated_noise_strength: float
    random_walk_strength_mbps: float
    chaotic_periodic_weight: float
    regime_switch_probability: float
    event_state_multiplier_min: float
    event_state_multiplier_max: float
    heavy_tail_burst_probability: float
    heavy_tail_burst_scale_mbps: float
    cluster_event_probability: float
    cluster_event_multiplier_min: float
    cluster_event_multiplier_max: float
    min_traffic_mbps: float
    mean_rate_per_ue_min_mbps: float
    mean_rate_per_ue_max_mbps: float
    ue_temporal_noise_std: float
    min_ues: int
    mobility_profile: str
    mobility_scale: float
    min_mobility_clusters: int
    max_mobility_clusters: int
    rh_region_spacing: float
    rh_position_jitter: float
    base_handover_intensity: float
    same_cluster_intensity: float
    circular_neighbor_intensity: float
    repeated_pair_intensity: float
    repeated_pair_pulse_intensity: float
    pulse_base: float
    pulse_amplitude: float
    max_outflow_fraction: float
    small_cell_outflow_allowance: float
    handover_noise_strength: float
    handover_burst_probability: float
    handover_burst_multiplier: float
    handover_cluster_jitter: float
    chaotic_pair_shuffle_probability: float
    chaotic_handover_multiplier_min: float
    chaotic_handover_multiplier_max: float
    chaotic_cluster_event_probability: float
    min_ue_denominator: float
    mbit_per_mb: float
    t_fwd_s: float
    t_drain_s: float
    rh_es_degree_mode: str
    fronthaul_enabled: bool
    rh_es_coverage_probability: float
    rh_es_min_degree: int
    rh_es_max_degree: int
    rh_es_degree_one_probability: float
    extra_fronthaul_probability: float
    midhaul_drop_probability: float
    midhaul_es_rc_degree_mode: str
    midhaul_es_rc_min_degree: int
    midhaul_es_rc_max_degree: int
    midhaul_rc_max_degree: int | None
    direct_rh_rc_enabled: bool
    direct_rh_rc_degree_mode: str
    direct_rh_rc_min_degree_per_rh: int
    direct_rh_rc_max_degree_per_rh: int
    direct_rh_rc_degree_one_probability: float
    direct_rh_rc_connectivity_probability: float
    direct_rh_rc_min_total_edges: int
    direct_rh_rc_max_total_edges: int
    direct_rh_rc_max_degree_per_rc: int | None
    direct_rh_rc_selected_rh_strategy: str
    direct_rh_rc_latency_min_ms: float
    direct_rh_rc_latency_max_ms: float
    direct_rh_rc_bandwidth_min_mbps: float
    direct_rh_rc_bandwidth_max_mbps: float
    direct_rh_rc_cost_min: float
    direct_rh_rc_cost_max: float
    rh_transport_budget_enabled: bool
    rh_transport_budget_max_total_degree_per_rh: int
    rh_transport_budget_min_total_degree_per_rh: int
    rh_transport_budget_allocation_mode: str
    rh_transport_budget_prefer_direct_fallback: bool
    rh_transport_budget_gateway_probability: float
    rh_transport_budget_direct_probability: float
    fronthaul_latency_min_ms: float
    fronthaul_latency_max_ms: float
    midhaul_latency_min_ms: float
    midhaul_latency_max_ms: float
    infeasible_latency_sentinel_ms: float
    fronthaul_bandwidth_min_mbps: float
    fronthaul_bandwidth_max_mbps: float
    midhaul_bandwidth_min_mbps: float
    midhaul_bandwidth_max_mbps: float
    infeasible_bandwidth_mbps: float
    fronthaul_cost_min: float
    fronthaul_cost_max: float
    midhaul_cost_min: float
    midhaul_cost_max: float
    infeasible_link_cost: float
    es_compute_capacity_min: float
    es_compute_capacity_max: float
    rc_compute_capacity_min: float
    rc_compute_capacity_max: float
    es_placement_cost_min: float
    es_placement_cost_max: float
    rc_placement_cost_min: float
    rc_placement_cost_max: float

    def __post_init__(self) -> None:
        if self.traffic_profile not in ALLOWED_PROFILES:
            raise ValueError(f"traffic_profile must be one of {sorted(ALLOWED_PROFILES)}")
        if self.mobility_profile not in ALLOWED_PROFILES:
            raise ValueError(f"mobility_profile must be one of {sorted(ALLOWED_PROFILES)}")
        if self.topology_profile not in ALLOWED_TOPOLOGY_PROFILES:
            raise ValueError(f"topology_profile must be one of {sorted(ALLOWED_TOPOLOGY_PROFILES)}")
        if self.rh_es_degree_mode not in {"legacy", "bounded", "episode_profile"}:
            raise ValueError("rh_es_degree_mode must be 'legacy', 'bounded', or 'episode_profile'")
        if not 0.0 <= self.rh_es_coverage_probability <= 1.0:
            raise ValueError("rh_es_coverage_probability must be in [0, 1]")
        if self.rh_es_min_degree <= 0 or self.rh_es_max_degree <= 0:
            raise ValueError("RH->ES degree bounds must be positive")
        if self.rh_es_min_degree > self.rh_es_max_degree:
            raise ValueError("rh_es_min_degree cannot exceed rh_es_max_degree")
        if not 0.0 <= self.rh_es_degree_one_probability <= 1.0:
            raise ValueError("rh_es_degree_one_probability must be in [0, 1]")
        if self.midhaul_es_rc_degree_mode not in {"legacy_drop", "bounded", "episode_profile"}:
            raise ValueError(
                "midhaul_es_rc_degree_mode must be 'legacy_drop', 'bounded', or 'episode_profile'"
            )
        if self.midhaul_es_rc_min_degree < 0 or self.midhaul_es_rc_max_degree < 0:
            raise ValueError("ES->RC degree bounds must be non-negative")
        if self.midhaul_es_rc_min_degree > self.midhaul_es_rc_max_degree:
            raise ValueError("midhaul_es_rc_min_degree cannot exceed midhaul_es_rc_max_degree")
        if self.midhaul_rc_max_degree is not None and self.midhaul_rc_max_degree < 0:
            raise ValueError("midhaul_rc_max_degree must be non-negative when provided")
        if self.direct_rh_rc_degree_mode not in {"bounded", "pairwise", "global_sparse"}:
            raise ValueError("direct_rh_rc_degree_mode must be 'bounded', 'pairwise', or 'global_sparse'")
        if self.direct_rh_rc_degree_mode == "global_sparse":
            if self.direct_rh_rc_min_total_edges < 0 or self.direct_rh_rc_max_total_edges < 0:
                raise ValueError("direct RH->RC global sparse edge totals must be non-negative")
            if self.direct_rh_rc_min_total_edges > self.direct_rh_rc_max_total_edges:
                raise ValueError("direct_rh_rc_min_total_edges cannot exceed direct_rh_rc_max_total_edges")
            if self.direct_rh_rc_max_degree_per_rh < 0:
                raise ValueError("direct_rh_rc_max_degree_per_rh must be non-negative in global_sparse mode")
            if self.direct_rh_rc_max_degree_per_rc is not None and self.direct_rh_rc_max_degree_per_rc < 0:
                raise ValueError("direct_rh_rc_max_degree_per_rc must be non-negative when provided")
        else:
            if self.direct_rh_rc_min_degree_per_rh <= 0 or self.direct_rh_rc_max_degree_per_rh <= 0:
                raise ValueError("direct RH->RC degree bounds must be positive")
            if self.direct_rh_rc_min_degree_per_rh > self.direct_rh_rc_max_degree_per_rh:
                raise ValueError("direct_rh_rc_min_degree_per_rh cannot exceed direct_rh_rc_max_degree_per_rh")
            if not 0.0 <= self.direct_rh_rc_degree_one_probability <= 1.0:
                raise ValueError("direct_rh_rc_degree_one_probability must be in [0, 1]")
        if self.rh_transport_budget_allocation_mode not in ALLOWED_TRANSPORT_BUDGET_MODES:
            raise ValueError(
                "rh_transport_budget_allocation_mode must be one of "
                f"{sorted(ALLOWED_TRANSPORT_BUDGET_MODES)}"
            )
        if self.rh_transport_budget_min_total_degree_per_rh < 0:
            raise ValueError("rh_transport_budget_min_total_degree_per_rh must be non-negative")
        if self.rh_transport_budget_max_total_degree_per_rh < 0:
            raise ValueError("rh_transport_budget_max_total_degree_per_rh must be non-negative")
        if self.rh_transport_budget_min_total_degree_per_rh > self.rh_transport_budget_max_total_degree_per_rh:
            raise ValueError(
                "rh_transport_budget_min_total_degree_per_rh cannot exceed "
                "rh_transport_budget_max_total_degree_per_rh"
            )
        if not 0.0 <= self.rh_transport_budget_gateway_probability <= 1.0:
            raise ValueError("rh_transport_budget_gateway_probability must be in [0, 1]")
        if not 0.0 <= self.rh_transport_budget_direct_probability <= 1.0:
            raise ValueError("rh_transport_budget_direct_probability must be in [0, 1]")

    @classmethod
    def from_project_config(
        cls,
        project_config: dict,
        *,
        seed: int | None = None,
        topology: TopologyConfig | None = None,
    ) -> "MockDataConfig":
        mock = project_config["mock_data"]
        split_policy = project_config.get("split_policy", {})
        validity = project_config.get("validity", {})
        return cls(
            topology=topology or TopologyConfig(
                num_rhs=int(project_config["topology"]["num_rhs"]),
                num_es=int(project_config["topology"]["num_es"]),
                num_rc=int(project_config["topology"]["num_rc"]),
                horizon=int(project_config["topology"]["horizon"]),
                num_splits=int(project_config["topology"]["num_splits"]),
            ),
            seed=int(mock["seed"] if seed is None else seed),
            topology_profile=str(project_config.get("topology_profile", mock.get("topology_profile", "legacy"))),
            allowed_split_names=tuple(split_policy.get("allowed_split_names", project_config["split_demand"]["split_names"])),
            ensure_each_rh_has_valid_action=bool(validity.get("ensure_each_rh_has_valid_action", True)),
            traffic_profile=normalize_traffic_profile(str(mock.get("traffic_profile", "smooth"))),
            base_traffic_mbps=float(mock["base_traffic_mbps"]),
            traffic_amplitude_mbps=float(mock["traffic_amplitude_mbps"]),
            wave_pi_multiplier=float(mock["wave_pi_multiplier"]),
            traffic_phase_rh_step=float(mock["traffic_phase_rh_step"]),
            cluster_traffic_bias_mbps=float(mock["cluster_traffic_bias_mbps"]),
            traffic_noise_std_mbps=float(mock["traffic_noise_std_mbps"]),
            per_rh_baseline_jitter_mbps=float(mock.get("per_rh_baseline_jitter_mbps", 0.0)),
            burst_probability=float(mock.get("burst_probability", 0.0)),
            burst_magnitude_mbps=float(mock.get("burst_magnitude_mbps", 0.0)),
            burst_duration_min=int(mock.get("burst_duration_min", 1)),
            burst_duration_max=int(mock.get("burst_duration_max", 1)),
            correlated_noise_strength=float(mock.get("correlated_noise_strength", 0.0)),
            random_walk_strength_mbps=float(mock.get("random_walk_strength_mbps", 0.0)),
            chaotic_periodic_weight=float(mock.get("chaotic_periodic_weight", 0.35)),
            regime_switch_probability=float(mock.get("regime_switch_probability", 0.03)),
            event_state_multiplier_min=float(mock.get("event_state_multiplier_min", 1.2)),
            event_state_multiplier_max=float(mock.get("event_state_multiplier_max", 2.2)),
            heavy_tail_burst_probability=float(mock.get("heavy_tail_burst_probability", 0.02)),
            heavy_tail_burst_scale_mbps=float(mock.get("heavy_tail_burst_scale_mbps", 30.0)),
            cluster_event_probability=float(mock.get("cluster_event_probability", 0.015)),
            cluster_event_multiplier_min=float(mock.get("cluster_event_multiplier_min", 1.1)),
            cluster_event_multiplier_max=float(mock.get("cluster_event_multiplier_max", 1.8)),
            min_traffic_mbps=float(mock["min_traffic_mbps"]),
            mean_rate_per_ue_min_mbps=float(mock["mean_rate_per_ue_min_mbps"]),
            mean_rate_per_ue_max_mbps=float(mock["mean_rate_per_ue_max_mbps"]),
            ue_temporal_noise_std=float(mock["ue_temporal_noise_std"]),
            min_ues=int(mock["min_ues"]),
            mobility_profile=normalize_mobility_profile(
                str(mock.get("mobility_profile", mock.get("traffic_profile", "smooth")))
            ),
            mobility_scale=float(mock["mobility_scale"]),
            min_mobility_clusters=int(mock["min_mobility_clusters"]),
            max_mobility_clusters=int(mock["max_mobility_clusters"]),
            rh_region_spacing=float(mock["rh_region_spacing"]),
            rh_position_jitter=float(mock["rh_position_jitter"]),
            base_handover_intensity=float(mock["base_handover_intensity"]),
            same_cluster_intensity=float(mock["same_cluster_intensity"]),
            circular_neighbor_intensity=float(mock["circular_neighbor_intensity"]),
            repeated_pair_intensity=float(mock["repeated_pair_intensity"]),
            repeated_pair_pulse_intensity=float(mock["repeated_pair_pulse_intensity"]),
            pulse_base=float(mock["pulse_base"]),
            pulse_amplitude=float(mock["pulse_amplitude"]),
            max_outflow_fraction=float(mock["max_outflow_fraction"]),
            small_cell_outflow_allowance=float(mock["small_cell_outflow_allowance"]),
            handover_noise_strength=float(mock.get("handover_noise_strength", 0.0)),
            handover_burst_probability=float(mock.get("handover_burst_probability", 0.0)),
            handover_burst_multiplier=float(mock.get("handover_burst_multiplier", 1.0)),
            handover_cluster_jitter=float(mock.get("handover_cluster_jitter", 0.0)),
            chaotic_pair_shuffle_probability=float(mock.get("chaotic_pair_shuffle_probability", 0.05)),
            chaotic_handover_multiplier_min=float(mock.get("chaotic_handover_multiplier_min", 0.7)),
            chaotic_handover_multiplier_max=float(mock.get("chaotic_handover_multiplier_max", 1.8)),
            chaotic_cluster_event_probability=float(mock.get("chaotic_cluster_event_probability", 0.02)),
            min_ue_denominator=float(mock["min_ue_denominator"]),
            mbit_per_mb=float(mock["mbit_per_mb"]),
            t_fwd_s=float(mock["t_fwd_s"]),
            t_drain_s=float(mock["t_drain_s"]),
            rh_es_degree_mode=str(mock.get("rh_es_degree_mode", "bounded")),
            fronthaul_enabled=bool(mock.get("fronthaul_enabled", True)),
            rh_es_coverage_probability=float(mock.get("rh_es_coverage_probability", 1.0)),
            rh_es_min_degree=int(mock.get("rh_es_min_degree", 1)),
            rh_es_max_degree=int(mock.get("rh_es_max_degree", 2)),
            rh_es_degree_one_probability=float(mock.get("rh_es_degree_one_probability", 0.70)),
            extra_fronthaul_probability=float(mock["extra_fronthaul_probability"]),
            midhaul_drop_probability=float(mock["midhaul_drop_probability"]),
            midhaul_es_rc_degree_mode=str(mock.get("midhaul_es_rc_degree_mode", "legacy_drop")),
            midhaul_es_rc_min_degree=int(mock.get("midhaul_es_rc_min_degree", 1)),
            midhaul_es_rc_max_degree=int(mock.get("midhaul_es_rc_max_degree", 1)),
            midhaul_rc_max_degree=mock.get("midhaul_rc_max_degree"),
            direct_rh_rc_enabled=bool(mock.get("direct_rh_rc_enabled", False)),
            direct_rh_rc_degree_mode=str(mock.get("direct_rh_rc_degree_mode", "pairwise")),
            direct_rh_rc_min_degree_per_rh=int(mock.get("direct_rh_rc_min_degree_per_rh", 1)),
            direct_rh_rc_max_degree_per_rh=int(mock.get("direct_rh_rc_max_degree_per_rh", 2)),
            direct_rh_rc_degree_one_probability=float(mock.get("direct_rh_rc_degree_one_probability", 0.70)),
            direct_rh_rc_connectivity_probability=float(mock.get("direct_rh_rc_connectivity_probability", 0.0)),
            direct_rh_rc_min_total_edges=int(mock.get("direct_rh_rc_min_total_edges", 0)),
            direct_rh_rc_max_total_edges=int(mock.get("direct_rh_rc_max_total_edges", 0)),
            direct_rh_rc_max_degree_per_rc=mock.get("direct_rh_rc_max_degree_per_rc"),
            direct_rh_rc_selected_rh_strategy=str(mock.get("direct_rh_rc_selected_rh_strategy", "random")),
            direct_rh_rc_latency_min_ms=float(mock.get("direct_rh_rc_latency_min_ms", mock["fronthaul_latency_min_ms"])),
            direct_rh_rc_latency_max_ms=float(mock.get("direct_rh_rc_latency_max_ms", mock["fronthaul_latency_max_ms"])),
            direct_rh_rc_bandwidth_min_mbps=float(
                mock.get("direct_rh_rc_bandwidth_min_mbps", mock["fronthaul_bandwidth_min_mbps"])
            ),
            direct_rh_rc_bandwidth_max_mbps=float(
                mock.get("direct_rh_rc_bandwidth_max_mbps", mock["fronthaul_bandwidth_max_mbps"])
            ),
            direct_rh_rc_cost_min=float(mock.get("direct_rh_rc_cost_min", mock["fronthaul_cost_min"])),
            direct_rh_rc_cost_max=float(mock.get("direct_rh_rc_cost_max", mock["fronthaul_cost_max"])),
            rh_transport_budget_enabled=bool(mock.get("rh_transport_budget_enabled", False)),
            rh_transport_budget_max_total_degree_per_rh=int(mock.get("rh_transport_budget_max_total_degree_per_rh", 2)),
            rh_transport_budget_min_total_degree_per_rh=int(mock.get("rh_transport_budget_min_total_degree_per_rh", 1)),
            rh_transport_budget_allocation_mode=str(mock.get("rh_transport_budget_allocation_mode", "legacy_independent")),
            rh_transport_budget_prefer_direct_fallback=bool(
                mock.get("rh_transport_budget_prefer_direct_fallback", mock.get("topology_profile") == "mixed")
            ),
            rh_transport_budget_gateway_probability=float(mock.get("rh_transport_budget_gateway_probability", 0.50)),
            rh_transport_budget_direct_probability=float(mock.get("rh_transport_budget_direct_probability", 1.0)),
            fronthaul_latency_min_ms=float(mock["fronthaul_latency_min_ms"]),
            fronthaul_latency_max_ms=float(mock["fronthaul_latency_max_ms"]),
            midhaul_latency_min_ms=float(mock["midhaul_latency_min_ms"]),
            midhaul_latency_max_ms=float(mock["midhaul_latency_max_ms"]),
            infeasible_latency_sentinel_ms=float(mock["infeasible_latency_sentinel_ms"]),
            fronthaul_bandwidth_min_mbps=float(mock["fronthaul_bandwidth_min_mbps"]),
            fronthaul_bandwidth_max_mbps=float(mock["fronthaul_bandwidth_max_mbps"]),
            midhaul_bandwidth_min_mbps=float(mock["midhaul_bandwidth_min_mbps"]),
            midhaul_bandwidth_max_mbps=float(mock["midhaul_bandwidth_max_mbps"]),
            infeasible_bandwidth_mbps=float(mock["infeasible_bandwidth_mbps"]),
            fronthaul_cost_min=float(mock["fronthaul_cost_min"]),
            fronthaul_cost_max=float(mock["fronthaul_cost_max"]),
            midhaul_cost_min=float(mock["midhaul_cost_min"]),
            midhaul_cost_max=float(mock["midhaul_cost_max"]),
            infeasible_link_cost=float(mock["infeasible_link_cost"]),
            es_compute_capacity_min=float(mock["es_compute_capacity_min"]),
            es_compute_capacity_max=float(mock["es_compute_capacity_max"]),
            rc_compute_capacity_min=float(mock["rc_compute_capacity_min"]),
            rc_compute_capacity_max=float(mock["rc_compute_capacity_max"]),
            es_placement_cost_min=float(mock["es_placement_cost_min"]),
            es_placement_cost_max=float(mock["es_placement_cost_max"]),
            rc_placement_cost_min=float(mock["rc_placement_cost_min"]),
            rc_placement_cost_max=float(mock["rc_placement_cost_max"]),
        )


def normalize_traffic_profile(profile: str) -> str:
    """Map dataset-specific traffic labels to canonical generator profiles."""

    normalized = str(profile)
    return TRAFFIC_PROFILE_ALIASES.get(normalized, normalized)


def normalize_mobility_profile(profile: str) -> str:
    """Map dataset-specific mobility labels to canonical generator profiles."""

    normalized = str(profile)
    return MOBILITY_PROFILE_ALIASES.get(normalized, normalized)

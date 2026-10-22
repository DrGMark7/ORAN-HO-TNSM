"""Centralized topology-intent normalization and validation for mock data YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.model.split_semantics import infer_split_names, is_direct_rc_split, is_gateway_split, semantics_for_split


ALLOWED_TOPOLOGY_PROFILES = {"gateway_only", "direct_only", "mixed", "mixed_episode_profiles", "legacy"}
ALLOWED_TRANSPORT_BUDGET_MODES = {"profile_aware", "weighted", "legacy_independent"}
ALLOWED_DIRECT_RH_RC_DEGREE_MODES = {"bounded", "pairwise", "global_sparse"}
ALLOWED_BACKHAUL_DEGREE_MODES = {"legacy_drop", "bounded", "episode_profile"}


class ConfigError(ValueError):
    """Raised when YAML configuration is semantically inconsistent."""


@dataclass(frozen=True)
class ConfigCompatibility:
    strict: bool
    allow_profile_overrides: bool
    warn_deprecated_keys: bool


@dataclass(frozen=True)
class EffectiveTopologySummary:
    profile: str
    gateway_family_active: bool
    direct_family_active: bool
    allowed_splits: tuple[str, ...]
    gateway_splits: tuple[str, ...]
    direct_splits: tuple[str, ...]
    fronthaul_enabled: bool
    direct_rh_rc_enabled: bool
    rh_transport_budget_enabled: bool
    max_total_degree_per_rh: int
    notes: tuple[str, ...]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizedTopologyConfig:
    topology_profile: str
    compatibility: ConfigCompatibility
    allowed_split_names: tuple[str, ...]
    fronthaul_enabled: bool
    direct_rh_rc_enabled: bool
    ensure_each_rh_has_valid_action: bool
    effective_topology: EffectiveTopologySummary
    warnings: tuple[str, ...]


def normalize_topology_config(params: dict[str, Any]) -> NormalizedTopologyConfig:
    topology_profile_present = "topology_profile" in params
    topology_profile = str(params.get("topology_profile", "legacy"))
    if topology_profile not in ALLOWED_TOPOLOGY_PROFILES:
        raise ConfigError(f"topology_profile must be one of {sorted(ALLOWED_TOPOLOGY_PROFILES)}")

    compatibility = _resolve_compatibility(
        params.get("config_compatibility", {}),
        topology_profile=topology_profile,
        topology_profile_present=topology_profile_present,
    )
    number = params["number"]
    num_rhs = int(number["num_rhs"])
    num_es = int(number["num_es"])
    num_rc = int(number["num_rc"])
    split_options = params["split_options"]
    split_names = infer_split_names(int(number["num_splits"]), split_options.get("names"))
    _validate_split_options(split_options, int(number["num_splits"]), split_names)

    split_policy = params.get("split_policy", {})
    direct = params.get("direct_rh_rc", {})
    fronthaul = params["fronthaul"]
    backhaul = params["backhaul"]
    transport_budget = params.get("rh_transport_budget", {})
    validity = params.get("validity", {})
    transport_budget_enabled = bool(transport_budget.get("enabled", False))
    transport_budget_mode = str(
        transport_budget.get(
            "allocation_mode",
            "profile_aware" if transport_budget_enabled and topology_profile != "legacy" else "legacy_independent",
        )
    )
    prefer_direct_fallback = bool(
        transport_budget.get("prefer_direct_fallback", topology_profile in {"mixed", "mixed_episode_profiles"})
    )

    allowed_split_names = tuple(
        str(name) for name in split_policy.get("allowed_split_names", _default_allowed_splits(topology_profile, split_names))
    )
    invalid_allowed = [name for name in allowed_split_names if name not in split_names]
    if invalid_allowed:
        raise ConfigError(
            "split_policy.allowed_split_names contains unknown split names "
            f"{invalid_allowed}; known split names are {list(split_names)}"
        )
    if not allowed_split_names:
        raise ConfigError("split_policy.allowed_split_names cannot be empty")
    for split_name in split_names:
        semantics_for_split(split_name)

    fronthaul_enabled = bool(fronthaul.get("enabled", _default_fronthaul_enabled(topology_profile)))
    direct_enabled = bool(direct.get("enabled", _default_direct_enabled(topology_profile, bool(direct))))
    ensure_valid = bool(validity.get("ensure_each_rh_has_valid_action", True))

    gateway_splits = tuple(name for name in split_names if is_gateway_split(name, split_names))
    direct_splits = tuple(name for name in split_names if is_direct_rc_split(name, split_names))
    allowed_gateway_splits = tuple(name for name in allowed_split_names if name in gateway_splits)
    allowed_direct_splits = tuple(name for name in allowed_split_names if name in direct_splits)

    warnings: list[str] = []
    notes: list[str] = []
    errors: list[str] = []

    def warning_message(message: str) -> str:
        return message.replace("ConfigError:", "ConfigWarning:", 1) if message.startswith("ConfigError:") else message

    def issue(message: str, *, strict_only: bool = True, always_error: bool = False) -> None:
        if always_error or (compatibility.strict and strict_only):
            errors.append(message)
        else:
            warnings.append(warning_message(message))

    def issue_profile_override(message: str) -> None:
        if compatibility.allow_profile_overrides:
            warnings.append(warning_message(message))
        else:
            errors.append(message)

    if compatibility.warn_deprecated_keys:
        _collect_deprecation_warnings(
            warnings,
            params,
            fronthaul=fronthaul,
            direct=direct,
            backhaul=backhaul,
            transport_budget=transport_budget,
        )

    _validate_direct_rh_rc_shape(direct, num_rhs=num_rhs, num_rc=num_rc)
    _validate_backhaul_shape(backhaul, num_es=num_es, num_rc=num_rc)

    if "S4D" in allowed_split_names and not direct_enabled:
        issue(
            "ConfigError: S4D is allowed by split_policy, but direct_rh_rc.enabled=false.\n"
            "Either remove S4D from split_policy.allowed_split_names or enable direct_rh_rc."
        )
    if allowed_gateway_splits and not fronthaul_enabled:
        issue(
            "ConfigError: gateway splits are allowed by split_policy, but fronthaul.enabled=false.\n"
            "Either remove gateway splits from split_policy.allowed_split_names or enable fronthaul."
        )
    if prefer_direct_fallback and (
        not direct_enabled or "S4D" not in allowed_split_names
    ):
        issue(
            "ConfigError: prefer_direct_fallback=true requires direct_rh_rc.enabled=true and "
            "S4D in split_policy.allowed_split_names."
        )
    if "direct_probability" in transport_budget and float(transport_budget.get("direct_probability", 1.0)) > 0.0 and not direct_enabled:
        issue(
            "ConfigError: rh_transport_budget.direct_probability > 0 but direct_rh_rc.enabled=false.\n"
            "Set direct_probability=0.0 or enable direct_rh_rc."
        )
    if "gateway_probability" in transport_budget and float(transport_budget.get("gateway_probability", 0.50)) > 0.0 and not fronthaul_enabled:
        issue(
            "ConfigError: rh_transport_budget.gateway_probability > 0 but fronthaul.enabled=false.\n"
            "Set gateway_probability=0.0 or enable fronthaul."
        )

    if topology_profile == "direct_only":
        if tuple(allowed_split_names) != ("S4D",):
            issue_profile_override(
                "ConfigError: topology_profile='direct_only' requires split_policy.allowed_split_names=['S4D']."
            )
        if not direct_enabled:
            issue_profile_override("ConfigError: topology_profile='direct_only' requires direct_rh_rc.enabled=true.")
        if fronthaul_enabled:
            issue_profile_override("ConfigError: topology_profile='direct_only' expects fronthaul.enabled=false.")
    elif topology_profile == "gateway_only":
        if "S4D" in allowed_split_names:
            issue_profile_override("ConfigError: topology_profile='gateway_only' must exclude S4D from split_policy.")
        if direct_enabled:
            issue_profile_override("ConfigError: topology_profile='gateway_only' expects direct_rh_rc.enabled=false.")
        if not fronthaul_enabled:
            issue_profile_override("ConfigError: topology_profile='gateway_only' requires fronthaul.enabled=true.")
    elif topology_profile in {"mixed", "mixed_episode_profiles"}:
        gateway_family_active = fronthaul_enabled and bool(allowed_gateway_splits)
        direct_family_active = direct_enabled and "S4D" in allowed_split_names
        if not gateway_family_active and not direct_family_active:
            raise ConfigError(
                "ConfigError: topology_profile='mixed' disables both gateway and direct families. "
                "Enable at least one family."
            )
        if not gateway_family_active:
            message = (
                "ConfigError: topology_profile='mixed' degenerates to direct-only behavior. "
                "Use topology_profile='direct_only' or enable a gateway family."
            )
            notes.append("mixed profile currently behaves like direct_only")
            issue_profile_override(message) if compatibility.strict else issue(message, strict_only=False)
        if not direct_family_active:
            message = (
                "ConfigError: topology_profile='mixed' degenerates to gateway-only behavior. "
                "Use topology_profile='gateway_only' or enable a direct family."
            )
            notes.append("mixed profile currently behaves like gateway_only")
            issue_profile_override(message) if compatibility.strict else issue(message, strict_only=False)

    if transport_budget_enabled:
        if transport_budget_mode not in ALLOWED_TRANSPORT_BUDGET_MODES:
            raise ConfigError(
                "rh_transport_budget.allocation_mode must be one of "
                f"{sorted(ALLOWED_TRANSPORT_BUDGET_MODES)}"
            )
        min_total = int(transport_budget.get("min_total_degree_per_rh", 1))
        max_total = int(transport_budget.get("max_total_degree_per_rh", 2))
        if max_total < min_total:
            raise ConfigError(
                "ConfigError: rh_transport_budget.max_total_degree_per_rh must be >= "
                "rh_transport_budget.min_total_degree_per_rh."
            )
        if ensure_valid and max_total < 1:
            raise ConfigError(
                "ConfigError: rh_transport_budget.max_total_degree_per_rh must be >= 1 when "
                "validity.ensure_each_rh_has_valid_action=true."
            )

    if errors:
        raise ConfigError(errors[0] if len(errors) == 1 else "\n".join(errors))

    effective_summary = EffectiveTopologySummary(
        profile=topology_profile,
        gateway_family_active=fronthaul_enabled and bool(allowed_gateway_splits),
        direct_family_active=direct_enabled and bool(allowed_direct_splits),
        allowed_splits=allowed_split_names,
        gateway_splits=allowed_gateway_splits,
        direct_splits=allowed_direct_splits,
        fronthaul_enabled=fronthaul_enabled,
        direct_rh_rc_enabled=direct_enabled,
        rh_transport_budget_enabled=transport_budget_enabled,
        max_total_degree_per_rh=int(transport_budget.get("max_total_degree_per_rh", 2)),
        notes=tuple(notes),
        warnings=tuple(warnings),
    )
    return NormalizedTopologyConfig(
        topology_profile=topology_profile,
        compatibility=compatibility,
        allowed_split_names=allowed_split_names,
        fronthaul_enabled=fronthaul_enabled,
        direct_rh_rc_enabled=direct_enabled,
        ensure_each_rh_has_valid_action=ensure_valid,
        effective_topology=effective_summary,
        warnings=tuple(warnings),
    )


def _resolve_compatibility(
    config: dict[str, Any],
    *,
    topology_profile: str,
    topology_profile_present: bool,
) -> ConfigCompatibility:
    strict_default = topology_profile_present and topology_profile != "legacy"
    if not topology_profile_present or topology_profile == "legacy":
        strict_default = False
    return ConfigCompatibility(
        strict=bool(config.get("strict", strict_default)),
        allow_profile_overrides=bool(config.get("allow_profile_overrides", False)),
        warn_deprecated_keys=bool(config.get("warn_deprecated_keys", True)),
    )


def _default_allowed_splits(topology_profile: str, split_names: tuple[str, ...]) -> tuple[str, ...]:
    gateway = tuple(name for name in split_names if is_gateway_split(name, split_names))
    direct = tuple(name for name in split_names if is_direct_rc_split(name, split_names))
    if topology_profile == "gateway_only":
        return gateway
    if topology_profile == "direct_only":
        return direct
    return split_names


def _default_direct_enabled(topology_profile: str, has_direct_section: bool) -> bool:
    if topology_profile == "direct_only":
        return True
    if topology_profile == "gateway_only":
        return False
    return has_direct_section


def _default_fronthaul_enabled(topology_profile: str) -> bool:
    return topology_profile != "direct_only"


def _validate_split_options(split_options: dict[str, Any], num_splits: int, split_names: tuple[str, ...]) -> None:
    if num_splits != len(split_names):
        raise ConfigError("number.num_splits must match len(split_options.names)")
    required_lists = (
        "es_compute_per_mbps",
        "rc_compute_per_mbps",
        "fronthaul_bw_per_mbps",
        "midhaul_bw_per_mbps",
        "max_fronthaul_latency_ms",
        "max_midhaul_latency_ms",
    )
    for key in required_lists:
        values = split_options[key]
        if len(values) != len(split_names):
            raise ConfigError(f"split_options.{key} length must match len(split_options.names)")


def _validate_direct_rh_rc_shape(direct: dict[str, Any], *, num_rhs: int, num_rc: int) -> None:
    degree_mode = str(direct.get("degree_mode", "pairwise"))
    if degree_mode not in ALLOWED_DIRECT_RH_RC_DEGREE_MODES:
        raise ConfigError(
            "direct_rh_rc.degree_mode must be one of "
            f"{sorted(ALLOWED_DIRECT_RH_RC_DEGREE_MODES)}"
        )
    if degree_mode == "global_sparse":
        if "min_total_edges" not in direct or "max_total_edges" not in direct:
            raise ConfigError(
                "ConfigError: direct_rh_rc.degree_mode='global_sparse' requires "
                "min_total_edges and max_total_edges."
            )
        min_total_edges = int(direct["min_total_edges"])
        max_total_edges = int(direct["max_total_edges"])
        if min_total_edges < 0 or max_total_edges < 0 or min_total_edges > max_total_edges:
            raise ConfigError(
                "ConfigError: direct_rh_rc min_total_edges and max_total_edges must satisfy "
                "0 <= min_total_edges <= max_total_edges."
            )
        if max_total_edges > num_rhs * num_rc:
            raise ConfigError(
                "ConfigError: direct_rh_rc.max_total_edges cannot exceed num_rhs * num_rc."
            )
        max_degree_per_rh = int(direct.get("max_degree_per_rh", 1))
        if max_degree_per_rh < 0:
            raise ConfigError("ConfigError: direct_rh_rc.max_degree_per_rh must be non-negative.")
        if str(direct.get("selected_rh_strategy", "random")) != "random":
            raise ConfigError("ConfigError: direct_rh_rc.selected_rh_strategy currently supports only 'random'.")
        max_degree_per_rc_raw = direct.get("max_degree_per_rc")
        max_degree_per_rc = None if max_degree_per_rc_raw is None else int(max_degree_per_rc_raw)
        if max_degree_per_rc is not None and max_degree_per_rc < 0:
            raise ConfigError("ConfigError: direct_rh_rc.max_degree_per_rc must be non-negative or null.")
        max_possible = min(
            num_rhs * num_rc,
            num_rhs * max_degree_per_rh,
            num_rc * max_degree_per_rc if max_degree_per_rc is not None else num_rhs * num_rc,
        )
        if max_total_edges > max_possible:
            raise ConfigError(
                "ConfigError: direct_rh_rc global_sparse degree constraints are infeasible for "
                f"max_total_edges={max_total_edges}."
            )


def _validate_backhaul_shape(backhaul: dict[str, Any], *, num_es: int, num_rc: int) -> None:
    degree_mode = str(backhaul.get("es_rc_degree_mode", "legacy_drop"))
    if degree_mode not in ALLOWED_BACKHAUL_DEGREE_MODES:
        raise ConfigError(
            "backhaul.es_rc_degree_mode must be one of "
            f"{sorted(ALLOWED_BACKHAUL_DEGREE_MODES)}"
        )
    if degree_mode not in {"bounded", "episode_profile"}:
        return
    es_rc_min_degree = int(backhaul.get("es_rc_min_degree", 1))
    es_rc_max_degree = int(backhaul.get("es_rc_max_degree", 1))
    if es_rc_min_degree < 0 or es_rc_max_degree < 0 or es_rc_min_degree > es_rc_max_degree:
        raise ConfigError(
            "ConfigError: backhaul.es_rc_min_degree and backhaul.es_rc_max_degree must satisfy "
            "0 <= es_rc_min_degree <= es_rc_max_degree."
        )
    if es_rc_max_degree > num_rc:
        raise ConfigError("ConfigError: backhaul.es_rc_max_degree cannot exceed num_rc.")
    rc_max_degree_raw = backhaul.get("rc_max_degree")
    if rc_max_degree_raw is None:
        return
    rc_max_degree = int(rc_max_degree_raw)
    if rc_max_degree < 0:
        raise ConfigError("ConfigError: backhaul.rc_max_degree must be non-negative or null.")
    min_total_edges = num_es * es_rc_min_degree
    max_total_capacity = num_rc * rc_max_degree
    if min_total_edges > max_total_capacity:
        raise ConfigError(
            "ConfigError: backhaul bounded degree is infeasible because "
            "num_es * es_rc_min_degree exceeds num_rc * rc_max_degree."
        )


def _collect_deprecation_warnings(
    warnings: list[str],
    params: dict[str, Any],
    *,
    fronthaul: dict[str, Any],
    direct: dict[str, Any],
    backhaul: dict[str, Any],
    transport_budget: dict[str, Any],
) -> None:
    if "topology_profile" not in params:
        warnings.append(
            "DeprecationWarning: missing topology_profile uses legacy compatibility mode. "
            "Set topology_profile explicitly for new configs."
        )
    if "extra_link_probability" in fronthaul:
        warnings.append(
            "DeprecationWarning: fronthaul.extra_link_probability is a legacy/advanced key and may conflict "
            "with topology intent or RH transport budget."
        )
    if "connectivity_probability" in direct:
        if str(direct.get("degree_mode", "pairwise")) == "bounded":
            warnings.append(
                "DeprecationWarning: direct_rh_rc.connectivity_probability is ignored when "
                "direct_rh_rc.degree_mode='bounded'; it is legacy pairwise-mode only."
            )
        elif str(direct.get("degree_mode", "pairwise")) == "global_sparse":
            warnings.append(
                "DeprecationWarning: direct_rh_rc.connectivity_probability is ignored when "
                "direct_rh_rc.degree_mode='global_sparse'; pairwise probability does not control exact edge counts."
            )
        else:
            warnings.append(
                "DeprecationWarning: direct_rh_rc.connectivity_probability is a legacy/advanced pairwise-mode key."
            )
    backhaul_degree_mode = str(backhaul.get("es_rc_degree_mode", "legacy_drop"))
    if backhaul_degree_mode in {"bounded", "episode_profile"} and "drop_probability" in backhaul:
        warnings.append(
            "DeprecationWarning: backhaul.drop_probability is ignored when "
            f"backhaul.es_rc_degree_mode={backhaul_degree_mode!r}; "
            "drop_probability is legacy random deletion, not precise sparse topology."
        )
    if "gateway_probability" in transport_budget:
        warnings.append(
            "DeprecationWarning: rh_transport_budget.gateway_probability is a legacy/advanced key that can "
            "conflict with topology intent."
        )
    if "direct_probability" in transport_budget:
        warnings.append(
            "DeprecationWarning: rh_transport_budget.direct_probability is a legacy/advanced key that can "
            "conflict with topology intent."
        )
    if "allocation_mode" in transport_budget and str(transport_budget.get("allocation_mode", "legacy_independent")) == "legacy_independent":
        warnings.append(
            "DeprecationWarning: rh_transport_budget.allocation_mode='legacy_independent' preserves legacy-like "
            "behavior and effectively bypasses the joint RH transport budget."
        )

"""Versioned ownership contract for problem and RL runtime configuration.

The legacy project predates this module and exposes one merged dictionary to
the evaluator.  That adapter remains available solely for reproducing legacy
datasets; v3 problem configuration and v2 RL configuration are parsed and
validated independently here.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import copy
import csv
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
import warnings


PROBLEM_CONFIG_SCHEMA_V3 = "oran_problem_config_v3"
RL_CONFIG_SCHEMA_V2 = "oran_rl_config_v2"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RL_CONFIG = _REPO_ROOT / "config" / "default.json"
_LEGACY_RUNTIME_DEFAULTS = _REPO_ROOT / "config" / "legacy_default_v1.json"


class ConfigurationContractError(ValueError):
    """Raised when a configuration violates ownership or v3/v2 schema rules."""


class ConfigurationOwnershipError(ConfigurationContractError):
    """Raised when a semantic field is declared by a non-canonical owner."""


# The registry is deliberately semantic rather than a list of implementation
# aliases.  New schemas are validated against it before any legacy adapter is
# allowed to run.
SEMANTIC_OWNERS: Mapping[str, str] = MappingProxyType({
    "dataset": "problem_yaml",
    "topology": "problem_yaml",
    "workload": "problem_yaml",
    "formulation": "problem_yaml",
    "objective": "problem_yaml",
    "generation": "problem_yaml",
    "instance_parameters": "materialized_csv",
    "observation": "rl_json",
    "reward": "rl_json",
    "ppo": "rl_json",
    "model": "rl_json",
    "training": "rl_json",
    "diagnostics": "rl_json",
})

_RL_ALLOWED_TOP_LEVEL = {
    "rl_config_schema_version", "experiment", "observation", "reward", "ppo",
    "model", "training", "diagnostics",
}
_RL_PROBLEM_OWNED_KEYS = {
    "cost_weights", "constraint", "handover_control", "reconfiguration_cost",
    "constraint_formulation_version", "handover_sla", "topology", "split_demand",
    "formulation", "objective", "workload", "generation", "capacity", "fronthaul",
    "midhaul", "backhaul", "direct_rh_rc", "service_deadline_ms",
}
_PROBLEM_ALLOWED_TOP_LEVEL = {
    "problem_config_schema_version", "dataset", "topology", "workload", "formulation",
    "generation", "instance_parameters", "traces",
}
_PROBLEM_RL_OWNED_KEYS = {
    "ppo", "model", "reward", "observation", "training", "diagnostics",
    "rl_config_schema_version", "observation_feature_mode",
}
_LEGACY_PROBLEM_ALIASES = {
    "number", "traffic", "mobility", "state_transfer", "fronthaul", "backhaul",
    "capacity", "split_options", "cost_weights", "constraint_formulation_version",
}


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return value


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _ownership_error(*, semantic_field: str, canonical_owner: str,
                     conflicting_source: str, conflicting_values: Any) -> ConfigurationOwnershipError:
    return ConfigurationOwnershipError(
        "configuration ownership conflict: "
        f"semantic field={semantic_field!r}; canonical owner={canonical_owner}; "
        f"conflicting source={conflicting_source}; conflicting values={conflicting_values!r}"
    )


@dataclass(frozen=True)
class ProblemConfig:
    path: Path
    sha256: str
    schema_version: str
    data: Mapping[str, Any]
    legacy_aliases_used: tuple[str, ...] = ()

    @property
    def is_legacy(self) -> bool:
        return self.schema_version != PROBLEM_CONFIG_SCHEMA_V3

    @property
    def formulation_status(self) -> str:
        if self.is_legacy:
            return "legacy"
        return str(self.data["formulation"]["status"])


@dataclass(frozen=True)
class RLConfig:
    path: Path
    sha256: str
    schema_version: str
    data: Mapping[str, Any]
    legacy_aliases_used: tuple[str, ...] = ()

    @property
    def is_legacy(self) -> bool:
        return self.schema_version != RL_CONFIG_SCHEMA_V2


@dataclass(frozen=True)
class InstanceParameterSource:
    """A YAML declaration that deliberately delegates a value to one CSV column."""

    semantic_field: str
    csv_file: str
    csv_column: str
    expected_unit: str
    scope: str


@dataclass(frozen=True)
class ResolvedInstanceParameter:
    source: InstanceParameterSource
    values: tuple[float, ...]

    def summary(self) -> dict[str, float | int]:
        return {
            "count": len(self.values), "minimum": min(self.values), "maximum": max(self.values),
            "unique_value_count": len(set(self.values)),
        }


INSTANCE_PARAMETER_SPECS: Mapping[str, InstanceParameterSource] = MappingProxyType({
    "es_compute_capacity": InstanceParameterSource(
        "es_compute_capacity", "es_nodes.csv", "compute_capacity", "compute_units", "node"
    ),
    "rc_compute_capacity": InstanceParameterSource(
        "rc_compute_capacity", "rc_nodes.csv", "compute_capacity", "compute_units", "node"
    ),
    "fronthaul_bandwidth": InstanceParameterSource(
        "fronthaul_bandwidth", "fronthaul_edges.csv", "bandwidth_mbps", "Mbps", "edge"
    ),
    "midhaul_bandwidth": InstanceParameterSource(
        "midhaul_bandwidth", "midhaul_edges.csv", "bandwidth_mbps", "Mbps", "edge"
    ),
    "direct_bandwidth": InstanceParameterSource(
        "direct_bandwidth", "direct_rh_rc_edges.csv", "bandwidth_mbps", "Mbps", "edge"
    ),
    "rh_service_deadline": InstanceParameterSource(
        "rh_service_deadline", "rh_nodes.csv", "service_deadline_ms", "ms", "rh"
    ),
})


@dataclass(frozen=True)
class ResolvedRunConfig:
    """Immutable references to independently owned problem and RL settings."""

    problem: ProblemConfig
    rl: RLConfig
    dataset_dir: Path | None
    dataset_metadata_hash: str | None
    runtime_config: Mapping[str, Any] | None
    instance_parameters: Mapping[str, ResolvedInstanceParameter] = MappingProxyType({})

    def audit_dict(self) -> dict[str, Any]:
        return {
            "problem_config": {
                "path": str(self.problem.path), "sha256": self.problem.sha256,
                "schema_version": self.problem.schema_version,
                "legacy_aliases_used": list(self.problem.legacy_aliases_used),
            },
            "rl_config": {
                "path": str(self.rl.path), "sha256": self.rl.sha256,
                "schema_version": self.rl.schema_version,
                "legacy_aliases_used": list(self.rl.legacy_aliases_used),
            },
            "formulation_status": self.problem.formulation_status,
            "formulation_version": (
                self.problem.data.get("formulation", {}).get("version")
                if not self.problem.is_legacy else self.runtime_config.get("constraint_formulation_version")
                if self.runtime_config is not None else None
            ),
            "dataset_metadata_hash": self.dataset_metadata_hash,
            "instance_parameters": {
                field: {
                    "source": parameter.source.csv_file,
                    "column": parameter.source.csv_column,
                    "unit": parameter.source.expected_unit,
                    "scope": parameter.source.scope,
                    "summary": parameter.summary(),
                }
                for field, parameter in self.instance_parameters.items()
            },
        }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigurationContractError(f"RL configuration must be a JSON object: {path}")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    # Import lazily to avoid a module cycle with the legacy config adapter.
    from src.utils.config import load_simple_yaml
    payload = load_simple_yaml(path)
    if not isinstance(payload, dict):
        raise ConfigurationContractError(f"problem configuration must be a YAML mapping: {path}")
    return payload


def _validate_rl_v2(raw: Mapping[str, Any]) -> None:
    for key in raw:
        if key in _RL_PROBLEM_OWNED_KEYS:
            raise _ownership_error(
                semantic_field=key, canonical_owner="problem_yaml",
                conflicting_source="rl_json", conflicting_values=raw[key],
            )
        if key not in _RL_ALLOWED_TOP_LEVEL:
            raise ConfigurationContractError(f"unknown key in {RL_CONFIG_SCHEMA_V2}: {key}")
    required = {"experiment", "observation", "reward", "ppo", "model", "training", "diagnostics"}
    missing = sorted(required - set(raw))
    if missing:
        raise ConfigurationContractError(f"{RL_CONFIG_SCHEMA_V2} missing required keys: {missing}")


def _translate_legacy_rl(raw: Mapping[str, Any]) -> dict[str, Any]:
    ppo = dict(raw.get("ppo", {}))
    return {
        "rl_config_schema_version": RL_CONFIG_SCHEMA_V2,
        "experiment": {"scope": raw.get("scope", "handover-aware-orchestration-phase-1")},
        "observation": {"feature_mode": raw.get("observation_feature_mode", "legacy_rh_only")},
        "reward": dict(raw.get("reward", {})),
        "ppo": {
            key: ppo[key] for key in (
                "learning_rate", "gamma", "clip_epsilon", "entropy_coef", "entropy_scale_mode",
                "value_coef", "update_epochs", "rollout_episodes",
            ) if key in ppo
        },
        "model": {
            "hidden_dim": dict(raw.get("model", {})).get("hidden_dim", 16),
            "temporal_window_size": ppo.get("temporal_window_size", 3),
        },
        "training": {"default_iterations": ppo.get("default_iterations", 1)},
        "diagnostics": {
            "enable_temporal_logit_delta": ppo.get("enable_temporal_logit_delta_diagnostic", False)
        },
    }


def load_rl_config(path: str | Path | None = None) -> RLConfig:
    config_path = Path(path) if path is not None else _DEFAULT_RL_CONFIG
    raw = _load_json(config_path)
    schema = raw.get("rl_config_schema_version")
    aliases: tuple[str, ...] = ()
    if schema == RL_CONFIG_SCHEMA_V2:
        _validate_rl_v2(raw)
    else:
        aliases = tuple(sorted(set(raw) & _RL_PROBLEM_OWNED_KEYS | {"scope", "observation_feature_mode"}))
        warnings.warn(
            f"legacy RL configuration translated from {config_path}; legacy keys={list(aliases)}",
            DeprecationWarning, stacklevel=2,
        )
        raw = _translate_legacy_rl(raw)
        schema = "legacy_rl_config_v1"
    return RLConfig(config_path, _sha256(config_path), str(schema), _freeze(raw), aliases)


def _validate_named_splits(formulation: Mapping[str, Any]) -> None:
    split_order = formulation.get("split_order")
    splits = formulation.get("splits")
    if not isinstance(split_order, list) or not isinstance(splits, Mapping):
        raise ConfigurationContractError("formulation.split_order and formulation.splits are required")
    if len(split_order) != len(set(split_order)):
        raise ConfigurationContractError("formulation.split_order contains duplicate split names")
    if set(split_order) != set(splits):
        raise ConfigurationContractError("formulation.split_order must contain every split record exactly once")
    for name in split_order:
        record = splits[name]
        try:
            compute = record["compute"]
            es_fraction = float(compute["es_fraction"])
            rc_fraction = float(compute["rc_fraction"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationContractError(f"formulation.splits.{name}.compute is invalid") from exc
        if not math.isclose(es_fraction + rc_fraction, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ConfigurationContractError(f"formulation.splits.{name}: ES and RC fractions must sum to 1.0")
    s4d = splits.get("S4D")
    if s4d is not None:
        transport = s4d.get("transport", {})
        if transport.get("direct") is None or transport.get("fronthaul") is not None or transport.get("midhaul") is not None:
            raise ConfigurationContractError("formulation.splits.S4D must use direct transport only")
    s4g = splits.get("S4G")
    if s4g is not None:
        transport = s4g.get("transport", {})
        if float(s4g.get("compute", {}).get("es_fraction", -1.0)) != 0.0:
            raise ConfigurationContractError("formulation.splits.S4G must not use ES compute")
        if transport.get("direct") is not None or transport.get("fronthaul") is None or transport.get("midhaul") is None:
            raise ConfigurationContractError("formulation.splits.S4G must use gateway transport")


def _validate_problem_v3(raw: Mapping[str, Any]) -> None:
    for key in raw:
        if key in _PROBLEM_RL_OWNED_KEYS:
            raise _ownership_error(
                semantic_field=key, canonical_owner="rl_json",
                conflicting_source="problem_yaml", conflicting_values=raw[key],
            )
        if key in _LEGACY_PROBLEM_ALIASES:
            raise ConfigurationContractError(f"{PROBLEM_CONFIG_SCHEMA_V3} rejects legacy alias: {key}")
        if key not in _PROBLEM_ALLOWED_TOP_LEVEL:
            raise ConfigurationContractError(f"unknown key in {PROBLEM_CONFIG_SCHEMA_V3}: {key}")
    required = {"dataset", "topology", "formulation"}
    missing = sorted(required - set(raw))
    if missing:
        raise ConfigurationContractError(f"{PROBLEM_CONFIG_SCHEMA_V3} missing required keys: {missing}")
    formulation = raw["formulation"]
    if not isinstance(formulation, Mapping):
        raise ConfigurationContractError("formulation must be a mapping")
    if formulation.get("status") not in {"draft", "active"}:
        raise ConfigurationContractError("formulation.status must be 'draft' or 'active'")
    csv_owned_instance_values = "instance_parameters" in raw
    if csv_owned_instance_values:
        _validate_instance_parameter_declarations(raw["instance_parameters"])
        compute = formulation.get("compute", {})
        transport = formulation.get("transport", {})
        if "capacities" in compute or "capacities" in transport:
            raise _ownership_error(
                semantic_field="capacity", canonical_owner="materialized_csv",
                conflicting_source="problem_yaml", conflicting_values="numeric capacity section",
            )
        if "service_deadline_ms" in formulation.get("latency", {}):
            raise _ownership_error(
                semantic_field="rh_service_deadline", canonical_owner="materialized_csv",
                conflicting_source="problem_yaml", conflicting_values=formulation["latency"]["service_deadline_ms"],
            )
    elif float(formulation.get("latency", {}).get("service_deadline_ms", float("nan"))) != 4.0:
        raise ConfigurationContractError("formulation.latency.service_deadline_ms must be exactly 4.0 ms")
    _validate_named_splits(formulation)
    if formulation["status"] == "active" and not csv_owned_instance_values:
        direct_latency = formulation.get("transport", {}).get("latency", {}).get("direct_ms")
        capacities = formulation.get("compute", {}).get("capacities", {})
        missing_active = []
        if direct_latency is None:
            missing_active.append("formulation.transport.latency.direct_ms")
        for node_type in ("es", "rc"):
            if capacities.get(node_type, {}).get("values") is None:
                missing_active.append(f"formulation.compute.capacities.{node_type}.values")
        if missing_active:
            raise ConfigurationContractError(
                "active formulation has unresolved required values: " + ", ".join(missing_active)
            )
    if "backhaul" in raw.get("topology", {}) and "midhaul" in raw.get("topology", {}):
        raise ConfigurationContractError("v3 problem YAML cannot contain both backhaul and midhaul")


def _validate_instance_parameter_declarations(declarations: Any) -> None:
    if not isinstance(declarations, Mapping):
        raise ConfigurationContractError("instance_parameters must be a mapping")
    if set(declarations) != set(INSTANCE_PARAMETER_SPECS):
        missing = sorted(set(INSTANCE_PARAMETER_SPECS) - set(declarations))
        extra = sorted(set(declarations) - set(INSTANCE_PARAMETER_SPECS))
        raise ConfigurationContractError(
            f"instance_parameters must declare each canonical field exactly once; missing={missing}, extra={extra}"
        )
    declared_sources: set[tuple[str, str]] = set()
    for field, expected in INSTANCE_PARAMETER_SPECS.items():
        declaration = declarations[field]
        if not isinstance(declaration, Mapping):
            raise ConfigurationContractError(f"instance_parameters.{field} must be a mapping")
        actual = (str(declaration.get("source")), str(declaration.get("column")))
        canonical = (expected.csv_file, expected.csv_column)
        if actual != canonical:
            raise _ownership_error(
                semantic_field=field, canonical_owner=f"{expected.csv_file}:{expected.csv_column}",
                conflicting_source="problem_yaml declaration", conflicting_values=actual,
            )
        if actual in declared_sources:
            raise ConfigurationOwnershipError(
                f"configuration ownership conflict: semantic field={field!r}; canonical owner=materialized_csv; "
                f"conflicting source={actual[0]}:{actual[1]}; conflicting values=duplicate declaration"
            )
        declared_sources.add(actual)


def load_problem_config(path: str | Path) -> ProblemConfig:
    config_path = Path(path)
    raw = _load_yaml(config_path)
    schema = raw.get("problem_config_schema_version")
    if schema == PROBLEM_CONFIG_SCHEMA_V3:
        _validate_problem_v3(raw)
        return ProblemConfig(config_path, _sha256(config_path), schema, _freeze(raw))
    aliases = tuple(sorted(set(raw) & _LEGACY_PROBLEM_ALIASES))
    warnings.warn(
        f"legacy problem configuration translated from {config_path}; legacy keys={list(aliases)}",
        DeprecationWarning, stacklevel=2,
    )
    return ProblemConfig(config_path, _sha256(config_path), "legacy_problem_config_v1", _freeze(raw), aliases)


def _legacy_runtime_adapter(rl: RLConfig, *, legacy_source: Path | None = None) -> Mapping[str, Any]:
    source = legacy_source or _LEGACY_RUNTIME_DEFAULTS
    base = _load_json(source)
    rl_data = _thaw(rl.data)
    base["scope"] = rl_data["experiment"]["scope"]
    base["observation_feature_mode"] = rl_data["observation"]["feature_mode"]
    base["reward"] = rl_data["reward"]
    base["ppo"] = dict(rl_data["ppo"])
    base["ppo"]["enable_temporal_logit_delta_diagnostic"] = rl_data["diagnostics"]["enable_temporal_logit_delta"]
    base["ppo"]["temporal_window_size"] = rl_data["model"]["temporal_window_size"]
    base["ppo"]["default_iterations"] = rl_data["training"]["default_iterations"]
    base["model"] = {"hidden_dim": rl_data["model"]["hidden_dim"]}
    return _freeze(base)


def runtime_project_config_from_resolved_problem(resolved: ResolvedRunConfig) -> dict[str, Any]:
    """Materialize the active runtime view from independently owned inputs.

    This is an internal bridge only: all scientific inputs continue to come
    from the v3 problem YAML or its declared CSV columns, while the RL fields
    come exclusively from the v2 JSON.  It lets the established environment
    and trainer construction path keep using one mapping without reviving a
    precedence merge between the two configuration files.
    """

    if resolved.problem.is_legacy:
        raise ConfigurationContractError("active runtime requires a versioned problem configuration")
    problem = _thaw(resolved.problem.data)
    rl = _thaw(resolved.rl.data)
    formulation = problem["formulation"]
    split_order = list(formulation["split_order"])
    splits = formulation["splits"]
    objective = formulation["objective"]
    weights = objective["weights"]
    handover_sla = objective["handover_sla"]
    handover_control = objective["handover_control"]
    reconfiguration = objective["reconfiguration"]
    transport = formulation["transport"]["demand"]

    # ``SplitResourceDemand`` remains the shared typed container.  Patch 1
    # consumes the named v3 records directly for transport/latency, so only
    # the compute fractions and semantic split names are meaningful here.
    return {
        "scope": rl["experiment"]["scope"],
        "observation_feature_mode": rl["observation"]["feature_mode"],
        "reward": rl["reward"],
        "ppo": {
            **rl["ppo"],
            "enable_temporal_logit_delta_diagnostic": rl["diagnostics"]["enable_temporal_logit_delta"],
            "temporal_window_size": rl["model"]["temporal_window_size"],
            "default_iterations": rl["training"]["default_iterations"],
        },
        "model": rl["model"],
        "topology": {
            "num_rhs": int(problem["topology"]["num_rhs"]),
            "num_es": int(problem["topology"]["num_es"]),
            "num_rc": int(problem["topology"]["num_rc"]),
            "num_splits": len(split_order),
            "horizon": int(problem["dataset"]["episode_length"]) * int(problem["dataset"]["num_episodes"]),
        },
        "split_demand": {
            "split_names": split_order,
            "es_compute_per_mbps": [float(splits[name]["compute"]["es_fraction"]) for name in split_order],
            "rc_compute_per_mbps": [float(splits[name]["compute"]["rc_fraction"]) for name in split_order],
            "fronthaul_bw_per_mbps": [float(transport["o7_2x_mbps"])] * len(split_order),
            "midhaul_bw_per_mbps": [0.0] * len(split_order),
            "max_fronthaul_latency_ms": [0.0] * len(split_order),
            "max_midhaul_latency_ms": [0.0] * len(split_order),
            "es_processing_latency_ms": [0.0] * len(split_order),
            "rc_processing_latency_ms": [0.0] * len(split_order),
            "max_end_to_end_latency_ms": [4.0] * len(split_order),
            "latency_config_source": "rh_nodes.csv:service_deadline_ms",
        },
        "cost_weights": {
            "placement": float(weights["placement"]),
            "compute_overload": float(weights["compute_violation"]),
            "link_violation": float(weights["bandwidth_violation"]),
            "pdcp_state_transfer": float(objective["handover_state_transfer"]["pdcp_weight"]),
            "rlc_state_transfer": float(objective["handover_state_transfer"]["rlc_weight"]),
            "handover_sla": float(handover_sla["weight"]),
            "invalid_action": float(weights["invalid_action"]),
        },
        "constraint": {
            "infeasible_path_penalty": float(weights["invalid_action"]),
            "latency_excess_penalty": 1.0,
            "min_effective_bandwidth_mbps": float(formulation["feasibility"]["min_effective_bandwidth_mbps"]),
            "latency_objective_mode": "end_to_end",
        },
        "handover_sla": {
            "base_latency_ms": float(handover_sla["base_latency_ms"]),
            "budget_ms": float(handover_sla["budget_ms"]),
            "mbps_to_mb_per_ms": float(handover_sla["mbps_to_mb_per_ms"]),
        },
        "handover_control": dict(handover_control),
        "reconfiguration_cost": dict(reconfiguration),
        "constraint_formulation_version": str(formulation["version"]),
        "formulation": formulation,
        "topology_profile": "mixed",
        "split_policy": {"allowed_split_names": split_order},
        "validity": {"ensure_each_rh_has_valid_action": True},
        "mock_data": {
            "fronthaul_enabled": True,
            "direct_rh_rc_enabled": bool(problem["topology"]["direct_rh_rc"]["enabled"]),
            "rh_transport_budget_enabled": True,
            "rh_transport_budget_max_total_degree_per_rh": 2,
        },
        "mobility_observation_schema_version": "daily_288_lagged_observation_v3",
        "effective_topology": {"es_rc_degree_mode": problem["topology"]["es_rc"]["mode"]},
        # Profile membership is owned by the materialized link CSVs and
        # episode_topology_profiles.csv. Training datasets commonly contain
        # four profiles, while held-out test datasets may intentionally contain
        # two different profiles, so the runtime adapter must not impose a
        # global count.
        "topology_episode_profiles": {},
    }


def project_config_compatibility_adapter(path: str | Path | None = None) -> dict[str, Any]:
    """Return the old merged runtime mapping for legacy consumers only.

    New v3 dataset loading uses :func:`resolve_run_config` and never gets
    problem values from the RL JSON.
    """

    config_path = Path(path) if path is not None else _DEFAULT_RL_CONFIG
    raw = _load_json(config_path)
    if raw.get("rl_config_schema_version") == RL_CONFIG_SCHEMA_V2:
        return _thaw(_legacy_runtime_adapter(load_rl_config(config_path)))
    warnings.warn(
        f"legacy merged project configuration loaded from {config_path}",
        DeprecationWarning, stacklevel=2,
    )
    return copy.deepcopy(raw)


def _metadata_hash(dataset_dir: Path | None) -> str | None:
    if dataset_dir is None:
        return None
    path = dataset_dir / "dataset_metadata.json"
    return _sha256(path) if path.exists() else None


def resolve_run_config(
    problem_path: str | Path,
    rl_path: str | Path | None = None,
    *,
    dataset_dir: str | Path | None = None,
    validate_csv: bool = True,
) -> ResolvedRunConfig:
    """Resolve independently-owned config files without precedence merging."""

    problem = load_problem_config(problem_path)
    rl = load_rl_config(rl_path)
    dataset = Path(dataset_dir) if dataset_dir is not None else None
    runtime: Mapping[str, Any] | None
    instance_parameters: Mapping[str, ResolvedInstanceParameter] = MappingProxyType({})
    if problem.is_legacy:
        runtime = _legacy_runtime_adapter(rl)
    else:
        runtime = None
        # A draft with explicit CSV ownership is structurally validated so it
        # can be inspected before Patch 1; a generic draft without an instance
        # remains inspectable without requiring CSV files.
        if dataset is not None and validate_csv and (
            problem.formulation_status == "active" or "instance_parameters" in problem.data
        ):
            instance_parameters = validate_materialized_dataset(problem, dataset)
    return ResolvedRunConfig(problem, rl, dataset, _metadata_hash(dataset), runtime, instance_parameters)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _assert_exact(field: str, expected: float, actual: float, source: str) -> None:
    if not math.isclose(expected, actual, rel_tol=0.0, abs_tol=1e-9):
        raise ConfigurationContractError(
            f"YAML/CSV mismatch for {field}: canonical owner=problem_yaml; "
            f"conflicting source={source}; conflicting values=({expected!r}, {actual!r})"
        )


def _validate_fixed_node_values(rows: list[dict[str, str]], field: str, values: Any, source: str) -> None:
    if not isinstance(values, (list, tuple)) or len(values) != len(rows):
        raise ConfigurationContractError(f"{field}.values must contain one value per materialized node")
    for expected, row in zip(values, rows, strict=True):
        _assert_exact(field, float(expected), float(row["cpu_capacity"]), source)


def _validate_fixed_edges(rows: list[dict[str, str]], *, feasible_key: str, field: str,
                          expected: float | None, source: str) -> None:
    if expected is None:
        return
    for row in rows:
        if str(row.get(feasible_key, "False")).lower() == "true":
            _assert_exact(field, float(expected), float(row[field]), source)


def _require_exact_ids(rows: list[dict[str, str]], id_column: str, expected_count: int, source: str) -> None:
    if id_column not in (rows[0] if rows else {}):
        raise ConfigurationContractError(f"{source} is missing required identifier column {id_column}")
    ids = [int(row[id_column]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ConfigurationContractError(f"{source} has duplicate {id_column} values")
    if set(ids) != set(range(expected_count)):
        raise ConfigurationContractError(
            f"{source} does not cover expected {id_column} values 0..{expected_count - 1}"
        )


def _positive_finite(value: str, field: str, source: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationContractError(f"{source}:{field} is not numeric") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ConfigurationContractError(f"{source}:{field} must be finite and positive, found {parsed!r}")
    return parsed


def _validate_csv_owned_instances(
    problem: ProblemConfig, root: Path, *, es_rows: list[dict[str, str]], rc_rows: list[dict[str, str]],
    rh_rows: list[dict[str, str]], fh_rows: list[dict[str, str]], mh_rows: list[dict[str, str]],
    direct_rows: list[dict[str, str]],
) -> Mapping[str, ResolvedInstanceParameter]:
    topology = problem.data["topology"]
    num_rhs, num_es, num_rc = (int(topology["num_rhs"]), int(topology["num_es"]), int(topology["num_rc"]))
    tables = {
        "es_nodes.csv": es_rows, "rc_nodes.csv": rc_rows, "rh_nodes.csv": rh_rows,
        "fronthaul_edges.csv": fh_rows, "midhaul_edges.csv": mh_rows,
        "direct_rh_rc_edges.csv": direct_rows,
    }
    declarations = problem.data["instance_parameters"]
    resolved: dict[str, ResolvedInstanceParameter] = {}
    _require_exact_ids(es_rows, "es_id", num_es, "es_nodes.csv")
    _require_exact_ids(rc_rows, "rc_id", num_rc, "rc_nodes.csv")
    _require_exact_ids(rh_rows, "rh_id", num_rhs, "rh_nodes.csv")
    for field, source in INSTANCE_PARAMETER_SPECS.items():
        rows = tables[source.csv_file]
        if not rows or source.csv_column not in rows[0]:
            raise ConfigurationContractError(
                f"declared CSV column is missing: semantic field={field}; "
                f"source={source.csv_file}; column={source.csv_column}"
            )
        if source.scope in {"node", "rh"}:
            values = tuple(_positive_finite(row[source.csv_column], source.csv_column, source.csv_file) for row in rows)
        else:
            feasible_key = "is_feasible"
            required = {"profile_id", feasible_key, "latency_ms", "bandwidth_mbps"}
            missing = required - set(rows[0])
            if missing:
                raise ConfigurationContractError(f"{source.csv_file} is missing canonical columns: {sorted(missing)}")
            values_list: list[float] = []
            for row in rows:
                feasible = str(row[feasible_key]).lower() == "true"
                bandwidth = float(row[source.csv_column])
                latency = float(row["latency_ms"])
                if feasible:
                    if not math.isfinite(bandwidth) or bandwidth <= 0.0:
                        raise ConfigurationContractError(
                            f"{source.csv_file}: feasible link bandwidth_mbps must be finite and positive"
                        )
                    if not math.isfinite(latency) or latency <= 0.0:
                        raise ConfigurationContractError(f"{source.csv_file}: feasible link latency_ms must be finite and positive")
                    values_list.append(bandwidth)
                elif bandwidth != 0.0 or latency != 1_000_000.0:
                    raise ConfigurationContractError(
                        f"{source.csv_file}: infeasible links must use bandwidth_mbps=0.0 and latency_ms=1000000.0"
                    )
            values = tuple(values_list)
        resolved[field] = ResolvedInstanceParameter(source, values)
    deadlines = resolved["rh_service_deadline"].values
    if len(deadlines) != num_rhs or any(value != 4.0 for value in deadlines):
        raise ConfigurationContractError("rh_nodes.csv: every service_deadline_ms must equal 4.0")
    _validate_all_to_all_midhaul(mh_rows, num_es, num_rc, profile_key="profile_id", feasible_key="is_feasible")
    return MappingProxyType(resolved)


def _validate_all_to_all_midhaul(
    rows: list[dict[str, str]], num_es: int, num_rc: int, *, profile_key: str, feasible_key: str,
) -> None:
    by_profile: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_profile.setdefault(row.get(profile_key, "static"), []).append(row)
    if not by_profile:
        raise ConfigurationContractError("all_to_all ES-to-RC validation failed: no materialized midhaul edges")
    for profile, profile_rows in by_profile.items():
        feasible = [row for row in profile_rows if row[feasible_key].lower() == "true"]
        pairs = {(int(row["es_id"]), int(row["rc_id"])) for row in feasible}
        if len(feasible) != num_es * num_rc or len(pairs) != num_es * num_rc:
            raise ConfigurationContractError(
                f"all_to_all ES-to-RC validation failed for profile {profile}: "
                f"expected {num_es * num_rc} feasible pairs, found {len(pairs)}"
            )


def validate_materialized_dataset(
    problem: ProblemConfig, dataset_dir: str | Path
) -> Mapping[str, ResolvedInstanceParameter]:
    """Validate a v3 problem YAML against its generated CSV instance.

    This intentionally ignores ``dataset_metadata.json`` problem-like fields:
    metadata is provenance and must never override canonical YAML semantics.
    """

    if problem.is_legacy:
        return MappingProxyType({})
    root = Path(dataset_dir)
    data = problem.data
    topology = data["topology"]
    formulation = data["formulation"]
    es_rows = _csv_rows(root / "es_nodes.csv")
    rc_rows = _csv_rows(root / "rc_nodes.csv")
    fh_rows = _csv_rows(root / "fronthaul_edges.csv")
    mh_rows = _csv_rows(root / "midhaul_edges.csv")
    direct_path = root / "direct_rh_rc_edges.csv"
    direct_rows = _csv_rows(direct_path) if direct_path.exists() else []
    rh_rows = _csv_rows(root / "rh_nodes.csv")
    for field, actual, expected in (
        ("topology.num_rhs", len(rh_rows), int(topology["num_rhs"])),
        ("topology.num_es", len(es_rows), int(topology["num_es"])),
        ("topology.num_rc", len(rc_rows), int(topology["num_rc"])),
    ):
        if actual != expected:
            raise ConfigurationContractError(
                f"YAML/CSV mismatch for {field}: canonical owner=problem_yaml; "
                f"conflicting source=csv; conflicting values=({expected!r}, {actual!r})"
            )
    if "instance_parameters" in data:
        return _validate_csv_owned_instances(
            problem, root, es_rows=es_rows, rc_rows=rc_rows, rh_rows=rh_rows,
            fh_rows=fh_rows, mh_rows=mh_rows, direct_rows=direct_rows,
        )
    capacities = formulation["compute"]["capacities"]
    _validate_fixed_node_values(es_rows, "formulation.compute.capacities.es", capacities["es"]["values"], "es_nodes.csv")
    _validate_fixed_node_values(rc_rows, "formulation.compute.capacities.rc", capacities["rc"]["values"], "rc_nodes.csv")
    transport = formulation["transport"]
    link_caps = transport["capacities"]
    latency = transport["latency"]
    _validate_fixed_edges(fh_rows, feasible_key="feasible", field="bandwidth_mbps", expected=link_caps["fronthaul_mbps"], source="fronthaul_edges.csv")
    _validate_fixed_edges(mh_rows, feasible_key="feasible", field="bandwidth_mbps", expected=link_caps["midhaul_mbps"], source="midhaul_edges.csv")
    _validate_fixed_edges(direct_rows, feasible_key="connected", field="bandwidth_mbps", expected=link_caps["direct_mbps"], source="direct_rh_rc_edges.csv")
    _validate_fixed_edges(fh_rows, feasible_key="feasible", field="latency_ms", expected=latency["fronthaul_ms"], source="fronthaul_edges.csv")
    _validate_fixed_edges(mh_rows, feasible_key="feasible", field="latency_ms", expected=latency["midhaul_ms"], source="midhaul_edges.csv")
    _validate_fixed_edges(direct_rows, feasible_key="connected", field="latency_ms", expected=latency.get("direct_ms"), source="direct_rh_rc_edges.csv")
    if topology["es_rc"]["mode"] == "all_to_all":
        _validate_all_to_all_midhaul(
            mh_rows, int(topology["num_es"]), int(topology["num_rc"]),
            profile_key="topology_profile_id", feasible_key="feasible",
        )
    return MappingProxyType({})


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value)
    return repr(value)


def dump_simple_yaml(payload: Mapping[str, Any], *, indent: int = 0) -> str:
    lines: list[str] = []
    for key in sorted(payload):
        value = payload[key]
        prefix = " " * indent + f"{key}:"
        if isinstance(value, Mapping):
            lines.append(prefix)
            lines.append(dump_simple_yaml(value, indent=indent + 2).rstrip("\n"))
        else:
            lines.append(f"{prefix} {_yaml_scalar(value)}")
    return "\n".join(lines) + "\n"


def write_effective_generation_config(resolved: ResolvedRunConfig, output: str | Path) -> Path:
    """Write a deterministic v3 audit snapshot; callers choose a new path."""

    if resolved.problem.is_legacy:
        raise ConfigurationContractError("effective v3 audit snapshots require a v3 problem YAML")
    if "instance_parameters" in resolved.problem.data:
        root = resolved.dataset_dir
        if root is None:
            raise ConfigurationContractError("CSV-owned v3 snapshots require dataset_dir")
        payload = {
            "canonical_problem_config_sha256": resolved.problem.sha256,
            "problem_config_schema_version": resolved.problem.schema_version,
            "formulation": {
                "version": resolved.problem.data["formulation"]["version"],
                "status": resolved.problem.formulation_status,
                "split_order": list(resolved.problem.data["formulation"]["split_order"]),
            },
            "instance_parameter_ownership": {
                field: {
                    "source": parameter.source.csv_file,
                    "column": parameter.source.csv_column,
                    "unit": parameter.source.expected_unit,
                    "scope": parameter.source.scope,
                    "non_authoritative_summary": parameter.summary(),
                    "csv_sha256": _sha256(root / parameter.source.csv_file),
                }
                for field, parameter in resolved.instance_parameters.items()
            },
            "resolved_counts": dict(resolved.problem.data["topology"]),
            "validation": "configuration complete; dataset structurally valid; ready for Patch 1",
            "rl_config_schema_version": resolved.rl.schema_version,
            "rl_config_sha256": resolved.rl.sha256,
        }
    else:
        payload = {
            "canonical_problem_config_sha256": resolved.problem.sha256,
            "problem_config_schema_version": resolved.problem.schema_version,
            "problem_config": _thaw(resolved.problem.data),
            "rl_config_schema_version": resolved.rl.schema_version,
            "rl_config_sha256": resolved.rl.sha256,
        }
    target = Path(output)
    target.write_text(dump_simple_yaml(payload), encoding="utf-8")
    return target

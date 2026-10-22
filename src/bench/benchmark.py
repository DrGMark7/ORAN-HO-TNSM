"""CSV/JSON output and policy-trace comparison for the MILP oracle."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.bench.optimizer import (
    COMPONENT_NAMES,
    MilpEpisodeResult,
    OrchestrationMilpOptimizer,
    assert_compatible_metadata,
    optimality_gap,
)
from src.model.inference.post_process import decode_flat_actions
from src.pipeline.schema import PlacementAction


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_oracle_outputs(
    output_dir: str | Path,
    optimizer: OrchestrationMilpOptimizer,
    results: Sequence[MilpEpisodeResult],
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    action_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    split_names = optimizer.episode.split_demand.split_names
    for result in results:
        for ti, (action, encoded) in enumerate(zip(result.actions, result.encoded_actions, strict=True)):
            absolute_t = result.absolute_start_t + ti
            for rh in range(optimizer.episode.config.num_rhs):
                split = int(action.split_id[rh])
                action_rows.append({
                    "episode_id": result.episode_id, "local_t": absolute_t % optimizer.episode_length,
                    "absolute_t": absolute_t, "rh_id": rh,
                    "encoded_action_id": int(encoded[rh]), "split_id": split,
                    "split_name": split_names[split], "es_id": int(action.es_id[rh]),
                    "rc_id": int(action.rc_id[rh]),
                })
        for evaluator_row, solver_row in zip(
            result.step_costs, result.solver_step_costs, strict=True
        ):
            step_rows.append({
                **dict(evaluator_row),
                **{
                    f"cplex_{name}": float(solver_row[name])
                    for name in (*COMPONENT_NAMES, "total_cost")
                },
                "optimization_mode": result.optimization_mode,
            })
        validation_error = result.solver_total_cost - result.evaluator_total_cost
        summaries.append({
            "episode_id": result.episode_id,
            "optimization_mode": result.optimization_mode,
            "solve_status": result.solver_stats.status_name,
            "raw_cplex_status_code": result.solver_stats.cplex_status_code,
            "raw_cplex_status": result.solver_stats.cplex_status_name,
            "termination_reason": result.solver_stats.termination_reason,
            "is_proven_optimal": result.solver_stats.is_proven_optimal,
            "objective_value": result.solver_total_cost,
            "best_bound": result.solver_stats.best_bound,
            "relative_mip_gap": result.solver_stats.relative_gap,
            "evaluator_total_cost": result.evaluator_total_cost,
            "validation_error": validation_error,
            "overall_feasible_rate": result.feasible_rate,
            "solve_time_seconds": result.solver_stats.solve_time_seconds,
            "build_time_seconds": result.solver_stats.build_time_seconds,
            "num_binary_variables": result.solver_stats.num_binary_variables,
            "num_continuous_variables": result.solver_stats.num_continuous_variables,
            "num_constraints": result.solver_stats.num_constraints,
        })
        stats.append({
            "episode_id": result.episode_id,
            "optimization_mode": result.optimization_mode,
            **asdict(result.solver_stats),
            "preparation_time_seconds": result.preparation_time_seconds,
            "extraction_time_seconds": result.extraction_time_seconds,
            "validation_time_seconds": result.validation_time_seconds,
        })
    _write_csv(root / "oracle_actions.csv", action_rows, (
        "episode_id", "local_t", "absolute_t", "rh_id", "encoded_action_id",
        "split_id", "split_name", "es_id", "rc_id",
    ))
    _write_csv(root / "oracle_step_costs.csv", step_rows, (
        "episode_id", "local_t", "absolute_t", *COMPONENT_NAMES, "total_cost", "feasible",
        *(f"cplex_{name}" for name in (*COMPONENT_NAMES, "total_cost")),
        "optimization_mode",
    ))
    _write_csv(root / "oracle_episode_summary.csv", summaries, (
        "episode_id", "optimization_mode", "solve_status", "raw_cplex_status_code",
        "raw_cplex_status", "termination_reason", "is_proven_optimal", "objective_value", "best_bound",
        "relative_mip_gap", "evaluator_total_cost", "validation_error", "overall_feasible_rate",
        "solve_time_seconds", "build_time_seconds", "num_binary_variables",
        "num_continuous_variables", "num_constraints",
    ))
    (root / "oracle_solver_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    modes = {result.optimization_mode for result in results}
    if len(modes) != 1:
        raise ValueError("one oracle output directory cannot mix optimization modes")
    metadata = dict(results[0].formulation_metadata)
    (root / "oracle_formulation_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    proven = sum(result.solver_stats.is_proven_optimal for result in results)
    mode = results[0].optimization_mode
    mode_label = {
        "mirror_soft": "upper-PWL soft-objective oracle",
        "hard_service_feasibility": "hard-service-feasible oracle",
    }[mode]
    qualifier = "" if proven == len(results) else "best-known "
    report = (
        "# MILP Oracle Report\n\n"
        f"Dataset SHA-256: `{optimizer.dataset_hash}`\n\n"
        f"Optimization mode: `{mode}` ({qualifier}{mode_label}).\n\n"
        f"Episodes solved: {len(results)}; proven optimal: {proven}.\n\n"
        "Every stored incumbent cost was recomputed per timestep and component with "
        "`OrchestrationConstraintEvaluator`. The reported best bound, MIP gap, and "
        "optimality status are native diagnostics for the internal upper-PWL model; "
        "they do not prove optimality of the exact nonlinear evaluator objective.\n"
    )
    (root / "MILP_ORACLE_REPORT.md").write_text(report, encoding="utf-8")


def load_action_trace(
    path: str | Path,
    optimizer: OrchestrationMilpOptimizer,
    episode_id: int,
) -> tuple[PlacementAction, ...]:
    rows: dict[tuple[int, int], int] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["episode_id"]) == episode_id:
                action_column = "encoded_action_id" if "encoded_action_id" in row else "action_id"
                rows[(int(row["local_t"]), int(row["rh_id"]))] = int(row[action_column])
    encoded = np.empty((optimizer.episode_length, optimizer.episode.config.num_rhs), dtype=np.int64)
    for t in range(optimizer.episode_length):
        for rh in range(optimizer.episode.config.num_rhs):
            try:
                encoded[t, rh] = rows[(t, rh)]
            except KeyError as exc:
                raise ValueError(f"action trace is missing episode={episode_id}, t={t}, rh={rh}") from exc
    return tuple(decode_flat_actions(row, optimizer.episode.config) for row in encoded)


def evaluate_policy_trace(
    optimizer: OrchestrationMilpOptimizer,
    episode_id: int,
    actions: Sequence[PlacementAction],
) -> tuple[dict[str, float], float, float]:
    if len(actions) != optimizer.episode_length:
        raise ValueError("policy trace must contain exactly one complete episode")
    totals = {name: 0.0 for name in COMPONENT_NAMES}
    feasible = 0
    previous = None
    start = episode_id * optimizer.episode_length
    for local_t, action in enumerate(actions):
        report = optimizer.evaluator.evaluate(start + local_t, action, previous)
        for name in COMPONENT_NAMES:
            totals[name] += float(getattr(report, name))
        feasible += report.feasible
        previous = action
    return totals, sum(totals.values()), feasible / len(actions)


def decision_agreement(
    model_actions: Sequence[PlacementAction], oracle_actions: Sequence[PlacementAction]
) -> dict[str, float | int]:
    if len(model_actions) != len(oracle_actions):
        raise ValueError("action traces must have equal length")
    model_s = np.vstack([a.split_id for a in model_actions]); oracle_s = np.vstack([a.split_id for a in oracle_actions])
    model_e = np.vstack([a.es_id for a in model_actions]); oracle_e = np.vstack([a.es_id for a in oracle_actions])
    model_r = np.vstack([a.rc_id for a in model_actions]); oracle_r = np.vstack([a.rc_id for a in oracle_actions])
    full = (model_s == oracle_s) & (model_e == oracle_e) & (model_r == oracle_r)
    return {
        "split_agreement_rate": float((model_s == oracle_s).mean()),
        "ES_agreement_rate": float((model_e == oracle_e).mean()),
        "RC_agreement_rate": float((model_r == oracle_r).mean()),
        "full_action_agreement_rate": float(full.mean()),
        "action_switch_count": int(np.any(
            np.stack([model_s, model_e, model_r], axis=2)[1:]
            != np.stack([model_s, model_e, model_r], axis=2)[:-1], axis=2
        ).sum()),
        "split_switch_count": int((model_s[1:] != model_s[:-1]).sum()),
        "ES_switch_count": int((model_e[1:] != model_e[:-1]).sum()),
        "RC_switch_count": int((model_r[1:] != model_r[:-1]).sum()),
    }


def compare_policy_to_oracle(
    optimizer: OrchestrationMilpOptimizer,
    policy_actions: Sequence[PlacementAction],
    oracle: MilpEpisodeResult,
    policy_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    assert_compatible_metadata(policy_metadata, oracle.formulation_metadata)
    components, total, feasible_rate = evaluate_policy_trace(optimizer, oracle.episode_id, policy_actions)
    result: dict[str, Any] = {
        "episode_id": oracle.episode_id, "model_total_cost": total,
        "oracle_total_cost": oracle.evaluator_total_cost,
        "optimality_gap": optimality_gap(total, oracle.evaluator_total_cost),
        "gap_label": "proven optimality gap" if oracle.solver_stats.is_proven_optimal else "gap to best-known CPLEX incumbent",
        "oracle_proven_optimal": oracle.solver_stats.is_proven_optimal,
        "oracle_mip_gap": oracle.solver_stats.relative_gap,
        "oracle_best_bound": oracle.solver_stats.best_bound,
        "model_feasible_rate": feasible_rate, "oracle_feasible_rate": oracle.feasible_rate,
    }
    for name in COMPONENT_NAMES:
        result[f"{name}_gap"] = components[name] - oracle.component_costs[name]
    result.update(decision_agreement(policy_actions, oracle.actions))
    return result

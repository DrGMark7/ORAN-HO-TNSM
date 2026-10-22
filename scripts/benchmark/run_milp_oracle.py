#!/usr/bin/env python3
"""Solve selected prebuilt-dataset episodes with the CPLEX MILP oracle."""

from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import sys
import time
from typing import Sequence

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bench.benchmark import evaluate_policy_trace, load_action_trace, write_oracle_outputs
from src.bench.optimizer import MilpSolveOptions, OrchestrationMilpOptimizer
from src.pipeline.schema import PlacementAction
from src.pipeline.data_pipeline import MockCsvDatasetLoader


SERVICE_WEIGHT_KEYS = (
    "compute_violation",
    "bandwidth_violation",
    "latency_violation",
)


def _project_config_with_service_weight(
    project_config: dict,
    service_weight: float | None,
) -> dict:
    """Return an in-memory config with only the three service weights overridden."""

    resolved = deepcopy(project_config)
    if service_weight is None:
        return resolved
    value = float(service_weight)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("--service-weight must be a finite nonnegative value")
    try:
        weights = resolved["formulation"]["objective"]["weights"]
    except KeyError as exc:
        raise ValueError("resolved project config has no formulation objective weights") from exc
    missing = [name for name in SERVICE_WEIGHT_KEYS if name not in weights]
    if missing:
        raise ValueError(f"resolved project config is missing service weights: {missing}")
    for name in SERVICE_WEIGHT_KEYS:
        weights[name] = value
    return resolved


def _episodes(text: str, count: int) -> list[int]:
    if text == "all": return list(range(count))
    if ":" in text:
        lo, hi = (int(value) for value in text.split(":", 1)); return list(range(lo, hi))
    return [int(text)]


def _episode_output_dir(
    output_root: Path,
    dataset_name: str,
    formulation_version: str,
    episode_id: int,
    optimization_mode: str = "mirror_soft",
) -> Path:
    """Return the stable per-episode output directory."""

    # base = output_root / dataset_name / formulation_version
    base = output_root / dataset_name
    if optimization_mode == "mirror_soft":
        return base / str(episode_id)
    return base / str(episode_id)


def _select_policy_mip_start(
    optimizer: OrchestrationMilpOptimizer,
    episode_id: int,
    trace_paths: Sequence[Path],
    *,
    optimization_mode: str,
    ready_file: Path | None,
    wait_seconds: float,
    native_hard_latency_deadline_ms: float | None = None,
) -> tuple[PlacementAction, ...] | None:
    """Return the best compatible policy trace as a CPLEX MIP start.

    The trace is always re-evaluated with the active exact evaluator.  In
    ``mirror_soft`` mode any connectivity-valid trace is a valid MILP start;
    in hard-service mode it must additionally be service-feasible at every
    timestep.  A missing/invalid trace therefore cannot alter the oracle
    formulation or make an infeasible action look feasible.
    """

    if not trace_paths:
        return None
    if wait_seconds < 0:
        raise ValueError("policy MIP-start wait time must be nonnegative")

    deadline = time.monotonic() + wait_seconds
    while ready_file is not None and not ready_file.is_file() and time.monotonic() < deadline:
        time.sleep(0.2)

    profile_id = optimizer.episode.topology_profile_id_for_episode(episode_id)
    optimizer.episode.activate_topology_profile(profile_id)
    candidates: list[tuple[float, float, Path, tuple[PlacementAction, ...]]] = []
    for trace_path in trace_paths:
        if not trace_path.is_file():
            continue
        try:
            actions = load_action_trace(trace_path, optimizer, episode_id)
            _, raw_j, feasible_rate = evaluate_policy_trace(optimizer, episode_id, actions)
        except (OSError, ValueError) as exc:
            print(f"[CPLEX] ignoring invalid policy MIP-start trace {trace_path}: {exc}", file=sys.stderr)
            continue
        if optimization_mode == "hard_service_feasibility" and feasible_rate < 1.0 - 1e-12:
            print(
                f"[CPLEX] policy trace is not hard-service-feasible and will not seed "
                f"episode {episode_id}: {trace_path} (rate={feasible_rate:.6f})",
                file=sys.stderr,
            )
            continue
        candidates.append((raw_j, feasible_rate, trace_path, actions))

    if not candidates:
        if ready_file is not None and not ready_file.is_file():
            print(
                f"[CPLEX] policy MIP-start manifest did not appear within {wait_seconds:g}s; "
                "using the normal oracle seed path.",
                file=sys.stderr,
            )
        return None
    candidates.sort(key=lambda item: item[0])
    if optimization_mode == "hard_service_feasibility":
        native_audits = optimizer.audit_native_hard_service_traces(
            episode_id,
            tuple(item[3] for item in candidates),
            native_hard_latency_deadline_ms=native_hard_latency_deadline_ms,
        )
        for (candidate_j, _, candidate_source, _), audit in zip(
            candidates, native_audits, strict=True
        ):
            if audit.feasible:
                print(
                    f"[CPLEX] native hard-service MIP-start audit passed: "
                    f"{candidate_source}; exact raw J={candidate_j:.10g}; "
                    f"checked_rows={audit.checked_rows}",
                    file=sys.stderr,
                )
            else:
                assert audit.first_violation is not None
                violation = audit.first_violation
                print(
                    f"[CPLEX] native hard-service MIP-start audit rejected: "
                    f"{candidate_source}; exact raw J={candidate_j:.10g}; "
                    f"{violation.kind}; margin={violation.margin:.17g}; "
                    f"row={violation.row}",
                    file=sys.stderr,
                )
        if not native_audits[0].feasible:
            print(
                "[CPLEX] lowest-exact-J trace is not native hard-service "
                "feasible; requesting the hard-service snapshot seed",
                file=sys.stderr,
            )
            return None

    raw_j, feasible_rate, source, actions = candidates[0]
    print(
        f"[CPLEX] using policy MIP start from {source}; exact raw J={raw_j:.10g}, "
        f"exact service-feasible rate={feasible_rate:.6f}",
        file=sys.stderr,
    )
    return actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("mock_data/testset/openireland_daily_288_8rh_relaxed_capacity_test_all_es_rc"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--episodes", default="0", help="one ID, half-open START:END, or all")
    parser.add_argument("--episode-length", type=int, default=288)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--mip-gap", type=float, default=1e-6)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--service-weight",
        type=float,
        help=(
            "runtime-only common weight for compute, bandwidth, and latency "
            "violations"
        ),
    )
    parser.add_argument(
        "--policy-mip-start",
        type=Path,
        action="append",
        default=[],
        help="policy rh_decisions.csv trace eligible to seed the matching episode",
    )
    parser.add_argument(
        "--policy-mip-start-ready-file",
        type=Path,
        help="optional manifest that confirms all policy traces have finished writing",
    )
    parser.add_argument(
        "--policy-mip-start-wait-seconds",
        type=float,
        default=0.0,
        help="wait for the optional policy MIP-start manifest before solving",
    )
    parser.add_argument(
        "--hard-service-snapshot-mip-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="seed hard-service episode solves with independent hard-feasible snapshots",
    )
    parser.add_argument(
        "--hard-service-snapshot-time-limit",
        type=float,
        default=10.0,
        help="per-snapshot CPLEX time limit in seconds for the hard-service seed",
    )
    parser.add_argument(
        "--optimization-mode",
        choices=("mirror_soft", "hard_service_feasibility"),
        default="mirror_soft",
    )
    parser.add_argument(
        "--native-hard-latency-deadline-ms",
        type=float,
        help=(
            "diagnostic-only native CPLEX upper-PWL hard-latency deadline; "
            "does not change the Python evaluator deadline"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark/milp_oracle"))
    args = parser.parse_args()
    if (
        args.optimization_mode == "hard_service_feasibility"
        and args.policy_mip_start
        and not args.hard_service_snapshot_mip_start
    ):
        raise ValueError(
            "hard-service policy MIP starts require the snapshot fallback; "
            "remove --no-hard-service-snapshot-mip-start"
        )
    if (
        args.native_hard_latency_deadline_ms is not None
        and args.optimization_mode != "hard_service_feasibility"
    ):
        raise ValueError("--native-hard-latency-deadline-ms requires hard_service_feasibility")
    loader = MockCsvDatasetLoader(args.dataset, args.config)
    episode = loader.load_episode()
    project_config = _project_config_with_service_weight(
        loader.project_config, args.service_weight
    )
    if args.service_weight is not None:
        print(
            "[CPLEX] runtime service weight: "
            f"compute=bandwidth=latency={float(args.service_weight):.10g}"
        )
    optimizer = OrchestrationMilpOptimizer(
        episode, project_config=project_config, library_path=args.library,
        dataset_dir=args.dataset, episode_length=args.episode_length,
    )
    options = MilpSolveOptions(
        time_limit_seconds=args.time_limit,
        relative_mip_gap=args.mip_gap,
        threads=args.threads,
        optimization_mode=args.optimization_mode,
        use_policy_mip_start=True,
        hard_service_snapshot_mip_start=args.hard_service_snapshot_mip_start,
        hard_service_snapshot_time_limit_seconds=args.hard_service_snapshot_time_limit,
        native_hard_latency_deadline_ms=args.native_hard_latency_deadline_ms,
    )
    selected = _episodes(args.episodes, optimizer.num_episodes)
    if not selected:
        raise ValueError("episode selection is empty")

    timing_totals = {
        "preparation": 0.0,
        "cxx_model_build": 0.0,
        "cplex_solve": 0.0,
        "extraction": 0.0,
        "evaluator_validation": 0.0,
    }
    proven_optimal = 0
    wall_started = time.perf_counter()

    with tqdm(selected, desc="MILP episodes", unit="episode", dynamic_ncols=True) as progress:
        for episode_id in progress:
            progress.set_description(f"MILP episode {episode_id}")
            policy_mip_start = _select_policy_mip_start(
                optimizer,
                episode_id,
                args.policy_mip_start,
                optimization_mode=args.optimization_mode,
                ready_file=args.policy_mip_start_ready_file,
                wait_seconds=args.policy_mip_start_wait_seconds,
                native_hard_latency_deadline_ms=args.native_hard_latency_deadline_ms,
            )
            result = optimizer.solve_episode(
                episode_id,
                options=options,
                policy_mip_start=policy_mip_start,
            )
            output = _episode_output_dir(
                args.output,
                args.dataset.name,
                optimizer.evaluator.formulation_version,
                episode_id,
                args.optimization_mode,
            )
            write_oracle_outputs(output, optimizer, [result])

            timing_totals["preparation"] += result.preparation_time_seconds
            timing_totals["cxx_model_build"] += result.solver_stats.build_time_seconds
            timing_totals["cplex_solve"] += result.solver_stats.solve_time_seconds
            timing_totals["extraction"] += result.extraction_time_seconds
            timing_totals["evaluator_validation"] += result.validation_time_seconds
            proven_optimal += int(result.solver_stats.is_proven_optimal)
            progress.set_postfix(
                status=result.solver_stats.status_name,
                cost=f"{result.evaluator_total_cost:.3f}",
                mip_gap=f"{result.solver_stats.relative_gap:.3g}",
            )
            tqdm.write(f"wrote {output}")

    wall_seconds = time.perf_counter() - wall_started
    episode_count = len(selected)
    print("\nMILP benchmark timing summary")
    print(f"  episodes completed:       {episode_count}")
    print(f"  proven optimal:           {proven_optimal}/{episode_count}")
    print(f"  total wall time:          {wall_seconds:.3f} s")
    for label, seconds in timing_totals.items():
        readable_label = label.replace("_", " ")
        print(
            f"  {readable_label:<24}{seconds:.3f} s total "
            f"({seconds / episode_count:.3f} s/episode)"
        )


if __name__ == "__main__":
    main()

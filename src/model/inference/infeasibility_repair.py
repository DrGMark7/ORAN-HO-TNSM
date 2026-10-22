"""Optional inference-time repair for jointly infeasible greedy actions."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import time
from typing import Any, Callable

import numpy as np

from src.model.constraint import ConstraintReport, OrchestrationConstraintEvaluator
from src.model.inference.post_process import (
    ActionSpaceSpec, decode_flat_actions, decode_tuple_action, encode_flat_actions,
)
from src.model.split_semantics import uses_direct_rh_rc
from src.pipeline.schema import PlacementAction


@dataclass(frozen=True)
class RepairResult:
    actions: np.ndarray
    diagnostics: dict[str, Any]
    changes: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class BeamRepairConfig:
    """Internal bounds for inference-time service-feasibility repair."""

    beam_width: int = 4
    beam_depth: int = 3


@dataclass(frozen=True)
class _BeamState:
    """One bounded-search state, always measured against greedy actions."""

    actions: np.ndarray
    report: ConstraintReport
    changes: tuple[dict[str, Any], ...]
    violation_score: float
    policy_logit_loss: float


def _check(
    evaluator: OrchestrationConstraintEvaluator,
    time_index: int,
    action: PlacementAction,
    previous_action: PlacementAction | None,
) -> ConstraintReport:
    """Evaluate without leaving a candidate's resource loads in env state."""
    prior = evaluator._last_resource_loads
    try:
        return evaluator.evaluate(time_index, action, previous_action)
    finally:
        evaluator._last_resource_loads = prior


def _violations(report: ConstraintReport) -> tuple[float, float, float]:
    v = report.violations
    return (
        float(v.get("es_compute_overload", 0.0)) + float(v.get("rc_compute_overload", 0.0)),
        sum(float(v.get(k, 0.0)) for k in (
            "fronthaul_bw_shortfall_mbps", "midhaul_bw_shortfall_mbps", "direct_rh_rc_bw_shortfall_mbps"
        )),
        float(v.get("end_to_end_latency_excess_ms", 0.0)),
    )


def _violation_score(report: ConstraintReport) -> float:
    """Service-only beam score; lower is closer to feasible."""

    return float(sum(_violations(report)))


def _implicated_rhs(
    evaluator: OrchestrationConstraintEvaluator,
    time_index: int,
    action: PlacementAction,
) -> list[int]:
    ep = evaluator.episode
    loads = evaluator.resource_loads(time_index, action)
    rhs: set[int] = set()
    for r, d in np.argwhere(loads.fronthaul > ep.fronthaul_bandwidth_mbps):
        rhs.add(int(r))
    for r, c in np.argwhere(loads.direct > ep.direct_rh_rc_bandwidth_mbps):
        rhs.add(int(r))
    for d, c in np.argwhere(loads.midhaul > ep.midhaul_bandwidth_mbps):
        for r in range(ep.config.num_rhs):
            split = int(action.split_id[r])
            record = evaluator.formulation["splits"][ep.split_demand.split_names[split]]["transport"]
            if (not uses_direct_rh_rc(split, ep.split_demand.split_names)
                    and int(action.es_id[r]) == int(d) and int(action.rc_id[r]) == int(c)
                    and record["midhaul"] is not None):
                rhs.add(r)
    for d in np.flatnonzero(loads.es > ep.es_compute_capacity):
        rhs.update(int(r) for r in range(ep.config.num_rhs) if int(action.es_id[r]) == int(d))
    for c in np.flatnonzero(loads.rc > ep.rc_compute_capacity):
        rhs.update(int(r) for r in range(ep.config.num_rhs) if int(action.rc_id[r]) == int(c))
    # The evaluator exports only aggregate latency excess; all current paths are
    # the conservative implicated set when that is the sole violation.
    if not rhs:
        rhs.update(range(ep.config.num_rhs))
    return sorted(rhs)


def repair_topk_action(
    *,
    masked_logits: np.ndarray,
    connectivity_mask: np.ndarray,
    evaluator: OrchestrationConstraintEvaluator,
    time_index: int,
    previous_action: PlacementAction | None,
    top_k: int = 5,
    max_iterations: int = 5,
    beam_width: int = 4,
    abort_requested: Callable[[], bool] | None = None,
) -> RepairResult:
    """Bounded beam search for a service-feasible greedy-policy repair.

    The search only uses already-computed masked logits.  Each depth adds one
    RH substitution relative to the original greedy joint action; retained beam
    states are ranked exclusively by service violation, then policy proximity.
    Candidate checks restore evaluator state; callers alone commit the returned
    action to ``env.step``.
    """
    if top_k < 2 or max_iterations < 1 or beam_width < 1:
        raise ValueError("top_k must be >=2, max_iterations must be >=1, and beam_width must be >=1")
    beam_config = BeamRepairConfig(beam_width=beam_width, beam_depth=max_iterations)
    logits = np.asarray(masked_logits, dtype=np.float64)
    mask = np.asarray(connectivity_mask, dtype=bool)
    if logits.shape != mask.shape:
        raise ValueError("masked_logits and connectivity_mask shapes must match")
    spec = ActionSpaceSpec.from_topology(evaluator.episode.config)
    original = logits.argmax(axis=1).astype(np.int64)
    original_action = decode_flat_actions(original, evaluator.episode.config)
    original_report = _check(evaluator, time_index, original_action, previous_action)
    oc, ob, ol = _violations(original_report)
    base = {
        "repair_triggered": not original_report.feasible,
        "repair_succeeded": False,
        "repair_iterations": 0,
        "num_actions_changed": 0,
        "original_service_feasible": original_report.feasible,
        "final_service_feasible": original_report.feasible,
        "original_V_CMP": oc, "original_V_BW": ob, "original_V_LAT": ol,
        "final_V_CMP": oc, "final_V_BW": ob, "final_V_LAT": ol,
        "original_J": original_report.total_cost, "final_J": original_report.total_cost, "delta_J": 0.0,
        "original_implicated_rhs_count": 0,
        "searched_implicated_rhs_count": 0,
        "search_depth_reached": 0,
        "max_beam_states": 0,
        "max_children_before_prune": 0,
        "search_stop_reason": "already_feasible" if original_report.feasible else "max_depth_reached",
        "runtime_aborted": False,
        "previous_action_fallback_used": False,
    }
    if abort_requested is not None and abort_requested():
        base.update({"runtime_aborted": True, "search_stop_reason": "runtime_aborted"})
        return RepairResult(original, base, ())
    if original_report.feasible:
        return RepairResult(original, base, ())

    base["original_implicated_rhs_count"] = len(
        _implicated_rhs(evaluator, time_index, original_action)
    )

    ranked = [
        sorted(np.flatnonzero(mask[r]), key=lambda a: (-logits[r, a], int(a)))[:top_k]
        for r in range(logits.shape[0])
    ]
    rank_by_action = [
        {int(action_id): rank for rank, action_id in enumerate(actions, start=1)}
        for actions in ranked
    ]

    def policy_loss(actions: np.ndarray) -> float:
        changed = np.flatnonzero(actions != original)
        return float(sum(logits[r, original[r]] - logits[r, actions[r]] for r in changed))

    def change_record(rh_id: int, action_id: int) -> dict[str, Any]:
        old = decode_tuple_action(int(original[rh_id]), spec)
        new = decode_tuple_action(int(action_id), spec)
        return {
            "rh_id": rh_id,
            "original_rank": rank_by_action[rh_id].get(int(original[rh_id]), 1),
            "original_split": old[0],
            "original_es": old[1],
            "original_rc": old[2],
            "replacement_rank": rank_by_action[rh_id][int(action_id)],
            "replacement_split": new[0],
            "replacement_es": new[1],
            "replacement_rc": new[2],
            "original_logit": float(logits[rh_id, original[rh_id]]),
            "replacement_logit": float(logits[rh_id, action_id]),
            "logit_loss": float(logits[rh_id, original[rh_id]] - logits[rh_id, action_id]),
        }

    def state_key(state: _BeamState) -> tuple[float, int, float, tuple[int, ...]]:
        return (
            state.violation_score,
            int(np.sum(state.actions != original)),
            state.policy_logit_loss,
            tuple(int(action) for action in state.actions),
        )

    beam = [_BeamState(
        actions=original,
        report=original_report,
        changes=(),
        violation_score=_violation_score(original_report),
        policy_logit_loss=0.0,
    )]
    seen = {tuple(int(action) for action in original)}
    searched_rhs: set[int] = set()

    for depth in range(1, beam_config.beam_depth + 1):
        base["search_depth_reached"] = depth
        base["max_beam_states"] = max(int(base["max_beam_states"]), len(beam))
        children: list[_BeamState] = []
        for state in sorted(beam, key=state_key):
            current_action = decode_flat_actions(state.actions, evaluator.episode.config)
            changed_rhs = set(np.flatnonzero(state.actions != original).tolist())
            implicated_rhs = _implicated_rhs(evaluator, time_index, current_action)
            searched_rhs.update(implicated_rhs)
            for rh_id in implicated_rhs:
                # Each depth represents one additional RH change.  Do not
                # revisit a previously changed RH or revert it to greedy.
                if rh_id in changed_rhs:
                    continue
                for action_id in ranked[rh_id][1:]:
                    if abort_requested is not None and abort_requested():
                        base.update({
                            "runtime_aborted": True,
                            "searched_implicated_rhs_count": len(searched_rhs),
                            "search_stop_reason": "runtime_aborted",
                        })
                        return RepairResult(original, base, ())
                    candidate = state.actions.copy()
                    candidate[rh_id] = int(action_id)
                    candidate_key = tuple(int(action) for action in candidate)
                    if candidate_key in seen:
                        continue
                    seen.add(candidate_key)
                    report = _check(
                        evaluator,
                        time_index,
                        decode_flat_actions(candidate, evaluator.episode.config),
                        previous_action,
                    )
                    changes = state.changes + (change_record(rh_id, int(action_id)),)
                    candidate_loss = policy_loss(candidate)
                    if report.feasible:
                        fc, fb, fl = _violations(report)
                        base.update({
                            "repair_succeeded": True,
                            "repair_iterations": depth,
                            "num_actions_changed": int(np.sum(candidate != original)),
                            "final_service_feasible": True,
                            "final_V_CMP": fc,
                            "final_V_BW": fb,
                            "final_V_LAT": fl,
                            "final_J": report.total_cost,
                            "delta_J": report.total_cost - original_report.total_cost,
                            "searched_implicated_rhs_count": len(searched_rhs),
                            "search_stop_reason": "feasible",
                        })
                        return RepairResult(candidate, base, changes)
                    children.append(_BeamState(
                        actions=candidate,
                        report=report,
                        changes=changes,
                        violation_score=_violation_score(report),
                        policy_logit_loss=candidate_loss,
                    ))
        if not children:
            base.update({
                "searched_implicated_rhs_count": len(searched_rhs),
                "search_stop_reason": "search_space_exhausted",
            })
            break
        base["max_children_before_prune"] = max(int(base["max_children_before_prune"]), len(children))
        beam = sorted(children, key=state_key)[:beam_config.beam_width]
    base["searched_implicated_rhs_count"] = len(searched_rhs)
    if previous_action is not None:
        try:
            previous = encode_flat_actions(previous_action, evaluator.episode.config)
            previous_is_currently_valid = (
                previous.shape == original.shape
                and np.all((previous >= 0) & (previous < mask.shape[1]))
                and bool(mask[np.arange(len(previous)), previous].all())
            )
        except (TypeError, ValueError):
            previous_is_currently_valid = False
        if previous_is_currently_valid:
            try:
                previous_report = _check(evaluator, time_index, previous_action, previous_action)
            except ValueError:
                previous_report = None
            if previous_report is not None and previous_report.feasible:
                fc, fb, fl = _violations(previous_report)
                base.update({
                    "previous_action_fallback_used": True,
                    "num_actions_changed": int(np.sum(previous != original)),
                    "final_service_feasible": True,
                    "final_V_CMP": fc, "final_V_BW": fb, "final_V_LAT": fl,
                    "final_J": previous_report.total_cost,
                    "delta_J": previous_report.total_cost - original_report.total_cost,
                    "search_stop_reason": "previous_action_fallback_feasible",
                })
                return RepairResult(previous, base, ())
    return RepairResult(original, base, ())


# Kept under its historical name for diagnostic compatibility.  In the
# deadline-aware search this limits the active best-first frontier, not a beam.
_DEADLINE_MAX_BEAM_WIDTH = 1024
_DEADLINE_MAX_CANDIDATE_EVALUATIONS = 100_000


def repair_topk_action_deadline(
    *,
    masked_logits: np.ndarray,
    connectivity_mask: np.ndarray,
    evaluator: OrchestrationConstraintEvaluator,
    time_index: int,
    previous_action: PlacementAction | None,
    top_k: int = 5,
    beam_width: int = 4,
    max_iterations: int = 3,
    deadline_ns: int | None = None,
    decision_start_ns: int | None = None,
) -> RepairResult:
    """Search Top-K actions with an anytime best-first priority frontier.

    Evaluator calls are atomic: a call started before ``deadline_ns`` always
    completes.  If it finds a feasible action, that action is returned even
    when completion overshoots the deadline.  Otherwise a deadline failure
    always falls back to the original greedy joint action.
    """
    if top_k < 2 or max_iterations < 1 or beam_width < 1:
        raise ValueError("top_k must be >=2, max_iterations must be >=1, and beam_width must be >=1")
    if decision_start_ns is not None and deadline_ns is None:
        raise ValueError("decision_start_ns requires deadline_ns")

    repair_start_ns = time.perf_counter_ns()
    logits = np.asarray(masked_logits, dtype=np.float64)
    mask = np.asarray(connectivity_mask, dtype=bool)
    if logits.shape != mask.shape:
        raise ValueError("masked_logits and connectivity_mask shapes must match")
    spec = ActionSpaceSpec.from_topology(evaluator.episode.config)
    original = logits.argmax(axis=1).astype(np.int64)
    original_action = decode_flat_actions(original, evaluator.episode.config)

    # The greedy feasibility check is required to preserve Algorithm 1's first
    # exit and is therefore allowed to complete even if inference used the
    # entire decision budget.
    original_report = _check(evaluator, time_index, original_action, previous_action)
    oc, ob, ol = _violations(original_report)
    base = {
        "repair_triggered": not original_report.feasible,
        "repair_succeeded": False,
        "repair_iterations": 0,
        "num_actions_changed": 0,
        "original_service_feasible": original_report.feasible,
        "final_service_feasible": original_report.feasible,
        "original_V_CMP": oc, "original_V_BW": ob, "original_V_LAT": ol,
        "final_V_CMP": oc, "final_V_BW": ob, "final_V_LAT": ol,
        "original_J": original_report.total_cost,
        "final_J": original_report.total_cost,
        "delta_J": 0.0,
        "deadline_enabled": deadline_ns is not None,
        "deadline_hit": False,
        "decision_budget_ms": (
            None if deadline_ns is None or decision_start_ns is None
            else max(0.0, (deadline_ns - decision_start_ns) / 1_000_000.0)
        ),
        "repair_elapsed_ms": 0.0,
        "deadline_overshoot_ms": 0.0,
        "search_depth_reached": 0,
        "max_beam_size_reached": 1,
        "candidate_evaluations": 0,
    }
    search_depth_reached = 0
    max_beam_size_reached = 1
    candidate_evaluations = 0

    def deadline_expired() -> bool:
        return deadline_ns is not None and time.perf_counter_ns() >= deadline_ns

    def result(
        actions: np.ndarray = original,
        changes: tuple[dict[str, Any], ...] = (),
    ) -> RepairResult:
        now_ns = time.perf_counter_ns()
        base.update({
            "deadline_hit": deadline_ns is not None and now_ns >= deadline_ns,
            "repair_elapsed_ms": (now_ns - repair_start_ns) / 1_000_000.0,
            "deadline_overshoot_ms": (
                0.0 if deadline_ns is None else max(0.0, (now_ns - deadline_ns) / 1_000_000.0)
            ),
            "search_depth_reached": search_depth_reached,
            "max_beam_size_reached": max_beam_size_reached,
            "candidate_evaluations": candidate_evaluations,
        })
        return RepairResult(actions, base, changes)

    if original_report.feasible or deadline_expired():
        return result()

    ranked = [
        sorted(np.flatnonzero(mask[r]), key=lambda a: (-logits[r, a], int(a)))[:top_k]
        for r in range(logits.shape[0])
    ]
    rank_by_action = [
        {int(action_id): rank for rank, action_id in enumerate(actions, start=1)}
        for actions in ranked
    ]

    def policy_loss(actions: np.ndarray) -> float:
        changed = np.flatnonzero(actions != original)
        return float(sum(logits[r, original[r]] - logits[r, actions[r]] for r in changed))

    def change_record(rh_id: int, action_id: int) -> dict[str, Any]:
        old = decode_tuple_action(int(original[rh_id]), spec)
        new = decode_tuple_action(int(action_id), spec)
        return {
            "rh_id": rh_id,
            "original_rank": rank_by_action[rh_id].get(int(original[rh_id]), 1),
            "original_split": old[0], "original_es": old[1], "original_rc": old[2],
            "replacement_rank": rank_by_action[rh_id][int(action_id)],
            "replacement_split": new[0], "replacement_es": new[1], "replacement_rc": new[2],
            "original_logit": float(logits[rh_id, original[rh_id]]),
            "replacement_logit": float(logits[rh_id, action_id]),
            "logit_loss": float(logits[rh_id, original[rh_id]] - logits[rh_id, action_id]),
        }

    def state_key(state: _BeamState) -> tuple[float, int, float, tuple[int, ...]]:
        return (
            state.violation_score,
            int(np.sum(state.actions != original)),
            state.policy_logit_loss,
            tuple(int(action) for action in state.actions),
        )

    original_state = _BeamState(
        actions=original,
        report=original_report,
        changes=(),
        violation_score=_violation_score(original_report),
        policy_logit_loss=0.0,
    )
    seen = {tuple(int(action) for action in original)}
    max_depth = logits.shape[0]
    # ``beam_width`` and ``max_iterations`` remain validated compatibility
    # parameters.  Deadline search deliberately does not use them to discard
    # a branch or cap depth; the wall-clock deadline and hard ceilings do that.
    del beam_width, max_iterations
    frontier: list[tuple[tuple[float, int, float, tuple[int, ...]], int, _BeamState]] = []
    insertion_order = 0

    def push_frontier(state: _BeamState) -> None:
        nonlocal insertion_order, max_beam_size_reached, frontier
        heapq.heappush(frontier, (state_key(state), insertion_order, state))
        insertion_order += 1
        if len(frontier) > _DEADLINE_MAX_BEAM_WIDTH:
            # This is a memory safety valve, not ordinary beam pruning: keep
            # the globally best pending states by the same deterministic key.
            frontier = heapq.nsmallest(_DEADLINE_MAX_BEAM_WIDTH, frontier)
            heapq.heapify(frontier)
        max_beam_size_reached = max(max_beam_size_reached, len(frontier))

    push_frontier(original_state)
    while frontier:
        if deadline_expired() or candidate_evaluations >= _DEADLINE_MAX_CANDIDATE_EVALUATIONS:
            return result()
        _, _, state = heapq.heappop(frontier)
        state_depth = int(np.sum(state.actions != original))
        search_depth_reached = max(search_depth_reached, state_depth)
        if state_depth >= max_depth:
            continue
        current_action = decode_flat_actions(state.actions, evaluator.episode.config)
        changed_rhs = set(np.flatnonzero(state.actions != original).tolist())
        for rh_id in _implicated_rhs(evaluator, time_index, current_action):
            if deadline_expired():
                return result()
            if rh_id in changed_rhs:
                continue
            for action_id in ranked[rh_id][1:]:
                # Deadline checks occur before beginning each atomic evaluator
                # call, never during one.
                if (deadline_expired()
                        or candidate_evaluations >= _DEADLINE_MAX_CANDIDATE_EVALUATIONS):
                    return result()
                candidate = state.actions.copy()
                candidate[rh_id] = int(action_id)
                candidate_key = tuple(int(action) for action in candidate)
                if candidate_key in seen:
                    continue
                seen.add(candidate_key)
                candidate_depth = state_depth + 1
                search_depth_reached = max(search_depth_reached, candidate_depth)
                report = _check(
                    evaluator,
                    time_index,
                    decode_flat_actions(candidate, evaluator.episode.config),
                    previous_action,
                )
                candidate_evaluations += 1
                changes = state.changes + (change_record(rh_id, int(action_id)),)
                if report.feasible:
                    fc, fb, fl = _violations(report)
                    base.update({
                        "repair_succeeded": True,
                        "repair_iterations": candidate_depth,
                        "num_actions_changed": int(np.sum(candidate != original)),
                        "final_service_feasible": True,
                        "final_V_CMP": fc, "final_V_BW": fb, "final_V_LAT": fl,
                        "final_J": report.total_cost,
                        "delta_J": report.total_cost - original_report.total_cost,
                    })
                    return result(candidate, changes)
                if deadline_expired():
                    return result()
                push_frontier(_BeamState(
                    actions=candidate,
                    report=report,
                    changes=changes,
                    violation_score=_violation_score(report),
                    policy_logit_loss=policy_loss(candidate),
                ))
    return result()

from __future__ import annotations

from contextlib import ExitStack
import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.ablation.eval_gppo_prebuilt_trace import run_prebuilt_gppo_trace
from scripts.ablation.eval_recurrent_prebuilt_trace import run_prebuilt_recurrent_trace
from src.model.inference import infeasibility_repair as repair_module


class _Clock:
    def __init__(self, now_ns: int) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns


def _report(feasible: bool, score: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(
        feasible=feasible,
        total_cost=score,
        violations={"es_compute_overload": 0.0 if feasible else score},
    )


class _Evaluator:
    def __init__(self, num_rhs: int, evaluate) -> None:
        self.episode = SimpleNamespace(config=SimpleNamespace(num_rhs=num_rhs))
        self._last_resource_loads = object()
        self.calls: list[tuple[int, ...]] = []
        self._evaluate = evaluate

    def evaluate(self, time_index, action, previous_action):
        key = tuple(int(value) for value in action)
        self.calls.append(key)
        self._last_resource_loads = object()
        return self._evaluate(key)


class DeadlineTopKRepairTests(unittest.TestCase):
    def _patch_search(self, num_rhs: int) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(patch.object(repair_module.ActionSpaceSpec, "from_topology", return_value=object()))
        stack.enter_context(patch.object(repair_module, "decode_flat_actions", side_effect=lambda values, _: values))
        stack.enter_context(patch.object(repair_module, "decode_tuple_action", side_effect=lambda value, _: (value, 0, 0)))
        stack.enter_context(patch.object(repair_module, "encode_flat_actions", side_effect=lambda values, _: np.asarray(values)))
        stack.enter_context(
            patch.object(repair_module, "_implicated_rhs", side_effect=lambda evaluator, time_index, action: list(range(num_rhs)))
        )
        return stack

    @staticmethod
    def _inputs(num_rhs: int) -> tuple[np.ndarray, np.ndarray]:
        logits = np.tile(np.array([2.0, 1.0]), (num_rhs, 1))
        return logits, np.ones_like(logits, dtype=bool)

    def test_existing_bounded_repair_still_returns_first_feasible_candidate(self) -> None:
        logits, mask = self._inputs(1)
        evaluator = _Evaluator(1, lambda action: _report(action == (1,)))
        with self._patch_search(1):
            result = repair_module.repair_topk_action(
                masked_logits=logits, connectivity_mask=mask, evaluator=evaluator,
                time_index=0, previous_action=None, top_k=2, max_iterations=1, beam_width=1,
            )
        np.testing.assert_array_equal(result.actions, [1])
        self.assertEqual(result.diagnostics["repair_iterations"], 1)
        self.assertEqual(evaluator.calls, [(0,), (1,)])

    def test_bounded_feasible_greedy_never_checks_previous_action(self) -> None:
        logits, mask = self._inputs(1)
        evaluator = _Evaluator(1, lambda action: _report(True))
        with self._patch_search(1):
            result = repair_module.repair_topk_action(
                masked_logits=logits, connectivity_mask=mask, evaluator=evaluator,
                time_index=0, previous_action=np.array([1]), top_k=2, max_iterations=1, beam_width=1,
            )
        np.testing.assert_array_equal(result.actions, [0])
        self.assertEqual(evaluator.calls, [(0,)])
        self.assertFalse(result.diagnostics["previous_action_fallback_used"])

    def test_bounded_topk_success_wins_over_previous_action_fallback(self) -> None:
        logits = np.array([[3.0, 2.0, 1.0]])
        mask = np.ones_like(logits, dtype=bool)
        evaluator = _Evaluator(1, lambda action: _report(action in {(1,), (2,)}))
        with self._patch_search(1):
            result = repair_module.repair_topk_action(
                masked_logits=logits, connectivity_mask=mask, evaluator=evaluator,
                time_index=0, previous_action=np.array([2]), top_k=2, max_iterations=1, beam_width=1,
            )
        np.testing.assert_array_equal(result.actions, [1])
        self.assertTrue(result.diagnostics["repair_succeeded"])
        self.assertFalse(result.diagnostics["previous_action_fallback_used"])
        self.assertEqual(evaluator.calls, [(0,), (1,)])

    def test_bounded_failure_uses_currently_feasible_previous_action(self) -> None:
        logits = np.array([[3.0, 2.0, 1.0]])
        mask = np.ones_like(logits, dtype=bool)
        evaluator = _Evaluator(1, lambda action: _report(action == (2,), score=0.5 if action == (2,) else 1.0))
        saved_loads = evaluator._last_resource_loads
        with self._patch_search(1):
            result = repair_module.repair_topk_action(
                masked_logits=logits, connectivity_mask=mask, evaluator=evaluator,
                time_index=0, previous_action=np.array([2]), top_k=2, max_iterations=1, beam_width=1,
            )
        np.testing.assert_array_equal(result.actions, [2])
        self.assertFalse(result.diagnostics["repair_succeeded"])
        self.assertTrue(result.diagnostics["previous_action_fallback_used"])
        self.assertTrue(result.diagnostics["final_service_feasible"])
        self.assertEqual(result.diagnostics["search_stop_reason"], "previous_action_fallback_feasible")
        self.assertEqual(evaluator.calls, [(0,), (1,), (2,)])
        self.assertIs(evaluator._last_resource_loads, saved_loads)

    def test_bounded_failure_rejects_infeasible_previous_action(self) -> None:
        logits = np.array([[3.0, 2.0, 1.0]])
        mask = np.ones_like(logits, dtype=bool)
        evaluator = _Evaluator(1, lambda action: _report(False))
        with self._patch_search(1):
            result = repair_module.repair_topk_action(
                masked_logits=logits, connectivity_mask=mask, evaluator=evaluator,
                time_index=0, previous_action=np.array([2]), top_k=2, max_iterations=1, beam_width=1,
            )
        np.testing.assert_array_equal(result.actions, [0])
        self.assertFalse(result.diagnostics["previous_action_fallback_used"])
        self.assertEqual(evaluator.calls, [(0,), (1,), (2,)])

    def test_bounded_failure_without_previous_action_preserves_greedy_fallback(self) -> None:
        logits, mask = self._inputs(1)
        evaluator = _Evaluator(1, lambda action: _report(False))
        with self._patch_search(1):
            result = repair_module.repair_topk_action(
                masked_logits=logits, connectivity_mask=mask, evaluator=evaluator,
                time_index=0, previous_action=None, top_k=2, max_iterations=1, beam_width=1,
            )
        np.testing.assert_array_equal(result.actions, [0])
        self.assertFalse(result.diagnostics["previous_action_fallback_used"])
        self.assertEqual(evaluator.calls, [(0,), (1,)])

    def test_bounded_failure_rejects_previous_action_invalid_under_current_mask(self) -> None:
        logits = np.array([[3.0, 2.0, 1.0]])
        mask = np.array([[True, True, False]])
        evaluator = _Evaluator(1, lambda action: _report(action == (2,), score=0.5))
        with self._patch_search(1):
            result = repair_module.repair_topk_action(
                masked_logits=logits, connectivity_mask=mask, evaluator=evaluator,
                time_index=0, previous_action=np.array([2]), top_k=2, max_iterations=1, beam_width=1,
            )
        np.testing.assert_array_equal(result.actions, [0])
        self.assertFalse(result.diagnostics["previous_action_fallback_used"])
        self.assertEqual(evaluator.calls, [(0,), (1,)])

    def test_bounded_repair_runtime_abort_stops_before_next_candidate(self) -> None:
        logits, mask = self._inputs(1)
        evaluator = _Evaluator(1, lambda action: _report(False))
        with self._patch_search(1):
            result = repair_module.repair_topk_action(
                masked_logits=logits, connectivity_mask=mask, evaluator=evaluator,
                time_index=0, previous_action=None, top_k=2, max_iterations=1, beam_width=1,
                abort_requested=lambda: len(evaluator.calls) >= 1,
            )
        np.testing.assert_array_equal(result.actions, [0])
        self.assertEqual(evaluator.calls, [(0,)])
        self.assertTrue(result.diagnostics["runtime_aborted"])
        self.assertEqual(result.diagnostics["search_stop_reason"], "runtime_aborted")

    def test_feasible_greedy_returns_immediately(self) -> None:
        logits, mask = self._inputs(1)
        evaluator = _Evaluator(1, lambda action: _report(True))
        with self._patch_search(1):
            result = repair_module.repair_topk_action_deadline(
                masked_logits=logits, connectivity_mask=mask, evaluator=evaluator,
                time_index=0, previous_action=None, deadline_ns=None,
            )
        np.testing.assert_array_equal(result.actions, [0])
        self.assertFalse(result.diagnostics["repair_triggered"])
        self.assertEqual(result.diagnostics["candidate_evaluations"], 0)

    def test_expired_deadline_does_not_start_candidate_evaluation(self) -> None:
        clock = _Clock(100)

        def evaluate(action):
            clock.now_ns = 200
            return _report(False)

        logits, mask = self._inputs(1)
        evaluator = _Evaluator(1, evaluate)
        saved_loads = evaluator._last_resource_loads
        with self._patch_search(1), patch.object(repair_module.time, "perf_counter_ns", side_effect=clock):
            result = repair_module.repair_topk_action_deadline(
                masked_logits=logits, connectivity_mask=mask, evaluator=evaluator,
                time_index=0, previous_action=None, deadline_ns=150,
            )
        np.testing.assert_array_equal(result.actions, [0])
        self.assertEqual(evaluator.calls, [(0,)])
        self.assertTrue(result.diagnostics["deadline_hit"])
        self.assertEqual(result.diagnostics["candidate_evaluations"], 0)
        self.assertIs(evaluator._last_resource_loads, saved_loads)

    def test_atomic_candidate_finishes_and_infeasible_result_falls_back(self) -> None:
        clock = _Clock(100)

        def evaluate(action):
            if action == (1,):
                clock.now_ns = 200
            return _report(False)

        logits, mask = self._inputs(1)
        evaluator = _Evaluator(1, evaluate)
        saved_loads = evaluator._last_resource_loads
        with self._patch_search(1), patch.object(repair_module.time, "perf_counter_ns", side_effect=clock):
            result = repair_module.repair_topk_action_deadline(
                masked_logits=logits, connectivity_mask=mask, evaluator=evaluator,
                time_index=0, previous_action=None, deadline_ns=150,
            )
        np.testing.assert_array_equal(result.actions, [0])
        self.assertEqual(evaluator.calls, [(0,), (1,)])
        self.assertEqual(result.diagnostics["candidate_evaluations"], 1)
        self.assertEqual(result.diagnostics["deadline_overshoot_ms"], 0.00005)
        self.assertIs(evaluator._last_resource_loads, saved_loads)

    def test_atomic_candidate_can_succeed_after_deadline(self) -> None:
        clock = _Clock(100)

        def evaluate(action):
            if action == (1,):
                clock.now_ns = 200
                return _report(True, 0.5)
            return _report(False)

        logits, mask = self._inputs(1)
        evaluator = _Evaluator(1, evaluate)
        with self._patch_search(1), patch.object(repair_module.time, "perf_counter_ns", side_effect=clock):
            result = repair_module.repair_topk_action_deadline(
                masked_logits=logits, connectivity_mask=mask, evaluator=evaluator,
                time_index=0, previous_action=None, deadline_ns=150,
            )
        np.testing.assert_array_equal(result.actions, [1])
        self.assertTrue(result.diagnostics["repair_succeeded"])
        self.assertTrue(result.diagnostics["deadline_hit"])
        self.assertGreater(result.diagnostics["deadline_overshoot_ms"], 0)

    def test_multi_rh_repair_continues_to_later_depth(self) -> None:
        logits, mask = self._inputs(2)
        evaluator = _Evaluator(2, lambda action: _report(action == (1, 1)))
        with self._patch_search(2):
            result = repair_module.repair_topk_action_deadline(
                masked_logits=logits, connectivity_mask=mask, evaluator=evaluator,
                time_index=0, previous_action=None, top_k=2, max_iterations=1, beam_width=1,
            )
        np.testing.assert_array_equal(result.actions, [1, 1])
        self.assertEqual(result.diagnostics["search_depth_reached"], 2)
        self.assertEqual(result.diagnostics["num_actions_changed"], 2)

    def test_frontier_expands_lower_violation_state_first(self) -> None:
        logits, mask = self._inputs(3)
        scores = {(1, 0, 0): 3.0, (0, 1, 0): 1.0, (0, 0, 1): 2.0}
        evaluator = _Evaluator(3, lambda action: _report(False, scores.get(action, 10.0)))
        with self._patch_search(3):
            repair_module.repair_topk_action_deadline(
                masked_logits=logits, connectivity_mask=mask, evaluator=evaluator,
                time_index=0, previous_action=None, top_k=2, beam_width=1,
            )
        # After the root's three candidates, the lower-violation (0, 1, 0)
        # state is popped first, so its first unseen child is evaluated next.
        self.assertEqual(evaluator.calls[4], (1, 1, 0))

    def test_low_ranked_branch_remains_available_until_it_finds_repair(self) -> None:
        logits, mask = self._inputs(3)

        def implicated(evaluator, time_index, action):
            key = tuple(int(value) for value in action)
            return [0, 1, 2] if key == (0, 0, 0) else ([1] if key == (0, 0, 1) else [])

        def evaluate(action):
            if action == (0, 0, 1):
                return _report(False, 30.0)
            return _report(action == (0, 1, 1), 1.0)

        evaluator = _Evaluator(3, evaluate)
        with self._patch_search(3), patch.object(repair_module, "_implicated_rhs", side_effect=implicated):
            result = repair_module.repair_topk_action_deadline(
                masked_logits=logits, connectivity_mask=mask, evaluator=evaluator,
                time_index=0, previous_action=None, top_k=2, beam_width=1,
            )
        np.testing.assert_array_equal(result.actions, [0, 1, 1])
        self.assertTrue(result.diagnostics["repair_succeeded"])
        self.assertIn((0, 0, 1), evaluator.calls)

    def test_frontier_safety_ceiling_discards_only_overflow_states(self) -> None:
        logits, mask = self._inputs(3)
        evaluator = _Evaluator(3, lambda action: _report(False))
        with self._patch_search(3), patch.object(repair_module, "_DEADLINE_MAX_BEAM_WIDTH", 2):
            result = repair_module.repair_topk_action_deadline(
                masked_logits=logits, connectivity_mask=mask, evaluator=evaluator,
                time_index=0, previous_action=None, top_k=2,
            )
        self.assertLessEqual(result.diagnostics["max_beam_size_reached"], 2)

    def test_candidate_evaluation_safety_ceiling_returns_original_action(self) -> None:
        logits, mask = self._inputs(2)
        evaluator = _Evaluator(2, lambda action: _report(False))
        with self._patch_search(2), patch.object(repair_module, "_DEADLINE_MAX_CANDIDATE_EVALUATIONS", 2):
            result = repair_module.repair_topk_action_deadline(
                masked_logits=logits, connectivity_mask=mask, evaluator=evaluator,
                time_index=0, previous_action=None, top_k=2,
            )
        np.testing.assert_array_equal(result.actions, [0, 0])
        self.assertEqual(result.diagnostics["candidate_evaluations"], 2)
        self.assertFalse(result.diagnostics["repair_succeeded"])

    def test_growing_search_never_evaluates_a_joint_action_twice(self) -> None:
        logits, mask = self._inputs(3)
        evaluator = _Evaluator(3, lambda action: _report(False))
        with self._patch_search(3):
            result = repair_module.repair_topk_action_deadline(
                masked_logits=logits, connectivity_mask=mask, evaluator=evaluator,
                time_index=0, previous_action=None, top_k=2, max_iterations=1, beam_width=1,
            )
        candidates = evaluator.calls[1:]
        self.assertEqual(len(candidates), len(set(candidates)))
        self.assertEqual(result.diagnostics["candidate_evaluations"], len(candidates))
        self.assertGreater(result.diagnostics["search_depth_reached"], 1)

    def test_absolute_deadline_includes_time_used_before_repair(self) -> None:
        clock = _Clock(80)

        def evaluate(action):
            if action == (1,):
                clock.now_ns = 110
            return _report(False)

        logits, mask = self._inputs(1)
        evaluator = _Evaluator(1, evaluate)
        with self._patch_search(1), patch.object(repair_module.time, "perf_counter_ns", side_effect=clock):
            result = repair_module.repair_topk_action_deadline(
                masked_logits=logits, connectivity_mask=mask, evaluator=evaluator,
                time_index=0, previous_action=None, decision_start_ns=0, deadline_ns=100,
            )
        self.assertTrue(result.diagnostics["deadline_hit"])
        self.assertEqual(result.diagnostics["decision_budget_ms"], 0.0001)
        self.assertEqual(evaluator.calls, [(0,), (1,)])




if __name__ == "__main__":
    unittest.main()

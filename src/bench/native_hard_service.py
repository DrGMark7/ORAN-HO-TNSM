"""Direct feasibility audit for the native hard-service CPLEX rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.bench.optimizer import MilpInstance


NATIVE_ROW_TOLERANCE = 1e-9


@dataclass(frozen=True)
class NativeRowViolation:
    kind: str
    margin: float
    row: str


@dataclass(frozen=True)
class NativeHardServiceAudit:
    feasible: bool
    first_violation: NativeRowViolation | None
    worst_margins: Mapping[str, NativeRowViolation]
    checked_rows: int


def audit_native_hard_service_instance(
    instance: "MilpInstance",
    *,
    tolerance: float = NATIVE_ROW_TOLERANCE,
) -> NativeHardServiceAudit:
    """Evaluate a submitted action assignment against every explicit native row.

    Continuous delay variables are assigned their minimum upper-PWL values;
    handover and reconfiguration auxiliaries are assigned their exact binary
    conjunctions. Those assignments are the objective-minimizing completion of
    the policy action trace.
    """

    if instance.optimization_mode != "hard_service_feasibility":
        raise ValueError("native hard-service audit requires hard_service_feasibility")
    if instance.mip_start_action.size == 0:
        raise ValueError("native hard-service audit requires a MIP-start action trace")

    selected = instance.mip_start_action.reshape(instance.horizon, instance.num_rhs)
    first: NativeRowViolation | None = None
    worst: dict[str, NativeRowViolation] = {}
    checked = 0

    def record(kind: str, margin: float, row: str) -> None:
        nonlocal first, checked
        checked += 1
        violation = NativeRowViolation(kind, float(margin), row)
        if kind not in worst or margin < worst[kind].margin:
            worst[kind] = violation
        if first is None and margin < -tolerance:
            first = violation

    max_delay = float(instance.upper_values[-1])
    es_delays = np.zeros((instance.horizon, instance.num_es), dtype=np.float64)
    rc_delays = np.zeros((instance.horizon, instance.num_rc), dtype=np.float64)

    for t in range(instance.horizon):
        chosen = np.zeros(instance.num_actions, dtype=np.int8)
        for rh in range(instance.num_rhs):
            action = int(selected[t, rh])
            in_owner_range = (
                int(instance.action_offsets[rh])
                <= action
                < int(instance.action_offsets[rh + 1])
            )
            record(
                "action_owner",
                0.0 if in_owner_range else -1.0,
                f"t={t},rh={rh},action={action}",
            )
            if in_owner_range:
                chosen[action] = 1
            one = int(chosen[int(instance.action_offsets[rh]):int(instance.action_offsets[rh + 1])].sum())
            record("one_action", -abs(one - 1), f"t={t},rh={rh},sum={one}")

        for es in range(instance.num_es):
            capacity = float(instance.es_capacity[es])
            load = sum(
                float(instance.es_load[t, action])
                for action in np.flatnonzero(chosen)
                if int(instance.action_es[action]) == es
            )
            rho = load / capacity
            rho_upper = float(instance.es_utilization_upper[t, es])
            record("es_rho_lower", rho, f"t={t},es={es},rho={rho:.17g}")
            record("es_rho_upper", rho_upper - rho, f"t={t},es={es},rho={rho:.17g},upper={rho_upper:.17g}")
            record("es_capacity", capacity - load, f"t={t},es={es},load={load:.17g},capacity={capacity:.17g}")
            delay = float(np.interp(rho, instance.upper_rho_points, instance.upper_values))
            es_delays[t, es] = delay
            record("es_delay_lower", delay, f"t={t},es={es},delay={delay:.17g}")
            record("es_delay_upper", max_delay - delay, f"t={t},es={es},delay={delay:.17g},upper={max_delay:.17g}")
            if rho_upper <= instance.utilization_cap + 1e-12:
                for i in range(instance.upper_rho_points.size - 1):
                    left = float(instance.upper_rho_points[i])
                    right = float(instance.upper_rho_points[i + 1])
                    if left >= instance.utilization_cap - 1e-12:
                        break
                    slope = float(
                        (instance.upper_values[i + 1] - instance.upper_values[i])
                        / (right - left)
                    )
                    line = slope * rho + float(instance.upper_values[i]) - slope * left
                    record("es_delay_chord", delay - line, f"t={t},es={es},segment={i},delay={delay:.17g},line={line:.17g}")
                    if right >= rho_upper - 1e-12:
                        break
            else:
                record("es_delay_pwl_equality", 0.0, f"t={t},es={es},rho={rho:.17g},delay={delay:.17g}")

        for rc in range(instance.num_rc):
            capacity = float(instance.rc_capacity[rc])
            load = sum(
                float(instance.rc_load[t, action])
                for action in np.flatnonzero(chosen)
                if int(instance.action_rc[action]) == rc
            )
            rho = load / capacity
            rho_upper = float(instance.rc_utilization_upper[t, rc])
            record("rc_rho_lower", rho, f"t={t},rc={rc},rho={rho:.17g}")
            record("rc_rho_upper", rho_upper - rho, f"t={t},rc={rc},rho={rho:.17g},upper={rho_upper:.17g}")
            record("rc_capacity", capacity - load, f"t={t},rc={rc},load={load:.17g},capacity={capacity:.17g}")
            delay = float(np.interp(rho, instance.upper_rho_points, instance.upper_values))
            rc_delays[t, rc] = delay
            record("rc_delay_lower", delay, f"t={t},rc={rc},delay={delay:.17g}")
            record("rc_delay_upper", max_delay - delay, f"t={t},rc={rc},delay={delay:.17g},upper={max_delay:.17g}")
            if rho_upper <= instance.utilization_cap + 1e-12:
                for i in range(instance.upper_rho_points.size - 1):
                    left = float(instance.upper_rho_points[i])
                    right = float(instance.upper_rho_points[i + 1])
                    if left >= instance.utilization_cap - 1e-12:
                        break
                    slope = float(
                        (instance.upper_values[i + 1] - instance.upper_values[i])
                        / (right - left)
                    )
                    line = slope * rho + float(instance.upper_values[i]) - slope * left
                    record("rc_delay_chord", delay - line, f"t={t},rc={rc},segment={i},delay={delay:.17g},line={line:.17g}")
                    if right >= rho_upper - 1e-12:
                        break
            else:
                record("rc_delay_pwl_equality", 0.0, f"t={t},rc={rc},rho={rho:.17g},delay={delay:.17g}")

        for rh in range(instance.num_rhs):
            selected_action = int(selected[t, rh])
            for es in range(instance.num_es):
                load = (
                    float(instance.fronthaul_load[t, selected_action])
                    if instance.action_gateway[selected_action]
                    and int(instance.action_es[selected_action]) == es
                    else 0.0
                )
                capacity = float(instance.fronthaul_capacity[rh * instance.num_es + es])
                record("fronthaul_capacity", capacity - load, f"t={t},rh={rh},es={es},load={load:.17g},capacity={capacity:.17g}")
            for rc in range(instance.num_rc):
                load = (
                    float(instance.direct_load[t, selected_action])
                    if not instance.action_gateway[selected_action]
                    and int(instance.action_rc[selected_action]) == rc
                    else 0.0
                )
                capacity = float(instance.direct_capacity[rh * instance.num_rc + rc])
                record("direct_capacity", capacity - load, f"t={t},rh={rh},rc={rc},load={load:.17g},capacity={capacity:.17g}")

        for es in range(instance.num_es):
            for rc in range(instance.num_rc):
                load = sum(
                    float(instance.midhaul_load[t, action])
                    for action in np.flatnonzero(chosen)
                    if instance.action_gateway[action]
                    and int(instance.action_es[action]) == es
                    and int(instance.action_rc[action]) == rc
                )
                capacity = float(instance.midhaul_capacity[es * instance.num_rc + rc])
                record("midhaul_capacity", capacity - load, f"t={t},es={es},rc={rc},load={load:.17g},capacity={capacity:.17g}")

        for rh in range(instance.num_rhs):
            selected_action = int(selected[t, rh])
            for action in range(int(instance.action_offsets[rh]), int(instance.action_offsets[rh + 1])):
                latency = float(
                    instance.action_transport_latency[action]
                    + instance.action_es_fraction[action]
                    * es_delays[t, int(instance.action_es[action])]
                    + instance.action_rc_fraction[action]
                    * rc_delays[t, int(instance.action_rc[action])]
                )
                deadline = float(instance.rh_deadline_ms[rh])
                maximum_latency = float(
                    instance.action_transport_latency[action]
                    + instance.action_es_fraction[action] * max_delay
                    + instance.action_rc_fraction[action] * max_delay
                )
                big_m = max(0.0, maximum_latency - deadline) + 1e-9
                rhs = deadline + big_m * (action != selected_action)
                record(
                    "hard_latency_upper_pwl",
                    rhs - latency,
                    f"t={t},rh={rh},action={action},encoded={instance.action_encoded_id[action]},selected={action == selected_action},latency={latency:.17g},rhs={rhs:.17g},deadline={deadline:.17g}",
                )

    selected_pairs = {
        (t, int(selected[t, rh]))
        for t in range(instance.horizon)
        for rh in range(instance.num_rhs)
    }
    for i in range(instance.ho_time.size):
        t = int(instance.ho_time[i])
        a = int(instance.ho_action_a[i])
        b = int(instance.ho_action_b[i])
        lhs = float((t, a) in selected_pairs and (t, b) in selected_pairs)
        rhs = float((t, a) in selected_pairs) + float((t, b) in selected_pairs) - 1.0
        record("handover_auxiliary", lhs - rhs, f"i={i},t={t},a={a},b={b},z={lhs:.0f},rhs={rhs:.0f}")

    if instance.reconf_enabled:
        for t in range(1, instance.horizon):
            for rh in range(instance.num_rhs):
                previous = int(selected[t - 1, rh])
                current = int(selected[t, rh])
                # Every generated q row is a lower bound on a binary
                # conjunction. Assigning q to that conjunction satisfies all
                # rows; zero-cost action pairs have no q variable or row.
                record("reconfiguration_auxiliary", 0.0, f"t={t},rh={rh},previous={previous},current={current}")

    return NativeHardServiceAudit(
        feasible=first is None,
        first_violation=first,
        worst_margins=worst,
        checked_rows=checked,
    )

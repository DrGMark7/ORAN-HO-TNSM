"""Graph-size normalization helpers for controlled PPO ablations."""

from __future__ import annotations

from typing import Final

from torch import Tensor


REFERENCE_NUM_RH: Final[int] = 8
ENTROPY_SCALE_MODES: Final[tuple[str, ...]] = ("legacy_sum", "reference_8rh")
REWARD_GRAPH_SCALE_MODES: Final[tuple[str, ...]] = (
    "legacy_total_cost",
    "reference_8rh",
)


def reference_8rh_factor(num_rh: int) -> float:
    """Return the reference graph-size factor ``8 / num_rh``."""

    if num_rh <= 0:
        raise ValueError("num_rh must be positive")
    return REFERENCE_NUM_RH / float(num_rh)


def entropy_scaling_factor(mode: str, num_rh: int) -> float:
    """Return the multiplier applied to summed per-RH entropy."""

    if mode == "legacy_sum":
        return 1.0
    if mode == "reference_8rh":
        return reference_8rh_factor(num_rh)
    raise ValueError(f"entropy_scale_mode must be one of {ENTROPY_SCALE_MODES}")


def scale_entropy(entropy_sum: Tensor, mode: str, num_rh: int) -> Tensor:
    """Scale joint entropy without changing its autograd path."""

    return entropy_sum * entropy_scaling_factor(mode, num_rh)


def reward_scaling_factor(mode: str, num_rh: int) -> float:
    """Return the multiplier applied to raw cost before reward transformation."""

    if mode == "legacy_total_cost":
        return 1.0
    if mode == "reference_8rh":
        return reference_8rh_factor(num_rh)
    raise ValueError(f"reward_graph_scale_mode must be one of {REWARD_GRAPH_SCALE_MODES}")


def reward_input_cost(total_cost: float, mode: str, num_rh: int) -> float:
    """Return reward-transform input while preserving the raw scientific cost."""

    return float(total_cost) * reward_scaling_factor(mode, num_rh)

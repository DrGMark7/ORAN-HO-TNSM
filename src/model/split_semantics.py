"""Centralized functional-split semantics.

Split IDs are implementation details. Code that needs split behavior should use
this module instead of hard-coding numeric IDs such as ``split_id == 3``.

Legacy 4-split configs may still use ``S4``. It is treated as gateway-mode
centralized split, equivalent to ``S4G``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CANONICAL_DIRECT_ES_ID = 0
LEGACY_SPLIT_NAMES_4 = ("S1", "S2", "S3", "S4")
SPLIT_NAMES_5 = ("S1", "S2", "S3", "S4G", "S4D")


@dataclass(frozen=True)
class SplitSemantics:
    """Behavior flags for one functional split option."""

    name: str
    path_type: str
    uses_es_compute: bool
    uses_midhaul: bool
    uses_direct_rh_rc: bool
    pdcp_anchor_type: str
    rlc_anchor_type: str
    canonical_es_id: int | None = None


def infer_split_names(num_splits: int, names: list[str] | tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Return explicit or legacy split names for supported split counts."""

    if names:
        split_names = tuple(str(name) for name in names)
        if len(split_names) != num_splits:
            raise ValueError("split name count must equal num_splits")
        return split_names
    if num_splits == 4:
        return LEGACY_SPLIT_NAMES_4
    if num_splits == 5:
        return SPLIT_NAMES_5
    raise ValueError("supported phase-1 split counts are 4 or 5")


def normalize_split_name(
    split_id_or_name: int | str,
    split_names: tuple[str, ...] | list[str] | None = None,
) -> str:
    """Return uppercase semantic split name for an ID or name."""

    if isinstance(split_id_or_name, str):
        return split_id_or_name.upper()
    if split_names is None:
        raise ValueError("split_names are required when resolving split IDs")
    split_id = int(split_id_or_name)
    if split_id < 0 or split_id >= len(split_names):
        raise ValueError(f"split_id out of range: {split_id}")
    return str(split_names[split_id]).upper()


def semantics_for_split(
    split_id_or_name: int | str,
    split_names: tuple[str, ...] | list[str] | None = None,
) -> SplitSemantics:
    """Return semantic flags for one split."""

    name = normalize_split_name(split_id_or_name, split_names)
    if name == "S1":
        return SplitSemantics(
            name=name,
            path_type="gateway",
            uses_es_compute=True,
            uses_midhaul=True,
            uses_direct_rh_rc=False,
            pdcp_anchor_type="rc",
            rlc_anchor_type="es",
        )
    if name in {"S2", "S3"}:
        return SplitSemantics(
            name=name,
            path_type="gateway",
            uses_es_compute=True,
            uses_midhaul=True,
            uses_direct_rh_rc=False,
            pdcp_anchor_type="rc",
            rlc_anchor_type="rc",
        )
    if name in {"S4", "S4G"}:
        return SplitSemantics(
            name=name,
            path_type="gateway",
            uses_es_compute=False,
            uses_midhaul=True,
            uses_direct_rh_rc=False,
            pdcp_anchor_type="rc",
            rlc_anchor_type="rc",
        )
    if name == "S4D":
        return SplitSemantics(
            name=name,
            path_type="direct",
            uses_es_compute=False,
            uses_midhaul=False,
            uses_direct_rh_rc=True,
            pdcp_anchor_type="rc",
            rlc_anchor_type="rc",
            canonical_es_id=CANONICAL_DIRECT_ES_ID,
        )
    raise ValueError(f"unsupported split name: {name}")


def contains_direct_rc_split(split_names: tuple[str, ...] | list[str]) -> bool:
    return any(is_direct_rc_split(name) for name in split_names)


def is_gateway_split(split_id_or_name: int | str, split_names: tuple[str, ...] | list[str] | None = None) -> bool:
    return semantics_for_split(split_id_or_name, split_names).path_type == "gateway"


def is_direct_rc_split(split_id_or_name: int | str, split_names: tuple[str, ...] | list[str] | None = None) -> bool:
    return semantics_for_split(split_id_or_name, split_names).uses_direct_rh_rc


def uses_es_compute(split_id_or_name: int | str, split_names: tuple[str, ...] | list[str] | None = None) -> bool:
    return semantics_for_split(split_id_or_name, split_names).uses_es_compute


def uses_midhaul(split_id_or_name: int | str, split_names: tuple[str, ...] | list[str] | None = None) -> bool:
    return semantics_for_split(split_id_or_name, split_names).uses_midhaul


def uses_direct_rh_rc(split_id_or_name: int | str, split_names: tuple[str, ...] | list[str] | None = None) -> bool:
    return semantics_for_split(split_id_or_name, split_names).uses_direct_rh_rc


def rlc_anchor_type(split_id_or_name: int | str, split_names: tuple[str, ...] | list[str] | None = None) -> str:
    return semantics_for_split(split_id_or_name, split_names).rlc_anchor_type


def pdcp_anchor_type(split_id_or_name: int | str, split_names: tuple[str, ...] | list[str] | None = None) -> str:
    return semantics_for_split(split_id_or_name, split_names).pdcp_anchor_type


def canonical_es_id_for_split(
    split_id_or_name: int | str,
    split_names: tuple[str, ...] | list[str] | None = None,
) -> int | None:
    return semantics_for_split(split_id_or_name, split_names).canonical_es_id


def split_path_type(split_id_or_name: int | str, split_names: tuple[str, ...] | list[str] | None = None) -> str:
    return semantics_for_split(split_id_or_name, split_names).path_type


def validate_split_coefficients(split_names: tuple[str, ...], coefficients: dict[str, Any]) -> None:
    """Validate semantic expectations that should not be silently violated."""

    for split_id, split_name in enumerate(split_names):
        semantics = semantics_for_split(split_name)
        es_compute = float(coefficients["es_compute_per_mbps"][split_id])
        midhaul_bw = float(coefficients["midhaul_bw_per_mbps"][split_id])
        if split_name.upper() in {"S4G", "S4D"} and not semantics.uses_es_compute and abs(es_compute) > 1e-12:
            raise ValueError(f"{split_name} must have zero ES compute coefficient")
        if semantics.uses_direct_rh_rc and abs(midhaul_bw) > 1e-12:
            raise ValueError(f"{split_name} must have zero midhaul bandwidth coefficient")

#!/usr/bin/env python3
"""Train one fixed cell of the GPPO/ST-GNN four-way ablation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

SCRIPTS_DIR = str(Path(__file__).resolve().parents[2])
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from scripts.ablation.train_gppo_prebuilt_metrics import main as gppo_main
from scripts.ablation.train_recurrent_prebuilt_metrics import main as stgnn_main


CELL_CONFIG: dict[str, tuple[Callable[[], None], str, str]] = {
    "gppo_no_ho": (gppo_main, "no_handover", "baseline_no_ho"),
    "gppo_ho": (gppo_main, "handover", "handover_aware"),
    "stgnn_no_ho": (stgnn_main, "no_handover", "baseline_no_ho"),
    "stgnn_ho": (stgnn_main, "handover", "handover_aware"),
}


def dispatch(cell: str, argv: list[str]) -> tuple[Callable[[], None], list[str]]:
    """Return the existing trainer and validated arguments for one cell."""

    main, feature_mode, reward_mode = CELL_CONFIG[cell]
    forwarded = list(argv)
    fixed = {"--feature_mode": feature_mode, "--reward_mode": reward_mode, "--ablation_cell": cell}
    for flag, expected in fixed.items():
        if flag in forwarded:
            index = forwarded.index(flag)
            actual = forwarded[index + 1] if index + 1 < len(forwarded) else None
            if actual != expected:
                raise ValueError(f"{cell} requires {flag}={expected}; got {actual}")
        else:
            forwarded.extend((flag, expected))
    return main, forwarded


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--help", action="help")
    parser.add_argument("--cell", choices=tuple(CELL_CONFIG), required=True)
    selected, remaining = parser.parse_known_args()
    trainer_main, forwarded = dispatch(selected.cell, [sys.argv[0], *remaining])
    sys.argv = forwarded
    trainer_main()


if __name__ == "__main__":
    main()

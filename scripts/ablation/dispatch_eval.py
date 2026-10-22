#!/usr/bin/env python3
"""Evaluate four ablation checkpoints under one HO-aware objective."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ablation.eval_gppo_prebuilt_trace import run_prebuilt_gppo_trace
from scripts.ablation.eval_recurrent_prebuilt_trace import run_prebuilt_recurrent_trace


CELLS = {
    "gppo_no_ho": ("gppo", "no_handover"), "gppo_ho": ("gppo", "handover"),
    "stgnn_no_ho": ("stgnn", "no_handover"), "stgnn_ho": ("stgnn", "handover"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True); parser.add_argument("--params", required=True)
    parser.add_argument("--episode_length", type=int, default=288); parser.add_argument("--num_episodes", type=int, required=True)
    for cell in CELLS: parser.add_argument(f"--{cell}_checkpoint", required=True)
    parser.add_argument("--policy", choices=("greedy",), default="greedy"); parser.add_argument("--device", default="auto")
    parser.add_argument("--infeasibility-repair", choices=("none", "topk"), default="none")
    parser.add_argument("--repair-top-k", type=int, default=5)
    parser.add_argument("--repair-max-iterations", type=int, default=5)
    parser.add_argument("--repair-beam-width", type=int, default=4)
    parser.add_argument("--repair-mode", choices=("bounded", "deadline"), default="bounded")
    parser.add_argument("--repair-decision-budget-seconds", type=float)
    parser.add_argument("--output_dir", required=True); parser.add_argument("--config", default="config/default.json")
    parser.add_argument("--seed", type=int, default=13); parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument(
        "--episodes",
        default="all",
        help="'all' or a comma-separated list such as 80,81,90,91",
    )
    parser.add_argument(
        "--stgnn_temporal_mode",
        choices=("recurrent", "reset_each_step"),
        default="recurrent",
    )
    parser.add_argument(
        "--allow_ood_dataset",
        action="store_true",
        help="allow evaluation on a dataset/episode layout different from checkpoint training metadata",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--observation_feature_mode",
        choices=("legacy_rh_only", "causal_resource_v1", "causal_resource_v2"),
        default="legacy_rh_only",
    )
    parser.add_argument(
        "--cells",
        help=(
            "optional comma-separated subset of ablation cells; used by the "
            "dual-observation benchmark to schedule one isolated cell per process"
        ),
    )
    return parser.parse_args()


def validate_metadata(path: Path, cell: str, args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from src.model.model import (
        STGNN_ARCHITECTURE_VERSION,
        STGNN_HIDDEN_RESET_MODE,
        STGNN_PPO_REPLAY_MODE,
        STGNN_TEMPORAL_FUSION_MODE,
    )
    from src.pipeline.data_pipeline import MockCsvDatasetLoader
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    project_config = MockCsvDatasetLoader(args.dataset_dir, args.config).project_config
    expected = {
        "ablation_cell": cell,
        "environment_class": "PrebuiltEpisodeWindowEnv",
        "constraint_formulation_version": project_config["constraint_formulation_version"],
        "observation_feature_mode": args.observation_feature_mode,
    }
    if cell.startswith("stgnn_"):
        expected.update(
            {
                "architecture_version": STGNN_ARCHITECTURE_VERSION,
                "recurrent_sequence_length": args.episode_length,
                "hidden_dim": int(project_config["model"]["hidden_dim"]),
                "num_gru_layers": 1,
                "temporal_fusion_mode": STGNN_TEMPORAL_FUSION_MODE,
                "hidden_reset_mode": STGNN_HIDDEN_RESET_MODE,
                "ppo_replay_mode": STGNN_PPO_REPLAY_MODE,
            }
        )
    if not getattr(args, "allow_ood_dataset", False):
        expected.update({"episode_length": args.episode_length, "num_episodes": args.num_episodes})
    for key, value in expected.items():
        actual = metadata.get(key, "legacy_rh_only") if key == "observation_feature_mode" else metadata.get(key)
        if actual != value:
            raise ValueError(f"{cell} checkpoint metadata mismatch: {key}={actual!r}, expected {value!r}")
    if tuple(metadata.get("split_names", ())) != ("S1", "S2", "S3", "S4G", "S4D"):
        raise ValueError(f"{cell} checkpoint split_names mismatch")
    if (not getattr(args, "allow_ood_dataset", False)
            and Path(metadata.get("dataset_dir", "")).resolve() != Path(args.dataset_dir).resolve()):
        raise ValueError(f"{cell} checkpoint dataset_dir mismatch")
    return metadata


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle: return list(csv.DictReader(handle))


def _summarize(cell: str, output: Path) -> dict[str, Any]:
    episodes, decisions = _read(output / "episode_summary.csv"), _read(output / "rh_decisions.csv")
    mean = lambda name: sum(float(row[name]) for row in episodes) / len(episodes)
    ordered = sorted(decisions, key=lambda row: (int(row["rh_id"]), int(row["absolute_t"])))
    switches = {name: 0 for name in ("action", "split", "es", "rc")}
    previous: dict[int, dict[str, str]] = {}
    for row in ordered:
        rh = int(row["rh_id"])
        if rh in previous:
            for name, column in (("action", "action_id"), ("split", "split_id"), ("es", "es_id"), ("rc", "rc_id")):
                switches[name] += row[column] != previous[rh][column]
        previous[rh] = row
    one_per_step = [row for row in decisions if int(row["rh_id"]) == 0]
    step_mean = lambda name: sum(float(row.get(name, 0.0)) for row in one_per_step) / len(one_per_step)
    split_usage = {name: sum(row["split_name"] == name for row in decisions) / len(decisions)
                   for name in ("S1", "S2", "S3", "S4G", "S4D")}
    return {"row_type": "cell", "cell": cell, "mean_total_cost": mean("mean_total_cost"),
            "overall_feasible_rate": mean("overall_feasible_rate"),
            "connectivity_feasible_rate": mean("connectivity_feasible_rate"),
            "latency_feasible_rate": mean("latency_feasible_rate"), "bandwidth_feasible_rate": mean("bandwidth_feasible_rate"),
            "compute_feasible_rate": mean("compute_feasible_rate"),
            "mean_placement_cost": step_mean("placement_cost"),
            "mean_link_violation_cost": step_mean("link_violation_cost"),
            "mean_compute_overload_cost": step_mean("compute_overload_cost"),
            "mean_handover_state_transfer_cost": step_mean("handover_state_transfer_cost"),
            "mean_handover_sla_cost": step_mean("handover_sla_cost"),
            "s4d_selected_ratio": split_usage["S4D"],
            **{f"split_usage_{name}": value for name, value in split_usage.items()},
            **{f"{k}_switch_count": v for k, v in switches.items()}}


def main() -> None:
    args = parse_args(); root = Path(args.output_dir); root.mkdir(parents=True, exist_ok=True)
    if args.repair_mode == "deadline" and (
        args.infeasibility_repair != "topk"
        or args.repair_decision_budget_seconds is None
        or not math.isfinite(args.repair_decision_budget_seconds)
        or args.repair_decision_budget_seconds <= 0
    ):
        raise ValueError("deadline repair requires --infeasibility-repair topk and a positive decision budget")
    selected_cells = list(CELLS)
    if args.cells:
        selected_cells = [value.strip() for value in args.cells.split(",") if value.strip()]
        unknown = sorted(set(selected_cells).difference(CELLS))
        if not selected_cells or unknown:
            raise ValueError(f"invalid --cells selection: {unknown or selected_cells}")
        if len(selected_cells) != len(set(selected_cells)):
            raise ValueError("--cells contains duplicate cell names")
    rows = []
    for cell in selected_cells:
        architecture, feature_mode = CELLS[cell]
        checkpoint = Path(getattr(args, f"{cell}_checkpoint")); validate_metadata(checkpoint, cell, args)
        output = root / cell
        kwargs = dict(dataset_dir=args.dataset_dir, params_path=args.params, checkpoint_path=checkpoint,
                      episode_length=args.episode_length, num_episodes=args.num_episodes, device=args.device,
                      config_path=args.config, output_dir=output, episodes=args.episodes,
                      policy="greedy", seed=args.seed, verbose=args.verbose,
                      # Schema mismatches remain strict by default. The explicit
                      # OOD opt-in records them as warnings in evaluation output.
                      allow_schema_mismatch=args.allow_ood_dataset)
        kwargs["infeasibility_repair"] = args.infeasibility_repair
        kwargs["repair_top_k"] = args.repair_top_k
        kwargs["repair_max_iterations"] = args.repair_max_iterations
        kwargs["repair_beam_width"] = args.repair_beam_width
        kwargs["repair_mode"] = args.repair_mode
        kwargs["repair_decision_budget_seconds"] = args.repair_decision_budget_seconds
        kwargs["observation_feature_mode"] = args.observation_feature_mode
        kwargs["max_episodes"] = args.max_episodes
        if architecture == "gppo":
            run_prebuilt_gppo_trace(**kwargs, feature_mode=feature_mode, reward_mode="handover_aware")
        else:
            run_prebuilt_recurrent_trace(**kwargs, temporal_mode=args.stgnn_temporal_mode)
        rows.append(_summarize(cell, output))
    # The shared factorial summary is written only by the traditional full
    # four-cell invocation.  Subset invocations write solely inside their
    # cell directory, so multiple cell processes can safely share ``root``.
    if set(selected_cells) != set(CELLS):
        return
    cost = {row["cell"]: float(row["mean_total_cost"]) for row in rows}
    effects = {"delta_ho_gppo": cost["gppo_no_ho"] - cost["gppo_ho"],
               "delta_ho_stgnn": cost["stgnn_no_ho"] - cost["stgnn_ho"],
               "delta_st_no_ho": cost["gppo_no_ho"] - cost["stgnn_no_ho"],
               "delta_st_ho": cost["gppo_ho"] - cost["stgnn_ho"],
               "interaction": (cost["stgnn_no_ho"] - cost["stgnn_ho"]) - (cost["gppo_no_ho"] - cost["gppo_ho"])}
    rows.append({"row_type": "factorial_effect", "cell": "effects", **effects})
    fields = sorted({key for row in rows for key in row}, key=lambda x: (x not in ("row_type", "cell"), x))
    with (root / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (root / "ABLATION_4WAY_REPORT.md").write_text(
        "# Four-Way Ablation\n\nAll policies were evaluated greedily on identical episodes under the HO-aware evaluator. "
        "Positive deltas mean the named addition reduced lower-is-better cost. Positive interaction means HO awareness benefited ST-GNN more than GPPO.\n\n"
        f"```json\n{json.dumps(effects, indent=2)}\n```\n\nSingle-seed results are preliminary.\n", encoding="utf-8")


if __name__ == "__main__": main()

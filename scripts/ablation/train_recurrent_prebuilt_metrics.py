"""Train Method A, minimal Recurrent PPO, on a prebuilt mock-data dataset."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import subprocess
from functools import partial

import numpy as np
import torch
from tqdm import tqdm, trange

from .training_metrics_common import (
    COMMON_FIELDNAMES,
    EVALUATION_DIAGNOSTIC_FIELDS,
    TRAINING_DIAGNOSTIC_FIELDS,
    build_prebuilt_scenario_env_and_model,
    empty_eval_metrics,
    evaluate_greedy_policy,
    evaluate_greedy_policy_diagnostics,
    format_metric,
    scenario_metric_fields,
)
from .training_persistence import (
    CSV_FLUSH_INTERVAL,
    PERIODIC_CHECKPOINT_INTERVAL,
    append_metrics_csv,
    prepare_finetune,
    prepare_resume,
    remove_periodic_checkpoints,
    save_periodic_checkpoint,
    save_resumable_checkpoint,
    validate_finetune_metadata,
    write_metrics_csv,
)
from src.model.model import (
    STGNN_ARCHITECTURE_VERSION,
    STGNN_HIDDEN_RESET_MODE,
    STGNN_PPO_REPLAY_MODE,
    STGNN_TEMPORAL_FUSION_MODE,
)
from src.model.training.recurrent_ppo import (
    HISTORICAL_DISTRIBUTION_IMPLEMENTATION_SOURCE,
    RecurrentPPOConfig,
    RecurrentPPOTrainer,
)
from src.model.runtime_modes import FEATURE_MODES, REWARD_MODES, reward_mode_weights, snapshot_for_feature_mode
from src.utils.ablation_metadata import (
    ablation_fingerprints,
    dataset_identity_metadata,
    mobility_checkpoint_metadata,
)
from src.utils.config import DEFAULT_CONFIG_PATH
from src.utils.timing import format_profile_block, sum_seconds, synchronized_time
from src.model.causal_observation import (
    OBSERVATION_FEATURE_MODES, LEGACY_OBSERVATION_FEATURE_MODE,
    observation_schema_for_mode, rh_feature_names,
)


FIELDNAMES = COMMON_FIELDNAMES + [
    "hidden_state_norm_mean",
    "hidden_state_norm_max",
    "hidden_state_delta_mean",
    "hidden_state_delta_max",
    "temporal_logit_delta_mean",
    "temporal_logit_delta_max",
    "sequence_recompute_logprob_max_abs_error",
    "sequence_recompute_value_max_abs_error",
    "episode_hidden_reset_count",
    "unexpected_hidden_reset_count",
    "hidden_state_reset_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Method A: Recurrent PPO metrics on a prebuilt mock-data dataset")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--episode_length", type=int, required=True)
    parser.add_argument("--num_episodes", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda device such as cuda:0")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output", default="outputs/recurrent_prebuilt_metrics.csv")
    parser.add_argument("--checkpoint_output", default=None)
    parser.add_argument("--feature_mode", choices=FEATURE_MODES, default="handover")
    parser.add_argument("--reward_mode", choices=REWARD_MODES, default="handover_aware")
    parser.add_argument("--observation_feature_mode", choices=OBSERVATION_FEATURE_MODES,
                        default=LEGACY_OBSERVATION_FEATURE_MODE)
    parser.add_argument(
        "--distribution_implementation",
        choices=("vectorized", "historical_non_vectorized"),
        default="vectorized",
        help="PPO distribution path; defaults to the current vectorized implementation",
    )
    parser.add_argument("--ablation_cell", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--verbose", action="store_true", help="print metrics every 10 iterations")
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=1,
        help="run greedy evaluation every N completed iterations unless disabled",
    )
    parser.add_argument(
        "--eval_scenarios",
        choices=("all", "current", "none"),
        default="all",
        help="greedy evaluation scope: all scenarios, current sampled scenario only, or none",
    )
    parser.add_argument(
        "--eval_milestones",
        default=None,
        help="Optional comma-separated evaluation iterations; each must match --eval_interval",
    )
    parser.add_argument("--evaluation_output", default=None)
    parser.add_argument("--resume", action="store_true", help="resume from the latest periodic checkpoint")
    parser.add_argument("--input-weight", dest="input_weight", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ignore-fingerprint", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_interval <= 0:
        raise ValueError("--eval_interval must be a positive integer")
    if args.input_weight is not None and args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.resume and args.input_weight is not None:
        raise ValueError("--resume and --input-weight cannot be used together")
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"--dataset_dir does not exist: {dataset_dir}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = None if args.checkpoint_output is None else Path(args.checkpoint_output)
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and checkpoint_path is None:
        raise ValueError("--resume requires --checkpoint_output")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    project_config, scenarios, model = build_prebuilt_scenario_env_and_model(
        dataset_dir,
        params_path=args.params,
        seed=args.seed,
        episode_length=args.episode_length,
        num_episodes=args.num_episodes,
        project_config_path=args.config,
        reward_mode=args.reward_mode,
        observation_feature_mode=args.observation_feature_mode,
    )
    env = scenarios[0].env
    trainer = RecurrentPPOTrainer(
        env,
        model,
        config=replace(
            RecurrentPPOConfig.from_project_config(project_config),
            distribution_implementation=args.distribution_implementation,
        ),
        device=args.device,
        scenarios=scenarios,
        scenario_seed=args.seed,
        snapshot_transform=partial(snapshot_for_feature_mode, feature_mode=args.feature_mode),
    )

    rows: list[dict[str, float | int]] = []
    historical_rows: list[dict[str, str]] = []
    append_only = args.input_weight is not None
    appended_row_count = 0
    completed_iteration = 0
    target_iterations = args.iterations
    if args.resume:
        assert checkpoint_path is not None
        resume = prepare_resume(output_path, checkpoint_path, FIELDNAMES, trainer)
        rows = resume.rows
        completed_iteration = resume.completed_iteration
        if args.verbose:
            print(f"Resuming from checkpoint: {resume.checkpoint_path}")
            print(f"Resume iteration: {completed_iteration}")
        if completed_iteration >= args.iterations:
            if resume.checkpoint_path != checkpoint_path:
                save_resumable_checkpoint(
                    checkpoint_path,
                    trainer,
                    method="recurrent",
                    metadata=resume.metadata,
                    completed_iteration=completed_iteration,
                )
                remove_periodic_checkpoints(checkpoint_path)
                print(f"Final checkpoint saved: {checkpoint_path}")
            print(
                "Resume checkpoint already satisfies requested iterations: "
                f"{completed_iteration} >= {args.iterations}"
            )
            return

    fine_tune = None
    if args.input_weight is not None:
        fine_tune = prepare_finetune(output_path, Path(args.input_weight), FIELDNAMES, trainer)
        historical_rows = fine_tune.rows
        completed_iteration = fine_tune.completed_iteration
        target_iterations = completed_iteration + args.iterations
        if args.verbose:
            print(f"Fine-tune checkpoint: {fine_tune.checkpoint_path}")
            print(f"Fine-tune continuation: {fine_tune.continuation_mode}")
            print(f"Fine-tune iterations: {completed_iteration} -> {target_iterations}")
            if args.ignore_fingerprint:
                print("Fine-tune warning: dataset fingerprint compatibility check is disabled")

    eval_milestones = (
        None
        if args.eval_milestones is None
        else {int(value) for value in args.eval_milestones.split(",")}
    )
    if eval_milestones is not None and (
        not eval_milestones
        or min(eval_milestones) <= 0
        or max(eval_milestones) > target_iterations
        or any(value % args.eval_interval != 0 for value in eval_milestones)
    ):
        raise ValueError("--eval_milestones must be positive multiples of --eval_interval within the target iteration count")
    if eval_milestones is not None and args.evaluation_output is None:
        raise ValueError("--evaluation_output is required with --eval_milestones")
    evaluation_rows: list[dict[str, float | int]] = []
    initial_actions: np.ndarray | None = None

    def checkpoint_metadata(last_train_metrics: dict[str, float]) -> dict[str, object]:
        """Keep periodic and final checkpoint metadata identical."""

        return {
            "ablation_cell": args.ablation_cell,
            "architecture_mode": "recurrent_stgnn",
            "architecture_version": STGNN_ARCHITECTURE_VERSION,
            "model_name": type(trainer.model).__name__,
            "trainer_name": type(trainer).__name__,
            "dataset_dir": str(dataset_dir),
            "params": args.params,
            "params_path": args.params,
            "config": args.config,
            "seed": args.seed,
            "episode_length": args.episode_length,
            "recurrent_sequence_length": args.episode_length,
            "hidden_dim": int(project_config["model"]["hidden_dim"]),
            "num_gru_layers": int(trainer.model.temporal_encoder.gru.num_layers),
            "temporal_fusion_mode": STGNN_TEMPORAL_FUSION_MODE,
            "hidden_reset_mode": STGNN_HIDDEN_RESET_MODE,
            "ppo_replay_mode": STGNN_PPO_REPLAY_MODE,
            "distribution_implementation": trainer.config.distribution_implementation,
            "historical_distribution_implementation_source": HISTORICAL_DISTRIBUTION_IMPLEMENTATION_SOURCE,
            "observation_schema_version": observation_schema_for_mode(
                env.episode, args.observation_feature_mode
            ),
            "observation_feature_mode": args.observation_feature_mode,
            "resource_observation_schema_version": (
                None
                if args.observation_feature_mode == LEGACY_OBSERVATION_FEATURE_MODE
                else observation_schema_for_mode(env.episode, args.observation_feature_mode)
            ),
            "observation_schema": rh_feature_names(env.episode, args.observation_feature_mode),
            "rh_feature_dim": trainer.model.rh_feature_dim,
            "num_episodes": args.num_episodes,
            "iterations": target_iterations,
            "evaluation_milestones": sorted(eval_milestones) if eval_milestones is not None else None,
            "feature_mode": args.feature_mode,
            "reward_mode": args.reward_mode,
            "num_rh": env.episode.config.num_rhs,
            "num_es": env.episode.config.num_es,
            "num_rc": env.episode.config.num_rc,
            "num_splits": env.episode.config.num_splits,
            "split_names": env.episode.split_demand.split_names,
            "environment_class": type(env).__name__,
            "device": str(trainer.device),
            "ppo_hyperparameters": vars(trainer.config),
            "entropy_scale_mode": trainer.config.entropy_scale_mode,
            "reward_graph_scale_mode": env.evaluator.reward_config.reward_graph_scale_mode,
            "entropy_scaling_factor": last_train_metrics["entropy_scaling_factor"],
            "reward_scaling_factor": last_train_metrics["reward_scaling_factor"],
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad
            ),
            "reward_weights": reward_mode_weights(env.episode, args.reward_mode),
            **env.evaluator.formulation_metadata(),
            "constraint_formulation_version": env.evaluator.formulation_version,
            "reward_transform": env.evaluator.reward_config.transform,
            "latency_objective_mode": env.evaluator.penalties.latency_objective_mode,
            "handover_control_enabled": env.evaluator.formulation_metadata()["handover_control_enabled"],
            "reconfiguration_cost": vars(env.evaluator.reconfiguration),
            **ablation_fingerprints(dataset_dir, args.config),
            **dataset_identity_metadata(dataset_dir),
            **mobility_checkpoint_metadata(dataset_dir, episode_order_mode=env.episode_order_mode),
            "temporal_diagnostics": {
                key: float(rows[-1][key]) if rows else float("nan")
                for key in (
                    "hidden_state_norm_mean",
                    "hidden_state_norm_max",
                    "hidden_state_delta_mean",
                    "hidden_state_delta_max",
                    "temporal_logit_delta_mean",
                    "temporal_logit_delta_max",
                    "sequence_recompute_logprob_max_abs_error",
                    "sequence_recompute_value_max_abs_error",
                    "episode_hidden_reset_count",
                    "unexpected_hidden_reset_count",
                )
            },
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip() or "unknown",
        }

    if fine_tune is not None:
        validate_finetune_metadata(
            fine_tune.metadata, checkpoint_metadata({
                "entropy_scaling_factor": float("nan"),
                "reward_scaling_factor": float(env.evaluator.formulation_metadata()["reward_scaling_factor"]),
            }),
            expected_method="recurrent",
            checkpoint_method=fine_tune.checkpoint_method,
            ignore_fingerprint=args.ignore_fingerprint,
        )

    def persist_rows() -> None:
        nonlocal appended_row_count
        if append_only:
            append_metrics_csv(output_path, FIELDNAMES, rows[appended_row_count:])
            appended_row_count = len(rows)
        else:
            write_metrics_csv(output_path, FIELDNAMES, rows)

    timing_records: list[dict[str, float]] = []
    print(
        f"method=A recurrent-prebuilt scenarios={len(scenarios)} dataset_dir={dataset_dir} "
        f"params={args.params} config={args.config} "
        f"T={env.episode.config.horizon} episode_length={args.episode_length} num_episodes={args.num_episodes} "
        f"RH={env.episode.config.num_rhs} device={trainer.device} "
        f"eval_scenarios={args.eval_scenarios} eval_interval={args.eval_interval}"
    )
    print(f"Training implementation mode: {trainer.config.distribution_implementation}")
    for iteration in trange(
        completed_iteration, target_iterations, desc="recurrent prebuilt training", unit="iter"
    ):
        iter_start = synchronized_time(trainer.device)
        train_start = synchronized_time(trainer.device)
        train_metrics = trainer.train(iterations=1)[0]
        train_elapsed = synchronized_time(trainer.device) - train_start
        completed_iteration = iteration + 1
        should_eval = (
            args.eval_scenarios != "none"
            and completed_iteration % args.eval_interval == 0
            and (eval_milestones is None or completed_iteration in eval_milestones)
        )
        if should_eval:
            eval_start = synchronized_time(trainer.device)
            if eval_milestones is None:
                eval_metrics = evaluate_greedy_policy(
                    trainer,
                    args.seed,
                    recurrent=True,
                    scenario_scope=args.eval_scenarios,
                )
            else:
                eval_metrics, _ = evaluate_greedy_policy_diagnostics(
                    trainer,
                    args.seed,
                    recurrent=True,
                    initial_actions=initial_actions,
                )
                evaluation_rows.append(
                    {
                        "iteration": completed_iteration,
                        "observation_feature_mode": args.observation_feature_mode,
                        "observation_schema_version": observation_schema_for_mode(
                            env.episode, args.observation_feature_mode
                        ),
                        **{
                            field: eval_metrics[field]
                            for field in EVALUATION_DIAGNOSTIC_FIELDS
                            if field in eval_metrics
                        },
                        "approx_kl": float(train_metrics["approx_kl"]),
                        "clip_fraction": float(train_metrics["clip_fraction"]),
                    }
                )
            eval_elapsed = synchronized_time(trainer.device) - eval_start
        else:
            eval_metrics = empty_eval_metrics()
            eval_elapsed = 0.0
        if should_eval:
            eval_metrics["eval_iteration"] = float(completed_iteration)
        iter_elapsed = synchronized_time(trainer.device) - iter_start
        rollout_elapsed = float(train_metrics.get("rollout_seconds", 0.0))
        update_elapsed = float(train_metrics.get("update_seconds", 0.0))
        train_overhead_elapsed = max(train_elapsed - rollout_elapsed - update_elapsed, 0.0)
        update_prepare_elapsed = float(train_metrics.get("update_prepare_seconds", 0.0))
        update_forward_elapsed = float(train_metrics.get("update_forward_seconds", 0.0))
        update_distribution_elapsed = float(train_metrics.get("update_distribution_seconds", 0.0))
        update_loss_elapsed = float(train_metrics.get("update_loss_seconds", 0.0))
        update_backward_elapsed = float(train_metrics.get("update_backward_seconds", 0.0))
        update_optimizer_elapsed = float(train_metrics.get("update_optimizer_seconds", 0.0))
        update_profiled_elapsed = (
            update_prepare_elapsed
            + update_forward_elapsed
            + update_distribution_elapsed
            + update_loss_elapsed
            + update_backward_elapsed
            + update_optimizer_elapsed
        )
        update_other_elapsed = max(update_elapsed - update_profiled_elapsed, 0.0)
        timing_records.append(
            {
                "rollout": rollout_elapsed,
                "ppo_update": update_elapsed,
                "update_prepare": update_prepare_elapsed,
                "update_forward": update_forward_elapsed,
                "update_distribution": update_distribution_elapsed,
                "update_loss": update_loss_elapsed,
                "update_backward": update_backward_elapsed,
                "update_optimizer": update_optimizer_elapsed,
                "update_other": update_other_elapsed,
                "train_overhead": train_overhead_elapsed,
                "train_total": train_elapsed,
                "greedy_eval": eval_elapsed,
                "iteration_total": iter_elapsed,
            }
        )
        row: dict[str, float | int] = {
            "iteration": iteration,
            "observation_feature_mode": args.observation_feature_mode,
            "observation_schema_version": observation_schema_for_mode(
                env.episode, args.observation_feature_mode
            ),
            **scenario_metric_fields(train_metrics, args.params),
            "loss": float(train_metrics.get("loss", np.nan)),
            "policy_loss": float(train_metrics.get("policy_loss", np.nan)),
            "value_loss": float(train_metrics.get("value_loss", np.nan)),
            "entropy": float(train_metrics.get("entropy", np.nan)),
            "approx_kl": float(train_metrics.get("approx_kl", np.nan)),
            "mean_reward": float(train_metrics.get("mean_reward", np.nan)),
            **eval_metrics,
            "avg_valid_action_count_per_rh": float(np.mean(trainer.valid_action_count_per_rh)),
            **{field: train_metrics[field] for field in TRAINING_DIAGNOSTIC_FIELDS},
            "hidden_state_norm_mean": float(train_metrics.get("hidden_state_norm_mean", np.nan)),
            "hidden_state_norm_max": float(train_metrics.get("hidden_state_norm_max", np.nan)),
            "hidden_state_delta_mean": float(train_metrics.get("hidden_state_delta_mean", np.nan)),
            "hidden_state_delta_max": float(train_metrics.get("hidden_state_delta_max", np.nan)),
            "temporal_logit_delta_mean": float(train_metrics.get("temporal_logit_delta_mean", np.nan)),
            "temporal_logit_delta_max": float(train_metrics.get("temporal_logit_delta_max", np.nan)),
            "sequence_recompute_logprob_max_abs_error": float(
                train_metrics.get("sequence_recompute_logprob_max_abs_error", np.nan)
            ),
            "sequence_recompute_value_max_abs_error": float(
                train_metrics.get("sequence_recompute_value_max_abs_error", np.nan)
            ),
            "episode_hidden_reset_count": float(train_metrics.get("episode_hidden_reset_count", np.nan)),
            "unexpected_hidden_reset_count": float(
                train_metrics.get("unexpected_hidden_reset_count", np.nan)
            ),
            "hidden_state_reset_count": float(train_metrics.get("hidden_state_reset_count", np.nan)),
        }
        rows.append(row)
        if completed_iteration % CSV_FLUSH_INTERVAL == 0:
            persist_rows()
            if args.verbose:
                print(f"CSV progress saved through iteration {completed_iteration}: {output_path}")
        if checkpoint_path is not None and completed_iteration % PERIODIC_CHECKPOINT_INTERVAL == 0:
            if completed_iteration % CSV_FLUSH_INTERVAL != 0:
                persist_rows()
                if args.verbose:
                    print(f"CSV progress saved through iteration {completed_iteration}: {output_path}")
            save_periodic_checkpoint(
                checkpoint_path,
                trainer,
                method="recurrent",
                metadata=checkpoint_metadata(train_metrics),
                completed_iteration=completed_iteration,
                log=print if args.verbose else lambda _message: None,
            )
        if args.verbose and (iteration + 1) % 10 == 0:
            tqdm.write(
                "iter={iteration} loss={loss} reward={reward} total_cost={total_cost} "
                "overall={overall} conn={conn}".format(
                    iteration=iteration + 1,
                    loss=format_metric(float(row["loss"])),
                    reward=format_metric(float(row["mean_reward"])),
                    total_cost=format_metric(float(row["mean_total_cost"])),
                    overall=format_metric(float(row["overall_feasible_rate"])),
                    conn=format_metric(float(row["connectivity_feasible_rate"])),
                )
            )

    output_start = synchronized_time(trainer.device)
    persist_rows()
    if args.evaluation_output is not None:
        evaluation_path = Path(args.evaluation_output)
        evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        with evaluation_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVALUATION_DIAGNOSTIC_FIELDS)
            writer.writeheader()
            writer.writerows(evaluation_rows)
    output_elapsed = synchronized_time(trainer.device) - output_start

    print(f"wrote metrics to {output_path} in {output_elapsed:.4f}s")
    checkpoint_elapsed = 0.0
    if checkpoint_path is not None:
        checkpoint_start = synchronized_time(trainer.device)
        save_resumable_checkpoint(
            checkpoint_path,
            trainer,
            method="recurrent",
            metadata=checkpoint_metadata(train_metrics),
            completed_iteration=completed_iteration,
        )
        remove_periodic_checkpoints(checkpoint_path)
        checkpoint_elapsed = synchronized_time(trainer.device) - checkpoint_start
        print(f"Final checkpoint saved: {checkpoint_path}")
    summary_rows = [*historical_rows, *rows]
    if summary_rows:
        first = summary_rows[0]
        last = rows[-1]
        print(
            "\n".join(
                [
                    "metrics_summary:",
                    f"  loss: {format_metric(float(first['loss']))} -> {format_metric(float(last['loss']))}",
                    f"  mean_reward: {format_metric(float(first['mean_reward']))} -> {format_metric(float(last['mean_reward']))}",
                    f"  mean_total_cost: {format_metric(float(first['mean_total_cost']))} -> {format_metric(float(last['mean_total_cost']))}",
                    f"  final_overall_feasible_rate: {format_metric(float(last['overall_feasible_rate']))}",
                    f"  final_connectivity_feasible_rate: {format_metric(float(last['connectivity_feasible_rate']))}",
                ]
            )
        )
    if timing_records:
        total_iter_total = sum_seconds(timing_records, "iteration_total")
        timing_summary_items = [
            ("rollout_collect", sum_seconds(timing_records, "rollout")),
            ("update_prepare", sum_seconds(timing_records, "update_prepare")),
            ("update_forward", sum_seconds(timing_records, "update_forward")),
            ("update_distribution", sum_seconds(timing_records, "update_distribution")),
            ("update_loss", sum_seconds(timing_records, "update_loss")),
            ("update_backward", sum_seconds(timing_records, "update_backward")),
            ("update_optimizer", sum_seconds(timing_records, "update_optimizer")),
            ("update_other", sum_seconds(timing_records, "update_other")),
            ("ppo_update_total", sum_seconds(timing_records, "ppo_update")),
            ("train_overhead", sum_seconds(timing_records, "train_overhead")),
            ("train_total", sum_seconds(timing_records, "train_total")),
            ("greedy_eval", sum_seconds(timing_records, "greedy_eval")),
            ("checkpoint_save", checkpoint_elapsed),
            ("metrics_csv_write", output_elapsed),
            ("iteration_total", total_iter_total),
        ]
        timing_summary_total = sum(seconds for _, seconds in timing_summary_items)
        print(
            format_profile_block(
                "timing_summary_total:",
                timing_summary_items,
                total_seconds=timing_summary_total,
            )
        )
        if total_iter_total > 0.0:
            share_items = [
                ("rollout_collect", sum_seconds(timing_records, "rollout") / total_iter_total),
                ("update_prepare", sum_seconds(timing_records, "update_prepare") / total_iter_total),
                ("update_forward", sum_seconds(timing_records, "update_forward") / total_iter_total),
                ("update_distribution", sum_seconds(timing_records, "update_distribution") / total_iter_total),
                ("update_loss", sum_seconds(timing_records, "update_loss") / total_iter_total),
                ("update_backward", sum_seconds(timing_records, "update_backward") / total_iter_total),
                ("update_optimizer", sum_seconds(timing_records, "update_optimizer") / total_iter_total),
                ("update_other", sum_seconds(timing_records, "update_other") / total_iter_total),
                ("ppo_update_total", sum_seconds(timing_records, "ppo_update") / total_iter_total),
                ("train_overhead", sum_seconds(timing_records, "train_overhead") / total_iter_total),
                ("greedy_eval", sum_seconds(timing_records, "greedy_eval") / total_iter_total),
            ]
            print(
                "\n".join(
                    ["timing_summary_share:"]
                    + [f"  {label:<22} {share:>9.2%}" for label, share in share_items]
                )
            )


if __name__ == "__main__":
    main()

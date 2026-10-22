"""Train the project-compatible five-split GPPO-style baseline on existing data."""

from __future__ import annotations

import argparse
import csv
from functools import partial
from pathlib import Path
import subprocess

import numpy as np
import torch
from tqdm import tqdm, trange

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
from .training_metrics_common import (
    COMMON_FIELDNAMES,
    EVALUATION_DIAGNOSTIC_FIELDS,
    TRAINING_DIAGNOSTIC_FIELDS,
    build_prebuilt_scenario_env_and_gppo_model,
    empty_eval_metrics,
    evaluate_greedy_policy,
    evaluate_greedy_policy_diagnostics,
    format_metric,
)
from src.model.runtime_modes import (
    FEATURE_MODES,
    REWARD_MODES,
    SUPPORTED_SPLIT_MODE,
    snapshot_for_feature_mode,
    validate_gppo_modes,
)
from src.model.training.ppo import PPOConfig, PPOTrainer
from src.utils.ablation_metadata import (
    ablation_fingerprints,
    dataset_identity_metadata,
    mobility_checkpoint_metadata,
)
from src.model.runtime_modes import reward_mode_weights
from src.utils.config import DEFAULT_CONFIG_PATH
from src.utils.timing import format_profile_block, sum_seconds, synchronized_time
from src.model.causal_observation import (
    OBSERVATION_FEATURE_MODES, LEGACY_OBSERVATION_FEATURE_MODE,
    observation_schema_for_mode, rh_feature_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the GPPO-style project-compatible five-split baseline on prebuilt data"
    )
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--episode_length", type=int, required=True)
    parser.add_argument("--num_episodes", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint_output", required=True)
    parser.add_argument("--feature_mode", choices=FEATURE_MODES, required=True)
    parser.add_argument("--reward_mode", choices=REWARD_MODES, required=True)
    parser.add_argument("--observation_feature_mode", choices=OBSERVATION_FEATURE_MODES,
                        default=LEGACY_OBSERVATION_FEATURE_MODE)
    parser.add_argument("--split_mode", default=SUPPORTED_SPLIT_MODE)
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=10,
        help="run greedy evaluation every N completed iterations",
    )
    parser.add_argument("--eval_scenarios", choices=("all", "current", "none"), default="all")
    parser.add_argument("--ablation_cell", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--verbose", action="store_true", help="print metrics every evaluation interval")
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
    validate_gppo_modes(args.feature_mode, args.reward_mode, args.split_mode)
    if args.eval_interval <= 0:
        raise ValueError("--eval_interval must be positive")
    if args.input_weight is not None and args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.resume and args.input_weight is not None:
        raise ValueError("--resume and --input-weight cannot be used together")

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"--dataset_dir does not exist: {dataset_dir}")
    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    project_config, scenarios, model = build_prebuilt_scenario_env_and_gppo_model(
        dataset_dir,
        params_path=args.params,
        seed=args.seed,
        episode_length=args.episode_length,
        num_episodes=args.num_episodes,
        reward_mode=args.reward_mode,
        project_config_path=args.config,
        observation_feature_mode=args.observation_feature_mode,
    )
    env = scenarios[0].env
    trainer = PPOTrainer(
        env,
        model,
        config=PPOConfig.from_project_config(project_config),
        device=args.device,
        snapshot_transform=partial(snapshot_for_feature_mode, feature_mode=args.feature_mode),
    )

    rows: list[dict[str, float | int | str]] = []
    historical_rows: list[dict[str, str]] = []
    append_only = args.input_weight is not None
    appended_row_count = 0
    completed_iteration = 0
    target_iterations = args.iterations
    if args.resume:
        resume = prepare_resume(output_path, checkpoint_path, COMMON_FIELDNAMES, trainer)
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
                    method="ppo",
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
        fine_tune = prepare_finetune(
            output_path, Path(args.input_weight), COMMON_FIELDNAMES, trainer
        )
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

    timing_records: list[dict[str, float]] = []
    checkpoint_metadata = {
        "ablation_cell": args.ablation_cell,
        "architecture_mode": "snapshot_non_recurrent",
        "model_name": type(trainer.model).__name__,
        "trainer_name": type(trainer).__name__,
        "model_type": "gppo",
        "method": "ppo",
        "split_mode": args.split_mode,
        "feature_mode": args.feature_mode,
        "observation_feature_mode": args.observation_feature_mode,
        "observation_schema_version": observation_schema_for_mode(
            env.episode, args.observation_feature_mode
        ),
        "observation_schema": rh_feature_names(env.episode, args.observation_feature_mode),
        "rh_feature_dim": trainer.model.rh_feature_dim,
        "reward_mode": args.reward_mode,
        "num_splits": env.episode.config.num_splits,
        "split_names": env.episode.split_demand.split_names,
        "dataset_dir": str(dataset_dir),
        "episode_length": args.episode_length,
        "num_episodes": args.num_episodes,
        "iterations": target_iterations,
        "evaluation_milestones": sorted(eval_milestones) if eval_milestones is not None else None,
        "params": args.params,
        "params_path": args.params,
        "config": args.config,
        "seed": args.seed,
        "num_rh": env.episode.config.num_rhs,
        "num_es": env.episode.config.num_es,
        "num_rc": env.episode.config.num_rc,
        "environment_class": type(env).__name__,
        "device": str(trainer.device),
        "ppo_hyperparameters": vars(trainer.config),
        "entropy_scale_mode": trainer.config.entropy_scale_mode,
        "reward_graph_scale_mode": env.evaluator.reward_config.reward_graph_scale_mode,
        "entropy_scaling_factor": float("nan"),
        "reward_scaling_factor": float(env.evaluator.formulation_metadata()["reward_scaling_factor"]),
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
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip() or "unknown",
    }
    if fine_tune is not None:
        validate_finetune_metadata(
            fine_tune.metadata,
            checkpoint_metadata,
            expected_method="ppo",
            checkpoint_method=fine_tune.checkpoint_method,
            ignore_fingerprint=args.ignore_fingerprint,
        )

    def persist_rows() -> None:
        nonlocal appended_row_count
        if append_only:
            append_metrics_csv(
                output_path, COMMON_FIELDNAMES, rows[appended_row_count:]
            )
            appended_row_count = len(rows)
        else:
            write_metrics_csv(output_path, COMMON_FIELDNAMES, rows)

    print(
        f"method=GPPO-style project-compatible-5split dataset_dir={dataset_dir} "
        f"params={args.params} config={args.config} "
        f"T={env.episode.config.horizon} episode_length={args.episode_length} "
        f"num_episodes={args.num_episodes} RH={env.episode.config.num_rhs} "
        f"device={trainer.device} feature_mode={args.feature_mode} reward_mode={args.reward_mode} "
        f"eval_interval={args.eval_interval}"
    )
    for iteration in trange(
        completed_iteration, target_iterations, desc="GPPO prebuilt training", unit="iter"
    ):
        iteration_start = synchronized_time(trainer.device)
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
                    trainer, args.seed, scenario_scope=args.eval_scenarios
                )
            else:
                eval_metrics, _ = evaluate_greedy_policy_diagnostics(
                    trainer, args.seed, initial_actions=initial_actions
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
        iteration_elapsed = synchronized_time(trainer.device) - iteration_start
        timing_records.append(
            {
                "ppo_train": train_elapsed,
                "greedy_eval": eval_elapsed,
                "iteration_total": iteration_elapsed,
            }
        )
        row: dict[str, float | int | str] = {
            "iteration": iteration,
            "observation_feature_mode": args.observation_feature_mode,
            "observation_schema_version": observation_schema_for_mode(
                env.episode, args.observation_feature_mode
            ),
            "scenario_id": 0,
            "scenario_name": scenarios[0].scenario_name,
            "params_path": str(args.params),
            "difficulty": "unknown",
            "loss": train_metrics["loss"],
            "policy_loss": train_metrics["policy_loss"],
            "value_loss": train_metrics["value_loss"],
            "entropy": train_metrics["entropy"],
            "approx_kl": train_metrics["approx_kl"],
            "mean_reward": train_metrics["mean_reward"],
            **eval_metrics,
            "avg_valid_action_count_per_rh": float(np.mean(trainer.valid_action_count_per_rh)),
            **{field: train_metrics[field] for field in TRAINING_DIAGNOSTIC_FIELDS},
        }
        rows.append(row)
        checkpoint_metadata["entropy_scaling_factor"] = train_metrics["entropy_scaling_factor"]
        checkpoint_metadata["reward_scaling_factor"] = train_metrics["reward_scaling_factor"]
        if completed_iteration % CSV_FLUSH_INTERVAL == 0:
            persist_rows()
            if args.verbose:
                print(f"CSV progress saved through iteration {completed_iteration}: {output_path}")
        if completed_iteration % PERIODIC_CHECKPOINT_INTERVAL == 0:
            if completed_iteration % CSV_FLUSH_INTERVAL != 0:
                persist_rows()
                if args.verbose:
                    print(f"CSV progress saved through iteration {completed_iteration}: {output_path}")
            save_periodic_checkpoint(
                checkpoint_path,
                trainer,
                method="ppo",
                metadata=checkpoint_metadata,
                completed_iteration=completed_iteration,
                log=print if args.verbose else lambda _message: None,
            )
        if args.verbose and should_eval:
            tqdm.write(
                "iter={iteration} loss={loss} reward={reward} total_cost={total_cost} "
                "overall={overall} conn={conn} train_s={train_s:.3f} eval_s={eval_s:.3f}".format(
                    iteration=iteration + 1,
                    loss=format_metric(float(row["loss"])),
                    reward=format_metric(float(row["mean_reward"])),
                    total_cost=format_metric(float(row["mean_total_cost"])),
                    overall=format_metric(float(row["overall_feasible_rate"])),
                    conn=format_metric(float(row["connectivity_feasible_rate"])),
                    train_s=train_elapsed,
                    eval_s=eval_elapsed,
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

    checkpoint_start = synchronized_time(trainer.device)
    save_resumable_checkpoint(
        checkpoint_path,
        trainer,
        method="ppo",
        metadata=checkpoint_metadata,
        completed_iteration=completed_iteration,
    )
    remove_periodic_checkpoints(checkpoint_path)
    checkpoint_elapsed = synchronized_time(trainer.device) - checkpoint_start
    print(f"wrote metrics to {output_path} in {output_elapsed:.4f}s")
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
        total_iteration_seconds = sum_seconds(timing_records, "iteration_total")
        timing_items = [
            ("ppo_train", sum_seconds(timing_records, "ppo_train")),
            ("greedy_eval", sum_seconds(timing_records, "greedy_eval")),
            ("metrics_csv_write", output_elapsed),
            ("checkpoint_save", checkpoint_elapsed),
            ("iteration_total", total_iteration_seconds),
        ]
        print(
            format_profile_block(
                "timing_summary_total:",
                timing_items,
                total_seconds=sum(seconds for _, seconds in timing_items),
            )
        )
        if total_iteration_seconds > 0.0:
            print(
                "\n".join(
                    [
                        "timing_summary_share:",
                        f"  {'ppo_train':<22} {sum_seconds(timing_records, 'ppo_train') / total_iteration_seconds:>9.2%}",
                        f"  {'greedy_eval':<22} {sum_seconds(timing_records, 'greedy_eval') / total_iteration_seconds:>9.2%}",
                    ]
                )
            )


if __name__ == "__main__":
    main()

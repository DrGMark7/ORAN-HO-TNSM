"""Small synchronous persistence helpers for long-running ablation training."""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Any, Callable

import numpy as np
import torch

from .infer_episode_trace import load_checkpoint_payload, save_model_checkpoint
from src.model.causal_observation import (
    CAUSAL_RESOURCE_OBSERVATION_FEATURE_MODE,
    CAUSAL_RESOURCE_OBSERVATION_SCHEMA,
    LEGACY_OBSERVATION_SCHEMA,
)


CSV_FLUSH_INTERVAL = 50
PERIODIC_CHECKPOINT_INTERVAL = 1_000


@dataclass(frozen=True)
class ResumeProgress:
    checkpoint_path: Path
    completed_iteration: int
    rows: list[dict[str, str]]
    metadata: dict[str, Any]
    continuation_mode: str = "exact_resume"
    checkpoint_method: str | None = None


def write_metrics_csv(
    output_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    """Rewrite the one configured CSV with every completed metric row."""

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_metrics_csv(
    output_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    """Append newly completed rows without rewriting existing fine-tune history."""

    if not rows:
        return
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerows(rows)


def periodic_checkpoint_path(checkpoint_path: Path, completed_iteration: int) -> Path:
    return checkpoint_path.with_name(
        f"{completed_iteration}_{checkpoint_path.stem}{checkpoint_path.suffix}"
    )


def periodic_checkpoints(checkpoint_path: Path) -> list[Path]:
    pattern = re.compile(
        rf"^(?P<iteration>\d+)_{re.escape(checkpoint_path.stem)}{re.escape(checkpoint_path.suffix)}$"
    )
    matches = [
        candidate
        for candidate in checkpoint_path.parent.glob(f"*_{checkpoint_path.stem}{checkpoint_path.suffix}")
        if pattern.match(candidate.name)
    ]
    return sorted(matches, key=lambda candidate: int(pattern.match(candidate.name).group("iteration")))


def capture_training_state(trainer: Any, completed_iteration: int) -> dict[str, Any]:
    """Capture only state that affects the next completed training iteration."""

    state: dict[str, Any] = {
        "completed_iteration": completed_iteration,
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "python_random_state": random.getstate(),
        "numpy_random_state": _serialize_numpy_random_state(np.random.get_state()),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "environment_states": [_capture_environment_state(env) for env in _trainer_environments(trainer)],
    }
    scenario_rng = getattr(trainer, "_scenario_rng", None)
    if scenario_rng is not None:
        state["scenario_rng_state"] = copy.deepcopy(scenario_rng.bit_generator.state)
    current_scenario = getattr(trainer, "current_scenario", None)
    if current_scenario is not None:
        state["current_scenario_id"] = int(current_scenario.scenario_id)
    return state


def restore_training_state(trainer: Any, state: dict[str, Any]) -> int:
    """Restore a checkpoint immediately before the next rollout is collected."""

    completed_iteration = int(state["completed_iteration"])
    trainer.optimizer.load_state_dict(state["optimizer_state_dict"])
    _move_optimizer_state_to_device(trainer.optimizer, trainer.device)
    random.setstate(state["python_random_state"])
    numpy_state = state["numpy_random_state"]
    if isinstance(numpy_state, dict):
        numpy_state = (
            numpy_state["algorithm"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    np.random.set_state(numpy_state)
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_states") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_states"])
    scenario_rng = getattr(trainer, "_scenario_rng", None)
    if scenario_rng is not None and state.get("scenario_rng_state") is not None:
        scenario_rng.bit_generator.state = state["scenario_rng_state"]
    for env, environment_state in zip(
        _trainer_environments(trainer), state.get("environment_states", []), strict=False
    ):
        _restore_environment_state(env, environment_state)
    if state.get("current_scenario_id") is not None and hasattr(trainer, "_activate_scenario"):
        scenario_id = int(state["current_scenario_id"])
        for scenario in trainer.scenarios:
            if scenario.scenario_id == scenario_id:
                trainer._activate_scenario(scenario)
                break
    return completed_iteration


def save_resumable_checkpoint(
    checkpoint_path: Path,
    trainer: Any,
    *,
    method: str,
    metadata: dict[str, Any],
    completed_iteration: int,
) -> None:
    """Atomically save and then read back a resumable checkpoint."""

    temporary_path = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    save_model_checkpoint(
        temporary_path,
        trainer.model,
        method=method,
        metadata=metadata,
        training_state=capture_training_state(trainer, completed_iteration),
    )
    _load_checkpoint_payload(temporary_path, trainer.device)
    temporary_path.replace(checkpoint_path)
    _load_checkpoint_payload(checkpoint_path, trainer.device)


def save_periodic_checkpoint(
    checkpoint_path: Path,
    trainer: Any,
    *,
    method: str,
    metadata: dict[str, Any],
    completed_iteration: int,
    log: Callable[[str], None],
) -> Path:
    """Save a new periodic checkpoint before removing older periodic files."""

    periodic_path = periodic_checkpoint_path(checkpoint_path, completed_iteration)
    previous_paths = periodic_checkpoints(checkpoint_path)
    save_resumable_checkpoint(
        periodic_path,
        trainer,
        method=method,
        metadata=metadata,
        completed_iteration=completed_iteration,
    )
    log(f"Periodic checkpoint saved at iteration {completed_iteration}: {periodic_path}")
    for previous_path in previous_paths:
        if previous_path != periodic_path and previous_path.exists():
            previous_path.unlink()
            log(f"Removed previous periodic checkpoint: {previous_path}")
    return periodic_path


def remove_periodic_checkpoints(checkpoint_path: Path) -> None:
    for periodic_path in periodic_checkpoints(checkpoint_path):
        periodic_path.unlink()


def prepare_resume(
    output_path: Path,
    checkpoint_path: Path,
    fieldnames: list[str],
    trainer: Any,
    *,
    trust_checkpoint: bool = False,
) -> ResumeProgress:
    """Select a checkpoint, validate its CSV, and restore trainer state."""

    candidates = periodic_checkpoints(checkpoint_path)
    selected_path = candidates[-1] if candidates else checkpoint_path
    if not selected_path.is_file():
        raise FileNotFoundError(
            "--resume was requested but no numbered periodic checkpoint or final checkpoint was found: "
            f"{checkpoint_path}"
        )
    payload = _load_checkpoint_payload(
        selected_path, trainer.device, trust_checkpoint=trust_checkpoint
    )
    training_state = payload.get("training_state")
    if not isinstance(training_state, dict) or "completed_iteration" not in training_state:
        raise ValueError(
            f"resume checkpoint is not resumable because training_state.completed_iteration is missing: {selected_path}"
        )
    if not output_path.is_file():
        raise FileNotFoundError(
            f"--resume requires the existing metrics CSV for the selected checkpoint: {output_path}"
        )
    rows = _load_rows_through_checkpoint(output_path, fieldnames, int(training_state["completed_iteration"]))
    trainer.model.load_state_dict(payload["model_state_dict"])
    completed_iteration = restore_training_state(trainer, training_state)
    if len(rows) != completed_iteration:
        raise ValueError(
            "metrics CSV does not contain one row for every completed iteration through "
            f"checkpoint iteration {completed_iteration}: {output_path}"
        )
    write_metrics_csv(output_path, fieldnames, rows)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"resume checkpoint metadata must be a dictionary: {selected_path}")
    return ResumeProgress(selected_path, completed_iteration, rows, metadata)


def prepare_finetune(
    output_path: Path,
    input_checkpoint_path: Path,
    fieldnames: list[str],
    trainer: Any,
    *,
    trust_checkpoint: bool = False,
) -> ResumeProgress:
    """Load an explicit fine-tune checkpoint while preserving the CSV bytes.

    Resumable repository checkpoints restore optimizer, RNG, scenario, and
    environment state. Older weight-only checkpoints remain supported by using
    the contiguous existing CSV length as the completed iteration count.
    """

    if not input_checkpoint_path.is_file():
        raise FileNotFoundError(f"--input-weight does not exist: {input_checkpoint_path}")
    if not output_path.is_file():
        raise FileNotFoundError(
            f"--input-weight requires the existing metrics CSV: {output_path}"
        )

    payload = _load_checkpoint_payload(
        input_checkpoint_path, trainer.device, trust_checkpoint=trust_checkpoint
    )
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"fine-tune checkpoint metadata must be a dictionary: {input_checkpoint_path}")

    training_state = payload.get("training_state")
    if isinstance(training_state, dict) and "completed_iteration" in training_state:
        completed_iteration = int(training_state["completed_iteration"])
        rows = _load_all_contiguous_rows(output_path, fieldnames)
        if len(rows) != completed_iteration:
            raise ValueError(
                "fine-tune metrics CSV does not end at the input checkpoint iteration: "
                f"csv_rows={len(rows)}, checkpoint_iteration={completed_iteration}, path={output_path}"
            )
        trainer.model.load_state_dict(payload["model_state_dict"])
        restored_iteration = restore_training_state(trainer, training_state)
        if restored_iteration != completed_iteration:
            raise ValueError("fine-tune checkpoint completed_iteration restore mismatch")
        continuation_mode = "exact_resume"
    else:
        rows = _load_all_contiguous_rows(output_path, fieldnames)
        completed_iteration = len(rows)
        trainer.model.load_state_dict(payload["model_state_dict"])
        continuation_mode = "weight_initialized_finetune"

    return ResumeProgress(
        input_checkpoint_path,
        completed_iteration,
        rows,
        metadata,
        continuation_mode=continuation_mode,
        checkpoint_method=payload.get("method"),
    )


def validate_finetune_metadata(
    checkpoint_metadata: dict[str, Any],
    expected_metadata: dict[str, Any],
    *,
    expected_method: str,
    checkpoint_method: str | None = None,
    ignore_fingerprint: bool = False,
) -> None:
    """Validate available training-contract metadata before a fine-tune begins."""

    checkpoint_method = checkpoint_method or checkpoint_metadata.get("method")
    if checkpoint_method is not None and checkpoint_method != expected_method:
        raise ValueError(
            "fine-tune checkpoint method mismatch: "
            f"checkpoint={checkpoint_method!r}, requested={expected_method!r}"
        )
    saved = checkpoint_metadata.get("metadata", checkpoint_metadata)
    if not isinstance(saved, dict):
        raise ValueError("fine-tune checkpoint nested metadata must be a dictionary")

    keys = (
        "ablation_cell",
        "architecture_mode",
        "architecture_version",
        "model_name",
        "model_type",
        "feature_mode",
        "reward_mode",
        "observation_feature_mode",
        "observation_schema_version",
        "resource_observation_schema_version",
        "rh_feature_dim",
        "num_rh",
        "num_es",
        "num_rc",
        "num_splits",
        "split_names",
        "episode_length",
        "num_episodes",
        "hidden_dim",
        "num_gru_layers",
        "temporal_fusion_mode",
        "hidden_reset_mode",
        "ppo_replay_mode",
        "distribution_implementation",
        "constraint_formulation_version",
        "config_fingerprint",
    )
    if not ignore_fingerprint:
        keys = (*keys, "dataset_fingerprint")

    # Early recurrent v1 checkpoints recorded the legacy mobility schema in
    # observation_schema_version while correctly recording the v1 resource
    # schema in resource_observation_schema_version.  Their model width and
    # resource feature contract are still v1, so accept only that precise,
    # checkpoint-visible alias.  Do not generalize this exception to GPPO,
    # legacy/v2 observations, or a missing resource schema.
    historical_recurrent_v1_schema_alias = (
        saved.get("architecture_mode") == "recurrent_stgnn"
        and saved.get("observation_feature_mode") == CAUSAL_RESOURCE_OBSERVATION_FEATURE_MODE
        and expected_metadata.get("observation_feature_mode") == CAUSAL_RESOURCE_OBSERVATION_FEATURE_MODE
        and saved.get("observation_schema_version") == LEGACY_OBSERVATION_SCHEMA
        and expected_metadata.get("observation_schema_version") == CAUSAL_RESOURCE_OBSERVATION_SCHEMA
        and saved.get("resource_observation_schema_version") == CAUSAL_RESOURCE_OBSERVATION_SCHEMA
        and expected_metadata.get("resource_observation_schema_version") == CAUSAL_RESOURCE_OBSERVATION_SCHEMA
    )
    mismatches: list[str] = []
    for key in keys:
        expected = expected_metadata.get(key)
        actual = saved.get(key)
        if key == "observation_schema_version" and historical_recurrent_v1_schema_alias:
            continue
        if expected is not None and actual is not None and actual != expected:
            mismatches.append(f"{key}: checkpoint={actual!r}, requested={expected!r}")
    if mismatches:
        raise ValueError("fine-tune checkpoint metadata mismatch: " + "; ".join(mismatches))


def _load_rows_through_checkpoint(
    output_path: Path,
    fieldnames: list[str],
    completed_iteration: int,
) -> list[dict[str, str]]:
    with output_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fieldnames:
            raise ValueError(f"metrics CSV schema mismatch during resume: {output_path}")
        all_rows = list(reader)
    retained = [row for row in all_rows if int(row["iteration"]) < completed_iteration]
    retained_iterations = [int(row["iteration"]) for row in retained]
    if retained_iterations != list(range(completed_iteration)):
        raise ValueError(
            "metrics CSV is missing, duplicated, or unordered before checkpoint iteration "
            f"{completed_iteration}: {output_path}"
        )
    return retained


def _load_all_contiguous_rows(
    output_path: Path,
    fieldnames: list[str],
) -> list[dict[str, str]]:
    with output_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fieldnames:
            raise ValueError(f"metrics CSV schema mismatch during fine-tune: {output_path}")
        rows = list(reader)
    iterations = [int(row["iteration"]) for row in rows]
    if iterations != list(range(len(rows))):
        raise ValueError(
            "metrics CSV is missing, duplicated, or unordered for weight-initialized fine-tune: "
            f"{output_path}"
        )
    return rows


def _load_checkpoint_payload(
    path: Path,
    device: torch.device,
    *,
    trust_checkpoint: bool = False,
) -> dict[str, Any]:
    payload = load_checkpoint_payload(
        path, map_location=device, trust_checkpoint=trust_checkpoint
    )
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"invalid resumable checkpoint payload: {path}")
    return payload


def _serialize_numpy_random_state(state: tuple[Any, ...]) -> dict[str, Any]:
    algorithm, keys, position, has_gauss, cached_gaussian = state
    return {
        "algorithm": str(algorithm),
        "keys": np.asarray(keys, dtype=np.uint32).tolist(),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _trainer_environments(trainer: Any) -> list[Any]:
    scenarios = getattr(trainer, "scenarios", None)
    if scenarios is not None:
        return [scenario.env for scenario in scenarios]
    return [trainer.env]


def _capture_environment_state(env: Any) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for attribute in (
        "time_index",
        "current_episode_id",
        "current_local_t",
        "_next_episode_id",
        "_episode_order",
    ):
        if hasattr(env, attribute):
            state[attribute] = copy.deepcopy(getattr(env, attribute))
    episode_order_rng = getattr(env, "_episode_order_rng", None)
    if episode_order_rng is not None:
        state["episode_order_rng_state"] = copy.deepcopy(episode_order_rng.bit_generator.state)
    return state


def _restore_environment_state(env: Any, state: dict[str, Any]) -> None:
    for attribute, value in state.items():
        if attribute != "episode_order_rng_state":
            setattr(env, attribute, value)
    episode_order_rng = getattr(env, "_episode_order_rng", None)
    if episode_order_rng is not None and state.get("episode_order_rng_state") is not None:
        episode_order_rng.bit_generator.state = state["episode_order_rng_state"]


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)

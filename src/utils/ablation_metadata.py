"""Reproducibility metadata shared by four-way ablation runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from src.utils.config import load_simple_yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ablation_fingerprints(dataset_dir: str | Path, config_path: str | Path) -> dict[str, Any]:
    dataset = Path(dataset_dir)
    files = {
        str(path.relative_to(dataset)): _sha256(path)
        for path in sorted(dataset.rglob("*"))
        if path.is_file()
    }
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"], capture_output=True, text=True, check=False
    )
    fingerprint_input = "".join(f"{name}:{digest}\n" for name, digest in files.items())
    return {
        "dataset_fingerprint": hashlib.sha256(fingerprint_input.encode()).hexdigest(),
        "dataset_file_hashes": files,
        "config_fingerprint": _sha256(Path(config_path)),
        "git_dirty": bool(status.stdout.strip()),
    }


def mobility_checkpoint_metadata(dataset_dir: str | Path, *, episode_order_mode: str) -> dict[str, Any]:
    params = load_simple_yaml(Path(dataset_dir) / "generation_params.yaml")
    observation = dict(params.get("mobility_observation", {}))
    return {
        "mobility_observation_schema_version": params.get(
            "mobility_observation_schema_version", "legacy_same_step_v1"
        ),
        "lag_steps": observation.get("lag_steps"),
        "rho_mode": observation.get("rho_mode"),
        "rho_ewma_alpha": observation.get("rho_ewma_alpha"),
        "warmup_steps": observation.get("warmup_steps"),
        "regime_randomization": bool(params.get("regime_randomization", False)),
        "phase_randomization": bool(params.get("phase_randomization", False)),
        "episode_order_mode": episode_order_mode,
    }


def dataset_identity_metadata(dataset_dir: str | Path) -> dict[str, str]:
    """Return stable dataset hashes recorded by the daily dataset exporter."""

    dataset = Path(dataset_dir)
    metadata_path = dataset / "dataset_metadata.json"
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {
            "dataset_dynamic_hash": str(payload.get("dynamic_data_hash", "unknown")),
            "dataset_topology_hash": str(payload.get("topology_hash", "unknown")),
            "dataset_config_hash": str(payload.get("config_hash", "unknown")),
            "topology_profile_set_hash": str(payload.get("topology_profile_set_hash", "static")),
        }

    # Legacy datasets predate dataset_metadata.json. Preserve compatibility
    # while still recording an immutable identity for every required field.
    fingerprint = ablation_fingerprints(dataset, dataset / "generation_params.yaml")[
        "dataset_fingerprint"
    ]
    return {
        "dataset_dynamic_hash": fingerprint,
        "dataset_topology_hash": fingerprint,
        "dataset_config_hash": _sha256(dataset / "generation_params.yaml"),
        "topology_profile_set_hash": "static",
    }

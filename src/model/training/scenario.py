"""Fixed-size training scenario contexts for PPO variants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.env.oran_ho_env import OranHandoverEnv
from src.model.inference.post_process import (
    action_mask_kwargs_from_episode,
    build_link_feasibility_action_mask,
    build_topology_profile_action_masks,
)
from src.pipeline.data_pipeline import HandoverGraphPipeline
from src.pipeline.schema import TopologyConfig


@dataclass
class TrainingScenario:
    """One fixed-size training scenario with its own episode, mask, and graph pipeline."""

    scenario_id: int
    scenario_name: str
    params_path: str
    env: OranHandoverEnv
    graph_pipeline: HandoverGraphPipeline
    link_action_mask: np.ndarray
    valid_action_count_per_rh: np.ndarray
    topology_profile_action_masks: dict[str, np.ndarray]

    @classmethod
    def from_env(
        cls,
        *,
        scenario_id: int,
        scenario_name: str,
        params_path: str,
        env: OranHandoverEnv,
    ) -> "TrainingScenario":
        graph_pipeline = HandoverGraphPipeline(env.episode)

        def build_mask(fronthaul_mask: np.ndarray, midhaul_mask: np.ndarray) -> np.ndarray:
            return build_link_feasibility_action_mask(
                env.episode.config,
                fronthaul_mask,
                midhaul_mask,
                env.episode.direct_rh_rc_mask,
                env.episode.split_demand.split_names,
                **action_mask_kwargs_from_episode(env.episode),
            )

        link_action_mask = build_mask(
            env.episode.fronthaul_mask,
            env.episode.midhaul_mask,
        )
        profile_masks = build_topology_profile_action_masks(env.episode)
        return cls(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            params_path=params_path,
            env=env,
            graph_pipeline=graph_pipeline,
            link_action_mask=link_action_mask,
            valid_action_count_per_rh=link_action_mask.sum(axis=1).astype(np.int64),
            topology_profile_action_masks=profile_masks,
        )

    def action_mask_for_time_index(self, time_index: int) -> np.ndarray:
        """Return the cached mask for the topology active at an absolute timestep."""

        if not self.topology_profile_action_masks:
            return self.link_action_mask
        episode_id_for_time = getattr(self.env, "episode_id_for_time_index", None)
        if episode_id_for_time is None:
            profile_id = self.env.episode.metadata.get("active_topology_profile_id")
        else:
            episode_id = int(episode_id_for_time(time_index))
            profile_id = self.env.episode.topology_profile_id_for_episode(episode_id)
        if profile_id is None:
            return self.link_action_mask
        return self.topology_profile_action_masks[str(profile_id)]


def scenario_name_from_params(params_path: str | Path) -> str:
    """Return a stable scenario name from a YAML path."""

    return Path(params_path).stem


def validate_same_topology(scenarios: list[TrainingScenario]) -> TopologyConfig:
    """Require all scenarios to share model/action dimensions."""

    if not scenarios:
        raise ValueError("at least one training scenario is required")
    reference = scenarios[0].env.episode.config
    for scenario in scenarios[1:]:
        config = scenario.env.episode.config
        mismatches = []
        if config.num_rhs != reference.num_rhs:
            mismatches.append(f"num_rhs {config.num_rhs} != {reference.num_rhs}")
        if config.num_es != reference.num_es:
            mismatches.append(f"num_es {config.num_es} != {reference.num_es}")
        if config.num_rc != reference.num_rc:
            mismatches.append(f"num_rc {config.num_rc} != {reference.num_rc}")
        if config.num_splits != reference.num_splits:
            mismatches.append(f"num_splits {config.num_splits} != {reference.num_splits}")
        if config.num_splits == reference.num_splits and (
            scenario.env.episode.split_demand.split_names != scenarios[0].env.episode.split_demand.split_names
        ):
            mismatches.append(
                f"split_names {scenario.env.episode.split_demand.split_names} "
                f"!= {scenarios[0].env.episode.split_demand.split_names}"
            )
        if mismatches:
            raise ValueError(
                f"scenario '{scenario.scenario_name}' has incompatible topology: "
                + ", ".join(mismatches)
            )
    return reference

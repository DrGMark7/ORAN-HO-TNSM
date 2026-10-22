"""CSV-to-graph pipeline for time-domain handover-aware mock data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch_geometric.data import HeteroData

from src.pipeline.schema import (
    DEFAULT_GRAPH_FEATURE_SCHEMA,
    MockDatasetPaths,
    MockEpisode,
    TopologyConfig,
    TopologyProfile,
)
from src.model.split_semantics import contains_direct_rc_split
from src.utils.config import (
    DEFAULT_DATA_PARAMS_PATH,
    cost_weights_from_config,
    load_data_generation_config,
    load_project_config,
    split_demand_from_config,
)
from src.utils.config_contract import (
    ConfigurationContractError,
    resolve_run_config,
    runtime_project_config_from_resolved_problem,
)


SUPPORTED_V3_RL_FORMULATION = "split_transport_utilization_latency_v1"


class MockCsvDatasetLoader:
    """Read exported mock CSV files and reconstruct a typed episode."""

    def __init__(
        self,
        dataset_dir: str | Path = Path("mock_data") / DEFAULT_DATA_PARAMS_PATH.stem,
        config_path: str | Path | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        generation_params_path = self.dataset_dir / "generation_params.yaml"
        resolved = resolve_run_config(
            generation_params_path if generation_params_path.exists() else DEFAULT_DATA_PARAMS_PATH,
            config_path,
            dataset_dir=self.dataset_dir,
        )
        self.resolved_run_config = resolved
        self.is_v3 = not resolved.problem.is_legacy
        if not resolved.problem.is_legacy:
            if resolved.problem.formulation_status == "draft":
                raise ConfigurationContractError(
                    "draft problem formulation cannot start training or evaluation"
                )
            formulation_version = str(resolved.problem.data["formulation"]["version"])
            if formulation_version != SUPPORTED_V3_RL_FORMULATION:
                raise ConfigurationContractError(
                    f"active v3 formulation is not implemented for RL loading: {formulation_version}"
                )
            self.project_config = runtime_project_config_from_resolved_problem(resolved)
            self.generation_config = {
                "topology": self.project_config["topology"],
                "episode_length": int(resolved.problem.data["dataset"]["episode_length"]),
                "topology_episode_profiles": self.project_config["topology_episode_profiles"],
            }
        else:
            # Legacy datasets retain their exact pre-Patch-0 equations and the
            # familiar merged mapping.  The mapping originates from the explicit
            # compatibility adapter, not from the v2 RL JSON.
            self.project_config = dict(resolved.runtime_config or load_project_config(config_path))
            generation_config = load_data_generation_config(
                generation_params_path if generation_params_path.exists() else DEFAULT_DATA_PARAMS_PATH
            )
            self.generation_config = generation_config
            self.project_config["topology"] = generation_config["topology"]
            self.project_config["mock_data"] = generation_config["mock_data"]
            self.project_config["split_demand"] = generation_config["split_demand"]
            self.project_config["topology_profile"] = generation_config.get("topology_profile", "legacy")
            self.project_config["config_compatibility"] = generation_config.get("config_compatibility", {})
            self.project_config["effective_topology"] = generation_config.get("effective_topology", {})
            self.project_config["split_policy"] = generation_config.get("split_policy", {})
            self.project_config["validity"] = generation_config.get("validity", {})
            self.project_config["mobility_observation_schema_version"] = generation_config.get(
                "mobility_observation_schema_version", "legacy_same_step_v1"
            )
            self.project_config["mobility_observation"] = generation_config.get("mobility_observation", {})
        self.paths = MockDatasetPaths(
            root_dir=str(self.dataset_dir),
            rh_timeseries=str(self.dataset_dir / "rh_timeseries.csv"),
            handover_edges=str(self.dataset_dir / "handover_edges.csv"),
            rh_nodes=str(self.dataset_dir / "rh_nodes.csv"),
            es_nodes=str(self.dataset_dir / "es_nodes.csv"),
            rc_nodes=str(self.dataset_dir / "rc_nodes.csv"),
            fronthaul_edges=str(self.dataset_dir / "fronthaul_edges.csv"),
            midhaul_edges=str(self.dataset_dir / "midhaul_edges.csv"),
            direct_rh_rc_edges=str(self.dataset_dir / "direct_rh_rc_edges.csv"),
            effective_generation_config=str(self.dataset_dir / "effective_generation_config.yaml"),
            episode_topology_profiles=str(self.dataset_dir / "episode_topology_profiles.csv"),
        )

    def load_episode(self, *, allow_large_transport_degree: bool = False) -> MockEpisode:
        sanity_check_csv_dataset(self.paths)
        rh_ts = pl.read_csv(self.paths.rh_timeseries)
        handover = pl.read_csv(self.paths.handover_edges)
        rh_nodes = pl.read_csv(self.paths.rh_nodes).sort("rh_id")
        es_nodes = pl.read_csv(self.paths.es_nodes).sort("es_id")
        rc_nodes = pl.read_csv(self.paths.rc_nodes).sort("rc_id")
        fronthaul = pl.read_csv(self.paths.fronthaul_edges)
        midhaul = pl.read_csv(self.paths.midhaul_edges)
        direct_path = Path(self.paths.direct_rh_rc_edges) if self.paths.direct_rh_rc_edges is not None else None
        direct = pl.read_csv(direct_path) if direct_path is not None and direct_path.exists() else None

        horizon = int(rh_ts["t"].max()) + 1
        num_rh = rh_nodes.height
        num_es = es_nodes.height
        num_rc = rc_nodes.height
        num_splits = int(self.project_config["topology"]["num_splits"])
        config = TopologyConfig(
            num_rhs=num_rh,
            num_es=num_es,
            num_rc=num_rc,
            horizon=horizon,
            num_splits=num_splits,
        )

        lambda_rt = np.zeros((horizon, num_rh), dtype=np.float64)
        num_ues_rt = np.zeros((horizon, num_rh), dtype=np.float64)
        s_pdcp_rt = np.zeros((horizon, num_rh), dtype=np.float64)
        s_rlc_rt = np.zeros((horizon, num_rh), dtype=np.float64)
        rho_rt = np.zeros((horizon, num_rh), dtype=np.float64)
        for row in rh_ts.iter_rows(named=True):
            t = int(row["t"])
            rh_id = int(row["rh_id"])
            num_ues_rt[t, rh_id] = float(row["n_ue"])
            lambda_rt[t, rh_id] = float(row["traffic_load_mbps"])
            s_pdcp_rt[t, rh_id] = float(row["s_pdcp_mb"])
            s_rlc_rt[t, rh_id] = float(row["s_rlc_mb"])
            rho_rt[t, rh_id] = float(row["rho"])

        handover_count_t = np.zeros((horizon, num_rh, num_rh), dtype=np.float64)
        handover_matrix_t = np.zeros((horizon, num_rh, num_rh), dtype=np.float64)
        observed_handover_count_t = np.zeros_like(handover_count_t)
        observed_handover_matrix_t = np.zeros_like(handover_matrix_t)
        has_observed = {"observed_handover_count", "observed_handover_prob"}.issubset(handover.columns)
        for row in handover.iter_rows(named=True):
            t = int(row["t"])
            src = int(row["src_rh_id"])
            dst = int(row["dst_rh_id"])
            handover_count_t[t, src, dst] = float(row["handover_count"])
            handover_matrix_t[t, src, dst] = float(row["handover_prob"])
            if has_observed:
                observed_handover_count_t[t, src, dst] = float(row["observed_handover_count"])
                observed_handover_matrix_t[t, src, dst] = float(row["observed_handover_prob"])

        def link_arrays(
            selected_fronthaul: pl.DataFrame,
            selected_midhaul: pl.DataFrame,
        ) -> tuple[np.ndarray, ...]:
            feasible_column = "is_feasible" if self.is_v3 else "feasible"
            fh_mask = np.zeros((num_rh, num_es), dtype=np.bool_)
            fh_latency = np.zeros((num_rh, num_es), dtype=np.float64)
            fh_bandwidth = np.zeros((num_rh, num_es), dtype=np.float64)
            fh_cost = np.zeros((num_rh, num_es), dtype=np.float64)
            for row in selected_fronthaul.iter_rows(named=True):
                rh_id, es_id = int(row["rh_id"]), int(row["es_id"])
                fh_mask[rh_id, es_id] = bool(row[feasible_column])
                fh_latency[rh_id, es_id] = float(row["latency_ms"])
                fh_bandwidth[rh_id, es_id] = float(row["bandwidth_mbps"])
                fh_cost[rh_id, es_id] = float(row["cost"])
            mh_mask = np.zeros((num_es, num_rc), dtype=np.bool_)
            mh_latency = np.zeros((num_es, num_rc), dtype=np.float64)
            mh_bandwidth = np.zeros((num_es, num_rc), dtype=np.float64)
            mh_cost = np.zeros((num_es, num_rc), dtype=np.float64)
            for row in selected_midhaul.iter_rows(named=True):
                es_id, rc_id = int(row["es_id"]), int(row["rc_id"])
                mh_mask[es_id, rc_id] = bool(row[feasible_column])
                mh_latency[es_id, rc_id] = float(row["latency_ms"])
                mh_bandwidth[es_id, rc_id] = float(row["bandwidth_mbps"])
                mh_cost[es_id, rc_id] = float(row["cost"])
            return fh_mask, mh_mask, fh_latency, mh_latency, fh_bandwidth, mh_bandwidth, fh_cost, mh_cost

        def direct_arrays(selected_direct: pl.DataFrame | None) -> tuple[np.ndarray, ...]:
            mask = np.zeros((num_rh, num_rc), dtype=np.bool_)
            latency = np.zeros((num_rh, num_rc), dtype=np.float64)
            bandwidth = np.zeros((num_rh, num_rc), dtype=np.float64)
            cost = np.zeros((num_rh, num_rc), dtype=np.float64)
            if selected_direct is not None:
                feasible_column = "is_feasible" if self.is_v3 else "connected"
                for row in selected_direct.iter_rows(named=True):
                    rh_id, rc_id = int(row["rh_id"]), int(row["rc_id"])
                    mask[rh_id, rc_id] = bool(row[feasible_column])
                    latency[rh_id, rc_id] = float(row["latency_ms"])
                    bandwidth[rh_id, rc_id] = float(row["bandwidth_mbps"])
                    cost[rh_id, rc_id] = float(row["cost"])
            return mask, latency, bandwidth, cost

        topology_profiles: dict[str, TopologyProfile] = {}
        episode_topology_profile_ids: tuple[str, ...] = ()
        profile_column = "profile_id" if self.is_v3 else "topology_profile_id"
        has_profiles = profile_column in fronthaul.columns or profile_column in midhaul.columns
        if has_profiles:
            if profile_column not in fronthaul.columns or profile_column not in midhaul.columns:
                raise ValueError(f"both link CSVs must include {profile_column}")
            profile_ids = sorted(set(fronthaul[profile_column].cast(pl.String).to_list()))
            if profile_ids != sorted(set(midhaul[profile_column].cast(pl.String).to_list())):
                raise ValueError("fronthaul and midhaul topology profile IDs must match")
            direct_profile_ids: list[str] | None = None
            if direct is not None and profile_column in direct.columns:
                direct_profile_ids = sorted(set(direct[profile_column].cast(pl.String).to_list()))
                if direct_profile_ids != ["static"] and profile_ids != direct_profile_ids:
                    raise ValueError("direct and gateway topology profile IDs must match")
            expected_profile_count = int(
                self.generation_config.get("topology_episode_profiles", {}).get(
                    "profile_count", len(profile_ids)
                )
            )
            if len(profile_ids) != expected_profile_count:
                raise ValueError(
                    f"expected {expected_profile_count} topology profiles, found {len(profile_ids)}"
                )
            for profile_id in profile_ids:
                arrays = link_arrays(
                    fronthaul.filter(pl.col(profile_column).cast(pl.String) == profile_id),
                    midhaul.filter(pl.col(profile_column).cast(pl.String) == profile_id),
                )
                selected_direct = direct
                if direct is not None and direct_profile_ids not in (None, ["static"]):
                    selected_direct = direct.filter(
                        pl.col(profile_column).cast(pl.String) == profile_id
                    )
                arrays = (*arrays, *direct_arrays(selected_direct))
                topology_profiles[profile_id] = TopologyProfile(profile_id, *arrays)
            assignment_path = Path(self.paths.episode_topology_profiles or "")
            if not assignment_path.exists():
                raise ValueError("profile-aware link CSVs require episode_topology_profiles.csv")
            assignment = pl.read_csv(assignment_path).sort("episode_id")
            expected_episode_ids = list(range(assignment.height))
            if assignment["episode_id"].cast(pl.Int64).to_list() != expected_episode_ids:
                raise ValueError("episode topology assignments must use contiguous zero-based episode IDs")
            episode_topology_profile_ids = tuple(
                assignment["topology_profile_id"].cast(pl.String).to_list()
            )
            episode_length = int(self.generation_config["episode_length"])
            expected_episode_count, remainder = divmod(horizon, episode_length)
            if remainder or len(episode_topology_profile_ids) != expected_episode_count:
                raise ValueError(
                    "episode topology assignment count must match horizon / episode_length"
                )
            active_profile_id = episode_topology_profile_ids[0]
            active = topology_profiles[active_profile_id]
            arrays = tuple(
                getattr(active, name).copy()
                for name in (
                    "fronthaul_mask", "midhaul_mask", "fronthaul_latency_ms", "midhaul_latency_ms",
                    "fronthaul_bandwidth_mbps", "midhaul_bandwidth_mbps", "fronthaul_cost", "midhaul_cost",
                    "direct_rh_rc_mask", "direct_rh_rc_latency_ms",
                    "direct_rh_rc_bandwidth_mbps", "direct_rh_rc_cost",
                )
            )
        else:
            arrays = (*link_arrays(fronthaul, midhaul), *direct_arrays(direct))
        (
            fronthaul_mask, midhaul_mask, fronthaul_latency_ms, midhaul_latency_ms,
            fronthaul_bandwidth_mbps, midhaul_bandwidth_mbps, fronthaul_cost, midhaul_cost,
            direct_rh_rc_mask, direct_rh_rc_latency_ms,
            direct_rh_rc_bandwidth_mbps, direct_rh_rc_cost,
        ) = arrays
        if direct is None and contains_direct_rc_split(
            tuple(self.project_config["split_demand"]["split_names"])
        ):
            raise ValueError("S4D split requires direct_rh_rc_edges.csv in the dataset")

        episode = MockEpisode(
            config=config,
            lambda_rt=lambda_rt,
            num_ues_rt=num_ues_rt,
            handover_count_t=handover_count_t,
            handover_matrix_t=handover_matrix_t,
            s_pdcp_rt=s_pdcp_rt,
            s_rlc_rt=s_rlc_rt,
            rho_rt=rho_rt,
            rh_x=rh_nodes["x"].to_numpy().astype(np.float64),
            rh_y=rh_nodes["y"].to_numpy().astype(np.float64),
            rh_region_id=rh_nodes["region_id"].to_numpy().astype(np.int64),
            fronthaul_mask=fronthaul_mask,
            midhaul_mask=midhaul_mask,
            fronthaul_latency_ms=fronthaul_latency_ms,
            midhaul_latency_ms=midhaul_latency_ms,
            fronthaul_bandwidth_mbps=fronthaul_bandwidth_mbps,
            midhaul_bandwidth_mbps=midhaul_bandwidth_mbps,
            fronthaul_cost=fronthaul_cost,
            midhaul_cost=midhaul_cost,
            direct_rh_rc_mask=direct_rh_rc_mask,
            direct_rh_rc_latency_ms=direct_rh_rc_latency_ms,
            direct_rh_rc_bandwidth_mbps=direct_rh_rc_bandwidth_mbps,
            direct_rh_rc_cost=direct_rh_rc_cost,
            es_compute_capacity=es_nodes["compute_capacity" if self.is_v3 else "cpu_capacity"].to_numpy().astype(np.float64),
            rc_compute_capacity=rc_nodes["compute_capacity" if self.is_v3 else "cpu_capacity"].to_numpy().astype(np.float64),
            es_placement_cost=es_nodes["base_cost"].to_numpy().astype(np.float64),
            rc_placement_cost=rc_nodes["base_cost"].to_numpy().astype(np.float64),
            split_demand=split_demand_from_config(self.project_config),
            cost_weights=cost_weights_from_config(self.project_config),
            metadata={
                "source": "csv",
                "dataset_dir": str(self.dataset_dir),
                "topology_profile": self.project_config.get("topology_profile", "legacy"),
                "config_compatibility": dict(self.project_config.get("config_compatibility", {})),
                "effective_topology": dict(self.project_config.get("effective_topology", {})),
                "latency_config_source": self.project_config["split_demand"].get(
                    "latency_config_source", "explicit"
                ),
                "allowed_split_names": tuple(
                    self.project_config.get("split_policy", {}).get(
                        "allowed_split_names",
                        self.project_config["split_demand"]["split_names"],
                    )
                ),
                "ensure_each_rh_has_valid_action": bool(
                    self.project_config.get("validity", {}).get("ensure_each_rh_has_valid_action", True)
                ),
                "fronthaul_enabled": bool(self.project_config["mock_data"].get("fronthaul_enabled", True)),
                "direct_rh_rc_enabled": bool(self.project_config["mock_data"].get("direct_rh_rc_enabled", False)),
                "rh_transport_budget_enabled": bool(
                    self.project_config["mock_data"].get("rh_transport_budget_enabled", False)
                ),
                "rh_transport_budget_max_total_degree_per_rh": int(
                    self.project_config["mock_data"].get("rh_transport_budget_max_total_degree_per_rh", 2)
                ),
                "rh_transport_budget_min_total_degree_per_rh": int(
                    self.project_config["mock_data"].get("rh_transport_budget_min_total_degree_per_rh", 1)
                ),
                "rh_transport_budget_allocation_mode": str(
                    self.project_config["mock_data"].get("rh_transport_budget_allocation_mode", "legacy_independent")
                ),
                "mobility_observation_schema_version": self.project_config.get(
                    "mobility_observation_schema_version", "legacy_same_step_v1"
                ),
                "mobility_observation_source": "dataset_columns" if has_observed else "unavailable_legacy",
                "topology_episode_profiles_enabled": has_profiles,
                "topology_profile_set_hash": self._topology_profile_set_hash(),
                "topology_episode_length": int(
                    self.generation_config.get(
                        "episode_length", self.generation_config["topology"]["horizon"]
                    )
                ),
                "allow_multiple_midhaul_routes_per_es": bool(
                    self.is_v3 or self._dataset_metadata().get("allow_multiple_midhaul_routes_per_es", False)
                ),
                "formulation_version": self.project_config["constraint_formulation_version"],
            },
            observed_handover_count_t=observed_handover_count_t if has_observed else None,
            observed_handover_matrix_t=observed_handover_matrix_t if has_observed else None,
            realized_outgoing_ratio_rt=np.divide(
                handover_count_t.sum(axis=2), np.maximum(num_ues_rt, 1.0)
            ),
            topology_profiles=topology_profiles,
            episode_topology_profile_ids=episode_topology_profile_ids,
            rh_service_deadline_ms=(
                rh_nodes["service_deadline_ms"].to_numpy().astype(np.float64)
                if self.is_v3 else None
            ),
        )
        episode.validate(allow_large_transport_degree=allow_large_transport_degree)
        return episode

    def _topology_profile_set_hash(self) -> str | None:
        value = self._dataset_metadata().get("topology_profile_set_hash")
        return None if value is None else str(value)

    def _dataset_metadata(self) -> dict[str, Any]:
        metadata_path = self.dataset_dir / "dataset_metadata.json"
        if not metadata_path.exists():
            return {}
        import json
        return dict(json.loads(metadata_path.read_text(encoding="utf-8")))


def sanity_check_csv_dataset(paths: MockDatasetPaths) -> None:
    """Validate the exported CSV schema before graph conversion."""

    rh_ts = pl.read_csv(paths.rh_timeseries)
    handover = pl.read_csv(paths.handover_edges)
    rh_nodes = pl.read_csv(paths.rh_nodes)
    es_nodes = pl.read_csv(paths.es_nodes)
    rc_nodes = pl.read_csv(paths.rc_nodes)
    fronthaul = pl.read_csv(paths.fronthaul_edges)
    midhaul = pl.read_csv(paths.midhaul_edges)
    direct_path = Path(paths.direct_rh_rc_edges) if paths.direct_rh_rc_edges is not None else None
    direct = pl.read_csv(direct_path) if direct_path is not None and direct_path.exists() else None

    horizon = int(rh_ts["t"].max()) + 1
    num_rh = rh_nodes.height
    if rh_ts.height != horizon * num_rh:
        raise ValueError("number of RH rows must equal T * num_rh")
    if handover.filter(pl.col("src_rh_id") == pl.col("dst_rh_id")).height > 0:
        raise ValueError("handover_edges.csv cannot contain src_rh_id == dst_rh_id")
    if handover.filter(pl.col("handover_count") < -1e-8).height > 0:
        raise ValueError("handover_count cannot contain negative values")
    if handover.filter((pl.col("handover_count") - pl.col("handover_count").round(0)).abs() > 1e-8).height > 0:
        raise ValueError("handover_count must be integer-valued")
    if handover.filter((pl.col("handover_prob") < -1e-8) | (pl.col("handover_prob") > 1.0 + 1e-8)).height > 0:
        raise ValueError("handover_prob must be in [0, 1]")

    outgoing = handover.group_by(["t", "src_rh_id"]).agg(pl.col("handover_prob").sum().alias("prob_sum"))
    if outgoing.filter(pl.col("prob_sum") > 1.0 + 1e-8).height > 0:
        raise ValueError("outgoing handover_prob sum must be <= 1 for each t, src_rh_id")
    outgoing_count = handover.group_by(["t", "src_rh_id"]).agg(pl.col("handover_count").sum().alias("count_sum"))
    joined_count = outgoing_count.join(
        rh_ts.select(["t", "rh_id", "n_ue"]),
        left_on=["t", "src_rh_id"],
        right_on=["t", "rh_id"],
    )
    if joined_count.filter(pl.col("count_sum") > pl.col("n_ue") + 1e-8).height > 0:
        raise ValueError("outgoing handover_count sum must be <= source n_ue")

    joined = handover.join(
        rh_ts.select(["t", "rh_id", "n_ue"]),
        left_on=["t", "src_rh_id"],
        right_on=["t", "rh_id"],
    ).with_columns(
        pl.when(pl.col("n_ue") > 0)
        .then((pl.col("handover_prob") - pl.col("handover_count") / pl.col("n_ue")).abs())
        .otherwise(pl.col("handover_prob").abs())
        .alias("prob_error")
    )
    if joined.filter(pl.col("prob_error") > 1e-8).height > 0:
        raise ValueError("handover_prob must equal handover_count / source n_ue within tolerance")

    valid_rh = set(rh_nodes["rh_id"].to_list())
    valid_es = set(es_nodes["es_id"].to_list())
    valid_rc = set(rc_nodes["rc_id"].to_list())
    if not set(fronthaul["rh_id"].to_list()).issubset(valid_rh):
        raise ValueError("fronthaul_edges.csv references invalid rh_id")
    if not set(fronthaul["es_id"].to_list()).issubset(valid_es):
        raise ValueError("fronthaul_edges.csv references invalid es_id")
    if not set(midhaul["es_id"].to_list()).issubset(valid_es):
        raise ValueError("midhaul_edges.csv references invalid es_id")
    if not set(midhaul["rc_id"].to_list()).issubset(valid_rc):
        raise ValueError("midhaul_edges.csv references invalid rc_id")
    fronthaul_keys = [
        *[column for column in ("topology_profile_id", "profile_id") if column in fronthaul.columns],
        "rh_id",
        "es_id",
    ]
    midhaul_keys = [
        *[column for column in ("topology_profile_id", "profile_id") if column in midhaul.columns],
        "es_id",
        "rc_id",
    ]
    if fronthaul.select(fronthaul_keys).is_duplicated().any():
        raise ValueError("fronthaul_edges.csv contains duplicate profile/RH/ES edges")
    if midhaul.select(midhaul_keys).is_duplicated().any():
        raise ValueError("midhaul_edges.csv contains duplicate profile/ES/RC edges")
    if direct is not None:
        if not set(direct["rh_id"].to_list()).issubset(valid_rh):
            raise ValueError("direct_rh_rc_edges.csv references invalid rh_id")
        if not set(direct["rc_id"].to_list()).issubset(valid_rc):
            raise ValueError("direct_rh_rc_edges.csv references invalid rc_id")
        direct_keys = [
            *[column for column in ("topology_profile_id", "profile_id") if column in direct.columns],
            "rh_id",
            "rc_id",
        ]
        if direct.select(direct_keys).is_duplicated().any():
            raise ValueError("direct_rh_rc_edges.csv contains duplicate profile/RH/RC edges")


class HandoverGraphPipeline:
    """Build per-time-step heterogeneous PyG graph snapshots."""

    def __init__(self, episode: MockEpisode) -> None:
        episode.validate()
        self.episode = episode

    @classmethod
    def from_csv(
        cls,
        dataset_dir: str | Path = Path("mock_data") / DEFAULT_DATA_PARAMS_PATH.stem,
        config_path: str | Path | None = None,
    ) -> "HandoverGraphPipeline":
        return cls(MockCsvDatasetLoader(dataset_dir, config_path).load_episode())

    def build_snapshot(self, time_index: int) -> HeteroData:
        """Return one PyG ``HeteroData`` snapshot."""

        if time_index < 0 or time_index >= self.episode.config.horizon:
            raise IndexError("time_index is outside the episode horizon")

        graph = HeteroData()
        for node_type, features in self._node_features(time_index).items():
            graph[node_type].x = torch.as_tensor(features, dtype=torch.float32)
        graph["rh"].feature_names = DEFAULT_GRAPH_FEATURE_SCHEMA.rh_node_features
        graph["es"].feature_names = DEFAULT_GRAPH_FEATURE_SCHEMA.es_node_features
        graph["rc"].feature_names = DEFAULT_GRAPH_FEATURE_SCHEMA.rc_node_features
        edge_index, edge_attr = self._edges(time_index)
        for edge_type, index in edge_index.items():
            graph[edge_type].edge_index = torch.as_tensor(index, dtype=torch.long)
            graph[edge_type].edge_attr = torch.as_tensor(edge_attr[edge_type], dtype=torch.float32)
        graph[("rh", "handover", "rh")].feature_names = DEFAULT_GRAPH_FEATURE_SCHEMA.handover_edge_features
        graph[("rh", "fronthaul", "es")].feature_names = DEFAULT_GRAPH_FEATURE_SCHEMA.fronthaul_edge_features
        graph[("es", "midhaul", "rc")].feature_names = DEFAULT_GRAPH_FEATURE_SCHEMA.midhaul_edge_features
        graph[("rh", "direct", "rc")].feature_names = DEFAULT_GRAPH_FEATURE_SCHEMA.direct_rh_rc_edge_features
        graph.time_index = time_index
        return graph

    def build_sequence(self) -> list[HeteroData]:
        return [self.build_snapshot(t) for t in range(self.episode.config.horizon)]

    def _node_features(self, time_index: int) -> dict[str, np.ndarray]:
        ep = self.episode
        rho = ep.rho_rt[time_index] if ep.rho_rt is not None else np.zeros(ep.config.num_rhs)
        rh_features = np.column_stack(
            [
                ep.num_ues_rt[time_index],
                ep.lambda_rt[time_index],
                ep.s_pdcp_rt[time_index],
                ep.s_rlc_rt[time_index],
                rho,
            ]
        ).astype(np.float32)
        es_features = np.column_stack([ep.es_compute_capacity]).astype(np.float32)
        rc_features = np.column_stack([ep.rc_compute_capacity]).astype(np.float32)
        return {"rh": rh_features, "es": es_features, "rc": rc_features}

    def _edges(
        self,
        time_index: int,
    ) -> tuple[dict[tuple[str, str, str], np.ndarray], dict[tuple[str, str, str], np.ndarray]]:
        ep = self.episode
        topology = None
        if ep.topology_profiles:
            episode_length = int(ep.metadata["topology_episode_length"])
            profile_id = ep.topology_profile_id_for_episode(time_index // episode_length)
            if profile_id is not None:
                topology = ep.topology_profiles[profile_id]
        fronthaul_mask = ep.fronthaul_mask if topology is None else topology.fronthaul_mask
        fronthaul_latency_ms = (
            ep.fronthaul_latency_ms if topology is None else topology.fronthaul_latency_ms
        )
        fronthaul_bandwidth_mbps = (
            ep.fronthaul_bandwidth_mbps if topology is None else topology.fronthaul_bandwidth_mbps
        )
        fronthaul_cost = ep.fronthaul_cost if topology is None else topology.fronthaul_cost
        midhaul_mask = ep.midhaul_mask if topology is None else topology.midhaul_mask
        midhaul_latency_ms = ep.midhaul_latency_ms if topology is None else topology.midhaul_latency_ms
        midhaul_bandwidth_mbps = (
            ep.midhaul_bandwidth_mbps if topology is None else topology.midhaul_bandwidth_mbps
        )
        midhaul_cost = ep.midhaul_cost if topology is None else topology.midhaul_cost
        direct_rh_rc_mask = ep.direct_rh_rc_mask if topology is None else topology.direct_rh_rc_mask
        direct_rh_rc_latency_ms = (
            ep.direct_rh_rc_latency_ms if topology is None else topology.direct_rh_rc_latency_ms
        )
        direct_rh_rc_bandwidth_mbps = (
            ep.direct_rh_rc_bandwidth_mbps
            if topology is None else topology.direct_rh_rc_bandwidth_mbps
        )
        direct_rh_rc_cost = ep.direct_rh_rc_cost if topology is None else topology.direct_rh_rc_cost
        edge_index: dict[tuple[str, str, str], np.ndarray] = {}
        edge_attr: dict[tuple[str, str, str], np.ndarray] = {}

        fh_src, fh_dst = np.indices((ep.config.num_rhs, ep.config.num_es)).reshape(2, -1)
        fh_type = ("rh", "fronthaul", "es")
        edge_index[fh_type] = np.vstack([fh_src, fh_dst]).astype(np.int64)
        edge_attr[fh_type] = np.column_stack(
            [
                fronthaul_mask[fh_src, fh_dst].astype(np.float64),
                fronthaul_latency_ms[fh_src, fh_dst],
                fronthaul_bandwidth_mbps[fh_src, fh_dst],
                fronthaul_cost[fh_src, fh_dst],
            ]
        ).astype(np.float32)

        mh_src, mh_dst = np.indices((ep.config.num_es, ep.config.num_rc)).reshape(2, -1)
        mh_type = ("es", "midhaul", "rc")
        edge_index[mh_type] = np.vstack([mh_src, mh_dst]).astype(np.int64)
        edge_attr[mh_type] = np.column_stack(
            [
                midhaul_mask[mh_src, mh_dst].astype(np.float64),
                midhaul_latency_ms[mh_src, mh_dst],
                midhaul_bandwidth_mbps[mh_src, mh_dst],
                midhaul_cost[mh_src, mh_dst],
            ]
        ).astype(np.float32)

        direct_src, direct_dst = np.indices((ep.config.num_rhs, ep.config.num_rc)).reshape(2, -1)
        direct_type = ("rh", "direct", "rc")
        edge_index[direct_type] = np.vstack([direct_src, direct_dst]).astype(np.int64)
        edge_attr[direct_type] = np.column_stack(
            [
                direct_rh_rc_mask[direct_src, direct_dst].astype(np.float64),
                direct_rh_rc_latency_ms[direct_src, direct_dst],
                direct_rh_rc_bandwidth_mbps[direct_src, direct_dst],
                direct_rh_rc_cost[direct_src, direct_dst],
            ]
        ).astype(np.float32)

        ho_src, ho_dst = np.nonzero(~np.eye(ep.config.num_rhs, dtype=bool))
        ho_type = ("rh", "handover", "rh")
        edge_index[ho_type] = np.vstack([ho_src, ho_dst]).astype(np.int64)
        if ep.observed_handover_matrix_t is None or ep.observed_handover_count_t is None:
            # Explicitly identified legacy runtime compatibility. New scientific datasets
            # carry immutable observed columns and never take this branch.
            observed_t = max(0, time_index - 1)
            observed_matrix = ep.handover_matrix_t
            observed_count = ep.handover_count_t
        else:
            observed_t = time_index
            observed_matrix = ep.observed_handover_matrix_t
            observed_count = ep.observed_handover_count_t
        edge_attr[ho_type] = np.column_stack(
            [
                observed_matrix[observed_t, ho_src, ho_dst],
                observed_count[observed_t, ho_src, ho_dst],
            ]
        ).astype(np.float32)
        return edge_index, edge_attr

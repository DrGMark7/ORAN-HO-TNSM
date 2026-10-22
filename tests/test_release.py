from __future__ import annotations

import random
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from pickle import UnpicklingError
from unittest.mock import patch

import numpy as np
import torch

from src.model.constraint import OrchestrationConstraintEvaluator
from src.model.gppo import GPPOActorCritic
from src.model.inference.post_process import (
    ActionSpaceSpec,
    decode_tuple_action,
    encode_tuple_action,
)
from src.model.model import STGNNActorCritic
from src.pipeline.data_pipeline import MockCsvDatasetLoader
from src.pipeline.schema import PlacementAction
from scripts.ablation.infer_episode_trace import (
    load_checkpoint_payload,
    load_model_checkpoint,
    save_model_checkpoint,
)
from scripts.ablation.training_persistence import capture_training_state, restore_training_state


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "default_32.json"
V2_ROOT = ROOT / "mock_data" / "v2"
TESTSET_ROOT = ROOT / "mock_data" / "testset"
CPLEX_LIBRARY = ROOT / "build" / "cplex" / "liboran_cplex_optimizer.so"
TOPOLOGIES = (8, 16, 32, 64)
TEST_CONDITIONS = ("Regular", "Hard1", "Hard2")


def _dataset_cases() -> list[tuple[Path, int, str]]:
    cases = [
        (
            V2_ROOT / f"openireland_daily_288_{topology}rh_relaxed_capacity_all_es_rc",
            topology,
            "training",
        )
        for topology in TOPOLOGIES
    ]
    for topology in TOPOLOGIES:
        for condition in TEST_CONDITIONS:
            suffix = {"Regular": "", "Hard1": "_hard1", "Hard2": "_hard2"}[condition]
            cases.append(
                (
                    TESTSET_ROOT
                    / f"openireland_daily_288_{topology}rh_relaxed_capacity_test_all_es_rc{suffix}",
                    topology,
                    condition,
                )
            )
    return cases


class SourceOnlyReleaseTests(unittest.TestCase):
    def test_outputs_placeholder_is_present(self) -> None:
        self.assertTrue((ROOT / "outputs" / ".gitkeep").is_file())

    def test_action_encoding_round_trip(self) -> None:
        spec = ActionSpaceSpec(5, 3, 2)
        for action in range(spec.num_actions_per_rh):
            self.assertEqual(encode_tuple_action(*decode_tuple_action(action, spec), spec), action)

    def test_gppo_forward_and_checkpoint_round_trip(self) -> None:
        torch.manual_seed(13)
        model = GPPOActorCritic(rh_feature_dim=5, hidden_dim=8, num_actions_per_rh=30).eval()
        features = torch.zeros(1, 2, 3, 5)
        edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
        attrs = torch.ones(3, 2)
        with torch.no_grad():
            expected = model(features, [edges, edges], [attrs, attrs])
        self.assertEqual(expected[0].shape, (1, 3, 30))

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.pt"
            torch.save(model.state_dict(), path)
            loaded = GPPOActorCritic(rh_feature_dim=5, hidden_dim=8, num_actions_per_rh=30).eval()
            loaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            with torch.no_grad():
                actual = loaded(features, [edges, edges], [attrs, attrs])

        for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
            torch.testing.assert_close(expected_tensor, actual_tensor, rtol=0, atol=0)

    def test_resumable_checkpoint_is_weights_only_compatible(self) -> None:
        torch.manual_seed(13)
        np.random.seed(17)
        model = GPPOActorCritic(rh_feature_dim=5, hidden_dim=8, num_actions_per_rh=30)
        trainer = SimpleNamespace(
            model=model,
            optimizer=torch.optim.Adam(model.parameters()),
            device=torch.device("cpu"),
            env=SimpleNamespace(),
        )
        training_state = capture_training_state(trainer, completed_iteration=3)
        expected_rng = (random.random(), np.random.random(), torch.rand(1))

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "resume.pt"
            save_model_checkpoint(
                path,
                model,
                method="ppo",
                metadata={"ablation_cell": "gppo_no_ho", "split_names": ["S1", "S2", "S3", "S4G", "S4D"]},
                training_state=training_state,
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)

        saved_rng = payload["training_state"]["numpy_random_state"]
        self.assertIsInstance(saved_rng, dict)
        self.assertIsInstance(saved_rng["keys"], list)
        self.assertEqual(restore_training_state(trainer, payload["training_state"]), 3)
        actual_rng = (random.random(), np.random.random(), torch.rand(1))
        self.assertEqual(actual_rng[:2], expected_rng[:2])
        torch.testing.assert_close(actual_rng[2], expected_rng[2], rtol=0, atol=0)

        legacy_state = dict(training_state)
        legacy_state["numpy_random_state"] = np.random.get_state()
        with tempfile.TemporaryDirectory() as temp:
            legacy_path = Path(temp) / "legacy-resume.pt"
            torch.save({"model_state_dict": model.state_dict(), "training_state": legacy_state}, legacy_path)
            with self.assertRaises(UnpicklingError):
                load_checkpoint_payload(legacy_path, map_location="cpu")
            trusted = load_checkpoint_payload(
                legacy_path, map_location="cpu", trust_checkpoint=True
            )
            self.assertIsInstance(trusted["training_state"]["numpy_random_state"], tuple)

    def test_full_checkpoint_round_trip_preserves_model_outputs(self) -> None:
        features = torch.zeros(1, 2, 3, 5)
        edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
        attrs = torch.ones(3, 2)
        graph = ([edges, edges], [attrs, attrs])
        for model_type in (GPPOActorCritic, STGNNActorCritic):
            with self.subTest(model=model_type.__name__):
                torch.manual_seed(101)
                model = model_type(rh_feature_dim=5, hidden_dim=8, num_actions_per_rh=30).eval()
                with torch.no_grad():
                    expected = (
                        model(features, *graph)
                        if model_type is GPPOActorCritic
                        else model.forward_recurrent(features, *graph)
                    )
                with tempfile.TemporaryDirectory() as temp:
                    checkpoint = Path(temp) / "checkpoint.pt"
                    metadata = {"rh_feature_dim": 5}
                    if model_type is STGNNActorCritic:
                        from src.model.model import STGNN_ARCHITECTURE_VERSION

                        metadata["architecture_version"] = STGNN_ARCHITECTURE_VERSION
                    save_model_checkpoint(checkpoint, model, metadata=metadata)
                    loaded = model_type(rh_feature_dim=5, hidden_dim=8, num_actions_per_rh=30).eval()
                    load_model_checkpoint(checkpoint, loaded, torch.device("cpu"))
                    with torch.no_grad():
                        actual = (
                            loaded(features, *graph)
                            if model_type is GPPOActorCritic
                            else loaded.forward_recurrent(features, *graph)
                        )
                for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
                    torch.testing.assert_close(expected_tensor, actual_tensor, rtol=0, atol=0)
                if model_type is STGNNActorCritic:
                    metadata["architecture_version"] = "incompatible-test-architecture"
                    with tempfile.TemporaryDirectory() as temp:
                        checkpoint = Path(temp) / "incompatible.pt"
                        save_model_checkpoint(checkpoint, model, metadata=metadata)
                        with self.assertRaisesRegex(ValueError, "incompatible ST-GNN checkpoint architecture"):
                            load_model_checkpoint(
                                checkpoint,
                                model_type(rh_feature_dim=5, hidden_dim=8, num_actions_per_rh=30),
                                torch.device("cpu"),
                            )

    def test_model_checkpoint_loader_rejects_untrusted_pickle_by_default(self) -> None:
        model = GPPOActorCritic(rh_feature_dim=5, hidden_dim=8, num_actions_per_rh=30)
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / "unpickled.txt"
            checkpoint = Path(temp) / "untrusted.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metadata": {"untrusted": _UntrustedMetadata(marker)},
                },
                checkpoint,
            )

            with self.assertRaises(UnpicklingError):
                load_model_checkpoint(checkpoint, model, torch.device("cpu"))
            self.assertFalse(marker.exists(), "unsafe checkpoint payload executed during default loading")
            load_model_checkpoint(
                checkpoint, model, torch.device("cpu"), trust_checkpoint=True
            )
            self.assertTrue(marker.is_file(), "explicit trusted loading did not enable legacy pickle")

    def test_dispatch_metadata_loader_rejects_untrusted_pickle_by_default(self) -> None:
        from scripts.ablation.dispatch_eval import validate_metadata

        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / "unpickled.txt"
            checkpoint = Path(temp) / "untrusted.pt"
            torch.save(
                {
                    "metadata": {
                        "ablation_cell": "gppo_no_ho",
                        "environment_class": "PrebuiltEpisodeWindowEnv",
                        "constraint_formulation_version": "test-formulation",
                        "observation_feature_mode": "causal_resource_v2",
                        "split_names": ["S1", "S2", "S3", "S4G", "S4D"],
                        "untrusted": _UntrustedMetadata(marker),
                    }
                },
                checkpoint,
            )
            args = SimpleNamespace(
                dataset_dir=Path(temp),
                config=CONFIG,
                observation_feature_mode="causal_resource_v2",
                allow_ood_dataset=True,
                episode_length=288,
                num_episodes=100,
            )
            fake_loader = SimpleNamespace(project_config={"constraint_formulation_version": "test-formulation"})

            with patch("src.pipeline.data_pipeline.MockCsvDatasetLoader", return_value=fake_loader):
                with self.assertRaises(UnpicklingError):
                    validate_metadata(checkpoint, "gppo_no_ho", args)
            self.assertFalse(marker.exists(), "unsafe checkpoint payload executed during metadata validation")

    def test_dispatch_metadata_checks_contract_and_ood_opt_in(self) -> None:
        from scripts.ablation.dispatch_eval import validate_metadata

        metadata = {
            "ablation_cell": "gppo_no_ho",
            "environment_class": "PrebuiltEpisodeWindowEnv",
            "constraint_formulation_version": "test-formulation",
            "observation_feature_mode": "causal_resource_v2",
            "split_names": ["S1", "S2", "S3", "S4G", "S4D"],
            "episode_length": 288,
            "num_episodes": 100,
            "dataset_dir": "/training/data",
        }
        args = SimpleNamespace(
            dataset_dir=Path("/held-out/data"),
            config=CONFIG,
            observation_feature_mode="causal_resource_v2",
            allow_ood_dataset=False,
            episode_length=288,
            num_episodes=100,
        )
        fake_loader = SimpleNamespace(project_config={"constraint_formulation_version": "test-formulation"})
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "valid.pt"
            torch.save({"metadata": metadata}, checkpoint)
            with patch("src.pipeline.data_pipeline.MockCsvDatasetLoader", return_value=fake_loader):
                with self.assertRaisesRegex(ValueError, "dataset_dir mismatch"):
                    validate_metadata(checkpoint, "gppo_no_ho", args)
                args.allow_ood_dataset = True
                self.assertEqual(validate_metadata(checkpoint, "gppo_no_ho", args), metadata)

                for field, invalid in (
                    ("ablation_cell", "stgnn_ho"),
                    ("constraint_formulation_version", "other-formulation"),
                    ("observation_feature_mode", "legacy_rh_only"),
                    ("split_names", ["S1", "S2", "S3", "S4G"]),
                ):
                    changed = dict(metadata)
                    changed[field] = invalid
                    torch.save({"metadata": changed}, checkpoint)
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        validate_metadata(checkpoint, "gppo_no_ho", args)

    def test_stgnn_forward_and_recurrent_state_shapes(self) -> None:
        torch.manual_seed(13)
        model = STGNNActorCritic(rh_feature_dim=5, hidden_dim=8, num_actions_per_rh=30).eval()
        features = torch.zeros(1, 2, 3, 5)
        edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
        attrs = torch.ones(3, 2)
        with torch.no_grad():
            logits, value, hidden = model.forward_recurrent(features, [edges, edges], [attrs, attrs])

        self.assertEqual(logits.shape, (1, 3, 30))
        self.assertEqual(value.shape, (1,))
        self.assertEqual(hidden.shape, (1, 3, 8))

    def test_public_help(self) -> None:
        commands = (
            [sys.executable, "scripts/train.py", "--help"],
            [sys.executable, "scripts/infer.py", "--help"],
            [sys.executable, "scripts/benchmark_cplex.py", "--help"],
            ["bash", "train.sh", "--help"],
        )
        for command in commands:
            with self.subTest(command=command):
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_public_train_and_infer_share_causal_observation_default(self) -> None:
        from scripts.infer import parser as infer_parser
        from scripts.train import parser as train_parser

        self.assertEqual(
            train_parser().parse_args([]).observation_feature_mode, "causal_resource_v2"
        )
        self.assertEqual(
            infer_parser().parse_args([]).observation_feature_mode, "causal_resource_v2"
        )

    def test_public_launchers_report_missing_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / "not-a-dataset"
            commands = (
                [sys.executable, "scripts/train.py", "--dataset", str(missing)],
                [sys.executable, "scripts/infer.py", "--dataset", str(missing)],
            )
            for command in commands:
                with self.subTest(command=command):
                    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn("dataset is missing", result.stderr)


class DatasetIntegrationTests(unittest.TestCase):
    def test_prebuilt_training_and_held_out_datasets(self) -> None:
        available = 0
        missing: list[str] = []
        for dataset, topology, role in _dataset_cases():
            with self.subTest(dataset=dataset.relative_to(ROOT).as_posix()):
                params = dataset / "generation_params.yaml"
                if not params.is_file():
                    missing.append(dataset.relative_to(ROOT).as_posix())
                    continue

                loader = MockCsvDatasetLoader(dataset, CONFIG)
                episode = loader.load_episode()
                self.assertEqual(episode.config.num_rhs, topology)
                self.assertEqual(episode.config.horizon, 28_800)
                self.assertEqual(loader.generation_config["episode_length"], 288)
                self.assertEqual(episode.config.num_splits, 5)
                if role == "training":
                    self.assertEqual(episode.config.horizon // 288, 100)
                available += 1

        if available == 0:
            self.skipTest(
                "no prebuilt training or held-out datasets are available; missing: "
                + ", ".join(missing)
            )

    def test_known_feasible_and_infeasible_actions_on_held_out_data(self) -> None:
        dataset = TESTSET_ROOT / "openireland_daily_288_8rh_relaxed_capacity_test_all_es_rc"
        if not (dataset / "generation_params.yaml").is_file():
            self.skipTest(f"held-out test dataset is unavailable: {dataset.relative_to(ROOT)}")

        loader = MockCsvDatasetLoader(dataset, CONFIG)
        episode = loader.load_episode()
        evaluator = OrchestrationConstraintEvaluator(episode, project_config=loader.project_config)
        feasible = PlacementAction(
            np.array([1, 0, 3, 1, 0, 1, 2, 0]),
            np.array([0, 1, 2, 0, 1, 0, 0, 1]),
            np.array([0, 0, 1, 0, 0, 0, 1, 0]),
        )
        infeasible = PlacementAction(
            np.full(8, 3), feasible.es_id.copy(), feasible.rc_id.copy()
        )

        self.assertTrue(evaluator.evaluate(0, feasible).feasible)
        report = evaluator.evaluate(0, infeasible)
        self.assertFalse(report.feasible)
        self.assertGreater(report.violations["midhaul_bw_shortfall_mbps"], 0)


class CplexIntegrationTests(unittest.TestCase):
    def test_native_one_step_solve_and_exact_evaluation(self) -> None:
        if not CPLEX_LIBRARY.is_file():
            self.skipTest(f"native CPLEX bridge is unavailable: {CPLEX_LIBRARY.relative_to(ROOT)}")
        dataset = TESTSET_ROOT / "openireland_daily_288_8rh_relaxed_capacity_test_all_es_rc"
        if not (dataset / "generation_params.yaml").is_file():
            self.skipTest(f"CPLEX smoke dataset is unavailable: {dataset.relative_to(ROOT)}")

        with tempfile.TemporaryDirectory() as temp:
            command = [
                sys.executable,
                "scripts/benchmark_cplex.py",
                "--topology",
                "8",
                "--dataset",
                str(dataset),
                "--episodes",
                "0",
                "--episode-length",
                "1",
                "--optimization-mode",
                "mirror_soft",
                "--time-limit",
                "1",
                "--threads",
                "1",
                "--output-dir",
                temp,
            ]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
            output = result.stdout + result.stderr
            if result.returncode != 0 and _is_cplex_license_error(output):
                self.skipTest(f"CPLEX license is unavailable: {output.strip()}")
            self.assertEqual(result.returncode, 0, output)
            self.assertTrue(any(Path(temp).rglob("*.csv")))


def _is_cplex_license_error(output: str) -> bool:
    normalized = output.lower()
    return any(
        marker in normalized
        for marker in (
            "cannot open license file",
            "license file not found",
            "no cplex license",
            "cplex license is not available",
            "license has expired",
        )
    )


def _write_unpickle_marker(path: str) -> None:
    Path(path).write_text("executed", encoding="utf-8")


class _UntrustedMetadata:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return _write_unpickle_marker, (str(self.marker),)


if __name__ == "__main__":
    unittest.main()

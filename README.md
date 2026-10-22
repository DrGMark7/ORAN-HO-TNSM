# ORAN-HO placement policies

This repository trains and evaluates four policies for placing O-RAN functional splits. Handover is an observed mobility and cost signal; the policy does not control the radio handover procedure. The five supported split actions are S1, S2, S3, S4G, and S4D. Training, inference, and the CPLEX benchmark share the same exact Python evaluator.

## Repository contents

| Path | Purpose |
| --- | --- |
| `src/` | Environment, data loader, policy models, evaluator, repair logic, and CPLEX bridge |
| `scripts/train.py` | Training entry point |
| `scripts/infer.py` | Inference entry point |
| `scripts/benchmark_cplex.py` | CPLEX benchmark entry point |
| `config/` | Training configurations |
| `mock_data/` | Prebuilt datasets; not included in the current source-only GitHub repository |
| `outputs/` | Generated metrics, checkpoints, and traces; ignored by Git |

## Installation

Linux and Python 3.12 are the verified platform. A CPU is sufficient for short software checks; full training is computationally substantial. Install a PyTorch build suited to your hardware if the default package wheel is unsuitable.

The tested environment used Python 3.12.13, PyTorch 2.11.0+cu128, PyTorch Geometric 2.8.0, NumPy 2.4.6, Polars 1.41.2, Gymnasium 1.3.0, Numba 0.67.0, and PyYAML 6.0.3. CUDA was unavailable on that host. A clean independent installation has not been verified.

Using `venv`:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Or using Conda:

```bash
conda env create -f environment.yml
conda activate oran-ho-release
```

The training and inference workflows use NumPy, Polars, PyTorch, PyTorch Geometric, Gymnasium, Numba, tqdm, and PyYAML. The CPLEX workflow also requires IBM CPLEX Studio and a valid license, a C++17 compiler, CMake, and OpenMP. The Python `docplex` package alone is not sufficient; this repository uses a native CPLEX bridge.

## Training

Training runs sequentially and logs each model variant separately. Defaults are a 288-step daily episode, 100 episodes, seed 13, causal resource v2 observations, and 60,000 iterations. Checkpoints are generated during training and are not included in the repository.

```bash
bash train.sh --models all --topology 8 --config config/default_32.json
bash train.sh --models stgnn --topology 16 --config config/default_32.json
```

Use `--dataset` to select another compatible prebuilt training dataset and `--output-dir` to change the output location. Other configurable options include `--iterations`, `--seed`, `--device`, and `--observation-feature-mode`.

A short run such as `--iterations 1 --eval-scenarios none` checks software execution; it is not a research result. By default, each variant writes `metrics.csv`, `checkpoint.pt`, and `train.log` under `outputs/training/<cell>/`. The `gppo` and `stgnn` names select variants without handover awareness; the `_ho` suffix selects handover-aware variants.

## Inference

Inference requires a checkpoint that matches the model architecture and observation mode. Checkpoints are read from `outputs/training/` by default. Use the individual checkpoint options to override paths:

- `--gppo_no_ho-checkpoint`
- `--gppo_ho-checkpoint`
- `--stgnn_no_ho-checkpoint`
- `--stgnn_ho-checkpoint`

Actions use greedy masking and the exact evaluator. Optional bounded Top-K repair defaults to K=5, five iterations, and beam width 4. Output CSVs include direct and repaired feasibility, costs, and timings. Test datasets differ from training datasets, so the public launcher permits a dataset identity difference while still checking architecture and formulation.

```bash
python scripts/infer.py --models all --topology 8 --condition Regular --episodes 0 --checkpoint-dir outputs/training
python scripts/infer.py --models gppo --topology 8 --condition Hard1 --episodes 0 --repair topk
```

`--condition` accepts `Regular`, `Hard1`, or `Hard2`. Use `--dataset` to select a different test dataset. Set `--repair none` to disable repair. The `--repair-mode deadline` option requires `--repair-decision-budget-seconds`.

## CPLEX benchmark

Build the native bridge against an installed CPLEX Studio distribution:

```bash
cmake -S src/bench/cpp -B build/cplex -DCMAKE_BUILD_TYPE=Release -DCPLEX_ROOT=/path/to/CPLEX_Studio2211
cmake --build build/cplex --parallel 4
python scripts/benchmark_cplex.py --topology 8 --condition Regular --episodes 0 --time-limit 120 --threads 1 --mip-gap 0.01
```

The CLI supports `--optimization-mode mirror_soft` and `hard_service_feasibility`. Use repeatable `--policy-mip-start <rh_decisions.csv>` arguments to provide policy traces as MIP starts. The native solver objective and exact evaluator cost are reported separately.

An incumbent found before a time limit is a best-known solution, not necessarily a certified optimum. `OptimalTol` means optimal within the configured native MIP gap, not exact nonlinear global optimality. Missing CPLEX libraries or a license cause an explicit failure; no substitute solver is used. Build artifacts stay under the ignored `build/` directory, and IBM libraries and license files are not copied into this repository.

The verified native build used CPLEX Studio 22.1.1, GCC 13.1, CMake 3.22.1, and OpenMP 4.5. One-timestep smoke solves succeeded for both optimization modes with exact post-solve feasibility. For a 288-step episode, a 10-second limit without a MIP start returned no incumbent; with a compatible trained-policy trace, it returned a feasible incumbent with 288/288 exact-feasible actions but did not prove optimality:

```bash
python scripts/benchmark_cplex.py --topology 8 --condition Regular --episodes 0 --episode-length 288 --time-limit 10 --threads 1 --mip-gap 0.01 --optimization-mode mirror_soft --policy-mip-start outputs/inference/gppo_no_ho/rh_decisions.csv
```

## Dataset availability and rights

The current source-only GitHub repository does not include `mock_data/`. The local dataset tree contains 7,819,515,833 bytes: 360 files in `v2/` and 553 files in `testset/`. All retained files matched their original counterparts by SHA-256 during validation. The supported 8/16/32/64-RH training and Regular/Hard1/Hard2 held-out layouts load; each has 100 assignments to four training or two held-out topology profiles, with no profile-hash overlap between training and test data.

Dataset redistribution is separate from the software license. The retained metadata does not establish redistribution permission for generated or OpenIreland-derived data. Confirm the data origins and rights before publishing or redistributing the datasets.

Sixteen `topology_contamination_check.json` files contain 32 absolute developer paths. These paths are provenance references, not runtime inputs. They remain unchanged so source parity can be checked. Before publication, a sanitization pass should remove the absolute path prefix from each `held_out_test_dataset` and `source_training_dataset` value, retain the dataset basename, and produce a new metadata checksum manifest for maintainer review.

No public dataset archive or download URL is available. If the data is distributed separately, publish an immutable version, a download location, and a SHA-256 manifest. Extract the files under `mock_data/v2/` and `mock_data/testset/`, verify them from the repository root with `sha256sum -c <manifest>`, and use `--dataset` when storing them elsewhere. The source code is MIT-licensed; that license does not grant rights to the separate datasets. GitHub LFS storage and bandwidth quotas also apply if LFS is chosen for future data distribution. See [GitHub LFS billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs) and [GitHub's LFS file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).

<details>
<summary>Source audit and maintainer notes</summary>

| Area | Purpose and release decision |
| --- | --- |
| `src/pipeline/`, `src/env/` | Load CSV datasets and topology data for training, inference, and MILP. |
| `src/model/` | Shared exact evaluator, split semantics, causal observations, policy architectures, training, inference, and Top-K repair. |
| `src/bench/` | Native CPLEX model and exact post-solve evaluation; source is retained and binary builds are ignored. |
| `src/utils/data_config*.py` | Runtime topology and profile normalization moved from the legacy `data/mock_generator/` modules without formula changes. |
| `scripts/ablation/` | Training and inference implementation modules used by the public launchers. Legacy data-generation helpers are not part of the public training flow. |
| `scripts/benchmark/run_milp_oracle.py` | Episode selection, MILP solve, and exact validation behind `benchmark_cplex.py`. |
| `config/` | RL settings and the legacy configuration adapter required by training. |
| `analysis/`, `tmp/`, notebooks, plotting and dataset-generation scripts, old shell entry points | Development-only flows omitted from the public interface. |

The C++ bridge retains the solver equations and interface. The CMake file omits a development experiment target. Training and inference retain the source equations; the launchers select paths and flags. The original training script expects an exact `episode_length × num_episodes` layout and cannot run a one-step episode against a 288-step dataset. Checkpoints must match the model architecture, observation mode, and formulation. Inference explicitly permits a test-dataset identity difference.

</details>

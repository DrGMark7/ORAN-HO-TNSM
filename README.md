# ORAN-HO placement policies

This repository trains and evaluates four policies for placing O-RAN functional splits. The agent chooses the split and vDU/vCU placement for each radio head. Handover is an observed mobility and cost signal; the agent does not control the radio handover procedure. The supported split actions are S1, S2, S3, S4G, and S4D. Training, inference, and the CPLEX benchmark use the same exact Python evaluator.

## Source-only checkout

The GitHub repository contains source and configuration only. It does not contain prebuilt datasets or pretrained checkpoints. Dataset-based commands require a compatible prebuilt dataset with its `generation_params.yaml`:

- Training data: `mock_data/v2/`
- Held-out data: `mock_data/testset/`

Place those directories at the repository root, or supply another dataset directory with `--dataset`. No download URL is provided. Generated metrics, logs, traces, and checkpoints go under `outputs/` by default. The tracked `outputs/.gitkeep` preserves the directory; generated contents are ignored by Git.

Run source-only maintenance checks with the documented Python dependencies:

```bash
python -m unittest discover -s tests -v
bash train.sh --help
python scripts/infer.py --help
python scripts/benchmark_cplex.py --help
```

Dataset and licensed CPLEX integration tests are included in the same unittest discovery command. They run when their exact data/native prerequisites are present and otherwise report `SKIP` with the missing path or license condition. A skip means that integration was not validated in that environment.

## Installation

Python 3.12 is the maintained release target. The locally validated Conda environment used Python 3.12.13, PyTorch 2.11.0+cu128, PyTorch Geometric 2.8.0, NumPy 2.4.6, Polars 1.41.2, Gymnasium 1.3.0, Numba 0.67.0, PyYAML 6.0.3, and docplex 2.32.264. This is a compatibility record, not a guarantee of a fresh installation. PyTorch and PyTorch Geometric wheels can depend on the operating system, Python, and CPU/CUDA choice; choose compatible builds for your machine before installing the remaining requirements.

For example, create an isolated environment and install the project requirements after selecting an appropriate PyTorch/PyG build for the platform:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

The CPLEX workflow additionally requires IBM CPLEX Studio, a valid license, a C++17 compiler, CMake, and OpenMP. The Python `docplex` package alone is not sufficient. IBM software and license files are not distributed here.

## Training

`train.sh` maps model names to the four release variants: `gppo` → `gppo_no_ho`, `gppo_ho` → `gppo_ho`, `stgnn` → `stgnn_no_ho`, and `stgnn_ho` → `stgnn_ho`. Use `--models all` to train all four sequentially. The default public observation mode is `causal_resource_v2`; it is selected by the CLI and overrides the legacy observation mode recorded in the JSON configuration files. Set `--observation-feature-mode` explicitly when reproducing a different checkpoint configuration.

```bash
bash train.sh --models all --topology 8 --config config/default_32.json
bash train.sh --models stgnn --topology 16 --config config/default_32.json
bash train.sh --models all --dataset /path/to/training-dataset --iterations 1 --eval-scenarios none
```

The dataset must contain a complete `episode_length × num_episodes` layout. The paper runs use 288-step daily episodes and 100 episodes. A one-iteration run still processes the configured complete episodes; it is a software smoke check only. It is not a paper result or a substitute for the 60,000-iteration experiments. No smoke checkpoint is a published research result. By default, per-variant metrics, checkpoints, and logs are written under `outputs/training/<variant>/`.

## Held-out inference

Inference requires checkpoints compatible with the model architecture, observation features, and constraint formulation. The public launcher allows the held-out dataset identity to differ from the training dataset; other required metadata remains checked.

```bash
python scripts/infer.py --models all --topology 8 --condition Regular --episodes 0 --checkpoint-dir outputs/training
python scripts/infer.py --models gppo --topology 8 --condition Hard1 --episodes 0 --repair topk
python scripts/infer.py --models all --dataset /path/to/held-out-dataset --episodes 0
```

`--condition` accepts `Regular`, `Hard1`, or `Hard2`. Individual checkpoint paths can be supplied with `--gppo_no_ho-checkpoint`, `--gppo_ho-checkpoint`, `--stgnn_no_ho-checkpoint`, and `--stgnn_ho-checkpoint`. Greedy inference uses the exact evaluator and optional bounded Top-K repair (K=5, five iterations, beam width 4 by default). Use `--repair none` to disable repair. Deadline mode requires `--repair-decision-budget-seconds`.

### Checkpoint trust

Checkpoint loading uses PyTorch `weights_only=True` by default and does not fall back to unrestricted pickle loading after an error. New resumable checkpoints store their NumPy random state using primitive values so they load through this safe path. Legacy checkpoints containing unsupported Python/NumPy objects require the explicit `--trust-checkpoint` flag in inference/evaluation or in the lower-level trainer resume/fine-tune commands. Only use that flag for a checkpoint from a trusted source: unrestricted pickle loading can execute code. Architecture, observation-feature mode/schema, formulation, split names, and recurrent metadata checks remain enforced; held-out dataset identity is relaxed only by the explicit public inference path.

## CPLEX benchmark

Build the native bridge against a local CPLEX Studio installation and run a bounded smoke solve:

```bash
cmake -S src/bench/cpp -B build/cplex -DCMAKE_BUILD_TYPE=Release -DCPLEX_ROOT=/path/to/CPLEX_Studio2211
cmake --build build/cplex --parallel 4
python scripts/benchmark_cplex.py --topology 8 --condition Regular --episodes 0 --episode-length 1 --time-limit 1 --threads 1 --optimization-mode mirror_soft
```

The solver supports `mirror_soft` and `hard_service_feasibility`; repeatable `--policy-mip-start <rh_decisions.csv>` arguments supply policy traces as MIP starts. The native objective and exact evaluator cost are reported separately. `OptimalTol` means optimal within the configured native MIP gap. A feasible incumbent returned at the time limit is a best-known solution and is not a certificate of optimality. CPLEX binaries, headers, libraries, and license files are not shipped. Build output stays under the ignored `build/` directory.

These commands are software checks, not full paper reproduction. Full reproduction requires the specified datasets, experimental settings, and 60,000 training iterations; smoke output is not a published research result.

## Data and licensing

The MIT license applies to released source code. It does not grant rights to external datasets or IBM CPLEX. Dataset redistribution rights remain under maintainer review; no dataset download, upload, or Git LFS setup is provided by this repository. Existing provenance JSON is unchanged in this release task and may contain development-host paths; provenance sanitization and external dataset hosting remain unresolved. Do not redistribute the datasets until their rights and provenance have been reviewed.

# ORAN-HO research code

This repository trains and evaluates four placement policies for O-RAN functional splits. Handover is an observed mobility and cost signal; the policy does not control a radio handover procedure. The five split actions are S1, S2, S3, S4G, and S4D. The exact Python evaluator is shared by training, inference, and the CPLEX benchmark's post-solve validation.

## Layout

- `src/`: environment, prebuilt CSV loader, policy models, PPO, exact evaluator, bounded Top-K repair, and MILP bridge.
- `scripts/train.py`, `scripts/infer.py`, `scripts/benchmark_cplex.py`: supported public interfaces.
- `mock_data/`: retained, byte-identical prebuilt datasets. `v2/` contains training and legacy test copies; `testset/` contains Regular, Hard1, and Hard2 held-out conditions. The development-only `nonflexible/`, `test/`, and `v1/` folders are intentionally absent.
- `config/`: RL configuration; `default_32.json` has hidden width 32 used by the training command below.
- `outputs/`: generated metrics, checkpoints, and traces (ignored by Git).

## Installation

Linux and Python 3.12 are the verified platform. A CPU works for smoke checks; full training is computationally substantial. Install PyTorch with a build appropriate for your hardware before the remaining packages if the default wheel is unsuitable.

The validation below reused an existing Conda environment named `gppo` with Python 3.12.13, PyTorch 2.11.0+cu128, PyTorch Geometric 2.8.0, NumPy 2.4.6, Polars 1.41.2, Gymnasium 1.3.0, Numba 0.67.0, and PyYAML 6.0.3. CUDA was unavailable on the validation host. A clean independent install has not been verified.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Or, from this directory:

```bash
conda env create -f environment.yml
conda activate oran-ho-release
```

The training and inference stack uses NumPy, Polars, PyTorch, PyTorch Geometric, Gymnasium, Numba, tqdm, and PyYAML. The CPLEX workflow additionally requires IBM CPLEX Studio with a valid license, a C++17 compiler, CMake, and OpenMP. The Python `docplex` package alone is insufficient: this repository uses a native CPLEX bridge.

## Training

Runs are sequential and independently logged. The default is the original 288 step daily episode, 100 episodes, seed 13, causal resource v2 observation, and 60,000 iterations. Checkpoints are created by training; none are bundled.

```bash
bash train.sh --models all --topology 8 --config config/default_32.json
bash train.sh --models stgnn --topology 16 --config config/default_32.json
```

Use `--dataset` for another compatible prebuilt training directory and `--output-dir` to place results elsewhere. `--iterations`, `--seed`, `--device`, and `--observation-feature-mode` are configurable. A short run such as `--iterations 1 --eval-scenarios none` is a software smoke check, not a paper result. The launcher reports each variant's exit status and stores `metrics.csv`, `checkpoint.pt`, and `train.log` in `outputs/training/<cell>/` by default. `gppo` and `stgnn` select the no-HO variants; `_ho` selects the handover-aware variants.

## Inference

Supply trained checkpoints from the same architecture and observation mode. The default checkpoint directory is `outputs/training/`; individual paths can be set with `--gppo_no_ho-checkpoint`, `--gppo_ho-checkpoint`, `--stgnn_no_ho-checkpoint`, and `--stgnn_ho-checkpoint`. Inference uses greedy masked actions, optional bounded Top-K repair (default K=5, 5 iterations, beam width 4), and the exact evaluator. The evaluator writes direct and repaired feasibility, costs, and timing fields in the output CSVs. Test data is out of distribution relative to a training checkpoint, so the public launcher explicitly permits the dataset identity difference while retaining architecture and formulation checks.

```bash
python scripts/infer.py --models all --topology 8 --condition Regular --episodes 0 --checkpoint-dir outputs/training
python scripts/infer.py --models gppo --topology 8 --condition Hard1 --episodes 0 --repair topk
```

`--condition` accepts `Regular`, `Hard1`, and `Hard2`; `--dataset` overrides the default testset directory. `--repair none` disables repair. `--repair-mode deadline` requires `--repair-decision-budget-seconds`.

## CPLEX benchmark

Build the native bridge against your licensed CPLEX Studio installation:

```bash
cmake -S src/bench/cpp -B build/cplex -DCMAKE_BUILD_TYPE=Release -DCPLEX_ROOT=/path/to/CPLEX_Studio2211
cmake --build build/cplex --parallel 4
python scripts/benchmark_cplex.py --topology 8 --condition Regular --episodes 0 --time-limit 120 --threads 1 --mip-gap 0.01
```

The verified native build used CPLEX Studio 22.1.1, GCC 13.1, CMake 3.22.1, and OpenMP 4.5. Build artifacts remain ignored under `build/`; IBM libraries and license files are never copied into this repository. The CLI accepts `--optimization-mode mirror_soft` or `hard_service_feasibility`, and repeatable `--policy-mip-start <rh_decisions.csv>`. The native solver objective and exact evaluator cost are reported separately. An incumbent after a limit is a best-known solution, not a certified optimum. `OptimalTol` means optimal within the configured native MIP gap, not exact nonlinear global optimality. An absent library or license causes an explicit failure; no substitute solver is used. One-timestep `mirror_soft` and `hard_service_feasibility` smoke solves both succeeded with exact post-solve feasibility. For a full 288-step episode, a 10-second limit without a MIP start returned no incumbent; the same limit with a compatible trained-policy trace returned a feasible incumbent and 288/288 exact-feasible actions, but did not prove optimality:

```bash
python scripts/benchmark_cplex.py --topology 8 --condition Regular --episodes 0 --episode-length 288 --time-limit 10 --threads 1 --mip-gap 0.01 --optimization-mode mirror_soft --policy-mip-start outputs/inference/gppo_no_ho/rh_decisions.csv
```

## Validation and troubleshooting

```bash
python -m unittest discover -s tests -v
bash train.sh --help
python scripts/infer.py --help
python scripts/benchmark_cplex.py --help
```

Run from the repository root. A missing `generation_params.yaml` means the dataset path is wrong. A checkpoint metadata mismatch usually means the configuration, observation mode, episode layout, or architecture differs from training. CPLEX build errors require checking `CPLEX_ROOT` and its license. Generated results stay under `outputs/` and are ignored by Git.

The retained `mock_data/` tree is 7,819,515,833 bytes: `v2/` has 360 files (3,132,911,948 bytes), and `testset/` has 553 files (4,686,603,885 bytes). Every retained file matched its original counterpart by SHA-256 during validation. All 8/16/32/64 RH training and Regular/Hard1/Hard2 held-out layouts load; each has 100 assignments to four training or two held-out topology profiles, with no training/test profile-hash overlap.

**Data rights are a separate publication gate from the software license.** No explicit redistribution permission for the generated or OpenIreland-derived material was found in the retained metadata. Do not publish until the maintainer confirms the data's origins and rights. Sixteen `topology_contamination_check.json` files contain 32 absolute developer paths. Those are provenance references, not runtime inputs; the files have not been edited so that source parity remains verifiable. A proposed publication-only sanitization would replace the absolute prefix through `/mock_data/` with an empty string in each `held_out_test_dataset` and `source_training_dataset` value, leaving the origin dataset basename. CSVs and scientific parameters would remain unchanged. That change requires a new metadata checksum manifest and maintainer review.

Git/LFS is not yet operational in this workspace: `.git/` is an empty read-only placeholder and `git lfs` is unavailable. A proposed local LFS rule set for the 40 large `rh_timeseries.csv` and `handover_edges.csv` files is documented outside the publication tree until rights and Git access are resolved. Those files total 4,323,597,351 bytes; the remaining data is still about 3.50 GB, so a versioned external dataset archive with a checksum manifest and documented downloader is the preferred distribution design. Keep readable metadata in ordinary Git only after the path and rights review. No license is asserted for this code until the maintainer chooses one.

If the maintainer chooses Git LFS after rights clearance, proposed `.gitattributes` patterns are `mock_data/v2/**/{rh_timeseries,handover_edges}.csv` and `mock_data/testset/**/{rh_timeseries,handover_edges}.csv` (expand each filename into its own rule). They cover all 40 files above 10 MB and no metadata. Confirm the account plan and run `git check-attr` and `git lfs ls-files` in a functional checkout before staging anything. GitHub currently includes 10 GiB of LFS storage and 10 GiB of monthly bandwidth on Free/Pro, with a 2 GB per-file limit; repeated clones and versions consume quota. See [GitHub LFS billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs) and [file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).

If the maintainer instead distributes data as an archive, publish an immutable version identifier, an explicit download location, and a SHA-256 manifest covering each retained file. Researchers should extract it as `mock_data/v2/` and `mock_data/testset/`, run `sha256sum -c <provided-manifest>` from the repository root, then pass `--dataset` for a nondefault location. No archive, downloader, or public URL exists yet; neither a Git clone nor a checkpoint download should be assumed to include them.

## Source audit and release decisions

| Original file/group | Role and call sites | Public decision | Evidence |
|---|---|---|---|
| `src/pipeline/{data_pipeline,schema}.py`, `src/env/*.py` | CSV and topology loading for trainers, traces, and MILP | Keep under `src/` | Dataset loader and training smoke |
| `src/model/{constraint,split_semantics,causal_observation,serial_numba}.py` | Shared exact evaluator, split semantics, causal state | Keep under `src/` | Imported by environment, inference, and MILP; unit tests |
| `src/model/{gppo,model}.py`, `src/model/training/{ppo,recurrent_ppo}.py` | Two actor architectures and optimizers | Keep under `src/` | Four-cell training smoke |
| `src/model/inference/*.py` | Flat action decoding, masks, Top-K search | Keep under `src/` | Action round trip and repair tests |
| `src/bench/{optimizer,benchmark,native_hard_service,helper}.py`, `src/bench/cpp/` | Native CPLEX model and exact post-solve re-evaluation | Keep source; ignore binary build | Native 22.1.1 build, solve, and exact evaluation |
| `data/mock_generator/{config,config_validation}.py` | Runtime topology and profile normalization called from `src/utils/config.py` | Move to `src/utils/data_config*.py`; update imports | Loader import and dataset test |
| `data/mockup_data.py` and data generation helpers | Called only by legacy generator functions in `training_metrics_common.py` | Remove legacy generator functions from public training helper | Call-site search; public trainer uses `build_prebuilt_scenario*` |
| `scripts/ablation/{train_*,eval_*,infer_episode_trace,training_metrics_common,training_persistence}.py` | Four-cell training and inference implementation | Keep as implementation modules under `scripts/ablation/` | Public launcher and tests |
| `scripts/benchmark/run_milp_oracle.py` | Episode selection, MILP solve and exact validation | Keep behind `benchmark_cplex.py` | Native library preflight and help |
| `analysis/`, `tmp/`, notebooks, plotting scripts, dataset generation scripts, old shell entry points | Development analyses or unsupported generation/plotting flows | Omit | No public call path imports them |
| `config/{default,default_32,legacy_default_v1}.json` | RL settings and compatibility adapter | Keep | Trainer traceback identified legacy adapter dependency; rerun |

The C++ bridge retains the original equations and solver interface. Its CMake file omits a development experiment target that is not part of the public bridge. The two data configuration helpers were copied without formula changes. Training and inference implementation files otherwise retain the source equations; new launchers select paths and flags. The original training script assumes an exact `episode_length × num_episodes` layout and cannot use a one step episode against a 288 step dataset. Checkpoint metadata requires matching architecture, observation mode, and formulation; inference explicitly opts into a test dataset identity difference.

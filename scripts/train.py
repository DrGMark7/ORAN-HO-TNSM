#!/usr/bin/env python3
"""Sequential public launcher for the four original ablation trainers."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = {'gppo': 'gppo_no_ho', 'gppo_ho': 'gppo_ho', 'stgnn': 'stgnn_no_ho', 'stgnn_ho': 'stgnn_ho'}

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', choices=(*CELLS, 'all'), default='all')
    p.add_argument('--topology', type=int, choices=(8, 16, 32, 64), default=8)
    p.add_argument('--dataset', type=Path, help='prebuilt training dataset (overrides topology default)')
    p.add_argument('--config', type=Path, default=ROOT/'config/default_32.json')
    p.add_argument('--iterations', type=int, default=60000)
    p.add_argument('--num-episodes', type=int, default=100)
    p.add_argument('--episode-length', type=int, default=288, help='288 for paper runs; smaller values are smoke only')
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--device', default='cpu')
    p.add_argument('--observation-feature-mode', choices=('legacy_rh_only','causal_resource_v1','causal_resource_v2'), default='causal_resource_v2')
    p.add_argument('--eval-interval', type=int, default=25)
    p.add_argument('--eval-scenarios', choices=('all','current','none'), default='all')
    p.add_argument('--output-dir', type=Path, default=ROOT/'outputs/training')
    return p

def main(argv=None):
    args = parser().parse_args(argv)
    if args.iterations < 1 or args.num_episodes < 1 or args.episode_length < 1:
        parser().error('iterations, num-episodes and episode-length must be positive')
    dataset = args.dataset or ROOT/f'mock_data/v2/openireland_daily_288_{args.topology}rh_relaxed_capacity_all_es_rc'
    dataset = dataset.resolve()
    params = dataset/'generation_params.yaml'
    if not params.is_file(): parser().error(f'dataset is missing generation_params.yaml: {dataset}')
    if not args.config.is_file(): parser().error(f'config is missing: {args.config}')
    selected = list(CELLS) if args.models == 'all' else [args.models]
    failures = []
    for name in selected:
        cell = CELLS[name]
        out = args.output_dir.resolve()/cell
        out.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(ROOT/'scripts/ablation/dispatch_train.py'), '--cell', cell,
            '--dataset_dir',str(dataset),'--params',str(params),'--config',str(args.config.resolve()),
            '--episode_length',str(args.episode_length),'--num_episodes',str(args.num_episodes),
            '--iterations',str(args.iterations),'--seed',str(args.seed),'--device',args.device,
            '--observation_feature_mode',args.observation_feature_mode,
            '--eval_interval',str(args.eval_interval),'--eval_scenarios',args.eval_scenarios,
            '--output',str(out/'metrics.csv'),'--checkpoint_output',str(out/'checkpoint.pt')]
        print(f'Training {cell}: {out}', flush=True)
        with (out/'train.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(f'{cell}: {"PASS" if result.returncode == 0 else "FAIL"} (exit {result.returncode})', flush=True)
        if result.returncode: failures.append(cell)
    return 1 if failures else 0

if __name__ == '__main__': raise SystemExit(main())

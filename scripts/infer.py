#!/usr/bin/env python3
"""Evaluate trained policies on one prebuilt test condition with the exact evaluator."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CELLS={'gppo':'gppo_no_ho','gppo_ho':'gppo_ho','stgnn':'stgnn_no_ho','stgnn_ho':'stgnn_ho'}

def parser():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--models',choices=(*CELLS,'all'),default='all')
 p.add_argument('--topology',type=int,choices=(8,16,32,64),default=8)
 p.add_argument('--condition',choices=('Regular','Hard1','Hard2'),default='Regular')
 p.add_argument('--dataset',type=Path)
 p.add_argument('--config',type=Path,default=ROOT/'config/default_32.json')
 p.add_argument('--checkpoint-dir',type=Path,default=ROOT/'outputs/training')
 for cell in CELLS.values(): p.add_argument(f'--{cell}-checkpoint',type=Path)
 p.add_argument('--episodes',default='0')
 p.add_argument('--num-episodes',type=int,default=100)
 p.add_argument('--episode-length',type=int,default=288)
 p.add_argument('--seed',type=int,default=13)
 p.add_argument('--device',default='cpu')
 p.add_argument('--observation-feature-mode',choices=('legacy_rh_only','causal_resource_v1','causal_resource_v2'),default='causal_resource_v2')
 p.add_argument('--repair',choices=('none','topk'),default='topk')
 p.add_argument('--repair-top-k',type=int,default=5)
 p.add_argument('--repair-max-iterations',type=int,default=5)
 p.add_argument('--repair-beam-width',type=int,default=4)
 p.add_argument('--repair-mode',choices=('bounded','deadline'),default='bounded')
 p.add_argument('--repair-decision-budget-seconds',type=float)
 p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/inference')
 return p

def main(argv=None):
 p=parser();a=p.parse_args(argv)
 suffix={'Regular':'','Hard1':'_hard1','Hard2':'_hard2'}[a.condition]
 dataset=(a.dataset or ROOT/f'mock_data/testset/openireland_daily_288_{a.topology}rh_relaxed_capacity_test_all_es_rc{suffix}').resolve()
 if not (dataset/'generation_params.yaml').is_file(): p.error(f'dataset is missing: {dataset}')
 selected=list(CELLS) if a.models=='all' else [a.models]
 checkpoints={cell:(getattr(a,f'{cell}_checkpoint') or a.checkpoint_dir/cell/'checkpoint.pt').resolve() for cell in CELLS.values()}
 for name in selected:
  cell=CELLS[name]
  if not checkpoints[cell].is_file(): p.error(f'checkpoint is missing for {cell}: {checkpoints[cell]}')
 cmd=[sys.executable,str(ROOT/'scripts/ablation/dispatch_eval.py'),'--dataset_dir',str(dataset),'--params',str(dataset/'generation_params.yaml'),
  '--config',str(a.config.resolve()),'--episode_length',str(a.episode_length),'--num_episodes',str(a.num_episodes),
  '--output_dir',str(a.output_dir.resolve()),'--episodes',a.episodes,'--cells',','.join(CELLS[x] for x in selected),
  '--seed',str(a.seed),'--device',a.device,'--observation_feature_mode',a.observation_feature_mode,
  '--infeasibility-repair',a.repair,'--repair-top-k',str(a.repair_top_k),'--repair-max-iterations',str(a.repair_max_iterations),
  '--repair-beam-width',str(a.repair_beam_width),'--repair-mode',a.repair_mode,'--allow_ood_dataset']
 for cell,path in checkpoints.items(): cmd.extend([f'--{cell}_checkpoint',str(path)])
 if a.repair_decision_budget_seconds is not None: cmd.extend(['--repair-decision-budget-seconds',str(a.repair_decision_budget_seconds)])
 return subprocess.call(cmd,cwd=ROOT)
if __name__=='__main__':raise SystemExit(main())

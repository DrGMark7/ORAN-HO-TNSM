#!/usr/bin/env python3
"""Run the native CPLEX oracle and exact evaluator on selected test episodes."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def parser():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--topology',type=int,choices=(8,16,32,64),default=8)
 p.add_argument('--condition',choices=('Regular','Hard1','Hard2'),default='Regular')
 p.add_argument('--dataset',type=Path)
 p.add_argument('--config',type=Path,default=ROOT/'config/default_32.json')
 p.add_argument('--library',type=Path,default=ROOT/'build/cplex/liboran_cplex_optimizer.so')
 p.add_argument('--episodes',default='0')
 p.add_argument('--episode-length',type=int,default=288)
 p.add_argument('--time-limit',type=float,default=120.0)
 p.add_argument('--threads',type=int,default=1)
 p.add_argument('--mip-gap',type=float,default=0.01)
 p.add_argument('--optimization-mode',choices=('mirror_soft','hard_service_feasibility'),default='hard_service_feasibility')
 p.add_argument('--policy-mip-start',type=Path,action='append',default=[])
 p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/cplex')
 return p

def main(argv=None):
 p=parser();a=p.parse_args(argv)
 suffix={'Regular':'','Hard1':'_hard1','Hard2':'_hard2'}[a.condition]
 dataset=(a.dataset or ROOT/f'mock_data/testset/openireland_daily_288_{a.topology}rh_relaxed_capacity_test_all_es_rc{suffix}').resolve()
 if not (dataset/'generation_params.yaml').is_file():p.error(f'dataset is missing: {dataset}')
 if not a.library.is_file():p.error(f'CPLEX native library is missing: {a.library}. Build it with cmake -S src/bench/cpp -B build/cplex -DCPLEX_ROOT=/path/to/CPLEX_Studio; cmake --build build/cplex')
 cmd=[sys.executable,str(ROOT/'scripts/benchmark/run_milp_oracle.py'),'--dataset',str(dataset),'--config',str(a.config.resolve()),
 '--library',str(a.library.resolve()),'--episodes',a.episodes,'--episode-length',str(a.episode_length),
 '--time-limit',str(a.time_limit),'--threads',str(a.threads),'--mip-gap',str(a.mip_gap),
 '--optimization-mode',a.optimization_mode,'--output',str(a.output_dir.resolve())]
 for path in a.policy_mip_start:cmd.extend(['--policy-mip-start',str(path.resolve())])
 return subprocess.call(cmd,cwd=ROOT)
if __name__=='__main__':raise SystemExit(main())

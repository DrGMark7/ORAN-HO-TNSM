from __future__ import annotations
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from src.model.gppo import GPPOActorCritic
from src.model.model import STGNNActorCritic
from src.model.inference.post_process import ActionSpaceSpec, decode_tuple_action, encode_tuple_action
from src.pipeline.data_pipeline import MockCsvDatasetLoader
from src.pipeline.schema import PlacementAction
from src.model.constraint import OrchestrationConstraintEvaluator

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'mock_data/v2/openireland_daily_288_8rh_relaxed_capacity_all_es_rc'

class ReleaseTests(unittest.TestCase):
 def test_dataset_and_action_space(self):
  self.assertTrue((DATA/'generation_params.yaml').is_file())
  episode=MockCsvDatasetLoader(DATA, ROOT/'config/default_32.json').load_episode()
  self.assertEqual(episode.config.num_rhs,8)
  self.assertEqual(episode.config.horizon,28800)
  spec=ActionSpaceSpec.from_topology(episode.config)
  self.assertGreater(spec.num_actions_per_rh,0)
  for action in range(spec.num_actions_per_rh):
   self.assertEqual(encode_tuple_action(*decode_tuple_action(action,spec),spec),action)
 def test_gppo_forward_and_checkpoint(self):
  torch.manual_seed(13)
  model=GPPOActorCritic(rh_feature_dim=5,hidden_dim=8,num_actions_per_rh=30).eval()
  features=torch.zeros(1,2,3,5)
  edges=torch.tensor([[0,1,2],[1,2,0]])
  attrs=torch.ones(3,2)
  with torch.no_grad(): actual=model(features,[edges,edges],[attrs,attrs])
  self.assertEqual(actual[0].shape,(1,3,30))
  with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temp:
   path=Path(temp)/'state.pt';torch.save(model.state_dict(),path)
   loaded=GPPOActorCritic(rh_feature_dim=5,hidden_dim=8,num_actions_per_rh=30).eval()
   loaded.load_state_dict(torch.load(path,weights_only=True))
   with torch.no_grad():again=loaded(features,[edges,edges],[attrs,attrs])
  for a,b in zip(actual,again):torch.testing.assert_close(a,b,rtol=0,atol=0)
 def test_stgnn_forward_and_hidden_state(self):
  torch.manual_seed(13)
  model=STGNNActorCritic(rh_feature_dim=5,hidden_dim=8,num_actions_per_rh=30).eval()
  features=torch.zeros(1,2,3,5)
  edges=torch.tensor([[0,1,2],[1,2,0]])
  attrs=torch.ones(3,2)
  with torch.no_grad():logits,value,hidden=model.forward_recurrent(features,[edges,edges],[attrs,attrs])
  self.assertEqual(logits.shape,(1,3,30))
  self.assertEqual(value.shape,(1,))
  self.assertEqual(hidden.shape,(1,3,8))
 def test_exact_feasible_and_infeasible(self):
  dataset=ROOT/'mock_data/testset/openireland_daily_288_8rh_relaxed_capacity_test_all_es_rc'
  loader=MockCsvDatasetLoader(dataset,ROOT/'config/default_32.json')
  episode=loader.load_episode()
  evaluator=OrchestrationConstraintEvaluator(episode,project_config=loader.project_config)
  feasible=PlacementAction(np.array([1,0,3,1,0,1,2,0]),np.array([0,1,2,0,1,0,0,1]),np.array([0,0,1,0,0,0,1,0]))
  infeasible=PlacementAction(np.full(8,3),feasible.es_id.copy(),feasible.rc_id.copy())
  self.assertTrue(evaluator.evaluate(0,feasible).feasible)
  report=evaluator.evaluate(0,infeasible)
  self.assertFalse(report.feasible)
  self.assertGreater(report.violations['midhaul_bw_shortfall_mbps'],0)
 def test_public_help(self):
  for command in ([sys.executable,'scripts/train.py','--help'],[sys.executable,'scripts/infer.py','--help'],[sys.executable,'scripts/benchmark_cplex.py','--help'],['bash','train.sh','--help']):
   with self.subTest(command=command):
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    self.assertEqual(result.returncode,0,result.stderr)
    self.assertIn('usage:',result.stdout)
 def test_cplex_optional(self):
  library=ROOT/'build/cplex/liboran_cplex_optimizer.so'
  if not library.is_file():
   self.skipTest('licensed native CPLEX bridge is not installed')
  with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temp:
   command=[sys.executable,'scripts/benchmark_cplex.py','--topology','8','--episodes','0',
            '--episode-length','1','--optimization-mode','mirror_soft',
            '--time-limit','1','--threads','1','--output-dir',temp]
   result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=180)
   self.assertEqual(result.returncode,0,result.stdout+result.stderr)
   self.assertTrue(any(Path(temp).rglob('*.csv')))
